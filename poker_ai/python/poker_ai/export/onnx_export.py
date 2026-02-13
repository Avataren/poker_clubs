"""Export trained Average Strategy network to ONNX format."""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.model.network import AverageStrategyNet


class AverageStrategyONNXWrapper(nn.Module):
    """Wrapper that flattens the AS network interface for ONNX export.

    ONNX doesn't handle complex Python logic well, so we create a simple
    forward pass that takes all inputs and returns action probabilities.

    Pre-computes position indices as registered buffers so that the ONNX
    graph uses constant tensors instead of Range ops (which tract cannot
    handle).
    """

    def __init__(self, as_net: AverageStrategyNet, max_seq_len: int = 30):
        super().__init__()
        self.as_net = as_net
        # Pre-compute position indices as constant buffers to avoid
        # torch.arange in the ONNX graph (tract doesn't support Range).
        self.register_buffer("pos_ids", torch.arange(max_seq_len, dtype=torch.long))
        self.register_buffer(
            "positions", torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        history_hidden = self._encode_history(action_history, history_lengths)
        return self.as_net.net.policy(obs, history_hidden, legal_mask)

    def _encode_history(
        self, action_seq: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Re-implementation of ActionHistoryTransformer.forward using
        pre-computed position buffers instead of torch.arange."""
        enc = self.as_net.history_encoder
        dtype = action_seq.dtype

        # Masking (uses pre-computed positions buffer)
        valid_mask_f = (self.positions < lengths.unsqueeze(1)).to(dtype)
        pad_mask = (1.0 - valid_mask_f) * torch.finfo(dtype).min

        # Project input and add positional embeddings (uses pre-computed pos_ids)
        x = enc.input_proj(action_seq)
        x = x + enc.pos_embed(self.pos_ids).unsqueeze(0)

        # Transformer encoder
        x = enc.transformer(x, src_key_padding_mask=pad_mask)

        # Mean pool over valid positions
        valid_counts = valid_mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        x = (x * valid_mask_f.unsqueeze(-1)).sum(dim=1) / valid_counts
        has_history = (valid_counts > 0.5).to(dtype)
        x = x * has_history

        return enc.output_proj(x)


def infer_config_from_checkpoint(checkpoint: dict, base_config: NFSPConfig) -> NFSPConfig:
    """Infer architecture fields from checkpoint weights, preserving other config values."""
    config = copy.deepcopy(base_config)
    as_state = checkpoint.get("as_net")
    if not isinstance(as_state, dict):
        return config

    hist_w = as_state.get("history_encoder.net.0.weight")
    if isinstance(hist_w, torch.Tensor) and hist_w.ndim == 2:
        config.history_hidden_dim = int(hist_w.shape[0])
        flat_input = int(hist_w.shape[1])
        if config.history_input_dim > 0 and flat_input % config.history_input_dim == 0:
            config.max_history_len = flat_input // config.history_input_dim

    trunk0_w = as_state.get("net.trunk.0.weight")
    if isinstance(trunk0_w, torch.Tensor) and trunk0_w.ndim == 2:
        config.hidden_dim = int(trunk0_w.shape[0])
        total_in = int(trunk0_w.shape[1])
        from poker_ai.model.state_encoder import STATIC_FEATURE_SIZE
        inferred_hist_hidden = total_in - STATIC_FEATURE_SIZE
        if inferred_hist_hidden > 0:
            config.history_hidden_dim = inferred_hist_hidden

    trunk4_w = as_state.get("net.trunk.4.weight")
    if isinstance(trunk4_w, torch.Tensor) and trunk4_w.ndim == 2:
        config.residual_dim = int(trunk4_w.shape[0])

    policy_out_w = as_state.get("net.policy_head.2.weight")
    if isinstance(policy_out_w, torch.Tensor) and policy_out_w.ndim == 2:
        config.num_actions = int(policy_out_w.shape[0])

    return config


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    """Resolve checkpoint path from common working directories."""
    raw_path = Path(checkpoint_path).expanduser()
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.append(Path.cwd() / raw_path)
        # .../poker_ai/python/poker_ai/export/onnx_export.py -> .../poker_ai/python
        python_root = Path(__file__).resolve().parents[2]
        candidates.append(python_root / raw_path)

    unique_candidates: list[Path] = []
    seen = set()
    for candidate in candidates:
        norm = str(candidate.resolve(strict=False))
        if norm in seen:
            continue
        seen.add(norm)
        unique_candidates.append(candidate.resolve(strict=False))
        if candidate.is_file():
            return candidate.resolve(strict=False)

    fallback_paths: list[str] = []
    for path in unique_candidates:
        latest = path.parent / "checkpoint_latest.pt"
        if latest.is_file():
            fallback_paths.append(str(latest))

    searched_paths = ", ".join(str(path) for path in unique_candidates)
    fallback_msg = ""
    if fallback_paths:
        fallback_msg = f" Try one of: {', '.join(sorted(set(fallback_paths)))}."

    raise FileNotFoundError(
        f"Checkpoint file not found: '{checkpoint_path}'. Searched: {searched_paths}.{fallback_msg}"
    )


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "poker_as_net.onnx",
    config: NFSPConfig | None = None,
    opset_version: int = 17,
    use_dynamo: bool = False,
) -> None:
    """Export the Average Strategy network to ONNX.

    Args:
        checkpoint_path: Path to training checkpoint (.pt file)
        output_path: Path for the output ONNX model
        config: NFSPConfig (uses default if None)
        opset_version: ONNX opset version
        use_dynamo: Use the torch.export-based ONNX exporter (requires onnxscript)
    """
    if config is None:
        config = NFSPConfig()

    checkpoint_file = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    config = infer_config_from_checkpoint(checkpoint, config)

    # Load model
    as_net = AverageStrategyNet(config)
    as_net.load_state_dict(checkpoint["as_net"])
    as_net.eval()

    wrapper = AverageStrategyONNXWrapper(as_net, max_seq_len=config.max_history_len)
    wrapper.eval()

    # Create dummy inputs
    from poker_ai.model.state_encoder import STATIC_FEATURE_SIZE
    batch_size = 1
    max_seq_len = config.max_history_len
    obs = torch.randn(batch_size, STATIC_FEATURE_SIZE)
    action_history = torch.randn(batch_size, max_seq_len, config.history_input_dim)
    history_lengths = torch.tensor([max_seq_len], dtype=torch.long)
    legal_mask = torch.ones(batch_size, config.num_actions, dtype=torch.bool)

    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Export
    try:
        # Only mark batch dimension as dynamic; keep seq_len fixed so that
        # tract-onnx (Rust) can resolve Reshape / Range nodes used by the
        # transformer's multi-head attention without symbolic dimension issues.
        dynamic = {
            "obs": {0: "batch_size"},
            "action_history": {0: "batch_size"},
            "history_lengths": {0: "batch_size"},
            "legal_mask": {0: "batch_size"},
            "action_probs": {0: "batch_size"},
        }
        torch.onnx.export(
            wrapper,
            (obs, action_history, history_lengths, legal_mask),
            str(output_file),
            input_names=["obs", "action_history", "history_lengths", "legal_mask"],
            output_names=["action_probs"],
            dynamic_axes=dynamic,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=use_dynamo,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "onnxscript" and use_dynamo:
            raise ModuleNotFoundError(
                "Dynamo ONNX export requires 'onnxscript'. Install it with "
                "`pip install onnxscript`, or rerun without `--dynamo`."
            ) from exc
        raise

    print(f"Exported ONNX model to {output_file}")


def verify_onnx(
    checkpoint_path: str,
    onnx_path: str,
    config: NFSPConfig | None = None,
    tolerance: float = 1e-5,
) -> bool:
    """Verify ONNX model outputs match PyTorch within tolerance."""
    import onnxruntime as ort

    if config is None:
        config = NFSPConfig()

    checkpoint_file = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    config = infer_config_from_checkpoint(checkpoint, config)

    # Load PyTorch model
    as_net = AverageStrategyNet(config)
    as_net.load_state_dict(checkpoint["as_net"])
    as_net.eval()

    # Create test inputs
    from poker_ai.model.state_encoder import STATIC_FEATURE_SIZE
    batch_size = 4
    max_seq_len = config.max_history_len
    obs = torch.randn(batch_size, STATIC_FEATURE_SIZE)
    action_history = torch.randn(batch_size, max_seq_len, config.history_input_dim)
    history_lengths = torch.tensor(
        [
            min(10, max_seq_len),
            min(20, max_seq_len),
            max_seq_len,
            min(5, max_seq_len),
        ],
        dtype=torch.long,
    )
    legal_mask = torch.ones(batch_size, config.num_actions, dtype=torch.bool)
    # Make some actions illegal
    legal_mask[0, 0] = False
    legal_mask[1, 8] = False

    # PyTorch output
    with torch.no_grad():
        pytorch_out = as_net(obs, action_history, history_lengths, legal_mask).numpy()

    # ONNX output
    session = ort.InferenceSession(str(Path(onnx_path).expanduser()))
    onnx_out = session.run(
        None,
        {
            "obs": obs.numpy(),
            "action_history": action_history.numpy(),
            "history_lengths": history_lengths.numpy(),
            "legal_mask": legal_mask.numpy(),
        },
    )[0]

    max_diff = np.abs(pytorch_out - onnx_out).max()
    match = max_diff < tolerance
    print(f"Max difference: {max_diff:.2e} | {'PASS' if match else 'FAIL'} (tolerance: {tolerance})")
    return match
