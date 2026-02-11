"""Export trained Average Strategy network to ONNX format."""

import copy
import torch
import torch.nn as nn
import numpy as np

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.model.network import AverageStrategyNet


class AverageStrategyONNXWrapper(nn.Module):
    """Wrapper that flattens the AS network interface for ONNX export.

    ONNX doesn't handle complex Python logic well, so we create a simple
    forward pass that takes all inputs and returns action probabilities.
    """

    def __init__(self, as_net: AverageStrategyNet):
        super().__init__()
        self.as_net = as_net

    def forward(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor,
        history_lengths: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep history_lengths as an explicit ONNX input for backend compatibility.
        # Current ActionHistoryMLP ignores lengths, so this is a no-op numerically.
        history_lengths_f = history_lengths.to(action_history.dtype).view(-1, 1, 1)
        action_history = action_history + history_lengths_f * 0.0
        return self.as_net(obs, action_history, history_lengths, legal_mask)


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


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "poker_as_net.onnx",
    config: NFSPConfig | None = None,
    opset_version: int = 17,
) -> None:
    """Export the Average Strategy network to ONNX.

    Args:
        checkpoint_path: Path to training checkpoint (.pt file)
        output_path: Path for the output ONNX model
        config: NFSPConfig (uses default if None)
        opset_version: ONNX opset version
    """
    if config is None:
        config = NFSPConfig()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = infer_config_from_checkpoint(checkpoint, config)

    # Load model
    as_net = AverageStrategyNet(config)
    as_net.load_state_dict(checkpoint["as_net"])
    as_net.eval()

    wrapper = AverageStrategyONNXWrapper(as_net)
    wrapper.eval()

    # Create dummy inputs
    from poker_ai.model.state_encoder import STATIC_FEATURE_SIZE
    batch_size = 1
    max_seq_len = config.max_history_len
    obs = torch.randn(batch_size, STATIC_FEATURE_SIZE)
    action_history = torch.randn(batch_size, max_seq_len, config.history_input_dim)
    history_lengths = torch.tensor([max_seq_len], dtype=torch.long)
    legal_mask = torch.ones(batch_size, config.num_actions, dtype=torch.bool)

    # Export
    torch.onnx.export(
        wrapper,
        (obs, action_history, history_lengths, legal_mask),
        output_path,
        input_names=["obs", "action_history", "history_lengths", "legal_mask"],
        output_names=["action_probs"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "action_history": {0: "batch_size", 1: "seq_len"},
            "history_lengths": {0: "batch_size"},
            "legal_mask": {0: "batch_size"},
            "action_probs": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")


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

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
    session = ort.InferenceSession(onnx_path)
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
