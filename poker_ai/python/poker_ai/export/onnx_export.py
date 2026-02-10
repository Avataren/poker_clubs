"""Export trained Average Strategy network to ONNX format."""

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
        return self.as_net(obs, action_history, history_lengths, legal_mask)


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

    # Load model
    as_net = AverageStrategyNet(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    as_net.load_state_dict(checkpoint["as_net"])
    as_net.eval()

    wrapper = AverageStrategyONNXWrapper(as_net)
    wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    max_seq_len = 50
    obs = torch.randn(batch_size, 441)
    action_history = torch.randn(batch_size, max_seq_len, 7)
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

    # Load PyTorch model
    as_net = AverageStrategyNet(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    as_net.load_state_dict(checkpoint["as_net"])
    as_net.eval()

    # Create test inputs
    batch_size = 4
    max_seq_len = 30
    obs = torch.randn(batch_size, 441)
    action_history = torch.randn(batch_size, max_seq_len, 7)
    history_lengths = torch.tensor([10, 20, 30, 5], dtype=torch.long)
    legal_mask = torch.ones(batch_size, config.num_actions, dtype=torch.bool)
    # Make some actions illegal
    legal_mask[0, 0] = False
    legal_mask[1, 7] = False

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
