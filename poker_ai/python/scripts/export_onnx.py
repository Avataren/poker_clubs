#!/usr/bin/env python3
"""Export trained model to ONNX format."""

import argparse

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.export.onnx_export import export_to_onnx, verify_onnx


def main():
    parser = argparse.ArgumentParser(description="Export poker AI to ONNX")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("-o", "--output", default="poker_as_net.onnx", help="Output ONNX path")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output matches PyTorch")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    config = NFSPConfig()
    export_to_onnx(args.checkpoint, args.output, config, args.opset)

    if args.verify:
        verify_onnx(args.checkpoint, args.output, config)


if __name__ == "__main__":
    main()
