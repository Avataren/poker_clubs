#!/usr/bin/env python3
"""Export trained model to ONNX format.

By default exports the Average Strategy (AS/GTO) network.
Use --br to export the Best Response (BR/shark) network instead.
Use --both to export both models at once.
"""

import argparse

from poker_ai.config.hyperparams import NFSPConfig
from poker_ai.export.onnx_export import export_to_onnx, export_br_to_onnx, verify_onnx


def main():
    parser = argparse.ArgumentParser(description="Export poker AI to ONNX")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output ONNX path (default: poker_as_net.onnx or poker_br_net.onnx)")
    parser.add_argument("--br", action="store_true",
                        help="Export BR (shark/exploiter) network instead of AS (GTO)")
    parser.add_argument("--both", action="store_true",
                        help="Export both AS and BR models")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output matches PyTorch")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="BR softmax temperature (lower = more greedy, default: 0.1)")
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use torch.export-based ONNX exporter (requires onnxscript)",
    )
    args = parser.parse_args()

    config = NFSPConfig()

    if args.both:
        as_out = args.output or "poker_as_net.onnx"
        # Derive BR output path from AS path
        br_out = as_out.replace("_as_", "_br_").replace("as_net", "br_net")
        if br_out == as_out:
            br_out = as_out.replace(".onnx", "_br.onnx")
        export_to_onnx(args.checkpoint, as_out, config, args.opset, use_dynamo=args.dynamo)
        export_br_to_onnx(args.checkpoint, br_out, config, args.opset,
                          temperature=args.temperature, use_dynamo=args.dynamo)
        if args.verify:
            verify_onnx(args.checkpoint, as_out, config)
    elif args.br:
        output = args.output or "poker_br_net.onnx"
        export_br_to_onnx(args.checkpoint, output, config, args.opset,
                          temperature=args.temperature, use_dynamo=args.dynamo)
    else:
        output = args.output or "poker_as_net.onnx"
        export_to_onnx(args.checkpoint, output, config, args.opset, use_dynamo=args.dynamo)
        if args.verify:
            verify_onnx(args.checkpoint, output, config)


if __name__ == "__main__":
    main()
