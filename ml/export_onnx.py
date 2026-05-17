"""
PEBBLE ONNX Export.

Exports the trained Pebble model to ONNX format for deployment
via WebGPU / ONNX Runtime Web.

Usage:
    python export_onnx.py --checkpoint checkpoints/checkpoint_latest.pt \
                          --output pebble.onnx
"""

import argparse
import json
from pathlib import Path

import torch

from pebble.config import PebbleConfig
from pebble.model import PebbleLMHeadModel


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    seq_len: int = 512,
    opset_version: int = 17,
):
    """Export Pebble model to ONNX format.

    Args:
        checkpoint_path: Path to the trained checkpoint.
        output_path: Output .onnx file path.
        seq_len: Sequence length for the exported model.
        opset_version: ONNX opset version.
    """
    device = torch.device("cpu")

    # Load config
    config_path = Path(checkpoint_path).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        config = PebbleConfig(**{
            k: v for k, v in config_dict.items()
            if k in PebbleConfig.__dataclass_fields__
        })
    else:
        config = PebbleConfig()

    # Load model
    model = PebbleLMHeadModel(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[Pebble] Model loaded: {model.count_parameters() / 1e6:.1f}M params")

    # Create dummy input
    dummy_input = torch.randint(0, config.vocab_size, (1, seq_len), dtype=torch.long)

    # Export
    print(f"[Pebble] Exporting to ONNX (opset={opset_version})...")

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter (handles sequential scan loops)
    )

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size_mb = Path(output_path).stat().st_size / 1e6
    print(f"[Pebble] ONNX model saved: {output_path} ({file_size_mb:.1f} MB)")
    print(f"[Pebble] Model verified successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Pebble to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="pebble.onnx")
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.seq_len)
