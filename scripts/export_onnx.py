from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vita.models import build_model
from vita.utils import class_names_from_config, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained classifier checkpoint to ONNX.")
    parser.add_argument("--config", required=True, help="Path to model config.")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fixed-batch", action="store_true", help="Disable dynamic batch axis.")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument(
        "--keep-io-fp32",
        action="store_true",
        help="For FP16 export, keep ONNX model inputs and outputs as FP32.",
    )
    return parser.parse_args()


class PrecisionWrapper(nn.Module):
    def __init__(self, model: nn.Module, precision: str, keep_io_fp32: bool):
        super().__init__()
        self.model = model
        self.precision = precision
        self.keep_io_fp32 = keep_io_fp32

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp16" and self.keep_io_fp32:
            logits = self.model(images.to(torch.float16))
            return logits.to(torch.float32)
        return self.model(images)


def export_model(model: nn.Module, dummy: torch.Tensor, output_path: Path, args, dynamic_axes) -> None:
    export_kwargs = {
        "export_params": True,
        "opset_version": args.opset,
        "do_constant_folding": True,
        "input_names": ["images"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "dynamo": False,
    }
    try:
        torch.onnx.export(model, dummy, output_path, **export_kwargs)
    except TypeError:
        export_kwargs.pop("dynamo", None)
        torch.onnx.export(model, dummy, output_path, **export_kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    class_names = class_names_from_config(cfg)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "class_names" in checkpoint:
        class_names = [str(name) for name in checkpoint["class_names"]]

    image_size = int(cfg["data"].get("image_size", 128))
    model = build_model(cfg, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    if args.precision == "fp16":
        model = model.half()
    model.eval()
    export_model_obj: nn.Module = PrecisionWrapper(model, args.precision, args.keep_io_fp32).to(device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_dtype = torch.float32 if args.keep_io_fp32 else torch.float16 if args.precision == "fp16" else torch.float32
    dummy = torch.randn(args.batch_size, 3, image_size, image_size, device=device, dtype=dummy_dtype)
    dynamic_axes = None
    if not args.fixed_batch:
        dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}}

    export_model(export_model_obj, dummy, output_path, args, dynamic_axes)

    metadata = {
        "onnx_path": str(output_path),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "model_name": cfg["model"]["name"],
        "class_names": class_names,
        "input_name": "images",
        "output_name": "logits",
        "input_shape": [args.batch_size, 3, image_size, image_size],
        "dynamic_batch": not args.fixed_batch,
        "opset": args.opset,
        "precision": args.precision,
        "io_precision": "fp32" if args.precision == "fp16" and args.keep_io_fp32 else args.precision,
        "preprocess": {
            "resize": [image_size, image_size],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"exported: {output_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
