from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .models import build_model
from .utils import class_names_from_config, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an eye-state checkpoint to ONNX.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--metadata-output", default=None, help="Optional metadata JSON path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--device", default="cpu", help="Export device, usually cpu.")
    return parser.parse_args()


def build_metadata(
    cfg: dict[str, Any],
    class_names: list[str],
    input_name: str,
    output_name: str,
) -> dict[str, Any]:
    image_size = int(cfg["data"].get("image_size", 128))
    return {
        "class_names": class_names,
        "input_name": input_name,
        "output_name": output_name,
        "preprocess": {
            "resize": [image_size, image_size],
            "interpolation": "bilinear",
            "antialias": True,
            "color_order": "RGB",
            "input_scale": 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    class_names = class_names_from_config(cfg)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "class_names" in checkpoint:
        class_names = [str(name) for name in checkpoint["class_names"]]

    model = build_model(cfg, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_name = "images"
    output_name = "logits"
    image_size = int(cfg["data"].get("image_size", 128))
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=int(args.opset),
        external_data=False,
        dynamo=False,
    )

    metadata = build_metadata(
        cfg=cfg,
        class_names=class_names,
        input_name=input_name,
        output_name=output_name,
    )
    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".json")
    save_json(metadata, metadata_path)
    print(f"Saved ONNX: {output_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
