from __future__ import annotations

import argparse
import time

import torch
from torch.cuda.amp import autocast

from .models import build_model
from .utils import CLASS_NAMES, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure model inference throughput.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    image_size = int(cfg["data"].get("image_size", 224))

    model = build_model(cfg, num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()

    x = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    for _ in range(args.warmup):
        with autocast(enabled=amp):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(args.steps):
        with autocast(enabled=amp):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    images = args.batch_size * args.steps
    fps = images / elapsed
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
