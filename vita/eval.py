from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.amp import autocast
from tqdm import tqdm

from .data import build_loader
from .models import build_model
from .utils import CLASS_NAMES, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an eye-state classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device, amp: bool) -> dict:
    model.eval()
    correct = 0
    total = 0
    for images, targets in tqdm(loader, leave=False, desc="eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
        preds = logits.argmax(dim=1)
        correct += int((preds == targets).sum().detach().cpu())
        total += targets.size(0)
    return {"accuracy": correct / max(total, 1)}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device_name = args.device or cfg["train"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    model = build_model(cfg, num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    loader = build_loader(cfg, split=args.split, train=False)
    metrics = evaluate(model, loader, device, amp=bool(cfg["train"].get("amp", True)))
    print(f"ACC: {metrics['accuracy']:.4f}")

    if args.output:
        save_json(metrics, Path(args.output))


if __name__ == "__main__":
    main()
