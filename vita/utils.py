from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml


CLASS_NAMES = ["closed", "open"]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(counts: list[int], device: torch.device) -> torch.Tensor:
    total = sum(counts)
    if total == 0 or any(c == 0 for c in counts):
        return torch.ones(len(counts), device=device)
    weights = [total / (len(counts) * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)
