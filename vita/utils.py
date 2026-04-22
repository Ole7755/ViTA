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


def update_confusion_matrix(
    matrix: list[list[int]], preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> None:
    for pred, target in zip(preds.view(-1).tolist(), targets.view(-1).tolist()):
        if 0 <= target < num_classes and 0 <= pred < num_classes:
            matrix[target][pred] += 1


def metrics_from_confusion(matrix: list[list[int]]) -> dict[str, Any]:
    num_classes = len(matrix)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(num_classes))
    per_class: dict[str, dict[str, float]] = {}
    recalls = []
    f1s = []

    for i, class_name in enumerate(CLASS_NAMES[:num_classes]):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(num_classes) if r != i)
        fn = sum(matrix[i][c] for c in range(num_classes) if c != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": sum(recalls) / num_classes if num_classes else 0.0,
        "macro_f1": sum(f1s) / num_classes if num_classes else 0.0,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def short_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"acc={metrics['accuracy']:.4f} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
