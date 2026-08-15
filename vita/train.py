from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .data import build_loader, build_loaders_by_dataset, label_counts
from .models import build_model
from .utils import (
    class_names_from_config,
    compute_class_weights,
    load_config,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an eye-state classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    amp: bool = False,
    num_classes: int | None = None,
    desc: str | None = None,
) -> tuple[float, float, dict[int, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    correct_samples = 0
    class_correct = [0 for _ in range(num_classes or 0)]
    class_total = [0 for _ in range(num_classes or 0)]

    iterator = tqdm(loader, leave=False, desc=desc or ("train" if training else "val"))
    with torch.set_grad_enabled(training):
        for images, targets in iterator:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = targets.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            preds = logits.argmax(dim=1)
            correct_samples += int((preds == targets).sum().detach().cpu())
            if num_classes is not None:
                for target, pred in zip(targets.detach().cpu(), preds.detach().cpu()):
                    target_index = int(target)
                    if 0 <= target_index < num_classes:
                        class_total[target_index] += 1
                        class_correct[target_index] += int(target_index == int(pred))
            iterator.set_postfix(
                loss=total_loss / max(total_samples, 1),
                acc=correct_samples / max(total_samples, 1),
            )

    per_class_accuracy = {
        index: class_correct[index] / class_total[index] if class_total[index] > 0 else 0.0
        for index in range(num_classes or 0)
    }
    return (
        total_loss / max(total_samples, 1),
        correct_samples / max(total_samples, 1),
        per_class_accuracy,
    )


def dataset_names_from_config(cfg: dict[str, Any]) -> list[str]:
    names: list[str] = []
    used_names: set[str] = set()
    for index, item in enumerate(cfg["data"]["datasets"], start=1):
        name = str(item.get("name") or Path(item["root"]).name or f"dataset_{index}")
        if name in used_names:
            name = f"{name}_{index}"
        used_names.add(name)
        names.append(name)
    return names


def dataset_label_sets_from_config(cfg: dict[str, Any]) -> list[set[int]]:
    return [
        {int(label) for label in item["class_map"].values()}
        for item in cfg["data"]["datasets"]
    ]


def dataset_metric_key(name: str) -> str:
    key = "".join(character if character.isalnum() or character == "_" else "_" for character in name)
    return key or "dataset"


def evaluate_dataset_loaders(
    model: nn.Module,
    loaders,
    dataset_names: list[str],
    split: str,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    num_classes: int,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for name, loader in zip(dataset_names, loaders):
        loss, accuracy, per_class_accuracy = run_epoch(
            model,
            loader,
            criterion,
            device,
            amp=amp,
            num_classes=num_classes,
            desc=f"{split}:{name}",
        )
        metrics[name] = {
            "loss": loss,
            "accuracy": accuracy,
            "per_class_accuracy": per_class_accuracy,
        }
    return metrics


def dataset_metric_fields(
    dataset_names: list[str], class_names: list[str]
) -> list[str]:
    fields: list[str] = []
    for split in ("train", "val"):
        for name in dataset_names:
            key = dataset_metric_key(name)
            fields.extend(
                [
                    f"{split}_{key}_loss",
                    f"{split}_{key}_accuracy",
                    *[f"{split}_{key}_{class_name}_accuracy" for class_name in class_names],
                ]
            )
    return fields


def flatten_dataset_metrics(
    metrics_by_split: dict[str, dict[str, dict[str, Any]]],
    dataset_names: list[str],
    class_names: list[str],
    dataset_label_sets: list[set[int]],
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for split, metrics in metrics_by_split.items():
        for dataset_index, name in enumerate(dataset_names):
            key = dataset_metric_key(name)
            dataset_metrics = metrics[name]
            row[f"{split}_{key}_loss"] = dataset_metrics["loss"]
            row[f"{split}_{key}_accuracy"] = dataset_metrics["accuracy"]
            per_class = dataset_metrics["per_class_accuracy"]
            for class_index, class_name in enumerate(class_names):
                field = f"{split}_{key}_{class_name}_accuracy"
                row[field] = (
                    per_class.get(class_index, 0.0)
                    if class_index in dataset_label_sets[dataset_index]
                    else ""
                )
    return row


def format_dataset_metrics(
    split: str,
    metrics: dict[str, dict[str, Any]],
    dataset_names: list[str],
    class_names: list[str],
    dataset_label_sets: list[set[int]],
) -> str:
    parts: list[str] = []
    for dataset_index, name in enumerate(dataset_names):
        dataset_metrics = metrics[name]
        class_parts = []
        for class_index, class_name in enumerate(class_names):
            value = dataset_metrics["per_class_accuracy"].get(class_index, 0.0)
            class_parts.append(
                f"{class_name}={value:.4f}"
                if class_index in dataset_label_sets[dataset_index]
                else f"{class_name}=na"
            )
        parts.append(
            f"{split}:{name}_acc={dataset_metrics['accuracy']:.4f} "
            + " ".join(class_parts)
        )
    return " ".join(parts)


def resolve_class_weights(
    cfg: dict[str, Any],
    class_names: list[str],
    counts: list[int],
    device: torch.device,
) -> torch.Tensor | None:
    if not bool(cfg["train"].get("class_weight", True)):
        return None

    explicit_weights = cfg["train"].get("class_weights")
    if explicit_weights is not None:
        if isinstance(explicit_weights, dict):
            weights = [float(explicit_weights.get(class_name, 1.0)) for class_name in class_names]
        else:
            weights = [float(value) for value in explicit_weights]
            if len(weights) != len(class_names):
                raise ValueError(
                    f"class_weights length {len(weights)} does not match classes {len(class_names)}"
                )
        return torch.tensor(weights, dtype=torch.float32, device=device)

    if bool(cfg["train"].get("class_weight", True)):
        return compute_class_weights(counts, device=device)
    return None


def is_early_stopping_improvement(score: float, best_score: float, min_delta: float) -> bool:
    return score > best_score + min_delta


def should_stop_early(epochs_without_improvement: int, patience: int) -> bool:
    return patience > 0 and epochs_without_improvement >= patience


def score_validation_metrics(
    class_names: list[str],
    val_metrics: dict[str, Any],
    selection_metric: str,
) -> float:
    if selection_metric == "accuracy":
        return float(val_metrics["accuracy"])
    if selection_metric == "min_closed_open":
        per_class = val_metrics["per_class_accuracy"]
        closed = float(per_class[class_names.index("closed")])
        open_ = float(per_class[class_names.index("open")])
        return min(closed, open_)
    if selection_metric == "mean_closed_open":
        per_class = val_metrics["per_class_accuracy"]
        closed = float(per_class[class_names.index("closed")])
        open_ = float(per_class[class_names.index("open")])
        return (closed + open_) / 2.0
    if selection_metric == "min_closed_open_plus_0_3_closed":
        per_class = val_metrics["per_class_accuracy"]
        closed = float(per_class[class_names.index("closed")])
        open_ = float(per_class[class_names.index("open")])
        return min(closed, open_) + 0.3 * closed
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def save_checkpoint(
    path: Path,
    cfg: dict[str, Any],
    class_names: list[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg,
            "class_names": class_names,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def serialize_dataset_metrics(
    metrics_by_split: dict[str, dict[str, dict[str, Any]]],
    class_names: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Convert integer class-index keys into readable metric keys."""

    serialized: dict[str, dict[str, dict[str, Any]]] = {}
    for split, dataset_metrics in metrics_by_split.items():
        serialized[split] = {}
        for dataset_name, metrics in dataset_metrics.items():
            per_class = metrics.get("per_class_accuracy", {})
            serialized[split][dataset_name] = {
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "per_class_accuracy": {
                    class_names[index]: float(per_class.get(index, 0.0))
                    for index in range(len(class_names))
                },
            }
    return serialized


def evaluate_best_dataset_metrics(
    best_path: Path,
    model: nn.Module,
    dataset_names: list[str],
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    num_classes: int,
    train_loaders,
    val_loaders,
    cfg: dict[str, Any],
    class_names: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate the saved best checkpoint on each configured dataset split."""

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    test_loaders = build_loaders_by_dataset(cfg, split="test", train=False)
    metrics_by_split = {
        "train": evaluate_dataset_loaders(
            model,
            train_loaders,
            dataset_names,
            "best-train",
            criterion,
            device,
            amp,
            num_classes,
        ),
        "val": evaluate_dataset_loaders(
            model,
            val_loaders,
            dataset_names,
            "best-val",
            criterion,
            device,
            amp,
            num_classes,
        ),
        "test": evaluate_dataset_loaders(
            model,
            test_loaders,
            dataset_names,
            "best-test",
            criterion,
            device,
            amp,
            num_classes,
        ),
    }
    serialized = serialize_dataset_metrics(metrics_by_split, class_names)
    best_checkpoint_metrics = dict(best_checkpoint.get("metrics", {}))
    best_checkpoint_metrics["dataset_metrics"] = serialized
    best_checkpoint["metrics"] = best_checkpoint_metrics
    torch.save(best_checkpoint, best_path)
    return serialized


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    class_names = class_names_from_config(cfg)
    set_seed(int(cfg.get("seed", 42)))

    device_name = args.device or cfg["train"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    output_dir = Path(cfg.get("output_dir", "outputs")) / cfg.get("experiment_name", "eye_cls")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output_dir / "config.json")

    train_loader = build_loader(cfg, split="train", train=True)
    val_loader = build_loader(cfg, split="val", train=False)
    report_dataset_metrics = bool(cfg["train"].get("report_dataset_metrics", False))
    save_best_dataset_metrics = bool(cfg["train"].get("save_best_dataset_metrics", True))
    track_dataset_metrics = report_dataset_metrics or save_best_dataset_metrics
    dataset_names = dataset_names_from_config(cfg) if track_dataset_metrics else []
    dataset_label_sets = dataset_label_sets_from_config(cfg) if track_dataset_metrics else []
    train_dataset_loaders = (
        build_loaders_by_dataset(cfg, split="train", train=False)
        if track_dataset_metrics
        else []
    )
    val_dataset_loaders = (
        build_loaders_by_dataset(cfg, split="val", train=False)
        if track_dataset_metrics
        else []
    )
    model = build_model(cfg, num_classes=len(class_names)).to(device)

    counts = label_counts(train_loader.dataset, num_classes=len(class_names))
    class_weights = resolve_class_weights(cfg, class_names=class_names, counts=counts, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 5e-5)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.05)),
    )
    epochs = int(cfg["train"].get("epochs", 20))
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(cfg["train"].get("amp", True))
    scaler = GradScaler("cuda", enabled=amp and device.type == "cuda")

    best_score = float("-inf")
    selection_metric = str(cfg["train"].get("selection_metric", "accuracy"))
    early_stopping_patience = int(cfg["train"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg["train"].get("early_stopping_min_delta", 0.0))
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0")
    epochs_without_improvement = 0
    history_path = output_dir / "metrics.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        val_class_fields = [f"val_{class_name}_accuracy" for class_name in class_names]
        per_dataset_fields = (
            dataset_metric_fields(dataset_names, class_names) if report_dataset_metrics else []
        )
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                *val_class_fields,
                *per_dataset_fields,
                "selection_score",
                "seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_loss, train_acc, _ = run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, amp=amp
            )
            val_loss, val_acc, val_per_class = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp=amp,
                num_classes=len(class_names),
            )
            dataset_metrics_by_split: dict[str, dict[str, dict[str, Any]]] = {}
            if report_dataset_metrics:
                dataset_metrics_by_split["train"] = evaluate_dataset_loaders(
                    model,
                    train_dataset_loaders,
                    dataset_names,
                    "train",
                    criterion,
                    device,
                    amp,
                    len(class_names),
                )
                dataset_metrics_by_split["val"] = evaluate_dataset_loaders(
                    model,
                    val_dataset_loaders,
                    dataset_names,
                    "val",
                    criterion,
                    device,
                    amp,
                    len(class_names),
                )
            scheduler.step()

            val_metrics = {
                "accuracy": val_acc,
                "per_class_accuracy": [
                    float(val_per_class.get(index, 0.0)) for index in range(len(class_names))
                ],
            }
            checkpoint_metrics = dict(val_metrics)
            if report_dataset_metrics:
                checkpoint_metrics["dataset_metrics"] = serialize_dataset_metrics(
                    dataset_metrics_by_split,
                    class_names,
                )
            score = score_validation_metrics(class_names, val_metrics, selection_metric)
            if is_early_stopping_improvement(score, best_score, early_stopping_min_delta):
                best_score = score
                epochs_without_improvement = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    cfg,
                    class_names,
                    model,
                    optimizer,
                    epoch,
                    checkpoint_metrics,
                )
            else:
                epochs_without_improvement += 1
            save_checkpoint(
                output_dir / "last.pt",
                cfg,
                class_names,
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
            )

            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                **{
                    f"val_{class_name}_accuracy": val_per_class.get(index, 0.0)
                    for index, class_name in enumerate(class_names)
                },
                **(
                    flatten_dataset_metrics(
                        dataset_metrics_by_split,
                        dataset_names,
                        class_names,
                        dataset_label_sets,
                    )
                    if report_dataset_metrics
                    else {}
                ),
                "selection_score": score,
                "seconds": time.perf_counter() - started,
            }
            writer.writerow(row)
            f.flush()
            val_class_text = " ".join(
                f"val_{class_name}_acc={val_per_class.get(index, 0.0):.4f}"
                for index, class_name in enumerate(class_names)
            )
            dataset_metric_text = ""
            if report_dataset_metrics:
                dataset_metric_text = " " + " ".join(
                    [
                        format_dataset_metrics(
                            "train",
                            dataset_metrics_by_split["train"],
                            dataset_names,
                            class_names,
                            dataset_label_sets,
                        ),
                        format_dataset_metrics(
                            "val",
                            dataset_metrics_by_split["val"],
                            dataset_names,
                            class_names,
                            dataset_label_sets,
                        ),
                    ]
                )
            print(
                "epoch={} train_loss={:.4f} train_acc={:.4f} "
                "val_loss={:.4f} val_acc={:.4f} {} selection_score={:.4f}{}".format(
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    val_class_text,
                    score,
                    dataset_metric_text,
                )
            )
            if should_stop_early(epochs_without_improvement, early_stopping_patience):
                print(
                    "early_stopping epoch={} patience={} min_delta={} best_{}={:.4f}".format(
                        epoch,
                        early_stopping_patience,
                        early_stopping_min_delta,
                        selection_metric,
                        best_score,
                    )
                )
                break

    if save_best_dataset_metrics:
        best_dataset_metrics = evaluate_best_dataset_metrics(
            best_path=output_dir / "best.pt",
            model=model,
            dataset_names=dataset_names,
            criterion=criterion,
            device=device,
            amp=amp,
            num_classes=len(class_names),
            train_loaders=train_dataset_loaders,
            val_loaders=val_dataset_loaders,
            cfg=cfg,
            class_names=class_names,
        )
        save_json(
            {
                "checkpoint": str(output_dir / "best.pt"),
                "epoch": int(torch.load(output_dir / "best.pt", map_location="cpu")["epoch"]),
                "dataset_metrics": best_dataset_metrics,
            },
            output_dir / "best_dataset_metrics.json",
        )
        print("best checkpoint dataset metrics:")
        for split, dataset_metrics in best_dataset_metrics.items():
            print(
                "  {}: {}".format(
                    split,
                    " ".join(
                        f"{name}_acc={metrics['accuracy']:.4f}"
                        for name, metrics in dataset_metrics.items()
                    ),
                )
            )

    print("best_{}={:.4f}".format(selection_metric, best_score))
    print("output_dir={}".format(output_dir))


if __name__ == "__main__":
    main()
