from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterator


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CLASS_NAMES = ["closed", "open"]
CLASS_NAME_ALIASES = {"sleepy": "closed", "awake": "open", "close": "closed"}
DEFAULT_PREPROCESS = {
    "resize": [64, 64],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an eye-state classifier.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint.")
    model_group.add_argument("--onnx", default=None, help="Path to ONNX model.")
    parser.add_argument("--metadata", default=None, help="Optional ONNX metadata JSON path.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional ImageFolder root. Accepts either root/<split>/<class> or root/<class>.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="Optional ONNXRuntime providers, e.g. CUDAExecutionProvider CPUExecutionProvider.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Number of ONNX batches to warm up before timed evaluation.",
    )
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    return parser.parse_args()


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"PyYAML is required to read YAML config: {config_path}. "
            "Use a JSON config or omit --config for ONNX evaluation with --metadata and --data-root."
        ) from exc

    return yaml.safe_load(text)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def class_names_from_config(cfg: dict[str, Any]) -> list[str]:
    if "class_names" in cfg:
        return [str(name) for name in cfg["class_names"]]

    names_by_label: dict[int, str] = {}
    for dataset in cfg.get("data", {}).get("datasets", []):
        for class_name, label in dataset.get("class_map", {}).items():
            canonical_name = CLASS_NAME_ALIASES.get(class_name, class_name)
            names_by_label[int(label)] = canonical_name

    if not names_by_label:
        return list(CLASS_NAMES)
    return [names_by_label[label] for label in sorted(names_by_label)]


def load_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    metadata_path = Path(path)
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_onnx_metadata_path(onnx_path: str | Path, metadata_path: str | Path | None) -> Path | None:
    if metadata_path is not None:
        return Path(metadata_path)
    candidate = Path(onnx_path).with_suffix(".json")
    return candidate if candidate.exists() else None


def resolve_onnx_providers(requested_providers: list[str] | None, device_name: str | None) -> list[str]:
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    if requested_providers:
        missing = [provider for provider in requested_providers if provider not in available_providers]
        if missing:
            raise ValueError(
                f"ONNXRuntime providers not available: {missing}. "
                f"Available providers: {available_providers}"
            )
        return requested_providers

    if device_name != "cpu" and "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def resolve_preprocess(cfg: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    preprocess = dict(DEFAULT_PREPROCESS)
    preprocess.update(metadata.get("preprocess", {}))
    image_size = cfg.get("data", {}).get("image_size")
    if image_size is not None and "preprocess" not in metadata:
        preprocess["resize"] = [int(image_size), int(image_size)]
    return preprocess


def resolve_batch_size(cfg: dict[str, Any], batch_size: int | None) -> int:
    if batch_size is not None:
        return int(batch_size)
    data_cfg = cfg.get("data", {})
    return int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 512)))


def class_map_from_config(cfg: dict[str, Any], class_names: list[str]) -> dict[str, int]:
    datasets = cfg.get("data", {}).get("datasets", [])
    if datasets:
        return {
            CLASS_NAME_ALIASES.get(str(name), str(name)): int(label)
            for name, label in datasets[0].get("class_map", {}).items()
        }
    return {class_name: index for index, class_name in enumerate(class_names)}


def resolve_sample_roots(
    cfg: dict[str, Any],
    data_root: str | Path | None,
    split: str,
) -> list[Path]:
    if data_root is not None:
        root = Path(data_root)
        split_root = root / split
        return [split_root if split_root.exists() else root]

    roots = []
    for dataset in cfg.get("data", {}).get("datasets", []):
        roots.append(Path(dataset["root"]) / split)
    if not roots:
        raise ValueError("ONNX evaluation needs --data-root when --config has no dataset root.")
    return roots


def is_image_file(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS


def class_folder_names(class_name: str) -> list[str]:
    names = [class_name]
    names.extend(
        alias
        for alias, canonical_name in CLASS_NAME_ALIASES.items()
        if canonical_name == class_name and alias != class_name
    )
    return names


def collect_samples_from_root(root: Path, class_map: dict[str, int]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_name, label in class_map.items():
        for folder_name in class_folder_names(class_name):
            folder = root / folder_name
            if not folder.exists():
                continue
            for path in sorted(folder.iterdir()):
                if is_image_file(path):
                    samples.append((path, int(label)))
            break
    return samples


def collect_sample_groups(
    cfg: dict[str, Any],
    class_names: list[str],
    data_root: str | Path | None,
    split: str,
) -> dict[str, list[tuple[Path, int]]]:
    class_map = class_map_from_config(cfg, class_names)
    groups: dict[str, list[tuple[Path, int]]] = {}

    if data_root is not None:
        root = Path(data_root)
        base_root = root / split if (root / split).exists() else root
        direct_samples = collect_samples_from_root(base_root, class_map)

        for child in sorted(base_root.iterdir()) if base_root.exists() else []:
            if not child.is_dir():
                continue
            child_root = child / split if (child / split).exists() else child
            child_samples = collect_samples_from_root(child_root, class_map)
            if child_samples:
                groups[child.name] = child_samples

        if groups:
            if direct_samples:
                groups[base_root.name or "root"] = direct_samples
            return groups
        if direct_samples:
            return {base_root.name or "root": direct_samples}
    else:
        for root in resolve_sample_roots(cfg, data_root=data_root, split=split):
            samples = collect_samples_from_root(root, class_map)
            if samples:
                groups[root.parent.name if root.name == split else root.name] = samples

    if not groups:
        raise RuntimeError("No evaluation images found. Check --data-root, --split, and class folders.")
    return groups


def collect_samples(
    cfg: dict[str, Any],
    class_names: list[str],
    data_root: str | Path | None,
    split: str,
) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for group_samples in collect_sample_groups(cfg, class_names, data_root=data_root, split=split).values():
        samples.extend(group_samples)
    return samples


def preprocess_image(path: Path, image_size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))


def iter_onnx_batches(
    samples: list[tuple[Path, int]],
    batch_size: int,
    image_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    desc: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    import numpy as np

    batch_images: list[np.ndarray] = []
    batch_targets: list[int] = []
    for path, target in progress(samples, leave=False, desc=desc):
        batch_images.append(preprocess_image(path, image_size=image_size, mean=mean, std=std))
        batch_targets.append(target)
        if len(batch_images) >= batch_size:
            yield np.stack(batch_images, axis=0).astype(np.float32), np.asarray(batch_targets)
            batch_images.clear()
            batch_targets.clear()
    if batch_images:
        yield np.stack(batch_images, axis=0).astype(np.float32), np.asarray(batch_targets)


def update_confusion(confusion: list[list[int]], targets: np.ndarray, preds: np.ndarray) -> int:
    correct = 0
    for target, pred in zip(targets, preds):
        target_index = int(target)
        pred_index = int(pred)
        if 0 <= target_index < len(confusion) and 0 <= pred_index < len(confusion):
            confusion[target_index][pred_index] += 1
        correct += int(target_index == pred_index)
    return correct


def empty_confusion(class_names: list[str]) -> list[list[int]]:
    return [[0 for _ in class_names] for _ in class_names]


def add_confusion(target: list[list[int]], source: list[list[int]]) -> None:
    for row_index, row in enumerate(source):
        for column_index, value in enumerate(row):
            target[row_index][column_index] += value


def summarize_metrics(
    confusion: list[list[int]],
    correct: int,
    total: int,
    class_names: list[str],
    total_seconds: float,
    inference_seconds: float,
) -> dict[str, Any]:
    per_class_accuracy = {}
    for index, class_name in enumerate(class_names):
        class_total = sum(confusion[index])
        per_class_accuracy[class_name] = (
            confusion[index][index] / class_total if class_total > 0 else 0.0
        )
    return {
        "accuracy": correct / max(total, 1),
        "class_names": class_names,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion,
        "num_samples": total,
        "total_seconds": total_seconds,
        "inference_seconds": inference_seconds,
        "samples_per_second": total / max(total_seconds, 1e-12),
        "inference_samples_per_second": total / max(inference_seconds, 1e-12),
    }


def evaluate_onnx_file(args: argparse.Namespace, cfg: dict[str, Any], class_names: list[str]) -> dict:
    import numpy as np
    import onnxruntime as ort

    metadata_path = resolve_onnx_metadata_path(args.onnx, args.metadata)
    metadata = load_metadata(metadata_path)
    if "class_names" in metadata:
        class_names = [str(name) for name in metadata["class_names"]]

    preprocess = resolve_preprocess(cfg, metadata)
    image_size = int(preprocess["resize"][0])
    mean = np.asarray(preprocess["mean"], dtype=np.float32)
    std = np.asarray(preprocess["std"], dtype=np.float32)
    batch_size = resolve_batch_size(cfg, args.batch_size)
    sample_groups = collect_sample_groups(cfg, class_names, data_root=args.data_root, split=args.split)
    samples = [sample for group_samples in sample_groups.values() for sample in group_samples]

    providers = resolve_onnx_providers(args.providers, args.device)
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    input_name = str(metadata.get("input_name") or session.get_inputs()[0].name)
    output_name = metadata.get("output_name")
    output_names = [str(output_name)] if output_name else None

    if args.warmup_batches > 0:
        warmup_samples = samples[: batch_size * int(args.warmup_batches)]
        for images, _ in iter_onnx_batches(
            warmup_samples,
            batch_size=batch_size,
            image_size=image_size,
            mean=mean,
            std=std,
            desc="warmup-onnx",
        ):
            session.run(output_names, {input_name: images})

    confusion = empty_confusion(class_names)
    correct = 0
    total = 0
    inference_seconds = 0.0
    subsets: dict[str, dict[str, Any]] = {}
    total_start = time.perf_counter()
    for subset_name, subset_samples in sample_groups.items():
        subset_confusion = empty_confusion(class_names)
        subset_correct = 0
        subset_total = 0
        subset_inference_seconds = 0.0
        subset_start = time.perf_counter()
        for images, targets in iter_onnx_batches(
            subset_samples,
            batch_size=batch_size,
            image_size=image_size,
            mean=mean,
            std=std,
            desc=f"eval-onnx:{subset_name}",
        ):
            inference_start = time.perf_counter()
            logits = session.run(output_names, {input_name: images})[0]
            elapsed = time.perf_counter() - inference_start
            inference_seconds += elapsed
            subset_inference_seconds += elapsed
            preds = np.argmax(logits, axis=1)
            subset_correct += update_confusion(subset_confusion, targets=targets, preds=preds)
            subset_total += int(targets.shape[0])

        subset_metrics = summarize_metrics(
            confusion=subset_confusion,
            correct=subset_correct,
            total=subset_total,
            class_names=class_names,
            total_seconds=time.perf_counter() - subset_start,
            inference_seconds=subset_inference_seconds,
        )
        subsets[subset_name] = subset_metrics
        add_confusion(confusion, subset_confusion)
        correct += subset_correct
        total += subset_total

    metrics = summarize_metrics(
        confusion=confusion,
        correct=correct,
        total=total,
        class_names=class_names,
        total_seconds=time.perf_counter() - total_start,
        inference_seconds=inference_seconds,
    )
    metrics.update(
        {
            "backend": "onnxruntime",
            "providers": session.get_providers(),
            "metadata": str(metadata_path) if metadata_path else None,
            "input_name": input_name,
            "output_name": output_name or session.get_outputs()[0].name,
            "batch_size": batch_size,
            "warmup_batches": max(0, int(args.warmup_batches)),
            "num_subsets": len(subsets),
            "subsets": subsets,
        }
    )
    return metrics


def evaluate_torch_model(model: Any, loader: Any, device: Any, amp: bool, class_names: list[str]) -> dict:
    import torch
    from torch.amp import autocast

    model.eval()
    num_classes = len(class_names)
    confusion = [[0 for _ in class_names] for _ in class_names]
    correct = 0
    total = 0
    inference_seconds = 0.0
    total_start = time.perf_counter()
    with torch.inference_mode():
        for images, targets in progress(loader, leave=False, desc="eval"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_start
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            target_values = targets.detach().cpu().numpy()
            correct += update_confusion(confusion, targets=target_values, preds=preds)
            total += int(targets.size(0))

    metrics = summarize_metrics(
        confusion=confusion,
        correct=correct,
        total=total,
        class_names=class_names[:num_classes],
        total_seconds=time.perf_counter() - total_start,
        inference_seconds=inference_seconds,
    )
    return metrics


def evaluate_checkpoint(args: argparse.Namespace, cfg: dict[str, Any], class_names: list[str]) -> dict:
    if not cfg:
        raise ValueError("--config is required when evaluating a PyTorch checkpoint.")

    import torch

    from .data import build_loader
    from .models import build_model

    device_name = args.device or cfg["train"].get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "class_names" in checkpoint:
        class_names = [str(name) for name in checkpoint["class_names"]]
    model = build_model(cfg, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])

    loader = build_loader(cfg, split=args.split, train=False)
    metrics = evaluate_torch_model(
        model,
        loader,
        device,
        amp=bool(cfg["train"].get("amp", True)),
        class_names=class_names,
    )
    metrics["backend"] = "pytorch"
    metrics["device"] = str(device)
    return metrics


def validate_args(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if args.checkpoint and not cfg:
        raise ValueError("--config is required when evaluating --checkpoint.")
    if args.onnx and not cfg and args.data_root is None:
        raise ValueError("ONNX evaluation without --config needs --data-root.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    validate_args(args, cfg)
    class_names = class_names_from_config(cfg)
    metrics = (
        evaluate_onnx_file(args, cfg, class_names)
        if args.onnx
        else evaluate_checkpoint(args, cfg, class_names)
    )
    print(f"ACC: {metrics['accuracy']:.4f}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Total FPS: {metrics['samples_per_second']:.2f}")
    print(f"Inference FPS: {metrics['inference_samples_per_second']:.2f}")
    if "providers" in metrics:
        print(f"Providers: {metrics['providers']}")

    if args.output:
        save_json(metrics, Path(args.output))


if __name__ == "__main__":
    main()
