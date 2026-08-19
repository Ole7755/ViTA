#!/usr/bin/env python3
"""Evaluate the four local blink-call ONNX models on the test dataset.

The default paths are the four model directories supplied for the 2026-08-15
comparison.  The script accepts both ``closed`` and ``close`` as the closed
class directory and recursively scans each subset, which is needed for the
``dataset4/video_*`` layout.

Two throughput numbers are reported:

* ``total_fps``: image loading + preprocessing + ONNX inference;
* ``inference_fps``: CPU ONNX inference time only.

The latter is useful for comparing model compute, while the former is closer
to an end-to-end application throughput number.  Predictions of the model's
``irrelevant`` class are counted as incorrect because this test set contains
only the ground-truth classes ``closed`` and ``open``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Iterator

import numpy as np
import onnxruntime as ort
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
GROUND_TRUTH_CLASSES = ("closed", "open")
CLASS_DIR_ALIASES = {
    "closed": ("closed", "close"),
    "open": ("open",),
}

DEFAULT_DATA_ROOT = Path("/Volumes/Elements/datasets/blink_data/test")
DEFAULT_MODELS = (
    (
        "ViTA",
        Path("/Users/ole/Library/Application Support/blink_call/blink_call_model_files/ViTA"),
    ),
    (
        "ConvNeXt-V2",
        Path("/Volumes/Elements/datasets/blink_data/模型权重/0815/convnext_v2/exports"),
    ),
    (
        "Swin",
        Path("/Volumes/Elements/datasets/blink_data/模型权重/0815/swin/exports"),
    ),
    (
        "Swin-V2",
        Path("/Volumes/Elements/datasets/blink_data/模型权重/0815/swin2"),
    ),
)

Sample = tuple[Path, int]


@dataclass(frozen=True)
class ModelFile:
    name: str
    path: Path
    metadata_path: Path | None


@dataclass
class Metrics:
    accuracy: float
    strict_accuracy: float
    coverage: float
    accepted_accuracy: float
    confidence_threshold: float
    total_fps: float
    inference_fps: float
    total_seconds: float
    inference_seconds: float
    num_samples: int
    confusion_matrix: list[list[int]]
    strict_confusion_matrix: list[list[int]]
    per_class_accuracy: dict[str, float]
    strict_per_class_accuracy: dict[str, float]
    subsets: dict[str, dict[str, Any]]
    providers: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate four blink-call ONNX models for ACC and FPS."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--model-dir",
        dest="model_dirs",
        type=Path,
        action="append",
        help=(
            "Model directory or ONNX file. Repeat four times to override the "
            "built-in model paths."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="A prediction is accepted only when softmax confidence is greater than this value.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Batches to run before timing each model (default: 5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate at most this many images in total; 0 means all images.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set ONNX Runtime intra-op CPU threads (mainly useful with CPU provider).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, **kwargs)


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_dir(root: Path, class_name: str) -> Path | None:
    for directory_name in CLASS_DIR_ALIASES[class_name]:
        candidate = root / directory_name
        if candidate.is_dir():
            return candidate
    return None


def collect_subset_samples(root: Path) -> list[Sample] | None:
    class_dirs = {
        class_name: find_class_dir(root, class_name)
        for class_name in GROUND_TRUTH_CLASSES
    }
    if all(class_dir is None for class_dir in class_dirs.values()):
        return None

    missing = [
        class_name
        for class_name, class_dir in class_dirs.items()
        if class_dir is None
    ]
    if missing:
        raise FileNotFoundError(
            f"Subset {root} is missing class directories: {', '.join(missing)}"
        )

    samples: list[Sample] = []
    for target, class_name in enumerate(GROUND_TRUTH_CLASSES):
        class_dir = class_dirs[class_name]
        assert class_dir is not None
        paths = image_files(class_dir)
        if not paths:
            raise RuntimeError(f"Class directory contains no images: {class_dir}")
        samples.extend((path, target) for path in paths)
    return samples


def collect_sample_groups(data_root: Path) -> dict[str, list[Sample]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    direct_samples = collect_subset_samples(data_root)
    if direct_samples is not None:
        return {data_root.name or "root": direct_samples}

    groups: dict[str, list[Sample]] = {}
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        samples = collect_subset_samples(child)
        if samples is not None:
            groups[child.name] = samples

    if not groups:
        raise RuntimeError(
            f"No valid subsets found under {data_root}; expected "
            "closed/close and open directories."
        )
    return groups


def resolve_model_file(name: str, root: Path) -> Path:
    if root.is_file():
        if root.suffix.lower() != ".onnx":
            raise ValueError(f"Model file must be an ONNX file: {root}")
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")

    candidates = sorted(root.rglob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No ONNX file found under {root}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected one ONNX file under {root} for {name}, found: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def resolve_metadata_path(model_path: Path) -> Path | None:
    sibling = model_path.with_suffix(".json")
    if sibling.is_file():
        return sibling

    json_candidates = sorted(model_path.parent.glob("*.json"))
    return json_candidates[0] if len(json_candidates) == 1 else None


def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "class_names": ["closed", "open", "irrelevant"],
            "preprocess": {
                "resize": [128, 128],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_providers() -> list[str]:
    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        raise RuntimeError(
            "CPUExecutionProvider is not available in this ONNX Runtime installation; "
            f"available providers: {available}"
        )
    return ["CPUExecutionProvider"]


def create_session(
    model_path: Path,
    providers: list[str],
    threads: int | None,
) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        if threads <= 0:
            raise ValueError(f"--threads must be positive, got {threads}")
        session_options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )


def load_batch(
    batch: list[Sample],
    image_size: tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    targets: list[int] = []
    for image_path, target in batch:
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize(
                    image_size, Image.Resampling.BILINEAR
                )
                array = np.asarray(image, dtype=np.float32) / 255.0
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc
        images.append(np.transpose((array - mean) / std, (2, 0, 1)))
        targets.append(target)
    return np.asarray(images, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def iter_batches(
    samples: list[Sample],
    batch_size: int,
    image_size: tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    description: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    batches = range(0, len(samples), batch_size)
    for start in progress(batches, desc=description, leave=False):
        yield load_batch(
            samples[start : start + batch_size],
            image_size=image_size,
            mean=mean,
            std=std,
        )


def update_confusion(
    confusion: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> int:
    valid = (
        (targets >= 0)
        & (targets < confusion.shape[0])
        & (predictions >= 0)
        & (predictions < confusion.shape[1])
    )
    np.add.at(confusion, (targets[valid], predictions[valid]), 1)
    return int(np.sum(targets == predictions))


def summarize(
    confusion: np.ndarray,
    strict_confusion: np.ndarray,
    correct: int,
    total: int,
    class_names: list[str],
    total_seconds: float,
    inference_seconds: float,
    providers: list[str],
    confidence_threshold: float,
) -> Metrics:
    per_class_accuracy: dict[str, float] = {}
    for class_index, class_name in enumerate(GROUND_TRUTH_CLASSES):
        class_total = int(confusion[class_index].sum())
        class_correct = (
            int(confusion[class_index, class_index])
            if class_index < confusion.shape[1]
            else 0
        )
        per_class_accuracy[class_name] = class_correct / max(class_total, 1)

    strict_total = int(strict_confusion.sum())
    strict_correct = sum(
        int(strict_confusion[class_index, class_index])
        for class_index, class_name in enumerate(GROUND_TRUTH_CLASSES)
        if class_index < strict_confusion.shape[1]
    )
    strict_per_class_accuracy = {
        class_name: (
            int(strict_confusion[class_index, class_index])
            / max(int(confusion[class_index].sum()), 1)
            if class_index < strict_confusion.shape[1]
            else 0.0
        )
        for class_index, class_name in enumerate(GROUND_TRUTH_CLASSES)
    }

    return Metrics(
        accuracy=correct / max(total, 1),
        strict_accuracy=strict_correct / max(total, 1),
        coverage=strict_total / max(total, 1),
        accepted_accuracy=strict_correct / max(strict_total, 1),
        confidence_threshold=confidence_threshold,
        total_fps=total / max(total_seconds, 1e-12),
        inference_fps=total / max(inference_seconds, 1e-12),
        total_seconds=total_seconds,
        inference_seconds=inference_seconds,
        num_samples=total,
        confusion_matrix=confusion.astype(int).tolist(),
        strict_confusion_matrix=strict_confusion.astype(int).tolist(),
        per_class_accuracy=per_class_accuracy,
        strict_per_class_accuracy=strict_per_class_accuracy,
        subsets={},
        providers=providers,
    )


def metrics_to_dict(metrics: Metrics) -> dict[str, Any]:
    return {
        "accuracy": metrics.accuracy,
        "raw_accuracy": metrics.accuracy,
        "strict_accuracy": metrics.strict_accuracy,
        "coverage": metrics.coverage,
        "accepted_accuracy": metrics.accepted_accuracy,
        "confidence_threshold": metrics.confidence_threshold,
        "total_fps": metrics.total_fps,
        "inference_fps": metrics.inference_fps,
        "total_seconds": metrics.total_seconds,
        "inference_seconds": metrics.inference_seconds,
        "num_samples": metrics.num_samples,
        "confusion_matrix": metrics.confusion_matrix,
        "strict_confusion_matrix": metrics.strict_confusion_matrix,
        "ground_truth_classes": list(GROUND_TRUTH_CLASSES),
        "per_class_accuracy": metrics.per_class_accuracy,
        "strict_per_class_accuracy": metrics.strict_per_class_accuracy,
        "providers": metrics.providers,
        "subsets": metrics.subsets,
    }


def evaluate_model(
    model: ModelFile,
    sample_groups: dict[str, list[Sample]],
    batch_size: int,
    warmup_batches: int,
    threads: int | None,
    confidence_threshold: float,
) -> dict[str, Any]:
    metadata = load_metadata(model.metadata_path)
    class_names = [str(name) for name in metadata.get("class_names", [])]
    if not class_names:
        class_names = ["closed", "open", "irrelevant"]
    preprocess = metadata.get("preprocess", {})
    resize = preprocess.get("resize", [128, 128])
    if len(resize) != 2:
        raise ValueError(f"Invalid preprocess.resize in {model.metadata_path}: {resize}")
    image_size = (int(resize[1]), int(resize[0]))
    mean = np.asarray(preprocess.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(preprocess.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    providers = resolve_providers()
    session = create_session(
        model.path,
        providers=providers,
        threads=threads,
    )
    input_name = str(metadata.get("input_name") or session.get_inputs()[0].name)
    output_name = str(metadata.get("output_name") or session.get_outputs()[0].name)

    all_samples = [sample for samples in sample_groups.values() for sample in samples]
    if warmup_batches > 0:
        warmup_samples = all_samples[: batch_size * warmup_batches]
        for images, _ in iter_batches(
            warmup_samples,
            batch_size=batch_size,
            image_size=image_size,
            mean=mean,
            std=std,
            description=f"warmup:{model.name}",
        ):
            session.run([output_name], {input_name: images})

    confusion = np.zeros((len(GROUND_TRUTH_CLASSES), len(class_names)), dtype=np.int64)
    strict_confusion = np.zeros_like(confusion)
    total_correct = 0
    total_samples = 0
    total_inference_seconds = 0.0
    subset_metrics: dict[str, dict[str, Any]] = {}
    total_started = time.perf_counter()

    for subset_name, samples in sample_groups.items():
        subset_confusion = np.zeros_like(confusion)
        subset_strict_confusion = np.zeros_like(confusion)
        subset_correct = 0
        subset_total = 0
        subset_inference_seconds = 0.0
        subset_started = time.perf_counter()
        for images, targets in iter_batches(
            samples,
            batch_size=batch_size,
            image_size=image_size,
            mean=mean,
            std=std,
            description=f"eval:{model.name}:{subset_name}",
        ):
            inference_started = time.perf_counter()
            logits = session.run([output_name], {input_name: images})[0]
            inference_elapsed = time.perf_counter() - inference_started
            if logits.ndim != 2 or logits.shape[1] != len(class_names):
                raise ValueError(
                    f"Unexpected output shape from {model.path}: {logits.shape}; "
                    f"expected [batch, {len(class_names)}]"
                )
            probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            predictions = np.argmax(probabilities, axis=1).astype(np.int64)
            confidences = probabilities[np.arange(len(predictions)), predictions]
            accepted = confidences > confidence_threshold
            subset_inference_seconds += inference_elapsed
            subset_correct += update_confusion(
                subset_confusion,
                targets=targets,
                predictions=predictions,
            )
            update_confusion(
                subset_strict_confusion,
                targets=targets[accepted],
                predictions=predictions[accepted],
            )
            subset_total += int(targets.shape[0])

        subset_elapsed = time.perf_counter() - subset_started
        subset_metrics[subset_name] = {
            "accuracy": subset_correct / max(subset_total, 1),
            "raw_accuracy": subset_correct / max(subset_total, 1),
            "strict_accuracy": sum(
                int(subset_strict_confusion[index, index])
                for index in range(min(len(GROUND_TRUTH_CLASSES), len(class_names)))
            )
            / max(subset_total, 1),
            "coverage": int(subset_strict_confusion.sum()) / max(subset_total, 1),
            "accepted_accuracy": sum(
                int(subset_strict_confusion[index, index])
                for index in range(min(len(GROUND_TRUTH_CLASSES), len(class_names)))
            )
            / max(int(subset_strict_confusion.sum()), 1),
            "confidence_threshold": confidence_threshold,
            "total_fps": subset_total / max(subset_elapsed, 1e-12),
            "inference_fps": subset_total / max(subset_inference_seconds, 1e-12),
            "total_seconds": subset_elapsed,
            "inference_seconds": subset_inference_seconds,
            "num_samples": subset_total,
            "confusion_matrix": subset_confusion.astype(int).tolist(),
            "strict_confusion_matrix": subset_strict_confusion.astype(int).tolist(),
            "per_class_accuracy": {
                class_name: (
                    int(subset_confusion[class_index, class_index])
                    / max(int(subset_confusion[class_index].sum()), 1)
                )
                for class_index, class_name in enumerate(GROUND_TRUTH_CLASSES)
            },
            "strict_per_class_accuracy": {
                class_name: (
                    int(subset_strict_confusion[class_index, class_index])
                    / max(int(subset_confusion[class_index].sum()), 1)
                )
                for class_index, class_name in enumerate(GROUND_TRUTH_CLASSES)
            },
        }
        confusion += subset_confusion
        strict_confusion += subset_strict_confusion
        total_correct += subset_correct
        total_samples += subset_total
        total_inference_seconds += subset_inference_seconds

    metrics = summarize(
        confusion=confusion,
        strict_confusion=strict_confusion,
        correct=total_correct,
        total=total_samples,
        class_names=class_names,
        total_seconds=time.perf_counter() - total_started,
        inference_seconds=total_inference_seconds,
        providers=session.get_providers(),
        confidence_threshold=confidence_threshold,
    )
    metrics.subsets = subset_metrics
    result = metrics_to_dict(metrics)
    result.update(
        {
            "name": model.name,
            "model": str(model.path),
            "metadata": str(model.metadata_path) if model.metadata_path else None,
            "class_names": class_names,
            "input_name": input_name,
            "output_name": output_name,
            "batch_size": batch_size,
            "warmup_batches": max(warmup_batches, 0),
            "preprocess": {
                "resize": list(resize),
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
        }
    )
    return result


def print_dataset_summary(sample_groups: dict[str, list[Sample]]) -> None:
    print("Dataset:", sum(len(samples) for samples in sample_groups.values()), "images")
    for subset_name, samples in sample_groups.items():
        counts = {
            class_name: sum(target == class_index for _, target in samples)
            for class_index, class_name in enumerate(GROUND_TRUTH_CLASSES)
        }
        print(
            f"  {subset_name}: {len(samples)} "
            f"(closed={counts['closed']}, open={counts['open']})"
        )


def print_result(result: dict[str, Any]) -> None:
    per_class = result["per_class_accuracy"]
    strict_per_class = result["strict_per_class_accuracy"]
    providers = ", ".join(result["providers"])
    print(
        f"{result['name']}: RAW={result['raw_accuracy']:.4%} | "
        f"STRICT={result['strict_accuracy']:.4%} | "
        f"COVERAGE={result['coverage']:.4%} | "
        f"ACCEPTED={result['accepted_accuracy']:.4%} | "
        f"threshold>{result['confidence_threshold']:.2f} | "
        f"total FPS={result['total_fps']:.2f} | "
        f"inference FPS={result['inference_fps']:.2f} | "
        f"closed={per_class['closed']:.4%} | open={per_class['open']:.4%} | "
        f"N={result['num_samples']} | providers={providers}"
    )
    print(
        "  strict: "
        + ", ".join(
            f"{class_name}={strict_per_class[class_name]:.4%}"
            for class_name in GROUND_TRUTH_CLASSES
        )
    )
    for subset_name, subset in result["subsets"].items():
        subset_per_class = subset["per_class_accuracy"]
        subset_strict_per_class = subset["strict_per_class_accuracy"]
        print(
            f"  {subset_name}: RAW={subset['raw_accuracy']:.4%}, "
            f"STRICT={subset['strict_accuracy']:.4%}, "
            f"COVERAGE={subset['coverage']:.4%}, "
            f"ACCEPTED={subset['accepted_accuracy']:.4%}, "
            f"closed ACC={subset_per_class['closed']:.4%}, "
            f"open ACC={subset_per_class['open']:.4%}, "
            f"total FPS={subset['total_fps']:.2f}, "
            f"inference FPS={subset['inference_fps']:.2f}, N={subset['num_samples']}"
        )
        print(
            "    strict: "
            + ", ".join(
                f"{class_name}={subset_strict_per_class[class_name]:.4%}"
                for class_name in GROUND_TRUTH_CLASSES
            )
        )


def resolve_models(model_dirs: list[Path] | None) -> list[ModelFile]:
    pairs = (
        [(path.name, path) for path in model_dirs]
        if model_dirs is not None
        else list(DEFAULT_MODELS)
    )
    models: list[ModelFile] = []
    for name, root in pairs:
        model_path = resolve_model_file(name, root)
        models.append(
            ModelFile(
                name=name,
                path=model_path,
                metadata_path=resolve_metadata_path(model_path),
            )
        )
    return models


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.warmup_batches < 0:
        raise ValueError(f"--warmup-batches cannot be negative, got {args.warmup_batches}")
    if args.limit < 0:
        raise ValueError(f"--limit cannot be negative, got {args.limit}")
    if not 0.0 <= args.confidence_threshold < 1.0:
        raise ValueError("--confidence-threshold must be >= 0 and < 1")

    sample_groups = collect_sample_groups(args.data_root)
    if args.limit:
        remaining = args.limit
        limited_groups: dict[str, list[Sample]] = {}
        for subset_name, samples in sample_groups.items():
            limited_groups[subset_name] = samples[:remaining]
            remaining -= len(limited_groups[subset_name])
            if remaining <= 0:
                break
        sample_groups = limited_groups

    models = resolve_models(args.model_dirs)
    print_dataset_summary(sample_groups)
    print("Models:")
    for model in models:
        metadata = f" (metadata: {model.metadata_path})" if model.metadata_path else ""
        print(f"  {model.name}: {model.path}{metadata}")
    print()

    results: list[dict[str, Any]] = []
    for index, model in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] Evaluating {model.name} ...")
        result = evaluate_model(
            model=model,
            sample_groups=sample_groups,
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            threads=args.threads,
            confidence_threshold=args.confidence_threshold,
        )
        results.append(result)
        print_result(result)
        print()

    report = {
        "data_root": str(args.data_root),
        "num_models": len(results),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
