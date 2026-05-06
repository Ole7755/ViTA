from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vita.data import build_loader
from vita.utils import class_names_from_config, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ONNX classifier on an ImageFolder split.")
    parser.add_argument("--config", required=True, help="Evaluation config.")
    parser.add_argument("--onnx", required=True, help="ONNX model path.")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSON from export_onnx.py.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    parser.add_argument("--threads", type=int, default=None, help="Optional ONNX Runtime CPU threads.")
    return parser.parse_args()


def load_metadata(path: str | None, onnx_path: str) -> dict:
    metadata_path = Path(path) if path else Path(onnx_path).with_suffix(".json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_session(onnx_path: str, threads: int | None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(
        onnx_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def fixed_batch_size(session: ort.InferenceSession) -> int | None:
    batch_dim = session.get_inputs()[0].shape[0]
    if isinstance(batch_dim, int) and batch_dim > 0:
        return batch_dim
    return None


def run_batch(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    images: np.ndarray,
    model_batch_size: int | None,
) -> np.ndarray:
    if model_batch_size is None or images.shape[0] == model_batch_size:
        return session.run([output_name], {input_name: images})[0]

    outputs = []
    for start in range(0, images.shape[0], model_batch_size):
        chunk = images[start : start + model_batch_size]
        actual_size = chunk.shape[0]
        if actual_size < model_batch_size:
            padding = np.zeros((model_batch_size - actual_size, *chunk.shape[1:]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, padding], axis=0)
        logits = session.run([output_name], {input_name: chunk})[0]
        outputs.append(logits[:actual_size])
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    metadata = load_metadata(args.metadata, args.onnx)
    class_names = [str(name) for name in metadata.get("class_names", class_names_from_config(cfg))]

    session = build_session(args.onnx, args.threads)
    input_name = metadata.get("input_name") or session.get_inputs()[0].name
    output_name = metadata.get("output_name") or session.get_outputs()[0].name
    model_batch_size = fixed_batch_size(session)

    loader = build_loader(cfg, split=args.split, train=False)
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = 0
    total = 0

    for images, targets in tqdm(loader, leave=False, desc="eval-onnx"):
        logits = run_batch(
            session,
            input_name,
            output_name,
            images.numpy().astype(np.float32),
            model_batch_size,
        )
        preds = np.argmax(logits, axis=1)
        targets_np = targets.numpy()
        for target, pred in zip(targets_np, preds):
            confusion[int(target), int(pred)] += 1
        correct += int((preds == targets_np).sum())
        total += int(targets_np.shape[0])

    per_class_accuracy = {}
    for index, class_name in enumerate(class_names):
        class_total = int(confusion[index].sum())
        per_class_accuracy[class_name] = (
            float(confusion[index, index]) / class_total if class_total > 0 else 0.0
        )

    metrics = {
        "accuracy": correct / max(total, 1),
        "class_names": class_names,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "samples": total,
        "onnx": args.onnx,
        "providers": session.get_providers(),
        "threads": args.threads,
        "model_batch_size": model_batch_size,
    }
    print(f"ACC: {metrics['accuracy']:.4f}")
    if args.output:
        save_json(metrics, Path(args.output))


if __name__ == "__main__":
    main()
