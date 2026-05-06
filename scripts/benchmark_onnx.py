from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime classifier throughput.")
    parser.add_argument("--onnx", required=True, help="ONNX model path.")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSON from export_onnx.py.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=None, help="Optional ONNX Runtime CPU threads.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON path.")
    return parser.parse_args()


def load_metadata(path: str | None, onnx_path: str) -> dict:
    metadata_path = Path(path) if path else Path(onnx_path).with_suffix(".json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata, args.onnx)
    input_shape = metadata.get("input_shape") or [args.batch_size, 3, 128, 128]
    image_size = int(args.image_size or input_shape[-1])

    options = ort.SessionOptions()
    if args.threads is not None:
        options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        args.onnx,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = metadata.get("input_name") or session.get_inputs()[0].name
    output_name = metadata.get("output_name") or session.get_outputs()[0].name
    x = np.random.randn(args.batch_size, 3, image_size, image_size).astype(np.float32)

    for _ in range(args.warmup):
        session.run([output_name], {input_name: x})

    started = time.perf_counter()
    for _ in range(args.steps):
        session.run([output_name], {input_name: x})
    elapsed = time.perf_counter() - started

    images = args.batch_size * args.steps
    fps = images / elapsed
    metrics = {
        "fps": fps,
        "latency_ms_per_image": elapsed * 1000.0 / images,
        "elapsed_seconds": elapsed,
        "images": images,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "image_size": image_size,
        "onnx": args.onnx,
        "providers": session.get_providers(),
        "threads": args.threads,
    }
    print(f"FPS: {fps:.2f}")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
