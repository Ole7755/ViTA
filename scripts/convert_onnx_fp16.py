from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a FP32 ONNX model to FP16.")
    parser.add_argument("--input", required=True, help="Input FP32 ONNX path.")
    parser.add_argument("--output", required=True, help="Output FP16 ONNX path.")
    parser.add_argument("--metadata", default=None, help="Optional FP32 metadata JSON path.")
    parser.add_argument(
        "--pure-fp16",
        action="store_true",
        help="Also convert model inputs/outputs to FP16. Default keeps I/O as FP32.",
    )
    return parser.parse_args()


def convert_float_to_float16(model: onnx.ModelProto, keep_io_types: bool) -> onnx.ModelProto:
    try:
        from onnxconverter_common.float16 import convert_float_to_float16 as convert
    except ImportError as exc:
        raise RuntimeError(
            "Missing onnxconverter-common. Install it with: pip install onnxconverter-common"
        ) from exc

    return convert(model, keep_io_types=keep_io_types)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(input_path)
    converted = convert_float_to_float16(model, keep_io_types=not args.pure_fp16)
    onnx.save(converted, output_path)
    onnx.checker.check_model(str(output_path))

    input_metadata_path = Path(args.metadata) if args.metadata else input_path.with_suffix(".json")
    metadata = {}
    if input_metadata_path.exists():
        with input_metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata.update(
        {
            "onnx_path": str(output_path),
            "source_onnx_path": str(input_path),
            "precision": "fp16",
            "io_precision": "fp16" if args.pure_fp16 else "fp32",
        }
    )
    output_metadata_path = output_path.with_suffix(".json")
    with output_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"converted: {output_path}")
    print(f"metadata: {output_metadata_path}")


if __name__ == "__main__":
    main()
