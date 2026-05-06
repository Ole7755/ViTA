#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/eye3_foodseg_convnext_tiny_128.yaml}"
CKPT="${CKPT:-outputs/eye3_foodseg_convnext_tiny_128_finetune/best.pt}"
OUTPUT="${OUTPUT:-exports/eye3_foodseg_convnext_tiny_128_fp16.onnx}"
DEVICE="${DEVICE:-cuda}"
OPSET="${OPSET:-18}"
FIXED_BATCH="${FIXED_BATCH:-1}"
KEEP_IO_FP32="${KEEP_IO_FP32:-1}"

if ! "${PYTHON_BIN}" scripts/export_onnx.py --help | grep -q -- "--precision"; then
  echo "scripts/export_onnx.py is out of date. Sync the latest file before FP16 export." >&2
  exit 1
fi

args=(
  scripts/export_onnx.py
  --config "${CONFIG}"
  --checkpoint "${CKPT}"
  --output "${OUTPUT}"
  --device "${DEVICE}"
  --opset "${OPSET}"
  --precision fp16
)

if [[ "${FIXED_BATCH}" == "1" ]]; then
  args+=(--fixed-batch)
fi

if [[ "${KEEP_IO_FP32}" == "1" ]]; then
  args+=(--keep-io-fp32)
fi

"${PYTHON_BIN}" "${args[@]}"
