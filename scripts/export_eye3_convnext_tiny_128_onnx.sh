#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/eye3_foodseg_convnext_tiny_128.yaml}"
CKPT="${CKPT:-outputs/eye3_foodseg_convnext_tiny_128_finetune/best.pt}"
OUTPUT="${OUTPUT:-exports/eye3_foodseg_convnext_tiny_128_fp32.onnx}"
DEVICE="${DEVICE:-cpu}"
OPSET="${OPSET:-18}"
FIXED_BATCH="${FIXED_BATCH:-1}"

args=(
  scripts/export_onnx.py
  --config "${CONFIG}"
  --checkpoint "${CKPT}"
  --output "${OUTPUT}"
  --device "${DEVICE}"
  --opset "${OPSET}"
)

if [[ "${FIXED_BATCH}" == "1" ]]; then
  args+=(--fixed-batch)
fi

"${PYTHON_BIN}" "${args[@]}"
