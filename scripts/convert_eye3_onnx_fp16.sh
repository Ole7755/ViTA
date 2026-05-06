#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${INPUT:-exports/eye3_foodseg_convnext_tiny_128_fp32.onnx}"
OUTPUT="${OUTPUT:-exports/eye3_foodseg_convnext_tiny_128_fp16.onnx}"
META="${META:-exports/eye3_foodseg_convnext_tiny_128_fp32.json}"
PURE_FP16="${PURE_FP16:-0}"

args=(
  scripts/convert_onnx_fp16.py
  --input "${INPUT}"
  --output "${OUTPUT}"
  --metadata "${META}"
)

if [[ "${PURE_FP16}" == "1" ]]; then
  args+=(--pure-fp16)
fi

"${PYTHON_BIN}" "${args[@]}"
