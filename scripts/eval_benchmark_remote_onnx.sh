#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ONNX="${ONNX:-exports/eye3_foodseg_convnext_tiny_128_fp32.onnx}"
META="${META:-exports/eye3_foodseg_convnext_tiny_128_fp32.json}"
OUT_DIR="${OUT_DIR:-outputs/eye3_foodseg_convnext_tiny_128_finetune/onnx}"
THREADS="${THREADS:-16}"

mkdir -p "${OUT_DIR}"

echo "==> eval ONNX eye3_foodseg test"
"${PYTHON_BIN}" scripts/eval_onnx.py \
  --config configs/eye3_foodseg_convnext_tiny_128.yaml \
  --onnx "${ONNX}" \
  --metadata "${META}" \
  --split test \
  --threads "${THREADS}" \
  --output "${OUT_DIR}/eye3_test_metrics.json"

echo "==> eval ONNX MRL test"
"${PYTHON_BIN}" scripts/eval_onnx.py \
  --config configs/remote_mrl_eye3_convnext_tiny_128_eval.yaml \
  --onnx "${ONNX}" \
  --metadata "${META}" \
  --split test \
  --threads "${THREADS}" \
  --output "${OUT_DIR}/mrl_test_metrics.json"

echo "==> eval ONNX OCE test"
"${PYTHON_BIN}" scripts/eval_onnx.py \
  --config configs/remote_oceye_eye3_convnext_tiny_128_eval.yaml \
  --onnx "${ONNX}" \
  --metadata "${META}" \
  --split test \
  --threads "${THREADS}" \
  --output "${OUT_DIR}/oceye_test_metrics.json"

echo "==> benchmark ONNX CPU"
"${PYTHON_BIN}" scripts/benchmark_onnx.py \
  --onnx "${ONNX}" \
  --metadata "${META}" \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --threads "${THREADS}" \
  --output "${OUT_DIR}/cpu_benchmark.json"

echo "saved results under ${OUT_DIR}"
