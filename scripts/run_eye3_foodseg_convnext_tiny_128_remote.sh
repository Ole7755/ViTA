#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
NAME="eye3_foodseg_convnext_tiny_128_finetune"
CONFIG="configs/eye3_foodseg_convnext_tiny_128.yaml"
OUT_DIR="outputs/${NAME}"
CKPT="${OUT_DIR}/best.pt"

echo "==> train ${NAME}"
"${PYTHON_BIN}" -m vita.train --config "${CONFIG}"

echo "==> eval test ${NAME}"
"${PYTHON_BIN}" -m vita.eval \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split test \
  --output "${OUT_DIR}/test_metrics.json"

echo "==> benchmark GPU ${NAME}"
"${PYTHON_BIN}" -m vita.benchmark \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --steps 200 \
  --output "${OUT_DIR}/gpu_benchmark.json"

echo "==> benchmark CPU ${NAME}"
"${PYTHON_BIN}" -m vita.benchmark \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --device cpu \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --threads 16 \
  --output "${OUT_DIR}/cpu_benchmark.json"
