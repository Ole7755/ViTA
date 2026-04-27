#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
NAME="mrl_mobilenetv3_large_128_finetune"
TRAIN_CFG="configs/mrl_mobilenetv3_large_128.yaml"
OCEYE_CFG="configs/oceye_mobilenetv3_large_128_eval.yaml"
OUT_DIR="outputs/${NAME}"
CKPT="${OUT_DIR}/best.pt"

echo "==> train ${NAME}"
"${PYTHON_BIN}" -m vita.train --config "${TRAIN_CFG}"

echo "==> eval MRL test ${NAME}"
"${PYTHON_BIN}" -m vita.eval \
  --config "${TRAIN_CFG}" \
  --checkpoint "${CKPT}" \
  --split test \
  --output "${OUT_DIR}/mrl_test_metrics.json"

echo "==> eval OCE test ${NAME}"
"${PYTHON_BIN}" -m vita.eval \
  --config "${OCEYE_CFG}" \
  --checkpoint "${CKPT}" \
  --split test \
  --output "${OUT_DIR}/oceye_test_metrics.json"

echo "==> benchmark GPU ${NAME}"
"${PYTHON_BIN}" -m vita.benchmark \
  --config "${TRAIN_CFG}" \
  --checkpoint "${CKPT}" \
  --device cuda \
  --batch-size 1 \
  --warmup 20 \
  --steps 200 \
  --output "${OUT_DIR}/gpu_benchmark.json"

echo "==> benchmark CPU ${NAME}"
"${PYTHON_BIN}" -m vita.benchmark \
  --config "${TRAIN_CFG}" \
  --checkpoint "${CKPT}" \
  --device cpu \
  --batch-size 1 \
  --warmup 20 \
  --steps 100 \
  --threads 16 \
  --output "${OUT_DIR}/cpu_benchmark.json"
