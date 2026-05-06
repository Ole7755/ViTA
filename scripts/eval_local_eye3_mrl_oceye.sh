#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
CKPT="${CKPT:-ckp/eye3_foodseg_convnext_tiny_128_finetune/best.pt}"
OUT_DIR="ckp/eye3_foodseg_convnext_tiny_128_finetune"

echo "==> eval MRL test"
"${PYTHON_BIN}" -m vita.eval \
  --config configs/local_mrl_eye3_convnext_tiny_128_eval.yaml \
  --checkpoint "${CKPT}" \
  --device "${DEVICE}" \
  --split test \
  --output "${OUT_DIR}/local_mrl_test_metrics.json"

echo "==> eval OCE test"
"${PYTHON_BIN}" -m vita.eval \
  --config configs/local_oceye_eye3_convnext_tiny_128_eval.yaml \
  --checkpoint "${CKPT}" \
  --device "${DEVICE}" \
  --split test \
  --output "${OUT_DIR}/local_oceye_test_metrics.json"

echo "saved ${OUT_DIR}/local_mrl_test_metrics.json"
echo "saved ${OUT_DIR}/local_oceye_test_metrics.json"
