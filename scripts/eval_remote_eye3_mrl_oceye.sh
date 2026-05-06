#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CKPT="${CKPT:-outputs/eye3_foodseg_convnext_tiny_128_finetune/best.pt}"
OUT_DIR="${OUT_DIR:-outputs/eye3_foodseg_convnext_tiny_128_finetune}"

echo "==> eval remote MRL test"
"${PYTHON_BIN}" -m vita.eval \
  --config configs/remote_mrl_eye3_convnext_tiny_128_eval.yaml \
  --checkpoint "${CKPT}" \
  --device "${DEVICE}" \
  --split test \
  --output "${OUT_DIR}/mrl_test_metrics.json"

echo "==> eval remote OCE test"
"${PYTHON_BIN}" -m vita.eval \
  --config configs/remote_oceye_eye3_convnext_tiny_128_eval.yaml \
  --checkpoint "${CKPT}" \
  --device "${DEVICE}" \
  --split test \
  --output "${OUT_DIR}/oceye_test_metrics.json"

echo "saved ${OUT_DIR}/mrl_test_metrics.json"
echo "saved ${OUT_DIR}/oceye_test_metrics.json"
