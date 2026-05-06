#!/usr/bin/env bash
set -euo pipefail

ONNX="${ONNX:-exports/eye3_foodseg_convnext_tiny_128_fp16.onnx}" \
META="${META:-exports/eye3_foodseg_convnext_tiny_128_fp16.json}" \
OUT_DIR="${OUT_DIR:-outputs/eye3_foodseg_convnext_tiny_128_finetune/onnx_fp16}" \
bash scripts/eval_benchmark_remote_onnx.sh
