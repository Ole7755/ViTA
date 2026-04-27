#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
THREADS="${THREADS:-16}"
WARMUP="${WARMUP:-20}"
STEPS="${STEPS:-200}"

run_one() {
  local name="$1"
  local cfg="$2"
  local ckpt_dir="ckp/${name}"
  local ckpt="${ckpt_dir}/best.pt"
  local output="${ckpt_dir}/local_m4_cpu_benchmark.json"

  echo "==> benchmark local CPU ${name}"
  "${PYTHON_BIN}" -m vita.benchmark \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --device cpu \
    --batch-size 1 \
    --warmup "${WARMUP}" \
    --steps "${STEPS}" \
    --threads "${THREADS}" \
    --output "${output}"
  echo "saved ${output}"
}

run_one "mrl_swin_t_128_finetune" "configs/mrl_swin_t_128.yaml"
run_one "mrl_convnext_tiny_128_finetune" "configs/mrl_convnext_tiny_128.yaml"
run_one "mrl_resnet50_128_finetune" "configs/mrl_resnet50_128.yaml"
