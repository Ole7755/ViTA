#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

run_one() {
  local name="$1"
  local train_cfg="$2"
  local oceye_cfg="$3"
  local out_dir="outputs/${name}"
  local ckpt="${out_dir}/best.pt"

  echo "==> train ${name}"
  "${PYTHON_BIN}" -m vita.train --config "${train_cfg}"

  echo "==> eval MRL test ${name}"
  "${PYTHON_BIN}" -m vita.eval \
    --config "${train_cfg}" \
    --checkpoint "${ckpt}" \
    --split test \
    --output "${out_dir}/mrl_test_metrics.json"

  echo "==> eval OCE test ${name}"
  "${PYTHON_BIN}" -m vita.eval \
    --config "${oceye_cfg}" \
    --checkpoint "${ckpt}" \
    --split test \
    --output "${out_dir}/oceye_test_metrics.json"

  echo "==> benchmark GPU ${name}"
  "${PYTHON_BIN}" -m vita.benchmark \
    --config "${train_cfg}" \
    --checkpoint "${ckpt}" \
    --device cuda \
    --batch-size 1 \
    --warmup 20 \
    --steps 200 \
    --output "${out_dir}/gpu_benchmark.json"

  echo "==> benchmark CPU ${name}"
  "${PYTHON_BIN}" -m vita.benchmark \
    --config "${train_cfg}" \
    --checkpoint "${ckpt}" \
    --device cpu \
    --batch-size 1 \
    --warmup 20 \
    --steps 100 \
    --threads 16 \
    --output "${out_dir}/cpu_benchmark.json"
}

run_one "mrl_swin_t_128_finetune" \
  "configs/mrl_swin_t_128.yaml" \
  "configs/oceye_swin_t_128_eval.yaml"

run_one "mrl_convnext_tiny_128_finetune" \
  "configs/mrl_convnext_tiny_128.yaml" \
  "configs/oceye_convnext_tiny_128_eval.yaml"

run_one "mrl_resnet50_128_finetune" \
  "configs/mrl_resnet50_128.yaml" \
  "configs/oceye_resnet50_128_eval.yaml"
