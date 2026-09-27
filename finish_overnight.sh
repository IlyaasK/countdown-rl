#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

adapter=runs/qwen35-countdown-overnight/final/adapter_model.safetensors
for ((attempt = 0; attempt < 720; attempt++)); do
  if [[ -s "$adapter" ]]; then
    break
  fi
  sleep 60
done

if [[ ! -s "$adapter" ]]; then
  echo "Overnight adapter was not saved within 12 hours" >&2
  exit 1
fi

# Allow the trainer to finish writing the other adapter metadata files.
sleep 15
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
.venv/bin/python evaluate.py \
  --reference-adapter runs/qwen35-countdown-direct-four/final \
  --adapter runs/qwen35-countdown-overnight/final \
  --candidate-label overnight-30000 \
  --output runs/qwen35-countdown-overnight/eval-512.jsonl \
  --tasks 512 --rollouts 1 --batch-size 8 --max-new-tokens 64 --direct
