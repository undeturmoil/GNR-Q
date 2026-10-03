#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! docker image inspect gnrq:paper >/dev/null 2>&1; then
    echo "gnrq:paper image not found; building..."
    docker build -t gnrq:paper .
fi

docker run --rm \
  --gpus all \
  --ipc=host \
  -v "$PWD:/workspace/GNR-Q" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  gnrq:paper -lc '
    cd /workspace/GNR-Q
    export PYTHONPATH=/workspace/GNR-Q/src
    python3 src/gnrq_v5_final_validation.py \
      --model Qwen/Qwen3-4B-Base \
      --group-size 128 \
      --rank 128 \
      --seq-len 128 \
      --train-blocks 256 \
      --eval-blocks 64 \
      --c4-blocks 64 \
      --batch-size 4 \
      --epochs 4 \
      --lr 3e-4 \
      --seed 42 \
      --save results/reproduced/v5_final_validation.json
  '
