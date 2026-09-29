#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

for rank in 32 64 128 256
do
    echo "=== Simulated W4 / rank ${rank} ==="

    python "${ROOT}/src/gnrq_poc_spark.py" \
        --bits 4 \
        --group-size 128 \
        --layer 27 \
        --rank "${rank}" \
        --seq-len 128 \
        --train-blocks 256 \
        --eval-blocks 64 \
        --batch-size 4 \
        --local-epochs 3 \
        --e2e-epochs 1 \
        --lr 3e-4 \
        --seed 42 \
        --save "w4_rank${rank}.pt"
done
