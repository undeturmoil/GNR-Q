#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

for seed in 1 7 42
do
    echo "=== Simulated W4 / seed ${seed} ==="

    python "${ROOT}/src/gnrq_poc_spark.py" \
        --bits 4 \
        --group-size 128 \
        --layer 27 \
        --rank 128 \
        --seq-len 128 \
        --train-blocks 256 \
        --eval-blocks 64 \
        --batch-size 4 \
        --local-epochs 3 \
        --e2e-epochs 1 \
        --lr 3e-4 \
        --seed "${seed}" \
        --save "w4_seed${seed}.pt"
done
