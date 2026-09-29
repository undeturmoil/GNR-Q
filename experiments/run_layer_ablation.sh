#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

for layer in 9 18 27
do
    echo "=== Simulated W4 / layer ${layer} ==="

    python "${ROOT}/src/gnrq_poc_spark.py" \
        --bits 4 \
        --group-size 128 \
        --layer "${layer}" \
        --rank 128 \
        --seq-len 128 \
        --train-blocks 256 \
        --eval-blocks 64 \
        --batch-size 4 \
        --local-epochs 3 \
        --e2e-epochs 1 \
        --lr 3e-4 \
        --seed 42 \
        --save "w4_layer${layer}.pt"
done
