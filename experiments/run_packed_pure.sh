#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "=== Packed W4 / pure GNR-Q ==="
python "${ROOT}/src/gnrq_packed_pure.py"
