#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "=== Packed W4 / low-rank projection control ==="
python "${ROOT}/src/gnrq_linear_control.py"
