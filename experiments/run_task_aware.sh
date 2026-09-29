#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "=== Packed W4 / task-aware auxiliary ==="
python "${ROOT}/src/gnrq_packed_task_aware.py"
