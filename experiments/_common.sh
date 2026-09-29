#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

OUT_DIR="${ROOT}/results/reproduced"
mkdir -p "${OUT_DIR}"

cd "${OUT_DIR}"
