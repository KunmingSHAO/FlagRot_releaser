#!/usr/bin/env bash
# Simulated KL tables (simulator env): the bit-width table (INT2 / INT3 / INT4, all encoders, 12 models) and the
# per-model INT2 comparison.  Needs the 12 models (blockflag/paths.py) and WikiText-2 (scripts/prepare_data.py).
#   GPUS=0,1,2,3 bash scripts/run_sim_tables.sh          (one job per GPU at a time; finished jobs are skipped)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python}
GPUS=${GPUS:-0}
$PY scripts/sim/sweep.py run --table unified --gpus "$GPUS"
$PY scripts/sim/sweep.py run --table int2 --gpus "$GPUS"
mkdir -p results
$PY scripts/sim/sweep.py report --table unified --json results/table_unified.json | tee results/table_unified.md
$PY scripts/sim/sweep.py report --table int2 --json results/table_int2.json | tee results/table_int2.md
