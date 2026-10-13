#!/usr/bin/env bash
# Task accuracy in vLLM: LongBench v1 (Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-32B, Qwen3.8-27B), RULER-128k
# (Llama-3.1-8B-Instruct) and AIME24+25 (Qwen3-8B), every mode of the eval configs, then the paired comparisons.
#
#   SIM_PY=<simulator-env python> EVAL_PY=<eval-env python> VLLM=<vLLM-env vllm binary> GPUS=0,1,2,3 \
#     bash scripts/run_vllm_tasks.sh [calib|tasks|compare]      (default: all three steps)
#
# Step calib writes the calibration files to $FLAGROT_CALIB_DIR (default ./calib); step tasks starts one vLLM server per
# GPU (serve_mode.sh) and runs lm-eval / reasoning.py against it (results/tasks/<suite>/<model>/<mode>/); finished
# tasks are skipped on a rerun.  LongBench v1 needs `python scripts/prepare_data.py longbench1` first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SIM_PY=${SIM_PY:-python}
EVAL_PY=${EVAL_PY:-python}
GPUS=${GPUS:-0}
STEPS=${1:-all}
EXP=integrations/vllm_flagrot/scripts/export_calib.py
LAUNCH=integrations/vllm_flagrot/eval/launch.py
CMP=integrations/vllm_flagrot/eval/compare.py

if [ "$STEPS" = all ] || [ "$STEPS" = calib ]; then
  G=${GPUS%%,*}
  for m in qwen8b llama8b_it qwen32b qwen38_27b; do
    CUDA_VISIBLE_DEVICES=$G $SIM_PY $EXP --model $m                                          # konlya, konlya2
    CUDA_VISIBLE_DEVICES=$G $SIM_PY $EXP --model $m --center auto --method flagrot --prot s4w128   # ':ca'
    CUDA_VISIBLE_DEVICES=$G $SIM_PY $EXP --model $m --kivi kivi                              # kivi2*, kivi2p8ff, kivi2p8ffid
    CUDA_VISIBLE_DEVICES=$G $SIM_PY $EXP --model $m --kivi kivi_sp8                          # kivi2p8ffs
  done
fi
if [ "$STEPS" = all ] || [ "$STEPS" = tasks ]; then
  $EVAL_PY $LAUNCH --suite longbench1 --gpus "$GPUS"                                          # 8B / 32B models
  $EVAL_PY $LAUNCH --suite longbench1 --models qwen38_27b --tasks lb1_chat --gpus "$GPUS"
  $EVAL_PY $LAUNCH --suite longbench1 --config longbench1_27b --gpus "$GPUS"
  $EVAL_PY $LAUNCH --suite ruler128k --gpus "$GPUS"
  $EVAL_PY $LAUNCH --suite aime --gpus "$GPUS"
fi
if [ "$STEPS" = all ] || [ "$STEPS" = compare ]; then
  for m in qwen8b llama8b_it qwen32b qwen38_27b; do $EVAL_PY $CMP --suite longbench1 --model $m; done
  $EVAL_PY $CMP --suite ruler128k --model llama8b_it
  $EVAL_PY $CMP --suite aime --model qwen8b
fi
