#!/usr/bin/env bash
# Pure-decode throughput in vLLM (vLLM env, one GPU, prefix caching off, CUDA graphs on):
#   * capacity / latency curves at 32k context on the dense Qwen3-8B and Qwen3-32B: batches up to the capacity of each
#     cache (vLLM's own concurrency cap), mean time per output token over 48 decode steps;
#   * Qwen3.8-27B at 8k context, batches 64 / 32 / 8 / 1.
#   GPU=0 PY=<vLLM-env python> bash scripts/run_decode_bench.sh
# Calibration files as for the task runs (scripts/run_vllm_tasks.sh calib).  Output: results/bench/*.json
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
B=integrations/vllm_flagrot/scripts/bench_e2e.py
mkdir -p results/bench
for model in qwen8b qwen32b; do
  for mode in bf16 konlya2:s4w128:ca kivi2p8ff:s4w128 kivi2p8ffid:s4w128 kivi2p8ffs:s4w128; do
    $PY $B --model $model --mode $mode --ctx 32768 --batches 256 128 64 32 16 8 4 1 --max_seqs 256 --cap_vllm \
      --stall_s 120 --out "results/bench/curve32k_${model}_${mode//:/_}.json"
  done
done
for mode in bf16 konlya:s4w128 konlya2:s4w128:ca kivi2:s4w128 kivi2id:s4w128 kivi2p8ff:s4w128 kivi2p8ffid:s4w128; do
  $PY $B --model qwen38_27b --mode $mode --ctx 8192 --batches 64 32 8 1 --out "results/bench/decode8k_qwen38_27b_${mode//:/_}.json"
done
