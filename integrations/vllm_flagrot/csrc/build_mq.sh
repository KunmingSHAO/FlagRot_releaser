#!/usr/bin/env bash
# Build libflagrot_mq.so (Phase 2i prefix-cached prefill attention on the INT4 pages, csrc/mq_prefill.cu): CUDA 12.9
# toolkit, sm_90a, static CUDA runtime (as build.sh).   integrations/vllm_flagrot/csrc/build_mq.sh [OUT_DIR]
# vllm_flagrot/prefill_mq.py calls this script automatically when the library is missing or older than its source.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../build}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
CUDA_HOME="${FLAGROT_CUDA_HOME:-/usr/local/cuda-12.9}"
TMP="$OUT/mqobj.$$"
mkdir -p "$TMP"
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -lineinfo \
  -Xcompiler -fPIC -Xptxas -v ${FLAGROT_EXTRA_FLAGS:-} -shared -cudart static -o "$TMP/libflagrot_mq.so" \
  "$HERE/mq_prefill.cu" 2> "$TMP/ptxas.log" || { cat "$TMP/ptxas.log"; exit 1; }
grep -E "registers|spill" "$TMP/ptxas.log" >&2 || true
mv -f "$TMP/libflagrot_mq.so" "$OUT/libflagrot_mq.so"
rm -rf "$TMP"
echo "$OUT/libflagrot_mq.so"
