#!/usr/bin/env bash
# Build libflagrot_kernels.so: the FlagRot KV kernels of kernels/csrc (attention + writers v1 / v3) + the flat C ABI (flagrot_capi.cu), CUDA 12.9
# toolkit, sm_90a, CUDA runtime linked statically (only the driver is shared with torch's CUDA 13 runtime).
#   integrations/vllm_flagrot/csrc/build.sh [OUT_DIR]        (FLAGROT_CUDA_HOME, FLAGROT_KERNELS_SRC override)
# vllm_flagrot/kernels.py calls this script automatically when the library is missing or older than its sources.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${FLAGROT_KERNELS_SRC:-$HERE/../../../kernels/csrc}"
SRC="$(cd "$SRC" && pwd)"
OUT="${1:-$HERE/../build}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
CUDA_HOME="${FLAGROT_CUDA_HOME:-/usr/local/cuda-12.9}"
NVCC="$CUDA_HOME/bin/nvcc"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -lineinfo -Xcompiler -fPIC
       -I"$SRC" ${FLAGROT_EXTRA_FLAGS:-})
TMP="$OUT/obj.$$"
mkdir -p "$TMP"
"$NVCC" "${FLAGS[@]}" -c "$SRC/attn_decode_v2.cu" -o "$TMP/attn_decode_v2.o" &
P1=$!
"$NVCC" "${FLAGS[@]}" -c "$SRC/attn_codebook.cu" -o "$TMP/attn_codebook.o" &
P2=$!
"$NVCC" "${FLAGS[@]}" -c "$SRC/attn_cb_write3.cu" -o "$TMP/attn_cb_write3.o" &
P3=$!
"$NVCC" "${FLAGS[@]}" -c "$HERE/flagrot_capi.cu" -o "$TMP/flagrot_capi.o"
wait $P1
wait $P2
wait $P3
"$NVCC" -shared -gencode arch=compute_90a,code=sm_90a -cudart static -o "$TMP/libflagrot_kernels.so" \
  "$TMP/flagrot_capi.o" "$TMP/attn_decode_v2.o" "$TMP/attn_codebook.o" "$TMP/attn_cb_write3.o"
mv -f "$TMP/libflagrot_kernels.so" "$OUT/libflagrot_kernels.so"
rm -rf "$TMP"
echo "$OUT/libflagrot_kernels.so"
