#!/usr/bin/env bash
# OpenAI-compatible `vllm serve` for one (model, KV mode) on one GPU (task-level evaluation, eval/launch.py).
#
#   serve_mode.sh <model> <kv_mode[:W]> <gpu> <port> [-- cmd ...]
#
#   model   : a blockflag.paths model key (qwen8b | llama8b_it | qwen32b | qwen38_27b | ...) or a model directory
#   kv_mode : bf16 | fp8 | int4pth (vLLM int4_per_token_head) | rtn_a (plain INT4, our kernel) | static_a (FlagRot-S)
#             | konlya (FlagRot, key-only 2x2, main) | konlya_c (FlagRot-C: konlya on the calibrated WUSH base)
#             | rtn_a2 | static_a2 | konlya2 | konlya_c2 (Phase 3a: the same four with INT2 pages, int2a:32)
#             | kivi2 | kivi2c (kivi2: FlagRot-KIVI / FlagRot-KIVI-C INT2, only with a window >= 32: kivi2:s4w128)
#             | kivi2id (plain KIVI INT2 in the same pages / kernels: the matched control of kivi2)
#             suffix (plugin modes only): ':W' or ':wW' = BF16 staging window, ':sS' = S BF16 sink tokens, ':sSwW' both,
#             e.g. konlya:128, konlya_c:s4w128
#             ':c' (int2c, plugin modes): static post-RoPE key centering (option "center" = flagrot_center_<model>.pt;
#             static_a2 / static_a / konlya_c2 / konlya_c take their centered calibrations *_a2c / *_ac / calib_cc),
#             e.g. konlya2:s4w128:c, rtn_a2:s4w128:c, konlya2:c (W = 0); ':ca' = the research kv_center = auto
#             (konlya2 / rtn_a2 at S4W128: flagrot_center_auto_<flagrot|rtn>_s4w128_<model>.pt, mu = 0 = off)
#             | kivi2p8ff / kivi2p8ffid / kivi2p8ffs (2.5-bit FlagRot-KIVI per key / KIVI only / fixed selection), see below
#             aliases (README): flagrot-int4 = konlya:s4w128, flagrot-int2-auto = konlya2:s4w128:ca,
#             flagrot-kivi-3.0 / kivi-3.0 = kivi2 / kivi2id:s4w128, flagrot-kivi-2.5 / kivi-2.5 / flagrot-kivi-2.5-fixed =
#             kivi2p8ff / kivi2p8ffid / kivi2p8ffs:s4w128
#   without a command : start the server detached, wait for /health, write $RUN_DIR/server.{pid,log,json}, exit 0
#                       (stop with `serve_mode.sh stop <port>`)
#   with `-- cmd ...` : start, wait, run cmd (env OPENAI_BASE_URL / SERVER_URL point at the server), scrape /metrics,
#                       stop the server; exit status = the command's
#   serve_mode.sh stop <port> : stop a detached server started by this script (process group from its pid file)
#
# env: MAX_MODEL_LEN (32768), GPU_MEM (0.90), MAX_NUM_SEQS (128), MAX_BATCHED (16384), PREFIX_CACHE (auto = on,
#      also with sinks / window since Phase 2k; 0 = off), SERVED_NAME (model key), TOOLS=1 (auto tool choice with
#      the model's parser), REASONING=1 (reasoning parser, Qwen3), THINK=0 (Qwen3 non-thinking default), METRICS_OUT (JSON written at the end / on stop),
#      SAMPLE_S (period of the background /metrics sampler, 5; 0 = off), RUN_DIR (results/serve/<name>_<port>),
#      CALIB_DIR (calibration files, default $FLAGROT_CALIB_DIR or ./calib of this checkout), EXTRA (more args),
#      CG_SIZES (CUDA-graph capture sizes, e.g. "1 2 4 8 16 32 64 128"), HEALTH_TIMEOUT (1800 s), VLLM (vllm binary, default
#      `vllm` on PATH), PYTHON (python with blockflag importable, for model-key lookup; default python3)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
VLLM=${VLLM:-vllm}
PYTHON=${PYTHON:-python3}
SCRAPE="$HERE/metrics_scrape.py"

stop_port() {   # stop the server of this script on port $1 (only process groups recorded in our pid files)
  local pf reg="$ROOT/results/serve/.ports/$1"
  [ -f "$reg" ] || return 0
  for pf in "$(cat "$reg")/server.pid"; do
    [ -f "$pf" ] || continue
    local pgid; pgid=$(cat "$pf")
    local d; d=$(dirname "$pf")
    if kill -0 "$pgid" 2>/dev/null; then
      [ -f "$d/metrics_out" ] && python3 "$SCRAPE" --port "$1" --out "$(cat "$d/metrics_out")" \
        --log "$d/server.log" --samples "$d/samples.jsonl" 2>/dev/null
      [ -f "$d/sampler.pid" ] && kill "$(cat "$d/sampler.pid")" 2>/dev/null
      kill -TERM -- "-$pgid" 2>/dev/null
      for _ in $(seq 60); do kill -0 "$pgid" 2>/dev/null || break; sleep 1; done
      kill -KILL -- "-$pgid" 2>/dev/null
    fi
    rm -f "$pf"
  done
  rm -f "$reg"
}
if [ "${1:-}" = stop ]; then stop_port "$2"; exit 0; fi

[ $# -ge 4 ] || { sed -n 2,22p "$0"; exit 2; }
MODEL=$1; MODE=$2; GPU=$3; PORT=$4; shift 4
[ "${1:-}" = "--" ] && shift
case $MODE in      # friendly aliases (README, mode table): the paper's configurations
  flagrot-int4) MODE=konlya:s4w128 ;;
  flagrot-int2-auto) MODE=konlya2:s4w128:ca ;;
  flagrot-kivi-3.0) MODE=kivi2:s4w128 ;;
  kivi-3.0) MODE=kivi2id:s4w128 ;;
  flagrot-kivi-2.5) MODE=kivi2p8ff:s4w128 ;;
  kivi-2.5) MODE=kivi2p8ffid:s4w128 ;;
  flagrot-kivi-2.5-fixed) MODE=kivi2p8ffs:s4w128 ;;
esac
CENTER=0
if [[ ":$MODE:" == *":c:"* ]]; then CENTER=1; MODE=$(echo "$MODE" | sed -E 's/:c(:|$)/\1/'); fi
if [[ ":$MODE:" == *":ca:"* ]]; then CENTER=auto; MODE=$(echo "$MODE" | sed -E 's/:ca(:|$)/\1/'); fi
KV=${MODE%%:*}; W=0; S=0
if [[ $MODE == *:* ]]; then
  SUF=${MODE#*:}
  if [[ $SUF =~ ^[0-9]+$ ]]; then W=$SUF
  elif [[ $SUF =~ ^(s([0-9]+))?(w([0-9]+))?$ ]]; then S=${BASH_REMATCH[2]:-0}; W=${BASH_REMATCH[4]:-0}
  else echo "bad mode suffix $SUF (W | wW | sS | sSwW)"; exit 2; fi
fi

case $MODEL in      # tool-call / reasoning parsers per model key
  qwen8b|qwen14b|qwen32b) TOOLP=hermes; REASP=qwen3 ;;
  qwen38_27b) TOOLP=qwen3_coder; REASP=qwen3 ;;
  llama8b_it) TOOLP=llama3_json; REASP= ;;
  mistral7b) TOOLP=mistral; REASP= ;;
  *) TOOLP=${TOOL_PARSER:-}; REASP=${REASONING_PARSER:-} ;;
esac
if [ -d "$MODEL" ]; then MPATH=$MODEL; MODEL=$(basename "$MODEL")
else MPATH=$(cd "$ROOT" && PYTHONPATH="$ROOT" "$PYTHON" -m blockflag.paths "$MODEL") || { echo "unknown model $MODEL"; exit 2; }
fi
CALIB_DIR=${CALIB_DIR:-${FLAGROT_CALIB_DIR:-$ROOT/calib}}

CEN_FILE=""      # int2c: key centering (':c' post, ':ca' auto), checked before the server starts
if [ "$CENTER" = 1 ]; then
  CEN_FILE=$CALIB_DIR/flagrot_center_$MODEL.pt
elif [ "$CENTER" = auto ]; then   # research kv_center = auto (rtn / flagrot, S4W128 or W0): mu = 0 means no centering
  case $KV in konlya2) METH=flagrot ;; rtn_a2) METH=rtn ;; *) echo "':ca' only for rtn_a2 / konlya2 (the INT2 auto choice)"; exit 2 ;; esac
  if [ "$S$W" = 4128 ]; then PROT=s4w128; elif [ "$S$W" = 00 ]; then PROT=w0; else echo "':ca' needs s4w128 or w0"; exit 2; fi
  CEN_FILE=$CALIB_DIR/flagrot_center_auto_${METH}_${PROT}_$MODEL.pt
fi
if [ -n "$CEN_FILE" ] && [ ! -f "$CEN_FILE" ]; then echo "center file $CEN_FILE missing (export_calib.py --center)"; exit 2; fi
fr() {  # plugin config: kv_cache_dtype, calib file, extra json (+ key centering, int2c)
  local cal=$2 cen=""
  if [ "$CENTER" = 1 ]; then      # key-dependent calibrations made on the centered keys
    case $cal in flagrot_static_a2) cal=flagrot_static_a2c ;; flagrot_static_a) cal=flagrot_static_ac ;;
                 flagrot_calib_c) cal=flagrot_calib_cc ;; esac
  fi
  [ -n "$CEN_FILE" ] && cen=", \"center\": \"$CEN_FILE\""
  echo "{\"flagrot\": {\"kv_cache_dtype\": \"$1\", \"calib\": \"$CALIB_DIR/${cal}_$MODEL.pt\", \"window\": $W, \"sink\": $S$3$cen}}"
}
case $KV in
  bf16) KVARGS=(--kv-cache-dtype auto) ;;
  fp8) KVARGS=(--kv-cache-dtype fp8) ;;
  int4pth) KVARGS=(--kv-cache-dtype int4_per_token_head) ;;
  rtn_a) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_int4_static flagrot_static_rtn_a ', "quant": "int4a"')") ;;
  static_a) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_int4_static flagrot_static_a ', "quant": "int4a"')") ;;
  konlya) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_int4 flagrot_calib ', "quant": "int4a", "kv_dyn": "cq+f+konly"')") ;;
  konlya_c) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_int4 flagrot_calib_c ', "quant": "int4a", "kv_dyn": "cq+f+konly"')") ;;
  # Phase 3a: INT2 pages (int2a:32 stored as int2ah:32, 2.5 bits per element); same calibration files except FlagRot-S
  # (headsel calibrated for int2a:32: flagrot_static_a2_<model>.pt, export_calib.py --static --fmt int2a:32)
  rtn_a2) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_int4_static flagrot_static_rtn_a ', "quant": "int2a"')") ;;
  static_a2) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_int4_static flagrot_static_a2 ', "quant": "int2a"')") ;;
  konlya2) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_int4 flagrot_calib ', "quant": "int2a", "kv_dyn": "cq+f+konly"')") ;;
  konlya_c2) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_int4 flagrot_calib_c ', "quant": "int2a", "kv_dyn": "cq+f+konly"')") ;;
  # kivi2: FlagRot-KIVI INT2 (KIVI per-channel keys over 32-token groups + per-group / per-token transform choice,
  # simulator kivi_sel; needs a window >= 32, e.g. kivi2:s4w128); kivi2c = FlagRot-KIVI-C (calibrated WUSH base,
  # calib_rot 32768).  Calibration flagrot_kivi[_c]_<model>.pt (export_calib.py --kivi kivi | kivi_c)
  kivi2) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi '')") ;;
  kivi2c) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi_c '')") ;;
  # kivi2id: plain KIVI (identity candidate only, no selection), the matched control of kivi2 (same pages / reader)
  kivi2id) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_id": 1')") ;;
  # kivi2s: static selection (same candidates / criterion as kivi2, frozen per layer / KV head / slot on the
  # calibration windows: flagrot_kivi_s_<model>.pt, export_calib.py --kivi kivi_s), the control of write-time selection
  kivi2s) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi_s ', "kivi_static": 1')") ;;
  # kivi2p8 / kivi2p8id: the 2.5-bit packing kivi2f8:32 (FP8 E4M3 scale + minimum, 80 B per token and KV head at head
  # 128), FlagRot-KIVI selection / KIVI only (research kivi_sel_p8 / kivi_p8); calibration flagrot_kivi_<model>.pt
  kivi2p8) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_f8": 1')") ;;
  kivi2p8id) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_f8": 1, "kivi_id": 1')") ;;
  # kivi2p8f / kivi2p8fid / kivi2f: kivi2p8 / kivi2p8id / kivi2 caches with the fast decode-attention kernel
  # (kivi_fast: same math, accumulation-order differences only)
  kivi2p8f) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_f8": 1, "kivi_fast": 1')") ;;
  kivi2p8fid) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_f8": 1, "kivi_id": 1, "kivi_fast": 1')") ;;
  kivi2f) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_fast": 1')") ;;
  # kivi2p8ff / kivi2p8ffid: kivi2p8f + fast prefix dequantization, fused LSE merge, faster key-group encoder
  # (kivi_ff: identical K / V, codes and flags)
  kivi2p8ff) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_f8": 1, "kivi_ff": 1')") ;;
  kivi2p8ffid) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_f8": 1, "kivi_id": 1, "kivi_ff": 1')") ;;
  # kivi2ff / kivi2ffid: the same for the 3.0-bit kivi2 / kivi2id caches (FP16 meta)
  kivi2ff) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_ff": 1')") ;;
  kivi2ffid) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_id": 1, "kivi_ff": 1')") ;;
  # kivi2h / kivi2hid: kivi2 / kivi2id with this tree's host-side plan fix only (old kernels, old dequant)
  kivi2h) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi '')") ;;
  kivi2hid) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi ', "kivi_id": 1')") ;;
  # kivi2p8ffs: the 2.5-bit static (fixed) selection, flags frozen per (layer, KV head, slot) under kivi2f8:32
  # (calibration flagrot_kivi_sp8_<model>.pt, export_calib.py --kivi kivi_sp8; research kivi_sel_static_p8), with the
  # kivi2p8ff paths
  kivi2p8ffs) KVARGS=(--attention-backend CUSTOM --kv-cache-dtype auto --additional-config "$(fr flagrot_kivi flagrot_kivi_sp8 ', "kivi_f8": 1, "kivi_static": 1, "kivi_ff": 1')") ;;
  *) echo "unknown kv mode $KV"; exit 2 ;;
esac
# kivi2: prefix-cache blocks of 32 tokens = whole KIVI key groups (a cached block is always fully encoded); hybrid
# models (qwen38_27b) raise the attention block to a multiple of 32 (the backend checks it)
case $KV in kivi2|kivi2c|kivi2id|kivi2s|kivi2p8|kivi2p8id|kivi2p8f|kivi2p8fid|kivi2f|kivi2p8ff|kivi2p8ffid|kivi2ff|kivi2ffid|kivi2h|kivi2hid|kivi2p8ffs) EXTRA="--block-size 32 ${EXTRA:-}" ;; esac
case $KV in bf16|fp8|int4pth) [ "$W" = 0 ] && [ "$S" = 0 ] && [ "$CENTER" = 0 ] || { echo "window / sinks / centering only for plugin modes"; exit 2; } ;; esac

PREFIX_CACHE=${PREFIX_CACHE:-auto}
# the plugin's BF16 window / sink tokens work with prefix caching (vllm_flagrot/prefix_cache.py): on for
# every mode (PREFIX_CACHE=0, or a ':nopc' mode suffix in eval/launch.py, turns it off)
[ "$PREFIX_CACHE" = auto ] && PREFIX_CACHE=1
[ "$PREFIX_CACHE" = 1 ] && PC=--enable-prefix-caching || PC=--no-enable-prefix-caching
OPT=()
[ "${TOOLS:-0}" = 1 ] && [ -n "$TOOLP" ] && OPT+=(--enable-auto-tool-choice --tool-call-parser "$TOOLP")
# Llama 3.1 JSON tool calls need vLLM's tool chat template: pass it with EXTRA="--chat-template <file>"
[ "${REASONING:-0}" = 1 ] && [ -n "$REASP" ] && OPT+=(--reasoning-parser "$REASP")
# CG_SIZES="1 2 4 8 16 32 64 128": fewer CUDA-graph capture sizes (vLLM default ~50); the plugin's piecewise capture is
# CPU-bound (5 s per size on an idle host, 20 s under heavy CPU load), so this cuts startup from minutes to ~1 min
if [ -n "${CG_SIZES:-}" ]; then
  OPT+=(--compilation-config "{\"cudagraph_capture_sizes\": [$(echo $CG_SIZES | tr ' ' ',')]}")
fi
# THINK=0: Qwen3 non-thinking by default (a request can still pass chat_template_kwargs.enable_thinking = true)
[ "${THINK:-}" = 0 ] && [ -n "$REASP" ] && OPT+=(--default-chat-template-kwargs '{"enable_thinking": false}')
SERVED_NAME=${SERVED_NAME:-$MODEL}
NAME=${MODEL}_${KV}_w${W}
[ "$S" = 0 ] || NAME=${MODEL}_${KV}_s${S}_w${W}
[ "$CENTER" = 1 ] && NAME=${NAME}_c
[ "$CENTER" = auto ] && NAME=${NAME}_ca
RUN_DIR=${RUN_DIR:-$ROOT/results/serve/${NAME}_${PORT}}
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/server.log

export PYTHONPATH="$ROOT/integrations/vllm_flagrot${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=$GPU
unset all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1,${no_proxy:-}" NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"

if curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then echo "port $PORT already serves"; exit 3; fi
CMD=("$VLLM" serve "$MPATH" --served-model-name "$SERVED_NAME" "$MPATH" --host 127.0.0.1 --port "$PORT" --dtype bfloat16
     --max-model-len "${MAX_MODEL_LEN:-32768}" --gpu-memory-utilization "${GPU_MEM:-0.90}"
     --max-num-seqs "${MAX_NUM_SEQS:-128}" --max-num-batched-tokens "${MAX_BATCHED:-16384}" $PC --seed 0
     "${KVARGS[@]}" "${OPT[@]}" ${EXTRA:-})
printf '%q ' "${CMD[@]}" > "$RUN_DIR/cmd.txt"; echo >> "$RUN_DIR/cmd.txt"
setsid "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$RUN_DIR/server.pid"
mkdir -p "$ROOT/results/serve/.ports"; echo "$RUN_DIR" > "$ROOT/results/serve/.ports/$PORT"
python3 - "$RUN_DIR/server.json" <<EOF
import json, sys, time
json.dump(dict(model="$MODEL", model_path="$MPATH", kv="$KV", window=$W, sink=$S, center="$CENTER", gpu="$GPU", port=$PORT, pid=$SPID,
               served_name="$SERVED_NAME", prefix_caching=bool($PREFIX_CACHE), max_model_len=${MAX_MODEL_LEN:-32768},
               gpu_mem=${GPU_MEM:-0.90}, max_num_seqs=${MAX_NUM_SEQS:-128}, calib_dir="$CALIB_DIR",
               started=time.strftime("%Y-%m-%d %H:%M:%S")), open(sys.argv[1], "w"), indent=1)
EOF
T0=$(date +%s)
until curl -sf "localhost:$PORT/health" >/dev/null 2>&1; do
  sleep 3
  if ! kill -0 $SPID 2>/dev/null; then echo "server $NAME died:"; tail -20 "$LOG"; rm -f "$RUN_DIR/server.pid"; exit 4; fi
  if [ $(( $(date +%s) - T0 )) -gt "${HEALTH_TIMEOUT:-1800}" ]; then echo "health timeout"; stop_port "$PORT"; exit 5; fi
done
echo "ready $NAME port $PORT gpu $GPU after $(( $(date +%s) - T0 )) s; $(grep -E 'KV cache size' "$LOG" | tail -1 | sed 's/.*\] //')"
[ -n "${METRICS_OUT:-}" ] && echo "$METRICS_OUT" > "$RUN_DIR/metrics_out"
if [ "${SAMPLE_S:-5}" != 0 ]; then
  setsid python3 "$SCRAPE" --port "$PORT" --sample "${SAMPLE_S:-5}" --samples "$RUN_DIR/samples.jsonl" \
    > /dev/null 2>&1 < /dev/null &
  echo $! > "$RUN_DIR/sampler.pid"
fi
[ $# -eq 0 ] && exit 0

export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" SERVER_URL="http://127.0.0.1:$PORT" SERVED_NAME
"$@"
RC=$?
stop_port "$PORT"
exit $RC
