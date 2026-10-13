# FlagRot: Selective Rotation for Low-Bit KV Caches

This repository contains the code for the paper *FlagRot: Selective Rotation for Low-Bit KV Caches*:

* **`blockflag/`** — the simulator. It runs Hugging Face `transformers` models with a fake-quantized KV cache
  (custom attention function) and measures KL(BF16 ‖ quantized) of the next-token distributions on WikiText-2.
  It implements FlagRot (a per-token / per-group choice between the identity, Walsh–Hadamard transforms of 32-dim
  sub-vectors and a whole-head Hadamard, chosen by a calibrated attention-weighted error criterion; the choice is stored
  in the sign bits of the scale words), FlagRot-KIVI (the same selection on KIVI's per-channel key groups), static
  key centering (`kv_center = auto`), and the baselines of the paper (RTN, KIVI, a RotateKV simulation, ...).
* **`integrations/vllm_flagrot/`** — an out-of-tree plugin for **vLLM 0.30** (attention backend `CUSTOM`): paged KV-cache
  writers and decode-attention CUDA kernels for the INT4, INT2 and KIVI-style FlagRot caches, CUDA-graph capture,
  prefix caching with BF16 sink / window tokens, the calibration export, a serving script, the task-evaluation
  harness and benchmarks.
* **`kernels/`** — CUDA sources of the decode-attention kernels and cache writers shared with the plugin, their PyTorch
  references and kernel unit tests.
* **`scripts/`** — data preparation and the reproduction scripts (simulated sweeps, vLLM task runs, decode benchmark).
* **`tests/`** — unit tests of the simulator.

## Installation

Hardware: the CUDA kernels target Hopper (`sm_90a`); experiments were run on NVIDIA H20 (96 GB) and H200 GPUs.
The simulator runs any model of the paper on a single 96 GB GPU (the ~27B models need most of it).

We used three Python environments (they can be merged if the versions are compatible):

| environment | used for | tested versions |
|---|---|---|
| simulator | `blockflag`, KL sweeps, calibration export | Python 3.12, torch 2.6.0+cu124, transformers 5.15 |
| vLLM | plugin, serving, decode benchmark | vLLM 0.30.0 (torch 2.13.0+cu130) |
| eval | task clients (`eval/launch.py`) | lm-eval 0.4.13, openai, datasets, math-verify |

```bash
# simulator environment
pip install -r requirements-sim.txt
pip install -e .

# vLLM environment
pip install vllm==0.30.0
pip install -e integrations/vllm_flagrot          # registers the plugin (entry point vllm.general_plugins)

# eval environment (client side only)
pip install -r requirements-eval.txt
```

**Kernels.** The plugin loads two shared libraries through `ctypes`, `libflagrot_kernels.so` (INT4 / INT2 pages:
`kernels/csrc` + `integrations/vllm_flagrot/csrc/flagrot_capi.cu`) and `libflagrot_kivi.so` (KIVI-style pages:
`integrations/vllm_flagrot/csrc/kivi.cu`), plus `libflagrot_mq.so` for prefix-cached prefill. They need a **CUDA 12.9
toolkit** (the `ptxas` of older toolkits serializes the `wgmma` pipelines); the CUDA runtime is linked statically, so
only the driver is shared with the vLLM environment. The libraries are built automatically on first use (into
`integrations/vllm_flagrot/build/`), or explicitly:

```bash
export FLAGROT_CUDA_HOME=/usr/local/cuda-12.9      # default
bash integrations/vllm_flagrot/csrc/build.sh        # libflagrot_kernels.so
bash integrations/vllm_flagrot/csrc/build_kivi.sh   # libflagrot_kivi.so
bash integrations/vllm_flagrot/csrc/build_mq.sh     # libflagrot_mq.so
```

**Models and data.** Models are referred to by short keys (`llama8b`, `qwen8b`, `llama8b_it`, `qwen32b`,
`qwen38_27b`, ...; the full list and the Hugging Face ids are in `blockflag/paths.py`). A key resolves to
`$BLOCKFLAG_MODEL_<KEY>` if set, else to `$BLOCKFLAG_MODEL_DIR/<repo name>` (default `./models/<repo name>`, e.g.
`./models/Llama-3.1-8B`) if that directory exists, else to the Hugging Face id. `python -m blockflag.paths` prints the
resolved locations.

```bash
python scripts/prepare_data.py wikitext      # data/wikitext-2-raw-v1-{test,train,validation}.parquet
python scripts/prepare_data.py longbench1    # data/longbench1/<task>.jsonl (only for the LongBench v1 runs)
```

Other locations: `BLOCKFLAG_DATA_DIR` (default `./data`), `BLOCKFLAG_CACHE` (token ids, calibration statistics;
default `./cache`), `BLOCKFLAG_RESULTS` (simulator run files; default `./results`) and `FLAGROT_CALIB_DIR` /
`BLOCKFLAG_CALIB_DIR` (plugin calibration files; default `./calib`).

## Quick start

### Simulate KL on one model

One configuration = one run of `blockflag.run`; the result (mean KL, per-window KL / NLL / top-1 agreement) is written to
`results/runs/<config id>.json`. FlagRot at INT2 (groups of 32) with 4 sink and 128 recent tokens kept in BF16, on the
first 16 test windows:

```bash
python -m blockflag.run --config '{"model": "llama3b", "kv_k": "int2a:32", "kv_v": "int2a:32",
  "kv_rb": 32, "kv_flags": "dyn", "kv_dyn": "cq+f+konly", "kv_center": "auto",
  "kv_sink": 4, "kv_resid": 128, "calib_windows": 4, "windows": 16}'
```

The same through the sweep driver (one job per GPU, finished jobs are skipped):

```bash
python scripts/sim/sweep.py run    --table unified --models llama3b --bits 2 --windows 16 --gpus 0
python scripts/sim/sweep.py report --table unified --models llama3b --bits 2 --windows 16
```

All configuration keys are documented at the top of `blockflag/run.py`; the cache formats (`int<b>a:<g>`,
`kivi<b>:<g>`, `kivi<b>f8:<g>`, ...) at the top of `blockflag/formats.py`.

### Serve a model with a FlagRot KV cache in vLLM

1. Export the calibration file(s) of the mode (simulator environment, one GPU, a few minutes; see below):

   ```bash
   python integrations/vllm_flagrot/scripts/export_calib.py --model qwen8b --kivi kivi   # -> calib/flagrot_kivi_qwen8b.pt
   ```

2. Serve (vLLM environment). `serve_mode.sh <model> <mode> <gpu> <port>` starts an OpenAI-compatible server in the
   background and waits until it is healthy; `serve_mode.sh stop <port>` stops it:

   ```bash
   VLLM=$(which vllm) bash integrations/vllm_flagrot/scripts/serve_mode.sh qwen8b kivi2p8ff:s4w128 0 8000
   curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model": "qwen8b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
   bash integrations/vllm_flagrot/scripts/serve_mode.sh stop 8000
   ```

   Equivalently, with plain `vllm serve`:

   ```bash
   vllm serve Qwen/Qwen3-8B --attention-backend CUSTOM --kv-cache-dtype auto --block-size 32 \
     --additional-config '{"flagrot": {"kv_cache_dtype": "flagrot_kivi", "calib": "calib/flagrot_kivi_qwen8b.pt",
                                       "window": 128, "sink": 4, "kivi_f8": 1, "kivi_ff": 1}}'
   ```

   and from Python, `make_llm(model, kv, window=128, sink=4, ...)` in `integrations/vllm_flagrot/scripts/common.py`
   builds the same `vllm.LLM`.

## Calibration files

All calibrations use 4 WikiText-2 *train* windows of 2048 tokens and exactly the simulator's code path
(`blockflag.run.build`); they are written to `$FLAGROT_CALIB_DIR` (default `./calib`) under the names the plugin
looks up. Run `export_calib.py` with the model key that you serve (e.g. `llama8b_it` for the instruct model).

| modes | file(s) | command (`integrations/vllm_flagrot/scripts/export_calib.py ...`) |
|---|---|---|
| `konlya`, `konlya2` | `flagrot_calib_<model>.pt` (query covariances C_q, output-projection energies) | `--model <model>` |
| `...:ca` (with `konlya2`, S4W128) | `flagrot_center_auto_flagrot_s4w128_<model>.pt` (centering vector; zero = off) | `--model <model> --center auto --method flagrot --prot s4w128` |
| `kivi2`, `kivi2id`, `kivi2p8ff`, `kivi2p8ffid` | `flagrot_kivi_<model>.pt` | `--model <model> --kivi kivi` |
| `kivi2p8ffs` | `flagrot_kivi_sp8_<model>.pt` (fixed per-layer / head / slot choices) | `--model <model> --kivi kivi_sp8` |
| `rtn_a` (plain INT4 RTN in the static kernels) | `flagrot_static_rtn_a_<model>.pt` | `make_rtn_calib.py --model <model> --from_config` |

## KV-cache modes

A mode is `<kv>[:s<S>w<W>][:ca]`: `:s4w128` keeps the first 4 tokens (sinks) and the 128 most recent tokens of every
sequence in BF16 (all quantized rows of the paper use S4W128; the KIVI modes need a window of at least one 32-token key
group), `:ca` adds static key centering. Bits per value count codes and per-group metadata (groups of 32, head dim 128).

| mode | alias | method in the paper | bits / value |
|---|---|---|---|
| `bf16` | | BF16 cache (stock vLLM) | 16 |
| `konlya:s4w128` | `flagrot-int4` | FlagRot, INT4 | 4.5 |
| `konlya2:s4w128:ca` | `flagrot-int2-auto` | FlagRot + auto centering, INT2 | 2.5 |
| `kivi2:s4w128` | `flagrot-kivi-3.0` | FlagRot-KIVI, INT2, FP16 scale + minimum | 3.0 |
| `kivi2id:s4w128` | `kivi-3.0` | KIVI only (same pages and kernels, no selection) | 3.0 |
| `kivi2p8ff:s4w128` | `flagrot-kivi-2.5` | FlagRot-KIVI, INT2, FP8 scale + minimum, per-key selection | 2.5 |
| `kivi2p8ffid:s4w128` | `kivi-2.5` | KIVI only, 2.5-bit packing | 2.5 |
| `kivi2p8ffs:s4w128` | `flagrot-kivi-2.5-fixed` | FlagRot-KIVI with fixed (calibrated, static) selection | 2.5 |

The aliases expand to the full mode strings above and are accepted by `serve_mode.sh`, `eval/launch.py` and
`bench_e2e.py --mode`. `fp8` and `int4pth` select vLLM's own FP8 and `int4_per_token_head` caches. Further modes in
`serve_mode.sh` (`rtn_a`, `static_a`, `konlya_c`, `rtn_a2`, `static_a2`, `konlya_c2`, `kivi2c`, `kivi2s`, `kivi2p8`,
`kivi2p8id`, `kivi2p8f*`, `kivi2ff*`, `kivi2h*`) are ablations and earlier kernel versions; `kivi2p8ff*` write the same
caches as `kivi2p8*` with faster kernels.

Kernel coverage: head dim 128 or 256. The decode kernels of the 2.5-bit KIVI pages (`kivi2p8ff*`) are instantiated for
GQA groups 1, 2, 4, 5, 6, 8 at head dim 128 and 1, 2, 4, 6 at head dim 256 (so, e.g., not for Llama-3.2-3B, group 3);
the task models of the paper have groups 4 (Llama-3.1-8B-Instruct, Qwen3-8B), 8 (Qwen3-32B) and 6 at head dim 256
(Qwen3.8-27B).

## Reproducing the paper

| paper item | how to reproduce |
|---|---|
| Simulated KL across bit widths (INT2 / INT3 / INT4; all encoders with and without selection; equal-size comparison with KIVI and RotateKV) | `bash scripts/run_sim_tables.sh` (= `scripts/sim/sweep.py run/report --table unified`) |
| Per-model KL of INT2 caches on the 12 models (appendix) | `scripts/sim/sweep.py run/report --table int2` (part of `run_sim_tables.sh`) |
| Task accuracy in vLLM: LongBench v1 (Qwen3-8B, Llama-3.1-8B-Instruct, Qwen3-32B, Qwen3.8-27B), RULER-128k (Llama-3.1-8B-Instruct), AIME24+25 (Qwen3-8B) | `bash scripts/run_vllm_tasks.sh` (calibration export, `eval/launch.py` with `eval/configs/{longbench1,longbench1_27b,ruler128k,aime}.yaml`, paired comparisons with `eval/compare.py`) |
| Decode throughput / capacity under latency budgets at 32k (Qwen3-8B, Qwen3-32B) and the Qwen3.8-27B pure-decode cost | `bash scripts/run_decode_bench.sh` (`integrations/vllm_flagrot/scripts/bench_e2e.py`) |
| Kernel correctness and simulator parity | `kernels/test_*.py`, `integrations/vllm_flagrot/tests/`, `integrations/vllm_flagrot/scripts/parity.py` |

Notes:

* The KL sweeps use all 2048-token WikiText-2 test windows (`windows = 1000` is clipped to the available windows) and
  4 calibration windows; `sweep.py report` prints geometric means over the 12-model suite and, for the selection rows,
  the number of models on which selection is significantly lower / higher than the same encoder without selection
  (paired 95 % bootstrap over windows). `--windows N` runs a cheaper subset (different configuration ids).
* Task runs use one vLLM server per GPU (`serve_mode.sh`, prefix caching on, CUDA graphs on) and lm-eval or
  `eval/reasoning.py` as clients; settings (context length, prompt truncation, chat template, sampling, seeds) are
  documented in the config files. Set `VLLM` to the vLLM binary of the vLLM environment and `EVALTOOLS` to the prefix
  of the eval environment if `eval/launch.py` is run from another environment. `eval/compare.py` prints paired
  differences with 95 % bootstrap intervals (LongBench / RULER: macro averages, items resampled within each task).
* The decode benchmark runs an offline engine with random prompts of exactly the context length, prefix caching off,
  and reports the mean time per output token over 48 decode steps per batch size.
* Unit tests: `pytest -q tests` (simulator), `python kernels/test_attn_v2.py --quick`, `python kernels/test_codebook.py
  --quick` (simulator environment + CUDA 12.9), and `python integrations/vllm_flagrot/tests/test_*.py` (vLLM
  environment; `test_kivi.py` takes a KIVI calibration file via `--calib`).

## Third-party material

* `integrations/vllm_flagrot/eval/lb1_tasks/official/*.json` are the prompt templates and length settings of the
  official LongBench code (THUDM/LongBench, MIT License; see `official/LICENSE`). The `lb1_*.yaml` and `ruler_fit`
  task files are derived from the LongBench and RULER tasks of lm-evaluation-harness (EleutherAI, MIT License) and
  import their metric code from the installed `lm_eval` package.
* The baselines are re-implemented inside the simulator (`blockflag/formats.py`, `blockflag/qattention.py`); the
  official code of the compared methods is not included: KIVI (https://github.com/jy-yuan/KIVI), RotateKV
  (https://github.com/ZunhaiSu/RotateKV), QuaRot (https://github.com/spcl/QuaRot), SpinQuant
  (https://github.com/facebookresearch/SpinQuant).

## License

Apache License 2.0 (see `LICENSE`), except for the third-party files listed above, which keep their own licenses.
