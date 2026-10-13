"""Filesystem locations.  Every entry can be overridden with an environment variable BLOCKFLAG_<NAME>.

Models are looked up as follows (first match wins):
  1. BLOCKFLAG_MODEL_<KEY>   (e.g. BLOCKFLAG_MODEL_LLAMA8B=/path/to/Llama-3.1-8B), a local directory or a HF id;
  2. <MODEL_DIR>/<repo name> (MODEL_DIR = $BLOCKFLAG_MODEL_DIR, default ./models; e.g. ./models/Llama-3.1-8B);
  3. the Hugging Face hub id (downloaded by transformers / vLLM on first use).

  python -m blockflag.paths <key>      prints the resolved location of one model key
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _p(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(f"BLOCKFLAG_{name}", str(default)))


MODEL_DIR = _p("MODEL_DIR", ROOT / "models")
# model key -> Hugging Face hub id
HF_IDS = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B",
    # task-level evaluation: the instruct model, calibrated like the base model
    "llama8b_it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen14b": "Qwen/Qwen3-14B",
    "mistral7b": "mistralai/Mistral-7B-v0.3",
    "qwen32b": "Qwen/Qwen3-32B",
    "qwen30b_moe": "Qwen/Qwen3-30B-A3B",
    # held-out models of the KV-cache study (r_b is selected on the four development models only)
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
    "llama3b": "meta-llama/Llama-3.2-3B",
    "qwen25_7b": "Qwen/Qwen2.5-7B",
    "qwen2_7b": "Qwen/Qwen2-7B",
    "gemma2_9b": "google/gemma-2-9b",
    "phi4": "microsoft/phi-4",
    "nemo12b": "mistralai/Mistral-Nemo-Base-2407",
    "glm4_9b": "THUDM/glm-4-9b-hf",
    # ~27B models: hybrid Gated-DeltaNet / full attention (KV cache in every 4th layer) and local / global attention
    "qwen38_27b": "Qwen/Qwen3.8-27B",
    "gemma3_27b": "google/gemma-3-27b-pt",
}


def model_path(key: str) -> str:
    """local directory of a model key if present, else its Hugging Face hub id (see the module docstring)."""
    env = os.environ.get(f"BLOCKFLAG_MODEL_{key.upper()}")
    if env:
        return env
    hf = HF_IDS[key]
    local = MODEL_DIR / hf.split("/")[-1]
    return str(local) if local.exists() else hf


class _Models(dict):
    """dict-like view: MODELS[key] resolves the location at access time."""

    def __getitem__(self, key):
        return model_path(key)

    def __contains__(self, key):
        return key in HF_IDS

    def keys(self):
        return HF_IDS.keys()

    def __iter__(self):
        return iter(HF_IDS)

    def __len__(self):
        return len(HF_IDS)

    def items(self):
        return [(k, model_path(k)) for k in HF_IDS]

    def get(self, key, default=None):
        return model_path(key) if key in HF_IDS else default


MODELS = _Models()
DATA_DIR = _p("DATA_DIR", ROOT / "data")
DATA = {
    # WikiText-2 raw v1 parquet files (scripts/prepare_data.py)
    "test": _p("TEST_PARQUET", DATA_DIR / "wikitext-2-raw-v1-test.parquet"),
    "train": _p("TRAIN_PARQUET", DATA_DIR / "wikitext-2-raw-v1-train.parquet"),
    "valid": _p("VALID_PARQUET", DATA_DIR / "wikitext-2-raw-v1-validation.parquet"),
    # calibration only (calib_data = fineweb, as WUSH-KV): first shard of FineWeb-Edu sample-10BT
    "fineweb": _p("FINEWEB_PARQUET", DATA_DIR / "fineweb_edu_sample10BT_000_00000.parquet"),
    # evaluation only (eval_data): PG-19 test books (hf emozilla/pg19-test @c5e39bf3) and Python files
    # of codeparrot/codeparrot-clean-valid (@4db92d2e, file-000000000054.json.gz)
    "pg19": _p("PG19_PARQUET", DATA_DIR / "pg19_test.parquet"),
    "code": _p("CODE_JSONGZ", DATA_DIR / "codeparrot_clean_valid_054.json.gz"),
    # evaluation only (eval_data): math web text (hf open-web-math/open-web-math @fde8ef8d,
    # data/train-00000-of-00114) and Wikipedia 20231101 zh / de (hf wikimedia/wikipedia @b04c8d1c, first shard each)
    "owm": _p("OWM_PARQUET", DATA_DIR / "owm_train-00000-of-00114.parquet"),
    "wiki_zh": _p("WIKI_ZH_PARQUET", DATA_DIR / "wiki_zh_train-00000-of-00006.parquet"),
    "wiki_de": _p("WIKI_DE_PARQUET", DATA_DIR / "wiki_de_train-00000-of-00020.parquet"),
}
CACHE = _p("CACHE", ROOT / "cache")                 # token ids and calibration statistics (re-creatable)
CALIB = _p("CALIB_DIR", ROOT / "calib")             # calibration files of the vLLM plugin (export_calib.py)
RESULTS = _p("RESULTS", ROOT / "results")           # simulator run files (results/runs/<config id>.json)


if __name__ == "__main__":
    for k in sys.argv[1:] or list(HF_IDS):
        print(model_path(k) if sys.argv[1:] else f"{k}\t{model_path(k)}")
