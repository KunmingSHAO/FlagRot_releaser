#!/usr/bin/env python3
"""Download the evaluation / calibration data into ./data (or $BLOCKFLAG_DATA_DIR).

  python scripts/prepare_data.py wikitext      # WikiText-2 raw v1 parquet files (simulator: calibration + KL evaluation)
  python scripts/prepare_data.py longbench1    # official LongBench v1 data (THUDM/LongBench data.zip) -> data/longbench1

WikiText-2: hf datasets Salesforce/wikitext, wikitext-2-raw-v1/{test,train,validation}-00000-of-00001.parquet; the test
and train files used in the paper have the sha256 checked below (validation is only used for hyper-parameter checks).
LongBench v1: hf datasets THUDM/LongBench (now zai-org/LongBench), data.zip (sha256 cb45b11a...); the <task>.jsonl files
are what eval/lb1_tasks/lb1_utils.py reads (env LB1_DIR, default data/longbench1).
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockflag.paths import DATA_DIR  # noqa: E402

WIKI = {"test": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
        "train": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
        "validation": None}
LB1_SHA = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wikitext():
    from huggingface_hub import hf_hub_download
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, want in WIKI.items():
        dst = DATA_DIR / f"wikitext-2-raw-v1-{split}.parquet"
        if not dst.exists():
            src = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                                  repo_type="dataset")
            shutil.copy(src, dst)
        got = sha256(dst)
        status = "ok" if want is None or got == want else f"MISMATCH (expected {want})"
        print(f"{dst}: sha256 {got} {status}")


def longbench1():
    from huggingface_hub import hf_hub_download
    out = DATA_DIR / "longbench1"
    out.mkdir(parents=True, exist_ok=True)
    z = Path(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset"))
    got = sha256(z)
    print(f"data.zip sha256 {got} {'ok' if got == LB1_SHA else 'MISMATCH (expected ' + LB1_SHA + ')'}")
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():
            if m.endswith(".jsonl"):
                (out / Path(m).name).write_bytes(zf.read(m))
    print(f"{out}: {len(list(out.glob('*.jsonl')))} task files")


if __name__ == "__main__":
    what = sys.argv[1:] or ["wikitext"]
    for w in what:
        {"wikitext": wikitext, "longbench1": longbench1}[w]()
