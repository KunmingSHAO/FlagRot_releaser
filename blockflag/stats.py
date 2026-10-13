"""Paired statistics over evaluation windows.

All configs of a model are evaluated on the same first-N TEST windows, so two configs are compared through the
per-window differences d_i = KL_a,i - KL_b,i (token-weighted means).  The CI is a percentile bootstrap over windows
(windows are the independent units: 2048 contiguous tokens each)."""
from __future__ import annotations

import numpy as np


def window_kl(res: dict) -> tuple[np.ndarray, np.ndarray]:
    w = res["per_window"]
    return np.array([x["kl"] for x in w]), np.array([x["tokens"] for x in w], dtype=float)


def paired_delta(a: dict, b: dict, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """KL(a) - KL(b): mean, relative change, bootstrap CI.  a and b must share the evaluation windows."""
    ka, ta = window_kl(a)
    kb, tb = window_kl(b)
    n = min(len(ka), len(kb))
    assert n > 1 and np.array_equal(ta[:n], tb[:n]), "configs were not evaluated on the same windows"
    d, t = ka[:n] - kb[:n], ta[:n]
    if not (np.isfinite(ka[:n]).all() and np.isfinite(kb[:n]).all()):
        raise FloatingPointError("non-finite per-window KL: refusing to compare")
    base = float((kb[:n] * t).sum() / t.sum())
    mean = float((d * t).sum() / t.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = (d[idx] * t[idx]).sum(1) / t[idx].sum(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return dict(delta=mean, rel=mean / base if base else float("nan"), lo=float(lo), hi=float(hi),
                rel_lo=float(lo) / base if base else float("nan"), rel_hi=float(hi) / base if base else float("nan"),
                n_windows=n, p_better=float((boots < 0).mean()))


def verdict(cmp: dict, min_effect: float) -> str:
    """'better' / 'worse' when the CI excludes 0 and the relative effect is at least min_effect, else 'tie'."""
    if cmp["hi"] < 0 and -cmp["rel"] >= min_effect:
        return "better"
    if cmp["lo"] > 0 and cmp["rel"] >= min_effect:
        return "worse"
    return "tie"


def paired_acc(a: dict, b: dict, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """mean-over-tasks accuracy of a minus b (lm-eval results with per_doc correctness), documents resampled
    independently per task (paired: same documents for a and b)."""
    tasks = [t for t in a["lm"] if t != "mean" and t in b["lm"]]
    rng = np.random.default_rng(seed)
    diffs, boots = [], np.zeros(n_boot)
    for t in tasks:
        x, y = np.array(a["lm"][t]["per_doc"]), np.array(b["lm"][t]["per_doc"])
        n = min(len(x), len(y))
        d = x[:n] - y[:n]
        diffs.append(d.mean())
        boots += d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    boots /= len(tasks)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return dict(delta=float(np.mean(diffs)), lo=float(lo), hi=float(hi), tasks=tasks,
                per_task={t: float(v) for t, v in zip(tasks, diffs)})
