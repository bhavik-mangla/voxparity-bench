"""Shared loading and statistics for the Lens-3 acoustic insight scripts.

Every interval is an ITEM-CLUSTERED bootstrap (the project convention, D010/D105):
items are resampled with replacement and every row of a drawn item comes along.
Implemented with per-item multiplicity weights, so any weighted statistic
(mean, WLS slope, fractional logit) bootstraps with one code path.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "src"))

N_BOOT = 2000
SEED = 20260915

# --------------------------------------------------------------------------- stats


def cluster_weights(items: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, n_rows) multiplicity weights from resampling items with replacement."""
    uniq = sorted(set(items))
    ix = {u: i for i, u in enumerate(uniq)}
    row_item = np.array([ix[i] for i in items])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    counts = np.zeros((n_boot, len(uniq)))
    for b in range(n_boot):
        counts[b] = np.bincount(draws[b], minlength=len(uniq))
    return counts[:, row_item]


def boot_ci(
    stat: Callable[[np.ndarray], float],
    items: list[str],
    base_w: np.ndarray | None = None,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Point estimate + 95% percentile CI of stat(weights) under item resampling."""
    n = len(items)
    if n == 0:
        return {"est": None, "lo": None, "hi": None, "n": 0, "items": 0}
    bw = np.ones(n) if base_w is None else np.asarray(base_w, float)
    est = stat(bw)
    out: dict[str, Any] = {"est": _r(est), "n": n, "items": len(set(items))}
    if len(set(items)) < 3:
        out.update(lo=None, hi=None)
        return out
    W = cluster_weights(items, n_boot)
    vals = np.array([stat(W[b] * bw) for b in range(n_boot)], dtype=float)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.quantile(vals, [0.025, 0.975]) if len(vals) else (np.nan, np.nan)
    out.update(lo=_r(lo), hi=_r(hi))
    # two-sided bootstrap p for "est differs from 0" (share of draws across 0)
    if len(vals):
        p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
        out["p_boot"] = round(float(min(1.0, p)), 4)
    return out


def _r(x: Any, k: int = 4) -> Any:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(xf) else round(xf, k)


def wmean(y: np.ndarray) -> Callable[[np.ndarray], float]:
    def f(w: np.ndarray) -> float:
        s = w.sum()
        return float((w * y).sum() / s) if s > 0 else np.nan

    return f


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares coefficients (X includes the intercept column)."""
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * sw, y * sw[:, 0], rcond=None)
    return beta


def wls_coef(X: np.ndarray, y: np.ndarray, j: int) -> Callable[[np.ndarray], float]:
    def f(w: np.ndarray) -> float:
        if (w > 0).sum() < X.shape[1] + 1:
            return np.nan
        return float(wls(X, y, w)[j])

    return f


def flogit(X: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int = 50) -> np.ndarray:
    """Fractional-response logit (quasi-binomial) by IRLS; y in [0, 1]."""
    beta = np.zeros(X.shape[1])
    ybar = np.clip((w * y).sum() / max(w.sum(), 1e-9), 1e-3, 1 - 1e-3)
    beta[0] = np.log(ybar / (1 - ybar))
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        v = np.clip(mu * (1 - mu), 1e-6, None)
        z = eta + (y - mu) / v
        new = wls(X, z, w * v)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def flogit_coef(X: np.ndarray, y: np.ndarray, j: int) -> Callable[[np.ndarray], float]:
    def f(w: np.ndarray) -> float:
        if (w > 0).sum() < X.shape[1] + 1:
            return np.nan
        return float(flogit(X, y, w)[j])

    return f


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 4:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


# --------------------------------------------------------------------------- data


@dataclass
class Ctx:
    freeze: dict[str, Any]
    items: dict[str, Any]
    manifest: dict[str, dict[str, Any]]
    clips: dict[tuple[str, str, str], dict[str, Any]]
    arms: list[Any]
    human_rows: list[dict[str, Any]]
    crossjudge: dict[str, dict[str, Any]]
    crossjudge2: dict[str, dict[str, Any]]
    ser_tags: dict[str, dict[str, Any]]
    extra: dict[str, Any] = field(default_factory=dict)


def load_ctx(bank: Path, main: Path, features: Path) -> Ctx:
    import yaml

    from voxparity.cli import _iter_item_files, load_item
    from voxparity.harness.crossjudge import load_acoustic_tags
    from voxparity.harness.final_analysis import load_arms
    from voxparity.harness.human_baseline import load_human_rows

    freeze = json.loads((bank / "freeze" / "2026-09-15" / "freeze.json").read_text())
    items: dict[str, Any] = {}
    for d in freeze["item_dirs"]:
        for f in _iter_item_files(bank / d):
            it = load_item(f)
            items[it.id] = it
    manifest = {
        r["sha256"]: r for r in yaml.safe_load((bank / "stimuli" / "manifest.yaml").read_text())
    }
    feats = json.loads(features.read_text())["clips"]
    clips = {(r["item_id"], r["variant_id"], r["engine"]): r for r in feats}
    arms = load_arms(str(bank / "runs" / "20260915-final-*"), items)
    human_rows = load_human_rows(str(main / "runs" / "game-20260925" / "human-*"))
    cj = json.loads((HERE / "docs/results/final/crossjudge.json").read_text())["verdicts"]
    cj2 = json.loads((HERE / "docs/results/final/crossjudge-gptaudio.json").read_text())["verdicts"]
    tags = load_acoustic_tags(
        f"{bank}/runs/20260915-final-cascadeemo-*,{bank}/runs/crossjudge/ser-topup"
    )
    return Ctx(freeze, items, manifest, clips, arms, human_rows, cj, cj2, tags)


def group_by(rows: list[Any], key: Callable[[Any], Any]) -> dict[Any, list[Any]]:
    out: dict[Any, list[Any]] = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out
