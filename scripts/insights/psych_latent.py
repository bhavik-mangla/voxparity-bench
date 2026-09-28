"""Q2 - latent structure: one "acts-on-audio" ability or several?

Four views, all on the 28 contestants (plus the null/ladders/instrument where a
view needs every AI respondent):

1. Dimensionality by held-out prediction: regularised logistic matrix
   factorisation (item intercept + rank-k respondent x item term, L2), cell-masked
   5-fold CV, k = 0..4. If rank 2+ does not beat rank 1 out of sample, a single
   latent ability explains the response matrix.
2. PCA of the centred contestants x cells matrix with Horn's parallel analysis
   (permute each cell's responses across systems, 500 draws).
3. Per-axis abilities: each contestant's mean credit on an axis's cue-bearing
   cells, and its AUDIO LIFT on the axis (audio credit minus its own text-twin
   credit on the same cells; twin-capable arms only). Pairwise Pearson across
   systems, item-clustered bootstrap CIs, and disattenuated by split-half
   reliability (items split at random, Spearman-Brown, 200 splits).
4. NMF (k = 2, 3) of the contestants x cells pass matrix; component weight by axis.
"""

from __future__ import annotations

import numpy as np
from psych_common import SEED, Data, contestants

AXES = [
    "delivery emotion",
    "sarcasm",
    "scene (environmental)",
    "second-speaker",
    "slot-noise",
    "speaker attribute",
    "disfluency",
    "neutral",
]
# Grouped abilities (enough cells for a stable per-system score)
GROUPS = {
    "emotion": ["delivery emotion", "sarcasm"],
    "scene": ["scene (environmental)"],
    "second-speaker": ["second-speaker"],
    "acoustic-degradation": ["slot-noise", "disfluency"],
    "speaker attribute": ["speaker attribute"],
    "neutral (no cue)": ["neutral"],
}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def lmf(
    Y: np.ndarray, k: int, lam: float = 1.0, iters: int = 400, lr: float = 0.05, seed: int = SEED
):
    """Logistic MF: logit P = c_i + u_j . v_i ; Adam on the masked log-lik + L2."""
    rng = np.random.default_rng(seed)
    n_resp, n_items = Y.shape
    m = ~np.isnan(Y)
    Yz = np.nan_to_num(Y)
    p = np.clip(np.nanmean(Y, 0), 0.02, 0.98)
    c = np.log(p / (1 - p))
    U = 0.1 * rng.standard_normal((n_resp, k))
    V = 0.1 * rng.standard_normal((n_items, k))
    params = [c, U, V]
    mom = [np.zeros_like(x) for x in params]
    vel = [np.zeros_like(x) for x in params]
    for t in range(1, iters + 1):
        Z = c[None, :] + U @ V.T
        G = (Yz - _sig(Z)) * m  # d ll / dZ
        gc = G.sum(0) - 0.01 * c
        gU = G @ V - lam * U
        gV = G.T @ U - lam * V
        for n_, (x, g) in enumerate(zip(params, (gc, gU, gV), strict=True)):
            mom[n_] = 0.9 * mom[n_] + 0.1 * g
            vel[n_] = 0.999 * vel[n_] + 0.001 * g * g
            mh = mom[n_] / (1 - 0.9**t)
            vh = vel[n_] / (1 - 0.999**t)
            x += lr * mh / (np.sqrt(vh) + 1e-8)
    return c, U, V


def cv_rank(Y: np.ndarray, ks=(0, 1, 2, 3, 4), lams=(0.3, 1.0, 3.0), folds: int = 5) -> dict:
    rng = np.random.default_rng(SEED)
    obs = np.argwhere(~np.isnan(Y))
    fold = rng.integers(0, folds, size=len(obs))
    out = {}
    for k in ks:
        best = None
        for lam in lams if k else (1.0,):
            lls, accs = [], []
            for f in range(folds):
                te = obs[fold == f]
                Ytr = Y.copy()
                Ytr[te[:, 0], te[:, 1]] = np.nan
                if k == 0:
                    pr = np.clip(np.nanmean(Ytr, 0), 1e-3, 1 - 1e-3)[te[:, 1]]
                else:
                    c, U, V = lmf(Ytr, k, lam)
                    pr = np.clip(
                        _sig(c[te[:, 1]] + (U[te[:, 0]] * V[te[:, 1]]).sum(1)), 1e-6, 1 - 1e-6
                    )
                y = Y[te[:, 0], te[:, 1]]
                lls.append(np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
                accs.append(np.mean((pr > 0.5) == (y > 0.5)))
            r = {
                "lambda": lam,
                "heldout_loglik": round(float(np.mean(lls)), 4),
                "heldout_acc": round(float(np.mean(accs)), 4),
            }
            if best is None or r["heldout_loglik"] > best["heldout_loglik"]:
                best = r
        out[f"rank{k}"] = best
    return out


def parallel_analysis(X: np.ndarray, n: int = 500) -> dict:
    Xc = X - np.nanmean(X, 0)
    Xc = np.nan_to_num(Xc)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(n):
        P = np.column_stack([rng.permutation(X[:, i]) for i in range(X.shape[1])])
        Pc = np.nan_to_num(P - np.nanmean(P, 0))
        e = np.linalg.svd(Pc, compute_uv=False) ** 2
        null.append(e / e.sum())
    null = np.array(null)
    q95 = np.quantile(null, 0.95, axis=0)
    n_retain = (
        int(np.argmax(ev[: len(q95)] <= q95[: len(ev)]))
        if np.any(ev[: len(q95)] <= q95)
        else len(ev)
    )
    return {
        "explained": [round(float(x), 4) for x in ev[:6]],
        "null95": [round(float(x), 4) for x in q95[:6]],
        "components_retained": n_retain,
    }


def pca_loadings(d: Data, X: np.ndarray) -> dict:
    Xc = np.nan_to_num(X - np.nanmean(X, 0))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    out = {}
    ax = np.array(d.axis)
    scores = U[:, :2] * S[:2]
    for c in range(2):
        v = Vt[c]
        if c == 0 and v.mean() < 0:  # PC1 signed so a higher score = more credit
            v = -v
            scores[:, 0] = -scores[:, 0]
        out[f"PC{c + 1}"] = {a: round(float(v[ax == a].mean() * np.sqrt(len(v))), 3) for a in AXES}
    return {"loadings_by_axis": out, "scores": scores}


def per_axis_scores(d: Data, rows: list[int], cells_mask: np.ndarray, lift: bool) -> np.ndarray:
    X = d.credit[np.ix_(rows, np.where(cells_mask)[0])]
    if lift:
        T = d.twin_passed[np.ix_(rows, np.where(cells_mask)[0])]
        # twin credit is not stored per cell; use strict twin pass vs strict audio pass
        A = d.passed[np.ix_(rows, np.where(cells_mask)[0])]
        return np.nanmean(A - T, 1)
    return np.nanmean(X, 1)


def group_mask(d: Data, g: str) -> np.ndarray:
    return np.isin(np.array(d.axis), GROUPS[g])


def split_half_rel(d: Data, rows: list[int], mask: np.ndarray, lift: bool, n: int = 200) -> float:
    rng = np.random.default_rng(SEED)
    cells = np.where(mask)[0]
    items = np.unique(d.item_of[cells])
    rs = []
    for _ in range(n):
        perm = rng.permutation(items)
        h = set(perm[: len(items) // 2])
        m1 = np.zeros(len(d.cells), bool)
        m2 = np.zeros(len(d.cells), bool)
        for c in cells:
            (m1 if d.item_of[c] in h else m2)[c] = True
        a = per_axis_scores(d, rows, m1, lift)
        b = per_axis_scores(d, rows, m2, lift)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            r = np.corrcoef(a[ok], b[ok])[0, 1]
            rs.append(2 * r / (1 + r))
    return float(np.median(rs)) if rs else float("nan")


def axis_correlations(d: Data, rows: list[int], lift: bool, n_boot: int = 1000) -> dict:
    names = list(GROUPS)
    masks = {g: group_mask(d, g) for g in names}
    S = np.column_stack([per_axis_scores(d, rows, masks[g], lift) for g in names])
    R = np.corrcoef(S.T)
    rel = {g: split_half_rel(d, rows, masks[g], lift) for g in names}
    rng = np.random.default_rng(SEED)
    items = np.unique(d.item_of)
    by_item = {i: np.where(d.item_of == i)[0] for i in items}
    B = []
    for _ in range(n_boot):
        pick = rng.choice(items, size=len(items), replace=True)
        idx = np.concatenate([by_item[i] for i in pick])
        cols = []
        for g in names:
            sel = idx[masks[g][idx]]
            X = d.passed[np.ix_(rows, sel)] - (d.twin_passed[np.ix_(rows, sel)] if lift else 0)
            if not lift:
                X = d.credit[np.ix_(rows, sel)]
            cols.append(np.nanmean(X, 1))
        B.append(np.corrcoef(np.column_stack(cols).T))
    B = np.array(B)
    lo, hi = np.nanquantile(B, [0.025, 0.975], axis=0)
    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            r = R[a, b]
            den = np.sqrt(max(rel[names[a]], 1e-3) * max(rel[names[b]], 1e-3))
            pairs.append(
                {
                    "a": names[a],
                    "b": names[b],
                    "r": round(float(r), 3),
                    "lo": round(float(lo[a, b]), 3),
                    "hi": round(float(hi[a, b]), 3),
                    "r_disattenuated": round(float(min(r / den, 1.5)), 3),
                }
            )
    return {
        "groups": names,
        "n_systems": len(rows),
        "reliability_split_half": {g: round(v, 3) for g, v in rel.items()},
        "cells_per_group": {g: int(masks[g].sum()) for g in names},
        "R": [[round(float(x), 3) for x in row] for row in R],
        "pairs": pairs,
        "scores": {d.labels[r]: [round(float(x), 4) for x in S[i]] for i, r in enumerate(rows)},
    }


def nmf(X: np.ndarray, k: int, iters: int = 2000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    M = ~np.isnan(X)
    Xz = np.nan_to_num(X)
    W = rng.random((X.shape[0], k)) + 0.1
    H = rng.random((k, X.shape[1])) + 0.1
    for _ in range(iters):
        WH = W @ H
        H *= (W.T @ (M * Xz)) / (W.T @ (M * WH) + 1e-9)
        WH = W @ H
        W *= ((M * Xz) @ H.T) / ((M * WH) @ H.T + 1e-9)
    err = np.sqrt(np.sum((M * (Xz - W @ H)) ** 2) / M.sum())
    return W, H, float(err)


def run(d: Data) -> dict:
    con = contestants(d)
    allai = list(range(len(d.labels)))
    Y = d.passed
    cv = cv_rank(Y[allai])
    cv_cue = cv_rank(Y[np.ix_(allai, np.where(d.cue)[0])], ks=(0, 1, 2, 3))
    pa = parallel_analysis(d.credit[con])
    pa_cue = parallel_analysis(d.credit[np.ix_(con, np.where(d.cue)[0])])
    pca = pca_loadings(d, d.credit[con])
    twin_rows = [j for j in con if np.isfinite(d.twin_passed[j]).sum() > 100]
    corr_raw = axis_correlations(d, con, lift=False)
    corr_lift = axis_correlations(d, twin_rows, lift=True)
    # general competence vs audio use: neutral-cell credit vs cue-bearing lift
    neu = np.nanmean(d.credit[np.ix_(twin_rows, np.where(~d.cue)[0])], 1)
    lift = np.nanmean(
        d.passed[np.ix_(twin_rows, np.where(d.cue)[0])]
        - d.twin_passed[np.ix_(twin_rows, np.where(d.cue)[0])],
        1,
    )
    cue_cr = np.nanmean(d.credit[np.ix_(con, np.where(d.cue)[0])], 1)
    neu_all = np.nanmean(d.credit[np.ix_(con, np.where(~d.cue)[0])], 1)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(2000):
        s = rng.integers(0, len(twin_rows), len(twin_rows))
        if np.std(neu[s]) > 0 and np.std(lift[s]) > 0:
            bs.append(np.corrcoef(neu[s], lift[s])[0, 1])
    # NMF
    nm = {}
    ax = np.array(d.axis)
    for k in (2, 3):
        W, H, err = nmf(d.passed[con], k)
        comp = {}
        for c in range(k):
            h = H[c] / H[c].mean()
            comp[f"C{c + 1}"] = {a: round(float(h[ax == a].mean()), 3) for a in AXES}
        nm[f"k{k}"] = {
            "rmse": round(err, 4),
            "component_weight_by_axis": comp,
            "system_weights": {
                d.labels[j]: [round(float(x), 3) for x in W[i]] for i, j in enumerate(con)
            },
        }
    return {
        "cv_rank_all_cells": cv,
        "cv_rank_cue_cells": cv_cue,
        "parallel_analysis_all": pa,
        "parallel_analysis_cue": pa_cue,
        "pca_loadings_by_axis": pca["loadings_by_axis"],
        "pca_scores": {
            d.labels[j]: [round(float(x), 3) for x in pca["scores"][i]] for i, j in enumerate(con)
        },
        "axis_corr_credit": corr_raw,
        "axis_corr_audio_lift": corr_lift,
        "twin_capable_systems": [d.labels[j] for j in twin_rows],
        "neutral_vs_cue_lift": {
            "r": round(float(np.corrcoef(neu, lift)[0, 1]), 3),
            "lo": round(float(np.quantile(bs, 0.025)), 3),
            "hi": round(float(np.quantile(bs, 0.975)), 3),
            "n_systems": len(twin_rows),
            "note": "system-level Pearson of neutral-cell credit (general tool competence) "
            "with cue-bearing audio-minus-own-twin strict pass (audio use); CI = bootstrap "
            "over systems",
        },
        "neutral_vs_cue_credit_r": round(float(np.corrcoef(neu_all, cue_cr)[0, 1]), 3),
        "nmf": nm,
    }
