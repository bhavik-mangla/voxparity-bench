"""Q1 - item response theory on the systems x cells matrix.

Model: unidimensional 1PL / 2PL fitted by marginal maximum likelihood with EM over
a fixed quadrature (Bock-Aitkin), ability prior N(0, 1), and weakly-informative
item priors (log a ~ N(0, 0.5^2), b ~ N(0, 2^2)) that keep the 2PL identified
with only ~32 respondents (Bayesian-regularised MML; posterior modes for items,
EAP + posterior SD for abilities).

Respondents are AI systems (28 contestants, the words-only null, two ladder rungs,
one instrument). Humans are placed afterwards on the fitted scale by EAP scoring
their majority-vote responses on the cells they answered, on the SELECTION basis
(the only basis humans share with the models, D114).

Also here: test information, item information by axis, held-out predictive
comparison of 1PL vs 2PL (cell-masked 5-fold CV), and item-count sufficiency.
"""

from __future__ import annotations

import numpy as np
from psych_common import SEED, Data, contestants, idx
from scipy.optimize import minimize

NODES = np.linspace(-4, 4, 41)
W = np.exp(-0.5 * NODES**2)
W /= W.sum()
LOGA_SD, B_SD = 0.5, 2.0


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _item_mstep(r: np.ndarray, n: np.ndarray, a0: np.ndarray, b0: np.ndarray, pl: int):
    """Maximise expected complete log-lik + log prior, all items jointly (separable)."""
    n_items = len(a0)

    def f(x: np.ndarray):
        if pl == 2:
            la, b = x[:n_items], x[n_items:]
        else:
            la, b = np.full(n_items, x[0]), x[1:]
        a = np.exp(la)
        z = a[:, None] * (NODES[None, :] - b[:, None])
        p = np.clip(_sig(z), 1e-9, 1 - 1e-9)
        ll = (r * np.log(p) + (n - r) * np.log(1 - p)).sum()
        lp = -0.5 * ((la / LOGA_SD) ** 2).sum() if pl == 2 else -0.5 * (la[0] / LOGA_SD) ** 2
        lp -= 0.5 * ((b / B_SD) ** 2).sum()
        g = r - n * p  # d ll / d z
        dz_da = (NODES[None, :] - b[:, None]) * a[:, None]  # dz/dloga
        dla = (g * dz_da).sum(1) - la / LOGA_SD**2
        db = (g * (-a[:, None])).sum(1) - b / B_SD**2
        if pl == 2:
            grad = np.concatenate([dla, db])
        else:
            grad = np.concatenate([[dla.sum() + (n_items - 1) * la[0] / LOGA_SD**2], db])
        return -(ll + lp), -grad

    x0 = np.concatenate([np.log(a0), b0]) if pl == 2 else np.concatenate([[np.log(a0.mean())], b0])
    res = minimize(f, x0, jac=True, method="L-BFGS-B")
    x = res.x
    if pl == 2:
        return np.exp(x[:n_items]), x[n_items:]
    return np.full(n_items, np.exp(x[0])), x[1:]


def posterior(Y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Respondent x node posterior (normalised), NaN = not administered."""
    P = np.clip(_sig(a[:, None] * (NODES[None, :] - b[:, None])), 1e-9, 1 - 1e-9)  # I x Q
    obs = ~np.isnan(Y)
    Yz = np.nan_to_num(Y)
    ll = Yz @ np.log(P) + (obs & (Yz == 0)).astype(float) @ np.log(1 - P)  # J x Q
    ll += np.log(W)[None, :]
    ll -= ll.max(1, keepdims=True)
    post = np.exp(ll)
    return post / post.sum(1, keepdims=True)


def fit(Y: np.ndarray, pl: int = 2, iters: int = 300, tol: float = 1e-6):
    _, n_items = Y.shape
    obs = ~np.isnan(Y)
    Yz = np.nan_to_num(Y)
    p = np.clip(np.nanmean(Y, 0), 0.02, 0.98)
    a, b = np.ones(n_items), -np.log(p / (1 - p))
    last = None
    for _ in range(iters):
        post = posterior(Y, a, b)
        n = obs.T.astype(float) @ post  # I x Q
        r = (Yz * obs).T @ post
        a, b = _item_mstep(r, n, a, b, pl)
        x = np.concatenate([a, b])
        if last is not None and np.max(np.abs(x - last)) < tol:
            break
        last = x
    post = posterior(Y, a, b)
    theta = post @ NODES
    se = np.sqrt(post @ NODES**2 - theta**2)
    return {"a": a, "b": b, "theta": theta, "se": se, "post": post}


def item_info(a: np.ndarray, b: np.ndarray, grid: np.ndarray) -> np.ndarray:
    P = _sig(a[:, None] * (grid[None, :] - b[:, None]))
    return (a[:, None] ** 2) * P * (1 - P)  # I x G


def pred_ll(
    Y_test: np.ndarray, post: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, float]:
    P = _sig(a[:, None] * (NODES[None, :] - b[:, None]))  # I x Q
    pr = np.clip(post @ P.T, 1e-9, 1 - 1e-9)  # J x I
    m = ~np.isnan(Y_test)
    y = Y_test[m]
    ll = float(np.mean(y * np.log(pr[m]) + (1 - y) * np.log(1 - pr[m])))
    acc = float(np.mean((pr[m] > 0.5) == (y > 0.5)))
    return ll, acc


def cv_compare(Y: np.ndarray, folds: int = 5, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(Y))
    fold = rng.integers(0, folds, size=len(obs))
    out: dict = {}
    base = []
    for pl in (1, 2):
        lls, accs = [], []
        for f in range(folds):
            Ytr = Y.copy()
            te = obs[fold == f]
            Ytr[te[:, 0], te[:, 1]] = np.nan
            Yte = np.full_like(Y, np.nan)
            Yte[te[:, 0], te[:, 1]] = Y[te[:, 0], te[:, 1]]
            m = fit(Ytr, pl=pl, iters=150)
            ll, acc = pred_ll(Yte, m["post"], m["a"], m["b"])
            lls.append(ll)
            accs.append(acc)
            if pl == 1:
                pbar = np.clip(np.nanmean(Ytr, 0), 1e-3, 1 - 1e-3)
                y = Yte[te[:, 0], te[:, 1]]
                pb = pbar[te[:, 1]]
                base.append(float(np.mean(y * np.log(pb) + (1 - y) * np.log(1 - pb))))
        out[f"{pl}PL"] = {
            "heldout_loglik": round(float(np.mean(lls)), 4),
            "heldout_acc": round(float(np.mean(accs)), 4),
        }
    out["item_mean_only"] = {"heldout_loglik": round(float(np.mean(base)), 4)}
    return out


def human_theta(d: Data, a: np.ndarray, b: np.ndarray, basis_Y_row: np.ndarray) -> dict:
    post = posterior(basis_Y_row[None, :], a, b)[0]
    th = float(post @ NODES)
    return {
        "theta": round(th, 3),
        "se": round(float(np.sqrt(post @ NODES**2 - th**2)), 3),
        "cells": int((~np.isnan(basis_Y_row)).sum()),
    }


def subsample_curves(d: Data, n_rep: int = 300, seed: int = SEED) -> dict:
    """Ranking recovery and DiD CI width vs number of ITEMS (whole items, all variants).

    Metric = the leaderboard metric: contestant audio credit minus the cascade's on
    identical cue-bearing cells (DiD vs the null uses the twin; we use the simpler
    audio-vs-cascade-audio contrast, which is defined for every arm).
    """
    rng = np.random.default_rng(seed)
    con = contestants(d)
    cas = idx(d, "cascadeopen")
    cue_cells = np.where(d.cue)[0]
    items = np.unique(d.item_of[cue_cells])
    by_item = {i: cue_cells[d.item_of[cue_cells] == i] for i in items}

    def diffs(cells: np.ndarray) -> np.ndarray:
        X = d.credit[np.ix_(con, cells)]
        c = d.credit[cas, cells]
        return np.nanmean(X - c[None, :], axis=1)

    full = diffs(cue_cells)

    def kendall(x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        s = 0
        t = 0
        for i in range(n):
            for j in range(i + 1, n):
                s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
                t += 1
        return s / t

    ks = [20, 40, 60, 80, 100, len(items)]
    res = []
    for k in ks:
        taus = []
        for _ in range(n_rep if k < len(items) else 1):
            pick = rng.choice(items, size=k, replace=False) if k < len(items) else items
            cells = np.concatenate([by_item[i] for i in pick])
            dd = diffs(cells)
            taus.append(kendall(dd, full))
        res.append(
            {
                "items": int(k),
                "kendall_tau_mean": round(float(np.mean(taus)), 4),
                "kendall_tau_p05": round(float(np.quantile(taus, 0.05)), 4),
            }
        )
    # CI half-width of the per-arm contrast vs items: bootstrap at full n, then 1/sqrt(n) projection
    B = []
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(1000):
        pick = rng2.choice(items, size=len(items), replace=True)
        cells = np.concatenate([by_item[i] for i in pick])
        B.append(diffs(cells))
    B = np.array(B)
    lo, hi = np.quantile(B, [0.025, 0.975], axis=0)
    hw_full = (hi - lo) / 2
    n0 = len(items)
    proj = {
        str(n): round(float(np.median(hw_full) * np.sqrt(n0 / n)), 4) for n in (n0, 200, 300, 500)
    }
    return {
        "cue_items_full": int(n0),
        "ranking_recovery": res,
        "median_halfwidth_by_items": proj,
        "halfwidth_note": (
            "median over contestants of the item-clustered 95% CI half-width of audio "
            "credit minus cascade audio credit (cue-bearing); projected as 1/sqrt(items) "
            "at fixed variants-per-item"
        ),
        "mde80_at_300_items": round(float(np.median(hw_full) * np.sqrt(n0 / 300) / 1.96 * 2.80), 4),
        "mde80_now": round(float(np.median(hw_full) / 1.96 * 2.80), 4),
    }


def run(d: Data) -> dict:
    Y = d.passed.copy()  # strict pass, all AI respondents
    m2 = fit(Y, pl=2)
    m1 = fit(Y, pl=1)
    grid = np.linspace(-3, 3, 121)
    info = item_info(m2["a"], m2["b"], grid)
    cue = d.cue
    tinfo = {
        "grid": grid,
        "all": info.sum(0),
        "cue": info[cue].sum(0),
        "neutral": info[~cue].sum(0),
    }
    axes = sorted(set(d.axis))
    ax_info = {ax: info[np.array(d.axis) == ax].sum(0) for ax in axes}
    th = m2["theta"]
    con = contestants(d)
    # information in the roster's ability range (5th-95th pct of contestants)
    lo, hi = np.quantile(th[con], [0.05, 0.95])
    band = (grid >= lo) & (grid <= hi)
    per_item_band = info[:, band].mean(1)
    ax_rows = []
    for ax in axes:
        sel = np.array(d.axis) == ax
        ax_rows.append(
            {
                "axis": ax,
                "cells": int(sel.sum()),
                "median_a": round(float(np.median(m2["a"][sel])), 3),
                "median_b": round(float(np.median(m2["b"][sel])), 3),
                "info_in_roster_band": round(float(per_item_band[sel].sum()), 3),
                "info_per_cell_in_band": round(float(per_item_band[sel].mean()), 4),
                "share_of_band_info": round(
                    float(per_item_band[sel].sum() / per_item_band.sum()), 4
                ),
            }
        )
    ax_rows.sort(key=lambda r: -r["info_per_cell_in_band"])
    # reliability
    rel = 1 - np.mean(m2["se"][con] ** 2) / np.var(th[con])
    # sufficiency: info-optimal vs random subsets of CELLS for theta recovery
    rng = np.random.default_rng(SEED)
    order = np.argsort(-per_item_band)

    def rank_corr(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    suff = []
    for k in (30, 60, 100, 150, 200, 250, len(d.cells)):
        sel_opt = order[:k]
        post = posterior(Y[:, sel_opt], m2["a"][sel_opt], m2["b"][sel_opt])
        t_opt = post @ NODES
        rs = []
        for _ in range(100):
            s = rng.choice(len(d.cells), size=k, replace=False)
            pr = posterior(Y[:, s], m2["a"][s], m2["b"][s])
            rs.append(rank_corr(pr @ NODES, th))
        se_opt = np.sqrt(post @ NODES**2 - t_opt**2)
        suff.append(
            {
                "cells": int(k),
                "rho_theta_infoopt": round(rank_corr(t_opt, th), 4),
                "rho_theta_random_mean": round(float(np.mean(rs)), 4),
                "rho_theta_random_p05": round(float(np.quantile(rs, 0.05)), 4),
                "mean_se_infoopt": round(float(np.mean(se_opt[con])), 3),
            }
        )
    # respondent bootstrap for item-parameter stability (resample systems)
    rngb = np.random.default_rng(SEED + 7)
    Ab = []
    for _ in range(60):
        s = rngb.integers(0, Y.shape[0], size=Y.shape[0])
        mb = fit(Y[s], pl=2, iters=80)
        Ab.append(np.corrcoef(mb["b"], m2["b"])[0, 1])
    # Selection basis incl. humans
    Ys = (d.sel >= 0.5).astype(float)
    Ys[np.isnan(d.sel)] = np.nan
    ms = fit(Ys, pl=2)
    hrow = np.where(np.isnan(d.human_sel), np.nan, (d.human_sel >= 0.5).astype(float))
    hth = human_theta(d, ms["a"], ms["b"], hrow)
    # model thetas on selection basis restricted to the human cells (like-for-like)
    hmask = ~np.isnan(hrow)
    post_h = posterior(Ys[:, hmask], ms["a"][hmask], ms["b"][hmask])
    th_sel_h = post_h @ NODES
    cv = cv_compare(Y)
    sub = subsample_curves(d)
    return {
        "m1": m1,
        "m2": m2,
        "ms": ms,
        "tinfo": tinfo,
        "ax_info": ax_info,
        "summary": {
            "respondents": int(Y.shape[0]),
            "cells": int(Y.shape[1]),
            "model": "2PL, MML-EM (41-node quadrature), priors log a~N(0,.5^2), "
            "b~N(0,2^2), theta~N(0,1)",
            "cv": cv,
            "marginal_reliability_contestants": round(float(rel), 3),
            "roster_theta_band": [round(float(lo), 2), round(float(hi), 2)],
            "test_info_peak_theta": round(float(grid[np.argmax(tinfo["all"])]), 2),
            "cue_info_peak_theta": round(float(grid[np.argmax(tinfo["cue"])]), 2),
            "neutral_info_peak_theta": round(float(grid[np.argmax(tinfo["neutral"])]), 2),
            "median_a": round(float(np.median(m2["a"])), 3),
            "median_a_cue": round(float(np.median(m2["a"][cue])), 3),
            "median_a_neutral": round(float(np.median(m2["a"][~cue])), 3),
            "median_b_cue": round(float(np.median(m2["b"][cue])), 3),
            "median_b_neutral": round(float(np.median(m2["b"][~cue])), 3),
            "b_stability_respondent_bootstrap_r_median": round(float(np.median(Ab)), 3),
            "b_stability_respondent_bootstrap_r_p05": round(float(np.quantile(Ab, 0.05)), 3),
            "by_axis": ax_rows,
            "sufficiency_cells": suff,
            "items_subsample": sub,
            "human_theta_selection_basis": hth,
            "rho_theta_strict_vs_selection": round(
                float(
                    np.corrcoef(
                        np.argsort(np.argsort(m2["theta"])), np.argsort(np.argsort(ms["theta"]))
                    )[0, 1]
                ),
                3,
            ),
        },
        "theta_rows": sorted(
            [
                {
                    "label": d.labels[j],
                    "name": d.names[j],
                    "role": d.roles[j],
                    "mode": d.modes[j],
                    "theta": round(float(m2["theta"][j]), 3),
                    "se": round(float(m2["se"][j]), 3),
                    "theta_sel_on_human_cells": round(float(th_sel_h[j]), 3),
                }
                for j in range(len(d.labels))
            ],
            key=lambda r: -r["theta"],
        ),
    }
