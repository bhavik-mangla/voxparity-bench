"""Q3 - clustering systems (and humans) by response pattern.

Basis: SELECTION credit >= 0.5 per cell (the basis humans share with models,
D114; the strict-pass ability ranking is rho 0.99 with it), so the human pool can
sit in the same space. The human "respondent" is the pooled majority (cell mean
selection credit >= 0.5) on the cells humans answered.

- pairwise Cohen's kappa on the binary correct vectors (overlapping cells), and
  raw first-tool agreement (which action was chosen, right or wrong);
- average-linkage hierarchical clustering on 1 - kappa; classical MDS 2-D;
- metadata association: mean within-group kappa minus mean between-group kappa
  for vendor, serving mode, weights (open/closed), permutation p (5000);
- transcript-likeness: on cue-bearing cells, the share where the system's audio
  action equals (a) its OWN text-twin action on the same words, (b) the words-only
  cascade's audio action; item-clustered bootstrap CIs;
- human-likeness: expected agreement of the system's action with a random human
  answer on the same cell, against the human-human ceiling (pairs of answers from
  different players on the same cell).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from psych_common import HUMAN, META, SEED, Data, contestants
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def kappa(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return np.nan
    po = np.mean(x == y)
    px, py = x.mean(), y.mean()
    pe = px * py + (1 - px) * (1 - py)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def tool_agree(a: list, b: list, cells: np.ndarray) -> float:
    v = [a[c] == b[c] for c in cells if a[c] is not None and b[c] is not None]
    return float(np.mean(v)) if v else np.nan


def _boot_share(flags: dict[int, bool], d: Data, n: int = 2000) -> dict:
    cells = np.array(sorted(flags))
    if not len(cells):
        return {}
    vals = np.array([flags[c] for c in cells], float)
    items = d.item_of[cells]
    uniq = np.unique(items)
    by = {i: vals[items == i] for i in uniq}
    rng = np.random.default_rng(SEED)
    B = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq))
        B.append(np.concatenate([by[i] for i in pick]).mean())
    return {
        "est": round(float(vals.mean()), 4),
        "lo": round(float(np.quantile(B, 0.025)), 4),
        "hi": round(float(np.quantile(B, 0.975)), 4),
        "n": len(cells),
    }


def run(d: Data, irt: dict | None = None) -> dict:
    labels = list(d.labels)
    S = np.where(np.isnan(d.sel), np.nan, (d.sel >= 0.5).astype(float))
    H = np.where(np.isnan(d.human_sel), np.nan, (d.human_sel >= 0.5).astype(float))
    R = np.vstack([S, H[None, :]])
    names = [*labels, HUMAN]
    n = len(names)
    K = np.full((n, n), 1.0)
    for i, j in combinations(range(n), 2):
        K[i, j] = K[j, i] = kappa(R[i], R[j])
    # tool agreement (models only; all cells and cue cells)
    allc = np.arange(len(d.cells))
    cue = np.where(d.cue)[0]
    TA = np.full((len(labels), len(labels)), 1.0)
    for i, j in combinations(range(len(labels)), 2):
        TA[i, j] = TA[j, i] = tool_agree(d.tool[i], d.tool[j], allc)
    # clustering (all AI + humans)
    D = 1 - np.nan_to_num(K, nan=0.0)
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    dn = dendrogram(Z, no_plot=True, labels=names)
    # classical MDS
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    xy = V[:, o[:2]] * np.sqrt(np.maximum(w[o[:2]], 0))
    # metadata association (contestants only)
    con = contestants(d)
    groups = {
        "vendor": [META.get(labels[j], {}).get("vendor", "?") for j in con],
        "serving mode": [d.modes[j] for j in con],
        "weights": [META.get(labels[j], {}).get("weights", "?") for j in con],
    }
    Kc = K[np.ix_(con, con)]
    rng = np.random.default_rng(SEED)
    assoc = {}
    for gname, g in groups.items():
        g = np.array(g)

        def stat(gg):
            same = gg[:, None] == gg[None, :]
            iu = np.triu_indices(len(gg), 1)
            s, t = same[iu], Kc[iu]
            return float(np.nanmean(t[s]) - np.nanmean(t[~s])) if s.any() and (~s).any() else np.nan

        obs = stat(g)
        null = [stat(rng.permutation(g)) for _ in range(5000)]
        assoc[gname] = {
            "within_minus_between_kappa": round(obs, 4),
            "perm_p": round(float((np.sum(np.array(null) >= obs) + 1) / 5001), 4),
            "groups": dict(zip(*np.unique(g, return_counts=True), strict=False)),
        }
        assoc[gname]["groups"] = {str(k): int(v) for k, v in assoc[gname]["groups"].items()}
    # Yen's Q3 on respondents: correlate 2PL residuals (strict pass minus the model's
    # expected pass at the system's posterior), i.e. shared style beyond ability level
    resid_assoc = None
    q3 = None
    if irt is not None:
        m2 = irt["m2"]
        nodes = np.linspace(-4, 4, 41)
        Pq = 1 / (1 + np.exp(-m2["a"][:, None] * (nodes[None, :] - m2["b"][:, None])))
        E = m2["post"] @ Pq.T
        Rz = d.passed - E
        Rc = Rz[con]
        Q = np.full((len(con), len(con)), 1.0)
        for i, j in combinations(range(len(con)), 2):
            mm = np.isfinite(Rc[i]) & np.isfinite(Rc[j])
            Q[i, j] = Q[j, i] = np.corrcoef(Rc[i][mm], Rc[j][mm])[0, 1]
        q3 = Q
        resid_assoc = {}
        for gname, g in groups.items():
            g = np.array(g)

            def st(gg):
                same = gg[:, None] == gg[None, :]
                iu = np.triu_indices(len(gg), 1)
                return float(np.mean(Q[iu][same[iu]]) - np.mean(Q[iu][~same[iu]]))

            ob = st(g)
            nl = [st(rng.permutation(g)) for _ in range(5000)]
            resid_assoc[gname] = {
                "within_minus_between_q3": round(ob, 4),
                "perm_p": round(float((np.sum(np.array(nl) >= ob) + 1) / 5001), 4),
            }
        iu = np.triu_indices(len(con), 1)
        top = np.argsort(-Q[iu])[:8]
        resid_assoc["top_pairs"] = [
            {
                "a": labels[con[iu[0][t]]],
                "b": labels[con[iu[1][t]]],
                "q3": round(float(Q[iu][t]), 3),
            }
            for t in top
        ]
        resid_assoc["median_q3"] = round(float(np.median(Q[iu])), 3)
    # delivery-sensitivity index (tool level): audio==own-twin on neutral cells minus on
    # cue cells, item-clustered bootstrap (items resampled jointly for both terms)
    dsi = []
    rng3 = np.random.default_rng(SEED)
    uitems = np.unique(d.item_of)
    by_item = {u: np.where(d.item_of == u)[0] for u in uitems}
    picks = [
        np.concatenate([by_item[u] for u in rng3.choice(uitems, len(uitems))]) for _ in range(2000)
    ]
    for j in range(len(labels)):
        same = np.array(
            [
                np.nan
                if d.tool[j][c] is None or d.twin_tool[j][c] is None
                else float(d.tool[j][c] == d.twin_tool[j][c])
                for c in range(len(d.cells))
            ]
        )
        if np.isfinite(same).sum() < 100:
            continue

        def val(ix, same=same):
            cm = d.cue[ix]
            return np.nanmean(same[ix][~cm]) - np.nanmean(same[ix][cm])

        est = val(np.arange(len(d.cells)))
        B = [val(ix) for ix in picks]
        dsi.append(
            {
                "label": labels[j],
                "name": d.names[j],
                "role": d.roles[j],
                "est": round(float(est), 4),
                "lo": round(float(np.quantile(B, 0.025)), 4),
                "hi": round(float(np.quantile(B, 0.975)), 4),
            }
        )
    dsi.sort(key=lambda r: -r["est"])
    # nearest neighbour
    nn = {}
    for i in range(n):
        k = K[i].copy()
        k[i] = -np.inf
        j = int(np.nanargmax(k))
        nn[names[i]] = {"nearest": names[j], "kappa": round(float(K[i, j]), 3)}
    # transcript-likeness
    cas = labels.index("cascadeopen")
    tl = []
    for j in range(len(labels)):
        own = {
            c: d.tool[j][c] == d.twin_tool[j][c]
            for c in cue
            if d.tool[j][c] is not None and d.twin_tool[j][c] is not None
        }
        casc = {
            c: d.tool[j][c] == d.tool[cas][c]
            for c in cue
            if d.tool[j][c] is not None and d.tool[cas][c] is not None
        }
        neu_own = {
            c: d.tool[j][c] == d.twin_tool[j][c]
            for c in np.where(~d.cue)[0]
            if d.tool[j][c] is not None and d.twin_tool[j][c] is not None
        }
        tl.append(
            {
                "label": labels[j],
                "name": d.names[j],
                "role": d.roles[j],
                "mode": d.modes[j],
                "same_as_own_twin_cue": _boot_share(own, d) if own else None,
                "same_as_own_twin_neutral": _boot_share(neu_own, d) if neu_own else None,
                "same_as_cascade_cue": _boot_share(casc, d) if j != cas else None,
                "kappa_with_cascade": round(float(K[j, cas]), 3),
                "kappa_with_humans": round(float(K[j, n - 1]), 3),
            }
        )
    # human-likeness (action agreement with a random human answer)
    hcells = [c for c in range(len(d.cells)) if d.cells[c] in d.human_raters]
    hh = []  # human-human pairs
    per_cell_pairs = {}
    for c in hcells:
        ans = d.human_raters[d.cells[c]]
        ps = [(a[0] == b[0]) for a, b in combinations(ans, 2) if a[2] != b[2]]
        if ps:
            per_cell_pairs[c] = float(np.mean(ps))
            hh.extend(ps)
    hl = []
    for j in range(len(labels)):
        flags = {}
        for c in hcells:
            t = d.tool[j][c]
            if t is None:
                continue
            flags[c] = float(np.mean([t == a[0] for a in d.human_raters[d.cells[c]]]))
        agree_all = _boot_share(flags, d)
        pair_cells = [c for c in per_cell_pairs if c in flags]
        hl.append(
            {
                "label": labels[j],
                "name": d.names[j],
                "agree_with_random_human": agree_all,
                "agree_on_multi_rater_cells": round(
                    float(np.mean([flags[c] for c in pair_cells])), 4
                )
                if pair_cells
                else None,
            }
        )
    hceil = {
        "human_human_agreement": round(float(np.mean(list(per_cell_pairs.values()))), 4),
        "cells_with_pairs": len(per_cell_pairs),
        "pairs": len(hh),
    }
    tl.sort(key=lambda r: -(r["same_as_own_twin_cue"] or {}).get("est", -1))
    hl.sort(key=lambda r: -r["agree_with_random_human"]["est"])
    return {
        "basis": "selection credit >= 0.5 (humans: cell mean >= 0.5, pooled majority)",
        "names": names,
        "kappa": [[round(float(x), 3) if np.isfinite(x) else None for x in row] for row in K],
        "tool_agreement": [[round(float(x), 3) for x in row] for row in TA],
        "linkage": Z.tolist(),
        "dendrogram_order": dn["ivl"],
        "mds_xy": {
            names[i]: [round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3)] for i in range(n)
        },
        "metadata_association": assoc,
        "nearest_neighbour": nn,
        "transcript_likeness": tl,
        "human_likeness": hl,
        "human_ceiling": hceil,
        "residual_q3_association": resid_assoc,
        "q3_matrix_contestants": None
        if q3 is None
        else [[round(float(x), 3) for x in r] for r in q3],
        "delivery_sensitivity_index": dsi,
    }
