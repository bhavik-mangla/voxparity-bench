"""Lens 4: which system properties predict listening?

Inputs
  * the cell cache written by models_load.py (per-cell audio/twin credit, probe,
    first tool, latency; primary Gemini-TTS engine, eligible arms only);
  * docs/results/final/paper_pareto.json (measured $/cell) and
    paper_leaderboard.json (Holm p, used only for cross-checking);
  * models_metadata.META (public metadata with sources).

Outputs docs/insights/models.json (every number the write-up quotes).

Conventions follow the frozen tables: item-clustered percentile bootstrap,
seed 20260915; "listening" = vs-floor = difference-in-differences of
audio-minus-twin against the words-only cascade on identical cue-bearing cells
(twin-less arms: audio credit minus cascade audio credit). Across-system
statistics use n <= 28 systems, so every across-system CI resamples SYSTEMS and
ITEMS jointly (systems are the unit of inference for a property claim; items
carry each system's measurement error).

    uv run --extra paper python scripts/insights/models_analysis.py
"""

from __future__ import annotations

import json
import math
import os
from datetime import date
from pathlib import Path
from statistics import median

import numpy as np
from models_metadata import META

np.seterr(all="ignore")  # Accelerate matmul emits spurious FP warnings on macOS

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "docs/results/final"
OUT = ROOT / "docs/insights"
CACHE = Path(
    os.environ.get(
        "VXP_INSIGHTS_CACHE", Path.home() / ".cache/voxparity-insights/models_cells.json"
    )
)
SEED = 20260915
NB = 4000
CASCADE = "cascadeopen"
CLARIFY = "ask_clarifying_question"
NON_CONTESTANT = {"cascadeopen", "cascverbatim", "cascadeemo", "ultravox8b"}
REF_DATE = date(2025, 1, 1)


# ----------------------------------------------------------------------------- data


def load() -> dict:
    d = json.loads(CACHE.read_text())
    arms = {}
    for lab, a in d["arms"].items():
        cells = {tuple(k.split("|", 1)): v for k, v in a["cells"].items()}
        arms[lab] = {**a, "cells": cells}
    return arms


ARMS = load()
CASC = ARMS[CASCADE]["cells"]
CUE_KEYS = sorted(
    k for k, v in CASC.items() if v["cue"] and v["a"] is not None and v["t"] is not None
)
ITEMS = sorted({k[0] for k in CUE_KEYS})
ITEM_IX = {it: i for i, it in enumerate(ITEMS)}
ALL_KEYS = sorted(k for k, v in CASC.items() if v["a"] is not None)
ALL_ITEMS = sorted({k[0] for k in ALL_KEYS})
ALL_IX = {it: i for i, it in enumerate(ALL_ITEMS)}
CONTESTANTS = sorted(lab for lab, a in ARMS.items() if a["role"] == "contestant")

pareto = {
    r["label"]: r for r in json.loads((FINAL / "paper_pareto.json").read_text())["pareto"]["arms"]
}
board = {
    r["label"]: r
    for r in json.loads((FINAL / "paper_leaderboard.json").read_text())["leaderboard"]["rows"]
}


def did_cells(lab: str) -> dict:
    """Per cue-bearing cell listening contribution against the floor."""
    c = ARMS[lab]["cells"]
    has_twin = any(v["t"] is not None for v in c.values())
    out = {}
    for k in CUE_KEYS:
        v = c.get(k)
        if v is None or v["a"] is None:
            continue
        cv = CASC[k]
        if has_twin:
            if v["t"] is None:
                continue
            out[k] = (v["a"] - v["t"]) - (cv["a"] - cv["t"])
        else:
            out[k] = v["a"] - cv["a"]
    return out


def flag_cells(lab: str, fn, keys=None, cue_only=False) -> dict:
    c = ARMS[lab]["cells"]
    out = {}
    for k, v in c.items():
        if keys is not None and k not in keys:
            continue
        if cue_only and not v["cue"]:
            continue
        x = fn(v)
        if x is not None:
            out[k] = float(x)
    return out


def credit(v):
    return v["a"]


def acted(v):
    return None if not v["has_audio"] else v["tool"] is not None


def probe(v):
    return None if v["p"] is None else bool(v["p"])


# ----------------------------------------------------------------------------- bootstrap


def _item_matrix(cells: dict, ix: dict) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros(len(ix))
    n = np.zeros(len(ix))
    for k, x in cells.items():
        if k[0] in ix:
            s[ix[k[0]]] += x
            n[ix[k[0]]] += 1
    return s, n


def _weights(n_items: int, nb: int = NB, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_items, size=(nb, n_items))
    w = np.zeros((nb, n_items))
    np.add.at(w, (np.arange(nb)[:, None], draw), 1.0)
    return w


W_CUE = _weights(len(ITEMS))
W_ALL = _weights(len(ALL_ITEMS), seed=SEED + 1)


def ci_mean(cells: dict, cue: bool = True) -> dict | None:
    ix, W = (ITEM_IX, W_CUE) if cue else (ALL_IX, W_ALL)
    s, n = _item_matrix(cells, ix)
    if n.sum() == 0:
        return None
    boot = (W @ s) / np.maximum(W @ n, 1e-12)
    return {
        "mean": round(float(s.sum() / n.sum()), 4),
        "lo": round(float(np.quantile(boot, 0.025)), 4),
        "hi": round(float(np.quantile(boot, 0.975)), 4),
        "n": int(n.sum()),
        "p_two_sided": round(float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))), 4),
    }


def paired(a: dict, b: dict, cue: bool = True) -> dict | None:
    ks = set(a) & set(b)
    return ci_mean({k: a[k] - b[k] for k in ks}, cue=cue)


def boot_means(cells: dict, cue: bool = True) -> tuple[float, np.ndarray]:
    ix, W = (ITEM_IX, W_CUE) if cue else (ALL_IX, W_ALL)
    s, n = _item_matrix(cells, ix)
    return float(s.sum() / n.sum()), (W @ s) / np.maximum(W @ n, 1e-12)


# ----------------------------------------------------------------------------- per-system table


def months_since(d: str) -> float:
    y, m, dd = map(int, d.split("-"))
    return (date(y, m, dd) - REF_DATE).days / 30.4375


SYS: dict[str, dict] = {}
BOOT: dict[str, dict[str, np.ndarray]] = {}
for lab in sorted(ARMS):
    a = ARMS[lab]
    c = a["cells"]
    did = did_cells(lab) if lab != CASCADE else {}
    cue_credit = flag_cells(lab, credit, cue_only=True)
    act = flag_cells(lab, acted)
    act_cue = flag_cells(lab, acted, cue_only=True)
    pr_cue = flag_cells(lab, probe, cue_only=True)
    twin_cue = {
        k: v["t"] for k, v in c.items() if v["cue"] and v["t"] is not None and v["a"] is not None
    }
    pr_all = flag_cells(lab, probe)
    clar = flag_cells(lab, lambda v: None if not v["has_audio"] else v["tool"] == CLARIFY)
    # hears-but-doesn't-act: cue cells the probe got right but the action missed
    heard = [k for k, v in c.items() if v["cue"] and v["p"] and v["a"] is not None]
    hbda = {k: float(not c[k]["ap"]) for k in heard}
    missed = [k for k, v in c.items() if v["cue"] and v["p"] is False and v["a"] is not None]
    lats = [v["lat"] for v in c.values() if v["lat"] is not None]
    m = META.get(lab, {})
    row = {
        "label": lab,
        "name": a["name"],
        "role": a["role"],
        "mode": a["mode"],
        **{
            k: m.get(k)
            for k in (
                "vendor",
                "family",
                "release",
                "date_basis",
                "open_weights",
                "params_total_b",
                "params_active_b",
                "arch",
                "speech_out",
                "serving",
                "sources",
            )
        },
        "twin": any(v["t"] is not None for v in c.values()),
        "vs_floor": ci_mean(did) if did else None,
        "cue_credit": ci_mean(cue_credit),
        "probe_acc_cue": ci_mean(pr_cue) if pr_cue else None,
        "twin_credit_cue": ci_mean(twin_cue) if twin_cue else None,
        "probe_acc_all": ci_mean(pr_all, cue=False) if pr_all else None,
        "act_rate": ci_mean(act, cue=False),
        "act_rate_cue": ci_mean(act_cue),
        "clarify_rate": ci_mean(clar, cue=False),
        "hears_not_acts": ci_mean(hbda) if hbda else None,
        "p_correct_given_heard": round(float(np.mean([c[k]["a"] for k in heard])), 4)
        if heard
        else None,
        "p_correct_given_missed": round(float(np.mean([c[k]["a"] for k in missed])), 4)
        if missed
        else None,
        "median_latency_s": round(median(lats), 3) if lats else None,
        "usd_per_cell": (pareto.get(lab) or {}).get("usd_per_cell"),
        "cost_basis": (pareto.get(lab) or {}).get("cost_basis"),
        "p_holm": (board.get(lab) or {}).get("p_holm"),
    }
    SYS[lab] = row
    BOOT[lab] = {}
    if did:
        BOOT[lab]["vs_floor"] = boot_means(did)[1]
    BOOT[lab]["cue_credit"] = boot_means(cue_credit)[1]
    if pr_cue:
        BOOT[lab]["probe_acc_cue"] = boot_means(pr_cue)[1]
    BOOT[lab]["act_rate_cue"] = boot_means(act_cue)[1]
    if twin_cue:
        BOOT[lab]["twin_credit_cue"] = boot_means(twin_cue)[1]
    if hbda:
        BOOT[lab]["hears_not_acts"] = boot_means(hbda)[1]

# sanity: our vs-floor must reproduce the frozen leaderboard to 3 dp
CHECK = {
    lab: (SYS[lab]["vs_floor"]["mean"], board[lab]["vs_cascade"]["mean"])
    for lab in CONTESTANTS
    if SYS[lab]["vs_floor"] and lab in board and isinstance(board[lab].get("vs_cascade"), dict)
}
assert all(abs(x - y) < 1e-3 for x, y in CHECK.values()), CHECK


# ----------------------------------------------------------------------------- rank stats


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y) -> float:
    x, y = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def theil_sen(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    sl = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(len(x))
        for j in range(i + 1, len(x))
        if x[j] != x[i]
    ]
    return float(np.median(sl)) if sl else float("nan")


def mde_rho(n: int) -> float:
    """Smallest |rho| detectable at 80% power, alpha .05 two-sided (Fisher z, var x1.06)."""
    z = (1.959964 + 0.841621) * math.sqrt(1.06 / (n - 3))
    return round(math.tanh(z), 2)


def across(labels: list[str], xfun, ykey: str, *, stat: str = "spearman", nb: int = 2000) -> dict:
    """Across-system association with a joint system x item bootstrap.

    x is a fixed system property (xfun(label) -> float|None); y is a measured
    per-system quantity whose item-bootstrap replicates live in BOOT.
    """
    labs = [
        lab for lab in labels if xfun(lab) is not None and ykey in BOOT[lab] and SYS[lab].get(ykey)
    ]
    x = np.array([xfun(lab) for lab in labs], float)
    y = np.array([SYS[lab][ykey]["mean"] for lab in labs], float)
    f = spearman if stat == "spearman" else theil_sen
    est = f(x, y)
    rng = np.random.default_rng(SEED + 7)
    reps = []
    for b in range(nb):
        pick = rng.integers(0, len(labs), len(labs))
        yb = np.array([BOOT[labs[i]][ykey][b % NB] for i in pick])
        xb = x[pick]
        if len(set(xb)) < 3:
            continue
        v = f(xb, yb)
        if not math.isnan(v):
            reps.append(v)
    # permutation p on the point estimate (system labels shuffled)
    perm = [f(x, rng.permutation(y)) for _ in range(2000)]
    p = (1 + sum(abs(v) >= abs(est) for v in perm)) / (1 + len(perm))
    return {
        "stat": stat,
        "estimate": round(est, 4),
        "lo": round(float(np.quantile(reps, 0.025)), 4),
        "hi": round(float(np.quantile(reps, 0.975)), 4),
        "p_perm": round(p, 4),
        "n_systems": len(labs),
        "mde_rho_80pct": mde_rho(len(labs)) if stat == "spearman" else None,
        "systems": labs,
    }


def group_diff(labels, gfun, ykey, nb: int = 2000) -> dict:
    """mean(y | g True) - mean(y | g False), systems x items bootstrap."""
    labs = [
        lab for lab in labels if gfun(lab) is not None and ykey in BOOT[lab] and SYS[lab].get(ykey)
    ]
    g = np.array([bool(gfun(lab)) for lab in labs])
    y = np.array([SYS[lab][ykey]["mean"] for lab in labs])
    est = y[g].mean() - y[~g].mean()
    rng = np.random.default_rng(SEED + 11)
    reps = []
    ia, ib = np.where(g)[0], np.where(~g)[0]
    for b in range(nb):
        pa = rng.choice(ia, len(ia))
        pb = rng.choice(ib, len(ib))
        reps.append(
            np.mean([BOOT[labs[i]][ykey][b % NB] for i in pa])
            - np.mean([BOOT[labs[i]][ykey][b % NB] for i in pb])
        )
    perm = []
    for _ in range(2000):
        gp = rng.permutation(g)
        perm.append(y[gp].mean() - y[~gp].mean())
    p = (1 + sum(abs(v) >= abs(est) for v in perm)) / (1 + len(perm))
    return {
        "diff": round(float(est), 4),
        "lo": round(float(np.quantile(reps, 0.025)), 4),
        "hi": round(float(np.quantile(reps, 0.975)), 4),
        "p_perm": round(p, 4),
        "n_true": int(g.sum()),
        "n_false": int((~g).sum()),
        "mean_true": round(float(y[g].mean()), 4),
        "mean_false": round(float(y[~g].mean()), 4),
    }


def date_x(lab):
    r = SYS[lab].get("release")
    return months_since(r) if r else None


def log_active(lab):
    p = SYS[lab].get("params_active_b")
    return math.log10(p) if p else None


def log_total(lab):
    p = SYS[lab].get("params_total_b")
    return math.log10(p) if p else None


def price_x(lab):
    u = SYS[lab].get("usd_per_cell")
    return u if (u is not None and u > 0) else None


def lat_x(lab):
    return SYS[lab].get("median_latency_s")


def probe_x(lab):
    p = SYS[lab].get("probe_acc_cue")
    return p["mean"] if p else None


def act_x(lab):
    return SYS[lab]["act_rate_cue"]["mean"]


C = CONTESTANTS
PROPS: dict[str, dict] = {
    "release_date_vs_listening": across(C, date_x, "vs_floor"),
    "release_date_vs_listening_slope_per_month": across(C, date_x, "vs_floor", stat="theil_sen"),
    "release_date_vs_cue_credit": across(C, date_x, "cue_credit"),
    "release_date_vs_probe": across(C, date_x, "probe_acc_cue"),
    "release_date_vs_twin_credit": across(C, date_x, "twin_credit_cue"),
    "release_date_vs_act_rate": across(C, date_x, "act_rate_cue"),
    "price_vs_listening": across(C, price_x, "vs_floor"),
    "latency_vs_listening": across(C, lat_x, "vs_floor"),
    "latency_vs_cue_credit": across(C, lat_x, "cue_credit"),
    "log_active_params_vs_listening": across(C, log_active, "vs_floor"),
    "log_total_params_vs_listening": across(C, log_total, "vs_floor"),
    "probe_vs_listening": across(C, probe_x, "vs_floor"),
    "probe_vs_cue_credit": across(C, probe_x, "cue_credit"),
    "act_rate_vs_listening": across(C, act_x, "vs_floor"),
    "act_rate_vs_cue_credit": across(C, act_x, "cue_credit"),
    "act_rate_vs_hears_not_acts": across(C, act_x, "hears_not_acts"),
    "release_date_vs_hears_not_acts": across(C, date_x, "hears_not_acts"),
    "probe_vs_hears_not_acts": across(C, probe_x, "hears_not_acts"),
}

GROUPS = {
    "open_minus_closed_listening": group_diff(C, lambda lab: SYS[lab]["open_weights"], "vs_floor"),
    "open_minus_closed_cue_credit": group_diff(
        C, lambda lab: SYS[lab]["open_weights"], "cue_credit"
    ),
    "realtime_minus_nonrealtime_listening": group_diff(
        C, lambda lab: SYS[lab]["serving"] == "realtime", "vs_floor"
    ),
    "realtime_minus_nonrealtime_probe": group_diff(
        C, lambda lab: SYS[lab]["serving"] == "realtime", "probe_acc_cue"
    ),
    "speech_out_minus_text_out_listening": group_diff(
        C, lambda lab: SYS[lab]["speech_out"], "vs_floor"
    ),
    "local_minus_hosted_listening": group_diff(
        C, lambda lab: SYS[lab]["serving"] == "local", "vs_floor"
    ),
    "realtime_minus_nonrealtime_hears_not_acts": group_diff(
        C, lambda lab: SYS[lab]["serving"] == "realtime", "hears_not_acts"
    ),
    "open_minus_closed_hears_not_acts": group_diff(
        C, lambda lab: SYS[lab]["open_weights"], "hears_not_acts"
    ),
    "released_2026H2_minus_earlier_listening": group_diff(
        C, lambda lab: SYS[lab]["release"] >= "2026-07-01", "vs_floor"
    ),
}

# partial Spearman: probe -> listening controlling act rate (residualised ranks)


def partial_spearman(labels, xf, yk, zf):
    labs = [
        lab for lab in labels if xf(lab) is not None and SYS[lab].get(yk) and zf(lab) is not None
    ]
    rx = _rank(np.array([xf(lab) for lab in labs]))
    ry = _rank(np.array([SYS[lab][yk]["mean"] for lab in labs]))
    rz = _rank(np.array([zf(lab) for lab in labs]))

    def res(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]

    return {
        "rho_partial": round(float(np.corrcoef(res(rx, rz), res(ry, rz))[0, 1]), 4),
        "n": len(labs),
    }


PARTIAL = {
    "probe_vs_listening_given_act_rate": partial_spearman(C, probe_x, "vs_floor", act_x),
    "act_rate_vs_listening_given_probe": partial_spearman(C, act_x, "vs_floor", probe_x),
    "release_date_vs_listening_given_probe": partial_spearman(C, date_x, "vs_floor", probe_x),
}

# within-family release trend (vendor fixed effects): demean date and y within family


def within_family_slope():
    fams: dict[str, list[str]] = {}
    for lab in C:
        fams.setdefault(SYS[lab]["family"], []).append(lab)
    # treat Gemini file/live, Qwen omni file/RT and OpenAI audio/realtime as one vendor line each
    vendor_line = {}
    for lab in C:
        vendor_line.setdefault(SYS[lab]["vendor"], []).append(lab)
    groups = [g for g in vendor_line.values() if len({SYS[lab]["release"] for lab in g}) >= 2]
    labs = [lab for g in groups for lab in g]

    def fit(ylist, labs_):
        xs, ys = [], []
        for g in groups:
            gl = [lab for lab in g if lab in labs_]
            if len(gl) < 2:
                continue
            xg = np.array([date_x(lab) for lab in gl])
            yg = np.array([ylist[labs_.index(lab)] for lab in gl])
            xs.extend(xg - xg.mean())
            ys.extend(yg - yg.mean())
        xs, ys = np.array(xs), np.array(ys)
        return float((xs * ys).sum() / (xs * xs).sum()) if (xs * xs).sum() else float("nan")

    y = [SYS[lab]["vs_floor"]["mean"] for lab in labs]
    est = fit(y, labs)
    reps = []
    for b in range(NB):
        yb = [BOOT[lab]["vs_floor"][b] for lab in labs]
        reps.append(fit(yb, labs))
    return {
        "slope_per_month": round(est, 4),
        "lo": round(float(np.quantile(reps, 0.025)), 4),
        "hi": round(float(np.quantile(reps, 0.975)), 4),
        "ci_basis": "item bootstrap only (vendor lines fixed); systems not resampled",
        "vendor_lines": {SYS[g[0]]["vendor"]: g for g in groups},
    }


WITHIN = within_family_slope()


# ----------------------------------------------------------------------------- generations


def decomposition(new: str, old: str) -> dict:
    """Hearing vs using decomposition of the change in cue-bearing credit.

    credit = p*c1 + (1-p)*c0 with p = probe accuracy, c1 = E[credit | probe right],
    c0 = E[credit | probe wrong] on cue cells both arms hold with a probe.
    Delta = dp*(c1bar - c0bar)  [hearing]  + pbar*dc1 + (1-pbar)*dc0  [using].
    """
    cn, co = ARMS[new]["cells"], ARMS[old]["cells"]
    ks = [
        k
        for k in CUE_KEYS
        if k in cn
        and k in co
        and cn[k]["p"] is not None
        and co[k]["p"] is not None
        and cn[k]["a"] is not None
        and co[k]["a"] is not None
    ]
    if not ks:
        return {"status": "n/a (no probe on one side)"}
    items = sorted({k[0] for k in ks})
    ix = {it: i for i, it in enumerate(items)}

    def parts(cells, weights):
        # weighted sums per item
        s = {
            "p": np.zeros(len(items)),
            "n": np.zeros(len(items)),
            "c1": np.zeros(len(items)),
            "n1": np.zeros(len(items)),
            "c0": np.zeros(len(items)),
            "n0": np.zeros(len(items)),
        }
        for k in ks:
            i = ix[k[0]]
            v = cells[k]
            s["n"][i] += 1
            if v["p"]:
                s["p"][i] += 1
                s["c1"][i] += v["a"]
                s["n1"][i] += 1
            else:
                s["c0"][i] += v["a"]
                s["n0"][i] += 1
        return s

    sn, so = parts(cn, None), parts(co, None)
    rng = np.random.default_rng(SEED + 3)
    Wb = np.zeros((NB, len(items)))
    np.add.at(Wb, (np.arange(NB)[:, None], rng.integers(0, len(items), (NB, len(items)))), 1.0)
    Wb = np.vstack([np.ones(len(items)), Wb])

    def comp(s):
        n = Wb @ s["n"]
        p = (Wb @ s["p"]) / n
        c1 = (Wb @ s["c1"]) / np.maximum(Wb @ s["n1"], 1e-9)
        c0 = (Wb @ s["c0"]) / np.maximum(Wb @ s["n0"], 1e-9)
        return p, c1, c0

    pn, c1n, c0n = comp(sn)
    po, c1o, c0o = comp(so)
    hear = (pn - po) * ((c1n + c1o) / 2 - (c0n + c0o) / 2)
    use = (pn + po) / 2 * (c1n - c1o) + (1 - (pn + po) / 2) * (c0n - c0o)
    total = hear + use

    def summ(v):
        return {
            "mean": round(float(v[0]), 4),
            "lo": round(float(np.quantile(v[1:], 0.025)), 4),
            "hi": round(float(np.quantile(v[1:], 0.975)), 4),
        }

    return {
        "cells": len(ks),
        "items": len(items),
        "delta_cue_credit": summ(total),
        "hearing_part": summ(hear),
        "using_part": summ(use),
        "probe_old": round(float(po[0]), 4),
        "probe_new": round(float(pn[0]), 4),
        "c1_old": round(float(c1o[0]), 4),
        "c1_new": round(float(c1n[0]), 4),
        "c0_old": round(float(c0o[0]), 4),
        "c0_new": round(float(c0n[0]), 4),
    }


def compare(new: str, old: str, kind: str) -> dict:
    """new minus old on identical cells."""
    row: dict = {
        "new": new,
        "old": old,
        "kind": kind,
        "new_name": SYS[new]["name"],
        "old_name": SYS[old]["name"],
        "new_release": SYS[new]["release"],
        "old_release": SYS[old]["release"],
    }
    dn, do = did_cells(new), did_cells(old)
    same_metric = SYS[new]["twin"] == SYS[old]["twin"]
    row["d_vs_floor"] = (
        paired(dn, do) if same_metric else "n/a (one side has no text twin: metrics differ)"
    )
    row["vs_floor_metric_new"] = "did" if SYS[new]["twin"] else "audio-minus-cascade-audio"
    row["vs_floor_metric_old"] = "did" if SYS[old]["twin"] else "audio-minus-cascade-audio"
    row["d_cue_credit"] = paired(
        flag_cells(new, credit, cue_only=True), flag_cells(old, credit, cue_only=True)
    )
    row["d_credit_all"] = paired(flag_cells(new, credit), flag_cells(old, credit), cue=False)
    pn, po = flag_cells(new, probe, cue_only=True), flag_cells(old, probe, cue_only=True)
    row["d_probe_cue"] = paired(pn, po) if pn and po else "n/a"
    row["d_act_rate"] = paired(flag_cells(new, acted), flag_cells(old, acted), cue=False)
    hn = {
        k: 1.0 - ARMS[new]["cells"][k]["ap"]
        for k in pn
        if pn[k] and ARMS[new]["cells"][k]["ap"] is not None
    }
    ho = {
        k: 1.0 - ARMS[old]["cells"][k]["ap"]
        for k in po
        if po[k] and ARMS[old]["cells"][k]["ap"] is not None
    }
    row["hears_not_acts_new"] = ci_mean(hn) if hn else None
    row["hears_not_acts_old"] = ci_mean(ho) if ho else None
    row["decomposition"] = decomposition(new, old)
    ln, lo = SYS[new]["median_latency_s"], SYS[old]["median_latency_s"]
    row["median_latency_s"] = {"new": ln, "old": lo}
    return row


GENERATIONS = [
    ("gemini38or", "gemini37or", "generation (file)"),
    ("geminilive", "gem25native", "generation (Live)"),
    ("gemini38live", "geminilive", "generation (Live)"),
    ("gemini38live", "gem25native", "generation (Live, two steps)"),
    ("mimo26flash", "mimo25", "generation (same size class)"),
    ("mimo26pro", "mimo25", "generation + size"),
    ("mimo26pro", "mimo26flash", "size (same generation)"),
    ("qwen3omni", "qwen25omni7b", "generation (open, local)"),
    ("qwen38omni", "qwen3omni", "generation (file; open->API)"),
    ("qwen38rtflash", "qwenrtflash", "generation (realtime)"),
    ("qwenaudio31rt", "qwen38rtflash", "sibling realtime lines, same month"),
    ("gemma412b", "gemma4e4b", "size / architecture (encoder-free 12B vs E4B)"),
    ("gptaudio", "gptaudiomini", "size (file)"),
    ("gptrt21", "gptrt21mini", "size (realtime)"),
    ("mimo26flash", "mimo26pro", "flash vs pro (same gen)"),
]
GEN = [compare(n, o, k) for n, o, k in GENERATIONS if n in SYS and o in SYS]

SERVING_PAIRS = [
    # (realtime, file, same_generation, description)
    ("gemini38live", "gemini38or", True, "Gemini 3.8 Live vs 3.8 Flash (file)"),
    ("qwen38rtflash", "qwen38omni", True, "Qwen3.8-Omni-Flash RT vs Qwen3.8-Omni file"),
    ("gptrt21", "gptaudio", False, "gpt-realtime-2.1 vs gpt-audio"),
    ("gptrt21mini", "gptaudiomini", False, "gpt-realtime-2.1-mini vs gpt-audio-mini"),
    ("geminilive", "gemini37or", False, "Gemini 3.1 Flash Live vs 3.7 Flash"),
    ("gem25native", "gemini37or", False, "Gemini 2.5 native Live vs 3.7 Flash"),
    ("qwenrtflash", "qwen3omni", False, "Qwen3.5-Omni-Flash RT vs Qwen3-Omni-30B"),
]
SERVE = [
    dict(compare(r, f, "realtime minus file"), same_generation=sg, description=d)
    for r, f, sg, d in SERVING_PAIRS
]


def meta_analysis(rows: list[dict], key: str) -> dict:
    """DerSimonian-Laird random effects over pairs (SE from bootstrap CI width).

    Pairs that share an arm (gemini37or twice) are not independent; treated as
    independent here, so the pooled CI is optimistic. The heterogeneity (I^2) is
    the quantity of interest.
    """
    ys, vs, labs = [], [], []
    for r in rows:
        e = r.get(key)
        if not isinstance(e, dict):
            continue
        se = (e["hi"] - e["lo"]) / (2 * 1.96)
        ys.append(e["mean"])
        vs.append(max(se, 1e-4) ** 2)
        labs.append(r["description"])
    y, v = np.array(ys), np.array(vs)
    w = 1 / v
    fe = (w * y).sum() / w.sum()
    Q = float((w * (y - fe) ** 2).sum())
    df = len(y) - 1
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    wr = 1 / (v + tau2)
    re = (wr * y).sum() / wr.sum()
    se_re = math.sqrt(1 / wr.sum())
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    # prediction interval (t approx with df-1 ~ use z for simplicity)
    pi = 1.96 * math.sqrt(tau2 + se_re**2)
    return {
        "k": len(y),
        "pooled_re": round(float(re), 4),
        "lo": round(float(re - 1.96 * se_re), 4),
        "hi": round(float(re + 1.96 * se_re), 4),
        "tau": round(math.sqrt(tau2), 4),
        "I2": round(I2, 3),
        "Q": round(Q, 2),
        "prediction_interval": [round(re - pi, 4), round(re + pi, 4)],
        "pairs": dict(zip(labs, [round(x, 4) for x in ys], strict=True)),
        "signs": {
            "positive_excl_0": int(
                sum(1 for r in rows if isinstance(r.get(key), dict) and r[key]["lo"] > 0)
            ),
            "negative_excl_0": int(
                sum(1 for r in rows if isinstance(r.get(key), dict) and r[key]["hi"] < 0)
            ),
        },
    }


META_SERVING = {
    k: meta_analysis(SERVE, k) for k in ("d_cue_credit", "d_probe_cue", "d_act_rate", "d_vs_floor")
}
META_SERVING_SAMEGEN = {
    k: meta_analysis([r for r in SERVE if r["same_generation"]], k)
    for k in ("d_cue_credit", "d_probe_cue", "d_act_rate")
}


# ----------------------------------------------------------------------------- pareto


def pareto_front(labels, xkey, ykey):
    pts = [
        (SYS[lab][xkey], SYS[lab][ykey]["mean"], lab)
        for lab in labels
        if SYS[lab].get(xkey) is not None and SYS[lab].get(ykey)
    ]
    front = []
    for x, y, lab in pts:
        if not any((x2 <= x and y2 >= y) and (x2 < x or y2 > y) for x2, y2, _ in pts):
            front.append(lab)
    return sorted(front, key=lambda lab: SYS[lab][xkey])


PARETO = {
    "cost_vs_listening": pareto_front(C, "usd_per_cell", "vs_floor"),
    "latency_vs_listening": pareto_front(C, "median_latency_s", "vs_floor"),
    "cost_vs_cue_credit": pareto_front([*C, CASCADE], "usd_per_cell", "cue_credit"),
    "latency_vs_cue_credit": pareto_front([*C, CASCADE], "median_latency_s", "cue_credit"),
    "note": "usd_per_cell is measured provider cost (or the documented estimate for Gemini Live); "
    "free/local arms are $0 by basis. Systems whose cost was not recorded (OpenAI/xAI/DashScope "
    "realtime, StepFun preview) are excluded from cost fronts, never treated as $0.",
}
# listening per dollar (paid, cost recorded, clears floor)
PER_DOLLAR = sorted(
    [
        {
            "label": lab,
            "name": SYS[lab]["name"],
            "vs_floor": SYS[lab]["vs_floor"]["mean"],
            "usd_per_1k_cells": round(SYS[lab]["usd_per_cell"] * 1000, 3),
            "listening_points_per_usd_per_1k": round(
                SYS[lab]["vs_floor"]["mean"] / (SYS[lab]["usd_per_cell"] * 1000), 3
            ),
        }
        for lab in C
        if SYS[lab].get("usd_per_cell") and SYS[lab]["vs_floor"] and SYS[lab]["vs_floor"]["lo"] > 0
    ],
    key=lambda r: -r["listening_points_per_usd_per_1k"],
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "method": {
            "freeze": "bank-freeze-2026-09-15",
            "engine": "gemini (primary)",
            "listening": "vs-floor: DiD of audio-minus-twin vs the words-only cascade on identical "
            "cue-bearing cells; twin-less arms: audio credit minus cascade audio credit",
            "within_system_ci": "item-clustered percentile bootstrap, 4000 resamples, "
            "seed 20260915",
            "across_system_ci": "joint bootstrap over systems (with replacement) and items "
            "(each system's "
            "item-bootstrap replicate), 2000 draws; permutation p over system labels",
            "power_note": "with n systems, the smallest Spearman |rho| detectable at 80% power is "
            "mde_rho_80pct (n=27: ~0.52); any |rho| below that which fails is NOT evidence "
            "of no effect",
            "reproduced_leaderboard_vs_floor": len(CHECK),
        },
        "systems": SYS,
        "properties": PROPS,
        "partial": PARTIAL,
        "within_vendor_release_slope": WITHIN,
        "groups": GROUPS,
        "generations": GEN,
        "serving_pairs": SERVE,
        "serving_meta": META_SERVING,
        "serving_meta_same_generation": META_SERVING_SAMEGEN,
        "pareto": PARETO,
        "listening_per_dollar": PER_DOLLAR,
    }
    (OUT / "models.json").write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    print("wrote", OUT / "models.json")


if __name__ == "__main__":
    main()
