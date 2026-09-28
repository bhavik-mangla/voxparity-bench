"""REANALYSIS D, tasks 5 (human-vs-model equivalence) and 6 (player-clustered CIs).

Human trials come from the lens-6 loader (insights/human:human_common.py, read from
the git ref): 27 players, 638 scored action answers, each paired with its probe.
Model cells are the eligible contestants' selection credit on the identical
(item, variant@engine) cells humans answered.

Every human number is re-estimated with a TWO-WAY (item x player) cluster
bootstrap (robust_common.two_way_boot): items and players are resampled
independently; in a human-vs-model contrast items are shared by both sides and
players only reweight the human side. The item-only CI is printed beside it.
Equivalence (TOST, alpha .05) = the 90% CI inside +/- margin.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np
import robust_common as rc

MARGINS = (0.05, 0.10)
BEST = ("gemini37or", "gemini38or", "qwen38omni", "mimo26pro")


def load_human() -> Any:
    os.environ.setdefault("BANK", str(rc.BANK))
    os.environ.setdefault("MAIN", str(rc.MAIN))
    os.environ.setdefault("VX_INSIGHT_CACHE", str(rc.CACHE.with_name(rc.CACHE.stem + "_human.pkl")))
    hc = rc.import_from_ref(rc.REF_HUMAN, "scripts/insights/human_common.py", "human_common")
    return hc.load()


def _axis_fn() -> Any:
    from voxparity.harness.paper_analyses import taxonomy_axis

    return taxonomy_axis


def run(b: rc.Bank) -> dict[str, Any]:
    d = load_human()
    taxonomy_axis = _axis_fn()
    items = d.items

    def axis(key: tuple[str, str]) -> str | None:
        return taxonomy_axis(items[key[0]], key[1].rsplit("@", 1)[0])

    H = d.humans
    contestants = sorted(k for k, r in d.roles.items() if r == "contestant")
    probe_arms = [
        a for a in contestants if any(m.probe_ok is not None for m in d.models[a].values())
    ]
    out: dict[str, Any] = {
        "players": len({t.player for t in H}),
        "answers": len(H),
        "top_player_share": round(
            max(np.bincount(np.unique([t.player for t in H], return_inverse=True)[1])) / len(H), 3
        ),
    }

    # ------------------------------------------------ rows: (item, player|None, heard, credit, who)
    cue_trials = [t for t in H if axis(t.key) is not None and t.probe_ok is not None]
    cue_cells = sorted({t.key for t in cue_trials})
    hrows = [(t.item, t.player, bool(t.probe_ok), t.credit, "H") for t in cue_trials]
    mrows = []
    for a in probe_arms:
        for k in cue_cells:
            m = d.models[a].get(k)
            if m is not None and m.probe_ok is not None:
                mrows.append((k[0], None, bool(m.probe_ok), m.credit, "M"))

    def arr(rows: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        heard = np.array([r[2] for r in rows], dtype=bool)
        y = np.array([r[3] for r in rows], dtype=float)
        who = np.array([r[4] == "H" for r in rows], dtype=bool)
        return heard, y, who

    rows = hrows + mrows
    heard, y, isH = arr(rows)

    def p_heard(mask: np.ndarray, w: np.ndarray) -> float | None:
        return rc.wmean(y[mask & heard], w[mask & heard])

    def bound(mask: np.ndarray, w: np.ndarray) -> float | None:
        ph = rc.wmean(heard[mask].astype(float), w[mask])
        if not ph:
            return None
        return ((rc.wmean(y[mask], w[mask]) or 0) - (1 - ph)) / ph

    stats = {
        "human_act_given_heard": lambda w: p_heard(isH, w),
        "human_lower_bound": lambda w: bound(isH, w),
        "models_pooled_act_given_heard": lambda w: p_heard(~isH, w),
        "models_minus_human_act_given_heard": lambda w: _sub(p_heard(~isH, w), p_heard(isH, w)),
        "models_minus_human_bound": lambda w: _sub(p_heard(~isH, w), bound(isH, w)),
        "human_heard_zero_credit": lambda w: rc.wmean(
            (y[isH & heard] == 0).astype(float), w[isH & heard]
        ),
    }
    hear_act = {}
    for name, f in stats.items():
        hear_act[name] = rc.cluster_suite(
            rows, lambda r: r[0], lambda r: r[1], lambda _rs, w, _wi, f=f: f(w)
        )
    out["act_given_heard"] = hear_act

    # per-arm P(act|heard) minus human (gemini-3.7-flash: 'level with humans')
    per_arm = {}
    for a in ("gemini37or", "gemini38or", "qwen38omni", "mimo26pro"):
        ar = [
            (k[0], None, bool(m.probe_ok), m.credit, "M")
            for k in cue_cells
            if (m := d.models[a].get(k)) is not None and m.probe_ok is not None
        ]
        rs = hrows + ar
        hh = np.array([r[2] for r in rs], dtype=bool)
        yy = np.array([r[3] for r in rs], dtype=float)
        ih = np.array([r[4] == "H" for r in rs], dtype=bool)

        def f(w: np.ndarray, hh: np.ndarray = hh, yy: np.ndarray = yy, ih: np.ndarray = ih) -> Any:
            return _sub(rc.wmean(yy[~ih & hh], w[~ih & hh]), rc.wmean(yy[ih & hh], w[ih & hh]))

        suite = rc.cluster_suite(rs, lambda r: r[0], lambda r: r[1], lambda _r, w, _wi, f=f: f(w))
        per_arm[a] = {
            **suite,
            "tost": {m: rc.tost(suite["two_way"], m) for m in MARGINS},
            "tost_item_only": {m: rc.tost(suite["item_only"], m) for m in MARGINS},
        }
    out["act_given_heard_per_arm_minus_human"] = per_arm

    # ------------------------------------------------ same-cell credit: model - human
    # human side = per-cell mean over that cell's answers (player-weighted in the bootstrap);
    # model side = selection credit; cells = those humans answered, identical for both.
    hc_by: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for t in H:
        hc_by[t.key].append(t)
    same: dict[str, Any] = {}
    for a in (*BEST, "cascadeopen"):
        if a not in d.models:
            continue
        block = {}
        for sub, keep in (
            ("all", lambda k: True),
            ("cue", lambda k: axis(k) is not None),
            ("neutral", lambda k: axis(k) is None),
        ):
            keys = [k for k in hc_by if keep(k) and k in d.models[a]]
            # one row per human answer, carrying the cell id and the model's credit
            rs = []
            for k in keys:
                for t in hc_by[k]:
                    rs.append((k, t.player, t.credit, d.models[a][k].credit))
            cells = sorted({r[0] for r in rs})
            cix = {c: i for i, c in enumerate(cells)}
            ci = np.array([cix[r[0]] for r in rs])
            hy = np.array([r[2] for r in rs])
            my = np.array([r[3] for r in rs])

            def stat(
                w: np.ndarray,
                wi: np.ndarray,
                ci: np.ndarray = ci,
                hy: np.ndarray = hy,
                my: np.ndarray = my,
                nc: int = len(cells),
            ) -> Any:
                # per-cell human mean (player-weighted answers), then the mean of
                # (model - human) over cells weighted by item draws; a cell whose
                # players all drop out of the draw leaves the contrast.
                sw = np.bincount(ci, weights=w, minlength=nc)
                sh = np.bincount(ci, weights=w * hy, minlength=nc)
                iw = np.zeros(nc)
                iw[ci] = wi  # constant within a cell
                mcell = np.zeros(nc)
                mcell[ci] = my
                ok = (sw > 0) & (iw > 0)
                if not ok.any():
                    return None
                hm = sh[ok] / sw[ok]
                return float(np.sum(iw[ok] * (mcell[ok] - hm)) / np.sum(iw[ok]))

            suite = rc.cluster_suite(
                rs, lambda r: r[0][0], lambda r: r[1], lambda _r, w, wi, stat=stat: stat(w, wi)
            )
            block[sub] = {
                "cells": len(cells),
                **suite,
                "tost": {m: rc.tost(suite["two_way"], m) for m in MARGINS},
                "tost_item_only": {m: rc.tost(suite["item_only"], m) for m in MARGINS},
            }
        same[a] = block
    out["same_cell_model_minus_human"] = same

    # human level on cue-bearing cells (answer-weighted), two-way
    cue_ans = [t for t in H if axis(t.key) is not None]
    rows_c = [(t.item, t.player, t.credit) for t in cue_ans]
    yc = np.array([r[2] for r in rows_c])
    out["human_cue_credit_individual"] = rc.cluster_suite(
        rows_c, lambda r: r[0], lambda r: r[1], lambda _r, w, _wi: rc.wmean(yc, w)
    )
    return out


def _sub(a: float | None, c: float | None) -> float | None:
    return None if a is None or c is None else a - c
