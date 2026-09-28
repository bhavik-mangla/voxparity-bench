# ruff: noqa: B023  (bootstrap closures are consumed inside the loop that defines them)
"""Strengthening analyses for the paper's human-vs-frontier and null claims.

Six questions, all on existing frozen data (no model calls, no spend):

  1. Frontier-4 (and gemini-3.7-flash) vs players on PROTECTIVE calls: unsafe-execute
     and missed-duty differences and the unsafe-execute RATIO, identical cells,
     two-way (item x player) CIs; same on life-safety protective cells.
  2. "Average parity is a cancellation": best-system-minus-players on identical
     cue-bearing cells decomposed by axis (contribution = cell share x axis gap)
     and by harm class, two-way CIs.
  3. Severity-weighted view: tier-weighted selection credit (3 tier schemes) and
     harm risk under all 9 tier x outcome weightings, best system minus players.
  4. Cells where the players' majority is right and all / most of the frontier-4
     are wrong, vs the reverse; by axis and harm class.
  5. The 12 twin-bearing systems that do not clear the words-only cascade:
     TOST equivalence of their difference-in-differences at +-0.05 / +-0.10.
  6. Recommended sentences (written into the .md).

Conventions: freeze bank-freeze-2026-09-15, first-turn scoring (D118), Gemini-TTS
cells for every human-model contrast (the leaderboard's human row), identical
cells on both sides, per-cell player mean then cell mean. Two-way bootstrap for
anything with players (robust_common.two_way_boot: items and players resampled
independently, players reweight the human side only); item-clustered for
model-only contrasts. 4000 resamples, seed 20260915.

Run from the pinned bank worktree:

    cd $VXP_BANK && uv run --project $VXP_CODE --extra paper --with scipy \\
        python $VXP_CODE/scripts/insights/strengthen_analysis.py

Writes docs/insights/strengthen.{json,md}.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harm_analysis as ha
import robust_common as rc
import robust_human as rh
from harm_rubric import OUTCOME_WEIGHTS, TIER_WEIGHTS

N_BOOT = 4000
FRONTIER4 = ("gemini37or", "gemini38or", "qwen38omni", "mimo26pro")
BEST = "gemini37or"
CASCADE = "cascadeopen"
ENGINE = "gemini"
NO_CALL = "__none__"
MARGINS = (0.05, 0.10)


# --------------------------------------------------------------------------- two-way engine


class TwoWay:
    """Cells x (human answers, model per-cell values). ``run(f)`` bootstraps a
    statistic f(H, M, cw, idx) where H/M map a value name to per-cell arrays
    (H = player-weighted per-cell human mean), cw = per-cell item-draw weight and
    idx = the original cell indices of the cells that survive the draw."""

    def __init__(
        self,
        cells: list[tuple[str, str]],
        answers: list[tuple[int, str, dict[str, float]]],
        model: dict[str, np.ndarray],
    ) -> None:
        self.cells = cells
        self.nc = len(cells)
        self.rows = answers
        self.ci = np.array([a[0] for a in answers])
        names = sorted(answers[0][2]) if answers else []
        self.hv = {n: np.array([a[2][n] for a in answers], dtype=float) for n in names}
        self.model = model

    def _stat(self, f: Any) -> Any:
        ci, nc = self.ci, self.nc

        def stat(_rows: Any, w: np.ndarray, wi: np.ndarray) -> float | None:
            sw = np.bincount(ci, weights=w, minlength=nc)
            iw = np.zeros(nc)
            iw[ci] = wi
            ok = (sw > 0) & (iw > 0)
            if not ok.any():
                return None
            H = {
                n: np.bincount(ci, weights=w * y, minlength=nc)[ok] / sw[ok]
                for n, y in self.hv.items()
            }
            M = {n: v[ok] for n, v in self.model.items()}
            return f(H, M, iw[ok], np.nonzero(ok)[0])

        return stat

    def _draws(self, item_only: bool) -> list[tuple[np.ndarray, np.ndarray]]:
        """The (answer weight, item weight) draws of robust_common.two_way_boot,
        generated once with the identical RNG sequence and reused per statistic."""
        cache = self.__dict__.setdefault("_dcache", {})
        if item_only in cache:
            return cache[item_only]
        items = sorted({self.cells[r[0]][0] for r in self.rows})
        players = [] if item_only else sorted({r[1] for r in self.rows})
        ii = np.array([items.index(self.cells[r[0]][0]) for r in self.rows])
        pp = np.array([players.index(r[1]) for r in self.rows]) if players else None
        rng = np.random.default_rng(rc.SEED)
        out = []
        for _ in range(N_BOOT):
            ci = np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items))
            wi = ci[ii].astype(float)
            if players:
                cp = np.bincount(
                    rng.integers(0, len(players), len(players)), minlength=len(players)
                )
                w = wi * cp[pp]
            else:
                w = wi
            out.append((w, wi))
        cache[item_only] = out
        return out

    def run(self, f: Any, *, item_only: bool = False, shift: float = 0.0) -> dict[str, Any]:
        """Same numbers as robust_common.two_way_boot (verified against
        players-band.json), plus the two-sided bootstrap p for mean - shift != 0
        (floored at 1/N_BOOT)."""
        stat = self._stat(f)
        one = np.ones(len(self.rows))
        point = stat(self.rows, one, one)
        vals = []
        for w, wi in self._draws(item_only):
            v = stat(self.rows, w, wi)
            if v is not None and not np.isnan(v):
                vals.append(v)
        out: dict[str, Any] = {
            "mean": None if point is None else round(float(point), 4),
            "items": len({c[0] for c in self.cells}),
            "players": 0 if item_only else len({r[1] for r in self.rows}),
            "n": len(self.rows),
        }
        if vals:
            v = np.array(vals)
            b = v - shift
            p = min(1.0, 2.0 * min(float(np.mean(b <= 0)), float(np.mean(b >= 0))))
            out.update(
                {
                    "lo": round(float(np.quantile(v, 0.025)), 4),
                    "hi": round(float(np.quantile(v, 0.975)), 4),
                    "lo90": round(float(np.quantile(v, 0.05)), 4),
                    "hi90": round(float(np.quantile(v, 0.95)), 4),
                    "p": max(p, 1.0 / N_BOOT),
                    "n_boot_ok": len(vals),
                }
            )
        return out


def wm(x: np.ndarray, w: np.ndarray) -> float | None:
    s = float(w.sum())
    return None if s == 0 else float((x * w).sum() / s)


def diff(a: str, b: str | None = None) -> Any:
    """model mean of a minus human mean of b (default: same name)."""
    b = b or a
    return lambda H, M, cw, idx: wm(M[a] - H[b], cw)


def ratio(a: str) -> Any:
    def f(H: Any, M: Any, cw: Any, idx: Any) -> float | None:
        h = wm(H[a], cw)
        m = wm(M[a], cw)
        return None if not h else m / h

    return f


def level(side: str, a: str) -> Any:
    return lambda H, M, cw, idx: wm((M if side == "M" else H)[a], cw)


def sig(e: dict[str, Any]) -> str:
    if e.get("lo") is None:
        return ""
    return "↑" if e["lo"] > 0 else ("↓" if e["hi"] < 0 else "")


def f2(e: dict[str, Any] | None, nd: int = 2, signed: bool = True) -> str:
    return rc.fmt(e, signed=signed, nd=nd)


# --------------------------------------------------------------------------- data


def load() -> dict[str, Any]:
    d = rh.load_human()
    taxonomy_axis = rh._axis_fn()
    items = d.items
    _items_h, rubric, freeze = ha.load_bank(rc.BANK)
    held = set(freeze.get("held_items", {}))

    def axis(k: tuple[str, str]) -> str | None:
        return taxonomy_axis(items[k[0]], k[1].rsplit("@", 1)[0])

    def harm_cell(k: tuple[str, str], tool: str | None) -> dict[str, Any]:
        iid, vid = k[0], k[1].rsplit("@", 1)[0]
        tools = [tool] if tool and tool != NO_CALL else []
        return ha.cell_record(items[iid], rubric[iid], vid, tools)

    by_cell: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for t in d.humans:
        if t.engine == ENGINE:
            by_cell[t.key].append(t)
    return {
        "d": d,
        "items": items,
        "rubric": rubric,
        "held": held,
        "axis": axis,
        "harm_cell": harm_cell,
        "by_cell": by_cell,
    }


def build_tw(
    L: dict[str, Any],
    keys: list[tuple[str, str]],
    arms: tuple[str, ...],
    hvals: Any,
    mvals: Any,
    *,
    require_all: bool = True,
) -> TwoWay | None:
    """hvals(trial) -> dict; mvals(label, key) -> dict (averaged over ``arms``).
    require_all: keep only cells every arm holds (identical cells); otherwise a
    pooled group averages the arms that hold the cell."""
    d = L["d"]
    have = lambda k: [a for a in arms if k in d.models.get(a, {})]  # noqa: E731
    keys = [k for k in keys if (len(have(k)) == len(arms) if require_all else have(k))]
    if not keys:
        return None
    answers = []
    for i, k in enumerate(keys):
        for t in L["by_cell"][k]:
            answers.append((i, t.player, hvals(t, k)))
    names = sorted(answers[0][2])
    model = {
        n: np.array([np.mean([mvals(a, k)[n] for a in have(k)]) for k in keys], dtype=float)
        for n in names
    }
    tw = TwoWay(keys, answers, model)
    tw.meta = {  # type: ignore[attr-defined]
        "cells": len(keys),
        "items": len({k[0] for k in keys}),
        "answers": len(answers),
        "players": len({a[1] for a in answers}),
    }
    return tw


# --------------------------------------------------------------------------- 1 protective


def q1(L: dict[str, Any]) -> dict[str, Any]:
    d, hc = L["d"], L["harm_cell"]
    keys = sorted(
        k for k in L["by_cell"] if k[0] not in L["held"] and hc(k, None)["role"] == "PROTECTIVE"
    )
    classes = ("UNSAFE-EXECUTE", "MISSED-DUTY", "SAFE-DEFER", "CORRECT")

    def hv(t: Any, k: Any) -> dict[str, float]:
        c = hc(k, t.tool)
        return {cl: float(c["cls"] == cl) for cl in classes}

    def mv(a: str, k: Any) -> dict[str, float]:
        c = hc(k, d.models[a][k].tool)
        return {cl: float(c["cls"] == cl) for cl in classes}

    out: dict[str, Any] = {}
    for group, arms in (("frontier4", FRONTIER4), ("gemini37or", (BEST,)), ("cascade", (CASCADE,))):
        block = {}
        for sub, keep in (
            ("protective", lambda k: True),
            ("life_safety_protective", lambda k: hc(k, None)["tier"] == "life-safety"),
        ):
            tw = build_tw(L, [k for k in keys if keep(k)], arms, hv, mv)
            if tw is None:
                continue
            r: dict[str, Any] = {**tw.meta}  # type: ignore[attr-defined]
            for cl, nm in (
                ("UNSAFE-EXECUTE", "unsafe"),
                ("MISSED-DUTY", "missed_duty"),
                ("SAFE-DEFER", "safe_defer"),
                ("CORRECT", "correct"),
            ):
                r[f"{nm}_model"] = tw.run(level("M", cl))
                r[f"{nm}_players"] = tw.run(level("H", cl))
                r[f"{nm}_diff"] = tw.run(diff(cl))
                r[f"{nm}_diff_item_only"] = tw.run(diff(cl), item_only=True)
            r["unsafe_ratio"] = tw.run(ratio("UNSAFE-EXECUTE"))
            r["unsafe_ratio_item_only"] = tw.run(ratio("UNSAFE-EXECUTE"), item_only=True)
            # errors that went toward the words: unsafe + missed duty vs players
            r["unsafe_plus_missed_diff"] = tw.run(
                lambda H, M, cw, idx: wm(
                    (M["UNSAFE-EXECUTE"] + M["MISSED-DUTY"])
                    - (H["UNSAFE-EXECUTE"] + H["MISSED-DUTY"]),
                    cw,
                )
            )
            block[sub] = r
        out[group] = block
    return out


# --------------------------------------------------------------------------- 2 decomposition


def q2(L: dict[str, Any]) -> dict[str, Any]:
    d, axis, hc = L["d"], L["axis"], L["harm_cell"]
    keys = sorted(k for k in L["by_cell"] if axis(k) is not None)

    def hv(t: Any, k: Any) -> dict[str, float]:
        return {"credit": t.credit}

    def mv(a: str, k: Any) -> dict[str, float]:
        return {"credit": d.models[a][k].credit}

    out: dict[str, Any] = {}
    for group, arms in (("gemini37or", (BEST,)), ("frontier4", FRONTIER4)):
        tw = build_tw(L, keys, arms, hv, mv)
        assert tw is not None
        cells = tw.cells
        ax = np.array([axis(k) for k in cells])
        tier = np.array(
            [
                "life-safety" if hc(k, None)["tier"] == "life-safety" else "other tiers"
                for k in cells
            ]
        )
        role = np.array([hc(k, None)["role"] if k[0] not in L["held"] else "HELD" for k in cells])
        r: dict[str, Any] = {**tw.meta, "overall": tw.run(diff("credit"))}  # type: ignore[attr-defined]
        r["overall_item_only"] = tw.run(diff("credit"), item_only=True)

        def decompose(lab: np.ndarray) -> dict[str, Any]:
            res = {}
            for g in sorted(set(lab)):
                gm = lab == g

                def contrib(H: Any, M: Any, cw: Any, idx: Any, gm: Any = gm) -> float | None:
                    m = gm[idx]
                    s = float(cw.sum())
                    return (
                        None if s == 0 else float(((M["credit"] - H["credit"]) * cw * m).sum() / s)
                    )

                def gap(H: Any, M: Any, cw: Any, idx: Any, gm: Any = gm) -> float | None:
                    m = gm[idx]
                    return wm((M["credit"] - H["credit"])[m], cw[m]) if m.any() else None

                def share(H: Any, M: Any, cw: Any, idx: Any, gm: Any = gm) -> float | None:
                    return wm(gm[idx].astype(float), cw)

                res[g] = {
                    "cells": int(gm.sum()),
                    "items": len({cells[i][0] for i in np.nonzero(gm)[0]}),
                    "answers": int(sum(1 for a in tw.rows if gm[a[0]])),
                    "share": round(float(gm.mean()), 4),
                    "gap": tw.run(gap),
                    "contribution": tw.run(contrib),
                    "model_credit": tw.run(
                        lambda H, M, cw, idx, gm=gm: (
                            wm(M["credit"][gm[idx]], cw[gm[idx]]) if gm[idx].any() else None
                        )
                    ),
                    "players_credit": tw.run(
                        lambda H, M, cw, idx, gm=gm: (
                            wm(H["credit"][gm[idx]], cw[gm[idx]]) if gm[idx].any() else None
                        )
                    ),
                }
            # leads/trails split on the point estimate, then gross positive / negative mass
            lead = [g for g, v in res.items() if (v["gap"]["mean"] or 0) > 0]
            trail = [g for g, v in res.items() if (v["gap"]["mean"] or 0) < 0]
            lm = np.isin(lab, lead)
            tm = np.isin(lab, trail)

            def mass(mask: np.ndarray) -> Any:
                def f(H: Any, M: Any, cw: Any, idx: Any) -> float | None:
                    s = float(cw.sum())
                    m = mask[idx]
                    return (
                        None if s == 0 else float(((M["credit"] - H["credit"]) * cw * m).sum() / s)
                    )

                return f

            lead_e, trail_e = tw.run(mass(lm)), tw.run(mass(tm))

            def gross(H: Any, M: Any, cw: Any, idx: Any) -> float | None:
                s = float(cw.sum())
                g = M["credit"] - H["credit"]
                return (
                    None
                    if s == 0
                    else float(((g * cw * lm[idx]).sum() - (g * cw * tm[idx]).sum()) / s)
                )

            def cancel_ratio(H: Any, M: Any, cw: Any, idx: Any) -> float | None:
                g = (M["credit"] - H["credit"]) * cw
                p = float((g * lm[idx]).sum())
                n = float(-(g * tm[idx]).sum())
                return None if max(p, n) <= 0 else min(p, n) / max(p, n)

            return {
                "groups": res,
                "leads": lead,
                "trails": trail,
                "lead_mass": lead_e,
                "trail_mass": trail_e,
                "gross_abs": tw.run(gross),
                "cancellation_ratio": tw.run(cancel_ratio),
            }

        r["by_axis"] = decompose(ax)
        r["by_harm_tier"] = decompose(tier)
        r["by_variant_role"] = decompose(role)

        # PRE-SPECIFIED heterogeneity contrasts (not selected on the point estimate):
        # gap on group A minus gap on group B, same draws.
        def gapdiff(ma: np.ndarray, mb: np.ndarray) -> Any:
            def f(H: Any, M: Any, cw: Any, idx: Any) -> float | None:
                a, b_ = ma[idx], mb[idx]
                if not a.any() or not b_.any():
                    return None
                g = M["credit"] - H["credit"]
                return wm(g[a], cw[a]) - wm(g[b_], cw[b_])  # type: ignore[operator]

            return f

        emo = np.isin(ax, ["delivery emotion", "sarcasm"])
        r["prespecified"] = {
            "emotional_minus_nonemotional_axes": tw.run(gapdiff(emo, ~emo)),
            "delivery_emotion_minus_rest": tw.run(
                gapdiff(ax == "delivery emotion", ax != "delivery emotion")
            ),
            "clean_minus_protective": tw.run(gapdiff(role == "CLEAN", role == "PROTECTIVE")),
            "life_safety_minus_other": tw.run(
                gapdiff(tier == "life-safety", tier != "life-safety")
            ),
            "gap_emotional_axes": tw.run(
                lambda H, M, cw, idx: wm((M["credit"] - H["credit"])[emo[idx]], cw[emo[idx]])
            ),
            "gap_nonemotional_axes": tw.run(
                lambda H, M, cw, idx: wm((M["credit"] - H["credit"])[~emo[idx]], cw[~emo[idx]])
            ),
            "emotional_cells": int(emo.sum()),
        }
        out[group] = r
    return out


# --------------------------------------------------------------------------- 3 severity


def q3(L: dict[str, Any]) -> dict[str, Any]:
    d, axis, hc = L["d"], L["axis"], L["harm_cell"]
    base = [k for k in L["by_cell"] if k[0] not in L["held"]]
    out: dict[str, Any] = {"tier_weights": TIER_WEIGHTS, "outcome_weights": OUTCOME_WEIGHTS}
    for arm in (BEST, "frontier4"):
        arms = FRONTIER4 if arm == "frontier4" else (BEST,)
        block: dict[str, Any] = {}
        for sub, keys in (
            ("cue", sorted(k for k in base if axis(k) is not None)),
            ("all", sorted(base)),
        ):
            # severity-weighted credit: per-cell weight = tier weight
            def hv(t: Any, k: Any) -> dict[str, float]:
                c = hc(k, t.tool)
                v = {"credit": t.credit}
                for ts in TIER_WEIGHTS:
                    for os_ in OUTCOME_WEIGHTS:
                        mw = max(TIER_WEIGHTS[ts].values())
                        v[f"risk|{ts}|{os_}"] = 100 * ha.harm(c, ts, os_) / mw
                return v

            def mv(a: str, k: Any) -> dict[str, float]:
                m = d.models[a][k]
                c = hc(k, m.tool)
                v = {"credit": m.credit}
                for ts in TIER_WEIGHTS:
                    for os_ in OUTCOME_WEIGHTS:
                        mw = max(TIER_WEIGHTS[ts].values())
                        v[f"risk|{ts}|{os_}"] = 100 * ha.harm(c, ts, os_) / mw
                return v

            tw = build_tw(L, keys, arms, hv, mv)
            assert tw is not None
            tiers = [hc(k, None)["tier"] for k in tw.cells]
            r: dict[str, Any] = {**tw.meta, "unweighted_credit_diff": tw.run(diff("credit"))}  # type: ignore[attr-defined]
            sw = {}
            for ts, W in TIER_WEIGHTS.items():
                wt = np.array([W[t] for t in tiers], dtype=float)

                def swc(H: Any, M: Any, cw: Any, idx: Any, wt: Any = wt) -> float | None:
                    return wm(M["credit"] - H["credit"], cw * wt[idx])

                sw[ts] = {
                    "model": tw.run(lambda H, M, cw, idx, wt=wt: wm(M["credit"], cw * wt[idx])),
                    "players": tw.run(lambda H, M, cw, idx, wt=wt: wm(H["credit"], cw * wt[idx])),
                    "diff": tw.run(swc),
                }
            r["severity_weighted_credit"] = sw
            risk = {}
            for ts in TIER_WEIGHTS:
                for os_ in OUTCOME_WEIGHTS:
                    n = f"risk|{ts}|{os_}"
                    risk[f"{ts}|{os_}"] = {
                        "model": tw.run(level("M", n)),
                        "players": tw.run(level("H", n)),
                        "model_minus_players": tw.run(diff(n)),
                    }
            r["risk"] = risk
            r["risk_sign_model_worse"] = sum(
                (v["model_minus_players"]["mean"] or 0) > 0 for v in risk.values()
            )
            r["risk_sig_model_worse"] = sum(
                (v["model_minus_players"].get("lo") or -1) > 0 for v in risk.values()
            )
            r["swc_sign_model_better"] = sum((v["diff"]["mean"] or 0) > 0 for v in sw.values())
            block[sub] = r
        out[arm] = block
    return out


# --------------------------------------------------------------------------- 4 disagreements


def q4(L: dict[str, Any]) -> dict[str, Any]:
    d, axis, hc = L["d"], L["axis"], L["harm_cell"]
    keys = sorted(k for k in L["by_cell"] if axis(k) is not None)

    def hv(t: Any, k: Any) -> dict[str, float]:
        return {"hit": float(t.hit)}

    def mv(a: str, k: Any) -> dict[str, float]:
        return {"hit": float(d.models[a][k].hit)}

    tw = build_tw(L, keys, FRONTIER4, hv, mv)
    assert tw is not None
    cells = tw.cells
    # frontier: number of the four that are right, per cell (static)
    nright = np.array([sum(d.models[a][k].hit for a in FRONTIER4) for k in cells])
    ax = np.array([axis(k) for k in cells])
    tier = np.array(
        ["life-safety" if hc(k, None)["tier"] == "life-safety" else "other tiers" for k in cells]
    )
    role = np.array([hc(k, None)["role"] if k[0] not in L["held"] else "HELD" for k in cells])

    def frac(pred: Any) -> Any:
        def f(H: Any, M: Any, cw: Any, idx: Any) -> float | None:
            return wm(pred(H["hit"], nright[idx]).astype(float), cw)

        return f

    defs = {
        "players_right_all4_wrong": lambda h, n: (h > 0.5 + 1e-9) & (n == 0),
        "players_right_3plus_wrong": lambda h, n: (h > 0.5 + 1e-9) & (n <= 1),
        "players_wrong_all4_right": lambda h, n: (h < 0.5 - 1e-9) & (n == 4),
        "players_wrong_3plus_right": lambda h, n: (h < 0.5 - 1e-9) & (n >= 3),
        "both_right_majority": lambda h, n: (h > 0.5 + 1e-9) & (n >= 3),
        "both_wrong_majority": lambda h, n: (h < 0.5 - 1e-9) & (n <= 1),
        "players_tie": lambda h, n: np.abs(h - 0.5) <= 1e-9,
    }
    out: dict[str, Any] = {**tw.meta, "fractions": {k: tw.run(frac(p)) for k, p in defs.items()}}  # type: ignore[attr-defined]
    # majority status is discontinuous under player resampling (a 1-1 split flips),
    # so the item-only interval (players fixed) is reported beside the two-way one
    out["fractions_item_only"] = {k: tw.run(frac(p), item_only=True) for k, p in defs.items()}
    for a, b in (
        ("players_right_all4_wrong", "players_wrong_all4_right"),
        ("players_right_3plus_wrong", "players_wrong_3plus_right"),
    ):
        pa_, pb_ = defs[a], defs[b]

        def mr(H: Any, M: Any, cw: Any, idx: Any, pa_: Any = pa_, pb_: Any = pb_) -> Any:
            return wm(
                pa_(H["hit"], nright[idx]).astype(float) - pb_(H["hit"], nright[idx]).astype(float),
                cw,
            )

        out["fractions"][f"{a}_minus_reverse"] = tw.run(mr)
        out["fractions_item_only"][f"{a}_minus_reverse"] = tw.run(mr, item_only=True)
    # point-estimate characterisation (counts of cells by class)
    hmaj = np.array([np.mean([t.hit for t in L["by_cell"][k]]) for k in cells])
    chars: dict[str, Any] = {}
    for name in (
        "players_right_all4_wrong",
        "players_right_3plus_wrong",
        "players_wrong_all4_right",
        "players_wrong_3plus_right",
    ):
        m = defs[name](hmaj, nright)
        chars[name] = {
            "cells": int(m.sum()),
            "by_axis": dict(Counter(ax[m]).most_common()),
            "by_tier": dict(Counter(tier[m]).most_common()),
            "by_role": dict(Counter(role[m]).most_common()),
            "list": [
                {
                    "cell": f"{cells[i][0]}/{cells[i][1]}",
                    "axis": ax[i],
                    "tier": hc(cells[i], None)["tier"],
                    "role": role[i],
                    "players_hit": round(float(hmaj[i]), 2),
                    "players_n": len(L["by_cell"][cells[i]]),
                    "frontier_right": int(nright[i]),
                }
                for i in np.nonzero(m)[0]
            ],
        }
    out["characterisation"] = chars
    out["denominators"] = {
        "by_axis": dict(Counter(ax).most_common()),
        "by_tier": dict(Counter(tier).most_common()),
        "by_role": dict(Counter(role).most_common()),
    }
    return out


# --------------------------------------------------------------------------- 5 equivalence


def q5(b: rc.Bank) -> dict[str, Any]:
    import robust_floor as rf

    F = b.cascade
    rows = []
    for A in b.contestants:
        if not A.twin:
            continue
        est, _ = rf.system_vs_floor(b, A, F)
        own = rc.cells_test({k: A.audio[k] - A.twin[k] for k in b.cue(A.audio) if k in A.twin})
        rows.append({"label": A.label, "name": rf._display(A.label), "did": est, "own_delta": own})
    adj = rc.holm({r["label"]: r["did"]["p"] for r in rows})
    for r in rows:
        r["holm_p"] = adj[r["label"]]
        r["clears"] = bool(r["holm_p"] < 0.05 and r["did"]["mean"] > 0)
        r["tost"] = {str(m): rc.tost(r["did"], m) for m in MARGINS}
    floor = rc.cells_test({k: F.audio[k] - F.twin[k] for k in b.cue(F.audio) if k in F.twin})
    non = [r for r in rows if not r["clears"]]
    rows.sort(key=lambda r: -r["did"]["mean"])
    return {
        "n_twin_bearing": len(rows),
        "n_clear": sum(r["clears"] for r in rows),
        "n_non_clearing": len(non),
        "cascade_floor_cue": floor,
        "equivalent": {
            str(m): sorted(r["name"] for r in non if r["tost"][str(m)]["equivalent"])
            for m in MARGINS
        },
        "non_clearing_sig_above_zero_unadjusted": sorted(
            r["name"] for r in non if r["did"]["lo"] > 0
        ),
        "non_clearing_sig_below_zero_unadjusted": sorted(
            r["name"] for r in non if r["did"]["hi"] < 0
        ),
        "rows": rows,
    }


# --------------------------------------------------------------------------- main


def main() -> None:
    L = load()
    out: dict[str, Any] = {
        "freeze": "bank-freeze-2026-09-15",
        "scoring": "first-turn (D118)",
        "engine": "Gemini-TTS cells for every human-model contrast",
        "bootstrap": {
            "resamples": N_BOOT,
            "seed": rc.SEED,
            "humans": "two-way item x player (robust_common.two_way_boot)",
            "models_only": "item-clustered (robust_common.cells_test)",
        },
        "frontier4": list(FRONTIER4),
        "best": BEST,
    }
    # consistency check against players-band.json (same statistic, same draws)
    pb = json.loads((rc.OUT / "players-band.json").read_text())
    out["q2"] = q2(L)
    ref = pb["model_minus_players_gemini_cue"][BEST]["two_way"]
    got = out["q2"][BEST]["overall"]
    out["check_players_band"] = {"reference": ref, "reproduced": got}
    assert abs(ref["mean"] - got["mean"]) < 1e-3 and abs(ref["lo"] - got["lo"]) < 1e-3, (ref, got)
    out["q1"] = q1(L)
    out["q3"] = q3(L)
    out["q4"] = q4(L)
    b = rc.Bank()
    out["q5"] = q5(b)
    import strengthen_audit as sa

    out["audit"] = sa.run(L, b, out)
    path = rc.OUT / "strengthen.json"
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
