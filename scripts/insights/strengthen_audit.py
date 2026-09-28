# ruff: noqa: B023  (bootstrap closures are consumed inside the loop that defines them)
"""Claims-audit analyses A1-A7 (paper v3 CLAIMS-AUDIT.md §4), run from
strengthen_analysis.main() and written into docs/insights/strengthen.json under
``audit``. Same conventions: freeze bank-freeze-2026-09-15, first-turn scoring,
Gemini-TTS cells, identical cells per contrast, two-way (item x player)
bootstrap wherever players enter (item-only beside it), item-clustered
otherwise; 4000 resamples, seed 20260915; 95% and 90% intervals. No spend.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import robust_common as rc
import strengthen_analysis as S

FRONT = S.FRONTIER4


def _both(tw: S.TwoWay, f: Any) -> dict[str, Any]:
    return {"two_way": tw.run(f), "item_only": tw.run(f, item_only=True)}


def _holm_count(ests: dict[str, dict[str, Any]], direction: int) -> dict[str, Any]:
    """Systems significant in ``direction`` (+1 above, -1 below), raw and Holm."""
    raw = sorted(k for k, e in ests.items() if (e["lo"] > 0 if direction > 0 else e["hi"] < 0))
    adj = rc.holm({k: e["p"] for k, e in ests.items()})
    holm = sorted(
        k
        for k, e in ests.items()
        if adj[k] < 0.05 and (e["mean"] > 0 if direction > 0 else e["mean"] < 0)
    )
    return {"raw": raw, "n_raw": len(raw), "holm": holm, "n_holm": len(holm), "holm_p": adj}


def _contestants(L: dict[str, Any]) -> list[str]:
    return sorted(k for k, r in L["d"].roles.items() if r == "contestant")


# --------------------------------------------------------------------------- A1


def a1(L: dict[str, Any]) -> dict[str, Any]:
    d, hc = L["d"], L["harm_cell"]
    base = [k for k in L["by_cell"] if k[0] not in L["held"]]
    contest = _contestants(L)

    def cls_vals(cl: str) -> tuple[Any, Any]:
        def hv(t: Any, k: Any) -> dict[str, float]:
            return {"y": float(hc(k, t.tool)["cls"] == cl)}

        def mv(a: str, k: Any) -> dict[str, float]:
            return {"y": float(hc(k, d.models[a][k].tool)["cls"] == cl)}

        return hv, mv

    diff = S.diff("y")
    specs = {
        "unsafe_protective": ("UNSAFE-EXECUTE", lambda c: c["role"] == "PROTECTIVE"),
        "unsafe_life_safety_protective": (
            "UNSAFE-EXECUTE",
            lambda c: c["role"] == "PROTECTIVE" and c["tier"] == "life-safety",
        ),
        "over_trigger_clean": ("OVER-TRIGGER", lambda c: c["role"] == "CLEAN"),
    }
    out: dict[str, Any] = {}
    for name, (cl, keep) in specs.items():
        keys = sorted(k for k in base if keep(hc(k, None)))
        hv, mv = cls_vals(cl)
        block: dict[str, Any] = {}
        groups: dict[str, tuple[tuple[str, ...], bool]] = {
            "frontier4": (FRONT, True),
            **{a: ((a,), True) for a in FRONT},
            "pooled28": (tuple(contest), False),
        }
        for g, (arms, req) in groups.items():
            tw = S.build_tw(L, keys, arms, hv, mv, require_all=req)
            if tw is None:
                continue
            block[g] = {
                **tw.meta,  # type: ignore[attr-defined]
                "model": tw.run(S.level("M", "y")),
                "players_rate": tw.run(S.level("H", "y")),
                "diff": _both(tw, diff),
                "ratio": _both(tw, S.ratio("y")),
            }
        # every system alone vs players, Holm across the 28
        per: dict[str, Any] = {}
        for a in contest:
            tw = S.build_tw(L, keys, (a,), hv, mv)
            if tw is None:
                continue
            per[a] = {**tw.meta, "diff": tw.run(diff)}  # type: ignore[attr-defined]
        ests = {a: v["diff"] for a, v in per.items() if v["diff"].get("p") is not None}
        block["per_system"] = per
        block["above_players"] = _holm_count(ests, +1)
        block["below_players"] = _holm_count(ests, -1)
        block["cells_with_player_answers"] = len(keys)
        out[name] = block
    return out


# --------------------------------------------------------------------------- A2


def a2(L: dict[str, Any], q2: dict[str, Any]) -> dict[str, Any]:
    d, axis = L["d"], L["axis"]
    keys = sorted(k for k in L["by_cell"] if axis(k) is not None)

    def hv(t: Any, k: Any) -> dict[str, float]:
        return {"credit": t.credit}

    def mv(a: str, k: Any) -> dict[str, float]:
        return {"credit": d.models[a][k].credit}

    lead = ["second-speaker", "slot-noise"]
    trail = ["scene (environmental)", "delivery emotion"]
    out: dict[str, Any] = {
        "axis_rule": "one axis per cell: paper_analyses.taxonomy_axis (Table 2 assignment; "
        "scene > speaker > channel > delivery precedence), so no cell is double-counted",
        "heterogeneity_sets": {"A": lead, "B": trail},
    }
    for g, arms in (("gemini37or", (S.BEST,)), ("frontier4", FRONT)):
        tw = S.build_tw(L, keys, arms, hv, mv)
        assert tw is not None
        ax = np.array([axis(k) for k in tw.cells])
        per = {}
        for a_ in sorted(set(ax)):
            m = ax == a_

            def gap(H: Any, M: Any, cw: Any, idx: Any, m: Any = m) -> float | None:
                mm = m[idx]
                return S.wm((M["credit"] - H["credit"])[mm], cw[mm]) if mm.any() else None

            e = _both(tw, gap)
            per[a_] = {
                "cells": int(m.sum()),
                **e,
                "excludes_zero_two_way": bool(
                    e["two_way"].get("lo") is not None
                    and (e["two_way"]["lo"] > 0 or e["two_way"]["hi"] < 0)
                ),
                "excludes_zero_item_only": bool(
                    e["item_only"].get("lo") is not None
                    and (e["item_only"]["lo"] > 0 or e["item_only"]["hi"] < 0)
                ),
                "contribution": q2[g]["by_axis"]["groups"][a_]["contribution"],
            }
        contrib_sum = round(sum(v["contribution"]["mean"] for v in per.values()), 4)
        ma, mb = np.isin(ax, lead), np.isin(ax, trail)

        def het(H: Any, M: Any, cw: Any, idx: Any) -> float | None:
            a, b_ = ma[idx], mb[idx]
            if not a.any() or not b_.any():
                return None
            g_ = M["credit"] - H["credit"]
            return S.wm(g_[a], cw[a]) - S.wm(g_[b_], cw[b_])  # type: ignore[operator]

        out[g] = {
            **tw.meta,  # type: ignore[attr-defined]
            "overall": _both(tw, S.diff("credit")),
            "per_axis": per,
            "contribution_sum": contrib_sum,
            "heterogeneity_A_minus_B": _both(tw, het),
        }
    return out


# --------------------------------------------------------------------------- A3


def a3(b: rc.Bank, q5: dict[str, Any]) -> dict[str, Any]:
    from voxparity.harness import paper_analyses as pa

    C = b.cascade
    cue = b.cue(C.audio)
    arms = b.contestants
    med = {}
    for k in cue:
        vals = [A.audio[k] for A in arms if k in A.audio]
        if vals:
            med[k] = float(np.median(vals)) - C.audio[k]
    median_level = {k: float(np.median([A.audio[k] for A in arms if k in A.audio])) for k in cue}
    out: dict[str, Any] = {
        "n_systems": len(arms),
        "cue_cells": len(cue),
        "median_system_minus_cascade": rc.cells_test(med),
        "median_system_level": rc.cells_test(median_level),
        "cascade_level": rc.cells_test({k: C.audio[k] for k in cue}),
        "median_of_system_means": round(
            float(np.median([np.mean([A.audio[k] for k in b.cue(A.audio)]) for A in arms])), 4
        ),
    }
    per = {}
    for A in arms:
        cells = {k: A.audio[k] - C.audio[k] for k in b.cue(A.audio) if k in C.audio}
        per[A.label] = {
            "name": pa.display(A.label),
            "mode": pa.arm_mode(A),
            "twin": bool(A.twin),
            "level_minus_cascade": rc.cells_test(cells),
        }
    ests = {a: v["level_minus_cascade"] for a, v in per.items()}
    adj = rc.holm({a: e["p"] for a, e in ests.items()})
    for a, v in per.items():
        e = v["level_minus_cascade"]
        v["holm_p"] = adj[a]
        v["above_raw"], v["below_raw"] = e["lo"] > 0, e["hi"] < 0
        v["above_holm"] = adj[a] < 0.05 and e["mean"] > 0
        v["below_holm"] = adj[a] < 0.05 and e["mean"] < 0
    out["per_system_level"] = per
    counts: dict[str, Any] = {}
    for mode in ("all", "file", "realtime", "local"):
        rs = [v for v in per.values() if mode == "all" or v["mode"] == mode]
        counts[mode] = {
            "n": len(rs),
            **{
                k: sum(v[k] for v in rs)
                for k in ("above_raw", "below_raw", "above_holm", "below_holm")
            },
        }
    out["level_counts_by_mode"] = counts
    # clears by mode: twin-bearing DiD (Holm across 23, q5) + twin-less level (Holm across 5)
    did = {r["label"]: r for r in q5["rows"]}
    twinless = {a: v for a, v in per.items() if not v["twin"]}
    adj5 = rc.holm({a: v["level_minus_cascade"]["p"] for a, v in twinless.items()})
    clears: dict[str, Any] = {}
    for mode in ("file", "realtime", "local"):
        tb = [a for a, v in per.items() if v["mode"] == mode and v["twin"]]
        tl = [a for a, v in twinless.items() if v["mode"] == mode]
        clears[mode] = {
            "twin_bearing": len(tb),
            "twin_bearing_clear": sorted(per[a]["name"] for a in tb if did[a]["clears"]),
            "twinless": len(tl),
            "twinless_above_holm5": sorted(
                per[a]["name"]
                for a in tl
                if adj5[a] < 0.05 and per[a]["level_minus_cascade"]["mean"] > 0
            ),
            "twinless_below_holm5": sorted(
                per[a]["name"]
                for a in tl
                if adj5[a] < 0.05 and per[a]["level_minus_cascade"]["mean"] < 0
            ),
        }
    out["clears_by_mode"] = clears
    out["note"] = "serving mode is confounded with vendor and generation (D115): descriptive only"
    return out


# --------------------------------------------------------------------------- A4


def a4(L: dict[str, Any]) -> dict[str, Any]:
    d, hc = L["d"], L["harm_cell"]
    keys = sorted(k for k in L["by_cell"] if k[0] not in L["held"])
    contest = _contestants(L)

    def hv(t: Any, k: Any) -> dict[str, float]:
        return {"risk": 100 * S.ha.harm(hc(k, t.tool)) / 10.0}

    def mv(a: str, k: Any) -> dict[str, float]:
        return {"risk": 100 * S.ha.harm(hc(k, d.models[a][k].tool)) / 10.0}

    per: dict[str, Any] = {}
    for a in [*contest, S.CASCADE]:
        tw = S.build_tw(L, keys, (a,), hv, mv)
        if tw is None:
            continue
        per[a] = {
            **tw.meta,  # type: ignore[attr-defined]
            "system": tw.run(S.level("M", "risk")),
            "players_rate": tw.run(S.level("H", "risk")),
            "diff": _both(tw, S.diff("risk")),
        }
    ests = {a: v["diff"]["two_way"] for a, v in per.items() if a in contest}
    ests_io = {a: v["diff"]["item_only"] for a, v in per.items() if a in contest}
    # answer bases: every scored game action answer vs the harm basis
    H = d.humans
    by_engine = Counter(t.engine for t in H)
    gem = [t for t in H if t.engine == S.ENGINE]
    return {
        "per_system": per,
        "above_players_two_way": _holm_count(ests, +1),
        "above_players_item_only": _holm_count(ests_io, +1),
        "all_positive_point": sum(e["mean"] > 0 for e in ests.values()),
        "closest": min(ests, key=lambda a: ests[a]["mean"]),
        "answer_bases": {
            "all_scored_action_answers": len(H),
            "players": len({t.player for t in H}),
            "by_engine": dict(by_engine),
            "gemini_answers": len(gem),
            "gemini_answers_held_items": sum(t.item in L["held"] for t in gem),
            "harm_basis_answers": sum(len(L["by_cell"][k]) for k in keys),
            "harm_basis_cells": len(keys),
            "explanation": (
                "the 638-answer basis counts every scored action answer on every stimulus "
                "engine (Gemini-TTS plus the other sources players heard); the harm basis "
                "keeps Gemini-TTS answers only (the leaderboard's engine, where every system "
                "has a same-clip cell) and drops held items; invariant controls are out of both"
            ),
        },
    }


# --------------------------------------------------------------------------- A5


def a5(L: dict[str, Any]) -> dict[str, Any]:
    from voxparity.harness.final_analysis import load_arm
    from voxparity.harness.human_baseline import selection_credit

    d, axis = L["d"], L["axis"]
    arm = load_arm(rc.BANK / "runs/20260915-exp-oracleaudio-gemini37or-gemini", L["items"])
    assert arm is not None
    note = {
        (iid, f"{vid}@{S.ENGINE}"): selection_credit(row.get("scores") or {})
        for (iid, vid), row in arm.audio_rows.items()
    }
    keys = sorted(k for k in L["by_cell"] if axis(k) is not None and k in note)

    def hv(t: Any, k: Any) -> dict[str, float]:
        return {"credit": t.credit}

    def mv(a: str, k: Any) -> dict[str, float]:
        return {"credit": note[k] if a == "NOTE" else d.models[a][k].credit}

    d.models.setdefault("NOTE", {k: True for k in note})  # presence only, for build_tw
    subsets = {
        "all cue": lambda a: True,
        "scene (environmental)": lambda a: a == "scene (environmental)",
        "second-speaker": lambda a: a == "second-speaker",
        "delivery emotion (excl. sarcasm)": lambda a: a == "delivery emotion",
        "delivery emotion + sarcasm": lambda a: a in ("delivery emotion", "sarcasm"),
    }
    out: dict[str, Any] = {
        "run": "20260915-exp-oracleaudio-gemini37or-gemini (own audio + oracle note)"
    }
    for name, keep in subsets.items():
        ks = [k for k in keys if keep(axis(k))]
        res = {}
        for lab, arms in (("note", ("NOTE",)), ("no_note", (S.BEST,))):
            tw = S.build_tw(L, ks, arms, hv, mv)
            if tw is None:
                continue
            res[lab] = {
                **tw.meta,  # type: ignore[attr-defined]
                "model": tw.run(S.level("M", "credit")),
                "players_rate": tw.run(S.level("H", "credit")),
                "diff": _both(tw, S.diff("credit")),
            }
        out[name] = res
    del d.models["NOTE"]
    return out


# --------------------------------------------------------------------------- A6


def a6(L: dict[str, Any], exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Named-but-not-acted, emotional vs other cues. ``exclude`` drops whole items
    (samecall_core.py passes the 19 legacy LLM-drafted items)."""
    items = L["items"]
    sc = json.loads((rc.OUT / "samecall.json").read_text())["cells_detail"]
    rows = []
    for c in sc:
        if not c["cue_bearing"] or c["item"] in exclude:
            continue
        it = items[c["item"]]
        v = next(x for x in it.variants if x.variant_id == c["variant"])
        sib = {str(x.gold.tool) for x in it.variants if x.variant_id != c["variant"]} - {
            str(v.gold.tool)
        }
        tool = c["tool"]
        not_acted = (not c["passed"]) and (
            tool is None or tool == "ask_clarifying_question" or tool in sib
        )
        words = (not c["passed"]) and tool in sib
        emo = c["axis"] in ("delivery emotion", "sarcasm")
        rows.append({**c, "not_acted": not_acted, "words": words, "emo": emo})
    out: dict[str, Any] = {
        "cue_cells": len(rows),
        "emotional_cells": sum(r["emo"] for r in rows),
        "emotion_def": "axis in (delivery emotion, sarcasm), as in §3.1/§5.2/Table A7",
        "words_def": "not passed and the tool is a sibling variant's gold (strict words' action)",
    }
    for coder in ("rule", "llm"):
        named = [r for r in rows if (r["codes"] or {}).get(coder) == r["variant"]]
        block: dict[str, Any] = {
            "named_emotional": sum(r["emo"] for r in named),
            "named_other": sum(not r["emo"] for r in named),
        }
        for dfn, pred in (
            ("not_acted", lambda r: r["not_acted"]),
            ("words", lambda r: r["words"]),
            ("strict_fail", lambda r: not r["passed"]),
        ):
            vals_e = {(r["item"], r["variant"]): float(pred(r)) for r in named if r["emo"]}
            vals_o = {(r["item"], r["variant"]): float(pred(r)) for r in named if not r["emo"]}

            # difference of rates, joint item-clustered bootstrap
            allk = sorted({**vals_e, **vals_o})
            items_ = sorted({k[0] for k in allk})
            idx = {i: n for n, i in enumerate(items_)}
            ki = np.array([idx[k[0]] for k in allk])
            y = np.array([vals_e.get(k, vals_o.get(k)) for k in allk])
            isE = np.array([k in vals_e for k in allk])

            def rates(w: np.ndarray) -> tuple[float, float] | None:
                we, wo = w[isE], w[~isE]
                if we.sum() == 0 or wo.sum() == 0:
                    return None
                return float((y[isE] * we).sum() / we.sum()), float((y[~isE] * wo).sum() / wo.sum())

            rng = np.random.default_rng(rc.SEED)
            draws = []
            for _ in range(S.N_BOOT):
                cnt = np.bincount(rng.integers(0, len(items_), len(items_)), minlength=len(items_))
                r_ = rates(cnt[ki].astype(float))
                if r_:
                    draws.append(r_)
            dr = np.array(draws)
            pt = rates(np.ones(len(allk)))
            assert pt

            def summ(point: float, v: np.ndarray) -> dict[str, Any]:
                return {
                    "mean": round(point, 4),
                    "lo": round(float(np.quantile(v, 0.025)), 4),
                    "hi": round(float(np.quantile(v, 0.975)), 4),
                    "lo90": round(float(np.quantile(v, 0.05)), 4),
                    "hi90": round(float(np.quantile(v, 0.95)), 4),
                }

            block[dfn] = {
                "emotional": {**summ(pt[0], dr[:, 0]), "k": int(y[isE].sum()), "n": int(isE.sum())},
                "other": {**summ(pt[1], dr[:, 1]), "k": int(y[~isE].sum()), "n": int((~isE).sum())},
                "difference": summ(pt[0] - pt[1], dr[:, 0] - dr[:, 1]),
                "items": len(items_),
            }
        out[coder] = block
    return out

    # --------------------------------------------------------------------------- A7


def a7(L: dict[str, Any], b: rc.Bank) -> dict[str, Any]:
    axis, hc = L["axis"], L["harm_cell"]
    arms = b.contestants
    cue = b.cue(b.cascade.audio)
    res: dict[str, Any] = {}
    for crit in ("selection", "strict_pass"):

        def right(A: Any, k: Any) -> bool | None:
            if k not in A.audio:
                return None
            if crit == "strict_pass":
                return bool(A.audio_passed.get(k))
            s = A.audio_rows[k].get("scores") or {}
            return bool(s.get("selection"))

        zero = []
        for k in cue:
            rr = [right(A, k) for A in arms]
            rr = [x for x in rr if x is not None]
            if rr and not any(rr):
                zero.append((k, len(rr)))
        cells = [k for k, _ in zero]
        hkeys = [(k[0], f"{k[1]}@{S.ENGINE}") for k in cells]
        hkeys = [k for k in hkeys if k in L["by_cell"]]
        entry: dict[str, Any] = {
            "cells": len(cells),
            "systems_per_cell_min": min((n for _, n in zero), default=None),
            "by_axis": dict(Counter(axis((k[0], f"{k[1]}@x")) for k in cells).most_common()),
            "by_tier": dict(
                Counter(
                    hc((k[0], f"{k[1]}@x"), None)["tier"] if k[0] not in L["held"] else "held"
                    for k in cells
                ).most_common()
            ),
            "by_role": dict(
                Counter(
                    hc((k[0], f"{k[1]}@x"), None)["role"] if k[0] not in L["held"] else "held"
                    for k in cells
                ).most_common()
            ),
            "list": [f"{k[0]}/{k[1]}" for k in cells],
            "players_answered_cells": len(hkeys),
        }
        if hkeys:
            answers = [
                (i, t.player, {"credit": t.credit, "hit": float(t.hit)})
                for i, k in enumerate(hkeys)
                for t in L["by_cell"][k]
            ]
            tw = S.TwoWay(
                hkeys, answers, {"credit": np.zeros(len(hkeys)), "hit": np.zeros(len(hkeys))}
            )
            entry["players_answers"] = len(answers)
            entry["players_credit"] = _both(tw, S.level("H", "credit"))
            entry["players_hit"] = _both(tw, S.level("H", "hit"))
            per = defaultdict(list)
            for k in hkeys:
                per[axis(k)].append(np.mean([t.credit for t in L["by_cell"][k]]))
            entry["players_cell_mean_by_axis"] = {
                a: {"cells": len(v), "mean": round(float(np.mean(v)), 3)} for a, v in per.items()
            }
        res[crit] = entry
    return res


def run(L: dict[str, Any], b: rc.Bank, out: dict[str, Any]) -> dict[str, Any]:
    return {
        "A1": a1(L),
        "A2": a2(L, out["q2"]),
        "A3": a3(b, out["q5"]),
        "A4": a4(L),
        "A5": a5(L),
        "A6": a6(L),
        "A7": a7(L, b),
    }
