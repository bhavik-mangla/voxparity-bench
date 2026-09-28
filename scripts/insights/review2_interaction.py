# ruff: noqa: E501  (report-rendering f-strings; wrapping them hurts readability)
"""Review round 2 re-cuts (E30, E31, spot checks): no model calls, no spend.

E30  emotion x cue-kind interaction on P(right action | heard), protocol-grounded
     core, first-turn scoring (D118):
       [(frontier - players) on emotion] - [(frontier - players) on other cues]
     under raw, guess-corrected (1/k and own-FA), pair-level and balanced-credit
     definitions, plus the within-model contrast (emotion - other) for the pooled
     roster and frontier-4, on identical cells and on the whole Gemini-TTS bank.
E31  frontier perception re-cut on cue cells admitted by an independent listener
     (gpt-audio-mini, gpt-audio cross-judges; human-validated cells), and a
     frontier-selection disclosure: frontier-4 re-selected on one half of the
     items and evaluated on the other (2-fold cross-fitting, plus 200 random
     splits for stability).

Same rows, bootstrap and seed as perception_robustness.py (humans two-way item x
player pigeonhole; models item-clustered; every contrast shares the item draws),
so the marginal rows reproduce docs/insights/perception-robustness.json.

    uv run --project . --extra paper python scripts/insights/review2_interaction.py

Writes docs/insights/review2-interaction.{json,md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_common as hc
import perception_robustness as pr
from human_insights import Ctx

from voxparity import private_data

XJ = {
    "gpt-audio-mini": hc.HERE / "docs/results/final/crossjudge.json",
    "gpt-audio": hc.HERE / "docs/results/final/crossjudge-gptaudio.json",
}
DEFS = ("raw", "corr_uniform", "corr_bias", "pair")
N_SPLITS = 200


def keep_fn(hrows: list[dict], bank: dict[str, list[dict]]) -> Any:
    legacy = set(pr.load_legacy())
    emo_items = {r["item"] for r in hrows if r["emo"]} | {
        r["item"] for rs in bank.values() for r in rs if r["emo"]
    }
    non_items = {r["item"] for r in hrows if r["cue"] and not r["emo"]} | {
        r["item"] for rs in bank.values() for r in rs if r["cue"] and not r["emo"]
    }

    def keep(sub: str, r: dict[str, Any]) -> bool:
        if sub == "any":
            return True
        if r["item"] in legacy:
            return False
        if sub == "emo":
            return bool(r["emo"]) or (bool(r["clean"]) and r["item"] in emo_items)
        if sub == "non":
            return (bool(r["cue"]) and not r["emo"]) or (
                bool(r["clean"]) and r["item"] in non_items
            )
        return True

    return keep


def metrics(s: pr.S) -> dict[str, Any]:
    a = pr.action_stats(s)
    out = {d: a[d]["aT"] for d in DEFS}
    out["balanced_credit"] = a["balanced_credit"]
    out["credit_cue"] = a["credit_cue"]
    return out


def interaction(
    H: dict[str, pr.Pop], G: dict[str, dict[str, pr.Pop]], bt: pr.Boot
) -> dict[str, Any]:
    """H = {'emo','non'}; G = group -> {'emo','non'}. One pass, shared weights."""
    pops = {f"h:{k}": v for k, v in H.items()}
    for g, pp in G.items():
        pops.update({f"{g}:{k}": v for k, v in pp.items()})

    def fn(s: dict[str, pr.S]) -> dict[str, Any]:
        he, hn = metrics(s["h:emo"]), metrics(s["h:non"])
        res: dict[str, Any] = {"humans": {"emo": he, "non": hn}}
        res["humans"]["emo_minus_non"] = {k: he[k] - hn[k] for k in he}
        for g in G:
            ge, gn = metrics(s[f"{g}:emo"]), metrics(s[f"{g}:non"])
            res[g] = {
                "emo": ge,
                "non": gn,
                "emo_minus_non": {k: ge[k] - gn[k] for k in ge},
                "minus_humans_emo": {k: ge[k] - he[k] for k in ge},
                "minus_humans_non": {k: gn[k] - hn[k] for k in gn},
                "interaction": {k: (ge[k] - he[k]) - (gn[k] - hn[k]) for k in ge},
            }
        return res

    return pr.run(pops, bt, fn)


def model_only(G: dict[str, dict[str, pr.Pop]], bt: pr.Boot) -> dict[str, Any]:
    pops = {f"{g}:{k}": v for g, pp in G.items() for k, v in pp.items()}

    def fn(s: dict[str, pr.S]) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for g in G:
            ge, gn = metrics(s[f"{g}:emo"]), metrics(s[f"{g}:non"])
            res[g] = {"emo": ge, "non": gn, "emo_minus_non": {k: ge[k] - gn[k] for k in ge}}
        return res

    return pr.run(pops, bt, fn)


def select_frontier(bank: dict[str, list[dict]], items: set[str], k: int = 4) -> list[str]:
    cred = {}
    for a, rows in bank.items():
        ys = [r["y"] for r in rows if r["cue"] and r["item"] in items]
        if ys:
            cred[a] = float(np.mean(ys))
    ranked = [a for a in sorted(cred, key=lambda a: -cred[a]) if a not in pr.PROBE_SUSPECT]
    return ranked[:k]


def perception_block(H: pr.Pop, F: pr.Pop, bt: pr.Boot) -> dict[str, Any]:
    def fn(s: dict[str, pr.S]) -> dict[str, Any]:
        ph, pf = pr.perception_stats(s["h"]), pr.perception_stats(s["f"])
        ah, af = pr.action_stats(s["h"]), pr.action_stats(s["f"])
        return {
            "humans": {
                "pi_bias": ph["pi_bias"],
                "acc_cue": ph["acc_cue"],
                "raw_aT": ah["raw"]["aT"],
                "corr_bias_aT": ah["corr_bias"]["aT"],
            },
            "frontier4": {
                "pi_bias": pf["pi_bias"],
                "acc_cue": pf["acc_cue"],
                "raw_aT": af["raw"]["aT"],
                "corr_bias_aT": af["corr_bias"]["aT"],
            },
            "frontier4_minus_humans": {
                "pi_bias": pf["pi_bias"] - ph["pi_bias"],
                "raw_aT": af["raw"]["aT"] - ah["raw"]["aT"],
                "corr_bias_aT": af["corr_bias"]["aT"] - ah["corr_bias"]["aT"],
            },
        }

    return pr.run({"h": H, "f": F}, bt, fn)


def bank_perception(F: pr.Pop, bt: pr.Boot) -> dict[str, Any]:
    return pr.run(
        {"f": F},
        bt,
        lambda s: {
            k: v
            for k, v in pr.perception_stats(s["f"]).items()
            if k in ("pi_bias", "acc_cue", "pi_uniform")
        },
    )


def admitted_sets() -> dict[str, set[tuple[str, str]]]:
    out: dict[str, set[tuple[str, str]]] = {}
    human: set[tuple[str, str]] = set()
    for name, path in XJ.items():
        v = json.loads(path.read_text())["verdicts"]
        xs = set()
        for key, r in v.items():
            iid, vid = key.split("/", 1)
            if r.get("xjudge") is True:
                xs.add((iid, vid))
            if r.get("human_check") is True:
                human.add((iid, vid))
        out[f"xjudge:{name}"] = xs
    out["xjudge:either"] = out["xjudge:gpt-audio-mini"] | out["xjudge:gpt-audio"]
    out["xjudge:both"] = out["xjudge:gpt-audio-mini"] & out["xjudge:gpt-audio"]
    out["human_validated"] = human
    out["independent (either xjudge or human)"] = out["xjudge:either"] | human
    return out


def restrict_cue(rows: list[dict], adm: set[tuple[str, str]] | None) -> list[dict]:
    """Cue rows only on admitted Gemini-TTS clips; clean rows kept (FA rate is a respondent property)."""
    if adm is None:
        return rows
    return [
        r
        for r in rows
        if not r["cue"] or (r["engine"] == "gemini" and (r["item"], r["variant"]) in adm)
    ]


def spot_checks(d: hc.Data) -> dict[str, Any]:
    """Hotel over-trigger control (selection vs strict with arguments), contestants only."""
    # The control cell (item, clean variant, its gold tool) is held-out: private data.
    ctl = private_data.load("insights/review2_interaction.json")["hotel_control"]
    key = (ctl["item"], f"{ctl['variant']}@gemini")
    contest = [a for a, r in d.roles.items() if r == "contestant"]
    tools: dict[str, int] = {}
    for a in contest:
        m = d.models[a].get(key)
        if m is not None:
            tools[str(m.tool)] = tools.get(str(m.tool), 0) + 1
    strict = 0
    for a in contest:
        f = hc.BANK / f"runs/20260915-final-{a}-gemini/records.jsonl"
        last = None
        for line in f.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if (
                r.get("item_id") == key[0]
                and r.get("variant_id") == ctl["variant"]
                and r.get("condition") == "audio"
                and not r.get("error")
            ):
                last = r
        strict += bool(last and (last.get("scores") or {}).get("passed"))
    return {
        "hotel_over_trigger_control": {
            "contestants": len(contest),
            "tool_counts": tools,
            "selection_correct": tools.get(ctl["gold_tool"], 0),
            "strict_with_arguments": strict,
        },
        "kokoro_same_cell": "docs/results/final/same_cell.md gemini37or gemini->kokoro, 117 cells: audio level -0.03 [-0.08, +0.01]; audio-minus-twin effect difference -0.03 [-0.08, +0.02]",
    }


def main() -> None:
    d = hc.load()
    ctx = Ctx(d)
    hrows, ident, bank = pr.build_rows(ctx)
    keep = keep_fn(hrows, bank)

    bank_credit = {
        a: float(np.mean([r["y"] for r in rows if r["cue"]])) for a, rows in bank.items()
    }
    ranked = sorted(bank_credit, key=lambda a: -bank_credit[a])
    frontier5 = ranked[: pr.FRONTIER_K]
    frontier4 = [a for a in frontier5 if a not in pr.PROBE_SUSPECT]

    items_all = sorted(
        {r["item"] for r in hrows}
        | {r["item"] for rs in bank.values() for r in rs}
        | {r["item"] for rs in ident.values() for r in rs}
    )
    players = sorted({r["player"] for r in hrows})
    bt = pr.Boot(items_all, players)

    def P(rows: list[dict], human: bool = False, name: str = "p") -> pr.Pop:
        return pr.Pop(name, rows, human=human)

    def split(rows: list[dict], human: bool = False) -> dict[str, pr.Pop]:
        return {k: P([r for r in rows if keep(k, r)], human) for k in ("emo", "non")}

    H = split(hrows, human=True)
    groups = {
        "pooled": split([r for a in ident for r in ident[a]]),
        "frontier4": split([r for a in frontier4 for r in ident[a]]),
        "frontier5": split([r for a in frontier5 for r in ident[a]]),
        "best": split(ident[ranked[0]]),
    }
    res: dict[str, Any] = {
        "freeze": d.meta.get("freeze"),
        "scoring": "first-turn (D118; final_analysis.load_arms default)",
        "frontier5": frontier5,
        "frontier4": frontier4,
        "counts": {
            "human_cue_emo": int(H["emo"].c["cue"].sum()),
            "human_cue_non": int(H["non"].c["cue"].sum()),
            "players": len(players),
            "items_emo": len({r["item"] for r in H["emo"].rows if r["cue"]}),
            "items_non": len({r["item"] for r in H["non"].rows if r["cue"]}),
        },
    }
    print("E30 identical cells", file=sys.stderr)
    res["E30_identical"] = interaction(H, groups, bt)
    print("E30 whole bank", file=sys.stderr)
    bgroups = {
        "pooled": split([r for a in bank for r in bank[a]]),
        "frontier4": split([r for a in frontier4 for r in bank[a]]),
    }
    res["E30_bank_within_model"] = model_only(bgroups, bt)

    # ---- E31a: perception on independently admitted cells
    print("E31 admitted subsets", file=sys.stderr)
    adm = admitted_sets()
    # E31 uses every item (the paper's perception headline is on all items, not core)
    core = [r for r in hrows if keep("any", r)]
    fr_ident = [r for a in frontier4 for r in ident[a] if keep("any", r)]
    e31: dict[str, Any] = {}
    for name, s in [("all (identical cells)", None), *adm.items()]:
        Hs = P(restrict_cue(core, s), True)
        Fs = P(restrict_cue(fr_ident, s))
        blk = perception_block(Hs, Fs, bt)
        blk["n_cue_cells"] = len({r["cell"] for r in Hs.rows if r["cue"]})
        blk["n_human_cue_answers"] = int(Hs.c["cue"].sum())
        e31[name] = blk
    res["E31_identical"] = e31
    # whole bank, per model family: does Gemini's probe shrink more than the others'?
    e31b: dict[str, Any] = {}
    fam = {
        "frontier4": frontier4,
        "frontier4_gemini": [a for a in frontier4 if "gemini" in a],
        "frontier4_nongemini": [a for a in frontier4 if "gemini" not in a],
    }
    for name, s in [("all", None), *adm.items()]:
        e31b[name] = {}
        for fname, arms in fam.items():
            rows = [r for a in arms for r in bank[a] if keep("any", r)]
            Fp = P(restrict_cue(rows, s))
            e31b[name][fname] = bank_perception(Fp, bt)
        e31b[name]["n_cue_cells"] = len(
            {r["cell"] for r in restrict_cue(bank[frontier4[0]], s) if r["cue"] and keep("any", r)}
        )
    res["E31_bank"] = e31b

    # ---- E31b: frontier selection disclosure (2-fold cross-fit + split stability)
    print("E31 cross-fit", file=sys.stderr)
    rng = np.random.default_rng(pr.SEED)
    perm = rng.permutation(items_all)
    A, Bh = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2 :])
    fA, fB = select_frontier(bank, A), select_frontier(bank, Bh)
    cf_ident = [r for a in fA for r in ident[a] if r["item"] in Bh] + [
        r for a in fB for r in ident[a] if r["item"] in A
    ]
    cf = {"frontier4_crossfit": split(cf_ident)}
    res["E31_crossfit"] = {
        "split_seed": pr.SEED,
        "frontier_selected_on_A": fA,
        "frontier_selected_on_B": fB,
        "interaction": interaction(H, cf, bt),
        "perception": perception_block(
            P(core, True), P([r for r in cf_ident if keep("any", r)]), bt
        ),
    }
    # stability over random splits (point estimates)
    member: dict[str, int] = {}
    stats: dict[str, list[float]] = {
        "emo_raw_minus_humans": [],
        "non_raw_minus_humans": [],
        "interaction_raw": [],
        "interaction_pair": [],
        "pi_bias_minus_humans": [],
        "emo_minus_non_raw": [],
    }
    hpt_e = pr._flat(metrics(pr.S(H["emo"], None)))
    hpt_n = pr._flat(metrics(pr.S(H["non"], None)))
    hpi = pr.perception_stats(pr.S(P(core, True), None))["pi_bias"]
    for i in range(N_SPLITS):
        rg = np.random.default_rng(pr.SEED + 1 + i)
        pm = rg.permutation(items_all)
        a_, b_ = set(pm[: len(pm) // 2]), set(pm[len(pm) // 2 :])
        fa, fb = select_frontier(bank, a_), select_frontier(bank, b_)
        for x in fa + fb:
            member[x] = member.get(x, 0) + 1
        rows = [r for a in fa for r in ident[a] if r["item"] in b_] + [
            r for a in fb for r in ident[a] if r["item"] in a_
        ]
        sp = split(rows)
        me = metrics(pr.S(sp["emo"], None))
        mn = metrics(pr.S(sp["non"], None))
        stats["emo_raw_minus_humans"].append(float(me["raw"]) - hpt_e["raw"])
        stats["non_raw_minus_humans"].append(float(mn["raw"]) - hpt_n["raw"])
        stats["interaction_raw"].append(
            float(me["raw"]) - hpt_e["raw"] - (float(mn["raw"]) - hpt_n["raw"])
        )
        stats["interaction_pair"].append(
            float(me["pair"]) - hpt_e["pair"] - (float(mn["pair"]) - hpt_n["pair"])
        )
        stats["emo_minus_non_raw"].append(float(me["raw"]) - float(mn["raw"]))
        pf = pr.perception_stats(pr.S(P([r for r in rows if keep("any", r)]), None))["pi_bias"]
        stats["pi_bias_minus_humans"].append(float(pf) - float(hpi))
    res["E31_split_stability"] = {
        "n_splits": N_SPLITS,
        "membership_share": {
            k: round(v / (2 * N_SPLITS), 3)
            for k, v in sorted(member.items(), key=lambda kv: -kv[1])
        },
        "point_estimates": {
            k: {
                "median": round(float(np.median(v)), 4),
                "p10": round(float(np.quantile(v, 0.1)), 4),
                "p90": round(float(np.quantile(v, 0.9)), 4),
            }
            for k, v in stats.items()
        },
    }
    # ---- E32: players' emotional-delivery cue credit (all items), for the oracle-note comparison
    Hemo_all = P([r for r in hrows if r["emo"]], True)
    res["E32_players_emotion_credit_all_items"] = {
        **pr.run(
            {"h": Hemo_all}, bt, lambda s: {"credit_cue": pr.action_stats(s["h"])["credit_cue"]}
        ),
        "n_answers": Hemo_all.n,
    }
    exp = json.loads((hc.HERE / "docs/results/exp/experiments.json").read_text())
    res["E32_note_by_axis_credit"] = {
        c["exp"]: {ax: c["by_axis"][ax]["credit"] for ax in ("emotion", "scene", "speaker")}
        for c in exp["note_controls"]["conditions"]
        if "gemini37or" in c["exp"]
    }
    res["checks"] = spot_checks(d)
    out = hc.OUT / "review2-interaction.json"
    out.write_text(json.dumps(res, indent=1, default=lambda o: None))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
