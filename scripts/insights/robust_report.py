# ruff: noqa: E501  (a renderer: long prose lines; the paper's own U+2212 minus is quoted verbatim)
"""REANALYSIS D driver: runs tasks 1-7 and writes docs/insights/robustness-v2.{json,md}.

    cd $VXP_BANK && uv run --project $VXP_CODE \
        python $VXP_CODE/scripts/insights/robust_report.py

Deterministic (seed 20260915); no model calls, no spend. The bottom-line prose in
the markdown is written from the numbers computed here; if a re-run changes a
count, the tables change and the prose flags must be re-read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import robust_common as rc
import robust_floor as rf
import robust_harm as rhm
import robust_human as rh

F, FN = rc.fmt, rc.fp


def stacked(b: rc.Bank) -> dict[str, Any]:
    """Every conservative choice at once: first-turn scoring (primary since D118),
    no legacy items, must clear all three floors (IUT), twin-bearing Holm family."""
    nb = b if rf.alt_mode() == "followup" else rf.with_turn1(b)
    legacy = set(rc.legacy_items())
    keep = {k[0] for a in b.primary for k in a.audio} - legacy
    return rf.leaderboard(nb, keep=keep)


def robustness_matrix(res: dict[str, Any]) -> list[dict[str, Any]]:
    lb = {r["label"]: r for r in res["floor"]["leaderboard"]["rows"]}
    t1 = {r["label"]: r for r in res["floor"]["ladder"]["leaderboard_alt"]["rows"]}
    lg = {r["label"]: r for r in res["floor"]["legacy"]["leaderboard"]["rows"]}
    st = {r["label"]: r for r in res["stacked"]["rows"]}
    ne = {r["label"]: r for r in res["floor"]["neutral"]["rows"]}
    out = []
    for lab, r in lb.items():
        tb = "twin_bearing"
        # The published clears under first-turn scoring (D118): the twin-bearing
        # Holm family on the gpt-oss floor (11 of 23).
        if not r["twin"] or not r["clears"][tb]["gptoss"]:
            continue
        checks = {
            "gpt-oss floor, single 28-system Holm family (mixed estimands)": r["clears"][
                "paper_mixed_28"
            ]["gptoss"],
            "twin-bearing Holm family (23)": r["clears"][tb]["gptoss"],
            "Sonnet 5 floor": r["clears"][tb]["sonnet5"],
            "DeepSeek-V4-Pro floor": r["clears"][tb]["dsv4pro"],
            "all three floors (IUT)": r["clears"][tb]["all_floors_iut"],
            "own delta > top floor's upper CI": r["clears"][tb]["own_delta_above_top_floor_upper"],
            f"{res['floor']['ladder']['alt_mode'].replace('_', '-')} scoring (sensitivity)": t1[
                lab
            ]["clears"][tb]["gptoss"],
            "without 19 legacy items": lg[lab]["clears"][tb]["gptoss"],
            "neutral DiD not significantly < 0": not ne[lab]["neutral_sig_negative"],
            "over-reaction-penalised DiD": ne[lab]["penalised_clears"],
            "stacked (first turn, no legacy, all floors)": st[lab]["clears"][tb]["all_floors_iut"],
        }
        out.append(
            {
                "label": lab,
                "name": r["name"],
                "did": r["gptoss"],
                "checks": checks,
                "passed": sum(checks.values()),
                "of": len(checks),
            }
        )
    return out


def main() -> None:
    b = rc.Bank()
    res: dict[str, Any] = {
        "method": {
            "freeze": b.freeze.get("freeze_id"),
            "engine": "gemini (Gemini-TTS), arms >= 90% coverage",
            "ci": "item-clustered percentile bootstrap, 4000 resamples, seed 20260915 "
            "(human two-way item x player: 2000 resamples)",
            "p": "two-sided bootstrap p by CI inversion, floored at 1/4000; Holm step-down",
            "tost": "equivalent iff the 90% CI lies inside +/- margin (alpha .05 per side)",
            "legacy_source": f"{rc.REF_ATLAS}:docs/insights/atlas.json llm_drafted_legacy",
            "harm_source": f"{rc.REF_HARM}:scripts/insights/harm_{{analysis,rubric}}.py",
            "human_source": f"{rc.REF_HUMAN}:scripts/insights/human_common.py",
        },
        "floor": rf.run(b),
        "stacked": stacked(b),
        "human": rh.run(b),
        "harm": rhm.run(b),
    }
    res["matrix"] = robustness_matrix(res)
    rc.OUT.mkdir(parents=True, exist_ok=True)
    (rc.OUT / "robustness-v2.json").write_text(
        json.dumps(res, indent=1, sort_keys=True, default=str) + "\n"
    )
    (rc.OUT / "robustness-v2.md").write_text(render(res))
    print("wrote", rc.OUT / "robustness-v2.md")


# --------------------------------------------------------------------------- bottom line


def bottom_line(res: dict[str, Any]) -> str:
    """Computed summary (D118: first-turn scoring is primary). Every count and name
    below is read from ``res``; nothing is hard-coded, so a re-run cannot leave the
    prose behind the tables."""
    fl = res["floor"]
    f = fl["floors"]
    lbd = fl["leaderboard"]
    lb = lbd["counts"]
    lad = fl["ladder"]
    alt = lad["leaderboard_alt"]["counts"]
    ne = fl["neutral"]
    st = res["stacked"]["counts"]["twin_bearing"]
    hu = res["human"]["act_given_heard"]
    hz = res["harm"]
    mg = hz["mandate_vs_guidance"]["all_items"]
    u = hz["unsafe_vs_cascade"]
    tb = "twin_bearing"
    rows = lbd["rows"]
    twin_rows = [r for r in rows if tb in r["clears"]]
    clears = [r for r in twin_rows if r["clears"][tb]["gptoss"]]
    core = [r for r in twin_rows if r["clears"][tb].get("all_floors_iut")]
    shift = [r for r in twin_rows if r["clears"][tb].get("own_delta_above_top_floor_upper")]
    fragile = [r["name"] for r in clears if r not in core]
    below_tl = [r["name"] for r in rows if r["below"].get("twinless", {}).get("gptoss")]
    neg_neutral = ne["neutral_sig_negative"]
    nrow = {r["label"]: r for r in ne["rows"]}
    clear_neutral = [nrow[r["label"]]["neutral_did"]["mean"] for r in clears if r["label"] in nrow]
    alt_name = lad["alt_mode"].replace("_", "-")
    ci = lambda e: F(e)  # noqa: E731
    names = lambda rs: ", ".join(r["name"] for r in rs)  # noqa: E731
    worse = u.get("worse_holm") or []
    return f"""## BOTTOM LINE (first-turn scoring primary, D118)

- **Null floors** (same Whisper transcripts, three text LLMs): gpt-oss-120b {ci(f["gptoss"]["cue"])},
  DeepSeek-V4-Pro {ci(f["dsv4pro"]["cue"])}, Claude Sonnet 5 {ci(f["sonnet5"]["cue"])}.
- **Twin-bearing systems (Holm within the 23 with a text path):** {len(clears)} of
  {lb[tb]["n"]} clear the gpt-oss floor: {names(clears)}.
- **Against every text model's floor** (intersection-union, Holm): {len(core)} of {lb[tb]["n"]}:
  {names(core)}. Fragile (clear gpt-oss only): {", ".join(fragile) or "none"}.
  {len(shift)} also clear the ultra-conservative shift test (own audio-minus-twin above the
  highest floor's upper 95% bound, {lbd["top_floor_upper_95"]:+.3f}).
- **Audio-induced caution:** clean-call DiD of the clears ranges {min(clear_neutral):+.2f} to
  {max(clear_neutral):+.2f}; significantly negative clean-call DiD: {", ".join(neg_neutral) or "none"}.
- **Twin-less systems (own Holm family, level contrast against the cascade's audio):**
  significantly below the cascade on accuracy: {", ".join(below_tl) or "none"}.
- **Sensitivity, {alt_name} scoring** (the scripted follow-up rung scored in the audio
  condition): gpt-oss floor clears {alt[tb]["gptoss"]}/{lb[tb]["n"]}, all three floors
  {alt[tb]["all_floors_iut"]}. Stacked conservative choices (first turn, no legacy items,
  all three floors): {st["all_floors_iut"]} of {lb[tb]["n"]}.
- **Humans (two-way item x player CIs):** P(right action | heard) models minus humans
  {ci(hu["models_minus_human_act_given_heard"]["two_way"])} (item-only
  {ci(hu["models_minus_human_act_given_heard"]["item_only"])}); the order-free human bound
  {F(hu["human_lower_bound"]["two_way"], signed=False)}, pooled models minus it
  {ci(hu["models_minus_human_bound"]["two_way"])} (not a separating claim).
- **Harm:** unsafe execution worse than the cascade after Holm (28):
  {", ".join(worse) or "none"}; better after Holm {u["better_holm"]}/28.
  Mandate minus guidance departure rate {F(mg["difference"])}: no difference detected,
  equivalence not established (smallest margin
  +/-{mg["tost"][0.1]["smallest_equivalence_margin"]:.2f}).
"""


# --------------------------------------------------------------------------- markdown


def render(res: dict[str, Any]) -> str:
    fl = res["floor"]
    L: list[str] = []
    a = L.append
    a("# Robustness v2: the null floor and the headline (REANALYSIS D)")
    a("")
    a(
        "Frozen bank `bank-freeze-2026-09-15`, Gemini-TTS cells, the 28 eligible contestants. "
        "Item-clustered percentile bootstrap (4000, seed 20260915); two-sided bootstrap p by CI "
        "inversion (floored at 1/4000, which is why Holm p bottoms out near 0.006-0.007); Holm "
        "step-down. Human CIs add a two-way (item x player) cluster bootstrap. Regenerate: "
        "`scripts/insights/regen-insights.sh` (all lenses, first-turn primary). "
        "No model calls, no spend."
    )
    a("")
    a(bottom_line(res))
    a("")

    # ---------------- task 1
    a("## 1. The null across three text LLMs")
    a("")
    a(
        "All three floors read the SAME cached Whisper transcripts (the replays are "
        "`exp-replay-*`); only the text model differs. Cue-bearing cells, n=206 on 163 items."
    )
    a("")
    rows = []
    for k in ("gptoss", "sonnet5", "dsv4pro"):
        v = fl["floors"][k]
        rows.append(
            [
                v["name"],
                F(v["cue"]),
                FN(v["cue"]["p"]),
                F(v["neutral"]),
                F(res["floor"]["ladder"]["floors_alt"][k]),
                F(fl["legacy"]["floors_without_legacy"][k]),
            ]
        )
    a(
        rc.md_table(
            [
                "floor",
                "audio - twin, cue",
                "p",
                "audio - twin, neutral",
                f"cue, {res['floor']['ladder']['alt_mode'].replace('_', '-')} scoring",
                "cue, without legacy items",
            ],
            rows,
        )
    )
    a("")
    pw = fl["floors"]["pairwise"]
    a(
        "The floors do not differ from each other: Sonnet minus gpt-oss "
        f"{F(pw['sonnet5 - gptoss'])}, DeepSeek minus gpt-oss {F(pw['dsv4pro - gptoss'])} "
        "(paired, identical cells)."
    )
    a("")
    d = fl["diagnosis_sonnet5"]
    a("### 1.1 What carries Sonnet 5's +0.05")
    a("")
    a(
        f"{d['discordant_cells']} of {d['cue_cells']} cue cells are discordant "
        f"({d['positive_cells']} positive, {d['negative_cells']} negative; sum "
        f"{d['sum_delta']:+.1f}, mean {d['mean_delta']:+.3f}). Every text LLM's twin sees the gold "
        "transcript once per item; its audio sees the Whisper transcript of that delivery. "
        "Mechanism per discordant cell:"
    )
    a("")
    rows = []
    for m, v in d["by_mechanism"].items():
        rows.append(
            [
                m,
                v["cells"],
                f"{v['positive']}/{v['negative']}",
                f"{v['sum_delta']:+.1f}",
                F(d["floor_without_mechanism"][m]),
            ]
        )
    a(
        rc.md_table(
            ["mechanism", "cells", "+/-", "sum of delta", "Sonnet floor with these cells zeroed"],
            rows,
        )
    )
    a("")
    for w in ("diagnosis_gptoss", "diagnosis_dsv4pro"):
        dd = fl[w]
        a(
            f"- {w.split('_')[1]}: mean {dd['mean_delta']:+.3f}; "
            + "; ".join(
                f"{m.split(':')[0]} {v['sum_delta']:+.1f} ({v['cells']} cells)"
                for m, v in dd["by_mechanism"].items()
            )
        )
    a("")
    a("Sonnet's discordant cells (L = legacy LLM-drafted item):")
    a("")
    leg = set(fl["legacy"]["legacy_items"])
    rows = []
    for r in d["discordant"]:
        rows.append(
            [
                f"{r['delta']:+.1f}",
                f"{r['item']}/{r['variant']}" + (" L" if r["item"] in leg else ""),
                r["axis"],
                r["mechanism"].split(":")[0].split("(")[0].strip(),
                f"{r['audio_tool']} vs {r['twin_tool']}",
                r["wer_whisper_vs_gold"],
                "" if r["gptoss_delta"] is None else f"{r['gptoss_delta']:+.1f}",
            ]
        )
    a(
        rc.md_table(
            ["delta", "cell", "axis", "mechanism", "audio vs twin tool", "WER", "gpt-oss delta"],
            rows,
        )
    )
    a("")

    # ---------------- task 1c/2 leaderboard
    lb = fl["leaderboard"]
    a("## 2. Every system against every floor; Holm by estimand family")
    a("")
    a(
        "Twin-bearing systems (23): diff-in-diff on cue-bearing cells. Twin-less systems (5: "
        "gpt-audio, gpt-audio-mini, Qwen3.8-Omni-Flash RT, Qwen3.5-Omni-Flash RT, VoiceChat 11B): "
        "audio credit minus the floor's audio credit, a LEVEL, Holm-corrected in its own family. "
        "p columns are Holm-adjusted within the system's own family; 'paper' is the published "
        "single 28-system family on the gpt-oss floor. IUT = intersection-union test that the "
        "system clears ALL three floors (p = max of the three). Shift = the system's own "
        f"audio-minus-twin must exceed the highest floor's upper 95% bound ({lb['top_floor_upper_95']:+.3f})."
    )
    a("")
    rows = []
    for r in lb["rows"]:
        fam = "twin_bearing" if r["twin"] else "twinless"
        h = r["holm"][fam]
        rows.append(
            [
                r["name"] + ("" if r["twin"] else " (level)"),
                F(r["gptoss"]),
                FN(r["holm"]["paper_mixed_28"]["gptoss"]),
                FN(h["gptoss"]),
                F(r["sonnet5"]),
                FN(h["sonnet5"]),
                F(r["dsv4pro"]),
                FN(h["dsv4pro"]),
                FN(h["all_floors_iut"]),
                FN(h.get("own_delta_above_top_floor_upper")) if r["twin"] else "n/a",
            ]
        )
    a(
        rc.md_table(
            [
                "system",
                "vs gpt-oss",
                "paper Holm",
                "family Holm",
                "vs Sonnet",
                "Holm",
                "vs DeepSeek",
                "Holm",
                "IUT Holm",
                "shift Holm",
            ],
            rows,
        )
    )
    a("")
    c = lb["counts"]
    a(
        f"Clear counts. Paper family (28, mixed): gpt-oss {c['paper_mixed_28']['gptoss']}, Sonnet "
        f"{c['paper_mixed_28']['sonnet5']}, DeepSeek {c['paper_mixed_28']['dsv4pro']}, all three "
        f"{c['paper_mixed_28']['all_floors_iut']}. Twin-bearing family (23): gpt-oss "
        f"{c['twin_bearing']['gptoss']}, Sonnet {c['twin_bearing']['sonnet5']}, DeepSeek "
        f"{c['twin_bearing']['dsv4pro']}, all three {c['twin_bearing']['all_floors_iut']}, shift "
        f"{c['twin_bearing']['own_delta_above_top_floor_upper']}. Below the floor in the twin family: "
        f"gpt-oss {c['twin_bearing']['below']['gptoss']}, Sonnet {c['twin_bearing']['below']['sonnet5']} "
        f"(Nemotron). Twin-less family (5): below on every floor "
        f"{c['twinless']['below']['gptoss']} (gpt-audio-mini, Qwen3.5-Omni-Flash RT, VoiceChat 11B); "
        "gpt-audio and Qwen3.8-Omni-Flash RT indistinguishable."
    )
    a("")

    # ---------------- ladder
    lad = fl["ladder"]
    a("### 2.1 Which turn is scored: first turn (primary, D118) vs the follow-up rung")
    a("")
    a(
        "The runner administers a follow-up rung whenever an audio cell's first call is a "
        "ladder trigger (usually `ask_clarifying_question`): the caller replies with a "
        "variant-specific scripted line (its own words and audio) and a final call is made. "
        "The text twin is one call per item, scored on its first turn only, so scoring the "
        "audio on the follow-up turn credits clarify-then-act episodes the twin cannot reach. "
        f"Primary scoring is therefore the FIRST turn (D064/D118; this run: "
        f"`{lad['primary_mode']}`); the follow-up scoring is the sensitivity column. "
        "'Ladder part' = follow-up DiD minus first-turn DiD."
    )
    a("")
    rows = [
        [
            r["name"],
            F(r["did_primary"]),
            F(r["did_alt"]),
            F(r["ladder_contribution"]),
            f"{r['two_turn_cue_cells']}/{r['cue_cells']}",
        ]
        for r in lad["rows"]
    ]
    a(
        rc.md_table(
            [
                "system",
                f"DiD ({lad['primary_mode'].replace('_', '-')}, primary)",
                f"DiD ({lad['alt_mode'].replace('_', '-')})",
                "ladder part",
                "2-turn cue cells",
            ],
            rows,
        )
    )
    a("")
    ct = lad["leaderboard_alt"]["counts"]
    a(
        f"{lad['alt_mode'].replace('_', '-').capitalize()} scoring, clears: paper family gpt-oss "
        f"{ct['paper_mixed_28']['gptoss']}/28; "
        f"twin family gpt-oss {ct['twin_bearing']['gptoss']}, Sonnet {ct['twin_bearing']['sonnet5']}, "
        f"DeepSeek {ct['twin_bearing']['dsv4pro']}, all three {ct['twin_bearing']['all_floors_iut']}. "
        "Two-turn cue cells: "
        + ", ".join(f"{k} {v['two_turn_cue_cells']}" for k, v in lad["alt_stats_floors"].items())
        + "."
    )
    a("")

    # ---------------- task 3
    ne = fl["neutral"]
    a("## 3. Neutral cells: does audio-induced caution inflate cue-bearing credit?")
    a("")
    a(
        "Neutral DiD = the same diff-in-diff on clean variants (no cue). A system that grows "
        "cautious whenever audio is present fails clean calls its twin passes, so its neutral DiD "
        "is negative. Over-reaction = share of neutral cells where the twin passes and the audio "
        "fails (the cascade's own rate is "
        f"{F(ne['cascade_audio_induced_overreaction'], signed=False)}). Balanced = 1/2 cue DiD + 1/2 "
        "neutral DiD (rewards neutral gains, so it credits the D107 twin-under-acting artefact). "
        "Penalised = cue DiD minus any EXCESS over-reaction beyond the cascade's (never rewards a "
        "neutral gain). Holm across the 23 twin-bearing systems."
    )
    a("")
    rows = []
    for r in ne["rows"]:
        rows.append(
            [
                r["name"],
                F(r["cue_did"]),
                F(r["neutral_did"]),
                F(r["audio_induced_overreaction"], signed=False),
                F(r["audio_induced_overreaction_minus_cascade"]),
                F(r["balanced"]),
                FN(r["balanced_p_holm"]),
                F(r["penalised"]),
                FN(r["penalised_p_holm"]),
            ]
        )
    a(
        rc.md_table(
            [
                "system",
                "cue DiD",
                "neutral DiD",
                "over-reaction",
                "minus cascade",
                "balanced",
                "Holm",
                "penalised",
                "Holm",
            ],
            rows,
        )
    )
    a("")
    a(
        f"Clears: cue DiD {ne['cue_clears_twin_family']}/23, balanced {ne['balanced_clears']}/23, "
        f"penalised {ne['penalised_clears']}/23. Neutral DiD significantly negative: "
        f"{', '.join(ne['neutral_sig_negative']) or 'none'}; significantly positive (twin "
        f"under-acts on clean calls): {', '.join(ne['neutral_sig_positive']) or 'none'}."
    )
    a("")

    # ---------------- task 4
    lg = fl["legacy"]
    a("## 4. Without the 19 legacy LLM-drafted items")
    a("")
    a(
        f"{lg['legacy_in_bank']} legacy items have Gemini-TTS cells ({lg['cue_cells_dropped']} "
        "cue-bearing cells dropped). On the legacy cue cells alone the Sonnet floor is "
        f"{F(lg['floors_on_legacy_only']['sonnet5'])} while gpt-oss and DeepSeek are "
        f"{F(lg['floors_on_legacy_only']['gptoss'])} and {F(lg['floors_on_legacy_only']['dsv4pro'])}."
    )
    a("")
    rows = []
    for r in lg["leaderboard"]["rows"]:
        fam = "twin_bearing" if r["twin"] else "twinless"
        h = r["holm"][fam]
        rows.append(
            [
                r["name"] + ("" if r["twin"] else " (level)"),
                F(r["gptoss"]),
                FN(r["holm"]["paper_mixed_28"]["gptoss"]),
                FN(h["gptoss"]),
                FN(h["sonnet5"]),
                FN(h["dsv4pro"]),
                FN(h["all_floors_iut"]),
            ]
        )
    a(
        rc.md_table(
            [
                "system",
                "vs gpt-oss",
                "paper Holm",
                "family Holm",
                "Sonnet Holm",
                "DeepSeek Holm",
                "IUT Holm",
            ],
            rows,
        )
    )
    a("")
    cl = lg["leaderboard"]["counts"]
    a(
        f"Clears without legacy: paper family {cl['paper_mixed_28']['gptoss']}/28; twin family gpt-oss "
        f"{cl['twin_bearing']['gptoss']}, Sonnet {cl['twin_bearing']['sonnet5']}, DeepSeek "
        f"{cl['twin_bearing']['dsv4pro']}, all three {cl['twin_bearing']['all_floors_iut']}."
    )
    a("")
    sc = res["stacked"]["counts"]["twin_bearing"]
    a(
        "Every conservative choice stacked (first-turn scoring, no legacy items, must clear all three "
        f"floors, twin-bearing Holm): {sc['all_floors_iut']} systems."
    )
    a("")

    # ---------------- matrix
    mat = res["matrix"]
    n_twin = sum(1 for r in res["floor"]["leaderboard"]["rows"] if r["twin"])
    mode = res["floor"]["ladder"]["primary_mode"].replace("_", "-")
    a(
        f"## 5. Robustness matrix for the {len(mat)} published clears "
        f"(twin-bearing Holm family, {len(mat)} of {n_twin}; {mode} scoring)"
    )
    a("")
    names = list(mat[0]["checks"]) if mat else []
    short = [
        "Holm 28",
        "twin fam",
        "Sonnet",
        "DeepSeek",
        "all 3",
        "shift",
        "alt turn",
        "no legacy",
        "neutral ok",
        "penalised",
        "stacked",
    ]
    rows = []
    for m in mat:
        rows.append(
            [m["name"], F(m["did"])]
            + ["yes" if m["checks"][n] else "**no**" for n in names]
            + [f"{m['passed']}/{m['of']}"]
        )
    a(rc.md_table(["system", "DiD", *short, "passed"], rows))
    a("")
    a("Column key: " + "; ".join(f"{s} = {n}" for s, n in zip(short, names, strict=True)) + ".")
    a("")

    # ---------------- task 5/6 humans
    hu = res["human"]
    a("## 6. Human numbers with player clustering (task 6) and equivalence (task 5)")
    a("")
    a(
        f"{hu['players']} players, {hu['answers']} scored answers; the most prolific player gave "
        f"{hu['top_player_share']:.0%}. Item-only is the published CI; two-way adds player "
        "resampling (the pigeonhole bootstrap, conservative); player-only holds items fixed; LOPO = "
        "leave-one-player-out range."
    )
    a("")
    rows = []
    lab = {
        "human_act_given_heard": "humans P(right action | heard)",
        "human_lower_bound": "humans order-free lower bound",
        "models_pooled_act_given_heard": "27 models pooled P(act | heard)",
        "models_minus_human_act_given_heard": "models minus humans",
        "models_minus_human_bound": "models minus human lower bound",
        "human_heard_zero_credit": "humans: heard, zero credit",
    }
    for k, name in lab.items():
        v = hu["act_given_heard"][k]
        lo = v.get("leave_one_player_out") or {}
        rows.append(
            [
                name,
                F(v["item_only"]),
                F(v["player_only"]),
                F(v["two_way"]),
                "" if not lo else f"{lo['min']:+.2f} to {lo['max']:+.2f}",
            ]
        )
    v = hu["human_cue_credit_individual"]
    lo = v["leave_one_player_out"]
    rows.append(
        [
            "humans' cue-bearing credit (0.64)",
            F(v["item_only"], signed=False),
            F(v["player_only"], signed=False),
            F(v["two_way"], signed=False),
            f"{lo['min']:.2f} to {lo['max']:.2f}",
        ]
    )
    hm = res["harm"]["humans"]
    for k, name in (
        ("unsafe_execute", "humans' unsafe execution (18%)"),
        ("over_trigger", "humans' over-trigger (21%)"),
        ("unsafe_minus_cascade_paired", "humans minus cascade, unsafe (paired)"),
    ):
        v = hm[k]
        lo = v["leave_one_player_out"]
        rows.append(
            [
                name,
                F(v["item_only"]),
                F(v["player_only"]),
                F(v["two_way"]),
                f"{lo['min']:+.2f} to {lo['max']:+.2f}",
            ]
        )
    a(
        rc.md_table(
            ["quantity", "item-only (published)", "player-only", "two-way", "LOPO range"], rows
        )
    )
    a("")
    a("### 6.1 Equivalence: 'humans match the best model'")
    a("")
    a(
        "Same cells, model selection credit minus the human per-cell mean. TOST at alpha .05: "
        "equivalent iff the 90% CI is inside the margin. 'smallest margin' = the tightest margin the "
        "data would certify."
    )
    a("")
    rows = []
    for arm, blk in hu["same_cell_model_minus_human"].items():
        for sub in ("cue", "all"):
            v = blk[sub]
            t_i, t_2 = v["tost_item_only"], v["tost"]
            rows.append(
                [
                    rf._display(arm),
                    sub,
                    v["cells"],
                    F(v["item_only"]),
                    f"[{t_i[0.1]['ci90'][0]:+.3f}, {t_i[0.1]['ci90'][1]:+.3f}]",
                    "yes" if t_i[0.10]["equivalent"] else "no",
                    "yes" if t_i[0.05]["equivalent"] else "no",
                    f"±{t_i[0.1]['smallest_equivalence_margin']:.3f}",
                    f"[{t_2[0.1]['ci90'][0]:+.3f}, {t_2[0.1]['ci90'][1]:+.3f}]",
                    "yes" if t_2[0.10]["equivalent"] else "no",
                    f"±{t_2[0.1]['smallest_equivalence_margin']:.3f}",
                ]
            )
    a(
        rc.md_table(
            [
                "model",
                "cells",
                "n",
                "minus human (item CI)",
                "90% CI item",
                "equiv ±0.10",
                "equiv ±0.05",
                "smallest margin",
                "90% CI two-way",
                "equiv ±0.10",
                "smallest margin",
            ],
            rows,
        )
    )
    a("")
    rows = []
    for arm, v in hu["act_given_heard_per_arm_minus_human"].items():
        t_i, t_2 = v["tost_item_only"][0.1], v["tost"][0.1]
        rows.append(
            [
                rf._display(arm),
                F(v["item_only"]),
                F(v["two_way"]),
                f"[{t_i['ci90'][0]:+.3f}, {t_i['ci90'][1]:+.3f}]",
                "yes" if t_i["equivalent"] else "no",
                f"[{t_2['ci90'][0]:+.3f}, {t_2['ci90'][1]:+.3f}]",
                "yes" if t_2["equivalent"] else "no",
            ]
        )
    a("P(right action | heard), model minus humans (the §5.1 'level with humans' claim):")
    a("")
    a(
        rc.md_table(
            [
                "model",
                "item CI",
                "two-way CI",
                "90% item",
                "equiv ±0.10",
                "90% two-way",
                "equiv ±0.10",
            ],
            rows,
        )
    )
    a("")

    # ---------------- harm
    hz = res["harm"]
    mg = hz["mandate_vs_guidance"]
    a("## 7. Harm claims")
    a("")
    a("### 7.1 Mandate vs guidance (TOST)")
    a("")
    a(
        "Pooled over the 28 contestants, protective cells; violation = unsafe execution or missed "
        "duty. The two classes are DISJOINT item sets, so items are resampled within each class "
        "(stratified bootstrap)."
    )
    a("")
    rows = []
    for tag in ("all_items", "guidance_without_legacy"):
        v = mg[tag]
        ps = v["per_system_difference"]
        rows.append(
            [
                tag.replace("_", " "),
                f"{v['mandate_rate']:.3f} ({v['difference']['items_mandate']} items)",
                f"{v['guidance_rate']:.3f} ({v['difference']['items_guidance']} items)",
                F(v["difference"]),
                f"[{v['difference']['lo90']:+.3f}, {v['difference']['hi90']:+.3f}]",
                "yes" if v["tost"][0.10]["equivalent"] else "no",
                f"±{v['tost'][0.1]['smallest_equivalence_margin']:.3f}",
                f"{ps['median']:+.3f} ({ps['systems_mandate_higher']}/{ps['n']} higher on mandate)",
            ]
        )
    a(
        rc.md_table(
            [
                "guidance set",
                "mandate rate",
                "guidance rate",
                "mandate - guidance",
                "90% CI",
                "equiv ±0.10",
                "smallest margin",
                "per-system median diff",
            ],
            rows,
        )
    )
    a("")
    a(f"Legacy items by norm class: {mg['legacy_norms']}.")
    a("")
    u = hz["unsafe_vs_cascade"]
    a("### 7.2 'Worse than the deaf cascade on unsafe execution', Holm across 28")
    a("")
    rows = [
        [
            r["name"],
            F(r["unsafe_minus_cascade"]),
            FN(r["unsafe_minus_cascade"]["p"]),
            FN(r["p_holm"]),
        ]
        for r in u["rows"][:6]
    ]
    a(rc.md_table(["system", "unsafe minus cascade (paired)", "raw p", "Holm p (28)"], rows))
    a("")
    a(
        f"Worse than the cascade: unadjusted {len(u['worse_unadjusted'])} "
        f"({', '.join(u['worse_unadjusted'])}); after Holm {len(u['worse_holm'])} "
        f"({', '.join(u['worse_holm'])}). Better than the cascade: unadjusted "
        f"{u['better_unadjusted']}/28, after Holm {u['better_holm']}/28."
    )
    a("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
