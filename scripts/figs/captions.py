# ruff: noqa: E501, RUF001  (caption prose f-strings; the typographic minus is intended)
"""Write paper/figures/captions.md: one caption per figure, first sentence = the
figure's message. Every number is formatted from the same JSON the figure reads.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from statistics import median

import data as D
from style import OUT


def pc(x: float) -> str:
    return f"{x * 100:.0f}%"


def ci(e: dict, nd: int = 2, pct: bool = False) -> str:
    f = (lambda v: f"{v * 100:.0f}%") if pct else (lambda v: f"{v:.{nd}f}")
    return f"{f(e['mean'])} [{f(e['lo'])}, {f(e['hi'])}]"


def s_ci(e: dict) -> str:
    return f"{e['mean']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]".replace("-", "−")


def s_ci_hu(e: dict) -> str:
    """s_ci with round-half-up on the printed value (0.045 -> +0.05), matching the
    paper's text for the stratified heard-emotion gap."""

    def f(v: float) -> str:
        return f"{Decimal(str(v)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):+.2f}"

    return f"{f(e['mean'])} [{f(e['lo'])}, {f(e['hi'])}]".replace("-", "−")


def _join(xs: list[str]) -> str:
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def build() -> str:
    at, lb = D.atlas(), D.leaderboard()
    ba = at["by_axis"]
    axes = list(D.AXES)
    hum_above = sum(ba[a]["humans"]["mean"] > ba[a]["cascade"]["mean"] for a in axes)
    med_near = sum(
        abs(ba[a]["median_system_credit"] - ba[a]["cascade"]["mean"]) <= 0.06 for a in axes
    )
    n_ax_cells = sum(ba[a]["cells"] for a in axes)
    cue_items: dict[str, set[str]] = {}
    for pat in at["patterns"]:
        cue_items.setdefault(pat["item"], set()).add(pat["axis"])
    n_cue_items = len(cue_items)
    n_two_axes = sum(len(v) > 1 for v in cue_items.values())
    n_ax_items = sum(ba[a]["items"] for a in [*axes, "channel"])

    # Fig 1b draws ONE fixed system on every axis (atlas "reference" = gemini-3.7-flash,
    # the overall cue-bearing leader that Section 4.1 compares with the players).
    # Ahead / level / behind the players' marker, with a 0.02 tolerance for "level".
    def _fx(a: str) -> float:
        return ba[a]["reference"]["credit"]["mean"]

    fx_ahead = [D.AXIS_LABEL[a] for a in axes if _fx(a) > ba[a]["humans"]["mean"] + 0.02]
    fx_level = [D.AXIS_LABEL[a] for a in axes if abs(_fx(a) - ba[a]["humans"]["mean"]) <= 0.02]
    fx_behind = [D.AXIS_LABEL[a] for a in axes if _fx(a) < ba[a]["humans"]["mean"] - 0.02]
    # per-axis gaps and the clean-vs-protective contrast, two-way (strengthen.json q2)
    st = D.load(D.INS / "strengthen.json")["q2"]["gemini37or"]
    n_axis_sig = sum(
        (g["gap"]["lo"] > 0 or g["gap"]["hi"] < 0) for g in st["by_axis"]["groups"].values()
    )
    st_cmp = st["prespecified"]["clean_minus_protective"]
    chan = ba["channel"]
    cnt = at["counts"]
    c = lb["counts"]
    rows = D.contestants()
    best = max((r for r in rows), key=lambda r: r["cue_credit"]["mean"])
    assert all(ba[a]["reference"]["label"] == best["label"] for a in axes), best["label"]
    best_minus_players = D.load(D.INS / "players-band.json")["model_minus_players_gemini_cue"][
        best["label"]
    ]["two_way"]
    rob = {r["label"]: r for r in D.robustness()["arm_vs_cascade"]}
    lofo_w = max(
        rob[r["label"]]["leave_one_family_out"]["max_mean"]
        - rob[r["label"]]["leave_one_family_out"]["min_mean"]
        for r in rows
    )
    fl = D.robust2()["floor"]["floors"]
    E = D.review2()["E30_identical"]
    hb = D.derived("heard_by_kind")
    hg = D.harm()["asymmetry"]
    groups = hg["groups"]
    f4h = D.derived("harm_frontier4")
    hh = D.robust2()["harm"]["humans"]
    n_probe = len(hb["arms"])
    ex = {a: D.exemplar(a) for a in axes}

    # ---- numbers for the new main figures
    f4names = ", ".join(
        D.short(next(r["name"] for r in rows if r["label"] == a)) for a in D.review2()["frontier4"]
    )
    all_ = groups["all contestants"]
    casc = groups["cascade (words only)"]
    n_all_sys = hg["n_contestants"]
    cn_rows = __import__("fig5_facts_feelings").rows()

    def _rng(key: str) -> str:
        v = [c["by_axis"][key]["credit"]["mean"] for _l, c in cn_rows]
        return f"{min(v):.2f}–{max(v):.2f}"

    g_own = cn_rows[0][1]["by_axis"]
    nf = D.notefull()
    nb = __import__("fig5_facts_feelings").note_rows()
    nb_g = dict(nb)
    g_a = nb_g["gemini-3.7-flash \u00b7 own audio"]
    gains = [g["emotion"]["description_minus_label"]["mean"] for _l, g in nb]
    side = [g[k]["description_minus_label"]["mean"] for _l, g in nb for k in ("other", "neutral")]

    def sg(x: float) -> str:
        return f"{x:+.2f}".replace("-", "−")

    gem_rows = [r for r in nf["description_note"]["rows"] if r["model"] == "gemini-3.7-flash"]
    gem_gap = " and ".join(
        f"{r['gap_other_minus_emotion']['label']:.2f} to {r['gap_other_minus_emotion']['description']:.2f} ({'own audio' if r['path'] == 'audio' else 'exact transcript'})"
        for r in sorted(gem_rows, key=lambda r: r["path"])
    )
    casc_lvl = lb["cascade"]["cue_credit"]

    out = [
        "# VoxParity figure captions (final spine)",
        "",
        "Generated by `scripts/figs/captions.py` from the same JSON the figures read. "
        "First sentence = the figure's message. Scoring: first turn throughout. "
        "Headings use the paper's numbering in reading order; file names keep their build names.",
        "",
        "## Reading order (for the writer)",
        "",
        "| paper number | file | section | width |",
        "|---|---|---|---|",
        "| Figure 1 | `fig1_overview.pdf` | §1 Introduction | full (6.5 in) |",
        "| Figure 2 | `fig5_facts_feelings.pdf` | §5.3–5.4 told the cue; description vs label only | full |",
        "| Figure 3 | `fig2_leaderboard.pdf` | §6 the null test leaderboard | full |",
        "| Figure A1 | `figA1_robustness.pdf` | appendix, robustness of the leaderboard | full |",
        "| Figure A2 | `figA4_sectors.pdf` | appendix, sectors | full |",
        "| Figure A3 | `figA_parity_by_cue.pdf` | appendix, companion to A2 (old Figure 1b) | half (3.15 in) |",
        "| Figure A4 | `fig4_asymmetry.pdf` | appendix, error direction by group (§4 now uses Table 3) | full |",
        "| Figure A5 | `fig3_heard.pdf` | appendix, heard emotion vs other heard cues (§5.1) | full |",
        "| Figure A6 | `figA2_taxonomy.pdf` | appendix, failure taxonomy | full |",
        "| Figure A7 | `figA6_cue_note.pdf` | appendix, cue note vs sham by group | half |",
        "| Figure A8 | `figA3_test_information.pdf` | appendix, test information | half |",
        "",
        "Superseded, still built so the v4 PAPER.md compiles: `fig1_glance.pdf` (replaced by `fig1_overview.pdf`). "
        "Not in the paper: `figA5_realtime.pdf` (Table A3 carries it). "
        "Interspeech column variants: `is_fig1_axes.pdf`, `is_fig2_leaderboard.pdf`, `is_fig3_heard.pdf`, `is_fig4_asymmetry.pdf` (no column variant of Figure 1b or Figure 2 yet).",
        "",
    ]

    out += [
        "## Figure 1 — `fig1_overview.pdf` (full width)",
        "",
        f"**VoxParity at a glance, and its headline: all {n_all_sys} systems err toward the words when the audio calls for protection.** "
        f"(a) The {cnt['items']} scenarios by cue type and sector (dot area = scenarios), each cue type with one example of what the audio adds and the action the rule then requires. "
        f"(b) Each system's rate of carrying out the routine request on protective calls (y) against its rate of over-reacting on clean calls (x); marker shape gives serving mode. "
        f"Every system sits above the dotted equal-rates line: pooled, {pc(all_['unsafe_execute']['mean'])} [{pc(all_['unsafe_execute']['lo'])}, {pc(all_['unsafe_execute']['hi'])}] against {pc(all_['over_trigger']['mean'])} [{pc(all_['over_trigger']['lo'])}, {pc(all_['over_trigger']['hi'])}], and for the four leading systems {pc(f4h['unsafe_execute']['mean'])} against {pc(f4h['over_trigger']['mean'])}. "
        f"Reference points with 95% intervals: the words-only cascade and the volunteer players, the only point on or across the line.",
        "",
    ]
    cmp_ = D.notefull()["stated_rule"]["composition_nonlegacy_cue_cells"]
    hs = D.notefull()["heard_split"]["paper"]
    out += [
        "## Figure A5 — `fig3_heard.pdf` (full width); Interspeech: `is_fig3_heard.pdf` (summary rows, column width)",
        "",
        f"**Pooled over rule modes, systems act on heard emotion less often than on other heard cues; within rule mode only gemini-3.7-flash's gap is significant.** "
        f"Share of right actions when the system's own perception question shows it missed the cue (hollow) and when it heard it (filled), split into emotional delivery and all other cues. "
        f"The four leading systems pooled (bold): {ci(E['frontier4.emo.raw'])} on heard emotion against {ci(E['frontier4.non.raw'])} on other heard cues; all {n_probe} systems with a perception answer pooled: {ci(E['pooled.emo.raw'])} against {ci(E['pooled.non.raw'])}. "
        f"Emotional cues come more often from scenarios that leave the rule unstated ({cmp_['feelings_emotion_and_sarcasm']['no_rule']} of {cmp_['feelings_emotion_and_sarcasm']['total']} against {cmp_['facts']['no_rule']} of {cmp_['facts']['total']} environmental, second-voice, speaker and masked-word cues); stratified by stated rule, the gap over all systems and the whole bank is {s_ci_hu(hs['all_27']['stratified_size_weighted'])} (pooled {s_ci_hu(hs['all_27']['pooled'])}), while gemini-3.7-flash's stays at {s_ci_hu(hs['gemini-3.7-flash']['stratified_size_weighted'])}. "
        f"Players ({ci(E['humans.emo.raw'])} and {ci(E['humans.non.raw'])}; they saw the scenario and the menu but not the stated policy) are drawn as the vertical band for reference; the comparison with them is suggestive only (Appendix A.8.4). "
        f"Cells are the protocol-grounded calls the players also answered. Points drawn faint rest on fewer than {__import__('fig3_heard').MIN_N} heard or missed calls. Shape = serving mode.",
        "",
    ]
    out += [
        "## Figure 2 — `fig5_facts_feelings.pdf` (full width)",
        "",
        f"**Told the cue, agents act on the facts of a call more readily than on the caller's state; describing the voice narrows the gap without closing it.** "
        f"(a) Told the cue in one line (for emotional delivery, a single label), gemini-3.7-flash on its own audio reaches {g_own['environmental']['credit']['mean']:.2f} on environmental sound and {g_own['second voice']['credit']['mean']:.2f} on a second voice but {ci(g_own['emotion']['credit'])} on emotional delivery; "
        f"across all six conditions (its own audio, the exact transcript, the Whisper transcript, and three text-only models on the Whisper transcript) emotion stays lowest, at {_rng('emotion')}, against {_rng('environmental')} and {_rng('second voice')}. "
        f"Hollow = the same model and input without the note; lines = 95% CIs, clustered by scenario. A sham note leaves credit unchanged (Figure A7). "
        f"(b) Credit on emotional-delivery calls when the cue note describes the voice as rendered (filled) and when it gives the emotion label only (hollow), on identical calls: "
        f"the description adds {sg(min(gains))} to {sg(max(gains))} across the five model and input rows (gemini-3.7-flash on its own audio {ci(g_a['emotion']['label'])} to {ci(g_a['emotion']['description'])}), "
        f"while credit on other cues and on clean calls moves by {sg(min(side))} to {sg(max(side))}. "
        f"Emotion stays below the other cues (tick) in every row; for gemini-3.7-flash the gap narrows from {gem_gap}. "
        f"Emotional delivery here excludes sarcasm (n = {g_a['emotion']['label']['n']} calls); on audio, the two notes were served through different routes (gemini-3.7-flash: OpenRouter and the Gemini API; Qwen3.8-Omni: its label-only run mixes OpenRouter and DashScope). "
        "Players are not drawn: neither panel uses their calls.",
        "",
    ]
    out += [
        "## Figure 3 — `fig2_leaderboard.pdf` (full width); Interspeech: `is_fig2_leaderboard.pdf` (panel b, column width)",
        "",
        f"**{c['twin_bearing_clear']} of {c['twin_bearing']} systems that can be tested against their own transcript act on the audio beyond the words-only null; the median system does not, and one of the seven production realtime agents with a transcript path does.** "
        f"(a) Credit on the calls with an audible cue (95% CI), against the words-only cascade ({ci(casc_lvl)}, dashed with grey band) and the players ({ci(D.players_band()[0])} on the calls they answered; solid line, edged band). "
        f"(b) Gain over the null: how much a system's actions change between its audio and its own transcript, minus the same change for the cascade; for the {c['twinless']} systems with no transcript input (lower block), audio credit minus the cascade's. "
        f"Filled = clears the null after Holm correction; hollow = does not; ▼ = significantly below the null. The light-blue underlay is the range when each scenario family is left out in turn; the grey band spans the null under two other text models. Shape = serving mode.",
        "",
    ]
    out += [
        "## Figure A4 — `fig4_asymmetry.pdf` (full width); Interspeech: `is_fig4_asymmetry.pdf` (column width)",
        "",
        f"**Every group of systems errs toward the words, and the words-only cascade most of all.** "
        f"Over-reaction on clean calls (left) and unsafe execution on protective calls (right): "
        f"the four leading systems {ci(f4h['over_trigger'], pct=True)} / {ci(f4h['unsafe_execute'], pct=True)}; "
        f"all {len(all_['arms'])} systems {ci(all_['over_trigger'], pct=True)} / {ci(all_['unsafe_execute'], pct=True)}; "
        f"words-only cascade {ci(casc['over_trigger'], pct=True)} / {ci(casc['unsafe_execute'], pct=True)}; "
        f"players, for reference, {ci(hh['over_trigger']['two_way'], pct=True)} / {ci(hh['unsafe_execute']['two_way'], pct=True)} (item and player resampled). "
        f"Group rates average member systems per call, bootstrapped over scenarios. Figure 1b gives each system's own operating point.",
        "",
    ]
    out += [
        "## Figure A3 — `figA_parity_by_cue.pdf` (half width); also Interspeech `is_fig1_axes.pdf`",
        "",
        f"**By cue type, the median system sits near the words-only cascade while the players sit above it on {hum_above} of 7.** "
        f"Credit on calls with an audible cue, per cue type: players (95% CI), {D.short(best['name'])} (the highest-credit system, fixed on every row), the median of the {c['contestants']} systems, and the words-only cascade (dashed; grey span from cascade to players). "
        f"The median system is within 0.06 of the cascade on {med_near} of 7 cue types. {D.short(best['name'])} is ahead of the players on {_join(fx_ahead)}, level on {_join(fx_level)} and behind on {_join(fx_behind)}; "
        f"{'no single cue-type gap excludes zero with players resampled' if n_axis_sig == 0 else f'{n_axis_sig} of 7 gaps exclude zero'}, and the gains and losses offset (overall {s_ci(best_minus_players)}). Rows with fewer than ten calls are drawn lighter.",
        "",
    ]
    _ = (ex, chan, n_cue_items, n_two_axes, n_ax_items, n_ax_cells, st_cmp, lofo_w, fl, f4names)
    # appendix
    fl_lb = {r["label"]: r for r in D.robust2()["floor"]["leaderboard"]["rows"]}
    counts = D.robust2()["floor"]["leaderboard"]["counts"]["twin_bearing"]
    nl_counts = D.robust2()["floor"]["legacy"]["leaderboard"]["counts"]["twin_bearing"]
    _ = fl_lb
    out += [
        "## Figure A1 — `figA1_robustness.pdf`",
        "",
        f"**The leaderboard does not depend on which text model defines the null or on the legacy items.** "
        f"Gain over the null per system (95% CI; filled = clears after Holm in its family) against the gpt-oss cascade (paper, {counts['gptoss']} of 23 testable systems clear), a Sonnet 5 replay ({counts['sonnet5']}), a DeepSeek-V4-Pro replay ({counts['dsv4pro']}), and gpt-oss without the 19 legacy LLM-drafted items ({nl_counts['gptoss']}). Rows as in Figure 3.",
        "",
    ]
    tx = {r["label"]: r for r in D.taxonomy()}
    shares = [
        tx[r["label"]]["all_cue_bearing"]["counts"].get("perceived, not acted", 0)
        / tx[r["label"]]["all_cue_bearing"]["n"]
        for r in rows
        if tx[r["label"]]["perception_measured"]
    ]
    no_opt = D.notefull()["probe"]["taxonomy_cue_cells"]
    pna_gt = sum(
        tx[r["label"]]["all_cue_bearing"]["counts"].get("perceived, not acted", 0)
        > no_opt[r["label"]]["not_heard"]
        for r in rows
        if tx[r["label"]]["perception_measured"]
    )
    big_no = sorted(
        (
            (no_opt[r["label"]]["no_option_named"], D.short(r["name"]))
            for r in rows
            if tx[r["label"]]["perception_measured"]
        ),
        reverse=True,
    )[:3]
    out += [
        "## Figure A6 — `figA2_taxonomy.pdf`",
        "",
        f"**Every system with a perception answer leaves {pc(min(shares))} to {pc(max(shares))} of the calls with an audible cue heard and not acted on; for {pna_gt} of {len(shares)}, cues heard and not acted on outnumber cues not heard.** "
        f"Share of the {tx[rows[0]['label']]['all_cue_bearing']['n']} calls with an audible cue per system: correct, heard but not acted on, heard and acted wrongly, not heard (perception from the system's own answer to a perception question), and calls whose perception answer names none of the options, counted as missing rather than as not heard ({_join([f'{n} for {m}' for n, m in big_no])}). Sorted by the heard-not-acted share (right margin; median {pc(median(shares))}). {len(shares)} systems with a perception answer.",
        "",
    ]
    ps = D.psych()["irt"]["summary"]
    out += [
        "## Figure A8 — `figA3_test_information.pdf` (half width)",
        "",
        f"**Calls with an audible cue carry the bank's information where the systems and players are.** "
        f"2PL test information over ability (strict pass): all cells (peak at θ {ps['test_info_peak_theta']:+.1f}), calls with an audible cue (peak {ps['cue_info_peak_theta']:+.1f}) and neutral cells (peak {ps['neutral_info_peak_theta']:+.1f}). Top strip: each system's θ (shape = serving mode). Dotted line: players, from the selection-basis refit (approximate). Marginal reliability over systems {ps['marginal_reliability_contestants']:.3f}.",
        "",
    ]
    small = [D.SECTOR_SHORT[s] for s, v in at["by_sector"].items() if v["small_n"]]
    sb = D.derived("sector_by_system")
    sysl = [x for x in sb["systems"] if x in {r["label"] for r in rows}]
    leaders = set()
    for x in sb["sectors"]:
        row = sb["credit"][x]
        top = max(row[y] for y in sysl if row[y] is not None)
        leaders |= {y for y in sysl if row[y] == top}
    hum_gt = sum(
        (sb["credit"][x]["humans"] or 0) > (sb["credit"][x]["cascadeopen"] or 0)
        for x in sb["sectors"]
    )
    out += [
        "## Figure A2 — `figA4_sectors.pdf`",
        "",
        f"**Players out-score the words-only cascade in {hum_gt} of {len(sb['sectors'])} sectors, and the leading system changes from sector to sector ({len(leaders)} different leaders).** "
        f"Mean credit on calls with an audible cue by sector (columns, count of such calls in brackets, same order as Figure 1) for players (tool-selection credit on the subset of cells they answered, as few as {min(sb['credit'][x]['humans_cells'] for x in sb['sectors'])} in a sector), the words-only cascade and the {c['contestants']} systems (ordered by overall credit on these calls). "
        f"‡ = fewer than 5 scenarios ({', '.join(small)}): read with care. Values: paper/figures/data/sector_by_system.json.",
        "",
    ]
    cd = D.conduct()
    neg = sum(p["credit_rt_minus_file_cue_bearing"]["hi"] < 0 for p in cd["pairs"])
    out += [
        "## Not in the paper — `figA5_realtime.pdf` (Table A3 carries these contrasts)",
        "",
        f"**Within a model family, realtime serving lowers credit on calls with an audible cue in {neg} of {len(cd['pairs'])} pairs, largely where it makes the system act less.** "
        f"Realtime minus file-mode serving on identical cells (95% CI) for {len(cd['pairs'])} within-family pairs: credit on calls with an audible cue ({neg} of {len(cd['pairs'])} significantly lower), the rate of making any tool call, and probe accuracy. * = the two sides are different model generations, so the contrast is not within-model.",
        "",
    ]
    conds = D.experiments()["note_controls"]["conditions"]
    g = next(
        x
        for x in conds
        if x["model"] == "google/gemini-3.7-flash"
        and x["path"] == "audio"
        and x["note"] == "oracle"
    )
    out += [
        "## Figure A7 — `figA6_cue_note.pdf` (half width)",
        "",
        f"**Naming the cue in a one-line label nearly solves scene, speaker and masked-word cells but lifts emotional delivery only to about 0.6.** "
        f"Cue-bearing credit with a note that names the cue (filled) vs a sham note (hollow), by axis, for gemini-3.7-flash on its own audio and for gpt-oss-120b on Whisper transcripts. With the note, gemini-3.7-flash reaches {ci(g['by_axis']['emotion']['credit'])} on emotion cells and {ci(g['by_axis']['scene']['credit'])} on scene and second-voice cells. "
        f"For emotional delivery this note is the label only; describing the voice as rendered lifts gemini-3.7-flash on its own audio to {ci(g_a['emotion']['description'])} on emotional-delivery calls, sarcasm excluded (Figure 2b). "
        f"Axis groups follow experiments.json (scene includes second voice).",
        "",
    ]
    out += [
        "## Not drawn as figures",
        "",
        "- Cost/latency Pareto and risk-vs-accuracy scatters: labels cannot be placed cleanly at print size (15+ points in one cluster); report them as tables (credit, $/cell, median latency; risk per 100 calls, accuracy) from docs/results/final/paper_pareto.json and docs/insights/harm.json.",
        "",
    ]
    # emit sections in the paper's reading order (Figure 1..3, then A1..A8, then the rest)
    text = "\n".join(out)
    head, *secs = text.split("\n## ")
    order = ["Figure 1 ", "Figure 2 ", "Figure 3 "] + [f"Figure A{i} " for i in range(1, 9)]

    def rank(sec: str) -> int:
        if sec.startswith("Reading order"):
            return -1
        return next((i for i, p in enumerate(order) if sec.startswith(p)), len(order))

    secs.sort(key=rank)
    return "\n## ".join([head, *secs])


def main() -> None:
    (OUT / "captions.md").write_text(build())
    print("wrote", OUT / "captions.md")


if __name__ == "__main__":
    main()
