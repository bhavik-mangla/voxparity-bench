# ruff: noqa: RUF001  (typographic minus signs in tick labels are intended)
"""Appendix figures (arXiv), redesigned per the v3 figure audit.

figA1_robustness       gain over the null under three text-LLM nulls and without the
                       legacy items (small multiples, shared rows = Fig. 2's order)
figA2_taxonomy         where the cue is lost, per system: 100% bars, no hatching,
                       sorted by the "heard, not acted on" share; a probe answer that
                       names no option is its own segment (counted as missing, not as
                       "not heard"; docs/insights/notefull.json)
figA3_test_information 2PL test information; direct labels; systems as a rug by mode
figA4_sectors          cue-bearing credit, systems x sectors (transposed), references
                       in their own block, no in-cell numbers
figA5_realtime         realtime minus file serving within a model family (7 pairs)
figA6_cue_note         credit when the cue is named in a note vs a sham note, by axis
"""

from __future__ import annotations

from typing import Any

import data as D
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from style import (
    ARXIV_FULL,
    ARXIV_HALF,
    BLUE,
    BLUE_LIGHT,
    FS,
    FS_TICK,
    GRID,
    INK,
    INK2,
    MODE_MARKER,
    MUTED_FILL,
    MUTED_TXT,
    NULL,
    NULL_BAND,
    ORANGE,
    ORANGE_LIGHT,
    SURFACE,
    VIOLET,
    fig_at,
    null_line,
    save,
)

SEQ = LinearSegmentedColormap.from_list(
    "vx_blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def _order() -> list[dict[str, Any]]:
    """Fig. 2's row order: twin-bearing by DiD, then the twin-less block."""
    rows = D.contestants()
    tw = sorted([r for r in rows if r["twin"]], key=lambda r: -r["vs_cascade"]["mean"])
    tl = sorted([r for r in rows if not r["twin"]], key=lambda r: -r["vs_cascade"]["mean"])
    return tw + tl


# ------------------------------------------------------------------ A1
def a1_robustness() -> list:
    order = _order()
    fl = {r["label"]: r for r in D.robust2()["floor"]["leaderboard"]["rows"]}
    nl = {r["label"]: r for r in D.robust2()["floor"]["legacy"]["leaderboard"]["rows"]}
    specs = [
        ("gpt-oss null (paper)", fl, "gptoss"),
        ("Sonnet 5 null", fl, "sonnet5"),
        ("DeepSeek-V4-Pro null", fl, "dsv4pro"),
        ("gpt-oss, no legacy items", nl, "gptoss"),
    ]
    n_tw = sum(r["twin"] for r in order)
    ys = [i if i < n_tw else i + 1.2 for i in range(len(order))]
    W, rowh, top, bottom = ARXIV_FULL, 0.118, 0.42, 0.4
    H = top + bottom + (ys[-1] + 1) * rowh
    fig = fig_at(W, H)
    namew, gap = 1.5, 0.1
    pw = (W - namew - 0.08 - 3 * gap) / 4
    for k, (title, src, key) in enumerate(specs):
        ax = fig.add_axes(
            ((namew + k * (pw + gap)) / W, bottom / H, pw / W, (ys[-1] + 1) * rowh / H)
        )
        ax.set_ylim(ys[-1] + 0.6, -0.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-0.32, 0.4)
        ax.set_xticks([-0.2, 0, 0.2], ["−.2", "0", ".2"])
        null_line(ax, 0.0)
        for r, y in zip(order, ys, strict=True):
            e = src[r["label"]][key]
            fam = "twin_bearing" if r["twin"] else "paper_mixed_28"
            clr = bool(src[r["label"]]["clears"][fam].get(key)) and e["mean"] > 0
            ax.plot([e["lo"], e["hi"]], [y, y], color=BLUE, lw=0.8)
            ax.plot(
                [e["mean"]],
                [y],
                ls="",
                marker=MODE_MARKER[r["mode"]],
                ms=3.4,
                mfc=BLUE if clr else SURFACE,
                mec=BLUE,
                mew=0.8,
                clip_on=False,
            )
        n_clear = sum(
            bool(src[r["label"]]["clears"]["twin_bearing"].get(key)) for r in order if r["twin"]
        )
        ax.text(
            -0.32,
            -1.1,
            f"{n_clear} of {n_tw} clear",
            fontsize=FS_TICK,
            color=INK2,
            va="bottom",
        )
        ax.text(-0.32, -2.5, title, fontsize=FS_TICK, fontweight="bold", va="bottom")
        ax.set_xlabel("gain over the null")
        if k == 0:
            for r, y in zip(order, ys, strict=True):
                ax.text(
                    -0.03,
                    y,
                    D.short(r["name"]),
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=FS_TICK,
                )
            ax.text(
                -0.03,
                (ys[n_tw - 1] + ys[n_tw]) / 2,
                "no transcript path:",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=FS_TICK,
                color=MUTED_TXT,
                style="italic",
            )
    return save(fig, "figA1_robustness")


# ------------------------------------------------------------------ A2
def a2_taxonomy() -> list:
    tx = {r["label"]: r for r in D.taxonomy()}
    rows = [r for r in D.contestants() if tx[r["label"]]["perception_measured"]]
    no_opt = D.notefull()["probe"]["taxonomy_cue_cells"]
    segs = [
        ("correct", "correct", BLUE),
        ("heard, not acted on", "perceived, not acted", ORANGE),
        ("heard, acted wrongly", "perceived, acted wrong", ORANGE_LIGHT),
        ("not heard", "not heard", MUTED_FILL),
        ("probe answer names no option", "no option named", GRID),
    ]

    def share(lab: str, key: str) -> float:
        c = tx[lab]["all_cue_bearing"]
        if key in ("not heard", "no option named"):
            s = no_opt[lab]
            assert s["not_heard"] + s["no_option_named"] == c["counts"]["not perceived"], lab
            return s["not_heard" if key == "not heard" else "no_option_named"] / c["n"]
        return c["counts"].get(key, 0) / c["n"]

    rows.sort(key=lambda r: -share(r["label"], "perceived, not acted"))
    n = len(rows)
    W, rowh, top, bottom = ARXIV_FULL, 0.13, 0.42, 0.38
    H = top + bottom + n * rowh
    fig = fig_at(W, H)
    namew = 1.5
    ax = fig.add_axes((namew / W, bottom / H, (W - namew - 0.5) / W, n * rowh / H))
    for i, r in enumerate(rows):
        left = 0.0
        for _lab, key, col in segs:
            v = share(r["label"], key)
            ax.barh(i, v, left=left, height=0.72, color=col, edgecolor=SURFACE, lw=1.0)
            left += v
        pna = share(r["label"], "perceived, not acted")
        ax.text(1.01, i, f"{pna * 100:.0f}%", va="center", fontsize=FS_TICK, color=INK)
        ax.text(
            -0.01,
            i,
            D.short(r["name"]),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
        )
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("share of cue-bearing cells")
    ax.text(
        1.01, -0.95, "heard,\nnot acted", fontsize=FS_TICK, color=INK2, va="bottom", linespacing=1.0
    )
    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(color=c, label=lab) for lab, _k, c in segs],
        loc="upper left",
        bbox_to_anchor=(namew / W - 0.01, 1.0),
        ncol=5,
        fontsize=FS_TICK,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.2,
        borderaxespad=0.2,
    )
    return save(fig, "figA2_taxonomy")


# ------------------------------------------------------------------ A3
def a3_test_information() -> list:
    ps = D.psych()["irt"]
    t = ps["test_information"]
    meth = D.psych()["method"]
    cue_n = meth["cue_cells"]
    all_n = meth["cells"]
    W, H = ARXIV_HALF, 2.3
    fig = fig_at(W, H)
    ax = fig.add_axes((0.46 / W, 0.4 / H, (W - 0.46 - 0.62) / W, (H - 0.4 - 0.42) / H))
    g = t["grid"]
    curves = [
        ("all", f"all cells ({all_n})", INK, "-"),
        ("cue", f"cue-bearing ({cue_n})", BLUE, "-"),
        ("neutral", f"neutral ({all_n - cue_n})", NULL, (0, (3, 2))),
    ]
    ymax = max(t["all"]) * 1.08
    for key, lab, col, ls in curves:
        ax.plot(g, t[key], color=col, lw=1.3, ls=ls)
        ax.text(
            g[-1] + 0.08, t[key][-1], lab.split(" (")[0], fontsize=FS_TICK, va="center", color=INK
        )
    h = ps["summary"]["human_theta_selection_basis"]["theta"]
    ax.axvline(h, color=INK, lw=0.7, ls=(0, (1, 1.5)))
    ax.text(h + 0.08, ymax * 0.97, "players", fontsize=FS_TICK, ha="left", va="top")
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(0, ymax)
    ax.set_ylabel("test information")
    ax.set_xlabel("ability θ (2PL)")
    # rug: contestants by serving mode, two jittered rows
    rug = fig.add_axes((0.46 / W, (H - 0.3) / H, (W - 0.46 - 0.62) / W, 0.2 / H), sharex=ax)
    rug.set_axis_off()
    rows = [r for r in ps["theta_rows"] if r["role"] == "contestant"]
    rows.sort(key=lambda r: r["theta"])
    for i, r in enumerate(rows):
        rug.plot(
            [r["theta"]],
            [i % 2],
            ls="",
            marker=MODE_MARKER[r["mode"]],
            ms=3.2,
            mfc=SURFACE,
            mec=BLUE,
            mew=0.7,
            clip_on=False,
        )
    rug.set_ylim(-0.7, 1.7)
    fig.text(
        (W - 0.6) / W,
        (H - 0.2) / H,
        f"{len(rows)} systems",
        fontsize=FS_TICK,
        va="center",
        color=INK2,
    )
    return save(fig, "figA3_test_information")


# ------------------------------------------------------------------ A4
def a4_sectors() -> list:
    sb = D.derived("sector_by_system")
    at = D.atlas()
    lb = {r["label"]: r for r in D.leaderboard()["rows"]}
    # same sector order as Fig. 1: by number of scenarios
    sectors = sorted(sb["sectors"], key=lambda s: -at["counts"]["sectors"][s])
    systems = [s for s in sb["systems"] if s in lb and lb[s]["role"] == "contestant"]
    systems.sort(key=lambda s: -lb[s]["cue_credit"]["mean"])
    refs = [("players", "humans"), ("words-only cascade", "cascadeopen")]
    nrow = len(systems) + len(refs)
    W, rowh, top, bottom = ARXIV_FULL, 0.125, 0.92, 0.12
    H = top + bottom + (nrow + 0.6) * rowh
    fig = fig_at(W, H)
    namew = 1.55
    aw = W - namew - 0.75
    ax = fig.add_axes((namew / W, bottom / H, aw / W, (nrow + 0.6) * rowh / H))
    ys = [i for i in range(len(refs))] + [len(refs) + 0.6 + i for i in range(len(systems))]
    labels = [lab for lab, _ in refs] + [D.short(lb[s]["name"]) for s in systems]
    keys = [k for _, k in refs] + systems
    for y, key, lab in zip(ys, keys, labels, strict=True):
        for j, sec in enumerate(sectors):
            v = sb["credit"][sec].get(key)
            if v is None:
                continue
            ax.add_patch(
                __import__("matplotlib").patches.Rectangle(
                    (j - 0.5, y - 0.5), 1, 1, facecolor=SEQ(v), edgecolor=SURFACE, lw=0.8
                )
            )
        ax.text(
            -0.6,
            y,
            lab,
            ha="right",
            va="center",
            fontsize=FS_TICK,
            fontweight="bold" if key == "humans" else None,
            color=MUTED_TXT if key == "cascadeopen" else INK,
        )
    ax.set_xlim(-0.5, len(sectors) - 0.5)
    ax.set_ylim(ys[-1] + 0.5, -0.5)
    ax.set_axis_off()
    for j, sec in enumerate(sectors):
        small = at["by_sector"][sec]["small_n"]
        ax.text(
            j,
            -0.75,
            f"{D.SECTOR_SHORT[sec]} ({sb['cue_cells'][sec]})" + (" \u2021" if small else ""),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=FS_TICK,
            color=MUTED_TXT if small else INK2,
        )
    cax = fig.add_axes(((namew + aw + 0.18) / W, bottom / H + 0.25, 0.1 / W, 0.35))
    import matplotlib as mpl

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=SEQ), cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([0, 0.5, 1], labels=["0", ".5", "1"])
    cb.ax.tick_params(labelsize=FS_TICK)
    cb.set_label("cue-bearing credit", fontsize=FS_TICK)
    return save(fig, "figA4_sectors")


# ------------------------------------------------------------------ A5
def a5_realtime() -> list:
    c = D.conduct()
    names = {r["label"]: D.short(r["name"]) for r in D.leaderboard()["rows"]}
    pairs = c["pairs"]
    metrics = [
        ("credit_rt_minus_file_cue_bearing", "cue-bearing credit"),
        ("act_rt_minus_file", "acts at all (any tool call)"),
        ("probe_rt_minus_file", "hears the cue (probe)"),
    ]
    n = len(pairs)
    W, rowh, top, bottom = ARXIV_FULL, 0.2, 0.32, 0.42
    H = top + bottom + n * rowh
    fig = fig_at(W, H)
    namew, gap = 2.3, 0.12
    pw = (W - namew - 0.08 - 2 * gap) / 3
    for k, (key, title) in enumerate(metrics):
        ax = fig.add_axes(((namew + k * (pw + gap)) / W, bottom / H, pw / W, n * rowh / H))
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        lim = 0.8 if key == "act_rt_minus_file" else 0.6
        ax.set_xlim(-lim, lim / 2)
        tk = [-0.6, -0.3, 0, 0.3] if lim > 0.6 else [-0.4, -0.2, 0, 0.2]
        ax.set_xticks(
            tk,
            [
                f"{t:+.1f}".replace("+0.0", "0").replace("-", "\u2212").replace("0.", ".")
                for t in tk
            ],
        )
        null_line(ax, 0.0)
        for i, p in enumerate(pairs):
            e = p.get(key)
            if not e:
                continue
            ax.plot([e["lo"], e["hi"]], [i, i], color=BLUE, lw=0.8)
            ax.plot([e["mean"]], [i], ls="", marker="s", ms=3.6, color=BLUE)
        ax.set_xlabel("realtime minus file", fontsize=FS_TICK)
        ax.text(-lim, -0.95, title, fontsize=FS_TICK, fontweight="bold", va="bottom")
        if k == 0:
            for i, p in enumerate(pairs):
                same = (
                    "different gen." in p["description"] or "different versions" in p["description"]
                )
                ax.text(
                    -0.03,
                    i,
                    f"{names[p['file']]} → {names[p['realtime']]}" + (" *" if same else ""),
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=FS_TICK,
                )
    fig.text(0.05 / W, 0.08 / H, "* different model generations", fontsize=FS_TICK, color=MUTED_TXT)
    return save(fig, "figA5_realtime")


# ------------------------------------------------------------------ A6
def a6_cue_note() -> list:
    conds = D.experiments()["note_controls"]["conditions"]

    def cond(model: str, path: str, note: str) -> dict[str, Any]:
        return next(
            c for c in conds if c["model"] == model and c["path"] == path and c["note"] == note
        )

    arms = [
        ("gemini-3.7-flash, own audio", "google/gemini-3.7-flash", "audio", BLUE),
        ("gpt-oss-120b, Whisper words", "groq:openai/gpt-oss-120b", "asr", NULL),
    ]
    axes_ = [
        ("emotion", "emotional delivery"),
        ("scene", "scene & second voice"),
        ("speaker", "speaker age"),
        ("slot", "masked word"),
        ("disfluency/truncation", "disfluency / silence"),
    ]
    W, H = ARXIV_HALF, 2.45
    fig = fig_at(W, H)
    namew, top, bottom = 1.12, 0.55, 0.38
    ax = fig.add_axes((namew / W, bottom / H, (W - namew - 0.2) / W, (H - top - bottom) / H))
    for i, (akey, alab) in enumerate(axes_):
        for j, (_lab, model, path, col) in enumerate(arms):
            y = i + (j - 0.5) * 0.36
            n_ = cond(model, path, "oracle")["by_axis"][akey]["credit"]
            s_ = cond(model, path, "sham")["by_axis"][akey]["credit"]
            ax.plot([s_["mean"], n_["mean"]], [y, y], color=MUTED_FILL, lw=1.0, zorder=1)
            ax.plot(
                [s_["mean"]],
                [y],
                ls="",
                marker="o",
                ms=3.4,
                mfc=SURFACE,
                mec=col,
                mew=0.8,
                zorder=3,
                clip_on=False,
            )
            ax.plot(
                [n_["mean"]], [y], ls="", marker="o", ms=3.8, color=col, zorder=4, clip_on=False
            )
        ax.text(
            -0.03,
            i,
            alab,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
        )
    ax.set_ylim(len(axes_) - 0.4, -0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("cue-bearing credit: sham note → cue note")
    from matplotlib.lines import Line2D

    hand = [
        Line2D([], [], ls="", marker="o", ms=3.8, color=c, label=lab) for lab, _m, _p, c in arms
    ]
    hand.append(
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=3.4,
            mfc=SURFACE,
            mec=MUTED_TXT,
            mew=0.8,
            label="hollow: sham note",
        )
    )
    fig.legend(
        handles=hand,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=1,
        fontsize=FS_TICK,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    _ = FS, VIOLET, NULL_BAND, BLUE_LIGHT, np
    return save(fig, "figA6_cue_note")


def main() -> list:
    out = []
    for f in (
        a1_robustness,
        a2_taxonomy,
        a3_test_information,
        a4_sectors,
        a5_realtime,
        a6_cue_note,
    ):
        out += f()
    return out


if __name__ == "__main__":
    import style

    style.setup()
    print(main())
