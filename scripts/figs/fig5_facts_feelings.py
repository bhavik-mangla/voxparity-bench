"""Figure for Section 5, "facts, not feelings" (arXiv full width, two panels).

(a) Told the cue as a one-word label: credit by cue kind (environmental sound, second
    voice, emotional delivery) with a one-line note naming the cue (filled, 95%
    item-clustered CI) and without it (hollow), for gemini-3.7-flash on its own
    audio, the exact transcript and the Whisper transcript, and for three
    text-only language models on the Whisper transcript. Source:
    docs/results/exp/cue-note-split.json (first-turn scoring). The no-note value is
    the with-note credit minus the paired note effect stored in the same file
    ("minus_baseline", identical cells), not a new analysis.
(b) The cue note for emotional delivery: credit on emotional-delivery calls when the
    note describes the voice as rendered (filled, the study's cue note) and when it
    gives the emotion as a single label (hollow, ablation), with 95% item-clustered
    CIs, for gemini-3.7-flash and Qwen3.8-Omni (own audio and exact transcript) and
    gpt-oss-120b (exact transcript); the tick marks credit on every other cue kind
    under the description. Source: docs/insights/notefull.json (description_note,
    identical cells, first-turn scoring; scripts/insights/notefull_analysis.py).

Players are not drawn: neither panel's cells are the players' cells.
"""

from __future__ import annotations

from typing import Any

import data as D
from matplotlib.lines import Line2D
from style import (
    ARXIV_FULL,
    AXIS,
    BLUE,
    CUE_COLOR,
    FS_PANEL,
    FS_TICK,
    INK,
    INK2,
    MUTED,
    MUTED_TXT,
    SURFACE,
    fig_at,
    save,
)

GEM = "google/gemini-3.7-flash"
CONDS = [  # (model, input path, row label)
    (GEM, "audio", "gemini-3.7-flash · own audio"),
    (GEM, "gold", "gemini-3.7-flash · exact transcript"),
    (GEM, "asr", "gemini-3.7-flash · Whisper transcript"),
    ("groq:openai/gpt-oss-120b", "asr", "gpt-oss-120b · Whisper transcript"),
    ("deepseek/deepseek-v4-pro", "asr", "DeepSeek-V4-Pro · Whisper transcript"),
    ("anthropic/claude-sonnet-5", "asr", "Claude Sonnet 5 · Whisper transcript"),
]
KINDS = [  # (cue-note-split key, label, colour)
    ("environmental", "environmental sound", CUE_COLOR["other"]),
    ("second voice", "second voice", BLUE),  # distinct from environmental (validated trio)
    ("emotion", "emotional delivery", CUE_COLOR["emotion"]),
]
GAP = 0.9  # blank sub-rows between conditions


def _cond(model: str, path: str) -> dict[str, Any]:
    hits = [
        c
        for c in D.cue_note_split()["conditions"]
        if c["model"] == model and c["path"] == path and c["note"] == "oracle"
    ]
    assert len(hits) == 1, (model, path, len(hits))
    return hits[0]


def rows() -> list[tuple[str, dict[str, Any]]]:
    return [(lab, _cond(m, p)) for m, p, lab in CONDS]


NOTE_ROWS = [  # (notefull model, path, row label) for panel (b)
    ("gemini-3.7-flash", "audio", "gemini-3.7-flash \u00b7 own audio"),
    ("gemini-3.7-flash", "transcript", "gemini-3.7-flash \u00b7 exact transcript"),
    ("Qwen3.8-Omni", "audio", "Qwen3.8-Omni \u00b7 own audio"),
    ("Qwen3.8-Omni", "transcript", "Qwen3.8-Omni \u00b7 exact transcript"),
    ("gpt-oss-120b", "transcript", "gpt-oss-120b \u00b7 exact transcript"),
]


def note_rows() -> list[tuple[str, dict[str, Any]]]:
    rows = D.notefull()["description_note"]["rows"]
    out = []
    for model, path, lab in NOTE_ROWS:
        hits = [r for r in rows if r["model"] == model and r["path"] == path]
        assert len(hits) == 1, (model, path, len(hits))
        out.append((lab, hits[0]["groups"]))
    return out


def arxiv() -> list:
    W, H = ARXIV_FULL, 3.15
    fig = fig_at(W, H)
    top, bottom = 0.34, 0.4
    namew, aw = 1.78, 1.78
    ax = fig.add_axes((namew / W, bottom / H, aw / W, (H - top - bottom) / H))
    step = len(KINDS) + GAP
    ymax = (len(CONDS) - 1) * step + len(KINDS) - 1
    order_ok = 0
    for i, (lab, c) in enumerate(rows()):
        y0 = i * step
        vals = {}
        for k, (key, _kl, col) in enumerate(KINDS):
            y = y0 + k
            e = c["by_axis"][key]["credit"]
            base = e["mean"] - c["by_axis"][key]["minus_baseline"]["mean"]
            vals[key] = e["mean"]
            ax.plot([base, e["mean"]], [y, y], color=MUTED, lw=1.0, zorder=2)
            ax.plot([e["lo"], e["hi"]], [y, y], color=col, lw=0.8, zorder=3)
            ax.plot(
                [base],
                [y],
                ls="",
                marker="o",
                ms=3.3,
                mfc=SURFACE,
                mec=MUTED_TXT,
                mew=0.8,
                zorder=4,
                clip_on=False,
            )
            ax.plot([e["mean"]], [y], ls="", marker="o", ms=3.8, color=col, zorder=5, clip_on=False)
            if i == 0:
                ax.text(
                    1.04,
                    y,
                    _kl,
                    transform=ax.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=FS_TICK,
                    color=col,
                )
        order_ok += vals["emotion"] < min(vals["environmental"], vals["second voice"])
        ax.text(
            -0.03,
            y0 + 1,
            lab,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
            color=INK,
            fontweight="bold" if i == 0 else None,
        )
        if i == 3:
            ax.axhline(y0 - GAP / 2 - 0.5, color="#e0dfdb", lw=0.6, xmin=-0.9, clip_on=False)
    assert order_ok == len(CONDS), "emotion is not lowest in every condition"
    ax.set_ylim(ymax + 0.7, -0.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.03)  # markers at 1.00 are not clipped
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
    for xv in (0.25, 0.5, 0.75):
        ax.axvline(xv, color="#f1f0ec", lw=0.5, zorder=0)
    ax.set_xlabel("credit on calls with the cue")

    hand = [
        Line2D(
            [], [], ls="", marker="o", ms=3.3, mfc=SURFACE, mec=MUTED_TXT, mew=0.8, label="no note"
        ),
        Line2D([], [], ls="", marker="o", ms=3.8, color=INK2, label="one-word label (95% CI)"),
    ]
    fig.legend(
        handles=hand,
        loc="upper left",
        bbox_to_anchor=(namew / W - 0.01, 1.0),
        ncol=2,
        fontsize=FS_TICK,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.25,
    )
    fig.text(0.0, (H - 0.13) / H, "(a)", fontsize=FS_PANEL, fontweight="bold", va="center")

    # ---- (b) the cue note for emotional delivery: description vs label only
    bx0 = 4.72
    bw = W - bx0 - 0.1
    bh = H - top - bottom - 0.42
    axb = fig.add_axes((bx0 / W, bottom / H, bw / W, bh / H))
    emo = CUE_COLOR["emotion"]
    below = 0
    nr = note_rows()
    for j, (lab, g) in enumerate(nr):
        e_lab, e_desc = g["emotion"]["label"], g["emotion"]["description"]
        oth = g["other"]["description"]
        below += e_desc["mean"] < oth["mean"]
        yl, yd = j + 0.1, j + 0.3
        axb.plot([e_lab["mean"], e_desc["mean"]], [yl, yd], color=MUTED, lw=1.0, zorder=2)
        axb.plot([e_lab["lo"], e_lab["hi"]], [yl, yl], color=MUTED_TXT, lw=0.8, zorder=3)
        axb.plot([e_desc["lo"], e_desc["hi"]], [yd, yd], color=emo, lw=0.8, zorder=3)
        axb.plot(
            [e_lab["mean"]],
            [yl],
            ls="",
            marker="o",
            ms=3.3,
            mfc=SURFACE,
            mec=MUTED_TXT,
            mew=0.8,
            zorder=4,
        )
        axb.plot([e_desc["mean"]], [yd], ls="", marker="o", ms=3.8, color=emo, zorder=5)
        axb.plot([oth["mean"]], [yd], ls="", marker="|", ms=6, mew=1.0, color=AXIS, zorder=4)
        axb.text(0.405, j - 0.12, lab, ha="left", va="center", fontsize=FS_TICK, color=INK)
    assert below == len(nr), "emotion is not below the other cues under the description"
    axb.set_ylim(len(nr) - 0.45, -0.42)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.set_xlim(0.4, 1.02)
    axb.set_xticks([0.4, 0.6, 0.8, 1], [".4", ".6", ".8", "1"])
    for xv in (0.6, 0.8):
        axb.axvline(xv, color="#f1f0ec", lw=0.5, zorder=0)
    axb.set_xlabel("credit on calls with the cue")
    hb = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=3.3,
            mfc=SURFACE,
            mec=MUTED_TXT,
            mew=0.8,
            label="label only",
        ),
        Line2D([], [], ls="", marker="o", ms=3.8, color=emo, label="description"),
        Line2D([], [], ls="", marker="|", ms=6, mew=1.0, color=AXIS, label="other cues"),
    ]
    fig.legend(
        handles=hb,
        loc="upper left",
        bbox_to_anchor=(bx0 / W - 0.012, (H - 0.3) / H),
        ncol=2,
        fontsize=FS_TICK,
        handletextpad=0.1,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    fig.text(
        bx0 / W,
        (H - 0.13) / H,
        "Cue note for emotional delivery",
        fontsize=FS_TICK,
        color=INK2,
        va="center",
    )
    fig.text(
        (bx0 - 0.3) / W, (H - 0.13) / H, "(b)", fontsize=FS_PANEL, fontweight="bold", va="center"
    )
    return save(fig, "fig5_facts_feelings")


def main() -> list:
    return arxiv()


if __name__ == "__main__":
    import style

    style.setup()
    print(main())
