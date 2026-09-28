"""Heard emotion is acted on less than other heard cues (arXiv full width; paper
Figure 2, Section 5.1) + Interspeech column variant.

P(right action | the system's own probe missed the cue) -> P(right action | it
heard the cue), per system, emotional delivery vs every other cue, on the
protocol-grounded core cells that players also answered. Players, the frontier-4
and all 27 probe-readable systems pooled are summary rows at the top; the players'
P(right | heard) is also a reference band in each panel.

Per-system values: paper/figures/data/heard_by_kind.json (scripts/figs/derive_heard.py,
checked against review2-interaction.json E30); group CIs: review2-interaction.json.
"""

from __future__ import annotations

from typing import Any

import data as D
from style import (
    ARXIV_FULL,
    CUE_COLOR,
    FS,
    FS_TICK,
    INK,
    IS_COL,
    MODE_MARKER,
    MUTED,
    MUTED_TXT,
    SURFACE,
    fig_at,
    save,
)

MIN_N = 15  # fewer heard or missed cells than this: drawn faint


def _data() -> dict[str, Any]:
    return D.derived("heard_by_kind")


def _summary(
    hb: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    E = D.review2()["E30_identical"]
    n_arms = len(hb["arms"])
    return [
        ("players", "humans", hb["humans"], E["humans.emo.raw"], E["humans.non.raw"]),
        (
            "four leading systems",
            "frontier4",
            hb["frontier4_pooled"],
            E["frontier4.emo.raw"],
            E["frontier4.non.raw"],
        ),
        (
            f"all {n_arms} systems pooled",
            "pooled",
            hb["pooled"],
            E["pooled.emo.raw"],
            E["pooled.non.raw"],
        ),
    ]


def _modes() -> dict[str, str]:
    return {r["label"]: r["mode"] for r in D.leaderboard()["rows"]}


def _names() -> dict[str, str]:
    return {r["label"]: D.short(r["name"]) for r in D.leaderboard()["rows"]}


def _draw_row(
    ax: Any,
    y: float,
    miss: float | None,
    heard: float | None,
    color: str,
    mk: str,
    faint: bool,
    ci: dict[str, Any] | None = None,
    ms: float = 4.0,
) -> None:
    al = 0.45 if faint else 1.0
    if miss is not None and heard is not None:
        ax.plot([miss, heard], [y, y], color=MUTED, lw=1.0, alpha=al, zorder=2)
    if ci is not None:
        ax.plot([ci["lo"], ci["hi"]], [y, y], color=color, lw=0.8, zorder=3)
    if miss is not None:
        ax.plot(
            [miss],
            [y],
            ls="",
            marker=mk,
            ms=ms * 0.9,
            mfc=SURFACE,
            mec=MUTED_TXT,
            mew=0.8,
            alpha=al,
            zorder=4,
        )
    if heard is not None:
        ax.plot([heard], [y], ls="", marker=mk, ms=ms, color=color, alpha=al, zorder=5)


def arxiv() -> list:
    hb = _data()
    f4 = set(D.review2()["frontier4"])
    modes, names = _modes(), _names()
    arms = sorted(hb["arms"], key=lambda a: -(hb["arms"][a]["emo"]["aT"] or 0))
    summ = _summary(hb)
    n_s = len(summ)
    ys_s = list(range(n_s))
    y0 = n_s + 0.9
    ys_a = [y0 + i for i in range(len(arms))]
    ymax = ys_a[-1]
    W, rowh, top, bottom = ARXIV_FULL, 0.118, 0.5, 0.42
    H = top + bottom + (ymax + 1) * rowh
    fig = fig_at(W, H)
    namew = 1.45
    pw = (W - namew - 0.1 - 0.3) / 2
    axes = []
    for k, kind in enumerate(("emo", "non")):
        x0 = namew + k * (pw + 0.3)
        ax = fig.add_axes((x0 / W, bottom / H, pw / W, (ymax + 1) * rowh / H))
        axes.append(ax)
        col = CUE_COLOR["emotion" if kind == "emo" else "other"]
        ax.set_ylim(ymax + 0.6, -0.6)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
        for xv in (0.25, 0.5, 0.75):
            ax.axvline(xv, color="#f1f0ec", lw=0.5, zorder=0)
        # players' P(right | heard) band
        hci = summ[0][3 if kind == "emo" else 4]
        ax.axvspan(hci["lo"], hci["hi"], ymin=0, ymax=1, color=INK, alpha=0.06, lw=0, zorder=0.5)
        ax.axvline(hci["mean"], color=INK, lw=0.8, zorder=1)
        for (_lab, key, pt, ce, cn), y in zip(summ, ys_s, strict=True):
            ci = ce if kind == "emo" else cn
            c = INK if key == "humans" else col
            _draw_row(
                ax,
                y,
                pt[kind]["aN"],
                pt[kind]["aT"],
                c,
                "D" if key == "humans" else "o",
                False,
                ci=ci,
            )
            ax.text(
                0.995,
                y,
                f"{pt[kind]['aT']:.2f}",
                ha="right",
                va="center",
                fontsize=FS_TICK,
                color=INK,
            )
        for a, y in zip(arms, ys_a, strict=True):
            v = hb["arms"][a][kind]
            faint = min(v["n_heard"], v["n_missed"]) < MIN_N
            _draw_row(ax, y, v["aN"], v["aT"], col, MODE_MARKER[modes[a]], faint)
        ax.axhline((ys_s[-1] + ys_a[0]) / 2, color="#e9e8e4", lw=0.6)
        ax.set_xlabel("P(right action)")
        ax.text(
            0,
            -0.9 - 0.2,
            "emotional delivery" if kind == "emo" else "other cues",
            fontsize=FS + 0.5,
            fontweight="bold",
            color=col,
            va="bottom",
            ha="left",
        )
    # names
    for (lab, key, *_r), y in zip(summ, ys_s, strict=True):
        axes[0].text(
            -0.03,
            y,
            lab,
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
            fontweight="bold" if key != "pooled" else None,
        )
    for a, y in zip(arms, ys_a, strict=True):
        axes[0].text(
            -0.03,
            y,
            names[a],
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
            fontweight="bold" if a in f4 else None,
        )
    # key
    from matplotlib.lines import Line2D

    hand = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=3.6,
            mfc=SURFACE,
            mec=MUTED_TXT,
            mew=0.8,
            label="own probe missed the cue",
        ),
        Line2D(
            [], [], ls="", marker="o", ms=4.0, color=CUE_COLOR["emotion"], label="heard: emotion"
        ),
        Line2D([], [], ls="", marker="o", ms=4.0, color=CUE_COLOR["other"], label="heard: other"),
        Line2D([], [], color=INK, lw=0.8, label="players, heard (95% CI band)"),
    ]
    fig.legend(
        handles=hand,
        loc="upper left",
        bbox_to_anchor=(0.1 / W, 1.0),
        ncol=4,
        fontsize=FS_TICK,
        handletextpad=0.3,
        columnspacing=1.1,
        borderaxespad=0.2,
    )
    fig.text(0.1 / W, (H - 0.3) / H, "bold: leading four", fontsize=FS_TICK, color=INK, va="center")
    return save(fig, "fig3_heard")


def interspeech() -> list:
    """Column width: the three summary rows only, emotion vs other, with CIs."""
    hb = _data()
    summ = _summary(hb)
    W, H = IS_COL, 1.6
    fig = fig_at(W, H)
    x0, top, bottom = 1.2, 0.3, 0.36
    ax = fig.add_axes((x0 / W, bottom / H, (W - x0 - 0.12) / W, (H - top - bottom) / H))
    ys = []
    for i, (_lab, key, pt, ce, cn) in enumerate(summ):
        for j, (kind, ci) in enumerate((("emo", ce), ("non", cn))):
            y = i * 2.6 + j * 1.0
            col = CUE_COLOR["emotion" if kind == "emo" else "other"]
            c = INK if key == "humans" else col
            _draw_row(
                ax,
                y,
                pt[kind]["aN"],
                pt[kind]["aT"],
                c,
                "D" if key == "humans" else "o",
                False,
                ci=ci,
                ms=3.6,
            )
            ys.append(y)
        for j, sub in enumerate(("emotion", "other")):
            ax.text(
                -0.03,
                i * 2.6 + j,
                sub,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=FS_TICK,
                color=INK,
            )
        short = {"humans": "players", "frontier4": "leading 4", "pooled": "all 27"}[key]
        ax.text(
            -0.36,
            i * 2.6 + 0.5,
            short,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
            fontweight="bold",
        )
    ax.set_ylim(ys[-1] + 0.7, -0.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("P(right action): missed → heard")
    from matplotlib.lines import Line2D

    hand = [
        Line2D(
            [], [], ls="", marker="o", ms=3.3, mfc=SURFACE, mec=MUTED_TXT, mew=0.8, label="missed"
        ),
        Line2D(
            [], [], ls="", marker="o", ms=3.6, color=CUE_COLOR["emotion"], label="heard: emotion"
        ),
        Line2D([], [], ls="", marker="o", ms=3.6, color=CUE_COLOR["other"], label="heard: other"),
    ]
    fig.legend(
        handles=hand,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        fontsize=FS_TICK,
        handletextpad=0.2,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    return save(fig, "is_fig3_heard")


def main() -> list:
    return arxiv() + interspeech()


if __name__ == "__main__":
    import style

    style.setup()
    print(main())
