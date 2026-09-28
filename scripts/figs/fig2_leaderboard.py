# ruff: noqa: RUF001  (typographic minus signs in tick labels are intended)
"""Figure 2: who acts on the audio (arXiv full width) + Interspeech column variant.

Rows = the 28 contestants. The 23 with a transcript-only twin are sorted by their
gain over the words-only null (difference in differences, cue-bearing cells);
the 5 without a twin sit in their own block, compared on accuracy.
(a) cue-bearing credit with the cascade and players as reference bands;
(b) gain over the null, 95% CI, leave-one-family-out range underneath.
Filled marker = clears the null after Holm; hollow = does not; a down-triangle
after the name = significantly below it. First-turn scoring throughout.
"""

from __future__ import annotations

from typing import Any

import data as D
from style import (
    ARXIV_FULL,
    BLUE,
    BLUE_LIGHT,
    FS,
    FS_TICK,
    HUMAN_ALPHA,
    INK,
    INK2,
    IS_COL,
    MODE_MARKER,
    MUTED_TXT,
    NULL,
    NULL_BAND,
    SURFACE,
    fig_at,
    fmt,
    human_band,
    mode_handles,
    null_line,
    save,
)


def _rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rob = {r["label"]: r for r in D.robustness()["arm_vs_cascade"]}
    twin, twinless = [], []
    for r in D.contestants():
        rr = dict(r)
        rr["lofo"] = rob[r["label"]]["leave_one_family_out"]
        rr["holm_clear"] = (
            bool(rob[r["label"]]["rejects_after_holm"]) and r["vs_cascade"]["mean"] > 0
        )
        (twin if r["twin"] else twinless).append(rr)
    twin.sort(key=lambda r: -r["vs_cascade"]["mean"])
    twinless.sort(key=lambda r: -r["vs_cascade"]["mean"])
    lb = D.leaderboard()
    assert sum(r["clears_floor"] for r in twin) == lb["counts"]["twin_bearing_clear"]
    assert sum(r["below_floor"] for r in twinless) == lb["counts"]["twinless_below"]
    assert all(r["holm_clear"] == r["clears_floor"] for r in twin + twinless)
    return twin, twinless


def _ypos(n_twin: int, n_less: int, gap: float = 2.0) -> tuple[list[float], list[float]]:
    yt = [float(i) for i in range(n_twin)]
    yl = [n_twin - 1 + gap + i for i in range(n_less)]
    return yt, yl


def _dot(ax: Any, x: float, y: float, r: dict[str, Any], ms: float = 4.0) -> None:
    mk = MODE_MARKER[r["mode"]]
    filled = r["clears_floor"]
    ax.plot(
        [x],
        [y],
        ls="",
        marker=mk,
        ms=ms if mk != "s" else ms * 0.9,
        mfc=BLUE if filled else SURFACE,
        mec=BLUE,
        mew=0.9,
        zorder=5,
        clip_on=False,
    )


def _name(r: dict[str, Any]) -> str:
    return D.short(r["name"]) + ("  ▼" if r["below_floor"] else "")


def arxiv() -> list:
    twin, less = _rows()
    lb = D.leaderboard()
    casc = lb["cascade"]["cue_credit"]
    hum, _basis = D.players_band()
    floors = D.robust2()["floor"]["floors"]
    base = floors["gptoss"]["cue"]["mean"]
    others = [floors[k]["cue"]["mean"] - base for k in ("dsv4pro", "sonnet5")]
    yt, yl = _ypos(len(twin), len(less))
    ymax = yl[-1]
    W = ARXIV_FULL
    top, bottom, rowh = 0.9, 0.42, 0.126
    H = top + bottom + (ymax + 1) * rowh
    fig = fig_at(W, H)
    namew = 1.62
    a_x0, a_w = namew + 0.05, 2.05
    b_x0, b_w = a_x0 + a_w + 0.38, W - (a_x0 + a_w + 0.38) - 0.12
    ax_h = (ymax + 1) * rowh
    axa = fig.add_axes((a_x0 / W, bottom / H, a_w / W, ax_h / H))
    axb = fig.add_axes((b_x0 / W, bottom / H, b_w / W, ax_h / H))

    for ax in (axa, axb):
        ax.set_ylim(ymax + 0.6, -0.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    # ---- (a) level
    axa.set_xlim(0, 0.7)
    axa.set_xticks(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], ["0", ".1", ".2", ".3", ".4", ".5", ".6", ".7"]
    )
    null_line(axa, casc["mean"], (casc["lo"], casc["hi"]))
    human_band(axa, hum["mean"], hum["lo"], hum["hi"])
    for rows, ys in ((twin, yt), (less, yl)):
        for r, y in zip(rows, ys, strict=True):
            e = r["cue_credit"]
            axa.plot([e["lo"], e["hi"]], [y, y], color=BLUE, lw=0.8, zorder=4)
            _dot(axa, e["mean"], y, r)
            axa.text(
                -0.012,
                y,
                _name(r),
                transform=axa.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=FS_TICK,
                color=INK,
            )
    axa.set_xlabel("credit on cue-bearing cells (first turn)")
    axa.text(
        casc["mean"],
        -1.0,
        "words-only\ncascade",
        ha="center",
        va="bottom",
        fontsize=FS_TICK,
        color=MUTED_TXT,
        linespacing=1.0,
    )
    axa.text(hum["mean"], -1.0, "players", ha="center", va="bottom", fontsize=FS_TICK, color=INK)

    # ---- (b) gain over the null
    axb.set_xlim(-0.3, 0.35)
    axb.set_xticks(
        [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], ["−.3", "−.2", "−.1", "0", ".1", ".2", ".3"]
    )
    null_line(axb, 0.0, (min(0.0, *others), max(0.0, *others)))
    for rows, ys in ((twin, yt), (less, yl)):
        for r, y in zip(rows, ys, strict=True):
            e, lo = r["vs_cascade"], r["lofo"]
            axb.plot(
                [lo["min_mean"], lo["max_mean"]],
                [y, y],
                color=BLUE_LIGHT,
                lw=6.5,
                solid_capstyle="butt",
                zorder=3,
            )
            axb.plot([e["lo"], e["hi"]], [y, y], color=BLUE, lw=0.8, zorder=4)
            _dot(axb, e["mean"], y, r)
    axb.set_xlabel("gain over the words-only null")
    axb.text(0.0, -1.0, "null", ha="center", va="bottom", fontsize=FS_TICK, color=MUTED_TXT)

    # block labels
    gy = (yt[-1] + yl[0]) / 2
    box = {"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6}
    axa.text(
        0.006,
        gy,
        "no transcript-only twin: compared on accuracy",
        ha="left",
        va="center",
        fontsize=FS_TICK,
        color=MUTED_TXT,
        style="italic",
        bbox=box,
        zorder=6,
    )
    axb.text(
        -0.295,
        gy,
        "audio credit minus the cascade's",
        ha="left",
        va="center",
        fontsize=FS_TICK,
        color=MUTED_TXT,
        style="italic",
        bbox=box,
        zorder=6,
    )

    # encoding key, stated in the figure above (b)
    n_clear = lb["counts"]["twin_bearing_clear"]
    kax = fig.add_axes((b_x0 / W, (H - 0.64) / H, b_w / W, 0.58 / H))
    kax.set_axis_off()
    kax.set_xlim(0, b_w)
    kax.set_ylim(-4.2, 0.4)
    kax.plot([0.05], [0], ls="", marker="o", ms=4.0, color=BLUE)
    kax.text(
        0.13,
        0,
        f"filled: clears the null after Holm ({n_clear} of {len(twin)})",
        va="center",
        fontsize=FS_TICK,
    )
    kax.plot([0.05], [-1], ls="", marker="o", ms=4.0, mfc=SURFACE, mec=BLUE, mew=0.9)
    kax.text(
        0.13,
        -1,
        "hollow: does not;  \u25bc = significantly below the null",
        va="center",
        fontsize=FS_TICK,
    )
    kax.plot([0.0, 0.1], [-2, -2], color=BLUE_LIGHT, lw=6.5, solid_capstyle="butt")
    kax.plot([0.0, 0.1], [-2, -2], color=BLUE, lw=0.8)
    kax.text(
        0.13,
        -2,
        "line: 95% CI;  shade: leave-one-family-out range",
        va="center",
        fontsize=FS_TICK,
    )
    kax.fill_between([0.0, 0.1], -3.3, -2.7, color=NULL_BAND, lw=0)
    kax.plot([0.02, 0.02], [-3.3, -2.7], color=NULL, lw=0.9, ls=(0, (2, 1.5)))
    kax.text(
        0.13, -3, "null: gpt-oss cascade; band: two other text LLMs", va="center", fontsize=FS_TICK
    )
    handles = mode_handles()
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.25 / W, 1.0),
        ncol=3,
        fontsize=FS_TICK,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    fig.text(0.02 / W, (H - 0.1) / H, "(a)", fontsize=8.5, fontweight="bold", va="center")
    fig.text((b_x0 - 0.3) / W, (H - 0.1) / H, "(b)", fontsize=8.5, fontweight="bold", va="center")
    _ = FS, INK2, NULL, NULL_BAND, HUMAN_ALPHA, fmt
    return save(fig, "fig2_leaderboard")


def interspeech() -> list:
    """Column width: gain over the null only, 23 twin-bearing systems + the 5 others."""
    twin, less = _rows()
    floors = D.robust2()["floor"]["floors"]
    base = floors["gptoss"]["cue"]["mean"]
    others = [floors[k]["cue"]["mean"] - base for k in ("dsv4pro", "sonnet5")]
    yt, yl = _ypos(len(twin), len(less), gap=1.8)
    ymax = yl[-1]
    W = IS_COL
    top, bottom, rowh = 0.5, 0.36, 0.112
    H = top + bottom + (ymax + 1) * rowh
    fig = fig_at(W, H)
    x0 = 1.5
    ax = fig.add_axes((x0 / W, bottom / H, (W - x0 - 0.1) / W, (ymax + 1) * rowh / H))
    ax.set_ylim(ymax + 0.6, -0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.3, 0.35)
    ax.set_xticks([-0.2, 0, 0.2], ["−.2", "0", ".2"])
    null_line(ax, 0.0, (min(0.0, *others), max(0.0, *others)))
    for rows, ys in ((twin, yt), (less, yl)):
        for r, y in zip(rows, ys, strict=True):
            e, lo = r["vs_cascade"], r["lofo"]
            ax.plot(
                [lo["min_mean"], lo["max_mean"]],
                [y, y],
                color=BLUE_LIGHT,
                lw=3.6,
                solid_capstyle="butt",
                zorder=3,
            )
            ax.plot([e["lo"], e["hi"]], [y, y], color=BLUE, lw=0.8, zorder=4)
            _dot(ax, e["mean"], y, r, ms=3.6)
            ax.text(
                -0.02,
                y,
                _name(r),
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=FS_TICK,
            )
    gy = (yt[-1] + yl[0]) / 2
    ax.axhline(gy, color="#e9e8e4", lw=0.6)
    ax.text(
        -0.295,
        gy,
        "no twin: accuracy minus cascade's",
        ha="left",
        va="center",
        fontsize=FS_TICK,
        color=MUTED_TXT,
        style="italic",
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6},
        zorder=6,
    )
    ax.set_xlabel("gain over the words-only null")
    from matplotlib.lines import Line2D

    fh = [
        Line2D([], [], ls="", marker="o", ms=3.6, color=BLUE, label="clears null (Holm)"),
        Line2D([], [], ls="", marker="o", ms=3.6, mfc=SURFACE, mec=BLUE, mew=0.9, label="does not"),
    ]
    fig.legend(
        handles=fh,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0 - 0.17 / H),
        ncol=2,
        fontsize=FS_TICK,
        handletextpad=0.2,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    fig.legend(
        handles=mode_handles(),
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        fontsize=FS_TICK,
        handletextpad=0.2,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    return save(fig, "is_fig2_leaderboard")


def main() -> list:
    return arxiv() + interspeech()


if __name__ == "__main__":
    import style

    style.setup()
    print(main())
