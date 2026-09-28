"""SUPERSEDED Figure 1 (v4; now fig1_overview.py) + its panel (b) as the appendix figure
figA_parity_by_cue and the Interspeech column variant is_fig1_axes.

VoxParity at a glance (arXiv full width; Interspeech column variant of (b)).

(a) the seven cue axes x 14 sectors: dot area = measured scenarios in that cell,
    the axis's scenario / cue-cell count, and one exemplar "what changes -> the
    action the protocol requires" per axis (read from atlas.json patterns);
(b) per axis: players (selection credit, 95% CI), one fixed system on every axis
    (gemini-3.7-flash, atlas "reference", the system Section 4.1 compares with the
    players; not a per-axis winner), the median of the 28 systems, and the words-only
    cascade (dashed reference) on cue-bearing cells.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import data as D
import numpy as np
from style import (
    ARXIV_FULL,
    BLUE,
    FS,
    FS_TICK,
    INK,
    INK2,
    IS_COL,
    MUTED,
    MUTED_TXT,
    NULL,
    NULL_BAND,
    SURFACE,
    fig_at,
    panel,
    save,
)

ROWS = [*D.AXES, "channel"]
# the fixed system drawn on every axis of panel (b): the overall cue-bearing leader
# that Section 4.1 compares with the players (atlas.json by_axis[...]["reference"])
FIXED = "gemini37or"
FIXED_NAME = "gemini-3.7-flash"


def _counts() -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    a = D.atlas()
    items: dict[tuple[str, str], set[str]] = defaultdict(set)
    ax_items: dict[str, set[str]] = defaultdict(set)
    ax_cells: dict[str, int] = defaultdict(int)
    for p in a["patterns"]:
        items[(p["axis"], p["sector"])].add(p["item"])
        ax_items[p["axis"]].add(p["item"])
        ax_cells[p["axis"]] += 1
    # the per-axis scenario and cell counts must equal atlas.json by_axis
    for ax, b in a["by_axis"].items():
        assert len(ax_items[ax]) == b["items"] and ax_cells[ax] == b["cells"], ax
    m = {ax: {s: len(items[(ax, s)]) for s in a["by_sector"]} for ax in ROWS}
    return m, {k: len(v) for k, v in ax_items.items()}, dict(ax_cells)


def _action(t: str) -> str:
    return t.replace("_", " ")


def _key(fig: Any, x0: float, y0: float, compact: bool = False) -> None:
    """Marker key drawn in figure inches (no box)."""
    W, H = fig.get_size_inches()
    entries = [
        ("players (95% CI)", "D", INK, INK),
        (FIXED_NAME, "o", BLUE, BLUE),
        ("median of 28 systems", "o", SURFACE, BLUE),
        ("words-only cascade", "|", NULL, NULL),
    ]
    dx = 0.0
    dy = 0.0
    for i, (lab, mk, fc, ec) in enumerate(entries):
        if compact:
            cx, cy = x0 + (i % 2) * 1.45, y0 - (i // 2) * 0.14
        else:
            cx, cy = x0, y0 - i * 0.135
        ax = fig.add_axes((cx / W, (cy - 0.05) / H, 0.12 / W, 0.1 / H))
        ax.set_axis_off()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        if mk == "|":
            ax.plot([0, 0], [-0.9, 0.9], color=NULL, lw=0.9, ls=(0, (2, 1.5)))
        else:
            ax.plot(
                [0], [0], ls="", marker=mk, ms=4.2 if mk != "D" else 3.8, mfc=fc, mec=ec, mew=0.9
            )
        fig.text((cx + 0.15) / W, cy / H, lab, fontsize=FS_TICK, va="center", color=INK)
    _ = dx, dy


def _headline_panel(ax: Any, rows: list[str], small: bool = False) -> None:
    a = D.atlas()["by_axis"]
    y = np.arange(len(rows))
    for i, axn in enumerate(rows):
        b = a[axn]
        if axn == "channel":
            ax.text(
                0.02,
                i,
                f"{b['cells']} cell: not summarised",
                fontsize=FS_TICK,
                color=MUTED_TXT,
                va="center",
            )
            continue
        casc = b["cascade"]["mean"]
        hum = b["humans"]
        faint = b["cells"] < 10
        al = 0.55 if faint else 1.0
        ax.plot(
            [casc, hum["mean"]], [i, i], color=NULL_BAND, lw=5.5, solid_capstyle="butt", zorder=1
        )
        ax.plot([casc, casc], [i - 0.32, i + 0.32], color=NULL, lw=0.9, ls=(0, (2, 1.5)), zorder=2)
        ax.plot([hum["lo"], hum["hi"]], [i, i], color=INK, lw=0.8, zorder=3)
        ax.plot([hum["mean"]], [i], ls="", marker="D", ms=3.8, color=INK, zorder=4)
        ax.plot(
            [b["median_system_credit"]],
            [i],
            ls="",
            marker="o",
            ms=4.2,
            mfc=SURFACE,
            mec=BLUE,
            mew=0.9,
            alpha=al,
            zorder=5,
            clip_on=False,
        )
        ref = b["reference"]
        assert ref["label"] == FIXED, (axn, ref["label"])
        ax.plot(
            [ref["credit"]["mean"]],
            [i],
            ls="",
            marker="o",
            ms=4.2,
            color=BLUE,
            alpha=al,
            zorder=6,
            clip_on=False,
        )
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("credit on calls with an audible cue")
    for xv in (0.25, 0.5, 0.75):
        ax.axvline(xv, color="#f1f0ec", lw=0.5, zorder=0)
    _ = y, small


def arxiv() -> list:
    a = D.atlas()
    m, ax_items, ax_cells = _counts()
    sectors = sorted(a["by_sector"], key=lambda s: -a["counts"]["sectors"][s])
    W, H = ARXIV_FULL, 3.3
    fig = fig_at(W, H)
    top_head = 0.74  # inches reserved above the rows
    bottom = 0.40
    row_h = (H - top_head - bottom) / len(ROWS)

    # ---- (a) text block
    tx0, tx1 = 0.02, 2.62
    mx0, mx1 = 2.72, 4.28
    bx0, bx1 = 4.62, 6.38
    ytop = H - top_head

    def yrow(i: int) -> float:
        return ytop - (i + 0.5) * row_h

    for i, axn in enumerate(ROWS):
        yc = yrow(i)
        chan = axn == "channel"
        col = MUTED_TXT if chan else INK
        fig.text(
            tx0 / W,
            (yc + 0.055) / H,
            D.AXIS_LABEL[axn],
            fontsize=FS,
            fontweight="bold",
            color=col,
            va="center",
        )
        ni, nc = ax_items[axn], ax_cells[axn]
        cnt = f"{ni} scenario{'s' if ni != 1 else ''} · {nc} cue cell{'s' if nc != 1 else ''}"
        fig.text(
            tx1 / W, (yc + 0.055) / H, cnt, fontsize=FS_TICK, color=col, ha="right", va="center"
        )
        if chan:
            ex = "a found film clip; not one of the seven axes"
        else:
            p = D.exemplar(axn)
            ex = f"{p['variant'].replace('_', ' ')} → {_action(p['gold'])}"
        fig.text(
            tx0 / W,
            (yc - 0.07) / H,
            ex,
            fontsize=FS_TICK,
            color=MUTED_TXT if chan else INK2,
            va="center",
            style="italic",
        )
    fig.text(
        tx0 / W,
        (ytop + 0.06) / H,
        "what the audio adds → the action the rule requires",
        fontsize=FS_TICK,
        color=INK2,
        va="bottom",
    )
    # totals, read from atlas counts and the leaderboard
    lb = D.leaderboard()
    foot = (
        f"{a['counts']['items']} scenarios in {len(a['counts']['sectors'])} sectors\n"
        f"{a['counts']['cue_cells']} cue-bearing cells · {lb['counts']['contestants']} systems"
    )
    fig.text(0.3 / W, (H - 0.06) / H, foot, fontsize=FS, color=INK, va="top", linespacing=1.4)

    # ---- (a) dot matrix
    axm = fig.add_axes(
        (mx0 / W, (ytop - len(ROWS) * row_h) / H, (mx1 - mx0) / W, len(ROWS) * row_h / H)
    )
    xs, ys, ss = [], [], []
    for i, axn in enumerate(ROWS):
        for j, s in enumerate(sectors):
            n = m[axn][s]
            if n:
                xs.append(j)
                ys.append(i)
                ss.append(n)
    ss_arr = np.array(ss, float)
    smax = max(ss)
    axm.scatter(xs, ys, s=4 + 62 * ss_arr / smax, color=INK, lw=0, zorder=3)
    for j in range(len(sectors)):
        axm.axvline(j, color="#f1f0ec", lw=0.5, zorder=0)
    axm.set_xlim(-0.6, len(sectors) - 0.4)
    axm.set_ylim(len(ROWS) - 0.5, -0.5)
    axm.set_axis_off()
    for j, s in enumerate(sectors):
        axm.text(
            j,
            -0.62,
            D.SECTOR_SHORT[s],
            rotation=90,
            fontsize=FS_TICK,
            ha="center",
            va="bottom",
            color=INK2,
            clip_on=False,
        )
    # size key under the matrix
    kx = mx0 + 0.02
    for k, n in enumerate((1, 5, 15)):
        cx = kx + k * 0.42
        kax = fig.add_axes(((cx) / W, 0.06 / H, 0.12 / W, 0.12 / H))
        kax.set_axis_off()
        kax.set_xlim(-1, 1)
        kax.set_ylim(-1, 1)
        kax.scatter([0], [0], s=4 + 62 * n / smax, color=INK, lw=0)
        fig.text((cx + 0.12) / W, 0.12 / H, f"{n}", fontsize=FS_TICK, va="center", color=INK2)
    fig.text(mx0 / W, 0.26 / H, "scenarios per sector", fontsize=FS_TICK, color=INK2, va="center")

    # ---- (b)
    axb = fig.add_axes(
        (bx0 / W, (ytop - len(ROWS) * row_h) / H, (bx1 - bx0) / W, len(ROWS) * row_h / H)
    )
    _headline_panel(axb, ROWS)
    _key(fig, bx0 + 0.05, H - 0.13)
    panel_y = ytop + 0.02
    fig.text(0.0 / W, (H - 0.13) / H, "(a)", fontsize=8.5, fontweight="bold", va="center")
    fig.text((bx0 - 0.3) / W, (H - 0.13) / H, "(b)", fontsize=8.5, fontweight="bold", va="center")
    _ = panel, panel_y, MUTED
    return save(fig, "fig1_glance")


def interspeech(name: str = "is_fig1_axes") -> list:
    """Column-width (3.15 in) version of panel (b) with axis labels as row names.
    Also the arXiv appendix figure figA_parity_by_cue (same drawing, same width)."""
    rows = list(D.AXES)
    W, H = IS_COL, 2.2
    fig = fig_at(W, H)
    left, right, top, bottom = 1.2, 0.14, 0.4, 0.36
    ax = fig.add_axes((left / W, bottom / H, (W - left - right) / W, (H - top - bottom) / H))
    _headline_panel(ax, rows)
    _, _ax_items, ax_cells = _counts()
    for i, axn in enumerate(rows):
        ax.text(
            -0.03,
            i,
            f"{D.AXIS_LABEL[axn]} ({ax_cells[axn]})",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
            color=INK,
        )
    _key(fig, 0.05, H - 0.12, compact=True)
    return save(fig, name)


def main() -> list:
    # fig1_glance (the v4 Figure 1) is superseded by fig1_overview; kept buildable so
    # the current PAPER.md still compiles until the writer switches.
    return arxiv() + interspeech() + interspeech("figA_parity_by_cue")


if __name__ == "__main__":
    import style

    style.setup()
    print(main())
