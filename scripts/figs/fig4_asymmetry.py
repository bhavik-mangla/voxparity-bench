"""Appendix figure: errors run toward the words, by group (arXiv full width) + Interspeech
column variant. Each system's own operating point is Figure 1b (fig1_overview.py, which
imports error_scatter from here).

Diverging bars: left = over-trigger rate on clean calls, right = unsafe
    execution (the routine action the words asked for) on protective calls, for
    players, the frontier-4, all 28 systems pooled and the words-only cascade;
    95% CIs (players: two-way item x player bootstrap).

Sources: docs/insights/harm.json (groups, per-system metrics),
docs/insights/robustness-v2.json (players, two-way CIs),
paper/figures/data/harm_frontier4.json (frontier-4, scripts/figs/derive_harm.py).
"""

from __future__ import annotations

from typing import Any

import data as D
from style import (
    ARXIV_FULL,
    BLUE,
    FS_TICK,
    INK,
    INK2,
    IS_COL,
    MODE_MARKER,
    MUTED_TXT,
    NULL,
    NULL_BAND,
    ORANGE,
    SURFACE,
    fig_at,
    save,
)


def _groups() -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    h = D.harm()["asymmetry"]["groups"]
    hum = D.robust2()["harm"]["humans"]
    f4 = D.derived("harm_frontier4")
    assert set(f4["arms"]) == set(D.review2()["frontier4"])
    n_all = len(h["all contestants"]["arms"])
    return [
        ("players", "human", hum["over_trigger"]["two_way"], hum["unsafe_execute"]["two_way"]),
        ("four leading systems", "f4", f4["over_trigger"], f4["unsafe_execute"]),
        (
            f"all {n_all} systems",
            "all",
            h["all contestants"]["over_trigger"],
            h["all contestants"]["unsafe_execute"],
        ),
        (
            "words-only cascade",
            "casc",
            h["cascade (words only)"]["over_trigger"],
            h["cascade (words only)"]["unsafe_execute"],
        ),
    ]


COL = {"human": INK, "f4": BLUE, "all": BLUE, "casc": NULL}


def _bars(ax: Any, groups: list, pct_labels: bool = True) -> None:
    for i, (_lab, key, ot, ue) in enumerate(groups):
        c = COL[key]
        alpha = 0.55 if key == "all" else 1.0
        ax.barh(i, -ot["mean"], height=0.62, color=c, alpha=alpha, lw=0, zorder=3)
        ax.barh(i, ue["mean"], height=0.62, color=c, alpha=alpha, lw=0, zorder=3)
        for e, sgn in ((ot, -1), (ue, 1)):
            ax.plot(
                [sgn * e["lo"], sgn * e["hi"]],
                [i, i],
                color=INK2 if key != "human" else "#9a9995",
                lw=0.8,
                zorder=4,
            )
        if pct_labels:
            ax.text(
                -ot["hi"] - 0.015,
                i,
                f"{ot['mean'] * 100:.0f}%",
                ha="right",
                va="center",
                fontsize=FS_TICK,
                color=INK,
            )
            ax.text(
                ue["hi"] + 0.015,
                i,
                f"{ue['mean'] * 100:.0f}%",
                ha="left",
                va="center",
                fontsize=FS_TICK,
                color=INK,
            )
    ax.axvline(0, color=INK, lw=0.6, zorder=5)
    ax.set_ylim(len(groups) - 0.45, -0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def arxiv() -> list:
    groups = _groups()
    W, H = ARXIV_FULL, 2.25
    fig = fig_at(W, H)
    namew, bottom, top = 1.12, 0.4, 0.28
    aw = W - namew - 0.35
    ax = fig.add_axes((namew / W, bottom / H, aw / W, (H - bottom - top) / H))
    _bars(ax, groups)
    ax.set_xlim(-0.45, 0.8)
    ticks = [-0.4, -0.2, 0, 0.2, 0.4, 0.6]
    ax.set_xticks(ticks, [f"{abs(t) * 100:.0f}%" for t in ticks])
    for i, (lab, key, *_r) in enumerate(groups):
        ax.text(
            -0.02,
            i,
            lab,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
            fontweight="bold" if key == "human" else None,
        )
    ax.text(
        -0.02,
        -0.72,
        "← over-trigger on clean calls",
        ha="right",
        va="center",
        fontsize=FS_TICK,
        color=INK2,
    )
    ax.text(
        0.02,
        -0.72,
        "unsafe execution on protective calls →",
        ha="left",
        va="center",
        fontsize=FS_TICK,
        color=INK2,
    )
    ax.set_xlabel("share of calls")
    _ = ORANGE, NULL_BAND
    return save(fig, "fig4_asymmetry")


def error_scatter(axb: Any, mode_key: bool = False) -> int:
    """Each system's operating point: over-trigger on clean calls (x) against unsafe
    execution on protective calls (y). The diagonal is equal rates; the words-only
    cascade and the players are reference points with 95% CIs (players: two-way
    item x player bootstrap). Returns the number of systems above the diagonal."""
    import numpy as np

    m = D.harm()["metrics"]
    lim = 0.9
    axb.plot([0, lim], [0, lim], color=MUTED_TXT, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
    n_side = 0
    for _lab, r in m.items():
        if r.get("role") != "contestant":
            continue
        x, y = r["over_trigger_rate"]["mean"], r["unsafe_execute_rate"]["mean"]
        n_side += y > x
        axb.plot(
            [x],
            [y],
            ls="",
            marker=MODE_MARKER[r["mode"]],
            ms=3.4,
            mfc=SURFACE,
            mec=BLUE,
            mew=0.8,
            zorder=3,
        )
    assert n_side == D.harm()["asymmetry"]["arms_under_side"]
    n_all = D.harm()["asymmetry"]["n_contestants"]
    cg = D.harm()["asymmetry"]["groups"]["cascade (words only)"]
    cx, cy = cg["over_trigger"], cg["unsafe_execute"]
    assert abs(cx["mean"] - m["cascadeopen"]["over_trigger_rate"]["mean"]) < 1e-9
    axb.plot([cx["lo"], cx["hi"]], [cy["mean"]] * 2, color=NULL, lw=0.7, zorder=4)
    axb.plot([cx["mean"]] * 2, [cy["lo"], cy["hi"]], color=NULL, lw=0.7, zorder=4)
    axb.plot([cx["mean"]], [cy["mean"]], ls="", marker="D", ms=3.6, color=NULL, zorder=4.5)
    hum = D.robust2()["harm"]["humans"]
    hx, hy = hum["over_trigger"]["two_way"], hum["unsafe_execute"]["two_way"]
    axb.plot([hx["lo"], hx["hi"]], [hy["mean"]] * 2, color=INK, lw=0.7, zorder=4)
    axb.plot([hx["mean"]] * 2, [hy["lo"], hy["hi"]], color=INK, lw=0.7, zorder=4)
    axb.plot([hx["mean"]], [hy["mean"]], ls="", marker="D", ms=3.8, color=INK, zorder=5)
    axb.set_xlim(0, 0.4)
    axb.set_ylim(0, 0.86)
    axb.set_xticks([0, 0.1, 0.2, 0.3, 0.4], ["0", "10", "20", "30", "40%"])
    axb.set_yticks([0, 0.25, 0.5, 0.75], ["0", "25", "50", "75%"])
    axb.set_xlabel("over-trigger on clean calls")
    axb.set_ylabel("unsafe execution on protective calls")
    axb.annotate(
        "players",
        (hx["mean"], hy["mean"]),
        xytext=(5, -9),
        textcoords="offset points",
        fontsize=FS_TICK,
        va="top",
        fontweight="bold",
    )
    axb.annotate(
        "words-only\ncascade",
        (cx["hi"], cy["mean"]),
        xytext=(2, 0),
        textcoords="offset points",
        fontsize=FS_TICK,
        color=MUTED_TXT,
        va="center",
        linespacing=1.0,
    )
    axb.text(
        0.012,
        0.845,
        f"{n_side} of {n_all} systems\nerr toward the words",
        ha="left",
        va="top",
        fontsize=FS_TICK,
        color=BLUE,
        fontweight="bold",
    )
    p0, p1 = axb.transData.transform([(0, 0), (0.4, 0.4)])
    ang = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    axb.text(
        0.335,
        0.35,
        "equal rates",
        ha="center",
        va="bottom",
        fontsize=FS_TICK,
        color=MUTED_TXT,
        rotation=ang,
        rotation_mode="anchor",
    )
    if mode_key:
        from matplotlib.lines import Line2D
        from style import MODE_LABEL

        hand = [
            Line2D(
                [],
                [],
                ls="",
                marker=MODE_MARKER[k],
                ms=3.4,
                mfc=SURFACE,
                mec=BLUE,
                mew=0.8,
                label=MODE_LABEL[k],
            )
            for k in ("file", "realtime", "local")
        ]
        axb.legend(
            handles=hand,
            loc="upper right",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=FS_TICK,
            handletextpad=0.1,
            borderaxespad=0.1,
            labelspacing=0.25,
            handlelength=1.0,
        )
    return n_side


def interspeech() -> list:
    groups = _groups()
    W, H = IS_COL, 1.45
    fig = fig_at(W, H)
    namew, bottom, top = 0.98, 0.34, 0.22
    ax = fig.add_axes((namew / W, bottom / H, (W - namew - 0.1) / W, (H - bottom - top) / H))
    _bars(ax, groups)
    ax.set_xlim(-0.5, 0.85)
    ticks = [-0.4, -0.2, 0, 0.2, 0.4, 0.6]
    ax.set_xticks(ticks, [f"{abs(t) * 100:.0f}" for t in ticks])
    for i, (lab, key, *_r) in enumerate(groups):
        ax.text(
            -0.03,
            i,
            lab,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FS_TICK,
            fontweight="bold" if key == "human" else None,
        )
    ax.text(-0.03, -0.78, "← over-trigger", ha="right", va="center", fontsize=FS_TICK, color=INK2)
    ax.text(0.03, -0.78, "unsafe execution →", ha="left", va="center", fontsize=FS_TICK, color=INK2)
    ax.set_xlabel("% of clean (left) or protective (right) calls")
    return save(fig, "is_fig4_asymmetry")


def main() -> list:
    return arxiv() + interspeech()


if __name__ == "__main__":
    import style

    style.setup()
    print(main())
