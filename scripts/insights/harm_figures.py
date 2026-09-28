"""Figures for the harm analysis (insights lens 5), drawn from harm.json alone.

Shares the paper figures' style (voxparity.harness.paper_figures): reference
palette, a second encoding (marker shape) wherever colour carries identity, thin
marks, recessive grid, no dual axes, deterministic PDF/PNG output.

1. fig_harm_risk_vs_accuracy: severity-weighted risk (y) against selection
   accuracy (x) per system; marker = serving mode.
2. fig_harm_severity_stack: each system's risk decomposed by severity tier
   (ordinal one-hue ramp, validated with the dataviz validator --ordinal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voxparity.harness.paper_figures import (
    GRID,
    INK,
    INK2,
    MODE_STYLE,
    SURFACE,
    YELLOW,
    _err,
    _Labeler,
    _save,
    _style,
)

# ordinal ramp (dataviz reference blue, steps 700/550/450/350/250): darkest = most severe
TIER_COLORS = ("#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef")
HUMAN_STYLE = (YELLOW, "*")


def _short(name: str) -> str:
    return (
        name.replace(" (local, MLX bf16)", "")
        .replace(" (local, 4-bit)", "")
        .replace(" (BaseTen upstream)", "")
        .replace(" (instrument)", " [instr.]")
    )


def fig_risk_vs_accuracy(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    lab = _Labeler(ax, gap=0.03)
    for key, m in sorted(data["metrics"].items(), key=lambda kv: kv[1]["risk"]["mean"]):
        if key == "humans (game)":
            col, mk = HUMAN_STYLE
        else:
            col, mk = MODE_STYLE.get(m["mode"], (INK, "o"))
        x, y = m["accuracy"]["mean"], m["risk"]["mean"]
        ax.errorbar(
            x,
            y,
            xerr=[[_err(m["accuracy"])[0]], [_err(m["accuracy"])[1]]],
            yerr=[[_err(m["risk"])[0]], [_err(m["risk"])[1]]],
            fmt=mk,
            color=col,
            ms=9 if mk == "*" else 5,
            mec=SURFACE,
            mew=0.6,
            elinewidth=0.5,
            ecolor=GRID if key != "humans (game)" else col,
            zorder=3,
        )
        lab(_short(m["name"]) + (" (own cells)" if key == "humans (game)" else ""), x, y)
    casc = data["metrics"]["cascadeopen"]["risk"]["mean"]
    ax.axhline(casc, color=INK2, lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.annotate(
        "words-only cascade",
        (0.02, casc),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=6,
        color=INK2,
    )
    ax.set_xlabel(
        "selection accuracy, all counterfactual cells (share of calls with the gold tool)"
    )
    ax.set_ylabel("severity-weighted risk\n(life-safety-unsafe equivalents per 100 calls)")
    ax.set_title("Accuracy and real-world risk diverge at the extremes", loc="left")
    for mode, (col, mk) in [*MODE_STYLE.items(), ("human (game)", HUMAN_STYLE)]:
        ax.scatter([], [], color=col, marker=mk, s=40 if mk == "*" else 18, label=mode)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.invert_yaxis()  # safer = higher on the page
    ax.text(
        0.99,
        0.02,
        "up = safer; right = more accurate. 95% item-clustered CIs.\n"
        "Humans are measured on their own 264 cells; see harm.md for paired contrasts.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=INK2,
    )
    fig.tight_layout()
    lab.flush()
    return _save(fig, out, "fig_harm_risk_vs_accuracy")


def fig_severity_stack(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    tiers = list(data["method"]["tier_weights"]["primary"])
    rows = sorted(data["metrics"].values(), key=lambda m: m["risk"]["mean"])
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ys = range(len(rows))
    left = [0.0] * len(rows)
    for t, col in zip(tiers, TIER_COLORS, strict=True):
        vals = [m["risk_by_tier"][t] for m in rows]
        ax.barh(
            list(ys),
            vals,
            left=left,
            height=0.72,
            color=col,
            edgecolor=SURFACE,
            linewidth=1.0,
            label=t,
            zorder=2,
        )
        left = [a + b for a, b in zip(left, vals, strict=True)]
    for y, m in zip(ys, rows, strict=True):
        e = m["risk"]
        ax.errorbar(
            e["mean"],
            y,
            xerr=[[_err(e)[0]], [_err(e)[1]]],
            fmt="none",
            ecolor=INK2,
            elinewidth=0.5,
            capsize=1.5,
            zorder=3,
        )
    ax.set_yticks(list(ys))
    ax.set_yticklabels(
        [_short(m["name"]) + (" *" if m["mode"] == "human" else "") for m in rows],
        fontsize=6.5,
    )
    for tick, m in zip(ax.get_yticklabels(), rows, strict=True):
        if m["mode"] in ("cascade", "human"):
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("severity-weighted risk (life-safety-unsafe equivalents per 100 calls)")
    ax.set_title("Where each system's risk comes from, by severity tier", loc="left")
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), title="severity tier", title_fontsize=6.5
    )
    ax.text(
        0.99,
        -0.09,
        "Sorted safest first. Whiskers: 95% item-clustered CI on the total.\n"
        "* humans on their own 264 cells (not the full bank).",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=INK2,
    )
    fig.tight_layout()
    return _save(fig, out, "fig_harm_severity_stack")


def render(data: dict[str, Any], out: Path) -> list[str]:
    plt = _style()
    files: list[str] = []
    files += fig_risk_vs_accuracy(plt, data, out)
    files += fig_severity_stack(plt, data, out)
    return files
