# ruff: noqa: RUF001
"""Lens 4 figures from docs/insights/models.json.

    uv run --extra paper python scripts/insights/models_figures.py

fig_models_1_release_vs_listening: release date vs vs-floor, vendor lines connected.
fig_models_2_realtime_vs_file: slope chart, file -> realtime on identical cells
    (cue-bearing credit, probe accuracy, act rate).
fig_models_3_hear_vs_use: change in cue-bearing credit between versions split into
    a hearing part (probe accuracy) and a using part (credit given the probe).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "docs/insights/models.json").read_text())
OUT = ROOT / "docs/insights/figures"
S = D["systems"]

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
VENDOR = {"Google": "#2a78d6", "OpenAI": "#eb6834", "Alibaba": "#1baf7a", "Xiaomi": "#eda100"}
OTHER = "#8d8c87"
MARK = {"file": "o", "realtime": "s", "local": "^"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,
    }
)

LINES = [  # (labels in release order, vendor, linestyle, legend)
    (["gemini37or", "gemini38or"], "Google", "-", "Gemini Flash (file)"),
    (["gem25native", "geminilive", "gemini38live"], "Google", "--", "Gemini Live"),
    (["gptaudiomini", "gptaudio"], "OpenAI", ":", "gpt-audio (file, mini/full)"),
    (["gptrt21mini", "gptrt21"], "OpenAI", "--", "gpt-realtime-2.1 (mini/full)"),
    (["qwen25omni7b", "qwen3omni", "qwen38omni"], "Alibaba", "-", "Qwen Omni (file/local)"),
    (["qwenrtflash", "qwen38rtflash"], "Alibaba", "--", "Qwen Omni Flash RT"),
    (["mimo25", "mimo26flash", "mimo26pro"], "Xiaomi", "-", "MiMo"),
]


def d(s: str) -> date:
    return date(*map(int, s.split("-")))


def color(lab: str) -> str:
    return VENDOR.get(S[lab]["vendor"], OTHER)


def contestants():
    return [lab for lab, r in S.items() if r["role"] == "contestant" and r["vs_floor"]]


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color=INK2, lw=1)
    ax.text(d("2025-06-01"), -0.018, "words-only floor (cascade)", color=INK2, fontsize=7)
    for labs, v, ls, name in LINES:
        labs = [lab for lab in labs if lab in S]
        ax.plot(
            [d(S[lab]["release"]) for lab in labs],
            [S[lab]["vs_floor"]["mean"] for lab in labs],
            ls=ls,
            color=VENDOR[v],
            lw=1.4,
            label=name,
            zorder=2,
        )
    for lab in contestants():
        r = S[lab]
        x, e = d(r["release"]), r["vs_floor"]
        ax.errorbar(
            x,
            e["mean"],
            yerr=[[e["mean"] - e["lo"]], [e["hi"] - e["mean"]]],
            fmt="none",
            ecolor=color(lab),
            alpha=0.35,
            lw=1,
            zorder=1,
        )
        ax.scatter(
            x,
            e["mean"],
            marker=MARK[r["serving"]],
            s=34,
            color=color(lab),
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
    lbl = {
        "qwen38omni": "Qwen3.8-Omni",
        "gemini37or": "gemini-3.7",
        "inkling": "Inkling",
        "stepaudio3": "StepAudio 3",
        "qwenrtflash": "Qwen3.5 RT",
        "gptaudiomini": "gpt-audio-mini",
        "voicechat11b": "VoiceChat 11B",
        "voxtral": "Voxtral",
        "phi4mm": "Phi-4-mm",
        "musespark12": "Muse Spark",
        "grokvoice": "Grok Voice",
        "nemotron": "Nemotron Omni",
        "qwenaudio31rt": "Qwen-Audio-3.1 RT",
        "gemma412b": "Gemma-4-12B",
        "gemma4e4b": "Gemma-4-E4B",
        "gem25native": "Gemini 2.5 Live",
        "gemini38live": "3.8 Live",
        "qwen25omni7b": "Qwen2.5-Omni-7B",
        "qwen3omni": "Qwen3-Omni-30B",
        "mimo25": "MiMo-V2.5",
    }
    off = {
        "inkling": (-30, 6),
        "gemini37or": (-12, 8),
        "musespark12": (5, 4),
        "gemini38live": (6, -9),
        "gemma412b": (-62, 5),
        "grokvoice": (-18, -11),
        "qwenaudio31rt": (6, -8),
        "stepaudio3": (6, 4),
        "phi4mm": (5, 5),
    }
    lbl["gemini38or"] = "gemini-3.8"
    lbl["mimo26pro"] = "MiMo-V2.6-Pro"
    off["gemini38or"] = (-52, 3)
    off["mimo26pro"] = (6, -6)
    for lab, t in lbl.items():
        if lab in S and S[lab]["vs_floor"]:
            ax.annotate(
                t,
                (d(S[lab]["release"]), S[lab]["vs_floor"]["mean"]),
                xytext=off.get(lab, (4, 3)),
                textcoords="offset points",
                fontsize=6.3,
                color=INK2,
            )
    for m, name in (("o", "file"), ("s", "realtime"), ("^", "local")):
        ax.scatter([], [], marker=m, color=INK2, s=26, label=name)
    ax.scatter([], [], marker="o", color=OTHER, s=26, label="other vendors")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("listening: vs words-only floor (DiD, cue-bearing)")
    ax.set_xlabel("public release of the served snapshot")
    p = D["properties"]["release_date_vs_listening"]
    t = D["properties"]["release_date_vs_listening_slope_per_month"]
    ax.set_title(
        f"Newer is not reliably better at listening: Spearman {p['estimate']:+.2f} "
        f"[{p['lo']:+.2f}, {p['hi']:+.2f}], slope {t['estimate'] * 100:+.1f} pts/month "
        f"[{t['lo'] * 100:+.1f}, {t['hi'] * 100:+.1f}], n={p['n_systems']}",
        fontsize=8,
        loc="left",
    )
    ax.legend(loc="lower left", ncol=2, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_models_1_release_vs_listening")


SHORT = {
    "gemini38live": "G3.8 Live",
    "gemini38or": "G3.8 file",
    "geminilive": "G3.1 Live",
    "gemini37or": "G3.7 file",
    "gem25native": "G2.5 Live",
    "gptrt21": "gpt-rt-2.1",
    "gptaudio": "gpt-audio",
    "gptrt21mini": "gpt-rt-mini",
    "gptaudiomini": "gpt-audio-mini",
    "qwen38rtflash": "Q3.8 RT",
    "qwen38omni": "Q3.8 file",
    "qwenrtflash": "Q3.5 RT",
    "qwen3omni": "Q3-Omni",
}


def fig2():
    rows = D["serving_pairs"]
    keys = [
        ("d_cue_credit", "cue_credit", "cue-bearing credit"),
        ("d_probe_cue", "probe_acc_cue", "probe accuracy (hearing)"),
        ("d_act_rate", "act_rate", "act rate (acting)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.6), sharey=False)
    for ax, (dk, sk, title) in zip(axes, keys, strict=True):
        ends = []
        for r in rows:
            f, rt = S[r["old"]], S[r["new"]]
            if not f.get(sk) or not rt.get(sk):
                continue
            v = S[r["new"]]["vendor"]
            ls = "-" if r["same_generation"] else "--"
            ax.plot(
                [0, 1],
                [f[sk]["mean"], rt[sk]["mean"]],
                color=VENDOR.get(v, OTHER),
                ls=ls,
                lw=1.4,
                marker="o",
                ms=4,
                markeredgecolor=SURFACE,
            )
            ends.append((rt[sk]["mean"], SHORT[r["new"]] + " vs " + SHORT[r["old"]]))
        ys = [e[0] for e in ends]
        gap = (max(ys) - min(ys) + 0.05) * 0.075
        placed = []
        for y, t in sorted(ends, reverse=True):
            y2 = min(y, placed[-1] - gap) if placed else y
            placed.append(y2)
            ax.annotate(
                t,
                (1, y),
                xytext=(1.06, y2),
                textcoords="data",
                fontsize=5.6,
                color=INK2,
                va="center",
                arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6),
            )
        m = D["serving_meta"].get(dk)
        ax.set_title(
            f"{title}\npooled RT-file {m['pooled_re']:+.2f}, I²={m['I2']:.2f}",
            fontsize=7.5,
            loc="left",
        )
        ax.set_xticks([0, 1], ["file", "realtime"])
        ax.set_xlim(-0.15, 2.3)
    axes[0].plot([], [], color=INK2, ls="-", label="same generation")
    axes[0].plot([], [], color=INK2, ls="--", label="different version")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.5)
    fig.suptitle(
        "Realtime vs file within a vendor (identical cells): hearing drops in 6/7 pairs; "
        "acting moves both ways",
        fontsize=8,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_models_2_realtime_vs_file")


def fig3():
    rows = [
        g
        for g in D["generations"]
        if "hearing_part" in g["decomposition"] and g["kind"].startswith(("generation", "size"))
    ]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, g in enumerate(rows):
        dc = g["decomposition"]
        h, u = dc["hearing_part"]["mean"], dc["using_part"]["mean"]
        ax.barh(i + 0.18, h, height=0.34, color="#2a78d6", edgecolor=SURFACE, lw=1)
        ax.barh(i - 0.18, u, height=0.34, color="#eb6834", edgecolor=SURFACE, lw=1)
        tot = dc["delta_cue_credit"]
        ax.plot([tot["lo"], tot["hi"]], [i, i], color=INK, lw=1)
        ax.plot(tot["mean"], i, marker="D", color=INK, ms=4)
    ax.set_yticks(
        range(len(rows)),
        [
            f"{g['new_name'].replace(' (local)', '')} vs {g['old_name'].replace(' (local)', '')}"
            for g in rows
        ],
        fontsize=6.5,
    )
    ax.axvline(0, color=INK2, lw=1)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(
                color="#2a78d6",
                label="hearing part: Δ probe accuracy × (credit|heard − credit|missed)",
            ),
            Patch(color="#eb6834", label="using part: Δ credit at fixed probe accuracy"),
            Line2D(
                [], [], marker="D", color=INK, lw=1, label="total Δ cue-bearing credit [95% CI]"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.35, -0.13),
        ncol=2,
        fontsize=6.3,
    )
    ax.set_xlabel("newer / larger minus older / smaller, identical cue-bearing cells")
    ax.set_title(
        "Version changes move actions through USE, not through HEARING", fontsize=8, loc="left"
    )
    fig.tight_layout()
    save(fig, "fig_models_3_hear_vs_use")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})  # byte-stable
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote", OUT)
