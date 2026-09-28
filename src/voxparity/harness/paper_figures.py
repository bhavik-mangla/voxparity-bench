"""Figures for the paper analyses (``voxparity analyze paper``).

Every figure is drawn from the ``paper_*`` data dict alone, written as PDF (for
the paper) and PNG (for the docs), with one shared style: the reference
categorical palette's first slots (colour-blind validated on adjacent pairs),
a second encoding (marker shape or hatch) wherever more than three categories
share a plot, thin marks, recessive grid, and no dual axes. Output is
deterministic: PDF creation dates and PNG software tags are stripped.

matplotlib is an optional dependency (the ``paper`` extra); the analyses run
without it and the CLI reports that figures were skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Reference palette, light mode (dataviz skill, references/palette.md).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
OUTCOME_COLORS = {
    "correct": BLUE,
    "perceived, acted wrong": ORANGE,
    "perceived, not acted": AQUA,
    "not perceived": YELLOW,
    "wrong (perception n/a)": "#b8b7b2",
}
OUTCOME_HATCH = {
    "correct": "",
    "perceived, acted wrong": "//",
    "perceived, not acted": "..",
    "not perceived": "xx",
    "wrong (perception n/a)": "",
}
MODE_STYLE = {  # colour + marker: never colour alone
    "file": (BLUE, "o"),
    "realtime": (ORANGE, "s"),
    "local": (AQUA, "^"),
    "cascade": (INK2, "D"),
}


def _style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,
            "svg.hashsalt": "voxparity",
            "hatch.linewidth": 0.5,
        }
    )
    return plt


def _save(fig: Any, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{name}.pdf", out / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None, "Producer": None})
    fig.savefig(png, bbox_inches="tight", dpi=200, metadata={"Software": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return [pdf.name, png.name]


def _mean(e: Any) -> float | None:
    return e.get("mean") if isinstance(e, dict) else None


def _err(e: Any) -> tuple[float, float]:
    if not isinstance(e, dict) or e.get("lo") is None:
        return (0.0, 0.0)
    return (e["mean"] - e["lo"], e["hi"] - e["mean"])


class _Labeler:
    """Greedy direct labels: queue them while plotting, ``flush`` once the axis
    limits are final; a label whose anchor sits within ``gap`` (axis fraction) of
    one already placed is skipped, so dense clusters stay legible. The tables
    carry every value, so an unlabelled point is never unidentifiable."""

    def __init__(self, ax: Any, gap: float = 0.04) -> None:
        self.ax, self.gap = ax, gap
        self.queue: list[tuple[str, float, float]] = []

    def __call__(self, text: str, x: float, y: float) -> None:
        self.queue.append((text, x, y))

    def flush(self) -> None:
        placed: list[tuple[float, float]] = []
        inv = self.ax.transAxes.inverted()
        for text, x, y in self.queue:
            tx, ty = inv.transform(self.ax.transData.transform((x, y)))
            if any(abs(tx - px) < 3 * self.gap and abs(ty - py) < self.gap for px, py in placed):
                continue
            placed.append((tx, ty))
            self.ax.annotate(
                text, (x, y), fontsize=5.5, xytext=(3, 2), textcoords="offset points", color=INK2
            )


# --------------------------------------------------------------------------- figures


def fig_taxonomy(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    from voxparity.harness.paper_analyses import NA_OUTCOME, OUTCOMES

    rows = sorted(
        data["taxonomy"],
        key=lambda r: (
            -(r["all_cue_bearing"]["counts"].get("correct", 0) / max(1, r["all_cue_bearing"]["n"]))
        ),
    )
    fig, ax = plt.subplots(figsize=(6.8, 0.26 * len(rows) + 1.2))
    ys = range(len(rows))
    left = [0.0] * len(rows)
    for o in (*OUTCOMES, NA_OUTCOME):
        vals = [
            r["all_cue_bearing"]["counts"].get(o, 0) / max(1, r["all_cue_bearing"]["n"])
            for r in rows
        ]
        if not any(vals):
            continue
        ax.barh(
            list(ys),
            vals,
            left=left,
            color=OUTCOME_COLORS[o],
            hatch=OUTCOME_HATCH[o],
            edgecolor=SURFACE,
            linewidth=1.0,
            height=0.72,
            label=o,
        )
        left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.set_yticks(list(ys), [f"{r['name']} (n={r['all_cue_bearing']['n']})" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of cue-bearing cells (Gemini-TTS)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    ax.set_title("Failure taxonomy: where the cue is lost", loc="left", pad=28)
    return _save(fig, out, "fig1_failure_taxonomy")


def fig_taxonomy_axes(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    import numpy as np

    from voxparity.harness.paper_analyses import AXES

    rows = [r for r in data["taxonomy"] if r["perception_measured"]]
    rows = sorted(rows, key=lambda r: r["name"].lower())
    axes_used = [a for a in AXES if any((r["by_axis"].get(a) or {}).get("n", 0) >= 3 for r in rows)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 0.25 * len(rows) + 1.6), sharey=True)
    for ax, outcome, title in (
        (axs[0], "correct", "correct"),
        (axs[1], "perceived, not acted", "perceived, not acted"),
    ):
        m = np.full((len(rows), len(axes_used)), np.nan)
        for i, r in enumerate(rows):
            for j, a in enumerate(axes_used):
                b = r["by_axis"].get(a)
                if b and b["n"] >= 3:
                    m[i, j] = b["counts"].get(outcome, 0) / b["n"]
        im = ax.imshow(m, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(rows)):
            for j in range(len(axes_used)):
                if not np.isnan(m[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{m[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=SURFACE if m[i, j] > 0.55 else INK,
                    )
        ax.set_xticks(range(len(axes_used)), axes_used, rotation=40, ha="right")
        ax.set_title(f"share {title}", loc="left")
        ax.grid(False)
    axs[0].set_yticks(range(len(rows)), [r["name"] for r in rows])
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02)
    return _save(fig, out, "fig1b_taxonomy_by_axis")


def fig_dissociation(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    d = data["dissociation"]
    rows = [
        r for r in d["arms"] if not r.get("status") and r.get("correct_given_probe", {}).get("gap")
    ]
    rows = sorted(rows, key=lambda r: r["correct_given_probe"]["p_given_true"])
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 0.24 * max(len(rows), 8) + 1.4))
    ax = axs[0]
    for i, r in enumerate(rows):
        c = r["correct_given_probe"]
        ax.plot([c["p_given_false"], c["p_given_true"]], [i, i], color=GRID, lw=1.5, zorder=1)
        ax.scatter(c["p_given_false"], i, color=YELLOW, marker="x", s=28, zorder=2)
        ax.scatter(c["p_given_true"], i, color=BLUE, marker="o", s=24, zorder=3)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(correct action)")
    ax.scatter([], [], color=BLUE, marker="o", label="probe right")
    ax.scatter([], [], color=YELLOW, marker="x", label="probe wrong")
    ax.legend(loc="lower right")
    ax.set_title("Hearing the cue vs acting on it", loc="left")

    ax = axs[1]
    band = d.get("cascade_band_cue_bearing")
    if isinstance(band, dict) and band.get("lo") is not None:
        ax.axhspan(
            band["lo"], band["hi"], color=GRID, alpha=0.9, lw=0, label="cascade null (95% CI)"
        )
    ax.axhline(0, color=INK2, lw=0.6)
    lab = _Labeler(ax)
    for r in d["arms"]:
        if r.get("role", "contestant") != "contestant" or r.get("status"):
            continue
        x, y = r.get("probe_accuracy_cue_bearing"), r.get("audio_minus_twin_cue_bearing")
        if not (isinstance(x, dict) and isinstance(y, dict)):
            continue
        col, mk = MODE_STYLE.get(r["mode"], (INK, "o"))
        ax.errorbar(
            x["mean"],
            y["mean"],
            xerr=[[_err(x)[0]], [_err(x)[1]]],
            yerr=[[_err(y)[0]], [_err(y)[1]]],
            fmt=mk,
            color=col,
            ms=5,
            elinewidth=0.6,
            capsize=0,
        )
        lab(r["name"], x["mean"], y["mean"])
    for mode, (col, mk) in MODE_STYLE.items():
        if mode != "cascade":
            ax.scatter([], [], color=col, marker=mk, label=mode)
    rho = d["across_arm_spearman"].get(
        "probe_accuracy_cue_bearing~audio_minus_twin_cue_bearing", {}
    )
    ax.set_xlabel("probe accuracy, cue-bearing cells")
    ax.set_ylabel("audio - twin, cue-bearing cells")
    r = rho.get("rho")
    ax.set_title(
        "Across arms: Spearman rho = "
        + ("n/a" if r is None else f"{r:.2f}")
        + f" (perm p {rho.get('p_perm')})",
        loc="left",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    lab.flush()
    return _save(fig, out, "fig2_dissociation")


def fig_conduct(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    pairs = data["conduct"]["pairs"]
    arms = {r["label"]: r for r in data["conduct"]["arms"]}
    if not pairs:
        return []
    metrics = (
        ("act_rate", "act rate"),
        ("no_call_rate", "no-call rate"),
        ("audio_credit", "audio credit"),
    )
    fig, axs = plt.subplots(1, 4, figsize=(8.2, 0.42 * len(pairs) + 1.3), sharey=True)
    for ax, (key, title) in zip(axs[:3], metrics, strict=True):
        for i, p in enumerate(pairs):
            f, r = arms[p["file"]][key], arms[p["realtime"]][key]
            ax.plot([_mean(f), _mean(r)], [i, i], color=GRID, lw=1.5, zorder=1)
            ax.scatter(_mean(f), i, color=BLUE, marker="o", s=24, zorder=2)
            ax.scatter(_mean(r), i, color=ORANGE, marker="s", s=24, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left")
    ax = axs[3]
    for i, p in enumerate(pairs):
        lf, lr = p["latency_median_s"]["file"], p["latency_median_s"]["realtime"]
        ax.plot([lf, lr], [i, i], color=GRID, lw=1.5, zorder=1)
        ax.scatter(lf, i, color=BLUE, marker="o", s=24, zorder=2)
        ax.scatter(lr, i, color=ORANGE, marker="s", s=24, zorder=3)
    ax.set_title("median latency (s)", loc="left")
    axs[0].set_yticks(
        range(len(pairs)),
        [p["description"].split(":")[0] + f"\n{p['file']} → {p['realtime']}" for p in pairs],
        fontsize=6,
    )
    axs[0].invert_yaxis()
    axs[0].scatter([], [], color=BLUE, marker="o", label="file mode")
    axs[0].scatter([], [], color=ORANGE, marker="s", label="realtime")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    fig.tight_layout()
    return _save(fig, out, "fig3_realtime_vs_file")


def fig_human_axes(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    hu = data.get("human")
    if not hu or not hu["by_axis"]["rows"]:
        return []
    from voxparity.harness.paper_analyses import display

    rows = hu["by_axis"]["rows"]
    series = (
        ("human", "humans", INK2, "o"),
        ("reference", display(hu["by_axis"]["reference_arm"]), BLUE, "s"),
        ("cascade", "cascade (words only)", ORANGE, "D"),
    )
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    w = 0.26
    for j, (key, lab, col, _mk) in enumerate(series):
        xs = [i + (j - 1) * w for i in range(len(rows))]
        ys = [_mean(r[key]) or 0 for r in rows]
        errs = list(zip(*[_err(r[key]) for r in rows], strict=True))
        ax.bar(xs, ys, width=w - 0.02, color=col, label=lab, edgecolor=SURFACE, linewidth=0.8)
        ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.6)
    ax.set_xticks(
        range(len(rows)),
        [f"{r['axis']}\n(n={r['cells']})" for r in rows],
        rotation=35,
        ha="right",
        fontsize=6,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("selection credit, identical cells")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=3)
    ax.set_title("Humans vs best model vs cascade, by axis", loc="left", pad=22)
    ax.grid(axis="x", visible=False)

    ax = axs[1]
    pts = hu["difficulty_agreement_pooled_models"]["points"]
    ax.scatter(
        [p["human"] for p in pts], [p["models"] for p in pts], s=10, color=BLUE, alpha=0.7, lw=0
    )
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls="--")
    sp = hu["difficulty_agreement_pooled_models"]["spearman_item"]
    ax.set_xlabel("human item mean")
    ax.set_ylabel("audio-native models, item mean")
    ax.set_title(
        "Item difficulty: rho = "
        + ("n/a" if sp.get("rho") is None else f"{sp['rho']:.2f} [{sp['lo']:.2f}, {sp['hi']:.2f}]"),
        loc="left",
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    return _save(fig, out, "fig4_human_vs_models")


def fig_leaderboard(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    """The hero figure: every system's cue-bearing action credit against the
    words-only floor and the human reference, one roster, one Holm family."""
    lb = data.get("leaderboard") or {}
    rows = [r for r in lb.get("rows", []) if r.get("cue_credit") and r.get("role") == "contestant"]
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["cue_credit"]["mean"])
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(rows) + 1.3))
    casc = (lb.get("cascade") or {}).get("cue_credit")
    hum = (lb.get("human") or {}).get("cue_credit")
    if casc:
        ax.axvspan(casc["lo"], casc["hi"], color=GRID, alpha=0.9, lw=0)
        ax.axvline(casc["mean"], color=INK2, lw=0.8, ls="--")
    if hum:
        ax.axvspan(hum["lo"], hum["hi"], color=AQUA, alpha=0.18, lw=0)
        ax.axvline(hum["mean"], color=GREEN, lw=0.9)
    for i, r in enumerate(rows):
        e = r["cue_credit"]
        col, mk = MODE_STYLE.get(r["mode"], (INK, "o"))
        filled = r.get("clears_floor")
        ax.errorbar(
            e["mean"],
            i,
            xerr=[[_err(e)[0]], [_err(e)[1]]],
            fmt=mk,
            mfc=col if filled else SURFACE,
            mec=RED if r.get("below_floor") else col,
            color=col,
            ms=5,
            elinewidth=0.8,
        )
    ax.set_yticks(range(len(rows)), [r["name"] + ("" if r.get("twin") else " †") for r in rows])
    ax.set_xlim(0, 1)
    for mode in ("file", "realtime", "local"):
        col, mk = MODE_STYLE[mode]
        ax.scatter([], [], color=col, marker=mk, label=mode)
    if casc:
        ax.plot([], [], color=INK2, ls="--", label="words-only cascade (floor), 95% CI shaded")
    if hum:
        ax.plot([], [], color=GREEN, label="humans (tool selection), 95% CI shaded")
    ax.legend(loc="lower right")
    ax.set_xlabel("correct action on cue-bearing cells (credit, 95% CI)")
    c = lb.get("counts", {})
    ax.set_title(
        f"{c.get('twin_bearing_clear')} of {c.get('twin_bearing')} audio-native systems with a "
        "text path act on delivery beyond the words-only floor\n(filled = audio moves its "
        "action more than the words alone do, Holm within family; red = significantly below;\n"
        "† no text path: audio accuracy vs the cascade's, its own Holm family)",
        loc="left",
    )
    return _save(fig, out, "fig0_leaderboard")


def fig_robustness(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    rows = data["robustness"].get("arm_vs_cascade", [])
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["estimate"]["mean"])
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(rows) + 1.0))
    ax.axvline(0, color=INK2, lw=0.6)
    for i, r in enumerate(rows):
        e = r["estimate"]
        lofo = r.get("leave_one_family_out") or {}
        if lofo:
            ax.plot(
                [lofo["min_mean"], lofo["max_mean"]],
                [i + 0.22] * 2,
                color=AQUA,
                lw=3,
                solid_capstyle="butt",
            )
        ax.errorbar(
            e["mean"],
            i,
            xerr=[[_err(e)[0]], [_err(e)[1]]],
            fmt="o",
            mfc=BLUE if r["rejects_after_holm"] else SURFACE,
            mec=BLUE,
            color=BLUE,
            ms=5,
            elinewidth=0.8,
        )
    ax.set_yticks(
        range(len(rows)),
        [
            r["name"]
            + ("" if r["metric"].startswith("diff") else " †")
            + {"exploratory": " (expl.)", None: " (outside family)"}.get(r.get("holm_family"), "")
            for r in rows
        ],
    )
    ax.plot([], [], color=AQUA, lw=3, label="leave-one-family-out range")
    ax.errorbar(
        [],
        [],
        xerr=[],
        fmt="o",
        color=BLUE,
        label="estimate, 95% CI (filled = Holm-significant)",
    )
    ax.legend(loc="lower right")
    ax.set_xlabel("vs cascade null, cue-bearing cells (identical cells)")
    ax.set_title(
        "Robustness of the headline deltas († no twin: audio vs cascade audio;\n"
        "outside family = ladder rung or instrument, unadjusted)",
        loc="left",
    )
    return _save(fig, out, "fig5_robustness")


def fig_reaction(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    rows = [r for r in data["reaction"] if r["action_sdt"].get("d_prime") is not None]
    if not rows:
        return []
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, key, title in (
        (axs[0], "action_sdt", "Actions: cue-driven action rate"),
        (axs[1], "probe_sdt", "Probes: cue report rate"),
    ):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls="--")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        lab = _Labeler(ax)
        for r in rows:
            s = r.get(key)
            if not isinstance(s, dict) or s.get("hit_rate") is None:
                continue
            col, mk = MODE_STYLE.get(r["mode"], (INK, "o"))
            ax.scatter(s["fa_rate"], s["hit_rate"], color=col, marker=mk, s=22)
            lab(r["name"], s["fa_rate"], s["hit_rate"])
        lab.flush()
        ax.set_xlabel("false alarm (clean cells)")
        ax.set_ylabel("hit (cue cells)")
        ax.set_title(title, loc="left")
    for mode, (col, mk) in MODE_STYLE.items():
        axs[0].scatter([], [], color=col, marker=mk, label=mode)
    axs[0].legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, out, "fig6_over_under_reaction")


def fig_pareto(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    rows = [r for r in data["pareto"]["arms"] if r["cue_credit"]]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    labs: list[_Labeler] = []
    for ax, key, xlabel, front_key in (
        (
            axs[0],
            "usd_per_cell",
            r"\$ per cell (symlog; free/local = \$0 by basis; unrecorded omitted)",
            "frontier_cost",
        ),
        (axs[1], "median_latency_s", "median latency per audio cell (s)", "frontier_latency"),
    ):
        front = set(data["pareto"][front_key])
        lab = _Labeler(ax)
        pts = sorted(
            (r[key], r["cue_credit"]["mean"])
            for r in rows
            if r["label"] in front and r[key] is not None
        )
        if pts:
            ax.plot(*zip(*pts, strict=True), color=GRID, lw=1.5, zorder=1, drawstyle="steps-post")
        for r in rows:
            if r[key] is None:
                continue
            col, mk = MODE_STYLE.get(r["mode"], (INK, "o"))
            ax.errorbar(
                r[key],
                r["cue_credit"]["mean"],
                yerr=[[_err(r["cue_credit"])[0]], [_err(r["cue_credit"])[1]]],
                fmt=mk,
                color=col,
                mfc=SURFACE if r.get("free_or_local") and key == "usd_per_cell" else col,
                ms=5,
                elinewidth=0.6,
                zorder=2,
            )
            lab(r["name"], r[key], r["cue_credit"]["mean"])
        ax.set_xlabel(xlabel)
        if key == "usd_per_cell":
            ax.set_xscale("symlog", linthresh=1e-4)
        labs.append(lab)
    axs[0].set_ylabel("audio credit, cue-bearing cells")
    for mode, (col, mk) in MODE_STYLE.items():
        axs[1].scatter([], [], color=col, marker=mk, label=mode)
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axs[0].set_title("Cost and latency vs acting on the cue", loc="left")
    fig.tight_layout()
    for lab in labs:
        lab.flush()
    return _save(fig, out, "fig7_pareto")


def fig_psychometrics(plt: Any, data: dict[str, Any], out: Path) -> list[str]:
    rows = data["psychometrics"]["rows"]
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for cue, col, mk, lab in (
        (False, INK2, "o", "neutral cells"),
        (True, BLUE, "s", "cue-bearing cells"),
    ):
        sub = [
            r
            for r in rows
            if (r["axis"] != "neutral") == cue and r["discrimination_rpb"] is not None
        ]
        ax.scatter(
            [r["difficulty_p"] for r in sub],
            [r["discrimination_rpb"] for r in sub],
            s=9,
            color=col,
            marker=mk,
            alpha=0.7,
            lw=0,
            label=lab,
        )
    zero = [r for r in rows if r["difficulty_p"] == 0]
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("pass rate across systems (difficulty p)")
    ax.set_ylabel("point-biserial discrimination")
    ax.set_title(f"Item psychometrics ({len(zero)} cells no system passes)", loc="left")
    ax.legend(loc="lower right")
    return _save(fig, out, "fig8_psychometrics")


FIGURES = (
    fig_taxonomy,
    fig_taxonomy_axes,
    fig_dissociation,
    fig_conduct,
    fig_human_axes,
    fig_leaderboard,
    fig_robustness,
    fig_reaction,
    fig_pareto,
    fig_psychometrics,
)


def render_figures(data: dict[str, Any], out: Path) -> list[str]:
    try:
        plt = _style()
    except ImportError:
        return []
    files: list[str] = []
    for f in FIGURES:
        files += f(plt, data, out)
    return files
