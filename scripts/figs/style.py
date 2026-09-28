# ruff: noqa: RUF001  (typographic minus signs in tick labels are intended)
"""One style for every VoxParity paper figure (v3).

Every figure is drawn at the width it is printed at and included with
``width=<that width>`` (scale 1.0), so the point sizes below are the point sizes
on paper. Nothing in any figure is smaller than 7 pt.

Widths (measured from the templates, not guessed):
  arXiv  : article, letterpaper, 1 in margins  -> \\textwidth = 6.5 in
  IS     : Interspeech two-column             -> \\columnwidth = 80 mm = 3.15 in

Semantic palette (the same role gets the same colour in every figure):
  humans / players            INK     #0b0b0b  (line, diamond, or 12% band; never a hue)
  words-only cascade (null)   NULL    #8f8e8a  (dashed line) + NULL_BAND #e4e3df
  audio-native systems        BLUE    #2a78d6  marker shape = serving mode
                                               (o file API, s realtime, ^ local)
  emotional delivery; and
  "heard, not acted on"       ORANGE  #eb6834
  other cue kinds             VIOLET  #4a3aa7
  not perceived / missed      MUTED   #b9b8b3 (hollow marker edge, bar fill #d0cfca)

Colour-blind check (dataviz skill validator, run on the three categorical hues
against the white page and the reference surface, all pairs):
  node validate_palette.js "#2a78d6,#eb6834,#4a3aa7" --mode light --pairs all
    lightness PASS, chroma PASS, CVD worst ΔE 13.0 (deutan, violet↔blue) PASS,
    normal-vision worst ΔE 16.3 PASS, contrast >= 3:1 PASS -> ALL CHECKS PASS
Ink and the two greys are neutral reference roles, not categorical hues (they fail
the validator's chroma/lightness band by design); their CVD separation from every
hue is >= 11.1 ΔE. Identity is never colour-alone: humans are also a diamond or a
labelled band, the cascade is also dashed, serving mode is a marker shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
INS = ROOT / "docs" / "insights"
FIN = ROOT / "docs" / "results" / "final"
EXP = ROOT / "docs" / "results" / "exp"
DATA = OUT / "data"

# ---------------------------------------------------------------- sizes (inches, points)
ARXIV_FULL = 6.5
ARXIV_HALF = 3.15
IS_COL = 3.15
IS_FULL = 6.85

FS = 7.5  # labels, annotations
FS_TICK = 7.0  # tick labels, the floor
FS_PANEL = 8.5  # panel letters (bold)

# ---------------------------------------------------------------- palette
INK = "#0b0b0b"
INK2 = "#52514e"  # secondary text
MUTED_TXT = "#6d6c68"
SURFACE = "#ffffff"
GRID = "#ecebe7"
AXIS = "#8f8e8a"
NULL = "#8f8e8a"
NULL_BAND = "#e4e3df"
HUMAN_BAND = "#0b0b0b"  # drawn at alpha HUMAN_ALPHA
HUMAN_ALPHA = 0.10
BLUE = "#2a78d6"
BLUE_LIGHT = "#b7d3f6"  # blue step 150: underlays (leave-one-family-out range)
ORANGE = "#eb6834"
ORANGE_LIGHT = "#f6b89c"
VIOLET = "#4a3aa7"
MUTED = "#b9b8b3"
MUTED_FILL = "#d0cfca"

MODE_MARKER = {"file": "o", "realtime": "s", "local": "^", "cascade": "D"}
MODE_LABEL = {"file": "file API", "realtime": "realtime", "local": "local"}
CUE_COLOR = {"emotion": ORANGE, "other": VIOLET}


def setup() -> Any:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FS,
            "axes.titlesize": FS,
            "axes.labelsize": FS,
            "axes.labelcolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": FS_TICK,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "text.color": INK,
        }
    )
    return plt


def panel(ax: Any, letter: str, x: float = -0.02, y: float = 1.0, ha: str = "right") -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight="bold",
        ha=ha,
        va="bottom",
    )


def fig_at(width: float, height: float) -> Any:
    return plt.figure(figsize=(width, height))


OVERLAPS: dict[str, list[tuple[str, str]]] = {}


def save(fig: Any, name: str, out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    OVERLAPS[name] = check_no_overlap(fig)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        # no bbox_inches: the file is exactly the authored size, so \includegraphics
        # at that width prints at scale 1.0
        fig.savefig(p, dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def xgrid(ax: Any) -> None:
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def null_line(
    ax: Any, x: float, band: tuple[float, float] | None = None, vertical: bool = True
) -> None:
    if band is not None:
        (ax.axvspan if vertical else ax.axhspan)(
            band[0], band[1], color=NULL_BAND, lw=0, zorder=0.5
        )
    (ax.axvline if vertical else ax.axhline)(x, color=NULL, lw=0.9, ls=(0, (3, 2)), zorder=1)


def human_band(ax: Any, mean: float, lo: float, hi: float, vertical: bool = True) -> None:
    """Players: solid ink line, CI as a very light ink band edged in thin ink
    (the cascade's band is warm grey and unedged, its line dashed)."""
    span = ax.axvspan if vertical else ax.axhspan
    line = ax.axvline if vertical else ax.axhline
    span(lo, hi, color=HUMAN_BAND, alpha=HUMAN_ALPHA * 0.6, lw=0, zorder=0.6)
    for v in (lo, hi):
        line(v, color=INK, lw=0.35, alpha=0.6, zorder=1.05)
    line(mean, color=INK, lw=1.0, zorder=1.1)


def mode_handles(
    modes: tuple[str, ...] = ("file", "realtime", "local"), color: str = BLUE
) -> list[Any]:
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], ls="", marker=MODE_MARKER[m], color=color, ms=4.2, label=MODE_LABEL[m])
        for m in modes
    ]


def direct_labels(
    ax: Any,
    pts: list[tuple[str, float, float]],
    offsets: dict[str, tuple[float, float]] | None = None,
    max_labels: int = 6,
    fs: float = FS_TICK,
    color: str = INK,
    pad_pt: float = 1.5,
) -> list[Any]:
    """Label at most ``max_labels`` points with leader lines and no collisions.

    ``pts`` = (text, x, y) in data coordinates. Each label starts at its requested
    offset (points; default up-right) and is pushed along y, alternately up and
    down, until its box clears every other label and every data marker box.
    Returns the text artists; raises if a clean placement is impossible, so an
    overlap can never ship silently.
    """
    assert len(pts) <= max_labels, f"{len(pts)} labels > {max_labels}"
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed: list[Bbox] = []
    marker_boxes = []
    for _t, x, y in pts:
        px, py = ax.transData.transform((x, y))
        marker_boxes.append(Bbox.from_extents(px - 3, py - 3, px + 3, py + 3))
    arts = []
    axbox = ax.get_window_extent(rend)
    for text, x, y in pts:
        dx, dy = (offsets or {}).get(text, (6.0, 4.0))
        best = None
        for step in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 10, -10]:
            t = ax.annotate(
                text,
                (x, y),
                xytext=(dx, dy + step * 4.0),
                textcoords="offset points",
                fontsize=fs,
                color=color,
                ha="left" if dx >= 0 else "right",
                va="center",
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": MUTED,
                    "lw": 0.5,
                    "shrinkA": 0,
                    "shrinkB": 2.5,
                },
            )
            bb = t.get_window_extent(rend).padded(pad_pt)
            clash = any(bb.overlaps(o) for o in placed) or any(bb.overlaps(m) for m in marker_boxes)
            inside = axbox.padded(40).contains(bb.x0, bb.y0) and axbox.padded(40).contains(
                bb.x1, bb.y1
            )
            if not clash and inside:
                best = (t, bb)
                break
            t.remove()
        if best is None:
            raise RuntimeError(f"cannot place label {text!r} without overlap")
        placed.append(best[1])
        arts.append(best[0])
    return arts


def check_no_overlap(fig: Any, pad: float = 0.0) -> list[tuple[str, str]]:
    """Every pair of visible text artists whose boxes intersect (after drawing)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(rend)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        texts.append((t.get_text(), bb.padded(pad)))
    bad = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1], texts[j][1]
            # shrink by 0.5 px so touching baselines are not flagged
            if a.x0 + 0.5 < b.x1 and b.x0 + 0.5 < a.x1 and a.y0 + 0.5 < b.y1 and b.y0 + 0.5 < a.y1:
                bad.append((texts[i][0], texts[j][0]))
    return bad


def fmt(x: float, nd: int = 2, signed: bool = False) -> str:
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "−")
