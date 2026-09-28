"""Shared loaders and statistics for REANALYSIS D (null-floor and headline robustness).

Run from the pinned bank worktree (its items, store and runs ARE the freeze):

    cd $VXP_BANK && uv run --project $VXP_CODE \
        python $VXP_CODE/scripts/insights/robust_report.py

Inputs (read only; nothing calls a model, nothing is spent):
  * frozen items + freeze : <BANK>/freeze/2026-09-15/freeze.json
  * final runs            : <BANK>/runs/20260915-final-*   (final_analysis.load_arms, D074 dedupe)
  * replay runs           : <BANK>/runs/20260915-exp-replay-{sonnet5,dsv4pro}-gemini
                            (the cascade's cached Whisper transcripts fed to another text LLM;
                            "audio" = Whisper transcript, "text_twin" = gold transcript)
  * human game imports    : <MAIN>/runs/game-20260925/human-*  and  <MAIN>/runs/web-trials.jsonl
  * sibling-lens code/data, read from git refs (never copied into this branch):
      insights/atlas:docs/insights/atlas.json         (the 19 legacy LLM-drafted items)
      insights/harm:scripts/insights/harm_{analysis,rubric}.py  (harm rubric, lens 5)
      insights/human:scripts/insights/human_common.py (human trial loader, lens 6)

Conventions are the paper's (paper_analyses / final_analysis): primary engine
Gemini-TTS, arms >= 90% coverage, invariant controls out, item-clustered
percentile bootstrap (4000 resamples, seed 20260915), bootstrap two-sided p by CI
inversion floored at 1/4000, Holm step-down.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20260915
N_BOOT = 4000
CONF = 0.95

BANK = Path(os.environ.get("VXP_BANK", Path.cwd()))
MAIN = Path(os.environ.get("VXP_MAIN", BANK.parent / "voxparity"))
HERE = Path(__file__).resolve().parents[2]
OUT = HERE / "docs" / "insights"
FREEZE = BANK / "freeze" / "2026-09-15" / "freeze.json"
FINAL_GLOB = str(BANK / "runs" / "20260915-final-*")
REPLAY_RUNS = {
    "sonnet5": BANK / "runs" / "20260915-exp-replay-sonnet5-gemini",
    "dsv4pro": BANK / "runs" / "20260915-exp-replay-dsv4pro-gemini",
}
HUMAN_GLOB = str(MAIN / "runs" / "game-20260925" / "human-*")
CACHE = Path(os.environ.get("ROBUST_CACHE", Path(tempfile.gettempdir()) / "vxp_robust.pkl"))

REF_ATLAS = "insights/atlas"
REF_HARM = "insights/harm"
REF_HUMAN = "insights/human"

TWINLESS_NOTE = (
    "no text path (D035 / realtime / full-duplex): scored as audio credit minus the "
    "cascade's audio credit on the same cells, a LEVEL contrast, not a diff-in-diff"
)


# --------------------------------------------------------------------------- git-ref sources


def git_show(ref_path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(HERE), "show", ref_path], check=True, capture_output=True, text=True
    ).stdout


_MODDIR = Path(tempfile.mkdtemp(prefix="vxp_robust_mods_"))


def import_from_ref(ref: str, relpath: str, name: str) -> Any:
    """Import a sibling lens's module straight from its branch (no copy in this tree)."""
    if name in sys.modules:
        return sys.modules[name]
    p = _MODDIR / f"{name}.py"
    p.write_text(git_show(f"{ref}:{relpath}"))
    if str(_MODDIR) not in sys.path:
        sys.path.insert(0, str(_MODDIR))
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def legacy_items() -> list[str]:
    atlas = json.loads(git_show(f"{REF_ATLAS}:docs/insights/atlas.json"))
    return sorted(k for k, e in atlas["items"].items() if e.get("llm_drafted_legacy"))


# --------------------------------------------------------------------------- loading


def _load() -> dict[str, Any]:
    from voxparity.cli import _iter_item_files, load_item
    from voxparity.harness.final_analysis import load_arm, load_arms

    freeze = json.loads(FREEZE.read_text())
    items: dict[str, Any] = {}
    for d in freeze.get("item_dirs", []):
        for f in _iter_item_files(BANK / d):
            it = load_item(f)
            items[it.id] = it
    arms = load_arms(FINAL_GLOB, items)
    replay = {}
    for lab, path in REPLAY_RUNS.items():
        a = load_arm(path, items)
        if a is not None:
            replay[lab] = a
    return {"items": items, "freeze": freeze, "arms": arms, "replay": replay}


class Bank:
    """Items, the eligible primary-engine arms (paper Context), and the replay floors."""

    def __init__(self) -> None:
        from voxparity.harness.paper_analyses import Context

        if CACHE.exists() and not os.environ.get("ROBUST_REBUILD"):
            raw = pickle.loads(CACHE.read_bytes())
        else:
            raw = _load()
            CACHE.write_bytes(pickle.dumps(raw))
        self.items: dict[str, Any] = raw["items"]
        self.freeze: dict[str, Any] = raw["freeze"]
        self.ctx = Context(raw["arms"], self.items, self.freeze)
        self.primary = self.ctx.primary
        self.by = {a.label: a for a in self.primary}
        self.cascade = self.ctx.cascade
        self.replay = raw["replay"]
        self.contestants = [a for a in self.primary if a.is_audio_native]

    def cue(self, keys: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return self.ctx.cue_keys(keys)

    def neutral(self, keys: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(k for k in keys if self.ctx.axis(k) is None)


# --------------------------------------------------------------------------- statistics


def boot_dist(
    values: list[float], clusters: list[str], *, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, np.ndarray] | None:
    """Point estimate + item-clustered bootstrap draws (identical draws to
    paper_analyses.boot_samples / final_analysis.cluster_bootstrap)."""
    from voxparity.harness.paper_analyses import boot_samples

    return boot_samples(values, clusters, n_boot=n_boot, seed=seed)


def summarise(point: float, boot: np.ndarray, *, shift: float = 0.0) -> dict[str, Any]:
    """95% CI, 90% CI (for TOST), and the two-sided bootstrap p for mean - shift != 0."""
    b = boot - shift
    p = min(1.0, 2.0 * min(float(np.mean(b <= 0)), float(np.mean(b >= 0))))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    lo90, hi90 = np.quantile(boot, [0.05, 0.95])
    return {
        "mean": round(float(point), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "lo90": round(float(lo90), 4),
        "hi90": round(float(hi90), 4),
        "p": max(p, 1.0 / len(boot)),
        "p_floor": p < 1.0 / len(boot),
    }


def cells_test(cells: dict[tuple[str, str], float], *, shift: float = 0.0) -> dict[str, Any] | None:
    ks = sorted(cells)
    if not ks:
        return None
    bs = boot_dist([cells[k] for k in ks], [k[0] for k in ks])
    if bs is None:
        return None
    out = summarise(*bs, shift=shift)
    out["n"] = len(ks)
    out["items"] = len({k[0] for k in ks})
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    from voxparity.harness.paper_analyses import holm as _holm

    return _holm(pvals)


def tost(est: dict[str, Any] | None, margin: float) -> dict[str, Any] | None:
    """Two one-sided tests at alpha .05 via the 90% CI: equivalent iff the 90% CI
    lies inside (-margin, +margin). Also reports the smallest margin that would pass."""
    if not est or est.get("lo90") is None:
        return None
    return {
        "margin": margin,
        "ci90": [est["lo90"], est["hi90"]],
        "equivalent": bool(est["lo90"] > -margin and est["hi90"] < margin),
        "smallest_equivalence_margin": round(max(abs(est["lo90"]), abs(est["hi90"])), 4),
    }


def two_way_boot(
    rows: list[Any],
    item_of: Callable[[Any], str],
    player_of: Callable[[Any], str | None],
    stat: Callable[..., float | None],
    *,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Two-way (item x player) cluster bootstrap (the 'pigeonhole' scheme): items and
    players are resampled independently and each row enters with weight
    (#draws of its item) x (#draws of its player). Rows with player None (model rows)
    get player weight 1, so a joint human-vs-model contrast resamples items for both
    sides and players for the human side only. ``stat(rows, weights)``."""
    items = sorted({item_of(r) for r in rows})
    players = sorted({p for r in rows if (p := player_of(r)) is not None})
    ii = np.array([items.index(item_of(r)) for r in rows])
    pp = np.array([players.index(p) if (p := player_of(r)) is not None else -1 for r in rows])
    w1 = np.ones(len(rows))
    point = stat(rows, w1, w1)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ci = np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items))
        cp = (
            np.bincount(rng.integers(0, len(players), len(players)), minlength=len(players))
            if players
            else np.zeros(0)
        )
        wi = ci[ii].astype(float)
        w = wi * np.where(pp >= 0, cp[np.maximum(pp, 0)] if players else 1, 1)
        v = stat(rows, w, wi)
        if v is not None and not np.isnan(v):
            vals.append(v)
    out: dict[str, Any] = {
        "mean": None if point is None else round(float(point), 4),
        "items": len(items),
        "players": len(players),
        "n": len(rows),
    }
    if vals:
        v = np.array(vals)
        out.update(
            {
                "lo": round(float(np.quantile(v, 0.025)), 4),
                "hi": round(float(np.quantile(v, 0.975)), 4),
                "lo90": round(float(np.quantile(v, 0.05)), 4),
                "hi90": round(float(np.quantile(v, 0.95)), 4),
                "n_boot_ok": len(vals),
            }
        )
    return out


def cluster_suite(
    rows: list[Any],
    item_of: Callable[[Any], str],
    player_of: Callable[[Any], str | None],
    stat: Callable[..., float | None],
    *,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """The same statistic under item-only, player-only and two-way cluster
    bootstraps, plus the leave-one-player-out range (a jackknife over players)."""
    out = {
        "two_way": two_way_boot(rows, item_of, player_of, stat, n_boot=n_boot),
        "item_only": two_way_boot(rows, item_of, lambda r: None, stat, n_boot=n_boot),
        "player_only": two_way_boot(rows, lambda r: "_", player_of, stat, n_boot=n_boot),
    }
    players = sorted({p for r in rows if (p := player_of(r)) is not None})
    pl = [player_of(r) for r in rows]
    vals = []
    for p in players:
        w = np.array([0.0 if q == p else 1.0 for q in pl])
        v = stat(rows, w, np.ones(len(rows)))
        if v is not None:
            vals.append((p, v))
    if vals:
        lo = min(vals, key=lambda x: x[1])
        hi = max(vals, key=lambda x: x[1])
        out["leave_one_player_out"] = {
            "min": round(float(lo[1]), 4),
            "min_dropped": lo[0][:8],
            "max": round(float(hi[1]), 4),
            "max_dropped": hi[0][:8],
        }
    return out


def wmean(x: np.ndarray, w: np.ndarray) -> float | None:
    s = float(w.sum())
    return None if s == 0 else float((x * w).sum() / s)


def fmt(e: dict[str, Any] | None, signed: bool = True, nd: int = 2) -> str:
    if not e or e.get("mean") is None:
        return "n/a"
    f = f"{{:{'+' if signed else ''}.{nd}f}}"
    s = f.format(e["mean"])
    if e.get("lo") is not None:
        s += f" [{f.format(e['lo'])}, {f.format(e['hi'])}]"
    return s


def fp(p: float | None) -> str:
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
