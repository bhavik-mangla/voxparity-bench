"""Loaders for the committed JSON the figures read. No number is typed by hand:
every value drawn comes from one of these files.

  docs/insights/atlas.json                   axes, sectors, per-axis references
  docs/results/final/paper_leaderboard.json  cue-bearing credit, DiD, Holm clears
  docs/results/final/paper_robustness.json   leave-one-family-out ranges
  docs/insights/robustness-v2.json           the three text-LLM floors; player CIs
  docs/insights/review2-interaction.json     frontier-4 membership, E30 groups
  docs/insights/harm.json                    error asymmetry groups
  docs/insights/psychometrics.json           2PL test information, thetas
  docs/results/final/paper_taxonomy.json     failure taxonomy per system
  docs/results/final/paper_conduct.json      realtime vs file pairs
  docs/results/exp/experiments.json          cue-note (oracle / sham) controls
  docs/results/exp/cue-note-split.json       cue note by cue kind (scene split)
  docs/insights/strengthen.json              A6 same-reply named-not-acted
  docs/insights/notefull.json                cue note as description vs label only
  paper/figures/data/*.json                  re-derived cuts (scripts/figs/derive_*.py)
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

from style import DATA, EXP, FIN, INS

from voxparity import private_data


@cache
def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atlas() -> dict[str, Any]:
    return load(INS / "atlas.json")


def leaderboard() -> dict[str, Any]:
    return load(FIN / "paper_leaderboard.json")["leaderboard"]


def robustness() -> dict[str, Any]:
    return load(FIN / "paper_robustness.json")["robustness"]


def robust2() -> dict[str, Any]:
    return load(INS / "robustness-v2.json")


def review2() -> dict[str, Any]:
    return load(INS / "review2-interaction.json")


def harm() -> dict[str, Any]:
    return load(INS / "harm.json")


def psych() -> dict[str, Any]:
    return load(INS / "psychometrics.json")


def taxonomy() -> list[dict[str, Any]]:
    return load(FIN / "paper_taxonomy.json")["taxonomy"]


def conduct() -> dict[str, Any]:
    return load(FIN / "paper_conduct.json")["conduct"]


def experiments() -> dict[str, Any]:
    return load(EXP / "experiments.json")


def cue_note_split() -> dict[str, Any]:
    return load(EXP / "cue-note-split.json")


def strengthen() -> dict[str, Any]:
    return load(INS / "strengthen.json")


def notefull() -> dict[str, Any]:
    return load(INS / "notefull.json")


def derived(name: str) -> dict[str, Any]:
    return load(DATA / f"{name}.json")


# ------------------------------------------------------------------ names
_STRIP = (
    r" \(local, MLX bf16\)",
    r" \(local, 4-bit\)",
    r" \(local\)",
    r" \(file\)",
    r" \(BaseTen upstream\)",
    r" \(instrument\)",
)


def short(name: str) -> str:
    """Display name: serving mode is carried by the marker shape, so the
    parenthesised route/quantisation tags are dropped."""
    for pat in _STRIP:
        name = re.sub(pat, "", name)
    return name.replace("native-audio Live", "native Live")


# the 7 cue axes, in item-count order, plus the 1-cell 'channel' tag
AXES = (
    "delivery emotion",
    "second-speaker",
    "disfluency",
    "scene (environmental)",
    "speaker attribute",
    "sarcasm",
    "slot-noise",
)
AXIS_LABEL = {
    "delivery emotion": "emotional delivery",
    "second-speaker": "second voice",
    "disfluency": "disfluency / silence",
    "scene (environmental)": "environmental sound",
    "speaker attribute": "speaker age",
    "sarcasm": "sarcasm",
    "slot-noise": "masked word",
    "channel": "channel",
}

SECTOR_SHORT = {
    "Banking, payments & fraud": "Banking",
    "Healthcare, pharmacy & triage": "Healthcare",
    "Emergency & public safety": "Emergency",
    "Retail, food & commerce": "Retail",
    "Workplace, recruitment & IT": "Workplace, IT",
    "Telecom, relay & subscriptions": "Telecom",
    "Travel & hospitality": "Travel",
    "Utilities, energy & home services": "Utilities",
    "Automotive & logistics": "Automotive",
    "Aviation, rail & maritime": "Aviation, rail",
    "Elder care & support lines": "Elder care",
    "Collections & debt": "Collections",
    "Insurance": "Insurance",
    "Government & public services": "Government",
}

# one exemplar per axis: (item, variant). Editorial choice of WHICH item; the text
# drawn (variant name, the words' default action, the gold) is read from atlas.json.
# The exemplars are held-out cells, so the choice ships with the bank's private data
# (voxparity.private_data); on a public checkout exemplar() raises PrivateDataUnavailable.
EXEMPLAR: Any = private_data.load(
    "figs/data.json", lambda d: {k: tuple(v) for k, v in d["exemplar"].items()}
)


def exemplar(axis: str) -> dict[str, Any]:
    item, var = EXEMPLAR[axis]
    p = next(p for p in atlas()["patterns"] if p["item"] == item and p["variant"] == var)
    assert p["axis"] == axis and p["protective_cell"], (axis, p)
    return p


def contestants() -> list[dict[str, Any]]:
    return [r for r in leaderboard()["rows"] if r["role"] == "contestant"]


def players_band() -> tuple[dict[str, Any], str]:
    """Players' cue-bearing credit for the Fig. 2 band: the two-way (item x player)
    bootstrap from docs/insights/players-band.json when that file exists, else the
    leaderboard's item-only interval. Returns (interval, basis label)."""
    p = INS / "players-band.json"
    if p.exists():
        b = load(p)["players_band_gemini_cue"]
        return b["two_way"], "two-way item x player bootstrap"
    return leaderboard()["human"]["cue_credit"], "item-clustered bootstrap"
