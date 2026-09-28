"""Bank freeze: the frozen stimulus selection a final run must use.

A freeze pins, for every usable (item, variant), the clip sha256 each engine
would contribute, computed with the ONE gate rule (`runner.gates_passed`, D047).
Variants a freeze excludes carry a ``freeze_exclusion`` gate on every row, so the
runner, `items ready` and the web export honour the exclusion with no second
rule. `check_frozen_bank` lets a run script prove its store and items still
match the freeze before spending anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from voxparity.harness.runner import gates_passed
from voxparity.stimuli.store import StimulusStore

ENGINES: tuple[str, ...] = (
    "gemini",
    "kokoro",
    "human",
    "found",
    "qwen3tts-cv",
    "qwen3tts-vd",
    "aura",
)
EXCLUSION_GATE = "freeze_exclusion"
RUNNABLE_REVIEW = ("screened", "reviewed", "validated")


def variant_axis(variant: Any) -> str:
    """Which manipulation axis a variant belongs to (D062/D084/D090)."""
    if variant.scene is not None:
        return "scene"
    if getattr(variant, "speaker", None) is not None:
        return "speaker"
    if getattr(variant, "channel", None) is not None:
        return "channel"
    return "delivery"


def variant_cue_class(variant: Any) -> str:
    """Same precedence as `scoring.stats.cue_class_map`: scene > speaker > emotion."""
    if variant.scene is not None:
        return f"scene:{variant.scene.kind.value}"
    if getattr(variant, "speaker", None) is not None:
        return f"speaker:{variant.speaker.value}"
    return str(variant.emotion.value)


def exclusion_gate(pattern: str, reason: str, freeze_id: str) -> dict[str, Any]:
    """A decision, not a measurement: it fails the row with its reason attached."""
    return {
        "passed": False,
        "verdict": "excluded",
        "pattern": pattern,
        "reason": reason,
        "freeze": freeze_id,
    }


def select_clips(store: StimulusStore, item: Any, engines: tuple[str, ...] = ENGINES) -> dict:
    """{variant_id: {engine: sha256}} for every clip `gates_passed` admits."""
    out: dict[str, dict[str, str]] = {}
    for v in item.variants:
        sel = {}
        for e in engines:
            rec = store.get(item.id, v.variant_id, e)
            if rec is not None and gates_passed(rec):
                sel[e] = rec.sha256
        out[v.variant_id] = sel
    return out


def ready_on(store: StimulusStore, item: Any, engine: str) -> bool:
    """`voxparity items ready` semantics: every variant has a passing clip on ENGINE."""
    for v in item.variants:
        rec = store.get(item.id, v.variant_id, engine)
        if rec is None or not gates_passed(rec):
            return False
    return True


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen_bank(
    freeze_path: Path, repo_root: Path, store_dir: Path | None = None, strict_items: bool = True
) -> list[str]:
    """Every way the live bank differs from the freeze; empty means frozen.

    Compares each item's per-engine clip selection, and (``strict_items``) each
    item file's sha256, so a run can refuse to start on a drifted bank.
    """
    from voxparity.cli import load_item

    freeze = json.loads(freeze_path.read_text())
    store = StimulusStore(store_dir or repo_root / "stimuli")
    problems: list[str] = []
    for entry in freeze["items"]:
        path = repo_root / entry["path"]
        if not path.exists():
            problems.append(f"{entry['id']}: item file missing ({entry['path']})")
            continue
        if strict_items and file_sha256(path) != entry["file_sha256"]:
            problems.append(f"{entry['id']}: item file changed since the freeze")
        item = load_item(path)
        live = select_clips(store, item, tuple(freeze["engines"]))
        for v in entry["variants"]:
            got = live.get(v["variant_id"], {})
            if got != v["clips"]:
                problems.append(
                    f"{entry['id']}/{v['variant_id']}: clips {got} != frozen {v['clips']}"
                )
    return problems
