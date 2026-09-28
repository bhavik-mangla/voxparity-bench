"""Human-gated engines (qwen3tts-cv/-vd): admitted to scored runs only by a
passing human_check, queued for the review console even when ungated, and every
other engine's gate rule is byte-for-byte what it was before the change."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest

from _held import needs_items, needs_module
from voxparity.harness.runner import HUMAN_GATED_ENGINES, gate_status, gates_passed
from voxparity.stimuli.store import StimulusRecord, StimulusStore

REPO = Path(__file__).parent.parent


def _legacy_gates_passed(rec: Any) -> bool:
    """Verbatim copy of runner.gates_passed before HUMAN_GATED_ENGINES existed."""
    gates = {k: v for k, v in (getattr(rec, "gates", {}) or {}).items() if isinstance(v, dict)}
    if not gates:
        return False
    asr = gates.get("asr_roundtrip")
    if asr is not None and gate_status(asr) == "fail":
        return False
    human = gates.get("human_check")
    if human is not None and gate_status(human) == "fail":
        return False
    decided_by_human = human is not None and gate_status(human) == "pass"
    for name, gate in gates.items():
        if name in ("asr_roundtrip", "human_check"):
            continue
        if name == "cue_check" and decided_by_human:
            continue
        if gate_status(gate) != "pass":
            return False
    return True


# absent, pass, fail, outage (D046), error
_STATES: list[dict | None] = [
    None,
    {"passed": True},
    {"passed": False},
    {"passed": False, "verdict": None},
    {"passed": False, "error": "429"},
]
_GATES = ("asr_roundtrip", "human_check", "cue_check", "ser_check")


def _rec(engine: str, combo: tuple) -> StimulusRecord:
    gates = {name: state for name, state in zip(_GATES, combo, strict=True) if state is not None}
    return StimulusRecord(
        item_id="i",
        variant_id="v",
        sha256="x",
        engine=engine,
        model="m",
        voice="vo",
        prompt="p",
        gates=gates,
    )


def _wav() -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 240)
    return buf.getvalue()


def test_engine_set_is_exactly_the_two_qwen_modes():
    assert frozenset({"qwen3tts-cv", "qwen3tts-vd"}) == HUMAN_GATED_ENGINES


@pytest.mark.parametrize("engine", ["gemini", "aura", "kokoro", "human", "camb", "qwen3tts"])
def test_other_engines_unchanged_on_every_gate_combination(engine):
    for combo in itertools.product(_STATES, repeat=len(_GATES)):
        rec = _rec(engine, combo)
        assert gates_passed(rec) is _legacy_gates_passed(rec), (engine, combo)


@pytest.mark.parametrize("engine", sorted(HUMAN_GATED_ENGINES))
def test_human_gated_engines_need_a_passing_human_check(engine):
    for combo in itertools.product(_STATES, repeat=len(_GATES)):
        rec = _rec(engine, combo)
        asr, human = combo[0], combo[1]
        expected = (
            human is not None
            and gate_status(human) == "pass"
            and _legacy_gates_passed(rec)  # ASR veto and other gates still apply
        )
        assert gates_passed(rec) is expected, (engine, combo)
        if asr is not None and gate_status(asr) == "fail":
            assert gates_passed(rec) is False  # ASR can still veto a human pass


def test_asr_alone_no_longer_admits_a_qwen_clip():
    asr_only = {"asr_roundtrip": {"passed": True}}
    judge_too = {**asr_only, "cue_check": {"passed": True}}
    for gates in (asr_only, judge_too):
        rec = _rec("qwen3tts-cv", tuple(gates.get(g) for g in _GATES))
        assert gates_passed(rec) is False
        assert gates_passed(_rec("gemini", tuple(gates.get(g) for g in _GATES))) is True
    human = {**asr_only, "human_check": {"passed": True}}
    assert gates_passed(_rec("qwen3tts-vd", tuple(human.get(g) for g in _GATES))) is True


@needs_module("voxparity.review.server")
@needs_items("vxp-bnkgr-0001")
def test_review_queue_admits_ungated_human_gated_clips(tmp_path):
    from voxparity.review.server import build_queue

    items = REPO / "items" / "pilot" / "t4" / "vxp-bnkgr-0001.yaml"
    store = StimulusStore(tmp_path)
    for engine, gates in (
        ("qwen3tts-cv", {"asr_roundtrip": {"passed": True}}),  # ungated cue: queued
        ("qwen3tts-vd", {"asr_roundtrip": {"passed": True}, "human_check": {"passed": True}}),
        ("gemini", {"asr_roundtrip": {"passed": True}}),  # ungated gemini: machine first
        ("kokoro", {"asr_roundtrip": {"passed": True}, "cue_check": {"passed": False}}),
    ):
        store.put(
            _wav(),
            StimulusRecord(
                item_id="vxp-bnkgr-0001",
                variant_id="sarcastic",
                sha256="",
                engine=engine,
                model="m",
                voice="v",
                prompt=engine,
                gates=gates,
            ),
        )
    queued = {(c["engine"], c["status"]) for c in build_queue(items, tmp_path, [])["clips"]}
    # D104 widened the console to kokoro (local engine) and found rows:
    # kokoro's judge-fail is now human-reviewable; ungated gemini still waits
    # for the machine gate first; human-gated engines queue even ungated.
    assert queued == {("qwen3tts-cv", "ungated"), ("kokoro", "fail")}
