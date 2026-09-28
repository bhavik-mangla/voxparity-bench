"""Found-audio ASR exemption: a `found` row whose recipe certified its transcript
is not vetoed by a failing ASR round-trip (D097 precedent), needs a passing
human_check instead, and every other engine's veto is unchanged."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import pytest

from test_human_gated_engines import _GATES, _STATES, _legacy_gates_passed, _rec
from voxparity.harness.runner import found_transcript_verified, gate_status, gates_passed
from voxparity.stimuli.store import StimulusRecord

REPO = Path(__file__).parent.parent

_VERIFIED = {
    "against": "FAA transcript (recipe transcript_faa)",
    "recipe": "items/found/recipes/faa1549.json",
}
_SCENES = {
    "channel_holder": {
        "channel": {"op": "phone_channel", "found_source": {"transcript_verified": _VERIFIED}}
    },
    "top_level_holder": {"found_source": {"transcript_verified": _VERIFIED}},
}
_NOT_VERIFIED = [
    None,
    {},
    {"channel": {"found_source": {"rebuild_verified": "bit-exact"}}},  # bytes, not words
    {"channel": {"found_source": {"transcript_verified": {"against": "FAA"}}}},  # no recipe
    {"found_source": {"transcript_verified": {"recipe": "r.json", "against": ""}}},
]


def _with_scene(rec: StimulusRecord, scene: dict | None) -> StimulusRecord:
    rec.scene = copy.deepcopy(scene)
    return rec


@pytest.mark.parametrize(
    "engine", ["gemini", "aura", "kokoro", "human", "camb", "qwen3tts-cv", "qwen3tts-vd"]
)
@pytest.mark.parametrize("scene", list(_SCENES.values()))
def test_provenance_on_any_other_engine_changes_nothing(engine, scene):
    """Copying the provenance block onto a non-found row must not lift its veto."""
    for combo in itertools.product(_STATES, repeat=len(_GATES)):
        plain = _rec(engine, combo)
        dressed = _with_scene(_rec(engine, combo), scene)
        assert not found_transcript_verified(dressed)
        assert gates_passed(dressed) is gates_passed(plain), (engine, combo)
        asr = combo[0]
        if asr is not None and gate_status(asr) == "fail":
            assert gates_passed(dressed) is False


@pytest.mark.parametrize("scene", _NOT_VERIFIED)
def test_found_without_transcript_provenance_keeps_legacy_rule(scene):
    for combo in itertools.product(_STATES, repeat=len(_GATES)):
        rec = _with_scene(_rec("found", combo), scene)
        assert not found_transcript_verified(rec)
        assert gates_passed(rec) is _legacy_gates_passed(rec), combo


@pytest.mark.parametrize("scene", list(_SCENES.values()))
def test_certified_found_row_needs_human_and_ignores_asr_failure(scene):
    for combo in itertools.product(_STATES, repeat=len(_GATES)):
        rec = _with_scene(_rec("found", combo), scene)
        assert found_transcript_verified(rec)
        human = combo[1]
        others_ok = all(
            state is None or gate_status(state) == "pass" or name == "cue_check"
            for name, state in zip(_GATES, combo, strict=True)
            if name not in ("asr_roundtrip", "human_check")
        )
        expected = human is not None and gate_status(human) == "pass" and others_ok
        assert gates_passed(rec) is expected, combo


def test_failing_asr_is_kept_as_evidence_not_rewritten():
    rec = _with_scene(
        _rec("found", ({"passed": False, "wer": 0.53}, {"passed": True}, None, None)),
        _SCENES["channel_holder"],
    )
    assert gates_passed(rec) is True
    assert rec.gates["asr_roundtrip"] == {"passed": False, "wer": 0.53}
