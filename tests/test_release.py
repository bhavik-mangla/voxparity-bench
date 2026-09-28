"""The public bundle physically excludes barred engines (docs/RELEASE-EXPORT.md),
and the Deepgram-free re-gate normalises only written forms (PX-010)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from voxparity.harness.export_web import PUBLIC_ENGINES
from voxparity.release import (
    EXCLUDED_ENGINES,
    LICENCE_BASIS,
    ReleaseError,
    redact,
    release_manifest,
    release_records,
)

PASS = {"asr_roundtrip": {"passed": True}, "cue_check": {"passed": True}}


def _row(engine: str, sha: str, gates: dict | None = None) -> dict:
    return {
        "item_id": "i",
        "variant_id": "v",
        "sha256": sha,
        "engine": engine,
        "gates": gates or PASS,
    }


@pytest.mark.parametrize("engine", sorted(EXCLUDED_ENGINES))
def test_every_excluded_engine_is_dropped(engine: str):
    rel = release_manifest([_row(engine, "x")])
    assert rel.rows == [] and rel.dropped[f"engine:{engine}"] == 1


def test_allow_list_and_deny_list_are_disjoint_and_all_allowed_have_a_basis():
    assert not set(PUBLIC_ENGINES) & EXCLUDED_ENGINES
    for engine in PUBLIC_ENGINES:
        assert engine in LICENCE_BASIS


def test_unknown_engine_is_dropped_not_shipped():
    assert release_manifest([_row("new-vendor", "x")]).rows == []


def test_allowed_rows_ship_with_licence_basis_and_found_is_recipe_only():
    rel = release_manifest([_row("gemini", "a"), _row("found", "b"), _row("kokoro", "c")])
    by = {r["sha256"]: r for r in rel.rows}
    assert set(by) == {"a", "b", "c"}
    assert all("licence_basis" in r for r in rel.rows)
    assert by["b"]["distribution"] == "recipe-only"
    assert by["a"]["distribution"] == "audio"


def test_qwen3tts_needs_a_human_pass():
    rows = [
        _row("qwen3tts-cv", "no-human"),
        _row("qwen3tts-cv", "human", {**PASS, "human_check": {"passed": True}}),
    ]
    assert release_manifest(rows).shas == {"human"}


def test_freeze_filter_keeps_only_frozen_usable():
    rel = release_manifest([_row("gemini", "a"), _row("gemini", "b")], frozenset({"a"}))
    assert rel.shas == {"a"} and rel.dropped["not-frozen-usable"] == 1


def test_missing_basis_is_an_error(monkeypatch):
    import voxparity.release as r

    monkeypatch.delitem(r.LICENCE_BASIS, "gemini")
    with pytest.raises(ReleaseError):
        release_manifest([_row("gemini", "a")])


def test_records_follow_released_stimuli_and_drop_deepgram_drivers():
    recs = [
        {"driver": "openrouter:x", "engine": "gemini", "stimulus_sha256": "a", "error": ""},
        {"driver": "openrouter:x", "engine": "gemini", "stimulus_sha256": "zz", "error": ""},
        {"driver": "openrouter:x", "engine": "gemini", "stimulus_sha256": "", "error": ""},
        {"driver": "openrouter:x", "engine": "aura", "stimulus_sha256": "", "error": ""},
        {"driver": "cascade-closed:deepgram-nova3+x", "engine": "gemini", "stimulus_sha256": "a"},
    ]
    kept, dropped = release_records(recs, frozenset({"a"}))
    assert [r["stimulus_sha256"] for r in kept] == ["a", ""]
    assert sum(dropped.values()) == 3


def test_error_strings_are_redacted():
    msg = "Rate limit for org-AbCdEf123456 user_abcdef99 key sk-proj1234567890"
    out = redact(msg)
    assert "AbCdEf123456" not in out and "abcdef99" not in out and "1234567890" not in out
    kept, _ = release_records(
        [
            {
                "driver": "d",
                "engine": "gemini",
                "stimulus_sha256": "a",
                "error": msg,
                "metrics": {"usage": {"cost": 1.0, "is_byok": False}},
            }
        ],
        frozenset({"a"}),
    )
    assert "org-REDACTED" in kept[0]["error"]
    assert kept[0]["metrics"]["usage"] == {"cost": 1.0}


# --- the Deepgram-free re-gate (scripts/regate_local_asr.py) -----------------


def _regate_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "regate_local_asr.py"
    spec = importlib.util.spec_from_file_location("regate_local_asr", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["regate_local_asr"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_spoken_readings_cover_cardinal_paired_and_digitwise():
    m = _regate_module()
    assert "eight thousand" in m.spoken_readings("8000")
    assert "fourteen twenty" in m.spoken_readings("1420")
    assert "fifteen thirty nine" in m.spoken_readings("1539")
    assert "three five zero" in m.spoken_readings("350")


def test_written_forms_normalise_but_wrong_numbers_stay_wrong():
    from test_schemas import make_item
    from voxparity.stimuli.validate import asr_roundtrip_gate

    m = _regate_module()
    item = make_item(
        transcript="I'd like to send eight thousand dollars to my grandson's account, please."
    )

    def best(hyp: str) -> float:
        return min(
            float(asr_roundtrip_gate(lambda _b, t=c: t, item, b"").detail["wer"])
            for c in m.hypothesis_candidates(hyp)
        )

    assert best("I'd like to send $8,000 to my grandson's account, please.") == 0.0
    assert best("I'd like to send $9,000 to my grandson's account, please.") > 0.0
