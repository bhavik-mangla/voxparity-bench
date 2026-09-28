"""Probe rows the paper treats as not applicable ship as not applicable."""

from __future__ import annotations

from voxparity.release import release_records


def test_voicechat_probe_rows_ship_as_not_applicable() -> None:
    row = {
        "run_id": "20260915-final-voicechat11b-gemini",
        "item_id": "vxp-x-0001",
        "variant_id": "a",
        "condition": "probe",
        "driver": "llamacpp:nemotron-voicechat-11b-4bit",
        "engine": "gemini",
        "stimulus_sha256": "abc",
        "scores": {"answer": None, "passed": False},
        "error": "",
        "metrics": {},
    }
    other = dict(row, run_id="20260915-final-gemini37or-gemini")
    kept, _ = release_records(
        [row, other], frozenset({"abc"}), allowed_items=frozenset({"vxp-x-0001"})
    )
    assert kept[0]["scores"]["applicable"] is False and "reason" in kept[0]["scores"]
    assert kept[1]["scores"] == {"answer": None, "passed": False}
