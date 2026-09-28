"""Rows recorded as not applicable (no transcript twin, no probe) are neither
passes nor failures: report, compare and rescore must leave them out."""

from __future__ import annotations

import json
from pathlib import Path

from voxparity.harness.compare import _cells, summarize_run
from voxparity.harness.report import rescore, summarize

NA_TWIN = {"applicable": False, "reason": "model requires audio in every request"}
NA_PROBE = {"applicable": False, "reason": "driver has no audio path to the LLM"}


def _rows() -> list[dict]:
    base = {"run_id": "r", "driver": "d", "engine": "gemini", "error": None}
    return [
        {**base, "item_id": "i1", "variant_id": "a", "condition": "audio",
         "tool_calls": [{"tool": "x"}], "scores": {"passed": True}},
        {**base, "item_id": "i1", "variant_id": "b", "condition": "audio",
         "tool_calls": [{"tool": "y"}], "scores": {"passed": False}},
        {**base, "item_id": "i1", "variant_id": "", "condition": "text_twin",
         "tool_calls": [], "scores": NA_TWIN},
        {**base, "item_id": "i1", "variant_id": "a", "condition": "probe",
         "tool_calls": [], "scores": NA_PROBE},
    ]  # fmt: skip


def test_summarize_leaves_not_applicable_rows_out() -> None:
    out = summarize(_rows())
    assert out["counts"]["audio"] == "1/2"
    assert out["counts"]["text_twin"] == "0/0"
    assert out["counts"]["probe"] == "0/0"
    assert out["text_twin_pass_rate"] is None and out["probe_pass_rate"] is None
    assert out["audio_minus_twin"] is None
    assert out["per_variant"]["i1/a"]["probe"] == "n/a"


def test_compare_and_run_summary_do_not_crash(tmp_path: Path) -> None:
    assert _cells(_rows(), "text_twin") == {}
    assert _cells(_rows(), "probe") == {}
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows()))
    summarize_run(run)


def test_rescore_keeps_restricted_and_not_applicable_rows(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    rows = [dict(r, restricted="derived fields only") for r in _rows()[:2]] + _rows()[2:]
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    item = type("Item", (), {"id": "i1", "variants": [], "design": "counterfactual"})()
    assert rescore(run, {"i1": item}) == 0
    kept = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
    assert [r["scores"] for r in kept] == [r["scores"] for r in rows]
