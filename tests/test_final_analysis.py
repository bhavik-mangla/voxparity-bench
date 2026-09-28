"""Final-matrix analysis: CI method, speaker_varies exclusion, n/a vs 0 for
capability-limited arms, same-cell pairing, skip-vs-error bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from test_schemas import make_item
from voxparity.harness.final_analysis import (
    cluster_bootstrap,
    dissociation,
    headline_row,
    hu_bootstrap,
    load_arm,
    null_floor,
    paired_bootstrap,
    parse_run_name,
    render_template,
    same_cell,
)
from voxparity.harness.report import probe_hu
from voxparity.schemas.item import PerceptionProbe
from voxparity.scoring.stats import cue_class_map


def _action(passed: bool) -> dict[str, Any]:
    return {"passed": passed, "credit": 1.0 if passed else 0.0}


def rows_for(
    item_id: str,
    *,
    engine: str = "gemini",
    audio: dict[str, bool] | None = None,
    twin: dict[str, bool] | None = None,
    probe: dict[str, str] | None = None,
    twin_na: bool = False,
    probe_na: bool = False,
    speaker_varies: bool = False,
    skip: tuple[str, ...] = (),
    error: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    base = {"item_id": item_id, "engine": engine, "driver": "d", "design": "counterfactual"}
    out: list[dict[str, Any]] = []
    audio = audio if audio is not None else {"happy": True, "angry": True}
    for vid, ok in audio.items():
        out.append(
            {
                **base,
                "variant_id": vid,
                "condition": "audio",
                "tool_calls": [{"tool": "a"}],
                "scores": _action(ok),
                "error": "",
                "metrics": {"latency_s": 1.0},
            }
        )
    for vid in skip:
        out.append(
            {
                **base,
                "variant_id": vid,
                "condition": "audio",
                "tool_calls": [],
                "scores": {},
                "error": "skipped: stimulus did not pass validation gates",
            }
        )
    for vid in error:
        out.append(
            {
                **base,
                "variant_id": vid,
                "condition": "audio",
                "tool_calls": [],
                "scores": {},
                "error": "HTTP 429: rate limit",
            }
        )
    if twin_na:
        out.append(
            {
                **base,
                "variant_id": "",
                "condition": "text_twin",
                "tool_calls": [],
                "scores": {"applicable": False, "reason": "audio required"},
                "error": "",
            }
        )
    else:
        tw = twin if twin is not None else {"happy": True, "angry": False}
        out.append(
            {
                **base,
                "variant_id": "",
                "condition": "text_twin",
                "tool_calls": [{"tool": "a"}],
                "scores": {v: _action(ok) for v, ok in tw.items()},
                "error": "",
            }
        )
    probe = probe if probe is not None else {"happy": "happy", "angry": "angry"}
    for vid, ans in probe.items():
        if probe_na:
            scores: dict[str, Any] = {"applicable": False, "reason": "no audio path"}
        else:
            scores = {"answer": ans, "gold": vid, "passed": ans == vid}
        row = {
            **base,
            "variant_id": vid,
            "condition": "probe",
            "tool_calls": [],
            "scores": scores,
            "error": "",
            "stimulus_sha256": f"{item_id}-{vid}",
        }
        if speaker_varies:
            row["speaker_varies"] = True
        out.append(row)
    return out


def write_run(root: Path, name: str, rows: list[dict[str, Any]]) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


@pytest.fixture
def items() -> dict[str, Any]:
    return {i: make_item(id=i) for i in ("vxp-test-0001", "vxp-test-0002", "vxp-test-0003")}


# ------------------------------------------------------------------ CI method


def test_cluster_bootstrap_is_seeded_and_brackets_the_mean():
    vals = [1, 1, 0, 0, 1, 0, 1, 1]
    clusters = ["i1", "i1", "i2", "i2", "i3", "i3", "i4", "i4"]
    a = cluster_bootstrap(vals, clusters)
    b = cluster_bootstrap(vals, clusters)
    assert a == b  # fixed seed -> byte-identical tables on re-run
    assert a["mean"] == 0.625 and a["items"] == 4 and a["n"] == 8
    assert a["lo"] <= a["mean"] <= a["hi"]


def test_clustering_widens_the_interval_versus_iid():
    # identical values within an item: 4 effective observations, not 40
    vals = [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10
    clustered = cluster_bootstrap(vals, [f"i{k // 10}" for k in range(40)])
    iid = cluster_bootstrap(vals, [f"c{k}" for k in range(40)])
    assert (clustered["hi"] - clustered["lo"]) > (iid["hi"] - iid["lo"])


def test_single_item_has_no_interval():
    e = cluster_bootstrap([1.0, 0.0], ["i1", "i1"])
    assert e["mean"] == 0.5 and e["lo"] is None and e["hi"] is None


def test_paired_bootstrap_uses_only_shared_cells():
    a = {("i1", "x"): 1.0, ("i2", "x"): 1.0, ("i3", "x"): 0.0}
    b = {("i1", "x"): 0.0, ("i2", "x"): 1.0, ("i9", "x"): 1.0}
    d = paired_bootstrap(a, b)
    assert d["n"] == 2 and d["mean"] == 0.5


def test_hu_point_matches_report_probe_hu(items):
    recs = rows_for("vxp-test-0001", probe={"happy": "happy", "angry": "happy"})
    recs += rows_for("vxp-test-0002", probe={"happy": "happy", "angry": "angry"})
    probes = [r for r in recs if r["condition"] == "probe"]
    triples = []
    for r in probes:
        cmap = cue_class_map(items[r["item_id"]])
        triples.append((r["item_id"], cmap[r["scores"]["gold"]], cmap[r["scores"]["answer"]]))
    assert hu_bootstrap(triples)["mean"] == probe_hu(probes, items)["hu"]


# ------------------------------------------------------------------ bookkeeping


def test_speaker_varies_pairs_are_excluded_and_counted(tmp_path, items):
    rows = rows_for("vxp-test-0001") + rows_for("vxp-test-0002", speaker_varies=True)
    d = write_run(tmp_path, "20260915-final-arm-gemini", rows)
    arm = load_arm(d, items)
    pd = headline_row(arm, items, frozenset())["pair_discrimination"]
    assert pd["pairs"] == 1 and pd["excluded_speaker_varies"] == 1 and pd["both_correct"] == 1


def test_capability_limited_arm_is_na_not_zero(tmp_path, items):
    na = write_run(
        tmp_path, "20260915-final-gptaudio-gemini", rows_for("vxp-test-0001", twin_na=True)
    )
    zero = write_run(
        tmp_path,
        "20260915-final-other-gemini",
        rows_for("vxp-test-0001", twin={"happy": False, "angry": False}),
    )
    h_na = headline_row(load_arm(na, items), items, frozenset())
    h_zero = headline_row(load_arm(zero, items), items, frozenset())
    assert h_na["twin_credit"] == "n/a" and h_na["audio_minus_twin"] == "n/a"
    assert h_na["n"]["twin"] == "n/a"
    assert h_zero["twin_credit"]["mean"] == 0.0 and h_zero["audio_minus_twin"]["mean"] == 1.0

    casc = write_run(
        tmp_path, "20260915-final-cascadeopen-gemini", rows_for("vxp-test-0001", probe_na=True)
    )
    h_casc = headline_row(load_arm(casc, items), items, frozenset())
    assert h_casc["probe_accuracy"] == "n/a" and h_casc["hu"] == "n/a"
    assert h_casc["pair_discrimination"] == "n/a"
    diss = dissociation([load_arm(na, items), load_arm(casc, items)])
    assert all(d["status"] == "n/a" for d in diss)


def test_probe_not_applicable_arm_leaves_perception_tables(tmp_path, items):
    """A full-duplex arm's probe answers are unparseable by construction: its
    probe rows are n/a with a reason (never wrong answers), while its actions
    are scored and compared to the cascade's audio like any twin-less arm."""
    from voxparity.harness.final_analysis import PROBE_NOT_APPLICABLE

    assert "voicechat11b" in PROBE_NOT_APPLICABLE
    rows = rows_for("vxp-test-0001", twin_na=True, probe={"happy": None, "angry": None})
    rows[-1]["error"] = "HTTP 500: stream closed"  # an errored probe is still n/a
    d = write_run(tmp_path, "20260915-final-voicechat11b-gemini", rows)
    arm = load_arm(d, items)
    assert not arm.probe and arm.probe_na == 2 and not arm.errors
    h = headline_row(arm, items, frozenset())
    assert h["probe_accuracy"] == "n/a" and h["hu"] == "n/a"
    assert h["pair_discrimination"] == "n/a"
    assert "full-duplex" in h["probe_not_applicable_reason"]
    assert h["audio_credit"]["mean"] == 1.0 and h["audio_minus_twin"] == "n/a"
    assert all(x["status"] == "n/a" for x in dissociation([arm]))
    # the same rows under an ordinary label are scored (and fail): the capability
    # is per arm, never inferred from the answers
    other = load_arm(write_run(tmp_path, "20260915-final-other-gemini", rows), items)
    assert other.probe and not any(other.probe.values())


def test_two_drivers_in_one_run_dedupe_to_one_cell(tmp_path, items):
    """A run whose tail was served by a second route (same model) keeps one row
    per cell, the latest clean one, and still counts as one arm."""
    first = rows_for("vxp-test-0001", audio={"happy": True}, error=("angry",))
    tail = [
        {**r, "driver": "realtime:vercel:x"}
        for r in rows_for("vxp-test-0001", audio={"angry": False}, probe={})
        if r["condition"] == "audio"
    ]
    arm = load_arm(write_run(tmp_path, "20260915-final-grokvoice-gemini", first + tail), items)
    assert arm.label == "grokvoice" and not arm.errors
    assert arm.audio == {("vxp-test-0001", "happy"): 1.0, ("vxp-test-0001", "angry"): 0.0}


def test_skips_are_coverage_not_errors_and_leave_twin_cells(tmp_path, items):
    rows = rows_for("vxp-test-0001", audio={"happy": True}, skip=("angry",))
    rows += rows_for("vxp-test-0002", audio={"happy": True}, error=("angry",))
    arm = load_arm(write_run(tmp_path, "20260915-final-arm-gemini", rows), items)
    h = headline_row(arm, items, frozenset())
    assert h["n"]["skipped"] == {"audio": 1} and h["n"]["errors"] == {"audio": 1}
    assert ("vxp-test-0001", "angry") not in arm.twin  # outside the run population
    assert ("vxp-test-0002", "angry") in arm.twin  # unmeasured audio, twin still valid


def test_load_records_dedupe_prefers_clean_retry(tmp_path, items):
    rows = rows_for("vxp-test-0001", audio={"happy": True}, error=("angry",))
    rows += rows_for("vxp-test-0001", audio={"angry": False})
    arm = load_arm(write_run(tmp_path, "20260915-final-arm-gemini", rows), items)
    assert arm.errors == {} and arm.audio[("vxp-test-0001", "angry")] == 0.0


# ------------------------------------------------------------------ pairing


def test_same_cell_pairs_identical_cells_only(tmp_path, items):
    g_rows = rows_for("vxp-test-0001") + rows_for(
        "vxp-test-0002", audio={"happy": False, "angry": False}
    )
    k_rows = rows_for("vxp-test-0002", engine="kokoro", audio={"happy": True, "angry": True})
    k_rows += rows_for("vxp-test-0003", engine="kokoro")
    arms = [
        load_arm(write_run(tmp_path, "20260915-final-gemini37or-gemini", g_rows), items),
        load_arm(write_run(tmp_path, "20260915-final-gemini37or-kokoro", k_rows), items),
    ]
    rows = same_cell(arms, labels=("gemini37or",))
    assert len(rows) == 1
    r = rows[0]
    assert (r["engine_a"], r["engine_b"], r["cells"], r["items"]) == ("gemini", "kokoro", 2, 1)
    assert r["audio_b_minus_a"]["mean"] == 1.0


def test_null_floor_is_difference_in_differences_on_shared_cells(tmp_path, items):
    casc = rows_for("vxp-test-0001", probe_na=True, audio={"happy": True, "angry": False})
    casc += rows_for("vxp-test-0002", probe_na=True, audio={"happy": True, "angry": False})
    nat = rows_for("vxp-test-0001") + rows_for("vxp-test-0002") + rows_for("vxp-test-0003")
    arms = [
        load_arm(write_run(tmp_path, "20260915-final-cascadeopen-gemini", casc), items),
        load_arm(write_run(tmp_path, "20260915-final-native-gemini", nat), items),
    ]
    (row,) = null_floor(arms)
    assert row["cascade_delta_full"]["mean"] == 0.0
    assert row["diff_in_diff"]["n"] == 4 and row["diff_in_diff"]["mean"] == 0.5
    assert row["inside_cascade_band"] is False


def test_parse_run_name_handles_hyphenated_engines():
    assert parse_run_name("20260915-final-gemini37or-qwen3tts-cv", "qwen3tts-cv") == "gemini37or"
    assert parse_run_name("20260915-final-cascadeopen-gemini", "gemini") == "cascadeopen"


def test_render_template_fills_numbers_and_marks_pending(tmp_path):
    body = {
        "status": "PARTIAL",
        "headline": [
            {
                "label": "g",
                "engine": "gemini",
                "audio_minus_twin": {"mean": 0.15, "lo": 0.1, "hi": 0.2, "n": 9},
            },
            {"label": "gpt", "engine": "gemini", "audio_minus_twin": "n/a"},
        ],
    }
    (tmp_path / "headline.json").write_text(json.dumps(body))
    (tmp_path / "headline.md").write_text("# t\n\nTABLE\n")
    out = render_template(
        "{{status}} {{headline[g/gemini].audio_minus_twin|sci}} "
        "{{headline[gpt].audio_minus_twin|sci}} {{headline[zzz].audio_credit|ci}} "
        "{{headline[g].audio_minus_twin.mean|.3f}}\n{{table:headline}}",
        tmp_path,
    )
    assert out == "PARTIAL +0.15 [+0.10, +0.20] (n=9) n/a [pending] 0.150\nTABLE"


def test_cue_split_separates_neutral_delivery_and_counts_flips(tmp_path):
    from voxparity.harness.final_analysis import flip_share, split_by_cue
    from voxparity.schemas.item import DeliveryVariant, GoldAction

    item = make_item(
        id="vxp-test-0009",
        variants=[
            DeliveryVariant(
                variant_id="calm",
                emotion="neutral",
                intensity=0.3,
                gold=GoldAction(tool="a", args={"x": ["1"]}, rationale="r"),
            ),
            DeliveryVariant(
                variant_id="urgent",
                emotion="urgent",
                intensity=0.8,
                gold=GoldAction(tool="b", rationale="r"),
            ),
        ],
        perception_probe=PerceptionProbe(
            question="q?",
            options=["calm", "urgent"],
            gold_by_variant={"calm": "calm", "urgent": "urgent"},
        ),
    )
    items = {item.id: item}
    cue, neutral = split_by_cue({(item.id, "calm"): 1.0, (item.id, "urgent"): 0.0}, items)
    assert list(cue) == [(item.id, "urgent")] and list(neutral) == [(item.id, "calm")]
    rows = rows_for(item.id, audio={"calm": True, "urgent": True}, probe={})
    arm = load_arm(write_run(tmp_path, "20260915-final-arm-gemini", rows), items)
    flips = flip_share([arm], items)["gemini"]
    assert flips["items_gold_flips"] == 1 and flips["cue_bearing_share"] == 0.5
