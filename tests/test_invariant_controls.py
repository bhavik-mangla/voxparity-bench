"""FLAG-008: invariant controls are scored apart from every headline number.

The regression guard is the first test: adding control records to a run must
leave every headline readout byte-identical, and a run with no controls must
not grow a new key at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_harness import ScriptedDriver, _seed_store
from test_schemas import make_item
from voxparity.harness.compare import compare_table, pair_discrimination, summarize_run
from voxparity.harness.leaderboard import leaderboard_data, leaderboard_markdown
from voxparity.harness.report import invariance, load_records, probe_hu, summarize
from voxparity.harness.runner import RunWriter, run_item
from voxparity.schemas.item import DeliveryVariant, GoldAction, PerceptionProbe


def _control_item(item_id: str = "vxp-ctrl-0001"):
    def gold():
        return GoldAction(tool="a", args={"x": ["1"]}, rationale="r")

    return make_item(
        id=item_id,
        design="invariant_control",
        invariance_rationale="Rule X: handle it the same whatever the voice.",
        variants=[
            DeliveryVariant(variant_id="calm", emotion="neutral", intensity=0.4, gold=gold()),
            DeliveryVariant(variant_id="panicked", emotion="urgent", intensity=0.9, gold=gold()),
        ],
        perception_probe=PerceptionProbe(
            question="How do they sound?",
            options=["calm", "panicked"],
            gold_by_variant={"calm": "calm", "panicked": "panicked"},
        ),
    )


def _rec(item_id: str, vid: str, cond: str, scores: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {
        "run_id": "r",
        "item_id": item_id,
        "variant_id": vid,
        "condition": cond,
        "driver": "d",
        "engine": "e",
        "stimulus_sha256": "",
        "response_text": "",
        "tool_calls": kw.pop("tool_calls", []),
        "scores": scores,
        "error": kw.pop("error", ""),
        "metrics": {},
        **kw,
    }


def _headline_records() -> list[dict[str, Any]]:
    return [
        _rec("vxp-cf-0001", "happy", "audio", {"passed": True}),
        _rec("vxp-cf-0001", "angry", "audio", {"passed": False}),
        _rec("vxp-cf-0001", "happy", "probe", {"answer": "happy", "gold": "happy", "passed": True}),
        _rec(
            "vxp-cf-0001", "angry", "probe", {"answer": "happy", "gold": "angry", "passed": False}
        ),
        _rec(
            "vxp-cf-0001", "", "text_twin", {"happy": {"passed": False}, "angry": {"passed": True}}
        ),
    ]


def _control_records(stamped: bool = True) -> list[dict[str, Any]]:
    d = {"design": "invariant_control"} if stamped else {}
    wrong = [{"tool": "b", "args": {}}]
    return [
        # ctrl-0001: right when calm, moved by panic -> over-reaction
        _rec("vxp-ctrl-0001", "calm", "audio", {"passed": True}, **d),
        _rec("vxp-ctrl-0001", "panicked", "audio", {"passed": False}, tool_calls=wrong, **d),
        _rec(
            "vxp-ctrl-0001",
            "calm",
            "probe",
            {"answer": "calm", "gold": "calm", "passed": True},
            **d,
        ),
        _rec(
            "vxp-ctrl-0001",
            "panicked",
            "probe",
            {"answer": "panicked", "gold": "panicked", "passed": True},
            **d,
        ),
        _rec(
            "vxp-ctrl-0001",
            "",
            "text_twin",
            {"calm": {"passed": True}, "panicked": {"passed": True}},
            **d,
        ),
        # ctrl-0002: downgraded when calm -> under-reaction
        _rec("vxp-ctrl-0002", "calm", "audio", {"passed": False}, tool_calls=wrong, **d),
        _rec("vxp-ctrl-0002", "panicked", "audio", {"passed": True}, **d),
        # ctrl-0003: invariant
        _rec("vxp-ctrl-0003", "calm", "audio", {"passed": True}, **d),
        _rec("vxp-ctrl-0003", "panicked", "audio", {"passed": True}, **d),
        # ctrl-0004: one cell unmeasured -> incomplete, never a failure (D046)
        _rec("vxp-ctrl-0004", "calm", "audio", {"passed": True}, **d),
        _rec("vxp-ctrl-0004", "panicked", "audio", {}, error="no stimulus for engine e", **d),
    ]


def _items() -> dict[str, Any]:
    return {f"vxp-ctrl-000{i}": _control_item(f"vxp-ctrl-000{i}") for i in range(1, 5)}


def _write(run: Path, rows: list[dict[str, Any]]) -> Path:
    run.mkdir(parents=True)
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def test_headline_is_byte_identical_with_controls_added():
    base = summarize(_headline_records())
    assert "invariant_controls" not in base  # control-free runs grow no new key
    mixed = summarize(_headline_records() + _control_records())
    controls = mixed.pop("invariant_controls")
    assert json.dumps(mixed, sort_keys=True) == json.dumps(base, sort_keys=True)
    assert controls["items"] == "1/3"


def test_unstamped_control_rows_are_excluded_via_items():
    base = summarize(_headline_records())
    mixed = summarize(_headline_records() + _control_records(stamped=False), _items())
    mixed.pop("invariant_controls")
    assert mixed == base


def test_invariance_metric_and_reaction_direction():
    inv = invariance(_control_records(), _items())
    assert inv["invariance_rate"] == round(1 / 3, 4)  # only ctrl-0003 held, of 3 complete
    assert inv["incomplete_items"] == 1
    assert inv["cells"] == "5/7"
    assert inv["per_variant_accuracy"] == round(5 / 7, 4)
    assert inv["by_side"]["composed"] == {"n": 4, "passed": 3, "accuracy": 0.75}
    assert inv["by_side"]["marked"] == {"n": 3, "passed": 2, "accuracy": round(2 / 3, 4)}
    assert (inv["under_reaction"], inv["over_reaction"]) == (1, 1)
    assert inv["per_item"]["vxp-ctrl-0001"]["variants"]["panicked"]["tool"] == "b"
    assert inv["per_item"]["vxp-ctrl-0004"]["invariant"] is None
    assert inv["text_twin_pass_rate"] == 1.0


def test_invariance_without_items_leaves_direction_unclassified():
    inv = invariance(_control_records())
    assert set(inv["by_side"]) == {"unclassified"}
    assert (inv["under_reaction"], inv["over_reaction"]) == (0, 0)


def test_probe_hu_and_pair_discrimination_exclude_controls(tmp_path: Path):
    cf_item = make_item(id="vxp-cf-0001")
    items = {**_items(), cf_item.id: cf_item}
    assert probe_hu(_headline_records() + _control_records(), items) == probe_hu(
        _headline_records(), items
    )
    a = _write(tmp_path / "a", _headline_records())
    b = _write(tmp_path / "b", _headline_records() + _control_records())
    assert pair_discrimination(b) == pair_discrimination(a)
    ra, rb = summarize_run(a), summarize_run(b)
    assert {k: rb[k] for k in ("audio", "probe", "text_twin", "errors")} == {
        k: ra[k] for k in ("audio", "probe", "text_twin", "errors")
    }


def test_compare_and_leaderboard_table_controls_separately(tmp_path: Path):
    a = _write(tmp_path / "a", _headline_records())
    b = _write(tmp_path / "b", _headline_records() + _control_records())
    only_a = compare_table([a])
    assert "INVARIANT CONTROLS" not in only_a
    both = compare_table([a, b], _items())
    assert both.startswith(only_a.splitlines()[0])
    assert "INVARIANT CONTROLS (FLAG-008)" in both and "0.33 (1/3)" in both
    md = leaderboard_markdown(leaderboard_data([a, b], _items()))
    assert "Invariant controls (not part of any number above)" in md
    assert "not part" not in leaderboard_markdown(leaderboard_data([a]))
    rows = leaderboard_data([a, b])["rows"]
    assert rows[0]["audio"] == rows[1]["audio"]
    assert rows[0]["audio_minus_twin"] == rows[1]["audio_minus_twin"]


def test_runner_stamps_design_on_every_row(tmp_path: Path):
    item = _control_item()
    store = _seed_store(tmp_path, item.id)
    # _seed_store renders happy/angry; add the control's variant ids.
    for vid in ("calm", "panicked"):
        from voxparity.stimuli.store import StimulusRecord

        store.put(
            b"fakewav" + vid.encode(),
            StimulusRecord(
                item_id=item.id,
                variant_id=vid,
                sha256="",
                engine="test",
                model="m",
                voice="v",
                prompt="p",
                gates={"cue_check": {"passed": True}},
            ),
        )
    writer = RunWriter(tmp_path / "runs", "c")
    run_item(ScriptedDriver(), item, store, "test", writer, "c")
    records = load_records(tmp_path / "runs" / "c")
    assert records and all(r["design"] == "invariant_control" for r in records)
    summary = summarize(records)
    assert summary["counts"]["audio"] == "0/0"  # nothing leaks into the headline
    assert summary["invariant_controls"]["items"] == "1/1"  # scripted 'a' x=1 is the gold


def test_three_variant_control_splits_delivery_and_source():
    """The found-audio control's shape: real composed, TTS composed, TTS panicked."""

    def gold():
        return GoldAction(tool="a", args={"x": ["1"]}, rationale="r")

    item = make_item(
        id="vxp-ctrl3-0001",
        design="invariant_control",
        invariance_rationale="Rule X.",
        variants=[
            DeliveryVariant(
                variant_id="composed_real",
                emotion="neutral",
                intensity=0.4,
                gold=gold(),
                source="found:faa1549/480.10-487.00",
            ),
            DeliveryVariant(
                variant_id="composed_tts",
                emotion="neutral",
                intensity=0.4,
                gold=gold(),
                source="tts:gemini/composed-twin",
            ),
            DeliveryVariant(
                variant_id="panicked_render",
                emotion="urgent",
                intensity=0.9,
                gold=gold(),
                source="tts:gemini/panicked-twin",
            ),
        ],
        perception_probe=PerceptionProbe(
            question="q?",
            options=["calm", "panicked"],
            gold_by_variant={
                "composed_real": "calm",
                "composed_tts": "calm",
                "panicked_render": "panicked",
            },
        ),
    )
    items = {item.id: item}
    d = {"design": "invariant_control"}
    wrong = [{"tool": "b", "args": {}}]
    rows = [
        # real calm voice downgraded; TTS calm fine; TTS panic moved the action
        _rec(item.id, "composed_real", "audio", {"passed": False}, tool_calls=wrong, **d),
        _rec(item.id, "composed_tts", "audio", {"passed": True}, **d),
        _rec(item.id, "panicked_render", "audio", {"passed": False}, tool_calls=wrong, **d),
    ]
    inv = invariance(rows, items)
    assert inv["items"] == "0/1" and inv["cells"] == "1/3"
    assert (inv["under_reaction"], inv["over_reaction"]) == (1, 1)
    assert inv["by_side"]["composed"] == {"n": 2, "passed": 1, "accuracy": 0.5}
    assert inv["composed_by_source"] == {
        "real": {"n": 1, "passed": 0, "accuracy": 0.0},
        "synthetic": {"n": 1, "passed": 1, "accuracy": 1.0},
    }
    cells = inv["per_item"][item.id]["variants"]
    assert cells["composed_real"]["source"] == "real"
    assert cells["panicked_render"]["side"] == "marked" and "source" not in cells["panicked_render"]
    # a missing third cell leaves the item incomplete, never a failure (D046)
    partial = invariance(rows[:2], items)
    assert partial["items"] == "0/0" and partial["incomplete_items"] == 1
    # headline untouched
    mixed = summarize(_headline_records() + rows, items)
    assert mixed.pop("invariant_controls")["composed_by_source"]["real"]["n"] == 1
    assert mixed == summarize(_headline_records())
