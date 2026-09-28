"""Human action baseline: selection basis, identical-cell contrasts, listener
counting by player, tie handling, and raw game counts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from test_final_analysis import rows_for, write_run
from test_schemas import make_item
from voxparity.harness.final_analysis import load_arm
from voxparity.harness.human_baseline import (
    human_baseline,
    majority_vote,
    rater_of,
    selection_credit,
    trial_counts,
)


def human_row(
    item_id: str,
    variant: str,
    tool: str | None,
    credit: float,
    *,
    player: str = "p1",
    session: str = "s1",
    engine: str = "gemini",
    condition: str = "audio",
    probe_ok: bool = True,
) -> dict[str, Any]:
    base = {
        "item_id": item_id,
        "variant_id": variant,
        "driver": f"human:{player}-{session}",
        "engine": engine,
        "error": "",
        "design": "counterfactual",
        "condition": condition,
    }
    if condition == "probe":
        gold = variant
        return {
            **base,
            "tool_calls": [],
            "scores": {"answer": gold if probe_ok else "x", "gold": gold, "passed": probe_ok},
        }
    return {
        **base,
        "tool_calls": [{"tool": tool, "args": {}}] if tool else [],
        "scores": {"credit": credit, "passed": credit == 1.0, "scored_on": "selection"},
    }


@pytest.fixture
def items() -> dict[str, Any]:
    return {i: make_item(id=i) for i in ("vxp-test-0001", "vxp-test-0002")}


def test_selection_credit_reconstructs_old_rows():
    assert selection_credit({"selection_credit": 0.5, "credit": 0.0}) == 0.5
    # a selection hit with a wrong argument: full credit 0, selection 1
    assert selection_credit({"selection": True, "credit": 0.0}) == 1.0
    # acceptable alternative: selection miss, partial credit carried
    assert selection_credit({"selection": False, "credit": 0.7}) == 0.7


def test_rater_of_splits_player_and_session():
    assert rater_of("human:0f1e2d3c-9a8b7c6d") == ("0f1e2d3c", "9a8b7c6d")


def test_majority_vote_scores_ties_at_mean_credit_and_counts_them():
    k1, k2 = ("i1", "v@gemini"), ("i2", "v@gemini")
    maj, ties = majority_vote(
        {k1: [("a", 1.0), ("a", 1.0), ("b", 0.0)], k2: [("a", 1.0), ("b", 0.0)]}
    )
    assert maj == {k1: 1.0, k2: 0.5} and ties == 1


def test_same_cell_uses_model_selection_credit_on_shared_cells_only(tmp_path, items):
    # model: happy right, angry wrong, on both items (full credit 0 on a selection hit
    # must NOT count against it: the human basis is selection)
    rows = rows_for("vxp-test-0001", audio={"happy": True, "angry": False})
    rows += rows_for("vxp-test-0002", audio={"happy": True, "angry": True})
    for r in rows:
        if r["condition"] == "audio" and r["variant_id"] == "happy" and r["item_id"].endswith("2"):
            r["scores"] = {"passed": False, "credit": 0.0, "selection": True}
    arm = load_arm(write_run(tmp_path, "20260915-final-m-gemini", rows), items)
    human = [
        human_row("vxp-test-0001", "happy", "a", 1.0),
        human_row("vxp-test-0001", "angry", "b", 1.0),
        human_row("vxp-test-0002", "happy", "a", 1.0, engine="kokoro"),  # no model cell
    ]
    out = human_baseline(human, [arm], items)
    (row,) = out["same_cell"]
    assert row["cells"] == 2 and row["engines"] == ["gemini"]
    assert row["human"]["mean"] == 1.0 and row["model_selection"]["mean"] == 0.5
    assert row["model_minus_human"]["mean"] == -0.5
    assert row["model_minus_human_cue_bearing"]["n"] == 2  # happy/angry are both cue-bearing


def test_listeners_are_distinct_players_and_controls_are_dropped(items):
    items["vxp-ctrl-0001"] = make_item(id="vxp-ctrl-0001")
    rows = [
        human_row("vxp-test-0001", "happy", "a", 1.0, player="p1", session="s1", engine="human"),
        human_row("vxp-test-0001", "happy", "a", 1.0, player="p1", session="s2", engine="human"),
        human_row("vxp-test-0001", "happy", "b", 0.0, player="p2", session="s3", engine="human"),
        human_row(
            "vxp-test-0001",
            "happy",
            None,
            0.0,
            player="p3",
            session="s4",
            engine="human",
            condition="probe",
        ),
        {**human_row("vxp-ctrl-0001", "happy", "a", 1.0), "design": "invariant_control"},
    ]
    recs = {("vxp-test-0001", "happy"), ("vxp-test-0001", "angry")}
    out = human_baseline(rows, [], items, recordings=recs)
    assert out["recordings"]["per_recording"] == {
        "vxp-test-0001/angry": 0,
        "vxp-test-0001/happy": 3,
    }
    assert out["recordings"]["with_3plus_listeners"] == 1
    assert out["counts"]["dropped_rows"] == {"invariant_control": 1}
    assert out["counts"]["players_imported"] == 2 and out["counts"]["audio_answers_scored"] == 3
    assert out["individual"]["mean"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["counts"]["raters_per_cell"] == {3: 1}


def test_trial_counts_skip_author_and_count_completion_only_answers(tmp_path: Path):
    lines = [
        {
            "kind": "trial",
            "session": "s1",
            "player": "p1",
            "cid": "c1",
            "action": "a",
            "batch": None,
        },
        {
            "kind": "session",
            "session": "s1",
            "player": "p1",
            "batch": None,
            "answers": {"act:c1": {"answer": "a"}, "act:c2": {"answer": "b"}, "listen:c1": {}},
        },
        {
            "kind": "trial",
            "session": "s2",
            "player": "me",
            "cid": "c1",
            "action": "a",
            "batch": "author",
        },
        {"session": "old", "item_id": "x", "action": "a"},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    c = trial_counts(p)
    assert c["players"] == 1 and c["sessions"] == 1 and c["completed_sessions"] == 1
    assert c["action_answers"] == 2 and c["answers_only_in_completion_row"] == 1
    assert c["author_sessions_excluded"] == 1 and c["legacy_rows_unmappable"] == 1
