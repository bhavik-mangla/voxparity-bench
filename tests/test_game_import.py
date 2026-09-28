"""`voxparity human` pooling of the web game's trials JSONL (D064/D069)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from test_schemas import make_item
from voxparity.cli import _rater_batches, app
from voxparity.stimuli.store import StimulusRecord, StimulusStore


def _rows(*rows: dict) -> str:
    return "\n".join(json.dumps(r) for r in rows) + "\n"


P = "0f1e2d3c-aaaa-bbbb-cccc-000000000000"
S1 = "11111111-2222-3333-4444-555555555555"
S2 = "99999999-2222-3333-4444-555555555555"


def test_sessions_pool_per_player_and_session_and_trial_rows_fill_gaps():
    raw = _rows(
        {
            "kind": "trial",
            "session": S1,
            "player": P,
            "batch": "friends",
            "cid": "abc",
            "action": "x",
            "args": {},
            "probe_answer": "calm",
            "plays": 1,
        },
        {
            "kind": "session",
            "session": S1,
            "player": P,
            "batch": "friends",
            "answers": {"act:def": {"answer": "y", "args": {}}},
        },
        # a second session whose tab closed before the completion POST
        {
            "kind": "trial",
            "session": S2,
            "player": P,
            "batch": "friends",
            "cid": "ghi",
            "action": "z",
            "args": {},
        },
    )
    batches = dict(_rater_batches(raw, "rater"))
    assert set(batches) == {"0f1e2d3c-11111111", "0f1e2d3c-99999999"}
    assert set(batches["0f1e2d3c-11111111"]) == {"act:def", "act:abc", "listen:abc"}
    assert set(batches["0f1e2d3c-99999999"]) == {"act:ghi"}


def test_author_sessions_never_import_and_batch_filter_applies():
    raw = _rows(
        {
            "kind": "session",
            "session": S1,
            "player": P,
            "batch": "author",
            "answers": {"act:a": {"answer": "x"}},
        },
        {
            "kind": "session",
            "session": S2,
            "player": P,
            "batch": "friends",
            "answers": {"act:b": {"answer": "x"}},
        },
    )
    assert [n for n, _ in _rater_batches(raw, "r")] == ["0f1e2d3c-99999999"]
    assert _rater_batches(raw, "r", {"other"}) == []


# --- simple vs full parity mode (D109) -------------------------------------


def _bank(tmp_path: Path) -> tuple[Path, Path, str]:
    """One item, one gated clip, and the cid the game keys its answers by."""
    item = make_item()
    items_dir = tmp_path / "items"
    items_dir.mkdir(exist_ok=True)
    (items_dir / f"{item.id}.yaml").write_text(yaml.safe_dump(item.model_dump(mode="json")))
    store_dir = tmp_path / "stimuli"
    store = StimulusStore(store_dir)
    store.put(
        b"RIFF",
        StimulusRecord(
            item_id=item.id,
            variant_id="happy",
            sha256="",
            engine="gemini",
            model="m",
            voice="v",
            prompt="p",
            gates={"asr_roundtrip": {"passed": True}},
        ),
    )
    cid = hashlib.sha1(f"gemini{item.id}happy".encode()).hexdigest()[:10]
    return items_dir, store_dir, cid


_N = [0]


def _import(tmp_path: Path, row: dict) -> dict:
    items_dir, store_dir, _ = _bank(tmp_path)
    _N[0] += 1
    export = tmp_path / f"trials{_N[0]}.jsonl"
    export.write_text(json.dumps(row) + "\n")
    out = tmp_path / f"runs{_N[0]}"
    res = CliRunner().invoke(
        app,
        [
            "human",
            str(export),
            str(items_dir),
            "--store-dir",
            str(store_dir),
            "--out-dir",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    recs = [
        json.loads(ln) for ln in (next(out.glob("human-*/records.jsonl"))).read_text().splitlines()
    ]
    return next(r for r in recs if r["condition"] == "audio")


CID = hashlib.sha1(b"geminivxp-test-0001happy").hexdigest()[:10]


def _session(mode: str, args: dict, answer: str = "a") -> dict:
    return {
        "kind": "session",
        "session": S1,
        "player": P,
        "batch": "friends",
        "mode": mode,
        "answers": {f"act:{CID}": {"answer": answer, "args": args}},
    }


def test_simple_mode_imports_and_scores_on_selection_only(tmp_path):
    """A menu-only answer with the RIGHT tool is full credit, though the gold
    takes an argument the player was never asked for (D109)."""
    rec = _import(tmp_path, _session("simple", {}))
    assert rec["metrics"]["mode"] == "simple"
    assert rec["scores"]["scored_on"] == "selection"
    assert rec["scores"]["selection"] is True
    assert rec["scores"]["credit"] == 1.0 and rec["scores"]["passed"] is True
    # never recorded as a measured argument failure: it was not measured
    assert rec["scores"]["structure"] is None and rec["scores"]["parameters"] is None


def test_full_mode_still_scores_arguments(tmp_path):
    right = _import(tmp_path, _session("full", {"x": "1"}))
    assert right["metrics"]["mode"] == "full"
    assert right["scores"]["scored_on"] == "full"
    assert right["scores"]["credit"] == 1.0 and right["scores"]["parameters"] is True

    missing = _import(tmp_path, _session("full", {}))
    assert missing["scores"]["selection"] is True
    assert missing["scores"]["credit"] == 0.0 and missing["scores"]["structure"] is False
    # the selection-only column is still there, so a full session can be pooled
    # with a simple one on the one basis they share
    assert missing["scores"]["selection_credit"] == 1.0


def test_none_of_these_reaches_a_gold_no_call(tmp_path):
    """The escape is not decoration: without it a human can never give the answer
    a model gives by emitting no tool call at all (When2Call golds)."""
    from voxparity.cli import NO_CALL

    rec = _import(tmp_path, _session("simple", {}, answer=NO_CALL))
    assert rec["tool_calls"] == []
    assert rec["scores"]["selection"] is False  # this item's gold IS a call
