"""The game bundle is shared on a public link, so export_web publishes audio only
from licence-cleared engines (an allow-list, never a deny-list)."""

from __future__ import annotations

from pathlib import Path

import pytest

from test_schemas import make_item
from voxparity.harness.export_web import PUBLIC_ENGINES, export_items, publishable
from voxparity.stimuli.store import StimulusRecord, StimulusStore

PASS = {"asr_roundtrip": {"passed": True}, "cue_check": {"passed": True}}


def _rec(engine: str, gates: dict, vid: str = "happy") -> StimulusRecord:
    return StimulusRecord(
        item_id="i",
        variant_id=vid,
        sha256="",
        engine=engine,
        model="m",
        voice="v",
        prompt="p",
        gates=gates,
    )


def test_allow_list_is_exactly_the_cleared_engines():
    assert set(PUBLIC_ENGINES) == {"gemini", "kokoro", "human", "found"}


@pytest.mark.parametrize(
    "engine", ["cartesia", "elevenlabs", "fishaudio", "inworld", "camb", "hume", "aura", "new-x"]
)
def test_barred_and_unknown_engines_are_never_published(engine: str):
    assert not publishable(_rec(engine, PASS))


def test_qwen3tts_needs_a_passing_human_check():
    assert not publishable(_rec("qwen3tts-cv", PASS))
    assert not publishable(_rec("qwen3tts-vd", {**PASS, "human_check": {"passed": False}}))
    assert publishable(_rec("qwen3tts-vd", {**PASS, "human_check": {"passed": True}}))


def test_export_drops_barred_engine_pairs_and_keeps_cleared_ones(tmp_path: Path):
    item = make_item()
    item.review = "screened"
    store = StimulusStore(tmp_path / "stimuli")
    for engine in ("gemini", "aura", "cartesia", "kokoro"):
        for vid in ("happy", "angry"):
            rec = _rec(engine, PASS, vid)
            rec.item_id = item.id
            store.put(f"wav-{engine}-{vid}".encode(), rec)
    payload = export_items([item], store, tmp_path / "out")
    engines = {v["engine"] for v in payload["items"][0]["variants"]}
    assert engines == {"gemini", "kokoro"}
    assert payload["clips_by_engine"] == {"gemini": 2, "kokoro": 2}
    assert len(list((tmp_path / "out" / "clips").iterdir())) == 4


def test_leak_only_drafts_publish_but_held_and_unscreened_do_not(tmp_path: Path, monkeypatch):
    import json

    from voxparity.authoring import screen
    from voxparity.harness.export_web import leak_only_drafts

    # the screen's HELD list is private data; hold a synthetic id so the rule is
    # exercised the same way on every checkout
    monkeypatch.setitem(screen.HELD, "vxp-held-0001", "held for the test")

    rows = [
        # latest row wins: an earlier pass/unmeasured is superseded
        {"item_id": "vxp-leak-0001", "status": "unmeasured"},
        {"item_id": "vxp-leak-0001", "status": "leak", "rule": "schema-lexical-leak (D105)"},
        # a leak without the schema rule is a real leak, not the by-design one
        {"item_id": "vxp-real-0001", "status": "leak"},
        # held/defective items never qualify
        {"item_id": "vxp-held-0001", "status": "leak", "rule": "schema-lexical-leak (D105)"},
        # latest row unmeasured: not eligible
        {"item_id": "vxp-unm-0001", "status": "leak", "rule": "x"},
        {"item_id": "vxp-unm-0001", "status": "unmeasured"},
    ]
    log = tmp_path / "screen.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert leak_only_drafts(log, tmp_path) == {"vxp-leak-0001"}

    store = StimulusStore(tmp_path / "stimuli")
    items = []
    for iid in ("vxp-leak-0001", "vxp-real-0001", "vxp-none-0001"):
        item = make_item(id=iid)
        item.review = "draft"
        items.append(item)
        for vid in ("happy", "angry"):
            rec = _rec("gemini", PASS, vid)
            rec.item_id = iid
            store.put(f"wav-{iid}-{vid}".encode(), rec)
    payload = export_items(items, store, tmp_path / "out", leak_only=leak_only_drafts(log))
    assert [i["id"] for i in payload["items"]] == ["vxp-leak-0001"]
    assert payload["published_leak_only_drafts"] == 1
    assert payload["held_for_review"] == 2
    # without the evidence the draft stays held, exactly as before
    assert export_items(items, store, tmp_path / "out2")["items"] == []
