"""Recording console (`voxparity record`): queue builder, resume, upload path."""

import json
from pathlib import Path

import yaml

from voxparity.record.server import (
    DISFLUENT_ITEM_KEYS,
    PROBE_LINES,
    build_queue,
    load_takes,
    save_take,
)
from voxparity.schemas.item import (
    DeliveryVariant,
    GoldAction,
    Item,
    PerceptionProbe,
    ToolDef,
    ToolParam,
)


def _write_item(dirp: Path, item_id: str, variants: list[tuple[str, str]]) -> None:
    """Minimal valid item: variants = [(variant_id, emotion), ...]."""
    tools = [
        ToolDef(name="a", description="", params=[ToolParam(name="x", type="string")]),
        ToolDef(name="b", description=""),
    ]
    golds = [
        GoldAction(tool="a", args={"x": ["1"]}, rationale="r"),
        GoldAction(tool="b", rationale="r"),
    ]
    item = Item.model_validate(
        {
            "id": item_id,
            "tier": "t4",
            "track": "turn",
            "policy_mode": "implicit",
            "domain": "testing",
            "transcript": f"Same words either way for {item_id}.",
            "scenario": "test scenario",
            "tools": tools,
            "variants": [
                DeliveryVariant(variant_id=vid, emotion=emo, intensity=0.6, gold=golds[i % 2])
                for i, (vid, emo) in enumerate(variants)
            ],
            "perception_probe": PerceptionProbe(
                question="q?",
                options=["one", "two"],
                gold_by_variant={vid: "one" for vid, _ in variants},
            ),
        }
    )
    (dirp / f"{item_id}.yaml").write_text(
        yaml.safe_dump(item.model_dump(mode="json"), sort_keys=False)
    )


def _bank(tmp_path: Path) -> Path:
    d = tmp_path / "items"
    d.mkdir()
    _write_item(d, "vxp-sarc-0001", [("sincere", "happy"), ("sarcastic", "sarcastic")])
    _write_item(d, "vxp-whisp-0001", [("routine", "neutral"), ("covert", "whispered")])
    _write_item(d, "vxp-slur-0001", [("alert", "neutral"), ("drowsy", "slurred")])
    _write_item(d, "vxp-hesit-0099", [("warm", "neutral"), ("halting", "anxious")])
    return d


class TestQueueBuilder:
    def test_priority_order_by_class(self, tmp_path):
        q = build_queue(_bank(tmp_path))
        classes = [e["cls"] for e in q]
        first = {c: classes.index(c) for c in dict.fromkeys(classes)}
        assert first["sarcastic_pair"] < first["whispered_pair"] < first["slurred"]
        assert first["slurred"] < first["stutter"] < first["voice_break"] < first["disfluent"]

    def test_sarcastic_pair_is_consecutive_twin_first(self, tmp_path):
        q = build_queue(_bank(tmp_path))
        pair = [e for e in q if e["cls"] == "sarcastic_pair"]
        assert [e["variant"] for e in pair] == ["sincere", "sarcastic"]
        # consecutive in the overall queue, identical transcript — the pair is the point
        idx = [q.index(e) for e in pair]
        assert idx[1] == idx[0] + 1
        assert pair[0]["transcript"] == pair[1]["transcript"]

    def test_pair_note_names_both_styles_on_both_entries(self, tmp_path):
        q = build_queue(_bank(tmp_path))
        pair = [e for e in q if e["cls"] == "whispered_pair"]
        assert len(pair) == 2
        for e in pair:
            assert e["pair_note"] is not None
            assert "back-to-back" in e["pair_note"]
            assert "first neutral, then whispered" in e["pair_note"]

    def test_probes_present_with_probe_kind(self, tmp_path):
        q = build_queue(_bank(tmp_path))
        probes = [e for e in q if e["kind"] == "probe"]
        assert len(probes) == len(PROBE_LINES) == 6
        assert {e["cls"] for e in probes} == {"stutter", "voice_break"}
        assert sum(e["cls"] == "stutter" for e in probes) == 3
        assert all(e["item"] is None for e in probes)
        assert any(e["transcript"].endswith("to Thursday.") for e in probes)

    def test_disfluent_items_queued_and_deduped(self, tmp_path):
        d = _bank(tmp_path)
        # confdis is BOTH a confused-class item and a disfluent-key item:
        # it must appear once, in the higher-priority confused class.
        _write_item(d, "vxp-confdis-0099", [("breezy", "happy"), ("lost", "confused")])
        q = build_queue(d)
        assert "confdis" in DISFLUENT_ITEM_KEYS
        confdis = [e for e in q if e["item"] == "vxp-confdis-0099"]
        assert len(confdis) == 2
        assert all(e["cls"] == "confused" for e in confdis)
        hesit = [e for e in q if e["item"] == "vxp-hesit-0099"]
        assert len(hesit) == 2
        assert all(e["cls"] == "disfluent" for e in hesit)

    def test_every_entry_id_unique_with_direction(self, tmp_path):
        q = build_queue(_bank(tmp_path))
        ids = [e["entry_id"] for e in q]
        assert len(ids) == len(set(ids))
        assert all(e["direction"] for e in q)


class TestResume:
    def test_load_takes_reads_highest_take(self, tmp_path):
        meta = tmp_path / "metadata.jsonl"
        rows = [
            {"entry": "vxp-sarc-0001__sarcastic", "take": 1},
            {"entry": "vxp-sarc-0001__sarcastic", "take": 2},
            {"entry": "probe__stutter_appointment", "take": 1},
        ]
        meta.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        takes = load_takes(meta)
        assert takes == {"vxp-sarc-0001__sarcastic": 2, "probe__stutter_appointment": 1}

    def test_load_takes_missing_file_is_empty(self, tmp_path):
        assert load_takes(tmp_path / "metadata.jsonl") == {}


class TestSaveTake:
    def test_writes_raw_wav_and_jsonl_via_injected_converter(self, tmp_path):
        calls = []

        def fake_converter(src: Path, dst: Path) -> None:
            calls.append((src, dst))
            dst.write_bytes(b"RIFF-fake-wav-" + src.read_bytes())

        row = save_take(
            tmp_path,
            "vxp-sarc-0001__sarcastic",
            take=3,
            webm=b"webm-bytes",
            duration_s=4.5,
            converter=fake_converter,
        )
        raw = tmp_path / "raw" / "vxp-sarc-0001__sarcastic__take3.webm"
        wav = tmp_path / "vxp-sarc-0001__sarcastic__take3.wav"
        assert raw.read_bytes() == b"webm-bytes"
        assert wav.exists() and calls == [(raw, wav)]
        assert row["entry"] == "vxp-sarc-0001__sarcastic"
        assert row["take"] == 3
        assert row["duration_s"] == 4.5
        assert row["wav_path"] == str(wav)
        import hashlib

        assert row["sha256"] == hashlib.sha256(wav.read_bytes()).hexdigest()
        lines = (tmp_path / "metadata.jsonl").read_text().splitlines()
        assert json.loads(lines[-1]) == row

    def test_probe_take_naming(self, tmp_path):
        def fake_converter(src: Path, dst: Path) -> None:
            dst.write_bytes(b"x")

        save_take(tmp_path, "probe__stutter_appointment", 1, b"w", 2.0, fake_converter)
        assert (tmp_path / "probe__stutter_appointment__take1.wav").exists()

    def test_appends_not_overwrites(self, tmp_path):
        def fake_converter(src: Path, dst: Path) -> None:
            dst.write_bytes(b"x")

        save_take(tmp_path, "e1", 1, b"w", 1.0, fake_converter)
        save_take(tmp_path, "e1", 2, b"w", 1.0, fake_converter)
        lines = (tmp_path / "metadata.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert load_takes(tmp_path / "metadata.jsonl") == {"e1": 2}
