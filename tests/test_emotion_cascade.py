"""cascade-open-emo: same ASR/LLM as cascade-open, plus a fixed acoustic block."""

import math
from pathlib import Path
from typing import Any

from test_harness import _seed_store
from test_schemas import make_item
from voxparity.adapters.cascade import EmotionCascadeDriver
from voxparity.harness.final_analysis import Arm
from voxparity.harness.report import load_records
from voxparity.harness.runner import RunWriter, run_item
from voxparity.providers.acoustic import (
    BLOCK_TEMPLATE,
    compose_user_message,
    format_block,
    prosody,
    sensevoice_from_posteriors,
)

TAGS: dict[str, Any] = {
    "tagger": "fake-tagger",
    "sensevoice": {
        "emotion_forced": "sad",
        "emotion_unforced": "unknown",
        "emotion_probs": {"sad": 0.31, "unknown": 0.6, "neutral": 0.09},
        "events": ["crying"],
    },
    "emotion2vec": {"probs": {"sad": 0.7, "neutral": 0.2, "angry": 0.1}, "top": "sad"},
    "prosody": {
        "speech_rate_wps": 1.6,
        "f0_median_hz": 181.0,
        "f0_range_st": 2.4,
        "voiced_fraction": 0.62,
        "loudness_dbfs": -24.2,
    },
}


class FakeGroq:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, int]] = []
        self.transcribed = 0

    def transcribe(self, wav: bytes) -> str:
        self.transcribed += 1
        return "please cancel my plan"

    def respond_with_tools(self, system: str, user: str, decls: list) -> tuple[str, list]:
        self.messages.append((system, user, len(decls)))
        if not decls:
            return "happy", []
        return "", [{"name": "a", "args": {"x": "1"}}]


class FakeTagger:
    name = "fake-tagger"

    def __init__(self) -> None:
        self.calls = 0

    def tag(self, wav: bytes, transcript: str) -> dict[str, Any]:
        self.calls += 1
        return TAGS


def test_block_template_is_stable():
    block = format_block(TAGS)
    assert block == (
        "[acoustic analysis of the caller audio (automatic): emotion=sad (p=0.31; unknown "
        "p=0.60); emotion2vec: sad 0.70, neutral 0.20, angry 0.10; audio events: crying; "
        "speech rate: 1.6 words/s (slow); pitch: median 181 Hz, range 2.4 semitones (narrow); "
        "voiced frames: 62%; loudness: -24 dBFS]"
    )
    assert format_block(TAGS) == block  # pure
    # no steering vocabulary in the fixed template
    for word in ("escalat", "should", "must", "recommend", "distress", "urgent"):
        assert word not in BLOCK_TEMPLATE.lower()
    assert compose_user_message("hi", block) == f"hi\n\n{block}"


def test_block_renders_missing_values_as_na():
    block = format_block({"sensevoice": {}, "prosody": {}})
    assert "emotion=n/a" in block and "audio events: none detected" in block
    assert "pitch: n/a" in block and "speech rate: n/a" in block


def test_sensevoice_forced_and_unforced_labels():
    lp = math.log
    out = sensevoice_from_posteriors(
        {"unknown": lp(0.6), "sad": lp(0.3), "neutral": lp(0.1)},
        {"speech": lp(0.8), "crying": lp(0.15), "cough": lp(0.05)},
    )
    assert out["emotion_unforced"] == "unknown"
    assert out["emotion_forced"] == "sad"
    assert out["events"] == ["crying"]  # speech never listed; cough below threshold


def test_prosody_silence_is_undefined_not_zero():
    import numpy as np

    out = prosody(np.zeros(16000, dtype=np.float32), "one two")
    assert out["speech_rate_wps"] is None and out["f0_median_hz"] is None


def test_twin_gets_no_block_and_audio_gets_one(tmp_path: Path):
    groq, tagger = FakeGroq(), FakeTagger()
    drv = EmotionCascadeDriver(groq=groq, tagger=tagger)
    assert drv.capabilities.perception_probe
    item = make_item()
    store = _seed_store(tmp_path, item.id)
    writer = RunWriter(tmp_path / "runs", "emo")
    rows = run_item(drv, item, store, "test", writer, "emo")
    assert rows == 5
    twin_msgs = [u for _s, u, n in groq.messages if n and "[acoustic analysis" not in u]
    assert twin_msgs == [item.transcript]  # byte-identical to cascade-open's twin input
    audio_msgs = [u for _s, u, n in groq.messages if "[acoustic analysis" in u]
    assert len(audio_msgs) == 4  # 2 action + 2 probe cells
    assert all(
        u == compose_user_message("please cancel my plan", format_block(TAGS)) for u in audio_msgs
    )
    # probe and action cell share one ASR call and one tagging pass per clip
    assert groq.transcribed == 2 and tagger.calls == 2

    records = load_records(tmp_path / "runs" / "emo")
    assert all(not r["error"] for r in records)
    for r in records:
        if r["condition"] == "text_twin":
            assert "acoustic" not in r["metrics"]
        else:
            assert r["metrics"]["acoustic"] == TAGS
            assert r["metrics"]["acoustic_block"] == format_block(TAGS)
            assert r["metrics"]["asr_transcript"] == "please cancel my plan"
    probes = [r for r in records if r["condition"] == "probe"]
    assert all("answer" in r["scores"] for r in probes)  # applicable, not n/a


def test_emotion_cascade_is_not_a_null_arm():
    arm = Arm(
        run="r", label="cascadeemo", engine="gemini", driver="cascade-open-emo:x", path=Path()
    )
    assert not arm.is_cascade
    assert arm.role == "ladder" and not arm.is_audio_native
    null = Arm(run="r", label="cascadeopen", engine="gemini", driver="cascade-open:x", path=Path())
    assert null.is_cascade
    verbatim = Arm(
        run="r",
        label="cascverbatim",
        engine="gemini",
        driver="cascade-open-verbatim:x",
        path=Path(),
    )
    assert verbatim.role == "ladder" and not verbatim.is_cascade


def test_instrument_and_holm_families():
    from voxparity.harness.final_analysis import holm_family

    uv = Arm(run="r", label="ultravox8b", engine="gemini", driver="llamacpp:u", path=Path())
    assert uv.role == "instrument" and not uv.is_audio_native
    assert holm_family("ultravox8b") is None
    assert holm_family("cascverbatim") is None
    # one roster, one family: run date is provenance, not a statistical grouping
    assert holm_family("gemini37or") == "all"
    assert holm_family("phi4mm") == "all"
    assert holm_family("voicechat11b") == "all"


def test_holm_families_split_by_estimand():
    """D118: systems with a text path (difference-in-differences) and systems
    without one (audio level vs the cascade's) are Holm-corrected separately."""
    from voxparity.harness.final_analysis import holm_family

    assert holm_family("gemini37or", twin=True) == "did"
    assert holm_family("voicechat11b", twin=False) == "level"
    assert holm_family("cascverbatim", twin=True) is None
