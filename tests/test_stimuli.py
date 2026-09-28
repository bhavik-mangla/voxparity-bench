from pathlib import Path

import pytest

from _held import needs_items
from voxparity.schemas.item import DeliveryVariant, Emotion, GoldAction
from voxparity.stimuli.store import StimulusRecord, StimulusStore
from voxparity.stimuli.style import style_instruction, tts_prompt
from voxparity.stimuli.validate import wer


def make_variant(emotion: Emotion, intensity: float) -> DeliveryVariant:
    return DeliveryVariant(
        variant_id="v",
        emotion=emotion,
        intensity=intensity,
        gold=GoldAction(tool="t", rationale="r"),
    )


class TestStyle:
    def test_every_emotion_has_a_style(self):
        for emotion in Emotion:
            text = style_instruction(make_variant(emotion, 0.5))
            assert "expressed clearly" in text

    def test_intensity_bands(self):
        low = style_instruction(make_variant(Emotion.SAD, 0.1))
        high = style_instruction(make_variant(Emotion.SAD, 0.9))
        assert "subtly" in low and "unmistakably" in high

    def test_prompt_contains_verbatim_transcript(self):
        prompt = tts_prompt("No, it's okay. I'll manage.", make_variant(Emotion.SAD, 0.8))
        assert prompt.endswith("No, it's okay. I'll manage.")


class TestWer:
    def test_identical_is_zero(self):
        assert wer("Fine, whatever, just book it.", "fine whatever just book it") == 0.0

    def test_substitution(self):
        assert wer("book the slot", "book a slot") == pytest.approx(1 / 3)

    def test_empty_hypothesis(self):
        assert wer("one two", "") == 1.0

    def test_spelling_variants_canonicalized(self):
        assert wer("Alright, go ahead.", "all right go ahead") == 0.0
        assert wer("OK, I'm gonna wait.", "okay i'm going to wait") == 0.0

    def test_real_defect_still_caught(self):
        # "that plan" rendered as "mathman" must still fail a 0.15 gate
        assert wer("Alright, enroll me in that plan.", "all right enroll me in mathman") > 0.15


class TestStore:
    def test_roundtrip_and_hash_integrity(self, tmp_path: Path):
        store = StimulusStore(tmp_path / "stimuli")
        rec = store.put(
            b"RIFFfakewav",
            StimulusRecord(
                item_id="vxp-x-0001",
                variant_id="a",
                sha256="",
                engine="gemini",
                model="m",
                voice="Kore",
                prompt="p",
            ),
        )
        reloaded = StimulusStore(tmp_path / "stimuli")
        got = reloaded.get("vxp-x-0001", "a", "gemini")
        assert got is not None and got.sha256 == rec.sha256
        assert reloaded.audio_path(got).read_bytes() == b"RIFFfakewav"

    def test_corruption_detected(self, tmp_path: Path):
        store = StimulusStore(tmp_path / "s")
        rec = store.put(
            b"good",
            StimulusRecord(
                item_id="vxp-x-0001",
                variant_id="a",
                sha256="",
                engine="gemini",
                model="m",
                voice="Kore",
                prompt="p",
            ),
        )
        (store.audio_dir / f"{rec.sha256}.wav").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="hash mismatch"):
            store.audio_path(rec)

    def test_save_is_atomic_no_tmp_left(self, tmp_path: Path):
        store = StimulusStore(tmp_path / "s")
        store.put(
            b"x",
            StimulusRecord(
                item_id="vxp-x-0001",
                variant_id="a",
                sha256="",
                engine="e",
                model="m",
                voice="v",
                prompt="p",
            ),
        )
        assert store.manifest_path.exists()
        assert not store.manifest_path.with_suffix(".yaml.tmp").exists()

    def test_gate_results_persist(self, tmp_path: Path):
        store = StimulusStore(tmp_path / "s")
        store.put(
            b"x",
            StimulusRecord(
                item_id="vxp-x-0001",
                variant_id="a",
                sha256="",
                engine="gemini",
                model="m",
                voice="Kore",
                prompt="p",
            ),
        )
        store.set_gate("vxp-x-0001", "a", "gemini", "asr_roundtrip", {"passed": True, "wer": 0.0})
        reloaded = StimulusStore(tmp_path / "s")
        rec = reloaded.get("vxp-x-0001", "a", "gemini")
        assert rec is not None and rec.gates["asr_roundtrip"]["passed"] is True


@needs_items("vxp-bnkgr-0001")
def test_generate_skip_check_tolerates_take_metadata(tmp_path: Path, monkeypatch):
    """Regression: the 'take' int inside gates must not break the gate-pass check."""
    from typer.testing import CliRunner

    from voxparity.cli import app

    store = StimulusStore(tmp_path / "s")
    store.put(
        b"x",
        StimulusRecord(
            item_id="vxp-bnkgr-0001",
            variant_id="grateful",
            sha256="",
            engine="gemini",
            model="m",
            voice="v",
            prompt="p",
            gates={"asr_roundtrip": {"passed": True}, "cue_check": {"passed": True}, "take": 2},
        ),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "fake")
    items = Path(__file__).parent.parent / "items" / "pilot" / "t4" / "vxp-bnkgr-0001.yaml"
    # --limit 0 and a gate-passing existing clip: must skip without touching the network
    r = CliRunner().invoke(
        app,
        [
            "stimuli",
            "generate",
            str(items),
            "--engine",
            "gemini",
            "--takes",
            "3",
            "--store-dir",
            str(tmp_path / "s"),
            "--limit",
            "1",
        ],
    )
    assert "AttributeError" not in (r.output + str(r.exception))


class TestCartesiaContract:
    """D031: the Aug-2026 Cartesia arm was rendered through a malformed control
    channel (wrapping SSML tag, stale version header, 5-bucket emotion map).
    These lock the corrected contract."""

    def test_emotion_map_covers_every_emotion_with_enum_values(self):
        from voxparity.providers.cartesia import _EMOTION_MAP

        # The 58-value Cartesia `generation_config.emotion` enum.
        enum = {
            "neutral",
            "happy",
            "excited",
            "enthusiastic",
            "elated",
            "euphoric",
            "triumphant",
            "amazed",
            "surprised",
            "flirtatious",
            "curious",
            "content",
            "peaceful",
            "serene",
            "calm",
            "grateful",
            "affectionate",
            "trust",
            "sympathetic",
            "anticipation",
            "mysterious",
            "angry",
            "mad",
            "outraged",
            "frustrated",
            "agitated",
            "threatened",
            "disgusted",
            "contempt",
            "envious",
            "sarcastic",
            "ironic",
            "sad",
            "dejected",
            "melancholic",
            "disappointed",
            "hurt",
            "guilty",
            "bored",
            "tired",
            "rejected",
            "nostalgic",
            "wistful",
            "apologetic",
            "hesitant",
            "insecure",
            "confused",
            "resigned",
            "anxious",
            "panicked",
            "alarmed",
            "scared",
            "proud",
            "confident",
            "distant",
            "skeptical",
            "contemplative",
            "determined",
        }
        for emotion in Emotion:
            assert emotion in _EMOTION_MAP
            assert _EMOTION_MAP[emotion] in enum

    def test_distinct_emotions_are_not_collapsed(self):
        """The old map sent frustrated/angry/urgent all as 'angry' and
        sad/anxious/resigned all as 'sad', so paired variants could render alike."""
        from voxparity.providers.cartesia import _EMOTION_MAP

        assert _EMOTION_MAP[Emotion.FRUSTRATED] != _EMOTION_MAP[Emotion.ANGRY]
        assert _EMOTION_MAP[Emotion.RESIGNED] != _EMOTION_MAP[Emotion.SAD]
        assert _EMOTION_MAP[Emotion.ANXIOUS] != _EMOTION_MAP[Emotion.SAD]
        assert len(set(_EMOTION_MAP.values())) == len(Emotion)

    def test_version_header_is_the_supported_one(self):
        from voxparity.providers.cartesia import VERSION

        assert VERSION == "2026-08-14"

    def test_transcript_is_never_wrapped_in_markup(self, monkeypatch):
        """Regression: we shipped `<emotion value="x">text</emotion>`, which is not
        valid Cartesia syntax (the real tag is self-closing) and left a stray
        closing tag in the transcript."""
        import voxparity.providers.cartesia as mod

        sent: dict = {}

        class FakeResp:
            status_code = 200
            content = b"RIFF"

        def fake_post(url, *, headers, json_body, timeout, tries):
            sent.update(json_body)
            sent["_headers"] = headers
            return FakeResp()

        monkeypatch.setattr(mod, "post_with_retry", fake_post)
        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        monkeypatch.setenv("VOXPARITY_ALLOW_CARTESIA", "1")
        client = mod.CartesiaClient()
        client.synthesize("Just cancel it.", make_variant(Emotion.RESIGNED, 0.7), "voice-1")

        assert sent["transcript"] == "Just cancel it."
        assert "<emotion" not in sent["transcript"]
        assert sent["generation_config"] == {"emotion": "resigned"}
        assert sent["voice"] == {"id": "voice-1"}
        assert "mode" not in sent["voice"]
        assert sent["_headers"]["Cartesia-Version"] == "2026-08-14"

    def test_lossy_mapping_is_recorded_in_provenance(self, monkeypatch):
        import voxparity.providers.cartesia as mod

        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        monkeypatch.setenv("VOXPARITY_ALLOW_CARTESIA", "1")
        client = mod.CartesiaClient()
        assert "lossy" in client.control_label(make_variant(Emotion.URGENT, 0.7))
        assert "lossy" not in client.control_label(make_variant(Emotion.RESIGNED, 0.7))


class TestElevenLabsPolicyGuard:
    def test_disabled_without_explicit_opt_in(self, monkeypatch):
        from voxparity.providers.elevenlabs import ElevenLabsClient, ElevenLabsError

        monkeypatch.delenv("VOXPARITY_ALLOW_ELEVENLABS", raising=False)
        with pytest.raises(ElevenLabsError, match="Prohibited Use Policy"):
            ElevenLabsClient()

    def test_no_silent_fallback_model(self):
        """The fallback stripped the emotion tag but kept the eleven_v3 label."""
        import voxparity.providers.elevenlabs as mod

        assert not hasattr(mod, "FALLBACK_MODEL")


class TestCartesiaRetired:
    def test_disabled_without_explicit_opt_in(self, monkeypatch):
        """D034: retired as a stimulus engine — 1/99 pair discrimination, and the
        vendor documents that emotion control needs the text to agree with the
        emotion, which is the opposite of a counterfactual item."""
        from voxparity.providers.cartesia import CartesiaClient, CartesiaError

        monkeypatch.delenv("VOXPARITY_ALLOW_CARTESIA", raising=False)
        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        with pytest.raises(CartesiaError, match="retired"):
            CartesiaClient()

    def test_opt_in_still_works_for_reproduction(self, monkeypatch):
        from voxparity.providers.cartesia import CartesiaClient

        monkeypatch.setenv("VOXPARITY_ALLOW_CARTESIA", "1")
        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        assert CartesiaClient() is not None


class TestStreamingWavHeader:
    """D035: Fish streams its response and writes a placeholder length, so every
    clip arrived declaring 2147483520 frames (~13.5 h). Players read to EOF so it
    is silent, but it poisoned every duration-derived metric."""

    def _wav(self, declared_data_size: int, samples: bytes) -> bytes:
        import struct

        return (
            b"RIFF"
            + struct.pack("<I", 36 + len(samples))
            + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data"
            + struct.pack("<I", declared_data_size)
            + samples
        )

    def test_placeholder_length_is_corrected(self):
        from voxparity.providers.fishaudio import _fix_streaming_wav_header

        samples = b"\x00\x01" * 1000
        fixed = _fix_streaming_wav_header(self._wav(0x7FFFFF80, samples))
        import struct
        import wave
        from io import BytesIO

        with wave.open(BytesIO(fixed)) as w:
            assert w.getnframes() == 1000
        assert struct.unpack("<I", fixed[4:8])[0] == len(fixed) - 8

    def test_correct_header_is_left_alone(self):
        from voxparity.providers.fishaudio import _fix_streaming_wav_header

        samples = b"\x00\x01" * 500
        original = self._wav(len(samples), samples)
        assert _fix_streaming_wav_header(original) == original

    def test_non_riff_input_is_untouched(self):
        from voxparity.providers.fishaudio import _fix_streaming_wav_header

        assert _fix_streaming_wav_header(b"not a wav at all") == b"not a wav at all"


def test_wav_seconds_ignores_a_lying_header(tmp_path):
    """The guard that stops a bad header reaching the metrics."""
    import struct

    from voxparity.harness.runner import wav_seconds

    samples = b"\x00\x01" * 24000  # exactly 1 second at 24 kHz mono 16-bit
    blob = (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
        + b"data"
        + struct.pack("<I", 0x7FFFFF80)
        + samples
    )
    path = tmp_path / "lying.wav"
    path.write_bytes(blob)
    seconds = wav_seconds(str(path))
    assert seconds is not None and 0.9 < seconds < 1.1


class TestOutageIsNotAVerdict:
    """D046: the regression that cost us an engine. A judge that could not answer
    has not rejected the clip — but the gate recorded the outage as
    `passed: false`, so 173 of Cartesia's 198 clips looked like engine failures
    when they were our own quota exhaustion. That phantom produced the '1/99 pair
    discrimination' figure that retired the engine."""

    def test_judge_error_records_no_verdict(self):
        from test_schemas import make_item
        from voxparity.providers.gemini import GeminiError
        from voxparity.stimuli.validate import cue_check_gate

        class OutageClient:
            def classify(self, *a, **k):
                raise GeminiError("rate-limited on all 4 key(s)")

        item = make_item()
        variant_id = item.variants[0].variant_id
        result = cue_check_gate(OutageClient(), item, variant_id, b"RIFF")
        assert result.passed is False
        # the load-bearing part: downstream must be able to tell this apart from
        # a judgement, or it counts against the engine
        assert result.detail["verdict"] is None
        assert "rate-limited" in str(result.detail["error"])
        assert "judge_answer" not in result.detail

    def test_a_real_disagreement_carries_a_verdict(self):
        from test_schemas import make_item
        from voxparity.stimuli.validate import cue_check_gate

        item = make_item()
        variant_id = item.variants[0].variant_id
        wrong = item.perception_probe.options[-1]
        gold = item.perception_probe.gold_by_variant[variant_id]

        class DisagreeingClient:
            def classify(self, *a, **k):
                return wrong

        result = cue_check_gate(DisagreeingClient(), item, variant_id, b"RIFF")
        assert result.detail["verdict"] == wrong
        assert result.detail.get("error") is None
        assert result.passed is (wrong == gold)
