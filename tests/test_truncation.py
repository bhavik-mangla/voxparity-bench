"""D083 truncation axis + disfluency-tolerant gate.

Covers: schema suffix enforcement, the truncate mix op (both tail flavors),
apply_scene recipe fields, slot_gap_wer's suffix-slot regression (the wirerb
gate-sweep crash), and disfluent_wer's exemption-without-loosening contract.
"""

from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pytest

from voxparity.schemas.item import SceneKind, SceneSpec
from voxparity.stimuli.assets import synthesize_asset
from voxparity.stimuli.scenes import Word, apply_scene
from voxparity.stimuli.validate import (
    disfluent_wer,
    has_disfluencies,
    slot_gap_wer,
    wer,
)


def _tone_wav(dur_s: float = 3.0, rate: int = 24000) -> bytes:
    import math
    import struct

    n = int(rate * dur_s)
    samples = [int(12000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(n)]
    buf = BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


def _wav_seconds(data: bytes) -> float:
    with wave.open(BytesIO(data), "rb") as f:
        return f.getnframes() / f.getframerate()


WORDS = [
    Word("pay", 0.1, 0.4),
    Word("the", 0.45, 0.6),
    Word("invoice", 0.65, 1.1),
    Word("reference", 1.2, 1.8),
    Word("whiskey", 1.9, 2.3),
    Word("bravo", 2.4, 2.8),
]


class TestTruncationScene:
    def test_line_drop_cuts_audio_and_records_recipe(self):
        scene = SceneSpec(kind=SceneKind.TRUNCATION, asset="synth:line_drop", slot="whiskey bravo")
        out, recipe = apply_scene(_tone_wav(), scene, words=WORDS)
        assert recipe["op"] == "truncate"
        assert recipe["slot"] == "whiskey bravo"
        assert recipe["slot_aligned_as"] == "whiskey bravo"
        assert "tail_sha256" in recipe
        # cut lands just before "whiskey" (1.9s - pad); the line-drop artifact
        # is ~0.5s, so the result must be well short of the 3s source
        assert recipe["cut_at_s"] == pytest.approx(1.86, abs=0.01)
        assert _wav_seconds(out) < 2.6

    def test_silence_tail_flavor_appends_dead_air_not_artifact(self):
        scene = SceneSpec(
            kind=SceneKind.TRUNCATION, asset="synth:silence_tail", slot="whiskey bravo"
        )
        out, recipe = apply_scene(_tone_wav(), scene, words=WORDS)
        assert recipe["tail"] == "silence"
        assert "tail_sha256" not in recipe
        # 1.86s of speech + 0.5s dead air
        assert _wav_seconds(out) == pytest.approx(2.36, abs=0.02)

    def test_truncation_requires_word_timings(self):
        scene = SceneSpec(kind=SceneKind.TRUNCATION, asset="synth:line_drop", slot="bravo")
        with pytest.raises(ValueError, match="word timings"):
            apply_scene(_tone_wav(), scene, words=None)

    def test_line_drop_asset_is_deterministic(self):
        assert synthesize_asset("synth:line_drop", seed=7) == synthesize_asset(
            "synth:line_drop", seed=7
        )


class TestTruncationSchema:
    def _item(self, slot: str):
        import yaml

        from voxparity.schemas.item import Item

        p = Path(__file__).resolve().parents[1] / "items/pilot/t4/vxp-wirerb-0001.yaml"
        base = yaml.safe_load(p.read_text())
        v = base["variants"][1]
        v["scene"] = {"kind": "truncation", "asset": "synth:line_drop", "slot": slot}
        return Item.model_validate(base)

    def test_suffix_slot_validates(self):
        item = self._item("whiskey four nine two bravo.")
        assert item.variants[1].scene.kind is SceneKind.TRUNCATION

    def test_non_suffix_slot_rejected(self):
        with pytest.raises(ValueError, match="suffix"):
            self._item("Send the invoice")


class TestSuffixSlotGate:
    def test_suffix_slot_does_not_crash_on_punctuation_only_tail(self):
        """Regression: slot ending at the final word left after == '.', which is
        truthy but has zero words — wer() raised and killed the gate sweep."""
        ref = "Send the payment, reference whiskey four nine two bravo."
        rate = slot_gap_wer(ref, "whiskey four nine two bravo", "send the payment reference")
        assert rate == 0.0

    def test_hallucinated_completion_in_gap_is_wildcard(self):
        ref = "Send the payment, reference whiskey four nine two bravo."
        rate = slot_gap_wer(
            ref, "whiskey four nine two bravo", "send the payment reference alpha seven"
        )
        assert rate == 0.0

    def test_errors_before_the_cut_still_count(self):
        ref = "Send the payment, reference whiskey four nine two bravo."
        rate = slot_gap_wer(ref, "whiskey four nine two bravo", "cancel the payment reference")
        assert rate > 0.0


class TestDisfluentWer:
    REF = "Um... sure. I— I guess so. Sure. Go ahead and post it."

    def test_detects_written_disfluency(self):
        assert has_disfluencies(self.REF)
        assert not has_disfluencies("Send the payment to First National.")

    def test_asr_filler_deletion_is_exempt(self):
        # Whisper-family ASR strips fillers by design (CrisperWhisper): the
        # cleaned hypothesis must round-trip at 0 while raw WER would fail.
        hyp = "sure i guess so sure go ahead and post it"
        assert disfluent_wer(self.REF, hyp) == 0.0
        assert wer(self.REF, hyp) > 0.15

    def test_content_word_error_still_vetoes(self):
        assert disfluent_wer(self.REF, "sure i guess so sure go ahead and cancel it") > 0.0

    def test_affirmative_mm_hmm_is_not_stripped(self):
        # _CANON maps mm-hmm -> yes BEFORE filler stripping; deleting the
        # affirmation must still count as an error.
        assert disfluent_wer("Mm-hmm, send it.", "send it") > 0.0


class TestSpeakerAxis:
    def _variant(self, speaker=None):
        from voxparity.schemas.item import DeliveryVariant, Emotion, GoldAction

        return DeliveryVariant(
            variant_id="v",
            emotion=Emotion.NEUTRAL,
            intensity=0.3,
            speaker=speaker,
            gold=GoldAction(tool="t", args={}, rationale="r"),
        )

    def test_child_frame_enters_prompt_and_voice(self):
        from voxparity.schemas.item import SpeakerProfile
        from voxparity.stimuli.style import tts_prompt
        from voxparity.stimuli.tts import GeminiBackend

        v = self._variant(SpeakerProfile.CHILD)
        assert "child's voice" in tts_prompt("Hello there.", v)
        assert GeminiBackend._PROFILE_VOICE["child"] == "Leda"

    def test_unspecified_speaker_prompt_is_unchanged(self):
        from voxparity.stimuli.style import tts_prompt

        v = self._variant(None)
        prompt = tts_prompt("Hello there.", v)
        assert "child" not in prompt and "elderly" not in prompt

    def test_no_gender_value_in_enum(self):
        # The Sep-6 research verdict: no legitimate voice-gender-conditional
        # protocol exists; the enum must never grow one casually.
        from voxparity.schemas.item import SpeakerProfile

        assert {p.value for p in SpeakerProfile} == {"adult", "child", "elderly"}

    def test_cue_class_pools_speaker_variants(self):
        from voxparity.scoring.stats import cue_class_map

        class P:
            options: ClassVar = ["an adult", "a young child"]
            gold_by_variant: ClassVar = {"a": "an adult", "c": "a young child"}

        class V:
            def __init__(self, vid, speaker):
                from voxparity.schemas.item import SpeakerProfile

                self.variant_id = vid
                self.scene = None
                self.speaker = SpeakerProfile(speaker) if speaker else None
                self.emotion = type("E", (), {"value": "neutral"})()

        class It:
            perception_probe = P()
            variants: ClassVar = [V("a", None), V("c", "child")]

        m = cue_class_map(It())
        assert m["a young child"] == "speaker:child"
        assert m["an adult"] == "neutral"


class TestPaidKeyFallback:
    def _pool(self, monkeypatch, free, paid):
        from voxparity.providers.keys import KeyPool

        monkeypatch.setenv("XX_API_KEYS", ",".join(free))
        monkeypatch.delenv("XX_API_KEY", raising=False)
        if paid:
            monkeypatch.setenv("XX_PAID_KEY", paid)
        else:
            monkeypatch.delenv("XX_PAID_KEY", raising=False)
        return KeyPool("XX")

    def test_paid_key_is_last_in_rotation(self, monkeypatch):
        pool = self._pool(monkeypatch, ["f1", "f2"], "paid")
        assert pool.keys == ["f1", "f2", "paid"]
        assert pool.paid_idx == 2
        assert not pool.on_paid
        pool.rotate()
        pool.rotate()
        assert pool.on_paid

    def test_release_paid_returns_to_free_keys(self, monkeypatch):
        pool = self._pool(monkeypatch, ["f1", "f2"], "paid")
        pool.rotate()
        pool.rotate()
        assert pool.on_paid
        pool.release_paid()
        assert pool.idx == 0 and not pool.on_paid
        # release when NOT on paid must not disturb sticky free rotation
        pool.rotate()
        pool.release_paid()
        assert pool.idx == 1

    def test_no_paid_key_changes_nothing(self, monkeypatch):
        pool = self._pool(monkeypatch, ["f1", "f2"], None)
        assert pool.paid_idx is None and pool.size == 2
        pool.release_paid()
        assert pool.idx == 0


class TestOverlayDtmf:
    def test_tones_past_clip_end_pad_and_stay_audible(self, tmp_path):
        import audioop

        from voxparity.stimuli.mix import dtmf, overlay_dtmf

        primary = tmp_path / "p.wav"
        primary.write_bytes(_tone_wav(1.0))
        tones = tmp_path / "t.wav"
        dtmf("55", tones)
        out = tmp_path / "o.wav"
        recipe = overlay_dtmf(primary, tones, out, start_s=2.0, snr_db=12.0)
        assert recipe["padded_to_s"] is not None and recipe["padded_to_s"] > 2.0
        with wave.open(str(out), "rb") as f:
            rate, pcm = f.getframerate(), f.readframes(f.getnframes())
        # the tone window (2.0s onward) must carry real energy, not the
        # near-zero the old span-RMS scaling produced over padded silence
        tone_span = pcm[int(2.0 * rate) * 2 :]
        assert audioop.rms(tone_span, 2) > 500

    def test_tones_over_speech_do_not_pad(self, tmp_path):
        from voxparity.stimuli.mix import dtmf, overlay_dtmf

        primary = tmp_path / "p.wav"
        primary.write_bytes(_tone_wav(3.0))
        tones = tmp_path / "t.wav"
        dtmf("55", tones)
        out = tmp_path / "o.wav"
        recipe = overlay_dtmf(primary, tones, out, start_s=0.5, snr_db=12.0)
        assert recipe["padded_to_s"] is None


class TestAuraBackend:
    def _backend(self):
        from voxparity.stimuli.tts import AuraBackend

        b = object.__new__(AuraBackend)
        b.voice = "thalia"
        b.model = "aura-2-thalia-en"

        class _Stub:
            def speak(self, text, voice="thalia"):
                return b"RIFFxxxxWAVE"

        b.client = _Stub()
        return b

    def _variant(self, emotion="neutral", speaker=None):
        from voxparity.schemas.item import DeliveryVariant, Emotion, GoldAction, SpeakerProfile

        return DeliveryVariant(
            variant_id="v",
            emotion=Emotion(emotion),
            intensity=0.3,
            speaker=SpeakerProfile(speaker) if speaker else None,
            gold=GoldAction(tool="t", args={}, rationale="r"),
        )

    def test_neutral_plain_transcript_renders(self):
        wav, voice, prompt = self._backend().synthesize("Send the payment.", self._variant())
        assert wav.startswith(b"RIFF") and voice == "thalia" and prompt == "neutral"

    def test_styled_variant_refused(self):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="style-less"):
            self._backend().synthesize("Send the payment.", self._variant("angry"))

    def test_speaker_variant_refused(self):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="speaker"):
            self._backend().synthesize("Send the payment.", self._variant(speaker="child"))

    def test_disfluent_transcript_refused(self):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="fillers"):
            self._backend().synthesize("Um... I— I think so.", self._variant())


class TestKokoroBackend:
    """D089: the local engine that ACCEPTS disfluent transcripts (probe-verified
    verbatim fillers) while refusing style — the mirror of Aura's contract."""

    def _backend(self):
        from voxparity.stimuli.tts import KokoroBackend

        b = object.__new__(KokoroBackend)
        b.voice = "af_heart"
        b.model = "prince-canuma/Kokoro-82M"

        class _Seg:
            audio: ClassVar = [0.0, 0.5, -0.5, 0.0]

        class _Model:
            def generate(self, **kw):
                return [_Seg()]

        b._model = _Model()
        return b

    def _variant(self, emotion="neutral", speaker=None):
        from voxparity.schemas.item import DeliveryVariant, Emotion, GoldAction, SpeakerProfile

        return DeliveryVariant(
            variant_id="v",
            emotion=Emotion(emotion),
            intensity=0.3,
            speaker=SpeakerProfile(speaker) if speaker else None,
            gold=GoldAction(tool="t", args={}, rationale="r"),
        )

    def test_disfluent_transcript_is_accepted(self):
        # The contract that separates kokoro from aura: aura refuses this
        # transcript because it deletes the fillers; kokoro speaks them.
        wav, voice, prompt = self._backend().synthesize(
            "Um... yes. I— I think so.", self._variant()
        )
        assert wav.startswith(b"RIFF") and voice == "af_heart" and prompt == "neutral"

    def test_styled_and_speaker_variants_refused(self):
        import pytest as _p

        with _p.raises(ValueError, match="style-less"):
            self._backend().synthesize("Send it.", self._variant("angry"))
        with _p.raises(ValueError, match="speaker"):
            self._backend().synthesize("Send it.", self._variant(speaker="child"))

    def test_native_rate_matches_project_stimulus_rate(self):
        import wave
        from io import BytesIO

        wav, _, _ = self._backend().synthesize("Send it.", self._variant())
        with wave.open(BytesIO(wav), "rb") as f:
            assert f.getframerate() == 24000 and f.getnchannels() == 1


class TestWerNormalizationD104:
    def test_hyphenated_numbers_equal_spaced(self):
        from voxparity.stimuli.validate import wer

        assert wer("flight thirty-nine heavy", "flight thirty nine heavy") == 0.0

    def test_canon_hyphen_forms_survive_splitting(self):
        from voxparity.stimuli.validate import wer

        assert wer("Mm-hmm, send it.", "yes send it") == 0.0

    def test_ah_is_a_filler(self):
        from voxparity.stimuli.validate import disfluent_wer

        assert (
            disfluent_wer("Ah... cactus one five two, uh, roger.", "cactus one five two roger")
            == 0.0
        )
