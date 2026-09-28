"""Scene assets + applicator (D065): determinism, slot alignment, recipes."""

import audioop
import math
import struct
import wave
from io import BytesIO
from typing import ClassVar

import pytest

from voxparity.schemas.item import SceneKind, SceneSpec
from voxparity.stimuli.assets import synthesize_asset
from voxparity.stimuli.scenes import Word, apply_scene, locate_slot

RATE = 24000


def _speech(dur_s: float = 2.0) -> bytes:
    n = int(RATE * dur_s)
    samples = [int(math.sin(2 * math.pi * 300 * i / RATE) * 0.4 * 32767) for i in range(n)]
    buf = BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


def _rms_window(wav: bytes, start_s: float, dur_s: float) -> int:
    with wave.open(BytesIO(wav), "rb") as f:
        pcm = f.readframes(f.getnframes())
    a = int(start_s * RATE) * 2
    return audioop.rms(pcm[a : a + int(dur_s * RATE) * 2], 2)


class TestAssets:
    def test_deterministic_per_seed(self):
        assert synthesize_asset("synth:traffic", seed=1) == synthesize_asset(
            "synth:traffic", seed=1
        )
        assert synthesize_asset("synth:traffic", seed=1) != synthesize_asset(
            "synth:traffic", seed=2
        )

    def test_all_kinds_render_nonsilent(self):
        for asset in (
            "synth:white_noise",
            "synth:radio_squelch",
            "synth:co_alarm_chirp",
            "synth:traffic",
            "synth:medical_beep",
            "synth:ecall_modem",
        ):
            wav = synthesize_asset(asset, dur_s=2.0)
            with wave.open(BytesIO(wav), "rb") as f:
                pcm = f.readframes(f.getnframes())
            assert audioop.rms(pcm, 2) > 200, asset

    def test_unknown_asset_raises(self):
        with pytest.raises(ValueError, match="unknown synth asset"):
            synthesize_asset("synth:kazoo")


class TestLocateSlot:
    WORDS: ClassVar[list[Word]] = [
        Word("descending", 0.0, 0.6),
        Word("flight", 0.6, 0.9),
        Word("level", 0.9, 1.2),
        Word("three", 1.2, 1.5),
        Word("five", 1.5, 1.8),
        Word("zero", 1.8, 2.1),
    ]

    def test_finds_padded_window(self):
        start, dur, matched = locate_slot(self.WORDS, "three five zero")
        assert start == pytest.approx(1.16, abs=0.01)
        assert start + dur == pytest.approx(2.14, abs=0.01)
        assert matched == "three five zero"

    def test_fuzzy_matches_asr_renormalized_numbers(self):
        # Live-observed: Deepgram heard "three five zero" as "three fifty".
        words = [
            Word("descending", 0.0, 0.6),
            Word("flight", 0.6, 0.9),
            Word("level", 0.9, 1.2),
            Word("three", 1.2, 1.5),
            Word("fifty", 1.5, 2.0),
            Word("skyhawk", 2.2, 2.7),
        ]
        start, dur, matched = locate_slot(words, "three five zero")
        assert matched == "three fifty"
        assert start == pytest.approx(1.16, abs=0.01)
        assert start + dur == pytest.approx(2.04, abs=0.01)

    def test_matches_through_normalization(self):
        # ASR casing/punctuation must not break alignment (scorer-shared norm).
        words = [Word("Three,", 0.0, 0.5), Word("FIVE", 0.5, 1.0), Word("zero.", 1.0, 1.5)]
        start, _dur, _matched = locate_slot(words, "three five zero")
        assert start == 0.0  # clamped pad

    def test_missing_slot_raises(self):
        with pytest.raises(ValueError, match="not found"):
            locate_slot(self.WORDS, "completely unrelated words here")


class TestApplyScene:
    def test_background_synth(self):
        scene = SceneSpec(
            kind=SceneKind.BACKGROUND, asset="synth:co_alarm_chirp", snr_db=6.0, seed=5
        )
        out, recipe = apply_scene(_speech(), scene)
        assert recipe["kind"] == "background"
        assert recipe["asset"] == "synth:co_alarm_chirp"
        assert _rms_window(out, 0, 2.0) > _rms_window(_speech(), 0, 2.0)

    def test_background_tts_uses_callable_and_requires_text(self):
        scene = SceneSpec(
            kind=SceneKind.BACKGROUND, asset="tts:prompter", text="say yes", snr_db=10.0, seed=1
        )
        calls: list[str] = []

        def fake_tts(text: str) -> bytes:
            calls.append(text)
            return _speech(0.8)

        _out, recipe = apply_scene(_speech(), scene, background_tts=fake_tts)
        assert calls == ["say yes"]
        assert recipe["text"] == "say yes"
        with pytest.raises(ValueError, match="background_tts"):
            apply_scene(_speech(), scene)

    def test_slot_noise_masks_only_the_slot(self):
        scene = SceneSpec(
            kind=SceneKind.SLOT_NOISE,
            asset="synth:radio_squelch",
            slot="five",
            snr_db=-15.0,
            seed=2,
        )
        words = [Word("three", 0.2, 0.6), Word("five", 0.8, 1.2), Word("zero", 1.4, 1.8)]
        out, recipe = apply_scene(_speech(), scene, words=words)
        clean = _speech()
        assert _rms_window(out, 0.85, 0.3) > _rms_window(clean, 0.85, 0.3) * 1.5
        assert _rms_window(out, 0.2, 0.3) == pytest.approx(_rms_window(clean, 0.2, 0.3), rel=0.02)
        assert recipe["slot"] == "five"

    def test_slot_noise_without_words_raises(self):
        scene = SceneSpec(
            kind=SceneKind.SLOT_NOISE, asset="synth:radio_squelch", slot="five", seed=2
        )
        with pytest.raises(ValueError, match="word timings"):
            apply_scene(_speech(), scene)

    def test_dtmf_scene(self):
        scene = SceneSpec(kind=SceneKind.DTMF, asset="dtmf:55", snr_db=0.0, seed=3)
        out, recipe = apply_scene(_speech(), scene)
        assert recipe["asset"] == "dtmf:55"
        assert len(out) > 44


class TestSlotGapWer:
    def test_live_case_hallucinated_digits_pass(self):
        from voxparity.stimuli.validate import slot_gap_wer

        # The shape of a live Aug-30 case: Deepgram under a -12 dB squelch over the slot.
        rate = slot_gap_wer(
            "Climbing flight level three five zero, Cessna two two four.",
            "three five zero",
            "climbing flight level three fifty eight cessna two twenty four",
        )
        # "twenty" vs "two" is a real out-of-slot error and still counts;
        # the hallucinated in-slot digits do not.
        assert rate == pytest.approx(1 / 7)

    def test_out_of_slot_errors_still_fail(self):
        from voxparity.stimuli.validate import slot_gap_wer

        rate = slot_gap_wer(
            "refill of metoprolol fifty milligrams please",
            "fifty milligrams",
            "refill of atenolol something something now",
        )
        assert rate > 0.4

    def test_clean_hypothesis_scores_zero(self):
        from voxparity.stimuli.validate import slot_gap_wer

        assert slot_gap_wer("send help to dock nine now", "dock nine", "send help to now") == 0.0
        assert (
            slot_gap_wer("send help to dock nine now", "dock nine", "send help to dock nine now")
            == 0.0
        )
