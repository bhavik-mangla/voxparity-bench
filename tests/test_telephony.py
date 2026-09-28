"""Telephone-channel DSP: codec degradation, seeded loss, composed pipeline.

The channel ops must be deterministic and recorded (D065/D086 recipe
discipline): every assertion here is about either the audio physics (energy
above the voiceband dies, the output stays non-silent) or the reproducibility
contract (same seed -> byte-identical file; the recipe names what happened).
ffmpeg-dependent tests skip gracefully on a machine without it.
"""

from __future__ import annotations

import audioop
import hashlib
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from voxparity.stimuli.telephony import codec_roundtrip, frame_drop, phone_channel

RATE = 24000

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

ALL_CODECS = ["g711u", "g711a", "g722", "g726", "g723_1"]
NARROWBAND = ["g711u", "g711a", "g726", "g723_1"]  # 8 kHz native: 6 kHz cannot survive


def _two_tone(path: Path, dur_s: float = 1.5) -> None:
    """440 Hz (inside the voiceband) + 6 kHz (well above it), 24 kHz mono."""
    n = int(RATE * dur_s)
    samples = [
        int(
            (
                0.35 * math.sin(2 * math.pi * 440 * i / RATE)
                + 0.35 * math.sin(2 * math.pi * 6000 * i / RATE)
            )
            * 32767
        )
        for i in range(n)
    ]
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(struct.pack(f"<{n}h", *samples))


def _read(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as f:
        return f.readframes(f.getnframes()), f.getframerate()


def _goertzel_power(pcm: bytes, sample_rate: int, freq: float) -> float:
    """Single-bin spectral power (Goertzel), normalized by window length —
    stdlib-only, phase-insensitive, exactly what a crude band probe needs."""
    n = len(pcm) // 2
    xs = struct.unpack(f"<{n}h", pcm)
    coeff = 2.0 * math.cos(2.0 * math.pi * freq / sample_rate)
    s1 = s2 = 0.0
    for x in xs:
        s0 = x + coeff * s1 - s2
        s2, s1 = s1, s0
    return (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (n * n)


@pytest.fixture()
def tone(tmp_path: Path) -> Path:
    path = tmp_path / "tone.wav"
    _two_tone(path)
    return path


@needs_ffmpeg
class TestCodecRoundtrip:
    @pytest.mark.parametrize("codec", ALL_CODECS)
    def test_output_valid_nonsilent_and_degraded(self, tone, tmp_path, codec):
        out = tmp_path / f"{codec}.wav"
        recipe = codec_roundtrip(tone, out, codec=codec)
        pcm, sample_rate = _read(out)
        assert sample_rate == RATE  # resampled back to the source rate
        assert audioop.rms(pcm, 2) > 500  # non-silent
        assert out.read_bytes() != tone.read_bytes()  # measurably not a copy
        # the in-band component survives the channel (within ~6 dB)
        in_pcm, _ = _read(tone)
        assert _goertzel_power(pcm, RATE, 440) > _goertzel_power(in_pcm, RATE, 440) * 0.25
        assert recipe["codec"] == codec
        assert recipe["source_rate"] == RATE

    @pytest.mark.parametrize("codec", NARROWBAND)
    def test_narrowband_codecs_kill_out_of_band_energy(self, tone, tmp_path, codec):
        out = tmp_path / f"{codec}.wav"
        codec_roundtrip(tone, out, codec=codec)
        in_pcm, _ = _read(tone)
        out_pcm, _ = _read(out)
        # 6 kHz is beyond an 8 kHz codec's Nyquist: >10x power drop expected.
        assert _goertzel_power(out_pcm, RATE, 6000) < _goertzel_power(in_pcm, RATE, 6000) / 10

    def test_deterministic_and_recipe_complete(self, tone, tmp_path):
        out1, out2 = tmp_path / "a.wav", tmp_path / "b.wav"
        r1 = codec_roundtrip(tone, out1, codec="g726", bitrate=24000)
        r2 = codec_roundtrip(tone, out2, codec="g726", bitrate=24000)
        assert out1.read_bytes() == out2.read_bytes()  # bit-exact re-run
        assert r1 == r2
        assert r1["op"] == "codec_roundtrip"
        assert r1["bitrate"] == 24000
        assert r1["codec_rate"] == 8000
        assert r1["input_sha256"] == hashlib.sha256(tone.read_bytes()).hexdigest()
        assert r1["ffmpeg_version"].startswith("ffmpeg version")

    def test_g726_bitrates_change_the_audio(self, tone, tmp_path):
        lo, hi = tmp_path / "16k.wav", tmp_path / "40k.wav"
        codec_roundtrip(tone, lo, codec="g726", bitrate=16000)
        codec_roundtrip(tone, hi, codec="g726", bitrate=40000)
        assert lo.read_bytes() != hi.read_bytes()

    def test_fixed_rate_codec_rejects_bitrate(self, tone, tmp_path):
        with pytest.raises(ValueError, match="fixed-rate"):
            codec_roundtrip(tone, tmp_path / "x.wav", codec="g711u", bitrate=16000)

    def test_invalid_g726_bitrate_rejected(self, tone, tmp_path):
        with pytest.raises(ValueError, match="bitrate"):
            codec_roundtrip(tone, tmp_path / "x.wav", codec="g726", bitrate=12345)


def test_unknown_codec_raises_naming_it(tmp_path):
    # codec validation precedes any ffmpeg call, so this runs everywhere
    src = tmp_path / "t.wav"
    _two_tone(src, dur_s=0.1)
    with pytest.raises(ValueError, match="opus"):
        codec_roundtrip(src, tmp_path / "x.wav", codec="opus")


class TestFrameDrop:
    def test_same_seed_is_byte_identical(self, tone, tmp_path):
        out1, out2 = tmp_path / "a.wav", tmp_path / "b.wav"
        r1 = frame_drop(tone, out1, seed=7, rate=0.5)
        r2 = frame_drop(tone, out2, seed=7, rate=0.5)
        assert out1.read_bytes() == out2.read_bytes()
        assert r1["dropped_frames"] == r2["dropped_frames"]
        assert 0 < len(r1["dropped_frames"]) < r1["n_frames"]

    def test_different_seed_drops_different_frames(self, tone, tmp_path):
        r1 = frame_drop(tone, tmp_path / "a.wav", seed=7, rate=0.5)
        r2 = frame_drop(tone, tmp_path / "b.wav", seed=8, rate=0.5)
        assert r1["dropped_frames"] != r2["dropped_frames"]

    def test_rate_zero_is_a_noop(self, tone, tmp_path):
        out = tmp_path / "same.wav"
        recipe = frame_drop(tone, out, seed=3, rate=0.0)
        assert out.read_bytes() == tone.read_bytes()
        assert recipe["dropped_frames"] == []

    def test_rate_one_silence_fill_is_silent(self, tone, tmp_path):
        out = tmp_path / "silent.wav"
        recipe = frame_drop(tone, out, seed=3, rate=1.0, fill="silence")
        pcm, _ = _read(out)
        assert audioop.rms(pcm, 2) == 0
        assert len(recipe["dropped_frames"]) == recipe["n_frames"]

    def test_repeat_fill_conceals_with_previous_frame(self, tone, tmp_path):
        out = tmp_path / "repeat.wav"
        recipe = frame_drop(tone, out, seed=11, rate=0.3, fill="repeat")
        in_pcm, sample_rate = _read(tone)
        out_pcm, _ = _read(out)
        assert len(out_pcm) == len(in_pcm)  # loss conceals, never shortens
        frame_bytes = int(sample_rate * recipe["frame_ms"] / 1000) * 2
        dropped = set(recipe["dropped_frames"])
        # pick a dropped frame whose predecessor was delivered: the concealment
        # must be that predecessor's bytes, and kept frames must be untouched
        checked = False
        for idx in sorted(dropped):
            if idx == 0 or idx - 1 in dropped:
                continue
            a, b = idx * frame_bytes, (idx + 1) * frame_bytes
            assert out_pcm[a:b] == in_pcm[a - frame_bytes : a]
            checked = True
            break
        assert checked
        for idx in range(recipe["n_frames"]):
            if idx not in dropped:
                a, b = idx * frame_bytes, (idx + 1) * frame_bytes
                assert out_pcm[a:b] == in_pcm[a:b]

    def test_unknown_fill_rejected(self, tone, tmp_path):
        with pytest.raises(ValueError, match="fill"):
            frame_drop(tone, tmp_path / "x.wav", seed=1, fill="extrapolate")


@needs_ffmpeg
class TestPhoneChannel:
    def test_steps_list_names_each_sub_op_in_order(self, tone, tmp_path):
        recipe = phone_channel(tone, tmp_path / "o.wav", seed=5, codec="g711u", loss_rate=0.05)
        assert [s["op"] for s in recipe["steps"]] == [
            "bandlimit",
            "codec_roundtrip",
            "frame_drop",
        ]
        assert recipe["op"] == "phone_channel"
        assert recipe["input_sha256"] == hashlib.sha256(tone.read_bytes()).hexdigest()

    def test_no_loss_omits_the_frame_drop_step(self, tone, tmp_path):
        recipe = phone_channel(tone, tmp_path / "o.wav", seed=5, loss_rate=0.0)
        assert [s["op"] for s in recipe["steps"]] == ["bandlimit", "codec_roundtrip"]

    def test_bandlimit_reduces_energy_above_3400(self, tone, tmp_path):
        out = tmp_path / "o.wav"
        phone_channel(tone, out, seed=5, codec="g711u")
        in_pcm, _ = _read(tone)
        out_pcm, sample_rate = _read(out)
        assert sample_rate == RATE  # downstream ops keep working
        # crude band-energy ratio: above-3400 relative to in-band, in vs out
        hi_in = _goertzel_power(in_pcm, RATE, 6000)
        lo_in = _goertzel_power(in_pcm, RATE, 440)
        hi_out = _goertzel_power(out_pcm, RATE, 6000)
        lo_out = _goertzel_power(out_pcm, RATE, 440)
        assert hi_out / lo_out < (hi_in / lo_in) / 10
        assert audioop.rms(out_pcm, 2) > 300  # still audible speech-band signal

    def test_deterministic_end_to_end(self, tone, tmp_path):
        out1, out2 = tmp_path / "a.wav", tmp_path / "b.wav"
        r1 = phone_channel(tone, out1, seed=9, codec="g726", loss_rate=0.1)
        r2 = phone_channel(tone, out2, seed=9, codec="g726", loss_rate=0.1)
        assert out1.read_bytes() == out2.read_bytes()
        assert r1 == r2


class TestChannelSpecIntegration:
    """D090 wiring: the schema field and the render-path application."""

    def _item_dict(self):
        import yaml

        # any item without a channel; a development-split item, path relative to the repo
        p = Path(__file__).resolve().parents[1] / "items/pilot/t4/vxp-wirerb-0001.yaml"
        return yaml.safe_load(p.read_text())

    def test_channel_field_validates_and_defaults_to_none(self):
        from voxparity.schemas.item import Item

        d = self._item_dict()
        item = Item.model_validate(d)
        assert all(v.channel is None for v in item.variants)
        d["variants"][0]["channel"] = {"codec": "g711u", "loss_rate": 0.02, "seed": 5}
        item2 = Item.model_validate(d)
        assert item2.variants[0].channel.codec == "g711u"
        assert item2.variants[0].channel.loss_rate == 0.02

    def test_channel_rejects_unknown_field_and_bad_loss_rate(self):
        import pytest as _p

        from voxparity.schemas.item import Item

        d = self._item_dict()
        d["variants"][0]["channel"] = {"codec": "g711u", "bogus": 1}
        with _p.raises(ValueError):
            Item.model_validate(d)
        d["variants"][0]["channel"] = {"codec": "g711u", "loss_rate": 1.5}
        with _p.raises(ValueError):
            Item.model_validate(d)

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
    def test_apply_channel_degrades_and_records_recipe(self):
        import audioop

        from voxparity.cli import _apply_channel
        from voxparity.schemas.item import DeliveryVariant, Emotion, GoldAction

        v = DeliveryVariant(
            variant_id="v",
            emotion=Emotion.NEUTRAL,
            intensity=0.3,
            channel={"codec": "g711u", "loss_rate": 0.0, "seed": 3},
            gold=GoldAction(tool="t", args={}, rationale="r"),
        )
        # a 1 s 300 Hz tone
        import math
        import struct
        import wave as _w
        from io import BytesIO

        rate, n = 24000, 24000
        samples = [int(9000 * math.sin(2 * math.pi * 300 * i / rate)) for i in range(n)]
        buf = BytesIO()
        with _w.open(buf, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(struct.pack(f"<{n}h", *samples))
        src = buf.getvalue()
        out, recipe = _apply_channel(src, v)
        assert recipe["op"] == "phone_channel"
        assert [s["op"] for s in recipe["steps"]][:2] == ["bandlimit", "codec_roundtrip"]
        assert out != src
        assert audioop.rms(out[44:], 2) > 100


class TestRadioBandLimitChannel:
    """FLAG-008: codec 'none' = analog radio / found-audio channel matching."""

    def test_codec_none_is_bandlimit_only_and_deterministic(self, tone, tmp_path):
        out1, out2 = tmp_path / "a.wav", tmp_path / "b.wav"
        kw = dict(seed=1, codec="none", low_hz=300, high_hz=3400, band_order=4)
        r1 = phone_channel(tone, out1, **kw)
        r2 = phone_channel(tone, out2, **kw)
        assert [s["op"] for s in r1["steps"]] == ["bandlimit"]
        assert r1["steps"][0]["order"] == 4
        assert r1 == r2 and out1.read_bytes() == out2.read_bytes()
        in_pcm, _ = _read(tone)
        out_pcm, rate = _read(out1)
        assert rate == RATE
        ratio_in = _goertzel_power(in_pcm, RATE, 6000) / _goertzel_power(in_pcm, RATE, 440)
        ratio_out = _goertzel_power(out_pcm, RATE, 6000) / _goertzel_power(out_pcm, RATE, 440)
        assert ratio_out < ratio_in / 10

    def test_single_pass_recipe_carries_no_order_key(self, tone, tmp_path):
        # D090 recipes predate band_order; the default must not change them.
        recipe = phone_channel(tone, tmp_path / "o.wav", seed=1, codec="none")
        assert "order" not in recipe["steps"][0]

    def test_codec_none_rejects_bitrate(self, tone, tmp_path):
        with pytest.raises(ValueError, match="bitrate"):
            phone_channel(tone, tmp_path / "x.wav", seed=1, codec="none", bitrate=16000)

    def test_channelspec_accepts_radio_band_and_rejects_inverted_band(self):
        from pydantic import ValidationError

        from voxparity.schemas.item import ChannelSpec

        spec = ChannelSpec(codec="none", low_hz=300, high_hz=3400)
        assert (spec.low_hz, spec.high_hz) == (300, 3400)
        assert ChannelSpec().high_hz == 3400.0  # existing specs unchanged
        with pytest.raises(ValidationError, match="low_hz"):
            ChannelSpec(codec="none", low_hz=3400, high_hz=300)
        with pytest.raises(ValidationError, match="bitrate"):
            ChannelSpec(codec="none", bitrate=16000)
