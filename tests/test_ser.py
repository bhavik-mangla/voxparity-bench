"""Tests for the local SER gate.

These MUST run with neither funasr, torch, nor any downloaded weights present:
CI is a free-tier box and the gate's whole point is that it is optional
infrastructure. funasr is therefore faked at the sys.modules level, which also
pins the exact call shape we verified against FunASR's emotion2vec inference()
(a path in; a list of dicts with "labels"/"scores" out).
"""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import types
import wave
from typing import Any, ClassVar

import pytest

from voxparity.providers import ser
from voxparity.providers.ser import (
    EMOTION_MAP,
    MODEL_LABELS,
    VOXPARITY_EMOTIONS,
    SERDependencyError,
    SERGate,
)

# The bilingual token form FunASR actually returns (tokens.txt ships
# "生气/angry"; the README shows the reversed order), plus the literal <unk>.
FUNASR_TOKENS = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
]


def make_wav(sample_rate: int, *, seconds: float = 0.5, channels: int = 1) -> bytes:
    """A 220 Hz sine, well below Nyquist at every rate under test."""
    n = int(sample_rate * seconds)
    frames = bytearray()
    for i in range(n):
        value = int(12000 * math.sin(2 * math.pi * 220 * i / sample_rate))
        frames += struct.pack("<h", value) * channels
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


class FakeAutoModel:
    """Stand-in for funasr.AutoModel with the verified generate() contract."""

    instances: ClassVar[list[FakeAutoModel]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        # Default: an unambiguous "angry" clip with scores that do NOT sum to 1,
        # so normalisation is actually exercised.
        self.scores: list[float] = [0.80, 0.05, 0.02, 0.01, 0.05, 0.01, 0.02, 0.02, 0.02]
        FakeAutoModel.instances.append(self)

    def generate(self, wav: str, **kwargs: Any) -> list[dict[str, Any]]:
        with wave.open(wav, "rb") as w:
            params = {
                "framerate": w.getframerate(),
                "channels": w.getnchannels(),
                "sampwidth": w.getsampwidth(),
                "nframes": w.getnframes(),
            }
        self.calls.append({"path": wav, "kwargs": kwargs, "wav": params})
        return [{"key": "test", "labels": list(FUNASR_TOKENS), "scores": list(self.scores)}]


@pytest.fixture
def fake_funasr(monkeypatch: pytest.MonkeyPatch) -> type[FakeAutoModel]:
    FakeAutoModel.instances = []
    module = types.ModuleType("funasr")
    module.AutoModel = FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", module)
    return FakeAutoModel


# --------------------------------------------------------------------------
# Mapping contract
# --------------------------------------------------------------------------
def test_mapping_covers_every_voxparity_emotion() -> None:
    assert set(EMOTION_MAP) == set(VOXPARITY_EMOTIONS)


def test_mapping_sources_are_real_model_labels() -> None:
    for emotion, mapping in EMOTION_MAP.items():
        assert mapping.sources, f"{emotion} has no sources"
        for src in mapping.sources:
            assert src in MODEL_LABELS, f"{emotion} maps to unknown model label {src!r}"


def test_lossy_mappings_are_declared_not_silent() -> None:
    """The four emotions with no native class must never claim `direct`."""
    for emotion in ("frustrated", "resigned", "urgent", "anxious"):
        mapping = EMOTION_MAP[emotion]
        assert mapping.fidelity != "direct", f"{emotion} claims a fidelity it does not have"
        assert mapping.note.strip(), f"{emotion} is lossy but undocumented"


def test_native_classes_are_direct() -> None:
    for emotion in ("neutral", "happy", "angry", "sad"):
        assert EMOTION_MAP[emotion].fidelity == "direct"


def test_unmapped_labels_are_the_models_escape_hatches() -> None:
    assert set(ser.UNMAPPED_LABELS) == {"other", "unknown"}


def test_to_voxparity_is_total_over_the_enum() -> None:
    dist = dict.fromkeys(MODEL_LABELS, 1 / len(MODEL_LABELS))
    assert set(ser.to_voxparity(dist)) == set(VOXPARITY_EMOTIONS)


def test_to_voxparity_accepts_raw_funasr_tokens() -> None:
    dist = {"生气/angry": 0.9, "<unk>": 0.1}
    projected = ser.to_voxparity(dist)
    assert projected["angry"] == pytest.approx(0.9)
    # frustrated shares angry's mass by design — this is the documented conflation.
    assert projected["frustrated"] == pytest.approx(0.9)
    assert ser.unmapped_mass(dist) == pytest.approx(0.1)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def test_classify_normalises_and_canonicalises(fake_funasr: type[FakeAutoModel]) -> None:
    gate = SERGate()
    dist = gate.classify(make_wav(16_000))
    assert set(dist) == set(MODEL_LABELS)
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-5)
    assert max(dist, key=lambda k: dist[k]) == "angry"


def test_classify_uses_the_documented_generate_kwargs(fake_funasr: type[FakeAutoModel]) -> None:
    gate = SERGate()
    gate.classify(make_wav(16_000))
    call = fake_funasr.instances[0].calls[0]
    assert call["kwargs"]["granularity"] == "utterance"
    assert call["kwargs"]["extract_embedding"] is False


def test_degenerate_scores_raise_rather_than_pass(fake_funasr: type[FakeAutoModel]) -> None:
    gate = SERGate()
    fake_funasr.instances[0].scores = [0.0] * len(FUNASR_TOKENS)
    with pytest.raises(ser.SERError):
        gate.classify(make_wav(16_000))


# --------------------------------------------------------------------------
# Gate output
# --------------------------------------------------------------------------
def test_gate_detail_is_json_serialisable(fake_funasr: type[FakeAutoModel]) -> None:
    gate = SERGate()
    passed, detail = gate.gate(make_wav(16_000), "angry")
    assert passed is True
    # The manifest is written with json.dump, so primitives only — round-tripping
    # must be an identity, not merely "does not raise".
    assert json.loads(json.dumps(detail)) == detail


def test_gate_detail_records_the_full_distribution(fake_funasr: type[FakeAutoModel]) -> None:
    gate = SERGate()
    _, detail = gate.gate(make_wav(16_000), "angry", threshold=0.5)
    assert detail["top_label"] == "angry"
    assert set(detail["distribution"]) == set(MODEL_LABELS)
    assert detail["intended_probability"] == pytest.approx(detail["distribution"]["angry"])
    assert detail["threshold"] == 0.5
    assert detail["license_flag"] == "PX-004"


def test_gate_fails_when_intended_mass_is_below_threshold(
    fake_funasr: type[FakeAutoModel],
) -> None:
    gate = SERGate()
    passed, detail = gate.gate(make_wav(16_000), "sad")
    assert passed is False
    assert detail["top_voxparity_emotion"] != "sad"


def test_conflated_emotions_carry_an_advisory(fake_funasr: type[FakeAutoModel]) -> None:
    gate = SERGate()
    _, direct = gate.gate(make_wav(16_000), "angry")
    _, conflated = gate.gate(make_wav(16_000), "frustrated")
    assert direct["advisory"] is None
    assert conflated["mapping_fidelity"] == "conflated"
    assert conflated["advisory"]


def test_gate_rejects_emotions_outside_our_enum(fake_funasr: type[FakeAutoModel]) -> None:
    gate = SERGate()
    with pytest.raises(ValueError, match="unknown intended emotion"):
        gate.gate(make_wav(16_000), "sarcastic")


# --------------------------------------------------------------------------
# Dependency handling
# --------------------------------------------------------------------------
def test_import_does_not_require_funasr() -> None:
    assert "funasr" not in sys.modules or sys.modules["funasr"] is not None
    assert "torch" not in sys.modules


def test_missing_funasr_raises_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "funasr", None)  # forces ImportError on import
    with pytest.raises(SERDependencyError) as exc:
        SERGate()
    message = str(exc.value)
    assert "funasr" in message
    assert "install" in message
    assert ser.MODEL_ID in message


# --------------------------------------------------------------------------
# Audio normalisation
# --------------------------------------------------------------------------
@pytest.fixture(params=["auto", "pure-python"])
def resampler_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Exercise both paths on every interpreter.

    audioop exists on 3.11/3.12 and is gone on 3.13+ (PEP 594), so whichever
    branch the local interpreter picks, the other one would otherwise never be
    tested until it breaks in production.
    """
    if request.param == "pure-python":
        monkeypatch.setattr(ser, "HAVE_AUDIOOP", False)
    return str(request.param)


@pytest.mark.parametrize("rate", [8_000, 16_000, 22_050, 24_000, 44_100, 48_000])
def test_resamples_to_16k_mono(rate: int, resampler_backend: str) -> None:
    out, info = ser.resample_to_16k_mono(make_wav(rate, seconds=0.5))
    with wave.open(io.BytesIO(out), "rb") as w:
        assert w.getframerate() == 16_000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        # Duration must survive the rate change; a rounding slop of one frame is
        # fine, a factor-of-N stretch is the bug this guards against.
        assert w.getnframes() / 16_000 == pytest.approx(0.5, abs=0.01)
    assert info["input_sample_rate"] == rate
    assert info["resampled"] is (rate != 16_000)
    assert info["duration_s"] == pytest.approx(0.5, abs=0.01)


def test_stereo_is_downmixed(resampler_backend: str) -> None:
    out, info = ser.resample_to_16k_mono(make_wav(44_100, seconds=0.3, channels=2))
    with wave.open(io.BytesIO(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getnframes() / 16_000 == pytest.approx(0.3, abs=0.01)
    assert info["input_channels"] == 2


@pytest.mark.parametrize("rate", [24_000, 44_100])
def test_model_only_ever_sees_16k_mono(rate: int, fake_funasr: type[FakeAutoModel]) -> None:
    gate = SERGate()
    gate.classify(make_wav(rate, seconds=0.4))
    seen = fake_funasr.instances[0].calls[0]["wav"]
    assert seen["framerate"] == 16_000
    assert seen["channels"] == 1
    assert seen["sampwidth"] == 2


def test_temp_file_is_cleaned_up(fake_funasr: type[FakeAutoModel]) -> None:
    import os

    gate = SERGate()
    gate.classify(make_wav(16_000))
    assert not os.path.exists(fake_funasr.instances[0].calls[0]["path"])


def test_non_wav_bytes_fail_loudly() -> None:
    with pytest.raises(ser.SERAudioError):
        ser.resample_to_16k_mono(b"this is not a wav file")
