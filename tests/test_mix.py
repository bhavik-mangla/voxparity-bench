"""Mixing primitives: determinism, SNR direction, slot locality (D062)."""

import audioop
import math
import struct
import wave
from pathlib import Path

import pytest

from voxparity.stimuli.mix import dtmf, mix_background, overlay_noise_slot

RATE = 16000


def _tone(path: Path, freq: float, dur_s: float, level: float = 0.4) -> None:
    n = int(RATE * dur_s)
    samples = [int(math.sin(2 * math.pi * freq * i / RATE) * level * 32767) for i in range(n)]
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(struct.pack(f"<{n}h", *samples))


def _rms_window(path: Path, start_s: float, dur_s: float) -> int:
    with wave.open(str(path), "rb") as f:
        pcm = f.readframes(f.getnframes())
    a = int(start_s * RATE) * 2
    b = a + int(dur_s * RATE) * 2
    return audioop.rms(pcm[a:b], 2)


@pytest.fixture()
def voices(tmp_path: Path) -> tuple[Path, Path]:
    primary = tmp_path / "primary.wav"
    noise = tmp_path / "noise.wav"
    _tone(primary, 440, 2.0)
    _tone(noise, 3000, 0.5)  # shorter than primary: exercises looping
    return primary, noise


def test_mix_background_deterministic(voices, tmp_path):
    primary, noise = voices
    out1, out2 = tmp_path / "a.wav", tmp_path / "b.wav"
    r1 = mix_background(primary, noise, out1, snr_db=10, seed=7)
    r2 = mix_background(primary, noise, out2, snr_db=10, seed=7)
    assert out1.read_bytes() == out2.read_bytes()
    assert r1["seed"] == r2["seed"] == 7
    out3 = tmp_path / "c.wav"
    mix_background(primary, noise, out3, snr_db=10, seed=8)
    assert out1.read_bytes() != out3.read_bytes()  # seed moves the excerpt


def test_mix_background_snr_direction(voices, tmp_path):
    primary, noise = voices
    quiet, loud = tmp_path / "q.wav", tmp_path / "l.wav"
    mix_background(primary, noise, quiet, snr_db=24, seed=1)
    mix_background(primary, noise, loud, snr_db=0, seed=1)
    # Lower SNR = more background energy in the mix.
    assert _rms_window(loud, 0, 2.0) > _rms_window(quiet, 0, 2.0)


def test_mix_background_delayed_onset(voices, tmp_path):
    primary, noise = voices
    out = tmp_path / "d.wav"
    mix_background(primary, noise, out, snr_db=0, seed=1, start_s=1.0)
    with wave.open(str(primary), "rb") as f:
        original = f.readframes(f.getnframes())
    with wave.open(str(out), "rb") as f:
        mixed = f.readframes(f.getnframes())
    half = int(1.0 * RATE) * 2
    assert mixed[:half] == original[:half]  # untouched before onset
    assert mixed[half:] != original[half:]


def test_overlay_noise_slot_is_local(voices, tmp_path):
    primary, noise = voices
    out = tmp_path / "slot.wav"
    recipe = overlay_noise_slot(
        primary, noise, out, slot_start_s=0.5, slot_dur_s=0.4, snr_db=-12, seed=3
    )
    # Heavy noise inside the slot, primary untouched outside it.
    clean = _rms_window(primary, 0.0, 0.4)
    assert _rms_window(out, 0.5, 0.4) > clean * 1.5
    assert _rms_window(out, 1.2, 0.6) == pytest.approx(_rms_window(primary, 1.2, 0.6), rel=0.01)
    assert recipe["slot_start_s"] == 0.5


def test_overlay_rejects_out_of_range(voices, tmp_path):
    primary, noise = voices
    with pytest.raises(ValueError):
        overlay_noise_slot(primary, noise, tmp_path / "x.wav", slot_start_s=5.0, slot_dur_s=0.5)


def test_dtmf_55_has_row_energy(tmp_path):
    out = tmp_path / "55.wav"
    recipe = dtmf("55", out)
    assert recipe["digits"] == "55"
    # '5' = 770 + 1336 Hz; check the file is non-silent and the expected length.
    with wave.open(str(out), "rb") as f:
        assert f.getnframes() == int(RATE * (0.25 + 0.12)) * 2
        pcm = f.readframes(f.getnframes())
    assert audioop.rms(pcm, 2) > 1000
