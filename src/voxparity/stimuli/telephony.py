"""Deterministic telephone-channel DSP (Sep-7 toolbox round, recommendation 2).

Every clip in the bank is studio-clean 24 kHz TTS, but real voice agents hear
8 kHz codec-degraded telephony. A benchmark whose stimuli are all studio-clean
risks measuring behavior that vanishes on a real phone line — so the channel
is a first-class, recorded manipulation, not an afterthought. Like every other
stimulus op (mix.py, D065), each function here is DETERMINISTIC and returns
the recipe dict that produced its output (inputs by sha256, all parameters,
the seed, the tool version), so a released clip is reproducible from the
manifest (D065/D086 recipe discipline; absence of measurement is never a
negative measurement, D046 — ffmpeg failures raise, they do not write files).

Three layers:
- ``codec_roundtrip`` — encode-then-decode through a real telephony codec via
  the local ffmpeg (G.711 mu/A-law, G.722, G.726, G.723.1 — all verified
  present on this machine; GSM/AMR are decode-only and unsupported here).
  Audio is resampled to the codec's native rate and back to the source rate,
  so downstream ops keep working. ffmpeg runs with ``-bitexact`` and metadata
  stripped: the same input produces byte-identical output.
- ``frame_drop`` — seeded packet loss: fixed-size frames dropped with a given
  probability, concealed by repeating the previous frame (what a real jitter
  buffer does) or by silence. Pure stdlib; the recipe records the ACTUAL
  dropped frame indices so reproducibility is auditable, not asserted.
- ``phone_channel`` — the composed pipeline: 300-3400 Hz band-limit (reusing
  assets.bandlimit_wav) -> codec round-trip -> optional frame drop, returning
  one recipe whose ``steps`` list carries every sub-recipe in order.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

from voxparity.stimuli.assets import bandlimit_wav
from voxparity.stimuli.mix import _SAMPLE_WIDTH, _read_mono, _sha256, _write_mono

# codec key -> (ffmpeg encoder, native sample rate, allowed bitrates, default bitrate)
# Rates per the ITU specs: G.711/G.726/G.723.1 are narrowband (8 kHz);
# G.722 is wideband (16 kHz). G.711/G.722 are fixed-rate (bitrate must be None).
_CODECS: dict[str, tuple[str, int, tuple[int, ...] | None, int | None]] = {
    "g711u": ("pcm_mulaw", 8000, None, None),
    "g711a": ("pcm_alaw", 8000, None, None),
    "g722": ("g722", 16000, None, None),
    "g726": ("adpcm_g726", 8000, (16000, 24000, 32000, 40000), 32000),
    "g723_1": ("g723_1", 8000, (5300, 6300), 6300),
}

_FILLS = ("repeat", "silence")


@lru_cache(maxsize=1)
def _ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()[0].strip()


def _run_ffmpeg(args: list[str], codec: str, stage: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"ffmpeg not found on PATH — cannot {stage} {codec}")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        raise RuntimeError(
            f"ffmpeg {stage} failed for codec {codec!r} (exit {proc.returncode}): "
            + " | ".join(tail)
        )


# Strip metadata and force bit-exact muxing so re-running a recipe reproduces
# the file byte for byte (verified live on this ffmpeg build).
_BITEXACT = ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact"]


def codec_roundtrip(
    in_path: Path,
    out_path: Path,
    *,
    codec: str = "g726",
    bitrate: int | None = None,
) -> dict[str, Any]:
    """Encode then decode ``in_path`` through a telephony codec via ffmpeg.

    The output is 16-bit mono PCM at the SOURCE sample rate (down to the
    codec's native rate and back), so it slots into the existing pipeline. The
    degradation is real: band loss from the resample, quantization noise from
    the codec — the same channel a caller's voice crosses on an actual line.
    """
    if codec not in _CODECS:
        raise ValueError(f"unknown codec {codec!r}; supported: {sorted(_CODECS)}")
    encoder, codec_rate, allowed, default = _CODECS[codec]
    if allowed is None:
        if bitrate is not None:
            raise ValueError(f"codec {codec!r} is fixed-rate; bitrate must be None")
    else:
        bitrate = default if bitrate is None else bitrate
        if bitrate not in allowed:
            raise ValueError(f"codec {codec!r} bitrate {bitrate} not in {allowed}")

    source_rate = _wav_rate(in_path)
    with tempfile.TemporaryDirectory() as td:
        enc = Path(td) / "enc.wav"
        enc_args = ["-i", str(in_path), "-ar", str(codec_rate), "-ac", "1", "-c:a", encoder]
        if bitrate is not None:
            enc_args += ["-b:a", str(bitrate)]
        _run_ffmpeg([*enc_args, *_BITEXACT, str(enc)], codec, "encode")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            [
                "-i",
                str(enc),
                "-ar",
                str(source_rate),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                *_BITEXACT,
                str(out_path),
            ],
            codec,
            "decode",
        )
    return {
        "op": "codec_roundtrip",
        "codec": codec,
        "bitrate": bitrate,
        "codec_rate": codec_rate,
        "source_rate": source_rate,
        "input_sha256": _sha256(in_path),
        "ffmpeg_version": _ffmpeg_version(),
    }


def _wav_rate(path: Path) -> int:
    """Frame rate from the WAV header alone (no payload read)."""
    with wave.open(str(path), "rb") as f:
        return f.getframerate()


def frame_drop(
    in_path: Path,
    out_path: Path,
    *,
    seed: int,
    rate: float = 0.02,
    frame_ms: int = 20,
    fill: str = "repeat",
) -> dict[str, Any]:
    """Seeded packet loss: drop each ``frame_ms`` frame with probability ``rate``.

    ``fill="repeat"`` conceals a dropped frame with the previous delivered
    frame (what a real jitter buffer plays; the very first frame, having no
    predecessor, conceals to silence); ``fill="silence"`` leaves a hard gap.
    Pure stdlib — no ffmpeg. The recipe records the actual dropped frame
    indices so a reader can confirm reproducibility instead of trusting it.
    """
    if fill not in _FILLS:
        raise ValueError(f"unknown fill {fill!r}; supported: {_FILLS}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"drop rate {rate} outside [0, 1]")
    pcm, sample_rate = _read_mono(in_path)
    frame_bytes = int(sample_rate * frame_ms / 1000) * _SAMPLE_WIDTH
    rng = random.Random(seed)
    out = bytearray()
    dropped: list[int] = []
    prev = b"\x00" * frame_bytes
    for idx, start in enumerate(range(0, len(pcm), frame_bytes)):
        frame = pcm[start : start + frame_bytes]
        if rng.random() < rate:
            dropped.append(idx)
            source = prev if fill == "repeat" else b""
            out += (source + b"\x00" * len(frame))[: len(frame)]
        else:
            out += frame
            prev = frame
    _write_mono(out_path, bytes(out), sample_rate)
    return {
        "op": "frame_drop",
        "input_sha256": _sha256(in_path),
        "seed": seed,
        "rate": rate,
        "frame_ms": frame_ms,
        "fill": fill,
        "sample_rate": sample_rate,
        "n_frames": (len(pcm) + frame_bytes - 1) // frame_bytes if pcm else 0,
        "dropped_frames": dropped,
    }


def phone_channel(
    in_path: Path,
    out_path: Path,
    *,
    seed: int,
    codec: str = "g711u",
    bitrate: int | None = None,
    loss_rate: float = 0.0,
    frame_ms: int = 20,
    fill: str = "repeat",
    low_hz: float = 300.0,
    high_hz: float = 3400.0,
    band_order: int = 1,
) -> dict[str, Any]:
    """The composed telephone channel: band-limit -> codec -> optional loss.

    ``codec="none"`` skips the codec round-trip, leaving a pure band-limit
    (plus optional loss): the analog VHF-AM radio path, and the FOUND-AUDIO
    channel-matching rule that passes a rendered twin through the same linear
    band as a found recording (FLAG-008 invariant controls).

    Band-limiting to 300-3400 Hz (the classic voiceband) reuses
    ``assets.bandlimit_wav``; the codec round-trip and seeded frame drop are
    the ops above. The returned recipe carries a ``steps`` list of the
    sub-recipes in application order, so the whole chain is reproducible from
    the manifest exactly like any single manipulation.
    """
    input_sha = _sha256(in_path)
    steps: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Normalize to 16-bit mono first: bandlimit_wav assumes that format.
        pcm, sample_rate = _read_mono(in_path)
        norm = tmp / "norm.wav"
        _write_mono(norm, pcm, sample_rate)
        band = tmp / "band.wav"
        if band_order < 1:
            raise ValueError("band_order must be >= 1")
        banded = norm.read_bytes()
        for _ in range(band_order):
            banded = bandlimit_wav(banded, low_hz, high_hz)
        band.write_bytes(banded)
        band_step: dict[str, Any] = {
            "op": "bandlimit",
            "low_hz": low_hz,
            "high_hz": high_hz,
            "rate": sample_rate,
        }
        if band_order > 1:  # keyed only when used: D090 recipes stay byte-identical
            band_step["order"] = band_order
        steps.append(band_step)
        coded = tmp / "coded.wav"
        if codec == "none":
            # Analog path (VHF-AM radio, FOUND-AUDIO channel matching): the
            # band-limit IS the channel; no digital codec is in the chain.
            if bitrate is not None:
                raise ValueError("codec 'none' has no bitrate")
            coded = band
        else:
            steps.append(codec_roundtrip(band, coded, codec=codec, bitrate=bitrate))
        if loss_rate > 0.0:
            steps.append(
                frame_drop(coded, out_path, seed=seed, rate=loss_rate, frame_ms=frame_ms, fill=fill)
            )
        else:
            out_pcm, out_rate = _read_mono(coded)
            _write_mono(out_path, out_pcm, out_rate)
    return {
        "op": "phone_channel",
        "input_sha256": input_sha,
        "codec": codec,
        "seed": seed,
        "loss_rate": loss_rate,
        "rate": sample_rate,
        "steps": steps,
    }
