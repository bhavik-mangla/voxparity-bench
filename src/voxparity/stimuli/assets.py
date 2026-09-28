"""Procedurally synthesized scene assets (D062/D065).

Every `synth:` asset is generated locally from a seed — no downloads, no
licenses, byte-reproducible. That covers the v0.1 scene items: a radio squelch
IS band noise, a CO alarm IS a beep pattern, and a traffic bed is shaped noise.
Recorded corpora (MUSAN, FSD50K CC0/CC-BY subsets) arrive later behind the
pack-recipe `fetch:` path; nothing in this module fetches anything.

Stdlib only, like mix.py.
"""

from __future__ import annotations

import math
import random
import struct
import wave
from io import BytesIO

_WIDTH = 2


def _pack(samples: list[int], rate: int) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(_WIDTH)
        f.setframerate(rate)
        f.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def _clip(v: float) -> int:
    return max(-32767, min(32767, int(v)))


def _white(rng: random.Random, n: int, level: float) -> list[float]:
    return [rng.uniform(-1, 1) * level for _ in range(n)]


def _lowpass(xs: list[float], rate: int, cutoff_hz: float) -> list[float]:
    """One-pole IIR — crude but plenty for a noise bed's spectral shape."""
    alpha = 1.0 / (1.0 + rate / (2 * math.pi * cutoff_hz))
    out, y = [], 0.0
    for x in xs:
        y += alpha * (x - y)
        out.append(y)
    return out


def _radio_squelch(rng: random.Random, rate: int, dur_s: float) -> list[int]:
    n = int(rate * dur_s)
    xs = _white(rng, n, 0.9)
    # Slight flutter so it reads as RF interference rather than pink hiss.
    return [
        _clip(x * (0.75 + 0.25 * math.sin(2 * math.pi * 7 * i / rate)) * 32767)
        for i, x in enumerate(xs)
    ]


def _co_alarm_chirp(rate: int, dur_s: float) -> list[int]:
    """UL 2034 temporal-4-ish pattern: four short 3.1 kHz beeps, then a pause."""
    beep_n, gap_n, pause_n = int(rate * 0.1), int(rate * 0.1), int(rate * 4.0)
    cycle: list[int] = []
    for _ in range(4):
        cycle += [
            _clip(math.sin(2 * math.pi * 3100 * i / rate) * 0.7 * 32767) for i in range(beep_n)
        ]
        cycle += [0] * gap_n
    cycle += [0] * pause_n
    n = int(rate * dur_s)
    return (cycle * (n // len(cycle) + 1))[:n]


def _traffic(rng: random.Random, rate: int, dur_s: float) -> list[int]:
    """Low-passed brown noise with slow swells — a distant-road bed."""
    n = int(rate * dur_s)
    y, brown = 0.0, []
    for _ in range(n):
        y = max(-1.0, min(1.0, y + rng.uniform(-1, 1) * 0.02))
        brown.append(y)
    shaped = _lowpass(brown, rate, 400)
    swell_period = rate * 3
    return [
        _clip(x * (0.6 + 0.4 * math.sin(2 * math.pi * i / swell_period)) * 6.0 * 32767)
        for i, x in enumerate(shaped)
    ]


def _medical_beep(rate: int, dur_s: float) -> list[int]:
    """Bedside-monitor cadence: one short 1 kHz beep per second."""
    beep_n, rest_n = int(rate * 0.15), int(rate * 0.85)
    cycle = [
        _clip(math.sin(2 * math.pi * 1000 * i / rate) * 0.5 * 32767) for i in range(beep_n)
    ] + [0] * rest_n
    n = int(rate * dur_s)
    return (cycle * (n // len(cycle) + 1))[:n]


def _ecall_modem(rate: int, dur_s: float) -> list[int]:
    """EN 16072-style in-band MSD signalling: a 2-second burst of alternating
    1.0 kHz / 1.5 kHz tones (100 ms per tone), then a pause — the audible
    signature of an eCall data transmission, not a bit-accurate modem."""
    tone_n = int(rate * 0.1)
    burst: list[int] = []
    for k in range(20):  # 20 x 100 ms = the 2.0 s burst
        freq = 1000.0 if k % 2 == 0 else 1500.0
        burst += [
            _clip(math.sin(2 * math.pi * freq * i / rate) * 0.7 * 32767) for i in range(tone_n)
        ]
    cycle = burst + [0] * int(rate * 4.0)
    n = int(rate * dur_s)
    return (cycle * (n // len(cycle) + 1))[:n]


def _line_drop(rng: random.Random, rate: int) -> list[int]:
    """A telephone-line drop artifact: one broadband click, a breath of static,
    then dead silence — the sound a caller's channel makes when it cuts out
    (the TRUNCATION scene kind's tail, D083)."""
    click = [_clip(x * 32767) for x in _white(rng, int(rate * 0.012), 0.95)]
    static = [
        _clip(x * 32767 * (1.0 - i / (rate * 0.08)))
        for i, x in enumerate(_white(rng, int(rate * 0.08), 0.25))
    ]
    silence = [0] * int(rate * 0.4)
    return click + static + silence


def _labored_breathing(rng: random.Random, rate: int, dur_s: float) -> list[int]:
    """Effortful, wheezy breathing: band-shaped noise under a slow in/out
    envelope with a faint tonal wheeze on the inhale (D086 breathing-only
    open-line family). Procedural approximation — flagged for the human
    listen pass; an FSD50K CC0 clip can replace it via the pack recipe."""
    n = int(rate * dur_s)
    xs = _lowpass(_white(rng, n, 0.9), rate, 1400.0)
    hp_bed = _lowpass(xs, rate, 250.0)
    out = []
    cycle = 3.2  # seconds per labored breath
    for i in range(n):
        t = i / rate
        phase = (t % cycle) / cycle
        # sharp effortful inhale, slower exhale, gap between breaths
        if phase < 0.35:
            env = math.sin(math.pi * phase / 0.35) ** 2
            wheeze = 0.18 * math.sin(2 * math.pi * 460 * t) * env
        elif phase < 0.8:
            env = 0.7 * math.sin(math.pi * (phase - 0.35) / 0.45) ** 2
            wheeze = 0.0
        else:
            env, wheeze = 0.04, 0.0
        out.append(_clip(((xs[i] - hp_bed[i]) * env + wheeze) * 32767 * 0.8))
    return out


def bandlimit_wav(wav: bytes, low_hz: float = 300.0, high_hz: float = 3400.0) -> bytes:
    """Telephone/broadcast-band a WAV: one-pole high-pass at ``low_hz`` plus
    one-pole low-pass at ``high_hz``. Used on tv_ad background voices so
    far-field media carries the channel cues (thin, boxy spectrum) a listener
    uses to tell a broadcast from a co-present speaker (Sep-6 false-trigger
    family)."""
    with wave.open(BytesIO(wav), "rb") as f:
        rate, n = f.getframerate(), f.getnframes()
        pcm = f.readframes(n)
    xs = [s / 32768.0 for s in struct.unpack(f"<{len(pcm) // _WIDTH}h", pcm)]
    lo = _lowpass(xs, rate, high_hz)
    hp_bed = _lowpass(lo, rate, low_hz)
    out = [_clip((a - b) * 32767) for a, b in zip(lo, hp_bed, strict=False)]
    return _pack(out, rate)


def synthesize_asset(
    asset_id: str, *, rate: int = 24000, dur_s: float = 8.0, seed: int = 0
) -> bytes:
    """Render a `synth:` asset to WAV bytes. Deterministic in (id, rate, dur, seed)."""
    kind = asset_id.removeprefix("synth:")
    rng = random.Random(seed)
    if kind == "white_noise":
        samples = [_clip(x * 32767) for x in _white(rng, int(rate * dur_s), 0.8)]
    elif kind == "radio_squelch":
        samples = _radio_squelch(rng, rate, dur_s)
    elif kind == "co_alarm_chirp":
        samples = _co_alarm_chirp(rate, dur_s)
    elif kind == "traffic":
        samples = _traffic(rng, rate, dur_s)
    elif kind == "medical_beep":
        samples = _medical_beep(rate, dur_s)
    elif kind == "ecall_modem":
        samples = _ecall_modem(rate, dur_s)
    elif kind == "line_drop":
        samples = _line_drop(rng, rate)  # fixed shape; dur_s does not apply
    elif kind == "labored_breathing":
        samples = _labored_breathing(rng, rate, dur_s)
    else:
        raise ValueError(f"unknown synth asset {asset_id!r}")
    return _pack(samples, rate)
