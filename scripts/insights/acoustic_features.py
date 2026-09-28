"""Per-clip acoustic features for every frozen clip (Lens 3: what makes a cue actionable).

Reads bank-freeze-2026-09-15 (freeze.json), the items it names and the content-
addressed WAV store, and writes one row per (item, variant, engine) clip with
deterministic, numpy-only prosody and spectrum features. No model, no network.

Features (all computed on the clip as heard, i.e. after any scene mix):
- duration_s, active_span_s (frames within 30 dB of the clip's p95 frame RMS),
  speech_rate_wps (transcript words / active span), leading_silence_s
- loudness_dbfs (RMS over active frames), dyn_range_db (p95 - p10 active frame dB),
  floor_db (p10 - p95 frame dB over the whole clip; a mixed bed raises it)
- F0 by normalised autocorrelation (NCCF, 40 ms frames, 60-500 Hz, voiced when the
  peak >= 0.5; the D079 autocorrelation approach): f0_median_st (semitones re
  100 Hz), f0_range_st (p10-p90), f0_jitter_st (median |frame-to-frame| step),
  voiced_fraction (voiced / active), periodicity (mean NCCF peak on voiced frames,
  a harmonicity proxy: low for breathy/whispered voice)
- zcr (mean zero-crossing rate over active frames)
- spectral_tilt_db_oct (slope of the long-term active spectrum, 100-5000 Hz, dB
  per octave), hf_ratio_db (energy >1 kHz minus <1 kHz), ltas (16 log bands,
  level-normalised, for a spectral distance)
- pause_count, pause_frac, pause_max_s (silent runs >= 150 ms inside the span)

Usage (from anywhere; paths default to the sibling worktrees):
    python scripts/insights/acoustic_features.py \
        --bank $VXP_BANK --stimuli $VXP_MAIN/stimuli \
        --out docs/insights/acoustic_features.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from voxparity.paths import bank_root

HERE = Path(__file__).resolve().parents[2]
ENGINES = ("gemini", "human", "kokoro", "qwen3tts-cv", "qwen3tts-vd", "found")
# Aura is excluded from every run by policy (Deepgram terms, D107): no features.

HOP_S = 0.010
RMS_WIN_S = 0.025
F0_WIN_S = 0.040
F0_MIN, F0_MAX = 60.0, 500.0
VOICED_NCCF = 0.5
ACTIVE_DB = 30.0
PAUSE_DB = 35.0
MIN_PAUSE_S = 0.15
N_BANDS = 16


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: sample width {sw}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    out: list[tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def nccf_f0(x: np.ndarray, sr: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """(f0 Hz or nan, nccf peak) per hop, via FFT autocorrelation with energy normalisation."""
    win = round(F0_WIN_S * sr)
    fr = frames(x, win, hop)
    fr = fr - fr.mean(axis=1, keepdims=True)
    nfft = 1 << (2 * win - 1).bit_length()
    spec = np.fft.rfft(fr, nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, :win]
    lag_min = int(sr / F0_MAX)
    lag_max = min(int(sr / F0_MIN), win - 2)
    sq = np.cumsum(fr**2, axis=1)
    total = sq[:, -1:]
    lags = np.arange(lag_min, lag_max + 1)
    # e0 = sum x[0:W-tau]^2 ; e_tau = sum x[tau:W]^2
    e0 = sq[:, win - 1 - lags]
    e_tau = total - np.concatenate([np.zeros((len(fr), 1)), sq], axis=1)[:, lags]
    denom = np.sqrt(np.maximum(e0 * e_tau, 1e-20))
    nccf = ac[:, lags] / denom
    best = np.argmax(nccf, axis=1)
    peak = nccf[np.arange(len(fr)), best]
    # octave guard: the smallest lag whose NCCF is within 0.9 of the peak
    f0 = np.full(len(fr), np.nan)
    for i in range(len(fr)):
        if peak[i] < VOICED_NCCF:
            continue
        cand = np.nonzero(nccf[i] >= 0.9 * peak[i])[0]
        # first local maximum among candidates
        j = int(cand[0])
        while j + 1 < nccf.shape[1] and nccf[i, j + 1] > nccf[i, j]:
            j += 1
        # parabolic interpolation
        if 0 < j < nccf.shape[1] - 1:
            a, b, c = nccf[i, j - 1], nccf[i, j], nccf[i, j + 1]
            d = a - 2 * b + c
            off = 0.5 * (a - c) / d if d != 0 else 0.0
        else:
            off = 0.0
        f0[i] = sr / (lags[j] + off)
    return f0, peak


def features(x: np.ndarray, sr: int, n_words: int) -> dict[str, Any]:
    hop = round(HOP_S * sr)
    win = round(RMS_WIN_S * sr)
    fr = frames(x, win, hop)
    rms = np.sqrt(np.mean(fr**2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    ref = float(np.percentile(db, 95))
    active = db > ref - ACTIVE_DB
    out: dict[str, Any] = {"duration_s": round(len(x) / sr, 3)}
    if not active.any():
        return out
    first = int(np.argmax(active))
    last = int(len(active) - 1 - np.argmax(active[::-1]))
    span = (last - first + 1) * HOP_S
    out["active_span_s"] = round(span, 3)
    out["leading_silence_s"] = round(first * HOP_S, 3)
    out["speech_rate_wps"] = round(n_words / span, 3) if span > 0 else None
    act_db = db[active]
    out["loudness_dbfs"] = round(float(20 * np.log10(np.sqrt(np.mean(rms[active] ** 2)))), 2)
    out["dyn_range_db"] = round(float(np.percentile(act_db, 95) - np.percentile(act_db, 10)), 2)
    out["floor_db"] = round(float(np.percentile(db, 10) - ref), 2)
    # pauses: quiet runs inside the active span
    inside = db[first : last + 1] < ref - PAUSE_DB
    prs = [(a, b) for a, b in runs(inside) if (b - a) * HOP_S >= MIN_PAUSE_S]
    out["pause_count"] = len(prs)
    out["pause_frac"] = round(sum(b - a for a, b in prs) * HOP_S / span, 4) if span else None
    out["pause_max_s"] = round(max(((b - a) * HOP_S for a, b in prs), default=0.0), 3)
    # zero-crossing rate over active frames
    zc = np.mean(np.abs(np.diff(np.signbit(fr).astype(np.int8), axis=1)), axis=1)
    out["zcr"] = round(float(zc[active].mean()), 4)
    # F0
    f0, peak = nccf_f0(x, sr, hop)
    m = min(len(f0), len(active))
    act = active[:m]
    v = act & np.isfinite(f0[:m])
    out["voiced_fraction"] = round(float(v.sum() / act.sum()), 4)
    out["periodicity"] = round(float(peak[:m][v].mean()), 4) if v.any() else None
    if v.sum() >= 10 and v.sum() / act.sum() >= 0.15:
        st = 12 * np.log2(f0[:m][v] / 100.0)
        # octave-error guard: drop frames more than 10 st from the clip median
        st = st[np.abs(st - np.median(st)) <= 10.0]
        out["f0_median_st"] = round(float(np.median(st)), 3)
        p10, p90 = np.percentile(st, [10, 90])
        out["f0_range_st"] = round(float(p90 - p10), 3)
        # frame-to-frame steps inside voiced runs only
        steps = []
        for a, b in runs(v):
            if b - a >= 3:
                s = 12 * np.log2(f0[a:b] / 100.0)
                d = np.abs(np.diff(s))
                steps.append(d[d < 6.0])
        out["f0_jitter_st"] = round(float(np.median(np.concatenate(steps))), 4) if steps else None
    else:
        out["f0_median_st"] = out["f0_range_st"] = out["f0_jitter_st"] = None
    # long-term average spectrum over active frames
    nfft = 1024
    sfr = frames(x, nfft, hop)[: len(active)]
    act_s = active[: len(sfr)]
    spec = np.mean(np.abs(np.fft.rfft(sfr[act_s] * np.hanning(nfft), axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    sdb = 10 * np.log10(spec + 1e-20)
    band = (freqs >= 100) & (freqs <= 5000)
    slope = np.polyfit(np.log2(freqs[band]), sdb[band], 1)[0]
    out["spectral_tilt_db_oct"] = round(float(slope), 3)
    lo = spec[(freqs >= 80) & (freqs < 1000)].sum()
    hi = spec[(freqs >= 1000) & (freqs <= 7500)].sum()
    out["hf_ratio_db"] = round(float(10 * np.log10(hi / lo)), 3) if lo > 0 and hi > 0 else None
    edges = np.geomspace(100, 7500, N_BANDS + 1)
    bands = np.array(
        [spec[(freqs >= a) & (freqs < b)].mean() for a, b in itertools.pairwise(edges)]
    )
    bdb = 10 * np.log10(bands + 1e-20)
    out["ltas"] = [round(float(b), 2) for b in (bdb - bdb.mean())]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", type=Path, default=bank_root())
    ap.add_argument("--stimuli", type=Path, default=HERE.parent / "voxparity" / "stimuli")
    ap.add_argument(
        "--out", type=Path, default=HERE / "docs" / "insights" / "acoustic_features.json"
    )
    a = ap.parse_args(argv)
    sys.path.insert(0, str(HERE / "src"))
    import yaml

    from voxparity.cli import _iter_item_files, load_item

    freeze = json.loads((a.bank / "freeze" / "2026-09-15" / "freeze.json").read_text())
    items: dict[str, Any] = {}
    for d in freeze["item_dirs"]:
        for f in _iter_item_files(a.bank / d):
            it = load_item(f)
            items[it.id] = it
    manifest = {
        r["sha256"]: r for r in yaml.safe_load((a.bank / "stimuli" / "manifest.yaml").read_text())
    }
    rows = []
    for it in freeze["items"]:
        item = items.get(it["id"])
        if item is None:
            continue
        n_words = len(item.transcript.split())
        for v in it["variants"]:
            if v.get("status") != "usable":
                continue
            for eng, sha in (v.get("clips") or {}).items():
                if eng not in ENGINES:
                    continue
                x, sr = read_wav(a.stimuli / f"{sha}.wav")
                f = features(x, sr, n_words)
                m = manifest.get(sha) or {}
                sc = m.get("scene") or {}
                rows.append(
                    {
                        "item_id": it["id"],
                        "variant_id": v["variant_id"],
                        "engine": eng,
                        "sha256": sha,
                        "voice": m.get("voice"),
                        "sr": sr,
                        "scene_op": sc.get("op"),
                        "scene_asset": sc.get("asset"),
                        "scene_snr_db": sc.get("snr_db"),
                        **f,
                    }
                )
        print(f"{it['id']}: {len(rows)} clips", file=sys.stderr, end="\r")
    rows.sort(key=lambda r: (r["item_id"], r["variant_id"], r["engine"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "freeze": freeze["freeze_id"],
                "method": {
                    "hop_s": HOP_S,
                    "f0": "NCCF autocorrelation, 40 ms, 60-500 Hz, voiced at peak>=0.5",
                    "active": f"frames within {ACTIVE_DB} dB of the clip p95",
                    "pause": f">= {MIN_PAUSE_S}s below p95-{PAUSE_DB} dB inside the span",
                    "tilt": "LTAS slope 100-5000 Hz, dB/octave",
                },
                "clips": rows,
            },
            indent=0,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwrote {len(rows)} clips -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
