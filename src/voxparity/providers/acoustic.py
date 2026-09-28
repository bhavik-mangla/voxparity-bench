"""Local acoustic tagger for the emotion-aware cascade arm (`cascade-open-emo`).

The open cascade (Groq Whisper -> gpt-oss-120b) is the words-only null floor
(D026): nothing about delivery survives ASR. This module produces the explicit
acoustic evidence the emotion-aware arm hands to the SAME text LLM, so the arm
separates "hearing" from "being told".

Three deterministic sources, all local, sharing nothing with any renderer or
model under test:

* **SenseVoiceSmall** (FunASR; D071 — the only model found that emits
  transcript + emotion + audio-event tags in one pass). Its fixed query slots
  are read straight from the CTC posteriors: slot 1 is the utterance emotion
  (HAPPY/SAD/ANGRY/NEUTRAL/FEARFUL/DISGUSTED/SURPRISED/EMO_UNKNOWN), slot 2 the
  utterance audio event (Speech/BGM/Applause/Laughter/Cry/Sneeze/Breath/Cough).
  One forward pass gives both the UNFORCED label (argmax incl. EMO_UNKNOWN) and
  the FORCED label (argmax excl. EMO_UNKNOWN, i.e. ``ban_emo_unk=True``, D072).
  Its transcript is not used: the ASR stays Groq Whisper so the only difference
  from cascade-open is the added evidence.
* **emotion2vec_plus_large** (D044; nine classes, cannot express frustrated,
  resigned or urgent). Class probabilities only.
* **Prosody**: speech rate (ASR word count over the active-speech span), pitch
  median and p10-p90 range (pYIN), voiced fraction and active loudness.

Limits that bound any claim made with this arm (report them, D044/D072):
urgency, sarcasm, whisper, slurring and second voices have NO label here; the
event slot is a single utterance-level decision that is "Speech" on almost every
clip, so background scenes (alarms, prompters, traffic) are mostly invisible to
it; both SER heads neutral-collapse. Model outputs are usable under FunASR Model
Licence v1.1 (PX-004 resolved, D060).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import os
import wave
from pathlib import Path
from typing import Any, Final, Protocol

TAGGER_VERSION: Final = "acoustic-tagger/1"
SENSEVOICE_ID: Final = "FunAudioLLM/SenseVoiceSmall"
DEFAULT_SENSEVOICE_DIR: Final = Path(__file__).resolve().parents[3] / "models" / "sensevoice-small"

# SenseVoice token ids for its query slots (tokenizer chn_jpn_yue_eng_ko_spectok).
SV_EMOTION_TOKENS: Final[dict[str, int]] = {
    "happy": 25001,
    "sad": 25002,
    "angry": 25003,
    "neutral": 25004,
    "fearful": 25005,
    "disgusted": 25006,
    "surprised": 25007,
    "unknown": 25009,
}
SV_EVENT_TOKENS: Final[dict[str, int]] = {
    "speech": 24993,
    "bgm": 24995,
    "laughter": 24997,
    "applause": 24999,
    "crying": 25010,
    "sneeze": 25011,
    "breath": 25012,
    "cough": 25013,
    "unknown_event": 25019,
}
EMOTION_SLOT: Final = 1
EVENT_SLOT: Final = 2

# Fixed, published thresholds (never tuned per cell).
EVENT_MIN_P: Final = 0.10
RATE_SLOW_WPS: Final = 2.0
RATE_FAST_WPS: Final = 3.5
RANGE_NARROW_ST: Final = 3.0
RANGE_WIDE_ST: Final = 8.0
MIN_VOICED: Final = 0.20

# The ONE template every audio cell gets. Changing it changes the arm; bump
# TAGGER_VERSION if you do.
BLOCK_TEMPLATE: Final = (
    "[acoustic analysis of the caller audio (automatic): "
    "emotion={emotion} (p={emotion_p}; unknown p={unknown_p}); "
    "emotion2vec: {e2v}; "
    "audio events: {events}; "
    "speech rate: {rate}; "
    "pitch: {pitch}; "
    "voiced frames: {voiced}; "
    "loudness: {loudness}]"
)
USER_TEMPLATE: Final = "{transcript}\n\n{block}"


class AcousticTagger(Protocol):
    name: str

    def tag(self, wav_bytes: bytes, transcript: str) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- formatting


def _p(x: Any) -> str:
    return "n/a" if x is None else f"{float(x):.2f}"


def _band(value: float | None, lo: float, hi: float, words: tuple[str, str, str]) -> str:
    if value is None:
        return "n/a"
    return words[0] if value < lo else words[2] if value > hi else words[1]


def format_block(tags: dict[str, Any]) -> str:
    """Render tags through the fixed template. Pure function of ``tags``."""
    sv = tags.get("sensevoice") or {}
    dist = sv.get("emotion_probs") or {}
    e2v = tags.get("emotion2vec") or {}
    e2v_probs = e2v.get("probs") or {}
    top3 = sorted(e2v_probs.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    e2v_txt = ", ".join(f"{k} {v:.2f}" for k, v in top3) if top3 else "n/a"
    events = sv.get("events") or []
    pros = tags.get("prosody") or {}
    wps = pros.get("speech_rate_wps")
    rate_band = _band(wps, RATE_SLOW_WPS, RATE_FAST_WPS, ("slow", "moderate", "fast"))
    rate = "n/a" if wps is None else f"{wps:.1f} words/s ({rate_band})"
    f0 = pros.get("f0_median_hz")
    rng = pros.get("f0_range_st")
    pitch = (
        "n/a"
        if f0 is None or rng is None
        else f"median {f0:.0f} Hz, range {rng:.1f} semitones "
        f"({_band(rng, RANGE_NARROW_ST, RANGE_WIDE_ST, ('narrow', 'moderate', 'wide'))})"
    )
    vf = pros.get("voiced_fraction")
    db = pros.get("loudness_dbfs")
    return BLOCK_TEMPLATE.format(
        emotion=sv.get("emotion_forced") or "n/a",
        emotion_p=_p(dist.get(sv.get("emotion_forced"))) if sv.get("emotion_forced") else "n/a",
        unknown_p=_p(dist.get("unknown")),
        e2v=e2v_txt,
        events=", ".join(events) if events else "none detected",
        rate=rate,
        pitch=pitch,
        voiced="n/a" if vf is None else f"{round(100 * vf)}%",
        loudness="n/a" if db is None else f"{db:.0f} dBFS",
    )


def compose_user_message(transcript: str, block: str) -> str:
    return USER_TEMPLATE.format(transcript=transcript, block=block)


# --------------------------------------------------------------------------- audio


def load_16k_float(wav_bytes: bytes) -> Any:
    """16 kHz mono float32 numpy array in [-1, 1] (via the SER gate's resampler)."""
    import numpy as np

    from voxparity.providers.ser import resample_to_16k_mono

    b, _ = resample_to_16k_mono(wav_bytes)
    with contextlib.closing(wave.open(io.BytesIO(b), "rb")) as w:
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def prosody(audio: Any, transcript: str, sr: int = 16_000) -> dict[str, Any]:
    """Deterministic prosody stats. Returns None fields where undefined."""
    import numpy as np

    frame, hop = 400, 160  # 25 ms / 10 ms
    out: dict[str, Any] = {
        "speech_rate_wps": None,
        "f0_median_hz": None,
        "f0_range_st": None,
        "voiced_fraction": None,
        "loudness_dbfs": None,
        "active_span_s": None,
        "words": len(transcript.split()),
    }
    if audio is None or len(audio) < frame:
        return out
    n = 1 + (len(audio) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(audio[idx] ** 2, axis=1) + 1e-12)
    ref = float(np.percentile(rms, 95))
    if ref <= 1e-5:
        return out
    active = rms > ref * 10 ** (-30 / 20)
    if not active.any():
        return out
    first, last = int(np.argmax(active)), int(n - 1 - np.argmax(active[::-1]))
    span_s = (last - first + 1) * hop / sr + (frame - hop) / sr
    out["active_span_s"] = round(span_s, 3)
    if out["words"] and span_s > 0:
        out["speech_rate_wps"] = round(out["words"] / span_s, 3)
    out["loudness_dbfs"] = round(20 * math.log10(float(np.sqrt(np.mean(rms[active] ** 2)))), 2)

    import librosa

    f0, voiced, _ = librosa.pyin(
        audio.astype(np.float64),
        fmin=60.0,
        fmax=500.0,
        sr=sr,
        frame_length=1024,
        hop_length=hop,
        center=True,
    )
    m = min(len(f0), n)
    act = active[:m]
    v = voiced[:m] & act & np.isfinite(f0[:m])
    if act.sum():
        out["voiced_fraction"] = round(float(v.sum() / act.sum()), 4)
    # Pitch over a mostly unvoiced clip (whisper) is octave noise: require at
    # least MIN_VOICED of active frames voiced before reporting it.
    if v.sum() >= 5 and act.sum() and v.sum() / act.sum() >= MIN_VOICED:
        vals = f0[:m][v]
        out["f0_median_hz"] = round(float(np.median(vals)), 2)
        p10, p90 = np.percentile(vals, [10, 90])
        out["f0_range_st"] = round(float(12 * math.log2(p90 / p10)), 3)
    return out


# --------------------------------------------------------------------------- local models


def sensevoice_from_posteriors(emo_logp: dict[str, float], ev_logp: dict[str, float]) -> dict:
    """Slot posteriors (log-probs) -> labels. Separated out for tests."""
    emo = {k: math.exp(v) for k, v in emo_logp.items()}
    ev = {k: math.exp(v) for k, v in ev_logp.items()}
    unforced = max(emo, key=lambda k: (emo[k], k))
    forced = max((k for k in emo if k != "unknown"), key=lambda k: (emo[k], k))
    events = sorted(
        (k for k, p in ev.items() if k not in ("speech", "unknown_event") and p >= EVENT_MIN_P),
        key=lambda k: (-ev[k], k),
    )
    return {
        "emotion_unforced": unforced,
        "emotion_forced": forced,
        "emotion_probs": {k: round(p, 4) for k, p in emo.items()},
        "event_top": max(ev, key=lambda k: (ev[k], k)),
        "event_probs": {k: round(p, 4) for k, p in ev.items()},
        "events": events,
    }


class LocalAcousticTagger:
    """SenseVoiceSmall + emotion2vec+ large + prosody, CPU, deterministic."""

    def __init__(self, *, use_emotion2vec: bool = True) -> None:
        import logging

        import torch

        logging.getLogger("funasr").setLevel(logging.ERROR)
        torch.manual_seed(0)
        from funasr import AutoModel  # type: ignore[import-untyped,import-not-found,unused-ignore]

        sv_dir = os.environ.get("VOXPARITY_SENSEVOICE_DIR") or (
            str(DEFAULT_SENSEVOICE_DIR) if DEFAULT_SENSEVOICE_DIR.exists() else SENSEVOICE_ID
        )
        if not Path(sv_dir).exists():
            alt = Path.home() / "Developer" / "voxparity" / "models" / "sensevoice-small"
            sv_dir = str(alt) if alt.exists() else SENSEVOICE_ID
        kw: dict[str, Any] = {"disable_update": True, "device": "cpu", "disable_pbar": True}
        if sv_dir == SENSEVOICE_ID:
            kw["hub"] = "hf"
        # FunASR's WavFrontend defaults to dither=1.0 (random noise added to the
        # fbank input): the emotion posteriors then move by ~0.02 run to run.
        # The frontend is rebuilt from frontend_conf per call, so it must be set
        # here, not on the instance.
        self._sv = AutoModel(model=sv_dir, frontend_conf={"dither": 0.0}, **kw)
        self._captured: dict[str, Any] = {}
        inner = self._sv.model
        orig = inner.ctc.log_softmax

        def capture(x: Any) -> Any:
            y = orig(x)
            self._captured["logp"] = y.detach().clone()
            return y

        inner.ctc.log_softmax = capture
        self._e2v = None
        if use_emotion2vec:
            from voxparity.providers.ser import SERGate

            self._e2v = SERGate()
        self.name = (
            f"{TAGGER_VERSION}:sensevoice-small+"
            + ("emotion2vec_plus_large+" if self._e2v else "")
            + "pyin"
        )

    def _sensevoice(self, audio: Any) -> dict[str, Any]:
        import torch

        self._captured.clear()
        self._sv.generate(
            input=torch.from_numpy(audio), language="en", use_itn=True, cache={}, disable_pbar=True
        )
        logp = self._captured["logp"][0]
        emo = {k: float(logp[EMOTION_SLOT, i]) for k, i in SV_EMOTION_TOKENS.items()}
        ev = {k: float(logp[EVENT_SLOT, i]) for k, i in SV_EVENT_TOKENS.items()}
        return sensevoice_from_posteriors(emo, ev)

    def tag(self, wav_bytes: bytes, transcript: str) -> dict[str, Any]:
        audio = load_16k_float(wav_bytes)
        tags: dict[str, Any] = {
            "tagger": self.name,
            "wav_sha256": hashlib.sha256(wav_bytes).hexdigest(),
            "sensevoice": self._sensevoice(audio),
        }
        if self._e2v is not None:
            dist = self._e2v.classify(wav_bytes)
            tags["emotion2vec"] = {
                "probs": {k: round(v, 4) for k, v in dist.items()},
                "top": max(dist, key=lambda k: (dist[k], k)),
            }
        tags["prosody"] = prosody(audio, transcript)
        return tags
