"""Local, vendor-independent speech-emotion gate (emotion2vec+ large via FunASR).

WHY THIS EXISTS
---------------
The current intended-cue gate is ``gemini-3.6-flash`` answering the item's own
perception probe (``stimuli/validate.py::cue_check_gate``). Two problems, one
measured and one structural:

* Measured — against a human rater on 64 clips the judge emitted "neutral" 32
  times for 20 neutral stimuli and "urgent" exactly once in 64 clips; Wagner
  unbiased hit rate on urgent was 0.08 where the human scored 0.75. A judge that
  almost never uses a label cannot certify that label.
* Structural — that judge is the same vendor as our best-performing stimulus
  engine (Gemini TTS) and the same family as a model under test. A gate that
  shares a vendor with the renderer cannot rule out that it is scoring its own
  house style rather than the delivery.

This module is the independent second opinion: a self-supervised SER model that
runs locally, has never seen our probe text, and shares nothing with any TTS
engine or any model under test. It is a *pre-filter and a cross-check*, not a
replacement for the >=3 human validators required by spec section 12.

MODEL
-----
``iic/emotion2vec_plus_large`` driven through ``funasr.AutoModel``. Its label
inventory is fixed at nine classes and is NOT our enum; see ``EMOTION_MAP`` for
the deliberately lossy projection onto VoxParity's eight-value ``Emotion``.

LICENCE — PX-004 (BLOCKING)
---------------------------
The FunASR *code* is Apache/MIT, but the emotion2vec+ *weights* ship under the
FunASR model licence, which is not CC BY 4.0 and is not automatically
redistributable or usable for arbitrary downstream publication. Gate output
derived from those weights therefore MUST NOT be published inside the CC BY 4.0
item/stimulus dataset until PX-004 is resolved (either a licence review that
clears derived scores as non-substantial, or a written grant). Until then: run
the gate locally, keep its detail dicts in the private manifest, and strip them
from any public export.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import tempfile
import warnings
import wave
from array import array
from dataclasses import dataclass
from typing import Any, Final, Literal

MODEL_ID: Final = "iic/emotion2vec_plus_large"

# emotion2vec+ was trained on 16 kHz speech; FunASR's frontend assumes fs=16000
# and does not resample for us, so every clip is normalised before it is handed
# over (see resample_to_16k_mono).
TARGET_SAMPLE_RATE: Final = 16_000

# Provisional. The threshold is NOT yet calibrated against the 64-clip
# human-rated set that condemned the Gemini gate; calibrate before this gate is
# allowed to reject anything, and record the chosen value per intended emotion.
DEFAULT_THRESHOLD: Final = 0.5

# The model's own nine classes, canonicalised to English. FunASR returns them as
# bilingual tokens ("生气/angry", "开心/happy", ...) plus the literal "<unk>"
# token for the ninth class; _canonical_label normalises both forms.
MODEL_LABELS: Final[tuple[str, ...]] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

VOXPARITY_EMOTIONS: Final[tuple[str, ...]] = (
    "neutral",
    "happy",
    "frustrated",
    "angry",
    "sad",
    "anxious",
    "resigned",
    "urgent",
)

Fidelity = Literal["direct", "approximate", "conflated"]


class SERError(RuntimeError):
    """Base class for gate failures that are ours, not the caller's."""


class SERDependencyError(SERError):
    """funasr/torch are not importable. Raised at construction, never at import."""


class SERAudioError(SERError, ValueError):
    """The WAV bytes could not be decoded into 16 kHz mono PCM."""


@dataclass(frozen=True)
class EmotionMapping:
    """One VoxParity emotion expressed in terms of the model's own classes.

    ``sources`` are summed, so mappings that share a source (frustrated/angry,
    resigned/sad) are explicitly non-exclusive: the projection is a mass
    attribution, not a partition. ``fidelity`` is what a reviewer should read
    before trusting a pass.
    """

    sources: tuple[str, ...]
    fidelity: Fidelity
    note: str


# The projection is lossy BY DESIGN and every entry says how. Four of our eight
# labels have no counterpart in a nine-way categorical taxonomy trained on
# acted-emotion corpora, and pretending otherwise is exactly the failure we are
# replacing.
#
# Emotions this model genuinely CANNOT express:
#   frustrated — folded into anger/disgust by every categorical SER corpus;
#                shares its mass with `angry`, so the two can never be separated.
#   resigned   — defeated low-arousal negative affect; corpora label it `sad`,
#                so it shares its mass with `sad` and cannot be separated.
#   urgent     — an AROUSAL construct, not a categorical emotion. There is no
#                counterpart at all; the entry below is a high-arousal proxy and
#                is the weakest row in this table. This is also the exact label
#                where the Gemini judge collapsed (1 use in 64 clips), so a weak
#                proxy still improves on the incumbent — but it must be reported
#                as a proxy, never as a measurement.
#   anxious    — approximated by `fearful`, the nearest true neighbour (anxiety
#                as sustained low-intensity fear). Usable as a presence check,
#                not as a fear-vs-anxiety discriminator.
EMOTION_MAP: Final[dict[str, EmotionMapping]] = {
    "neutral": EmotionMapping(("neutral",), "direct", "Native class."),
    "happy": EmotionMapping(("happy",), "direct", "Native class."),
    "angry": EmotionMapping(("angry",), "direct", "Native class; shares mass with frustrated."),
    "sad": EmotionMapping(("sad",), "direct", "Native class; shares mass with resigned."),
    "frustrated": EmotionMapping(
        ("angry", "disgusted"),
        "conflated",
        "No native class; folded into anger/disgust. Cannot be separated from angry.",
    ),
    "anxious": EmotionMapping(
        ("fearful",),
        "approximate",
        "Nearest neighbour only: anxiety as sustained low-intensity fear.",
    ),
    "resigned": EmotionMapping(
        ("sad",),
        "conflated",
        "No native class; defeated low-arousal affect is labelled sad. Inseparable from sad.",
    ),
    "urgent": EmotionMapping(
        ("fearful", "surprised", "angry"),
        "conflated",
        "No native class: urgency is arousal, not category. High-arousal proxy only.",
    ),
}

# Probability mass this projection deliberately discards. "other"/"unknown" are
# the model's own escape hatches and carry no VoxParity meaning; surfacing the
# residual in the gate detail keeps a silently-uncertain clip visible to review.
UNMAPPED_LABELS: Final[tuple[str, ...]] = tuple(
    label for label in MODEL_LABELS if not any(label in m.sources for m in EMOTION_MAP.values())
)


def _canonical_label(token: str) -> str:
    """Normalise a FunASR emotion token to its English class name.

    Tokens arrive bilingual and the ordering has changed between model releases
    ("生气/angry" in tokens.txt, "angry/生气" in the README), so pick whichever
    side is a known English class rather than trusting the position. Unknown
    tokens are returned lowercased and end up in the unmapped residual instead
    of being silently dropped.
    """
    raw = token.strip()
    if raw in {"<unk>", "unk"}:
        return "unknown"
    for part in raw.split("/"):
        candidate = part.strip().lower()
        if candidate in MODEL_LABELS:
            return candidate
    return raw.lower()


# --------------------------------------------------------------------------
# Audio normalisation
# --------------------------------------------------------------------------
# audioop was REMOVED from the stdlib in Python 3.13 (PEP 594) — not merely
# deprecated — and the repo already runs 3.14, so it is absent by default. It
# comes back only if `audioop-lts` is installed. Its ratecv() carries proper
# filter state across the conversion, so use it when present and fall back to a
# box-prefilter + linear interpolation otherwise.
try:  # pragma: no cover - presence depends on the interpreter/env, not on us
    with warnings.catch_warnings():
        # On 3.11/3.12 the import itself warns about the 3.13 removal; that is
        # news to nobody here and must not pollute the repo's test output.
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop as _audioop
except ImportError:  # pragma: no cover
    _audioop = None  # type: ignore[assignment]

HAVE_AUDIOOP: Final = _audioop is not None


def _pcm_to_int16(frames: bytes, sampwidth: int) -> array[int]:
    """Widen/narrow raw PCM frames to a signed 16-bit sample array."""
    if sampwidth == 2:
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder == "big":
            # WAV payloads are little-endian; array() uses native order.
            samples.byteswap()
        return samples
    if sampwidth == 1:
        # 8-bit WAV is unsigned by definition of the format.
        return array("h", ((b - 128) << 8 for b in frames))
    if sampwidth == 3:
        out = array("h")
        for i in range(0, len(frames) - 2, 3):
            # Keep the top 16 bits; sign lives in the high byte.
            value = frames[i + 1] | (frames[i + 2] << 8)
            out.append(value - 65536 if value >= 32768 else value)
        return out
    if sampwidth == 4:
        out = array("h")
        for i in range(0, len(frames) - 3, 4):
            value = int.from_bytes(frames[i : i + 4], "little", signed=True)
            out.append(value >> 16)
        return out
    raise SERAudioError(f"unsupported WAV sample width: {sampwidth} bytes")


def _downmix(samples: array[int], channels: int) -> array[int]:
    if channels == 1:
        return samples
    if channels < 1:
        raise SERAudioError(f"invalid channel count: {channels}")
    n = len(samples) // channels
    out = array("h", bytes(2 * n))
    for i in range(n):
        base = i * channels
        out[i] = sum(samples[base : base + channels]) // channels
    return out


def _resample(samples: array[int], src_sr: int, dst_sr: int) -> array[int]:
    if src_sr == dst_sr or not samples:
        return samples

    ratio = src_sr / dst_sr
    if ratio > 1:
        # Crude anti-alias before decimation: without it, 24k/44.1k TTS output
        # folds everything above 8 kHz back into the band the SER frontend
        # actually looks at. A box filter is not a good lowpass, but it is cheap
        # in pure Python and removes the worst of the fold-back. This path is
        # fine for feeding a 16 kHz model; it is NOT fit for publishing audio.
        width = max(1, int(ratio))
        if width > 1:
            filtered = array("h", bytes(2 * len(samples)))
            acc = 0
            for i, s in enumerate(samples):
                acc += s
                if i >= width:
                    acc -= samples[i - width]
                filtered[i] = acc // min(i + 1, width)
            samples = filtered

    n_out = max(1, int(len(samples) / ratio))
    out = array("h", bytes(2 * n_out))
    last = len(samples) - 1
    for i in range(n_out):
        pos = i * ratio
        j = int(pos)
        if j >= last:
            out[i] = samples[last]
            continue
        frac = pos - j
        a = samples[j]
        b = samples[j + 1]
        out[i] = int(a + (b - a) * frac)
    return out


def resample_to_16k_mono(wav_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    """Return (16 kHz mono 16-bit WAV bytes, JSON-safe info about the input).

    Defensive by design: stimuli reach us at 22.05/24/44.1 kHz depending on the
    engine, and a mismatched rate degrades emotion2vec silently rather than
    loudly, which is precisely the class of failure this gate exists to catch.
    """
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            src_sr = w.getframerate()
            frames = w.readframes(w.getnframes())
    except wave.Error as e:
        raise SERAudioError(f"not a decodable PCM WAV: {e}") from e

    if not frames:
        raise SERAudioError("WAV contains no audio frames")

    samples = _downmix(_pcm_to_int16(frames, sampwidth), channels)

    if HAVE_AUDIOOP and src_sr != TARGET_SAMPLE_RATE:
        converted, _ = _audioop.ratecv(samples.tobytes(), 2, 1, src_sr, TARGET_SAMPLE_RATE, None)
        resampled = array("h")
        resampled.frombytes(converted)
    else:
        resampled = _resample(samples, src_sr, TARGET_SAMPLE_RATE)

    payload = resampled
    if sys.byteorder == "big":
        payload = array("h", resampled)
        payload.byteswap()

    buf = io.BytesIO()
    with contextlib.closing(wave.open(buf, "wb")) as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(TARGET_SAMPLE_RATE)
        out.writeframes(payload.tobytes())

    info: dict[str, Any] = {
        "input_sample_rate": src_sr,
        "input_channels": channels,
        "input_sample_width_bytes": sampwidth,
        "resampled": src_sr != TARGET_SAMPLE_RATE,
        "resampler": "audioop.ratecv" if HAVE_AUDIOOP else "pure-python-linear",
        "duration_s": round(len(resampled) / TARGET_SAMPLE_RATE, 4),
    }
    return buf.getvalue(), info


# --------------------------------------------------------------------------
# Projection helpers
# --------------------------------------------------------------------------
def to_voxparity(dist: dict[str, float]) -> dict[str, float]:
    """Project a model distribution onto VoxParity's eight-value enum.

    The result is a mass attribution, NOT a probability distribution: sources
    are shared (angry feeds both `angry` and `frustrated`) so the values can sum
    to more than 1, and the "other"/"unknown" residual is dropped so they can
    also sum to less. Read ``EMOTION_MAP[...].fidelity`` before comparing two of
    these numbers against each other.
    """
    canonical: dict[str, float] = {}
    for label, p in dist.items():
        key = _canonical_label(label)
        canonical[key] = canonical.get(key, 0.0) + float(p)
    return {
        emotion: round(sum(canonical.get(src, 0.0) for src in mapping.sources), 6)
        for emotion, mapping in EMOTION_MAP.items()
    }


def _normalise(labels: list[Any], scores: list[Any]) -> dict[str, float]:
    """Canonicalise FunASR's label/score pair into a probability distribution.

    FunASR softmaxes internally but then drops "unuse*" tokens from the returned
    vector, so the surviving scores need not sum to 1 — renormalise rather than
    trust them.
    """
    if len(labels) != len(scores):
        raise SERError(f"funasr returned {len(labels)} labels for {len(scores)} scores")
    if not labels:
        raise SERError("funasr returned an empty label set")

    dist: dict[str, float] = {}
    for label, score in zip(labels, scores, strict=True):
        key = _canonical_label(str(label))
        dist[key] = dist.get(key, 0.0) + float(score)

    total = sum(dist.values())
    if total <= 0 or not math.isfinite(total):
        raise SERError(f"funasr returned a degenerate score vector (sum={total})")
    return {k: round(v / total, 6) for k, v in dist.items()}


def unmapped_mass(dist: dict[str, float]) -> float:
    """Probability the projection throws away (model `other`/`unknown` + strays)."""
    used = {src for m in EMOTION_MAP.values() for src in m.sources}
    return round(
        sum(float(p) for label, p in dist.items() if _canonical_label(label) not in used), 6
    )


class SERGate:
    """emotion2vec+ large as a local cue gate.

    Constructing this loads torch and downloads weights on first use; importing
    the module does neither.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        hub: str = "hf",
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.hub = hub
        auto_model = _import_auto_model()
        kwargs: dict[str, Any] = {
            "model": model_id,
            "hub": hub,
            # Skips FunASR's startup version check: a gate must not depend on a
            # network call to a vendor endpoint to run.
            "disable_update": True,
        }
        if device is not None:
            kwargs["device"] = device
        kwargs.update(model_kwargs or {})
        try:
            self._model = auto_model(**kwargs)
        # funasr/torch raise a wide variety of bare Exception subclasses on a bad
        # model id, a missing hub, or an OOM; all of them mean the same thing here.
        except Exception as e:
            raise SERError(f"could not load {model_id!r} via funasr: {e}") from e

    def classify(self, wav_bytes: bytes) -> dict[str, float]:
        """Return a normalised distribution over the model's own nine labels."""
        prepared, _ = resample_to_16k_mono(wav_bytes)
        return _normalise(*self._generate(prepared))

    def gate(
        self,
        wav_bytes: bytes,
        intended_emotion: str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[bool, dict[str, Any]]:
        """Judge one clip against its intended cue.

        Returns ``(passed, detail)``. ``detail`` is JSON-serialisable primitives
        only (str/float/int/bool/None and dicts thereof) because it is written
        verbatim into the stimulus manifest — subject to PX-004, which blocks
        publishing it in the CC BY 4.0 dataset.
        """
        intended = str(intended_emotion).lower()
        if intended not in EMOTION_MAP:
            raise ValueError(
                f"unknown intended emotion {intended_emotion!r}; expected one of "
                f"{', '.join(VOXPARITY_EMOTIONS)}"
            )

        prepared, audio_info = resample_to_16k_mono(wav_bytes)
        dist = _normalise(*self._generate(prepared))

        projected = to_voxparity(dist)
        mapping = EMOTION_MAP[intended]
        intended_p = projected[intended]
        top_label = max(dist, key=lambda k: dist[k])
        top_emotion = max(projected, key=lambda k: projected[k])
        passed = intended_p >= threshold

        # A pass on a conflated label says "the right family of affect is
        # present", never "the intended cue was rendered". Reviewers and the
        # human validators need that distinction in writing, per section 12.
        advisory = None
        if mapping.fidelity != "direct":
            advisory = (
                f"{intended!r} has no native class in this model; "
                f"result is {mapping.fidelity}: {mapping.note}"
            )

        detail: dict[str, Any] = {
            "gate": "ser_local",
            "model": self.model_id,
            "hub": self.hub,
            "intended": intended,
            "intended_probability": intended_p,
            "threshold": float(threshold),
            "passed": passed,
            "top_label": top_label,
            "top_probability": dist[top_label],
            "top_voxparity_emotion": top_emotion,
            "distribution": dist,
            "voxparity_mass": projected,
            "unmapped_mass": unmapped_mass(dist),
            "mapping_fidelity": mapping.fidelity,
            "mapping_sources": list(mapping.sources),
            "mapping_note": mapping.note,
            "advisory": advisory,
            "audio": audio_info,
            "license_flag": "PX-004",
        }
        return passed, detail

    def _generate(self, wav_16k_bytes: bytes) -> tuple[list[Any], list[Any]]:
        """Run FunASR on a temp file and pull out (labels, scores).

        FunASR's documented entry point takes a path; handing it a path also
        keeps us off its undocumented in-memory branches, which have changed
        shape between releases.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_16k_bytes)
            path = tmp.name
        try:
            res = self._model.generate(
                path,
                granularity="utterance",
                extract_embedding=False,
            )
        except SERError:
            raise
        except Exception as e:
            raise SERError(f"funasr inference failed: {e}") from e
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

        if not res:
            raise SERError("funasr returned no results")
        first = res[0]
        try:
            return list(first["labels"]), list(first["scores"])
        except (KeyError, TypeError) as e:
            raise SERError(f"unexpected funasr result shape: {first!r}") from e


def _import_auto_model() -> Any:
    """Import funasr.AutoModel, or explain exactly how to install it.

    Deliberately called from SERGate.__init__ and nowhere at module scope: the
    CLI imports every gate module eagerly, and a heavyweight torch import (or a
    hard failure on a machine that will never run this gate) is not acceptable
    just to look up EMOTION_MAP.
    """
    try:
        from funasr import AutoModel  # type: ignore[import-not-found,unused-ignore]
    except ImportError as e:
        raise SERDependencyError(
            "the local SER gate needs FunASR and PyTorch, which are not installed.\n"
            '  uv pip install "funasr>=1.2.0" torch torchaudio\n'
            "  (or: pip install 'funasr>=1.2.0' torch torchaudio)\n"
            f"First construction then downloads {MODEL_ID} (~1.2 GB) from the "
            "configured hub. Weights are under the FunASR model licence — see "
            "PX-004 before publishing any output."
        ) from e
    return AutoModel
