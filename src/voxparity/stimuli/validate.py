"""Validation gates for generated stimuli (blueprint §9, commandment 7).

Gate 1 — ASR round-trip: an independent pass transcribes the audio; word error
rate against the intended transcript must not exceed ``WER_GATE`` (content
fidelity — the Big Bench Audio pattern). The transcriber is injected; the CLI
prefers Deepgram Nova-3 (shares nothing with any stimulus TTS engine) and falls
back to Gemini only when no Deepgram key is present.

Gate 2 — intended-cue check: an audio judge answers the item's own perception
probe; the answer must equal the probe gold for this variant. This machine gate
is a *pre-filter* — the ≥3 independent human validators (spec §12) remain
mandatory before release; passing here never substitutes for them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from voxparity.providers.gemini import GeminiClient, GeminiError
from voxparity.schemas.item import DeliveryVariant, Item, SceneKind

WER_GATE = 0.15


# Spelling variants that ASR engines emit interchangeably; never semantic changes.
_CANON = {
    "alright": "all right",
    "ok": "okay",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "yeah": "yes",
    "yep": "yes",
    "mm-hmm": "yes",
    "uh-huh": "yes",
}


def _words(s: str) -> list[str]:
    tokens = re.sub(r"[^\w\s'-]", " ", s.lower()).split()
    out: list[str] = []
    for t in tokens:
        # _CANON first (it owns hyphenated forms like "mm-hmm"); THEN split
        # remaining hyphens — "thirty-nine" and "thirty nine" are the same
        # utterance and ASR engines emit either form (faafd live-hit, Sep 13)
        canon = _CANON.get(t, t)
        for part in canon.split():
            out.extend(part.split("-"))
    return out


# Closed filler lexicon for disfluent-transcript items (D083). Deliberately
# tiny: only pure hesitation vocalizations that carry no propositional content.
# "like"/"well"/"so" are excluded — semantic in some contexts. Note _CANON runs
# first, so affirmative "mm-hmm"/"uh-huh" become "yes" and are never stripped.
_FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "hmm", "mm", "ah"}


def _strip_disfluencies(tokens: list[str]) -> list[str]:
    """Drop fillers and collapse immediate word repeats ("I I think" -> "I think")."""
    out: list[str] = []
    for t in tokens:
        if t in _FILLERS:
            continue
        if out and out[-1] == t:
            continue
        out.append(t)
    return out


def has_disfluencies(reference: str) -> bool:
    """True when the transcript itself is written disfluent — a filler token or
    an immediate self-repair repeat. Gates whether the tolerant WER applies."""
    tokens = _words(reference)
    return tokens != _strip_disfluencies(tokens)


def disfluent_wer(reference: str, hypothesis: str) -> float:
    """WER with disfluencies normalized out of BOTH sides (D083).

    Whisper-family ASR deletes fillers as an intended transcription style
    (CrisperWhisper, arXiv 2408.16589), so a transcript that WRITES "Um... I—
    I think so" can never round-trip verbatim through the independent gate —
    the hesitations are the stimulus, not an error. Stripping the closed filler
    lexicon and collapsing immediate repeats on both reference and hypothesis
    scores content-word fidelity in full while exempting exactly the tokens the
    manipulation places. Content errors still veto: "cancel" for "send" is a
    substitution after normalization like before it.
    """
    ref = _strip_disfluencies(_words(reference))
    hyp = _strip_disfluencies(_words(hypothesis))
    return wer(" ".join(ref), " ".join(hyp))


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein distance over words."""
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        raise ValueError("empty reference")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def slot_gap_wer(reference: str, slot: str, hypothesis: str, max_gap: int = 6) -> float:
    """WER for a slot-masked clip: the reference's slot is a wildcard gap.

    Whatever the ASR emitted where the mask sits is neither right nor wrong —
    live-observed: Deepgram produced a confident "three fifty eight" under a
    -12 dB squelch over "three five zero" (the Careless-Whisper failure mode,
    and the phenomenon the slot-noise axis exists to measure). Scoring those
    hallucinated tokens as insertions would flunk the manipulation itself, so
    the hypothesis is split at every point into (pre, gap<=max_gap, post) and
    the best pre/post alignment against the reference around the slot wins.
    Errors OUTSIDE the masked window still count in full.
    """
    if slot not in reference:
        raise ValueError(f"slot {slot!r} not in reference")
    before, _, after = reference.partition(slot)
    before, after = " ".join(before.split()), " ".join(after.split())
    hyp = _words(hypothesis)
    n_ref = len(_words(before)) + len(_words(after))
    if n_ref == 0:
        raise ValueError("reference is empty outside the slot")
    best = 1.0
    for i in range(len(hyp) + 1):
        for j in range(i, min(i + max_gap, len(hyp)) + 1):
            pre, post = " ".join(hyp[:i]), " ".join(hyp[j:])
            errors = 0.0
            for ref_part, hyp_part in ((before, pre), (after, post)):
                # word-level emptiness, not string emptiness: a slot that ends
                # at the final punctuation leaves after == "." — truthy, but
                # zero words, and wer() on it raises (live-hit: wirerb suffix
                # slot crashed the gate sweep, Sep 6)
                if _words(ref_part):
                    errors += wer(ref_part, hyp_part) * len(_words(ref_part))
                else:
                    errors += len(_words(hyp_part))
            best = min(best, errors / n_ref)
    return best


@dataclass(frozen=True)
class GateResult:
    passed: bool
    detail: dict[str, object]


PREMIX_POLICY = (
    "caller voice gated BEFORE the mix: the bed carries speech by design, and ASR "
    "transcribing it is the manipulation, not a content error (bank-finish, Sep 14)"
)


def speech_bearing_bed(variant: DeliveryVariant) -> bool:
    """A background scene whose bed is rendered SPEECH (``tts:`` assets: a second
    speaker, a TV advert). Only these gate ASR on the caller's pre-mix audio;
    every other variant keeps the mix-level gate unchanged."""
    scene = variant.scene
    return (
        scene is not None and scene.kind == SceneKind.BACKGROUND and scene.asset.startswith("tts:")
    )


def caller_asr_gate(
    transcribe: Callable[[bytes], str],
    item: Item,
    variant: DeliveryVariant,
    final_wav: bytes,
    primary_wav: bytes | None = None,
) -> GateResult:
    """ASR round-trip under the speech-bed policy.

    Speech-bearing beds with the pre-mix render available: gate the caller's
    words on ``primary_wav`` and record the policy plus the primary's hash (the
    mix recipe's ``primary_sha256`` ties it to the final clip). Everything else,
    including every non-scene variant, is exactly ``asr_roundtrip_gate`` on the
    final clip (with the slot wildcard for slot-masked variants).
    """
    import hashlib

    if speech_bearing_bed(variant) and primary_wav is not None:
        r = asr_roundtrip_gate(transcribe, item, primary_wav)
        return GateResult(
            r.passed,
            {
                **r.detail,
                "gated_on": "primary pre-mix render",
                "policy": PREMIX_POLICY,
                "primary_sha256": hashlib.sha256(primary_wav).hexdigest(),
            },
        )
    slot = variant.scene.slot if variant.scene is not None else None
    return asr_roundtrip_gate(transcribe, item, final_wav, slot)


def asr_roundtrip_gate(
    transcribe: Callable[[bytes], str],
    item: Item,
    wav_bytes: bytes,
    slot: str | None = None,
) -> GateResult:
    """``slot`` marks a masked span (D065): the gate then scores the clip with
    slot_gap_wer — full transcript, but the masked window is a wildcard, so ASR
    hallucinations inside it are neither right nor wrong while every error
    outside it still counts."""
    transcript = transcribe(wav_bytes)
    detail: dict[str, object] = {}
    if slot is not None:
        rate = slot_gap_wer(item.transcript, slot, transcript)
    elif has_disfluencies(item.transcript):
        # The transcript is written disfluent BY DESIGN (D083): the tolerant
        # WER is the gate, with the raw rate recorded for audit.
        rate = disfluent_wer(item.transcript, transcript)
        detail["wer_raw"] = round(wer(item.transcript, transcript), 4)
        detail["disfluency_normalized"] = True
    else:
        rate = wer(item.transcript, transcript)
    detail.update({"wer": round(rate, 4), "asr_transcript": transcript})
    return GateResult(rate <= WER_GATE, detail)


CUE_JUDGE_PREFIX = (
    "Judge ONLY the vocal delivery — tone, pace, energy, prosody. Deliberately "
    "IGNORE what the words mean; the same sentence can be spoken many ways, and "
    "the wording must not influence your answer. "
)


def cue_judge_question(probe_question: str) -> str:
    """The cue judge's question: the delivery-only instruction plus the item's
    own probe. One definition, so a cross-judge asks exactly what the gate did."""
    return f"{CUE_JUDGE_PREFIX}{probe_question}"


def cue_check_gate(
    client: GeminiClient, item: Item, variant_id: str, wav_bytes: bytes
) -> GateResult:
    probe = item.perception_probe
    gold = probe.gold_by_variant.get(variant_id)
    if gold is None:
        return GateResult(
            False, {"error": f"no probe gold for variant {variant_id}", "verdict": None}
        )
    question = cue_judge_question(probe.question)
    try:
        answer = client.classify(wav_bytes, question, probe.options)
    except GeminiError as e:
        # A judge that could not answer has NOT rejected the clip. Recording an
        # outage as `passed: false` is how 173 of 198 Cartesia clips came to look
        # like engine failures when they were our own quota exhaustion — and that
        # phantom produced the "1/99 pair discrimination" that retired the engine
        # (D046). `verdict: None` marks the cell as unmeasured so every consumer
        # can exclude it from a denominator instead of counting it against the
        # engine.
        return GateResult(False, {"error": str(e), "verdict": None})
    from voxparity.providers.gemini import TEXT_MODEL

    return GateResult(
        answer == gold,
        {"judge_answer": answer, "gold": gold, "judge_model": TEXT_MODEL, "verdict": answer},
    )
