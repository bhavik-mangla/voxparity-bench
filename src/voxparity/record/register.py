"""Register accepted self-recorded takes into the stimulus store (D096/D097).

`voxparity record register` turns the recording console's ledger
(``metadata.jsonl``) into ``engine=human`` manifest rows, gated by the D097
policy for human recordings. The policy exists because a single ASR is the
wrong instrument for a human take: 6 of 7 single-transcriber "fails" in the
first batch passed a second, independent transcriber.

Gate policy, in order, every step recorded on the ``asr_roundtrip`` row:

1. **Cross-transcriber best-of.** Deepgram first; only if it fails is Gemini
   asked. Either passing admits the take, and the row names which one.
2. **Hyp-side filler strip for impaired deliveries** (slurred / breathless /
   confused). A naturalistic "uh" inserted by a slurring speaker IS the
   manipulation, so filler tokens are removed from the HYPOTHESIS only (the
   reference is the script and is never edited); ``wer_raw`` stays on the row.
3. **Script-read acceptance** when BOTH transcribers degrade. The speaker read
   the exact transcript on the console, so content is certified by procedure,
   and both transcripts are kept as evidence. Accepted only when the two
   hypotheses DIVERGE from each other (as in the D097 brthls case: two
   different fabrications). If both ASRs agree on the same wrong words, that is
   evidence of a misread, and the take fails.

A transcriber outage is recorded as ``verdict: None`` (unmeasured), never as a
fail or as a script-read pass (D046).

Idempotent: the latest take per entry is considered, and it is skipped when its
sha256 is already registered (as the row's hash or, for scene variants whose
mixed clip hashes differently, as ``take_sha256``). A row that already carries a
``human_check`` is never replaced without ``force``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voxparity.schemas.item import DeliveryVariant, Emotion, Item, SceneKind
from voxparity.stimuli.store import StimulusRecord, StimulusStore
from voxparity.stimuli.validate import (
    _FILLERS,
    PREMIX_POLICY,
    WER_GATE,
    _words,
    disfluent_wer,
    has_disfluencies,
    slot_gap_wer,
    wer,
)

POLICY = "D097"
IMPAIRED_DELIVERIES: frozenset[Emotion] = frozenset(
    {Emotion.SLURRED, Emotion.BREATHLESS, Emotion.CONFUSED}
)
# Order is the policy: Deepgram (shares nothing with any renderer) first.
TRANSCRIBER_LABELS = {"deepgram": "deepgram-nova3", "gemini": "gemini"}

Transcriber = Callable[[bytes], str]


def strip_hyp_fillers(hypothesis: str) -> str:
    """Remove filler tokens from an ASR hypothesis (never from the reference)."""
    return " ".join(t for t in _words(hypothesis) if t not in _FILLERS)


def _rate(reference: str, hypothesis: str, slot: str | None) -> float:
    if slot is not None:
        return slot_gap_wer(reference, slot, hypothesis)
    if has_disfluencies(reference):
        return disfluent_wer(reference, hypothesis)
    return wer(reference, hypothesis)


def human_asr_gate(
    reference: str,
    wav: bytes,
    transcribers: dict[str, Transcriber],
    impaired: bool,
    slot: str | None = None,
) -> dict[str, Any]:
    """The D097 gate for one human take. Returns the ``asr_roundtrip`` detail."""
    transcripts: dict[str, str] = {}
    wers: dict[str, float] = {}
    outages: dict[str, str] = {}
    for name in ("deepgram", "gemini"):
        fn = transcribers.get(name)
        if fn is None:
            continue
        try:
            hyp = fn(wav)
        except Exception as e:  # an outage is not a verdict (D046)
            outages[name] = str(e)[:200]
            continue
        transcripts[name] = hyp
        raw = _rate(reference, hyp, slot)
        wers[name] = round(raw, 4)
        if raw <= WER_GATE:
            return {
                "passed": True,
                "wer": round(raw, 4),
                "transcriber": TRANSCRIBER_LABELS[name],
                "asr_transcript": hyp,
                "transcripts": transcripts,
                "wers": wers,
                "policy": POLICY,
                "note": "cross-transcriber best-of"
                + ("" if name == "deepgram" else f"; deepgram wer {wers.get('deepgram')}"),
            }
        if impaired:
            stripped = _rate(reference, strip_hyp_fillers(hyp), slot)
            if stripped <= WER_GATE:
                return {
                    "passed": True,
                    "wer": round(stripped, 4),
                    "wer_raw": round(raw, 4),
                    "transcriber": f"{TRANSCRIBER_LABELS[name]}+hyp-filler-strip",
                    "asr_transcript": hyp,
                    "transcripts": transcripts,
                    "wers": wers,
                    "policy": POLICY,
                    "note": "impaired delivery: filler tokens stripped from the hypothesis "
                    "only; they are the manipulation, not a content error",
                }

    base = {"transcripts": transcripts, "wers": wers, "policy": POLICY}
    if outages and len(transcripts) < 2:
        # One transcriber could not be asked: we have not measured the second
        # opinion the policy needs, so this is unmeasured, never a fail.
        return {
            **base,
            "passed": False,
            "verdict": None,
            "error": "; ".join(f"{k}: {v}" for k, v in outages.items()),
            "note": "transcriber unavailable; re-run register (D046)",
        }
    if len(transcripts) < 2:
        return {**base, "passed": False, "note": "only one transcriber configured"}

    dg, gm = transcripts["deepgram"], transcripts["gemini"]
    divergence = wer(dg, gm) if _words(dg) else 1.0
    if divergence > WER_GATE:
        return {
            **base,
            "passed": True,
            "wer": None,
            "transcriber": "dual-ASR-degraded",
            "asr_divergence": round(divergence, 4),
            "note": "script-read acceptance: the speaker read the exact transcript and BOTH "
            "transcribers degraded with divergent hypotheses (both kept as evidence); "
            "content certified by procedure (D097)",
        }
    return {
        **base,
        "passed": False,
        "asr_divergence": round(divergence, 4),
        "note": "both transcribers agree on words that differ from the script: likely a "
        "misread, not an ASR collapse; re-record",
    }


def latest_takes(metadata_path: Path) -> dict[str, dict[str, Any]]:
    """entry_id -> the metadata row of its highest take."""
    out: dict[str, dict[str, Any]] = {}
    for line in metadata_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prev = out.get(row["entry"])
        if prev is None or int(row["take"]) > int(prev["take"]):
            out[row["entry"]] = row
    return out


@dataclass
class RegisterOutcome:
    entry: str
    status: str  # registered | skipped | failed | unmeasured (not persisted)
    reason: str = ""
    sha256: str | None = None
    gate: dict[str, Any] = field(default_factory=dict)


SceneFn = Callable[[bytes, DeliveryVariant], tuple[bytes, dict[str, Any] | None]]


def register_takes(
    metadata_path: Path,
    items: dict[str, Item],
    store: StimulusStore,
    transcribers: dict[str, Transcriber],
    speaker: str = "bhavik",
    scene_fn: SceneFn | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[RegisterOutcome]:
    """Gate and register every unregistered latest take. See module docstring."""
    known: set[str] = set()
    for rec in store.records():
        known.add(rec.sha256)
        ts = rec.gates.get("take_sha256")
        if isinstance(ts, str):
            known.add(ts)

    outcomes: list[RegisterOutcome] = []
    for entry, row in sorted(latest_takes(metadata_path).items()):
        item_id, sep, variant_id = entry.partition("__")
        if not sep or item_id == "probe":
            outcomes.append(RegisterOutcome(entry, "skipped", "probe: reference asset, no variant"))
            continue
        item = items.get(item_id)
        variant = (
            next((v for v in item.variants if v.variant_id == variant_id), None) if item else None
        )
        if item is None or variant is None:
            outcomes.append(RegisterOutcome(entry, "skipped", "no such item/variant in bank"))
            continue
        existing = store.get(item_id, variant_id, "human")
        needs_scene = variant.scene is not None or variant.channel is not None
        # a scene variant registered as the raw take (sha == take sha, no recipe)
        # lacks its manipulation: that row is a defect, not a registration
        unscened = (
            needs_scene
            and existing is not None
            and existing.sha256 == row["sha256"]
            and not existing.scene
        )
        if row["sha256"] in known and not unscened:
            outcomes.append(RegisterOutcome(entry, "skipped", "already registered", row["sha256"]))
            continue
        wav_path = metadata_path.parent / Path(row["wav_path"]).name
        take_bytes = wav_path.read_bytes()
        if hashlib.sha256(take_bytes).hexdigest() != row["sha256"]:
            outcomes.append(RegisterOutcome(entry, "failed", f"sha256 mismatch for {wav_path}"))
            continue
        if existing is not None and "human_check" in existing.gates and not force:
            outcomes.append(
                RegisterOutcome(entry, "skipped", "existing human row carries a human_check")
            )
            continue

        wav, recipe = take_bytes, None
        if variant.scene is not None or variant.channel is not None:
            if scene_fn is None:
                outcomes.append(
                    RegisterOutcome(entry, "failed", "variant has a scene/channel; no scene_fn")
                )
                continue
            # the manipulation is part of the stimulus: an unscened take of a
            # scene variant is a missing stimulus, not a plain one (D065)
            wav, recipe = scene_fn(take_bytes, variant)

        slot = variant.scene.slot if variant.scene is not None else None
        # Background beds are additive post-processing over the take the speaker
        # read, so the take itself is the caller's pre-mix voice: gate it there.
        premix = (
            recipe is not None
            and variant.scene is not None
            and (variant.scene.kind == SceneKind.BACKGROUND)
        )
        gate = human_asr_gate(
            item.transcript,
            take_bytes if premix else wav,
            transcribers,
            impaired=variant.emotion in IMPAIRED_DELIVERIES,
            slot=None if premix else slot,
        )
        if premix:
            gate.update(
                {
                    "gated_on": "human take before the scene mix",
                    "premix_policy": PREMIX_POLICY,
                    "primary_sha256": row["sha256"],
                }
            )
        if gate.get("verdict", True) is None:
            # never persist an unmeasured content gate: gates_passed would not
            # see a failure on it, so the row would qualify unheard (D046)
            outcomes.append(RegisterOutcome(entry, "unmeasured", gate.get("error", ""), gate=gate))
            continue
        if not gate["passed"]:
            outcomes.append(RegisterOutcome(entry, "failed", gate.get("note", ""), gate=gate))
            continue
        if not dry_run:
            gates: dict[str, Any] = {"asr_roundtrip": gate, "take": int(row["take"])}
            if recipe is not None:
                gates["take_sha256"] = row["sha256"]
            if unscened:
                # the audio a listener hears changes; nothing heard before carries over
                gates["human_relisten"] = {
                    "passed": False,
                    "verdict": None,
                    "note": "scene now applied to a take registered without it; "
                    "needs a human ruling on the mixed clip",
                    "superseded_sha256": existing.sha256 if existing else None,
                }
            rec = store.put(
                wav,
                StimulusRecord(
                    item_id=item_id,
                    variant_id=variant_id,
                    sha256="",
                    engine="human",
                    model=speaker,
                    voice=speaker,
                    prompt=f"human recording take {row['take']} (D096 console)",
                    gates=gates,
                    scene=recipe,
                ),
            )
            known.add(rec.sha256)
            known.add(row["sha256"])
            sha = rec.sha256
        else:
            sha = row["sha256"]
        outcomes.append(
            RegisterOutcome(
                entry,
                "registered",
                str(gate.get("transcriber", "")),
                sha,
                gate,
            )
        )
    return outcomes
