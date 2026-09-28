"""Apply a variant's SceneSpec to its rendered speech (D062/D065).

The scene is the manipulation on the acoustic-context axes, so this step is as
much a part of stimulus identity as the TTS render itself: the returned recipe
(asset provenance, SNR, seed, resolved slot window) is stored on the manifest
record, making every published scene clip reproducible from its parts.

Asset resolution:
- ``synth:*``  — procedurally generated here (assets.py), license-free.
- ``real:*``   — recorded environmental beds resolved through committed pack
  recipes (packs.py); the verified cache file feeds the mixer exactly like a
  synth temp file, and the recipe's license/attribution ride the scene recipe
  onto the manifest (published-dataset attribution requirement).
- ``dtmf:<digits>`` — keypad tones (mix.dtmf).
- ``tts:*``    — a second TTS voice speaking ``scene.text``; the caller supplies
  the render function so this module stays engine-agnostic.

Slot placement needs word timings for the PRIMARY render — supplied by the
caller (Deepgram's word timestamps in the CLI path, anything in tests). Slot
tokens are matched with the same normalization the scorer uses for strings.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from voxparity.schemas.item import SceneKind, SceneSpec
from voxparity.scoring.toolcall import NormalizedExactMatcher
from voxparity.stimuli.assets import synthesize_asset
from voxparity.stimuli.mix import dtmf, mix_background, overlay_dtmf, overlay_noise_slot, truncate
from voxparity.stimuli.packs import load_recipe, resolve_real_asset

_PAD_S = 0.04  # widen the mask slightly past the word boundaries


def _resolve_asset(asset_id: str, tmp: Path, seed: int, packs_dir: Path) -> Path:
    """A mixable WAV path for a non-tts, non-dtmf asset id.

    ``synth:`` renders into the scene's temp dir; ``real:`` resolves to the
    hash-verified pack cache (packs.py) — which fails loudly when the cache is
    missing or corrupted, never silently substituting synthesis.
    """
    if asset_id.startswith("real:"):
        return resolve_real_asset(asset_id, packs_dir=packs_dir)
    path = tmp / "asset.wav"
    path.write_bytes(synthesize_asset(asset_id, seed=seed))
    return path


class Word(NamedTuple):
    text: str
    start: float
    end: float


_FUZZY_FLOOR = 0.7

_NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}


def _digit_fold(tokens: list[str]) -> str | None:
    """Concatenated digit string for an all-numeric token run, else None.

    ASR renormalizes spoken digit strings — live-observed: Deepgram returned
    "three fifty" for a spoken "three five zero". Both fold to "350", which is
    the identity that actually matters for masking the right span.
    """
    out = []
    for tok in tokens:
        if tok.isdigit():
            out.append(tok)
        elif tok in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[tok])
        else:
            return None
    return "".join(out) if out else None


def locate_slot(words: list[Word], slot: str) -> tuple[float, float, str]:
    """Return (start_s, dur_s, matched_text) of ``slot`` in the aligned words.

    Exact contiguous token match first. When the ASR renormalizes the words —
    live-observed: Deepgram returned "three fifty" for a spoken "three five
    zero" — fall back to the best contiguous window by character similarity,
    accepted only above a floor. The matched text goes into the recipe so a
    fuzzy hit is auditable, never silent.
    """
    from difflib import SequenceMatcher

    norm = NormalizedExactMatcher._norm
    target = norm(slot).split()
    tokens = [norm(w.text) for w in words]

    def span(i: int, j: int) -> tuple[float, float, str]:
        start = max(words[i].start - _PAD_S, 0.0)
        end = words[j].end + _PAD_S
        return start, end - start, " ".join(tokens[i : j + 1])

    for i in range(len(tokens) - len(target) + 1):
        if tokens[i : i + len(target)] == target:
            return span(i, i + len(target) - 1)

    target_digits = _digit_fold(target)
    if target_digits is not None:
        for width in range(1, len(tokens) + 1):
            for i in range(len(tokens) - width + 1):
                if _digit_fold(tokens[i : i + width]) == target_digits:
                    return span(i, i + width - 1)

    target_str = " ".join(target)
    best, best_ij = 0.0, None
    for width in (len(target) - 1, len(target), len(target) + 1):
        if width < 1:
            continue
        for i in range(len(tokens) - width + 1):
            ratio = SequenceMatcher(None, target_str, " ".join(tokens[i : i + width])).ratio()
            if ratio > best:
                best, best_ij = ratio, (i, i + width - 1)
    if best_ij is not None and best >= _FUZZY_FLOOR:
        return span(*best_ij)
    raise ValueError(f"slot {slot!r} not found in aligned words {tokens} (best fuzzy {best:.2f})")


def apply_scene(
    primary_wav: bytes,
    scene: SceneSpec,
    *,
    background_tts: Callable[[str], bytes] | None = None,
    words: list[Word] | None = None,
    packs_dir: Path = Path("stimuli/packs"),
) -> tuple[bytes, dict[str, Any]]:
    """Mix the scene into the rendered speech; return (wav_bytes, recipe)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        primary = tmp / "primary.wav"
        primary.write_bytes(primary_wav)
        out = tmp / "out.wav"

        if scene.kind is SceneKind.SLOT_NOISE:
            if words is None:
                raise ValueError("slot_noise needs word timings for the primary render")
            assert scene.slot is not None  # schema validator guarantees this
            start_s, dur_s, matched = locate_slot(words, scene.slot)
            asset = _resolve_asset(scene.asset, tmp, scene.seed, packs_dir)
            recipe = overlay_noise_slot(
                primary,
                asset,
                out,
                slot_start_s=start_s,
                slot_dur_s=dur_s,
                snr_db=scene.snr_db,
                seed=scene.seed,
            )
            recipe["slot"] = scene.slot
            recipe["slot_aligned_as"] = matched
        elif scene.kind is SceneKind.TRUNCATION:
            if words is None:
                raise ValueError("truncation needs word timings for the primary render")
            assert scene.slot is not None  # schema validator guarantees this
            start_s, _dur_s, matched = locate_slot(words, scene.slot)
            # synth:silence_tail = the speaker stopping (raw cut, no artifact);
            # anything else (synth:line_drop) is appended as the drop artifact.
            tail = None
            if scene.asset != "synth:silence_tail":
                tail = _resolve_asset(scene.asset, tmp, scene.seed, packs_dir)
            recipe = truncate(
                primary,
                out,
                cut_at_s=start_s,
                tail_path=tail,
                tail_snr_db=scene.snr_db,
            )
            recipe["slot"] = scene.slot
            recipe["slot_aligned_as"] = matched
        elif scene.kind is SceneKind.DTMF:
            asset = tmp / "asset.wav"
            dtmf(scene.asset.removeprefix("dtmf:"), asset)
            recipe = overlay_dtmf(
                primary,
                asset,
                out,
                start_s=scene.start_s,
                snr_db=scene.snr_db,
            )
        else:  # BACKGROUND
            if scene.asset.startswith("tts:"):
                if background_tts is None:
                    raise ValueError(f"{scene.asset!r} needs a background_tts renderer")
                if not scene.text:
                    raise ValueError("a tts: scene asset needs scene.text")
                asset = tmp / "asset.wav"
                asset.write_bytes(background_tts(scene.text))
            else:
                asset = _resolve_asset(scene.asset, tmp, scene.seed, packs_dir)
            recipe = mix_background(
                primary,
                asset,
                out,
                snr_db=scene.snr_db,
                seed=scene.seed,
                start_s=scene.start_s,
            )
        recipe["asset"] = scene.asset
        recipe["kind"] = scene.kind.value
        if scene.text:
            recipe["text"] = scene.text
        if scene.asset.startswith("real:"):
            # Published-dataset attribution: the recorded bed's provenance
            # travels on the manifest with the clip it was mixed into.
            real = load_recipe(scene.asset, packs_dir=packs_dir)
            recipe["asset_license"] = real.get("license")
            recipe["asset_attribution"] = real.get("attribution")
        return out.read_bytes(), recipe
