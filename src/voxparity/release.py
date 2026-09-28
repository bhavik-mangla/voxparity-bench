"""Release filter: what may enter the public VoxParity bundle (docs/RELEASE-EXPORT.md).

Three rules, all enforced here so the export cannot drift from the register
(docs/POLICY-EXCLUSIONS.md):

1. **Stimulus rows** ship only when ``export_web.publishable`` admits them (an
   ALLOW-list: gemini, kokoro, human, found, and qwen3tts-* with a passing
   human_check). ``EXCLUDED_ENGINES`` is checked as well, as defence in depth: a
   later edit that widens the allow-list still cannot let a barred engine
   through. With a freeze, a row must also be one of the frozen usable clips.
2. **Every shipped row carries a ``licence_basis``** (PX-005). An allowed engine
   with no basis is an error, never a silent default.
3. **Run records** ship only for released items: an audio row needs a released
   stimulus, and a text-twin row (no audio) needs its item to be released,
   either named in ``allowed_items`` or, when that is not given, evidenced by a
   released audio row of the same item in the same batch. Never from an
   excluded driver family, and with account identifiers redacted from error
   strings.

With a public development split, pass the dev item ids as ``allowed_items`` to
both functions: a text-twin row carries a held-out item's gold in its scores,
so an engine-only check would publish the held-out bank's answers (release
blocker B1).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from voxparity.harness.export_web import publishable

# Barred by their terms or by Bhavik's source policy (PX-001/002/006/007/008/009/
# 010/025, D099). Checked in addition to the allow-list.
EXCLUDED_ENGINES = frozenset(
    {"elevenlabs", "fishaudio", "inworld", "cartesia", "hume", "camb", "aura"}
)
# Driver families excluded from published run records: any Deepgram-backed
# system (PX-010, D107). Matched as substrings of the record's ``driver``.
EXCLUDED_DRIVER_MARKERS = ("deepgram", "cascade-closed", "aura")

# Engine -> licence basis for the shipped audio. Dates are when the basis was
# last verified at source; re-verify before each release.
LICENCE_BASIS: dict[str, dict[str, str]] = {
    "gemini": {
        "basis": "Gemini API Additional Terms of Service: Google claims no ownership "
        "of generated content; released by the authors under CC BY 4.0",
        "url": "https://ai.google.dev/gemini-api/terms",
        "verified": "2026-09-25",
        "note": "most clips rendered on the unpaid tier (PX-003: inputs may be used "
        "by Google for training; disclosed)",
    },
    "kokoro": {
        "basis": "Kokoro-82M weights and voice packs Apache-2.0, rendered locally",
        "url": "https://huggingface.co/hexgrad/Kokoro-82M",
        "verified": "2026-09-25",
    },
    "qwen3tts-cv": {
        "basis": "Qwen3-TTS weights Apache-2.0, rendered locally; no output clause",
        "url": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "verified": "2026-09-25",
    },
    "qwen3tts-vd": {
        "basis": "Qwen3-TTS weights Apache-2.0, rendered locally; no output clause",
        "url": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "verified": "2026-09-25",
    },
    "human": {
        "basis": "recorded by a member of the team, who consented to their release "
        "under CC BY 4.0 (D118; no separate signed release required)",
        "url": "docs/archive/ethics/RECORDING-RELEASE.md",
        "verified": "2026-09-26",
    },
    "found": {
        "basis": "public domain per recipe (items/found/recipes/); shipped as a "
        "rebuild recipe, not as audio",
        "url": "docs/FOUND-AUDIO.md",
        "verified": "2026-09-13",
    },
}

# Account identifiers seen in provider error bodies (96 OpenAI org ids and 3
# user ids in the frozen run records, Sep 25).
_REDACT = [
    (re.compile(r"org-[A-Za-z0-9]{6,}"), "org-REDACTED"),
    (re.compile(r"user[_-][A-Za-z0-9]{6,}"), "user-REDACTED"),
    (re.compile(r"proj_[A-Za-z0-9]{6,}"), "proj-REDACTED"),
    (re.compile(r"(sk-|gsk_|AIza|xai-)[A-Za-z0-9_-]{8,}"), "KEY-REDACTED"),
]


class ReleaseError(ValueError):
    """The export would ship something the register does not clear."""


def redact(text: str) -> str:
    for pat, rep in _REDACT:
        text = pat.sub(rep, text)
    return text


def row_is_releasable(row: dict[str, Any]) -> bool:
    engine = str(row.get("engine", ""))
    if engine in EXCLUDED_ENGINES:
        return False
    rec = SimpleNamespace(engine=engine, gates=row.get("gates") or {})
    return bool(publishable(rec))


@dataclass
class ManifestRelease:
    rows: list[dict[str, Any]]
    dropped: Counter[str] = field(default_factory=Counter)

    @property
    def shas(self) -> frozenset[str]:
        return frozenset(r["sha256"] for r in self.rows)


def release_manifest(
    rows: Iterable[dict[str, Any]],
    frozen_shas: frozenset[str] | None = None,
    *,
    allowed_items: frozenset[str] | None = None,
) -> ManifestRelease:
    """Filter manifest rows to the public set and attach ``licence_basis``.

    ``allowed_items``, when given, is the set of item ids being released (the
    dev split); a row for any other item is dropped.
    """
    out = ManifestRelease(rows=[])
    for row in rows:
        engine = str(row.get("engine", ""))
        if allowed_items is not None and row.get("item_id") not in allowed_items:
            out.dropped["item-not-released"] += 1
            continue
        if not row_is_releasable(row):
            out.dropped[f"engine:{engine}"] += 1
            continue
        if frozen_shas is not None and row.get("sha256") not in frozen_shas:
            out.dropped["not-frozen-usable"] += 1
            continue
        basis = LICENCE_BASIS.get(engine)
        if basis is None:
            raise ReleaseError(f"allowed engine {engine!r} has no licence_basis (PX-005)")
        shipped = dict(row)
        shipped["licence_basis"] = dict(basis)
        shipped["distribution"] = "recipe-only" if engine == "found" else "audio"
        out.rows.append(shipped)
    return out


def record_is_releasable(
    rec: dict[str, Any],
    released_shas: frozenset[str],
    allowed_items: frozenset[str] | None = None,
) -> bool:
    """Per-row check. A text-twin row passes only with ``allowed_items``: on its
    own a row cannot show that its item is released (``release_records``
    supplies the batch evidence when ``allowed_items`` is absent)."""
    driver = str(rec.get("driver", "")).lower()
    if any(m in driver for m in EXCLUDED_DRIVER_MARKERS):
        return False
    if str(rec.get("engine", "")) in EXCLUDED_ENGINES:
        return False
    if allowed_items is not None and str(rec.get("item_id")) not in allowed_items:
        return False
    sha = rec.get("stimulus_sha256") or ""
    if sha:
        return sha in released_shas
    # text-twin rows carry no audio; they ship with their engine's run, and
    # only for a released item
    return allowed_items is not None and str(rec.get("engine", "")) in LICENCE_BASIS


def _probe_not_applicable(rec: dict[str, Any]) -> str | None:
    """Reason a run's probe rows are not applicable, keyed by the run's arm label."""
    from voxparity.harness.final_analysis import PROBE_NOT_APPLICABLE

    parts = str(rec.get("run_id", "")).split("-")
    return PROBE_NOT_APPLICABLE.get(parts[2]) if len(parts) > 3 else None


def release_records(
    records: Iterable[dict[str, Any]],
    released_shas: frozenset[str],
    *,
    allowed_items: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Filter run records to the public set.

    ``allowed_items`` is the set of released item ids (the dev split). Without
    it, a text-twin row ships only when a released audio row of the same item
    appears in ``records``: an engine being licensed says nothing about whether
    the item is public, and a twin row's scores name the item's gold actions.
    """
    records = list(records)
    items_ok = allowed_items
    if items_ok is None:
        items_ok = frozenset(
            str(r.get("item_id"))
            for r in records
            if (r.get("stimulus_sha256") or "") and record_is_releasable(r, released_shas)
        )
    kept: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for rec in records:
        if not record_is_releasable(rec, released_shas, items_ok):
            dropped[str(rec.get("engine", ""))] += 1
            continue
        out = dict(rec)
        if out.get("condition") == "probe" and _probe_not_applicable(out):
            # The paper treats these probes as not applicable (the answers cannot be
            # parsed); ship them that way so no tool counts them as wrong answers.
            out["scores"] = {"applicable": False, "reason": _probe_not_applicable(out)}
        if out.get("error"):
            out["error"] = redact(str(out["error"]))
        metrics = dict(out.get("metrics") or {})
        usage = metrics.get("usage")
        if isinstance(usage, dict):
            metrics["usage"] = {k: v for k, v in usage.items() if k != "is_byok"}
        out["metrics"] = metrics
        kept.append(out)
    return kept, dropped


# --- per-provider output restrictions (POLICY-EXCLUSIONS PX-031/037/041/042) ---
#
# Some providers' terms allow publishing scores derived from a response but not
# the response itself. Their rows ship with derived fields only: the chosen tool
# names, scores and metrics stay; response text, tool arguments and any ASR
# transcript are removed. Never re-score such a row: its arguments are gone.
RESTRICTED_NOTE = "derived fields only (provider terms)"


def restriction_reason(rec: dict[str, Any]) -> str | None:
    """Why a record may ship derived fields only, or None when it is unrestricted."""
    driver = str(rec.get("driver", "")).lower()
    upstream = str((rec.get("metrics") or {}).get("upstream_provider") or "").lower()
    if "nemotron" in driver and ":free" in driver:
        return "PX-037"
    if "grok" in driver or driver.startswith("xai"):
        return "PX-031"
    if "streamlake" in upstream:
        return "PX-041/PX-042"
    return None


def restrict_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Apply the derived-fields-only rule when a provider row requires it."""
    reason = restriction_reason(rec)
    if reason is None:
        return rec
    out = {k: v for k, v in rec.items() if k != "response_text"}
    out["tool_calls"] = [{"tool": (c or {}).get("tool")} for c in rec.get("tool_calls") or []]
    metrics = dict(out.get("metrics") or {})
    metrics.pop("asr_transcript", None)
    out["metrics"] = metrics
    out["restricted"] = f"{RESTRICTED_NOTE}; {reason}"
    return out


# --- manifest rows: what the public copy may say about each clip ---------------

# Keys in a gate that can carry a vendor ASR's words (PX-010: Deepgram output
# never ships). Every ASR transcript is removed; the verdict and WER stay.
_ASR_TEXT_KEYS = ("asr_transcript", "mix_asr_transcript", "transcript", "hypothesis")


def public_manifest_row(
    row: dict[str, Any],
    regate: dict[str, Any] | None = None,
    rater_alias: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Scrub one released manifest row for publication.

    - ASR transcripts are removed from every gate (Deepgram, PX-010), as is
      ``scene.slot_aligned_as`` (Deepgram word-timing text).
    - ``regate`` (a row of the local Whisper re-gate) is attached as
      ``gates.asr_regate_local`` with its verdict and WER only.
    - human raters are pseudonymised through ``rater_alias`` (a mapping the
      caller keeps stable across rows).
    """
    out = dict(row)
    gates: dict[str, Any] = {}
    for name, gate in (row.get("gates") or {}).items():
        if not isinstance(gate, dict):
            gates[name] = gate
            continue
        g = {k: v for k, v in gate.items() if k not in _ASR_TEXT_KEYS}
        if isinstance(g.get("transcripts"), dict):
            g.pop("transcripts")
        if isinstance(g.get("error"), str):
            g["error"] = redact(g["error"])
        if isinstance(g.get("ratings"), list) and rater_alias is not None:
            ratings = []
            for r in g["ratings"]:
                r = dict(r)
                if "rater" in r:
                    key = str(r["rater"])
                    r["rater"] = rater_alias.setdefault(key, f"rater-{len(rater_alias) + 1}")
                ratings.append(r)
            g["ratings"] = ratings
        gates[name] = g
    if regate is not None:
        gates["asr_regate_local"] = {
            "asr_engine": regate.get("local_asr"),
            "passed": regate.get("local_passed"),
            "wer": regate.get("local_wer"),
        }
    out["gates"] = gates
    if isinstance(out.get("scene"), dict):
        scene = dict(out["scene"])
        scene.pop("slot_aligned_as", None)
        out["scene"] = scene
    return out
