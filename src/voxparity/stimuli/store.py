"""Content-addressed stimulus store + manifest.

Audio lives outside git at ``stimuli/<sha256>.wav`` (gitignored). The manifest
(``stimuli/manifest.yaml``, committed) is the provenance record: for every
(item, variant) it stores the hash, engine/voice/model, the exact style prompt,
and validation-gate results. Runs resolve audio through the manifest, never by
path convention — a missing or hash-mismatched file fails loudly.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StimulusRecord:
    item_id: str
    variant_id: str
    sha256: str
    engine: str
    model: str
    voice: str
    prompt: str
    gates: dict[str, Any] = field(default_factory=dict)
    scene: dict[str, Any] | None = None  # mixing recipe for scene variants (D065)
    # The voice itself changes between this row and its twin (Qwen VoiceDesign
    # re-invents the speaker per render, D100). Such a pair varies speaker AND
    # delivery, so it is excluded from pair discrimination; audio-minus-twin
    # stays valid per variant. Serialized only when true.
    speaker_varies: bool = False


def _row(rec: StimulusRecord) -> dict[str, Any]:
    row = asdict(rec)
    if not row.get("speaker_varies"):
        row.pop("speaker_varies", None)
    return row


def speaker_varying_shas(store_dir: Path) -> frozenset[str]:
    """Hashes of stimuli whose rows carry ``speaker_varies: true``."""
    if not (store_dir / "manifest.yaml").exists():
        return frozenset()
    return frozenset(r.sha256 for r in StimulusStore(store_dir).records() if r.speaker_varies)


def repair_wav_header(wav: bytes) -> bytes:
    """Rewrite RIFF/data chunk sizes to the real byte count.

    Streaming TTS endpoints emit a placeholder length: 334 clips in this store
    declare 2147483520 or 2147483647 frames (~24 hours) for two-second audio,
    across Cartesia, Fish and older un-manifested renders. Players and our own
    `wav_seconds` cope by reading to EOF, but a PUBLISHED clip carrying a
    24-hour duration is a defect in the dataset, so the repair happens at the
    export boundary (D051).

    Deliberately not applied in-place in the store: the store is content-addressed
    by sha256, and rewriting bytes would invalidate every hash in the manifest and
    in every recorded run.
    """
    import struct

    if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return wav
    out = bytearray(wav)
    pos = 12
    while pos + 8 <= len(out):
        chunk_id = bytes(out[pos : pos + 4])
        (declared,) = struct.unpack("<I", out[pos + 4 : pos + 8])
        body = pos + 8
        if chunk_id == b"data":
            actual = len(out) - body
            if declared != actual:
                struct.pack_into("<I", out, pos + 4, actual)
                struct.pack_into("<I", out, 4, len(out) - 8)
            return bytes(out)
        pos = body + declared + (declared % 2)
        if declared == 0 or pos <= body:
            break
    return bytes(out)


class StimulusStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.audio_dir = root
        self.manifest_path = root / "manifest.yaml"
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[tuple[str, str, str], StimulusRecord] = {}
        self._dirty: set[tuple[str, str, str]] = set()
        if self.manifest_path.exists():
            raw = yaml.safe_load(self.manifest_path.read_text()) or []
            for entry in raw:
                rec = StimulusRecord(**entry)
                self._records[(rec.item_id, rec.variant_id, rec.engine)] = rec

    def put(self, wav_bytes: bytes, record: StimulusRecord) -> StimulusRecord:
        digest = hashlib.sha256(wav_bytes).hexdigest()
        record.sha256 = digest
        (self.audio_dir / f"{digest}.wav").write_bytes(wav_bytes)
        key = (record.item_id, record.variant_id, record.engine)
        self._records[key] = record
        self._dirty.add(key)
        self._save()
        return record

    def get(self, item_id: str, variant_id: str, engine: str) -> StimulusRecord | None:
        return self._records.get((item_id, variant_id, engine))

    def audio_path(self, record: StimulusRecord) -> Path:
        path = self.audio_dir / f"{record.sha256}.wav"
        if not path.exists():
            raise FileNotFoundError(f"stimulus audio missing: {path}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.sha256:
            raise ValueError(f"hash mismatch for {path} — store corrupted")
        return path

    def set_gate(
        self, item_id: str, variant_id: str, engine: str, gate: str, result: dict[str, Any]
    ) -> None:
        rec = self._records[(item_id, variant_id, engine)]
        rec.gates[gate] = result
        self._dirty.add((item_id, variant_id, engine))
        self._save()

    def records(self) -> list[StimulusRecord]:
        return sorted(self._records.values(), key=lambda r: (r.item_id, r.variant_id, r.engine))

    def _save(self) -> None:
        """Atomic AND dirty-aware: disk is the source of truth for every row
        this process has NOT written; only rows in ``self._dirty`` override.

        v1 of this merge (FLAG-006) kept "rows this process changed" by keeping
        the whole in-memory table — but a long-lived writer's UNCHANGED rows are
        stale copies of load-time state, and writing them back resurrected
        records a faster writer had since replaced (live-hit: a gate sweep
        resurrected a superseded wirerb render across two re-renders, D076).
        The dirty set distinguishes "mine" from "merely loaded"; everything
        else re-reads from disk under the lock.
        """
        import fcntl
        import os

        lock = self.manifest_path.with_suffix(".lock")
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                merged: dict[tuple[str, str, str], StimulusRecord] = {}
                if self.manifest_path.exists():
                    raw = yaml.safe_load(self.manifest_path.read_text()) or []
                    for entry in raw:
                        rec = StimulusRecord(**entry)
                        merged[(rec.item_id, rec.variant_id, rec.engine)] = rec
                for key in self._dirty:
                    if key in self._records:
                        merged[key] = self._records[key]
                self._records = merged
                data = [_row(r) for r in self.records()]
                tmp = self.manifest_path.with_suffix(".yaml.tmp")
                tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
                os.replace(tmp, self.manifest_path)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
