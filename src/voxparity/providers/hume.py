"""Hume Octave TTS — free-text delivery direction (D034 candidate).

Octave takes a per-utterance ``description`` field carrying free-form emotional
direction, separate from the text being spoken. That separation is what we want:
the instruction cannot leak into the words, so the ASR round-trip gate stays a
real content check.

LICENCE (PX-006): the free and Starter tiers are non-commercial only. Fine for
pilot iteration; no Hume audio may ship in the CC BY 4.0 dataset without a paid
tier or written permission. Registered in docs/POLICY-EXCLUSIONS.md at adoption
time, per the working rule there.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import httpx

from voxparity.providers.http import post_with_retry
from voxparity.providers.keys import EXHAUSTED_STATUSES, KeyPool
from voxparity.schemas.item import DeliveryVariant

BASE = "https://api.hume.ai/v0"
# MUST be "1". Verified live 2026-08-29: Octave 2 rejects the request with
# "Octave 2 does not support the 'description' parameter" — the free-text delivery
# direction, which is the entire reason Hume is a candidate, exists only on Octave
# 1 (English + Spanish). Newer is not better here. Pinned rather than left to
# Hume's auto-routing, which would be a silent confound either way.
VERSION = os.environ.get("VOXPARITY_HUME_VERSION", "1")
DEFAULT_VOICE = os.environ.get("VOXPARITY_HUME_VOICE", "Male English Actor")
VOICE_PROVIDER = os.environ.get("VOXPARITY_HUME_VOICE_PROVIDER", "HUME_AI")
DESCRIPTION_LIMIT = 1000


class HumeError(RuntimeError):
    pass


@dataclass
class HumeClient:
    api_key: str | None = None
    timeout: float = 180.0

    def __post_init__(self) -> None:
        try:
            self.pool = KeyPool("HUME", keys=[self.api_key] if self.api_key else [])
        except RuntimeError as e:
            raise HumeError(str(e)) from e
        self.api_key = self.pool.current

    def _headers(self) -> dict[str, str]:
        # The TTS endpoint uses the API key directly. The separate secret key is
        # only for browser token exchange and is deliberately unused here.
        return {"X-Hume-Api-Key": self.pool.current, "Content-Type": "application/json"}

    def synthesize(self, transcript: str, variant: DeliveryVariant, voice: str) -> bytes:
        from voxparity.stimuli.style import style_instruction

        description = style_instruction(variant)[:DESCRIPTION_LIMIT]
        body: dict[str, Any] = {
            "utterances": [
                {
                    "text": transcript,
                    "description": description,
                    "voice": {"name": voice, "provider": VOICE_PROVIDER},
                    "speed": 1.0,
                }
            ],
            # Defaults to mp3; the store and the ASR gate both expect wav.
            "format": {"type": "wav"},
            "version": VERSION,
            "num_generations": 1,
            # Instant mode requires a fixed voice; we pin one, but set this
            # explicitly so a voice-less variant would not 400 silently.
            "instant_mode": False,
        }
        tried = 0
        while True:
            try:
                resp = post_with_retry(
                    f"{BASE}/tts",
                    headers=self._headers(),
                    json_body=body,
                    timeout=self.timeout,
                    tries=2,
                )
            except (httpx.TransportError, TimeoutError) as e:
                raise HumeError(f"tts: {e}") from e
            if resp.status_code in EXHAUSTED_STATUSES:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
            break
        if resp.status_code != 200:
            raise HumeError(f"tts: HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            generation = resp.json()["generations"][0]
        except (KeyError, IndexError, ValueError) as e:
            raise HumeError(f"unexpected response shape: {resp.text[:300]}") from e
        return base64.b64decode(generation["audio"])

    def control_label(self, variant: DeliveryVariant) -> str:
        from voxparity.stimuli.style import style_instruction

        return f'description="{style_instruction(variant)[:DESCRIPTION_LIMIT]}"'
