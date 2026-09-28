"""Inworld TTS-2 — a dedicated ``instruction`` field (D034 candidate).

The delivery direction travels in its own top-level field, entirely separate from
the text, and Inworld documents explicitly that instruction tags are "removed from
the text before synthesis". That guarantee is what Fish lacks, and it means the
ASR round-trip gate is measuring content fidelity rather than tag leakage.

Also the best licence position of any candidate: Inworld assigns all right, title
and interest in outputs to the caller, with no clause restricting use in an
AI-testing dataset. That makes it the only shortlist engine whose audio could ship
in the public CC BY 4.0 release without a permission negotiation.

Two API details that cost people a 401 or a silent no-op:
- The key from the portal is ALREADY base64; it is pasted verbatim after "Basic ".
  Re-encoding it is the most common failure.
- ``inworld-tts-2-flash`` ignores steering entirely. The model id is pinned to
  ``inworld-tts-2`` because a flash request would succeed while quietly discarding
  the emotion, producing flat clips that pass ASR and fail the cue gate.
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

BASE = "https://api.inworld.ai"
MODEL = os.environ.get("VOXPARITY_INWORLD_MODEL", "inworld-tts-2")
DEFAULT_VOICE = os.environ.get("VOXPARITY_INWORLD_VOICE", "Ashley")
# STABLE | BALANCED | CREATIVE — replaces `temperature`, which tts-2 ignores.
DELIVERY_MODE = os.environ.get("VOXPARITY_INWORLD_DELIVERY", "CREATIVE")
SAMPLE_RATE = 44100


class InworldError(RuntimeError):
    pass


@dataclass
class InworldClient:
    api_key: str | None = None
    timeout: float = 180.0

    def __post_init__(self) -> None:
        try:
            self.pool = KeyPool("INWORLD", keys=[self.api_key] if self.api_key else [])
        except RuntimeError as e:
            raise InworldError(str(e)) from e
        self.api_key = self.pool.current
        self.last_usage: dict[str, Any] | None = None

    def _headers(self) -> dict[str, str]:
        # The portal key is already base64 — pasted verbatim, never re-encoded.
        return {
            "Authorization": f"Basic {self.pool.current}",
            "Content-Type": "application/json",
        }

    def list_voices(self) -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{BASE}/voices/v1/voices",
            headers=self._headers(),
            params={"filter": 'source = "SYSTEM" AND lang_code = "en"', "pageSize": 50},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise InworldError(f"list-voices: HTTP {resp.status_code}: {resp.text[:200]}")
        return list(resp.json().get("voices", []))

    def synthesize(self, transcript: str, variant: DeliveryVariant, voice_id: str) -> bytes:
        from voxparity.stimuli.style import style_instruction

        body: dict[str, Any] = {
            "text": transcript,
            "voiceId": voice_id,
            "modelId": MODEL,
            # Field form, not inline tags: Inworld says to pick one, and the field
            # keeps the transcript byte-identical for the ASR gate.
            "instruction": style_instruction(variant),
            "deliveryMode": DELIVERY_MODE,
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": SAMPLE_RATE},
            "applyTextNormalization": "ON",
        }
        tried = 0
        while True:
            try:
                resp = post_with_retry(
                    f"{BASE}/tts/v1/voice",
                    headers=self._headers(),
                    json_body=body,
                    timeout=self.timeout,
                    tries=2,
                )
            except (httpx.TransportError, TimeoutError) as e:
                raise InworldError(f"tts: {e}") from e
            if resp.status_code in EXHAUSTED_STATUSES:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
            break
        if resp.status_code != 200:
            raise InworldError(f"tts: HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            payload = resp.json()
            audio = payload["audioContent"]
        except (KeyError, ValueError) as e:
            raise InworldError(f"unexpected response shape: {resp.text[:300]}") from e
        self.last_usage = payload.get("usage")
        # LINEAR16 arrives with a WAV header already attached.
        return base64.b64decode(audio)

    def control_label(self, variant: DeliveryVariant) -> str:
        from voxparity.stimuli.style import style_instruction

        label = f'instruction="{style_instruction(variant)}" delivery={DELIVERY_MODE}'
        if self.last_usage:
            label += f" chars={self.last_usage.get('processedCharactersCount')}"
        return label
