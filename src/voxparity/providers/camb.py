"""Camb.ai Mars TTS — free-text instruction plus INLINE delivery tags (D034).

Two control channels, and the second is the interesting one:

- ``user_instructions`` — whole-clip tone guidance, free text. Only honoured when
  ``speech_model="mars-instruct"``; on any other model it is silently ignored,
  which would yield a batch of flat clips that pass ASR and fail the cue gate
  while looking like a TTS quality problem. The model id is therefore pinned, not
  defaulted.
- inline tags in the text itself, e.g.
  ``"[sighing] I suppose so. [brightening] Actually, yes!"`` — MID-UTTERANCE
  delivery control, which no engine currently in the bank can express. That is
  the direct enabler for D030's trajectory-flip and incongruence families, and it
  is free text rather than an enum.

LICENCE (PX-007): free-tier terms were not reachable at adoption time. No Camb
audio ships until that is resolved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from voxparity.providers.http import post_with_retry
from voxparity.providers.keys import EXHAUSTED_STATUSES, KeyPool
from voxparity.schemas.item import DeliveryVariant

BASE = "https://client.camb.ai/apis"
# Must be explicit: the default (mars-8.1-flash-beta) ignores user_instructions
# without erroring.
MODEL = os.environ.get("VOXPARITY_CAMB_MODEL", "mars-instruct")
DEFAULT_VOICE_ID = int(os.environ.get("VOXPARITY_CAMB_VOICE_ID", "147320"))
LANGUAGE = os.environ.get("VOXPARITY_CAMB_LANGUAGE", "en-us")
# mars-instruct renders at 22.05 kHz, unlike our 24 kHz Gemini stimuli — a
# per-engine confound worth recording rather than silently resampling.
SAMPLE_RATE = 22050


class CambError(RuntimeError):
    pass


@dataclass
class CambClient:
    api_key: str | None = None
    timeout: float = 180.0

    def __post_init__(self) -> None:
        try:
            self.pool = KeyPool("CAMB", keys=[self.api_key] if self.api_key else [])
        except RuntimeError as e:
            raise CambError(str(e)) from e
        self.api_key = self.pool.current
        self.last_credits: str | None = None

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.pool.current, "Content-Type": "application/json"}

    def list_voices(self) -> list[dict[str, Any]]:
        resp = httpx.get(f"{BASE}/list-voices", headers=self._headers(), timeout=self.timeout)
        if resp.status_code != 200:
            raise CambError(f"list-voices: HTTP {resp.status_code}: {resp.text[:200]}")
        return list(resp.json().get("voices", []))

    def synthesize(self, transcript: str, variant: DeliveryVariant, voice_id: int) -> bytes:
        from voxparity.stimuli.style import style_instruction

        body: dict[str, Any] = {
            "text": transcript,
            "language": LANGUAGE,
            "voice_id": voice_id,
            "speech_model": MODEL,
            "user_instructions": style_instruction(variant),
            "output_configuration": {"format": "wav", "sample_rate": SAMPLE_RATE},
            "voice_settings": {"speaking_rate": 1.0},
        }
        tried = 0
        while True:
            try:
                resp = post_with_retry(
                    f"{BASE}/tts-stream",
                    headers=self._headers(),
                    json_body=body,
                    timeout=self.timeout,
                    tries=2,
                )
            except (httpx.TransportError, TimeoutError) as e:
                raise CambError(f"tts: {e}") from e
            if resp.status_code in EXHAUSTED_STATUSES:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
            break
        if resp.status_code != 200:
            raise CambError(f"tts: HTTP {resp.status_code}: {resp.text[:300]}")
        # Errors come back as JSON with a 200-adjacent shape on some paths; writing
        # a JSON blob into a .wav would only surface at the gate.
        content_type = resp.headers.get("Content-Type", "")
        if "audio" not in content_type:
            raise CambError(f"expected audio, got {content_type!r}: {resp.text[:300]}")
        self.last_credits = resp.headers.get("X-Credits-Required")
        return resp.content

    def control_label(self, variant: DeliveryVariant) -> str:
        from voxparity.stimuli.style import style_instruction

        label = f'user_instructions="{style_instruction(variant)}" model={MODEL}'
        if self.last_credits:
            label += f" credits={self.last_credits}"
        return label
