"""StepFun driver — StepAudio 3 chat over StepFun's OpenAI-compatible API.

StepFun's chat completions endpoint speaks the same shape the OpenRouter
adapter already sends: a base64 ``input_audio`` content part alongside an
OpenAI-style ``tools`` array in one call. Verified 2026-09-20 from
platform.stepfun.ai/docs (en/api-reference/chat/chat-completion-create.md):
``POST https://api.stepfun.ai/v1/chat/completions``, ``tools`` supported,
temperature 0.0-2.0, audio "currently only mp3 and wav are supported".

Two documented differences from OpenRouter, both handled here:

- the audio part's ``data`` is a data URI (``data:audio/wav;base64,...``)
  with no separate ``format`` key;
- there is no catalogue with modality metadata, so preflight checks the
  ``/models`` listing for the pinned id and makes one cheap text call. That
  cannot prove the model HEARS (the D032 hazard); the smoke items' probe
  answers are the hearing check before any full run.

Default model ``stepaudio-3-chat-preview`` — "Free (limited time)" per the
pricing page, and the preview id "will be retired when the free trial
concludes" (MODEL-SCAN-2026-09 §2.2), so every run records the id and date.
OpenRouter's quirks (attribution headers, provider pinning, the credits
floor, Mistral's top_p carve-out) stay OpenRouter-only.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import httpx

from voxparity.adapters.openrouter import OpenRouterDriver, OpenRouterError
from voxparity.providers.keys import KeyPool

BASE = os.environ.get("VOXPARITY_STEPFUN_BASE", "https://api.stepfun.ai/v1")
MODEL = os.environ.get("VOXPARITY_STEPFUN_MODEL", "stepaudio-3-chat-preview")
# Free tier is 10 requests/minute (measured live 2026-09-20: HTTP 429 "request
# limited RPM reached, current: 11, limit: 10"). A serial run stays under it
# with a >=6s floor between requests; 429s that slip through are waited out,
# not rotated — one key, and the minute window always reopens.
MIN_INTERVAL_S = float(os.environ.get("VOXPARITY_STEPFUN_MIN_INTERVAL_S", "6.2"))
RATE_LIMIT_RETRIES = 4


class StepFunError(OpenRouterError):
    pass


class StepFunDriver(OpenRouterDriver):
    def __init__(self, model: str | None = None) -> None:
        self.model = model or MODEL
        self.name = f"stepfun:{self.model}"
        self.pool = KeyPool("STEPFUN")
        self.base_url = BASE
        # The chat model takes "speech or text turn by turn" (model doc), so
        # the transcript twin exists — unlike the gpt-audio family (D035).
        self._text_twin = True
        self._last_request_at = 0.0

    def _headers(self) -> dict[str, str]:
        # No OpenRouter attribution headers; StepFun only wants the bearer key.
        return {"Authorization": f"Bearer {self.pool.current}"}

    def _audio_part(self, audio_path: str) -> dict[str, Any]:
        """StepFun documents a data-URI ``data`` field and no ``format`` key."""
        wav = base64.b64encode(Path(audio_path).read_bytes()).decode()
        return {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{wav}"}}

    def _pin_upstream(self, body: dict[str, Any]) -> None:
        """Single vendor — OpenRouter's routing pin must never leak in here."""

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Pace requests under the free tier's 10 RPM and wait out any 429.

        The inherited path treats 429 as key exhaustion and rotates the pool —
        right for OpenRouter credit limits, wrong here: there is one key and
        the per-minute window reopens by itself, so an un-retried 429 would
        write an error row for a cell nothing is wrong with (the D046 class).
        """
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            wait = self._last_request_at + MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()
            try:
                return super()._post(body)
            except OpenRouterError as e:
                if "HTTP 429" in str(e) and attempt < RATE_LIMIT_RETRIES:
                    time.sleep(15.0 * (attempt + 1))
                    continue
                raise
        raise AssertionError("unreachable")

    def preflight(self) -> None:
        """Confirm the key works and the preview model id still exists.

        The preview id is retired when the free trial ends, so a dead id must
        fail once here rather than 796 times mid-run (D032). StepFun's model
        listing carries no modality metadata, so unlike OpenRouter this cannot
        veto a deaf model — the 3-item smoke's probe rows are that check.
        """
        try:
            resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=60.0)
        except httpx.TransportError as e:
            raise StepFunError(f"preflight: {e}") from e
        if resp.status_code != 200:
            raise StepFunError(f"preflight: HTTP {resp.status_code}: {resp.text[:200]}")
        ids = {m.get("id") for m in resp.json().get("data", [])}
        if self.model not in ids:
            raise StepFunError(
                f"model {self.model!r} is not in the StepFun model listing "
                f"({len(ids)} models). Preview ids are retired when the free "
                "trial concludes — check platform.stepfun.ai/docs for the "
                "current StepAudio 3 chat id."
            )
        # Listing presence does not prove this key may call it: one cheap call.
        try:
            self._post(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                }
            )
        except OpenRouterError as e:
            raise StepFunError(f"preflight call to {self.model!r} failed: {e}") from e
