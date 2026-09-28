"""Groq (free tier): Whisper-large-v3-turbo STT and gpt-oss-120b (open weights) with native
tools — the OPEN cascade arm. Shares nothing with Gemini, so a cascade built on
it is the cleanest "does audio matter at all?" baseline.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from voxparity.providers.keys import EXHAUSTED_STATUSES, KeyPool

BASE = "https://api.groq.com/openai/v1"
STT_MODEL = os.environ.get("VOXPARITY_GROQ_STT", "whisper-large-v3-turbo")
# llama-3.3-70b-versatile moved to Groq enterprise-only on 2026-08-16 (still listed
# publicly, but unreachable on a free key); gpt-oss-120b is the
# strongest open-weights model it serves with native tool calling.
LLM_MODEL = os.environ.get("VOXPARITY_GROQ_LLM", "openai/gpt-oss-120b")


class GroqError(RuntimeError):
    pass


@dataclass
class GroqClient:
    timeout: float = 120.0
    pool: KeyPool = field(default_factory=lambda: KeyPool("GROQ"))

    # Per-minute token limits (8k TPM on the free tier) 429 every key in a
    # tight loop; wait for the window instead of recording an error row.
    rate_waits: int = 6
    max_wait_s: float = 60.0

    def _request(
        self, path: str, *, json_body: Any = None, files: Any = None, data: Any = None
    ) -> Any:
        tried = 0
        waits = 0
        while True:
            headers = {"Authorization": f"Bearer {self.pool.current}"}
            last: httpx.Response | None = None
            for attempt in range(3):
                try:
                    last = httpx.post(
                        f"{BASE}{path}",
                        json=json_body,
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=self.timeout,
                    )
                except httpx.TransportError:
                    time.sleep(3 * 2**attempt)
                    continue
                if last.status_code >= 500:
                    time.sleep(3 * 2**attempt)
                    continue
                break
            if last is not None and last.status_code in EXHAUSTED_STATUSES:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
                if last.status_code == 429 and waits < self.rate_waits and _per_minute(last):
                    waits += 1
                    time.sleep(min(self.max_wait_s, _retry_after(last)))
                    tried = 0
                    continue
            if last is None:
                raise GroqError("transport failure after retries")
            if last.status_code != 200:
                raise GroqError(f"HTTP {last.status_code}: {last.text[:300]}")
            return last.json()

    def transcribe(self, wav_bytes: bytes) -> str:
        data = self._request(
            "/audio/transcriptions",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
            data={"model": STT_MODEL, "response_format": "json", "language": "en"},
        )
        return str(data.get("text", "")).strip()

    def respond_with_tools(
        self, system: str, user_text: str, tool_decls: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        body: dict[str, Any] = {
            "model": LLM_MODEL,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        }
        if tool_decls:
            body["tools"] = [{"type": "function", "function": d} for d in tool_decls]
            body["tool_choice"] = "auto"
        data = self._request("/chat/completions", json_body=body)
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": fn.get("name", ""), "args": args})
        return (msg.get("content") or "").strip(), calls


def _per_minute(resp: httpx.Response) -> bool:
    """A per-minute (TPM/RPM) limit, which clears by waiting; a per-DAY limit
    does not, and must surface as an error so a resume loop takes over."""
    text = resp.text.lower()
    return "per day" not in text and "(rpd)" not in text and "(tpd)" not in text


def _retry_after(resp: httpx.Response) -> float:
    try:
        return max(1.0, float(resp.headers.get("retry-after", "20")))
    except ValueError:
        return 20.0


def openai_tool_decl(gemini_decl: dict[str, Any]) -> dict[str, Any]:
    """Gemini functionDeclaration -> OpenAI function schema (lowercase JSON types)."""
    params = gemini_decl.get("parameters")
    out: dict[str, Any] = {"name": gemini_decl["name"], "description": gemini_decl["description"]}
    if params:
        props = {
            k: {"type": v["type"].lower(), "description": v.get("description", "")}
            for k, v in params["properties"].items()
        }
        out["parameters"] = {
            "type": "object",
            "properties": props,
            "required": params.get("required", []),
        }
    return out
