"""Deepgram Nova-3 batch transcription — the independent ASR for the round-trip
gate (an ASR engine that shares nothing with any stimulus TTS engine, so no
provider grades its own homework). $200 signup credit ≈ 26k+ minutes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from voxparity.providers.http import post_with_retry
from voxparity.providers.keys import EXHAUSTED_STATUSES, KeyPool

MODEL = os.environ.get("VOXPARITY_DEEPGRAM_MODEL", "nova-3")


class DeepgramError(RuntimeError):
    pass


@dataclass
class DeepgramClient:
    api_key: str | None = None
    timeout: float = 120.0

    def __post_init__(self) -> None:
        try:
            self.pool = KeyPool("DEEPGRAM", keys=[self.api_key] if self.api_key else [])
        except RuntimeError as e:
            raise DeepgramError(str(e)) from e
        self.api_key = self.pool.current

    def transcribe(self, wav_bytes: bytes) -> str:
        url = f"https://api.deepgram.com/v1/listen?model={MODEL}&smart_format=false"
        tried = 0
        while True:
            headers = {
                "Authorization": f"Token {self.pool.current}",
                "Content-Type": "audio/wav",
            }
            try:
                resp = post_with_retry(
                    url, headers=headers, content=wav_bytes, timeout=self.timeout, tries=2
                )
            except (httpx.TransportError, TimeoutError) as e:
                raise DeepgramError(str(e)) from e
            if resp.status_code in EXHAUSTED_STATUSES:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
            break
        if resp.status_code != 200:
            raise DeepgramError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            alt = resp.json()["results"]["channels"][0]["alternatives"][0]
            return str(alt["transcript"])
        except (KeyError, IndexError) as e:
            raise DeepgramError(f"unexpected response shape: {resp.text[:300]}") from e

    def speak(self, text: str, voice: str = "thalia") -> bytes:
        """Aura-2 TTS: plain text -> WAV bytes (linear16 mono 24 kHz).

        Aura has NO style surface (voice + speed only), which is exactly why it
        serves as the scene-axis second engine (D087): neutral-delivery items
        need no control, and a fourth control family — none — resists preset
        learning. Clips rendered here MUST be gated by Gemini ASR, never by
        this vendor's own Nova-3 (cross-grading; see cli gate wiring).
        """
        url = (
            f"https://api.deepgram.com/v1/speak?model=aura-2-{voice}-en"
            "&encoding=linear16&container=wav&sample_rate=24000"
        )
        tried = 0
        while True:
            headers = {
                "Authorization": f"Token {self.pool.current}",
                "Content-Type": "application/json",
            }
            try:
                resp = post_with_retry(
                    url, headers=headers, json_body={"text": text}, timeout=self.timeout, tries=2
                )
            except (httpx.TransportError, TimeoutError) as e:
                raise DeepgramError(str(e)) from e
            if resp.status_code in EXHAUSTED_STATUSES:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
            break
        if resp.status_code != 200:
            raise DeepgramError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.content

    def transcribe_words(self, wav_bytes: bytes) -> list[tuple[str, float, float]]:
        """Word-level (text, start_s, end_s) timings — the slot aligner for
        scene rendering (D065). Same endpoint as transcribe; words come free."""
        url = f"https://api.deepgram.com/v1/listen?model={MODEL}&smart_format=false"
        headers = {
            "Authorization": f"Token {self.pool.current}",
            "Content-Type": "audio/wav",
        }
        resp = post_with_retry(url, headers=headers, content=wav_bytes, timeout=self.timeout)
        if resp.status_code != 200:
            raise DeepgramError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            alt = resp.json()["results"]["channels"][0]["alternatives"][0]
            return [(w["word"], float(w["start"]), float(w["end"])) for w in alt["words"]]
        except (KeyError, IndexError) as e:
            raise DeepgramError(f"unexpected response shape: {resp.text[:300]}") from e
