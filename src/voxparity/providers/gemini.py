"""Thin Gemini REST client (free tier) for TTS synthesis, transcription, and
audio judging. REST over httpx keeps dependencies minimal; retries handle the
free tier's per-minute rate limits.

Never spends non-free credits: all default models are on the Gemini free tier
(verified Aug 2026). Key loaded from GEMINI_API_KEY (see .env, gitignored).
"""

from __future__ import annotations

import base64
import io
import os
import wave
from dataclasses import dataclass
from typing import Any

import httpx

from voxparity.providers.http import post_with_retry
from voxparity.providers.keys import EXHAUSTED_STATUSES, KeyPool

BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
TTS_MODEL = os.environ.get("VOXPARITY_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TEXT_MODEL = os.environ.get("VOXPARITY_JUDGE_MODEL", "gemini-3.6-flash")
# Dedicated transcription model (ai.google.dev/gemini-api/docs/transcribe,
# verified Sep 2026): verbatim is the documented default mode — "preserving raw
# filler words ('um', 'uh', 'like', 'you know'), repetitions, pauses, and false
# starts" — and we pin it explicitly anyway so a default change upstream cannot
# silently clean our disfluent stimuli.
TRANSCRIBE_MODEL = os.environ.get("VOXPARITY_TRANSCRIBE_MODEL", "gemini-3.5-transcribe")
_MAX_TRIES = 5


def forced_choice_prompt(question: str, options: list[str]) -> str:
    """The forced-choice instruction every audio judge receives, verbatim.

    Shared by the Gemini cue judge and the independent cross-judge
    (``harness/crossjudge.py``) so the two answer the SAME prompt: a
    disagreement between them is then about the listener, not the wording."""
    listed = "\n".join(f"- {o}" for o in options)
    return (
        f"{question}\nAnswer with EXACTLY one of these options, verbatim, "
        f"and nothing else:\n{listed}"
    )


class GeminiError(RuntimeError):
    pass


@dataclass
class GeminiClient:
    api_key: str | None = None
    timeout: float = 120.0

    def __post_init__(self) -> None:
        try:
            self.pool = KeyPool("GEMINI", keys=[self.api_key] if self.api_key else [])
        except RuntimeError as e:
            raise GeminiError(str(e)) from e
        self.api_key = self.pool.current

    def _generate(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE}/models/{model}:generateContent"
        tried = 0
        while True:
            headers = {"x-goog-api-key": self.pool.current}
            try:
                # short per-key backoff; exhausted keys rotate instead of sleeping
                resp = post_with_retry(
                    url, headers=headers, json_body=body, timeout=self.timeout, tries=2
                )
            except TimeoutError:
                resp = None
            except httpx.TransportError as e:
                raise GeminiError(f"{model}: {e}") from e
            if resp is None or resp.status_code in EXHAUSTED_STATUSES:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
                if resp is None:
                    raise GeminiError(f"{model}: rate-limited on all {self.pool.size} key(s)")
            if resp.status_code != 200:
                raise GeminiError(f"{model}: HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            # a successful billed call routes the NEXT request back to free keys
            self.pool.release_paid()
            return data  # type: ignore[no-any-return]

    @staticmethod
    def _first_part(data: dict[str, Any]) -> dict[str, Any]:
        try:
            return data["candidates"][0]["content"]["parts"][0]  # type: ignore[no-any-return]
        except (KeyError, IndexError) as e:
            raise GeminiError(f"unexpected response shape: {str(data)[:300]}") from e

    def synthesize(self, prompt: str, voice: str = "Kore") -> bytes:
        """TTS: style-prompted text -> WAV bytes (PCM s16le mono, 24 kHz)."""
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
            },
        }
        part = self._first_part(self._generate(TTS_MODEL, body))
        inline = part.get("inlineData")
        if not inline:
            raise GeminiError(f"no audio in response: {str(part)[:200]}")
        pcm = base64.b64decode(inline["data"])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)
        return buf.getvalue()

    def _audio_prompt(self, instruction: str, wav_bytes: bytes) -> str:
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": instruction},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(wav_bytes).decode(),
                            }
                        },
                    ]
                }
            ]
        }
        part = self._first_part(self._generate(TEXT_MODEL, body))
        text = part.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise GeminiError("empty text response")
        return text.strip()

    def respond_with_tools(
        self,
        model: str,
        system: str,
        user_parts: list[dict[str, Any]],
        tool_decls: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Model-under-test call: system prompt + user parts (audio and/or text) +
        function declarations. Returns (text, raw functionCall dicts)."""
        body: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": user_parts}],
        }
        if tool_decls:
            body["tools"] = [{"functionDeclarations": tool_decls}]
        data = self._generate(model, body)
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as e:
            raise GeminiError(f"unexpected response shape: {str(data)[:300]}") from e
        text = " ".join(p["text"] for p in parts if "text" in p).strip()
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        return text, calls

    def generate_json(self, prompt: str, model: str | None = None, temperature: float = 0.9) -> Any:
        """Text-only call with JSON response mode; returns parsed JSON. Authoring
        keeps the creative default; judges pass temperature=0."""
        import json

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": temperature,
            },
        }
        part = self._first_part(self._generate(model or TEXT_MODEL, body))
        text = part.get("text", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise GeminiError(f"non-JSON response: {text[:200]}") from e

    def transcribe(self, wav_bytes: bytes) -> str:
        return self._audio_prompt(
            "Transcribe this audio verbatim. Output only the spoken words, nothing else.",
            wav_bytes,
        )

    # ---- dedicated transcription model (interactions API) -------------------

    class _Exhausted(RuntimeError):
        """Internal: quota/rate-limit at some step of the upload+interact
        sequence — the whole sequence restarts on the next key (an uploaded
        file belongs to the key that uploaded it, so steps never mix keys)."""

    def transcribe_verbatim(self, wav_bytes: bytes, model: str | None = None) -> str:
        """ASR via the dedicated ``gemini-3.5-transcribe`` model, verbatim mode
        pinned explicitly (docs contract: preserves fillers, repetitions,
        pauses, false starts). Uses the Files API (the documented input route:
        upload, then pass the file uri to POST /v1beta/interactions)."""
        model = model or TRANSCRIBE_MODEL
        tried = 0
        while True:
            try:
                text = self._transcribe_once(model, wav_bytes)
            except GeminiClient._Exhausted:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
                raise GeminiError(f"{model}: rate-limited on all {self.pool.size} key(s)") from None
            # a successful billed call routes the NEXT request back to free keys
            self.pool.release_paid()
            return text

    def _transcribe_once(self, model: str, wav_bytes: bytes) -> str:
        key = self.pool.current
        uri = self._upload_wav(key, wav_bytes)
        body = {
            "model": model,
            "input": [{"type": "audio", "uri": uri, "mime_type": "audio/wav"}],
            "generation_config": {"transcription_config": {"mode": {"type": "verbatim"}}},
        }
        data = self._post_step(f"{BASE}/interactions", {"x-goog-api-key": key}, json_body=body)
        text = data.get("output_text", "")
        if not text:
            pieces = [
                c.get("text", "")
                for step in data.get("steps") or []
                for c in step.get("content") or []
                if c.get("type") == "text" and c.get("text")
            ]
            text = " ".join(pieces)
        if not isinstance(text, str) or not text.strip():
            raise GeminiError(
                f"{model}: no transcript in response "
                f"(status={data.get('status')!r}): {str(data)[:300]}"
            )
        return text.strip()

    def _upload_wav(self, key: str, wav_bytes: bytes) -> str:
        """Files API resumable upload (start -> upload,finalize); returns the
        file uri, valid only for `key`."""
        start = self._post_step(
            UPLOAD_URL,
            {
                "x-goog-api-key": key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(wav_bytes)),
                "X-Goog-Upload-Header-Content-Type": "audio/wav",
            },
            json_body={"file": {"display_name": "voxparity-clip.wav"}},
            raw=True,
        )
        upload_url = start.headers.get("x-goog-upload-url", "")
        if not upload_url:
            raise GeminiError(f"files upload: no x-goog-upload-url header: {start.text[:200]}")
        fin = self._post_step(
            upload_url,
            {
                "x-goog-api-key": key,
                "X-Goog-Upload-Command": "upload, finalize",
                "X-Goog-Upload-Offset": "0",
            },
            content=wav_bytes,
        )
        uri = (fin.get("file") or {}).get("uri", "")
        if not uri:
            raise GeminiError(f"files upload: no file.uri in response: {str(fin)[:200]}")
        return str(uri)

    def _post_step(
        self,
        url: str,
        headers: dict[str, str],
        *,
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        raw: bool = False,
    ) -> Any:
        """One HTTP step of the transcription sequence: quota statuses raise
        _Exhausted (rotate + restart the sequence), everything else non-200 is
        a definitive GeminiError."""
        try:
            resp = post_with_retry(
                url,
                headers=headers,
                json_body=json_body,
                content=content,
                timeout=self.timeout,
                tries=2,
            )
        except TimeoutError:
            raise GeminiClient._Exhausted(url) from None
        except httpx.TransportError as e:
            raise GeminiError(f"{url}: {e}") from e
        if resp.status_code in EXHAUSTED_STATUSES:
            raise GeminiClient._Exhausted(url)
        if resp.status_code != 200:
            raise GeminiError(f"{url}: HTTP {resp.status_code}: {resp.text[:300]}")
        return resp if raw else resp.json()

    def classify(self, wav_bytes: bytes, question: str, options: list[str]) -> str:
        """Forced-choice audio judgment; returns one of `options` (or raises)."""
        from voxparity.scoring.toolcall import match_option

        answer = self._audio_prompt(forced_choice_prompt(question, options), wav_bytes)
        matched = match_option(answer, options)
        if matched is None:
            raise GeminiError(f"judge answered off-menu: {answer!r}")
        return matched
