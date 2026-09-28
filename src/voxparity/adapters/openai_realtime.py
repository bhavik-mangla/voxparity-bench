"""OpenAI-dialect committed-turn drivers: OpenAI Realtime, xAI Grok Voice,
Alibaba Qwen-Omni-Realtime. IMPLEMENTED FROM DOCS, UNTESTED LIVE — no keys for
any of the three exist yet (each needs the provider's API key).
Every row they write carries ``realtime.live_tested: false`` until a live smoke
run flips the class attribute.

The three share one event vocabulary (``input_audio_buffer.append`` /
``.commit``, ``response.create``, ``response.function_call_arguments.done``,
``response.done``) and differ in session shape, tool format, audio rates and
whether a text user turn exists:

- OpenAI: GA ``session.type=realtime``, ``audio.input.turn_detection: null``;
  flat tools; 24 kHz in; text twin yes.
- xAI: ``turn_detection: {"type": null}``, ``audio.input.format.rate``; flat
  tools; 24 kHz in (8-48 kHz accepted); text twin yes.
- Qwen: flat ``input_audio_format: pcm``, ``turn_detection: null``; nested
  Chat-format tools; 16 kHz in; text twin NO (per docs).

Sources (fetched 2026-09-14): OpenAI
https://developers.openai.com/api/docs/guides/realtime-conversations ; xAI
https://docs.x.ai/docs/guides/voice/agent ; Qwen
https://www.alibabacloud.com/help/en/model-studio/realtime (updated 2026-09-04),
https://www.alibabacloud.com/help/en/model-studio/client-events and
https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling .
Cross-checked against tau2-bench ``src/tau2/voice/audio_native/{openai,xai,qwen}``
(MIT, Copyright (c) 2025 Sierra Research).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import urlsplit

from voxparity.adapters.base import SessionContext
from voxparity.adapters.gemini_file import tool_decl
from voxparity.adapters.realtime import (
    RealtimeDriver,
    RealtimeError,
    RealtimeQuota,
    TurnState,
    WebSocketLike,
    b64,
)
from voxparity.providers.groq import openai_tool_decl
from voxparity.providers.keys import KeyPool

AUDIO_DELTAS = ("response.output_audio.delta", "response.audio.delta")
TRANSCRIPT_DELTAS = ("response.output_audio_transcript.delta", "response.audio_transcript.delta")
TEXT_DELTAS = ("response.output_text.delta", "response.text.delta")
RATE_LIMIT_CODES = ("rate_limit_exceeded", "insufficient_quota", "quota_exceeded", "throttling")


class OpenAIDialectDriver(RealtimeDriver):
    """Shared event handling; subclasses provide URL, auth and session shape."""

    key_env: str = ""
    base_url: str = ""
    live_tested = False
    supports_reasoning_effort = True

    def __init__(self, model: str, pool: KeyPool | None = None, **kw: Any) -> None:
        super().__init__(model, **kw)
        self._pool = pool

    @property
    def pool(self) -> KeyPool:
        if self._pool is None:
            try:
                self._pool = KeyPool(self.key_env)
            except RuntimeError as e:
                raise RealtimeError(
                    f"{self.key_env}_API_KEY is not set — {self.provider} realtime needs a key "
                    "from the operator"
                ) from e
        return self._pool

    def url(self) -> str:
        return f"{self.base_url}?model={self.model}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.pool.current}"}

    def rotate_key(self) -> bool:
        return self.pool.rotate()

    def key_index(self) -> int | None:
        return self._pool.idx if self._pool is not None else None

    # -- session ----------------------------------------------------------------
    def tools_payload(self, ctx: SessionContext) -> list[dict[str, Any]]:
        return [{"type": "function", **openai_tool_decl(tool_decl(t))} for t in ctx.tools]

    def session_config(self, ctx: SessionContext) -> dict[str, Any]:
        raise NotImplementedError

    def with_reasoning(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """``reasoning.effort`` (OpenAI gpt-realtime-2.x: minimal..xhigh; xAI
        grok-voice: none..high). Omitted when unset, so the provider default
        applies and the row records ``reasoning_effort: null``."""
        if self.reasoning_effort:
            cfg["reasoning"] = {"effort": self.reasoning_effort}
        return cfg

    async def setup(self, ws: WebSocketLike, ctx: SessionContext, st: TurnState) -> None:
        await ws.send(json.dumps({"type": "session.update", "session": self.session_config(ctx)}))
        deadline = time.perf_counter() + self.connect_timeout_s
        while True:
            msg = await self.recv_json(ws, deadline)
            kind = msg.get("type", "")
            st.seen(kind)
            if kind == "session.updated":
                st.reasoning_echo = reasoning_echo(msg.get("session"))
                return
            if kind == "error":
                raise _error(msg)

    async def send_audio_turn(self, ws: WebSocketLike, pcm: bytes, st: TurnState) -> None:
        for chunk in self.chunks(pcm):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64(chunk)}))
            await self.pace_sleep(chunk)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        st.committed_ack = False
        await ws.send(json.dumps({"type": "response.create"}))
        st.t_commit = time.perf_counter()

    async def send_text_turn(self, ws: WebSocketLike, text: str, st: TurnState) -> None:
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
        await ws.send(json.dumps({"type": "response.create"}))
        st.t_commit = time.perf_counter()

    # -- events -------------------------------------------------------------------
    def handle(self, msg: dict[str, Any], st: TurnState) -> bool:
        return handle_openai_event(msg, st)


class OpenAIRealtimeDriver(OpenAIDialectDriver):
    provider = "openai"
    key_env = "OPENAI"
    base_url = "wss://api.openai.com/v1/realtime"
    input_rate = 24000
    output_rate = 24000
    supports_text_input = True
    turn_control_doc = "https://developers.openai.com/api/docs/guides/realtime-conversations"

    def __init__(self, model: str | None = None, voice: str = "marin", **kw: Any) -> None:
        super().__init__(
            model or os.environ.get("VOXPARITY_OPENAI_REALTIME_MODEL", "gpt-realtime-2.1"), **kw
        )
        self.voice = voice

    def session_config(self, ctx: SessionContext) -> dict[str, Any]:
        fmt = {"type": "audio/pcm", "rate": self.input_rate}
        cfg: dict[str, Any] = {
            "type": "realtime",
            "model": self.model,
            "instructions": ctx.system_prompt,
            "output_modalities": ["audio"],
            "audio": {
                "input": {"format": fmt, "turn_detection": None},
                "output": {"format": fmt, "voice": self.voice},
            },
        }
        if ctx.tools:
            cfg["tools"] = self.tools_payload(ctx)
            cfg["tool_choice"] = "auto"
        return self.with_reasoning(cfg)


class AzureOpenAIRealtimeDriver(OpenAIRealtimeDriver):
    """Azure OpenAI / Foundry GA Realtime: the OpenAI wire unchanged, served from
    ``wss://<resource>.openai.azure.com/openai/v1/realtime?model=<deployment>``
    with an ``api-key`` header (https://learn.microsoft.com/en-us/azure/foundry/
    openai/concepts/realtime-2, updated 2026-09-23). ``model`` is the DEPLOYMENT
    name, which the operator chooses in the portal; the row records it verbatim.
    ``reasoning.effort`` is documented there as minimal/low/medium/high.

    Key: ``AZURE_OPENAI_API_KEY(S)``, else ``FOUNDRY_API``; endpoint:
    ``AZURE_OPENAI_ENDPOINT``, else ``FOUNDRY_AZURE_OPENAI_ENDPOINT`` (any URL on
    the resource host; only the host is used).
    """

    provider = "azure"
    key_env = "AZURE_OPENAI"
    turn_control_doc = (
        "https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/realtime-audio-websockets"
    )

    def __init__(self, model: str | None = None, **kw: Any) -> None:
        super().__init__(
            model or os.environ.get("VOXPARITY_AZURE_REALTIME_DEPLOYMENT", "gpt-realtime-2.1"),
            **kw,
        )

    @property
    def pool(self) -> KeyPool:
        if self._pool is None:
            try:
                self._pool = KeyPool(self.key_env)
            except RuntimeError:
                raw = os.environ.get("FOUNDRY_API", "")
                keys = [k.strip() for k in raw.split(",") if k.strip()]
                if not keys:
                    raise RealtimeError(
                        "AZURE_OPENAI_API_KEY (or FOUNDRY_API) is not set — Azure realtime "
                        "needs a Foundry resource key"
                    ) from None
                self._pool = KeyPool("FOUNDRY", keys=keys)
        return self._pool

    @property
    def base_url(self) -> str:  # type: ignore[override]
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get(
            "FOUNDRY_AZURE_OPENAI_ENDPOINT", ""
        )
        host = urlsplit(endpoint.strip()).netloc
        if not host:
            raise RealtimeError("AZURE_OPENAI_ENDPOINT (or FOUNDRY_AZURE_OPENAI_ENDPOINT) not set")
        return f"wss://{host}/openai/v1/realtime"

    def headers(self) -> dict[str, str]:
        return {"api-key": self.pool.current}


class GrokVoiceDriver(OpenAIDialectDriver):
    provider = "xai"
    key_env = "XAI"
    base_url = "wss://api.x.ai/v1/realtime"
    # xAI accepts 8-48 kHz PCM; 24 kHz is our stimulus rate, so no resampling.
    input_rate = 24000
    output_rate = 24000
    supports_text_input = True
    turn_control_doc = "https://docs.x.ai/docs/guides/voice/agent"

    def __init__(self, model: str | None = None, voice: str = "eve", **kw: Any) -> None:
        super().__init__(
            model or os.environ.get("VOXPARITY_XAI_REALTIME_MODEL", "grok-voice-think-fast-2.0"),
            **kw,
        )
        self.voice = voice

    def session_config(self, ctx: SessionContext) -> dict[str, Any]:
        fmt = {"type": "audio/pcm", "rate": self.input_rate}
        cfg: dict[str, Any] = {
            "instructions": ctx.system_prompt,
            "voice": self.voice,
            "turn_detection": {"type": None},
            "audio": {"input": {"format": fmt}, "output": {"format": fmt}},
        }
        if ctx.tools:
            cfg["tools"] = self.tools_payload(ctx)
        return self.with_reasoning(cfg)


class QwenOmniRealtimeDriver(OpenAIDialectDriver):
    """Qwen3.5-Omni realtime (tool calling is documented for 3.5-omni plus/flash
    only; tau2 found qwen3-omni-flash-realtime accepts tools but never calls them).

    Text twin: Alibaba's client-events reference says conversation.item.create
    supports "Only items of the function_call_output type", so no text user turn
    exists and the twin is recorded not-applicable (D035). tau2-bench sends an
    ``input_text`` message anyway — an undocumented path we do not rely on.
    """

    provider = "qwen"
    key_env = "DASHSCOPE"
    input_rate = 16000
    output_rate = 24000
    supports_text_input = False
    supports_reasoning_effort = False
    turn_control_doc = "https://www.alibabacloud.com/help/en/model-studio/client-events"

    def __init__(self, model: str | None = None, voice: str | None = None, **kw: Any) -> None:
        super().__init__(
            model or os.environ.get("VOXPARITY_QWEN_REALTIME_MODEL", "qwen3.5-omni-flash-realtime"),
            **kw,
        )
        self.voice = voice

    @property
    def base_url(self) -> str:  # type: ignore[override]
        # Live debugging, Sep 20: the per-workspace maas host IS the documented
        # intl realtime endpoint and completes session.update with a valid key
        # (the earlier 1006 there was a malformed key). The shared
        # dashscope-intl host ACCEPTS the websocket but returns AccessDenied on
        # every session.update — a trap: connect success there proves nothing.
        explicit = os.environ.get("DASHSCOPE_REALTIME_URL")
        if explicit:
            return explicit
        ws_id = os.environ.get("DASHSCOPE_WORKSPACE_ID")
        if ws_id:
            return f"wss://{ws_id}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime"
        return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"

    def headers(self) -> dict[str, str]:
        h = super().headers()
        ws_id = os.environ.get("DASHSCOPE_WORKSPACE_ID")
        if ws_id:
            h["X-DashScope-WorkspaceId"] = ws_id
        return h

    def tools_payload(self, ctx: SessionContext) -> list[dict[str, Any]]:
        # Nested Chat-Completions format, not the flat Realtime format.
        return [{"type": "function", "function": openai_tool_decl(tool_decl(t))} for t in ctx.tools]

    def session_config(self, ctx: SessionContext) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "instructions": ctx.system_prompt,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": None,
        }
        if self.voice:
            cfg["voice"] = self.voice
        if ctx.tools:
            cfg["tools"] = self.tools_payload(ctx)
        return cfg


class QwenAudioRealtimeDriver(QwenOmniRealtimeDriver):
    """Qwen-Audio realtime (``qwen-audio-3.1-realtime-plus``), Alibaba's
    dedicated speech-to-speech line, distinct from the Omni family.

    Differences from Qwen-Omni, per
    https://www.alibabacloud.com/help/en/model-studio/qwen-audio-realtime-user-guides
    (fetched 2026-09-25): ``voice`` is REQUIRED and only honoured on the first
    ``session.update`` (default ``beth_v3.1``, an English system voice);
    ``conversation.item.create`` documents a user ``message`` with
    ``input_text`` content, so the text twin exists (unlike Omni, D035); and the
    documented transcript event is ``response.audio_transcript.done`` only, so
    the full transcript is taken from it when no deltas arrived. Same nested
    Chat-format tools, manual mode ``turn_detection: null``, 16 kHz PCM in.
    """

    provider = "qwen-audio"
    supports_text_input = True
    turn_control_doc = (
        "https://www.alibabacloud.com/help/en/model-studio/qwen-audio-realtime-user-guides"
    )

    def __init__(self, model: str | None = None, voice: str | None = None, **kw: Any) -> None:
        super().__init__(
            model
            or os.environ.get(
                "VOXPARITY_QWEN_AUDIO_REALTIME_MODEL", "qwen-audio-3.1-realtime-plus"
            ),
            voice=voice or os.environ.get("VOXPARITY_QWEN_AUDIO_VOICE", "beth_v3.1"),
            **kw,
        )

    def handle(self, msg: dict[str, Any], st: TurnState) -> bool:
        if msg.get("type") == "response.audio_transcript.done" and not st.transcript:
            st.seen("response.audio_transcript.done")
            st.mark_response()
            st.transcript.append(msg.get("transcript", ""))
            return False
        return super().handle(msg, st)


def handle_openai_event(msg: dict[str, Any], st: TurnState) -> bool:
    """Fold one OpenAI-dialect server event into ``st``; True when the turn is over.

    Module-level so the Vercel AI Gateway driver can reuse it on the native
    provider event the gateway forwards in each normalized event's ``raw``.
    """
    kind = msg.get("type", "")
    st.seen(kind)
    if kind == "error":
        raise _error(msg)
    if kind == "input_audio_buffer.committed":
        st.committed_ack = True
    elif kind in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"):
        # With turn_detection null these should never fire; if they do, the
        # provider is running VAD on our replay and the cell is not tier B.
        st.server_vad_events += 1
    elif kind == "conversation.item.input_audio_transcription.completed":
        st.input_transcript.append(msg.get("transcript", ""))
    elif kind == "response.created":
        st.mark_response()
    elif kind in AUDIO_DELTAS:
        st.add_audio(msg.get("delta", ""))
    elif kind in TRANSCRIPT_DELTAS:
        st.mark_response()
        st.transcript.append(msg.get("delta", ""))
    elif kind in TEXT_DELTAS:
        st.mark_response()
        st.text.append(msg.get("delta", ""))
    elif kind == "response.function_call_arguments.done":
        st.add_call(msg.get("name", ""), msg.get("arguments", "{}"), msg.get("call_id"))
    elif kind == "response.output_item.done":
        item = msg.get("item") or {}
        if item.get("type") == "function_call":
            st.add_call(item.get("name", ""), item.get("arguments", "{}"), item.get("call_id"))
    elif kind == "response.cancelled":
        st.interrupted = True
    elif kind == "response.done":
        resp = msg.get("response") or {}
        if resp.get("usage") is not None:
            st.usage = resp["usage"]
        elif msg.get("usage") is not None:
            st.usage = msg["usage"]
        for item in resp.get("output") or []:
            if item.get("type") == "function_call":
                st.add_call(item.get("name", ""), item.get("arguments", "{}"), item.get("call_id"))
        status = resp.get("status")
        if status == "failed":
            raise _error({"error": (resp.get("status_details") or {}).get("error") or resp})
        if status == "cancelled":
            st.interrupted = True
        st.done_reason = "response_done" if status in (None, "completed") else str(status)
        return True
    return False


def reasoning_echo(session: Any) -> Any:
    """The effort the server says it applied, from a session echo; None if absent."""
    if isinstance(session, dict) and isinstance(session.get("reasoning"), dict):
        return session["reasoning"].get("effort")
    return None


def _error(msg: dict[str, Any]) -> RealtimeError:
    err = msg.get("error") if isinstance(msg.get("error"), dict) else msg
    code = str((err or {}).get("code") or (err or {}).get("type") or "")
    text = str((err or {}).get("message") or err)[:300]
    if code in RATE_LIMIT_CODES or "rate limit" in text.lower() or "quota" in text.lower():
        return RealtimeQuota(f"{code}: {text}")
    return RealtimeError(f"{code}: {text}" if code else text)
