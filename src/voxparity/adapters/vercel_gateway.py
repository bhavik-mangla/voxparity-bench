"""Vercel AI Gateway realtime driver: ``--driver realtime:vercel:<creator>/<model>[@effort]``.

One committed-turn driver for every realtime model the gateway serves over its
normalized WebSocket (verified live Sep 25 2026 with an AI Gateway key):

- endpoint ``wss://ai-gateway.vercel.sh/v4/ai/realtime-model?ai-model-id=<id>``
  (the URL ``@ai-sdk/gateway`` 4.0.92 builds in ``toGatewayRealtimeUrl``);
- auth: ``Authorization: Bearer <AI Gateway API key>`` on the upgrade request.
  The documented browser path mints a single-use ``vcst_`` secret with
  ``POST https://ai-gateway.vercel.sh/v1/realtime/client-secrets`` and sends it
  as the ``ai-gateway-auth.<token>`` subprotocol; both work from Python, the
  header saves a round trip. NOTE: minting succeeds for ANY model string,
  including ids that do not exist, so a token is never evidence the model is
  reachable — ``preflight`` opens a real session instead;
- client events are the AI SDK's provider-neutral ``RealtimeModelV4ClientEvent``
  (``session-update``, ``input-audio-append``, ``input-audio-commit``,
  ``conversation-item-create``, ``response-create``);
- server events are normalized, but every one carries the provider's native
  event in ``raw``. We parse ``raw`` with the SAME handlers the direct drivers
  use for OpenAI and xAI (``handle_openai_event``), so a tool call is read
  identically whether it came direct or through the gateway. Gemini's native
  messages are folded by ``handle_gemini_raw`` below, which follows the direct
  driver's NON_BLOCKING rule (fix/gemini-live-nonblocking).

Per-upstream behaviour, all observed live, not assumed:

- ``openai/*`` and ``spacexai/grok-voice-*``: ``turnDetection: disabled``
  reaches the provider as ``turn_detection: null`` (session echo), tools pass
  through, and ``providerOptions`` are merged into the provider session, so
  ``{"reasoning": {"effort": e}}`` is applied and echoed back; an invalid value
  is rejected at setup ("Supported values are: 'minimal', 'low', 'medium',
  'high', and 'xhigh'" for gpt-realtime-2.1). Tier B, same as direct.
- ``google/gemini-*-live*``: ``turnDetection: disabled`` makes the gateway
  close the socket silently, and so does a ``realtimeInputConfig``
  passthrough, so server VAD stays ON and the commit is ``audioStreamEnd``.
  That is VAD-bound replay: fidelity tier **C**, ``deterministic_turns`` False.
  Thinking: ``providerOptions.google.thinkingConfig.thinkingLevel`` (low/high
  accepted on 3.8-live-extended-thinking; ``minimal`` closes the socket). The
  direct ``realtime:gemini-live`` driver keeps manual activity control and is
  the preferred route for Gemini.
- ``openai/gpt-live-1`` is NOT reachable here: the upstream answers "not
  supported in realtime mode". It has its own ``/v1/live/sessions`` endpoint and
  no tool schema at all (client delegation only), so it cannot score T4.

Billing notes: the gateway meters xAI voice per CLIENT MESSAGE
(``realtime_client_message_cost`` $0.004 in the catalog) plus session seconds,
so audio is sent in as few frames as the 256 KB message cap allows.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from voxparity.adapters.base import SessionContext
from voxparity.adapters.gemini_file import tool_decl
from voxparity.adapters.openai_realtime import handle_openai_event, reasoning_echo
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

GATEWAY_WS = "wss://ai-gateway.vercel.sh/v4/ai/realtime-model"
GATEWAY_HTTP = "https://ai-gateway.vercel.sh"
DEFAULT_MODEL = "openai/gpt-realtime-2.1"
GOOGLE_TRAILING_SILENCE_S = 1.0
MAX_MESSAGE_BYTES = 256 * 1024  # gateway session limit; larger frames are rejected
QUOTA_HINTS = ("insufficient", "credit", "balance", "payment required", "rate limit", "quota")


def gateway_pool() -> KeyPool:
    """``AI_GATEWAY_API_KEY(S)`` (Vercel's own variable name), falling back to
    ``VERCEL_GATEWAY`` (comma-separated), which is how this repo's .env names it."""
    try:
        return KeyPool("AI_GATEWAY")
    except RuntimeError:
        raw = os.environ.get("VERCEL_GATEWAY", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise RealtimeError(
                "AI_GATEWAY_API_KEY (or VERCEL_GATEWAY) is not set — create a key under "
                "Vercel dashboard > AI Gateway > API Keys"
            ) from None
        return KeyPool("VERCEL_GATEWAY", keys=keys)


class VercelGatewayRealtimeDriver(RealtimeDriver):
    provider = "vercel"
    live_tested = True
    supports_reasoning_effort = True
    output_rate = 24000
    supports_text_input = True
    turn_control_doc = "https://vercel.com/docs/ai-gateway/modalities/realtime"

    def __init__(self, model: str | None = None, pool: KeyPool | None = None, **kw: Any) -> None:
        name = model or DEFAULT_MODEL
        google = name.startswith("google/")
        # Google through the gateway is VAD-bound (turnDetection cannot be
        # disabled). Measured live Sep 25: the turn only completes when audio is
        # streamed at real-time pace AND followed by trailing silence; bursts,
        # or pacing without the silence, never end the turn. OpenAI/xAI take
        # large unpaced frames (fewer client messages, which xAI bills).
        kw.setdefault("chunk_ms", 100 if google else 2000)
        kw.setdefault("pace", 1.0 if google else 0.0)
        super().__init__(name, **kw)
        self.trailing_silence_s = GOOGLE_TRAILING_SILENCE_S if google else 0.0
        self.creator = self.model.split("/", 1)[0] if "/" in self.model else ""
        if not self.creator:
            raise ValueError(f"gateway model ids are <creator>/<model>, got {self.model!r}")
        if self.model.startswith("openai/gpt-live"):
            raise RealtimeError(
                "gpt-live models use /v1/live/sessions with client delegation and no tool "
                "schema; they cannot be driven as a committed-turn tool-calling session"
            )
        self._pool = pool
        self._last_raw: str | None = None
        # Gemini Live takes 16 kHz in; OpenAI and xAI take our 24 kHz stimuli as is.
        self.input_rate = 16000 if self.is_google else 24000
        self.fidelity_tier = "C" if self.is_google else "B"
        # a 2 s PCM16 chunk at 24 kHz is ~128 KB of base64: well under the cap
        step_b64 = int(self.input_rate * self.chunk_ms / 1000) * 2 * 4 // 3
        if step_b64 > MAX_MESSAGE_BYTES - 1024:
            raise ValueError(f"chunk_ms={self.chunk_ms} exceeds the gateway's 256 KB frame cap")

    @property
    def is_google(self) -> bool:
        return self.creator == "google"

    @property
    def pool(self) -> KeyPool:
        if self._pool is None:
            self._pool = gateway_pool()
        return self._pool

    def url(self) -> str:
        return f"{GATEWAY_WS}?ai-model-id={self.model}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.pool.current}"}

    def rotate_key(self) -> bool:
        return self.pool.rotate()

    def key_index(self) -> int | None:
        return self._pool.idx if self._pool is not None else None

    def extra_metrics(self) -> dict[str, Any]:
        return {
            "serving_path": "vercel-ai-gateway",
            "upstream": self.creator,
            "turn_control": "vad+audioStreamEnd" if self.is_google else "manual-commit",
            "trailing_silence_s": self.trailing_silence_s,
        }

    # -- session ------------------------------------------------------------------
    def provider_options(self) -> dict[str, Any] | None:
        e = self.reasoning_effort
        if not e:
            return None
        if self.is_google:
            return {"google": {"thinkingConfig": {"thinkingLevel": e}}}
        return {"reasoning": {"effort": e}}

    def session_config(self, ctx: SessionContext) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "instructions": ctx.system_prompt,
            "outputModalities": ["audio"],
            "inputAudioFormat": {"type": "audio/pcm", "rate": self.input_rate},
            "outputAudioFormat": {"type": "audio/pcm", "rate": self.output_rate},
        }
        if self.is_google:
            # probe answers are scored on the spoken reply's transcript
            cfg["outputAudioTranscription"] = {}
            cfg["inputAudioTranscription"] = {}
        else:
            cfg["turnDetection"] = {"type": "disabled"}
        if ctx.tools:
            cfg["tools"] = [gateway_tool(t) for t in ctx.tools]
        opts = self.provider_options()
        if opts:
            cfg["providerOptions"] = opts
        return cfg

    async def setup(self, ws: WebSocketLike, ctx: SessionContext, st: TurnState) -> None:
        self._last_raw = None
        await ws.send(json.dumps({"type": "session-update", "config": self.session_config(ctx)}))
        # OpenAI/xAI: session-created arrives BEFORE our update is applied; only
        # session-updated proves the config (tools, turn control, effort) took.
        # Google: session-created wraps setupComplete, which is the ack.
        ack = "session-created" if self.is_google else "session-updated"
        deadline = time.perf_counter() + self.connect_timeout_s
        while True:
            msg = await self.recv_json(ws, deadline)
            kind = msg.get("type", "")
            st.seen(f"gw:{kind}")
            if kind == "error":
                raise gateway_error(msg)
            if kind == ack:
                raw = msg.get("raw")
                if not self.is_google and isinstance(raw, dict):
                    st.reasoning_echo = reasoning_echo(raw.get("session"))
                return

    async def send_audio_turn(self, ws: WebSocketLike, pcm: bytes, st: TurnState) -> None:
        pcm = pcm + b"\x00\x00" * int(self.input_rate * self.trailing_silence_s)
        for chunk in self.chunks(pcm):
            await ws.send(json.dumps({"type": "input-audio-append", "audio": b64(chunk)}))
            await self.pace_sleep(chunk)
        await ws.send(json.dumps({"type": "input-audio-commit"}))
        st.committed_ack = False
        if not self.is_google:  # Google: audioStreamEnd already ends the turn
            await ws.send(json.dumps({"type": "response-create"}))
        st.t_commit = time.perf_counter()

    async def send_text_turn(self, ws: WebSocketLike, text: str, st: TurnState) -> None:
        item = {"type": "text-message", "role": "user", "text": text}
        await ws.send(json.dumps({"type": "conversation-item-create", "item": item}))
        if not self.is_google:
            await ws.send(json.dumps({"type": "response-create"}))
        st.t_commit = time.perf_counter()

    # -- events ---------------------------------------------------------------------
    def handle(self, msg: dict[str, Any], st: TurnState) -> bool:
        kind = msg.get("type", "")
        st.seen(f"gw:{kind}")
        raw = msg.get("raw")
        if kind == "error" and not isinstance(raw, dict):
            raise gateway_error(msg)  # the gateway's own error, no upstream event
        if not isinstance(raw, dict):
            return False
        # One upstream message can fan out into several normalized events that
        # all carry the SAME raw (Gemini: audio + transcript in one frame); fold
        # it once or the transcript is doubled.
        key = json.dumps(raw, sort_keys=True)
        if key == self._last_raw:
            return False
        self._last_raw = key
        if self.is_google:
            return handle_gemini_raw(raw, st)
        return handle_openai_event(raw, st)


def handle_gemini_raw(msg: dict[str, Any], st: TurnState) -> bool:
    """Fold one native Gemini Live message (forwarded in ``raw``); True = turn over.

    3.8 Live tool calls are NON_BLOCKING: measured through the gateway on Sep 25,
    extended thinking speaks a filler, sends ``turnComplete`` with
    ``interactionStatus: IN_PROGRESS``, and the ``toolCall`` follows ~1.4 s
    later. An IN_PROGRESS ``turnComplete`` therefore never ends the turn; the
    tool call, or an ``IDLE`` status after it, does.
    """
    if "error" in msg:
        raise gateway_error({"message": str(msg["error"])})
    if "usageMetadata" in msg:
        st.usage = msg["usageMetadata"]
    sc = msg.get("serverContent")
    if isinstance(sc, dict):
        for part in (sc.get("modelTurn") or {}).get("parts") or []:
            if "inlineData" in part:
                st.add_audio((part["inlineData"] or {}).get("data", ""))
            elif part.get("text") and not part.get("thought"):
                st.mark_response()
                st.text.append(part["text"])
        if (sc.get("outputTranscription") or {}).get("text"):
            st.mark_response()
            st.transcript.append(sc["outputTranscription"]["text"])
        if (sc.get("inputTranscription") or {}).get("text"):
            st.input_transcript.append(sc["inputTranscription"]["text"])
        if sc.get("interrupted"):
            st.interrupted = True
        status = sc.get("interactionStatus")
        if status:
            st.seen(f"interactionStatus:{status}")
        if sc.get("turnComplete"):
            st.seen("turnComplete")
            st.done_reason = "turn_complete"
            return status != "IN_PROGRESS"
        if status == "IDLE" and st.event_counts.get("turnComplete"):
            return True
    tc = msg.get("toolCall")
    if isinstance(tc, dict):
        if st.event_counts.get("turnComplete"):
            st.seen("toolCall:after_turnComplete")
        for fc in tc.get("functionCalls") or []:
            st.add_call(fc.get("name", ""), fc.get("args") or {}, fc.get("id"))
        st.done_reason = "tool_call"
        return True
    return False


def gateway_tool(tool: Any) -> dict[str, Any]:
    """A ``RealtimeModelV4ToolDefinition``. ``parameters`` is REQUIRED there:
    a no-argument tool (``escalate_to_human``) sent without it made the gateway
    close a Gemini session silently (live, Sep 25), so it gets an empty object."""
    decl = {"type": "function", **openai_tool_decl(tool_decl(tool))}
    decl.setdefault("parameters", {"type": "object", "properties": {}})
    return decl


def gateway_error(msg: dict[str, Any]) -> RealtimeError:
    raw_any = msg.get("raw")
    raw: dict[str, Any] = raw_any if isinstance(raw_any, dict) else {}
    err_any = raw.get("error")
    err: dict[str, Any] = err_any if isinstance(err_any, dict) else {}
    code = str(msg.get("code") or err.get("code") or "")
    text = str(msg.get("message") or err.get("message") or msg)[:300]
    if any(h in text.lower() for h in QUOTA_HINTS) or code in ("rate_limit_exceeded",):
        return RealtimeQuota(f"{code}: {text}")
    return RealtimeError(f"{code}: {text}" if code else text)


def credit_balance(key: str) -> dict[str, str]:
    """``GET /v1/credits`` -> {"balance": "...", "total_used": "..."} (USD strings).
    Diffing it around a run is the only per-arm cost measure the gateway gives
    realtime sessions (no per-response cost field was observed on the socket)."""
    import httpx

    r = httpx.get(
        f"{GATEWAY_HTTP}/v1/credits", headers={"Authorization": f"Bearer {key}"}, timeout=20
    )
    r.raise_for_status()
    data = r.json()
    return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}
