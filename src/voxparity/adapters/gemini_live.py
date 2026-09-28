"""Gemini Live committed-turn driver — LIVE-TESTED (free tier, Sep 2026).

Turn control (https://ai.google.dev/gemini-api/docs/live-guide, updated
2026-09-04): ``realtimeInputConfig.automaticActivityDetection.disabled = true``
in setup, then ``realtimeInput.activityStart`` -> audio chunks ->
``realtimeInput.activityEnd``. The server echoes ``voiceActivity`` with the audio
offset it received up to, which we record as a truncation check.

Wire format (https://ai.google.dev/api/live, updated 2026-09-04): one JSON
object per frame; audio ``audio/pcm;rate=16000`` in, 24 kHz PCM out. Native
audio models allow only the AUDIO response modality, so the text we score for
probes is ``outputAudioTranscription``.

Tools (https://ai.google.dev/gemini-api/docs/live-tools): on
``gemini-3.1-flash-live-preview`` function calling is synchronous — "the model
will not start responding until you've sent the tool response" — so the turn is
complete at the first ``toolCall`` message and we close without replying.
Consequence, verified live: tool-call cells carry no ``usageMetadata`` (it only
arrives with ``turnComplete``); cost for those cells is estimated from
``audio_sent_s`` x the documented 25 audio tokens/s, never from usage.

Text input: 3.1 accepts ``realtimeInput.text`` during the conversation
(``clientContent`` is only for seeding history there); 2.5 native-audio models
take ``clientContent`` turns. Verified live on 3.1 with VAD disabled.

Asynchronous (NON_BLOCKING) function calling (live-guide, fetched 2026-09-25):
3.8 Live defaults to NON_BLOCKING and 3.8 Live Extended Thinking supports ONLY
NON_BLOCKING, tracked by ``interaction_status`` (``IN_PROGRESS`` vs ``IDLE``).
Measured 2026-09-25 on extended thinking: the model speaks a filler, sends
``turnComplete`` with ``interactionStatus: IN_PROGRESS``, and only THEN sends the
``toolCall``. Closing the turn at that ``turnComplete`` scored every cell as
no-call — a harness artifact, not conduct. So: a ``turnComplete`` marked
IN_PROGRESS never ends the turn (we wait for the toolCall or IDLE), and an
optional grace window (``VOXPARITY_GEMINI_LIVE_LATE_TOOL_S``, default 0 = the
frozen behaviour) keeps listening after an unmarked ``turnComplete`` when tools
were offered and none was called. Late calls are counted in
``realtime.events`` (``toolCall:after_turnComplete``).

Optional setup knobs, recorded on every row under ``realtime.gemini_live``:
``VOXPARITY_GEMINI_LIVE_AFFECTIVE=1`` -> ``enableAffectiveDialog`` (native-audio
models only; "not supported in Gemini 3.1 Flash Live") and
``VOXPARITY_GEMINI_LIVE_THINKING_LEVEL`` -> ``thinkingConfig.thinkingLevel``
(required by 3.8 Live Extended Thinking).

SPEND RULE: by default this driver never uses ``GEMINI_PAID_KEY`` — Live runs
on the free keys only; when every free key is exhausted the cell errors and a
later re-run resumes it. The one exception is the paid final-matrix policy
(D025/PX-003: free-tier inputs are used for training, so frozen-bank runs must
NOT touch a free key): setting ``VOXPARITY_GEMINI_LIVE_ALLOW_PAID=1`` opts in
to the full pool — combined with ``GEMINI_API_KEYS=`` / ``GEMINI_API_KEY=``
(both empty) that pool holds ONLY the paid key, which is the provable
paid-only configuration.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from voxparity.adapters.base import SessionContext
from voxparity.adapters.gemini_file import tool_decl
from voxparity.adapters.realtime import (
    RealtimeDriver,
    RealtimeError,
    RealtimeQuota,
    TurnState,
    WebSocketLike,
    b64,
    classify,
)
from voxparity.providers.keys import KeyPool

DEFAULT_MODEL = os.environ.get("VOXPARITY_GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


def free_gemini_pool() -> KeyPool:
    pool = KeyPool("GEMINI")
    free = [k for i, k in enumerate(pool.keys) if i != pool.paid_idx]
    if not free:
        raise RealtimeError(
            "no free GEMINI keys; the paid key is excluded from Live runs "
            "(set VOXPARITY_GEMINI_LIVE_ALLOW_PAID=1 for a paid-only run)"
        )
    return KeyPool("GEMINI_LIVE_FREE", keys=free)


def gemini_live_pool() -> KeyPool:
    """The SPEND RULE gate: free keys only, unless the paid final-matrix opt-in
    (``VOXPARITY_GEMINI_LIVE_ALLOW_PAID=1``) is set, in which case the full
    GEMINI pool is used (paid key still last, so any free keys present are
    exhausted first)."""
    if os.environ.get("VOXPARITY_GEMINI_LIVE_ALLOW_PAID", "") == "1":
        return KeyPool("GEMINI")
    return free_gemini_pool()


class GeminiLiveDriver(RealtimeDriver):
    provider = "gemini-live"
    live_tested = True
    input_rate = 16000
    output_rate = 24000
    supports_text_input = True
    turn_control_doc = "https://ai.google.dev/gemini-api/docs/live-guide"

    def __init__(
        self,
        model: str | None = None,
        pool: KeyPool | None = None,
        late_tool_grace_s: float | None = None,
        affective_dialog: bool | None = None,
        thinking_level: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(model or DEFAULT_MODEL, **kw)
        self.pool = pool if pool is not None else gemini_live_pool()
        env = os.environ.get
        self.late_tool_grace_s = (
            late_tool_grace_s
            if late_tool_grace_s is not None
            else float(env("VOXPARITY_GEMINI_LIVE_LATE_TOOL_S", "0") or 0)
        )
        self.affective_dialog = (
            affective_dialog
            if affective_dialog is not None
            else env("VOXPARITY_GEMINI_LIVE_AFFECTIVE", "") == "1"
        )
        self.thinking_level = thinking_level or env("VOXPARITY_GEMINI_LIVE_THINKING_LEVEL") or None
        self._tools_offered = False
        self._grace_state: TurnState | None = None

    def url(self) -> str:
        return WS_URL

    def headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.pool.current}

    def rotate_key(self) -> bool:
        return self.pool.rotate()

    def key_index(self) -> int | None:
        return self.pool.idx

    @property
    def text_via_client_content(self) -> bool:
        return "2.5" in self.model or "native-audio" in self.model

    def setup_message(self, ctx: SessionContext) -> dict[str, Any]:
        setup: dict[str, Any] = {
            "model": f"models/{self.model}",
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "systemInstruction": {"parts": [{"text": ctx.system_prompt}]},
            "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}},
            "outputAudioTranscription": {},
            "inputAudioTranscription": {},
        }
        if ctx.tools:
            setup["tools"] = [{"functionDeclarations": [tool_decl(t) for t in ctx.tools]}]
        if self.affective_dialog:
            setup["enableAffectiveDialog"] = True
        if self.thinking_level:
            setup["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        self._tools_offered = bool(ctx.tools)
        self._grace_state = None  # new turn: no grace window carried over
        return {"setup": setup}

    async def recv_json(self, ws: WebSocketLike, deadline: float) -> dict[str, Any]:
        grace = getattr(self, "_grace_state", None)
        if grace is not None and grace.grace_until is not None:
            remaining = min(deadline, grace.grace_until) - time.perf_counter()
            if remaining <= 0:
                return {"_grace_expired": True}
            try:
                return await super().recv_json(ws, time.perf_counter() + remaining)
            except Exception:
                if time.perf_counter() >= grace.grace_until - 0.01:
                    return {"_grace_expired": True}
                raise
        return await super().recv_json(ws, deadline)

    def result(self, st: TurnState, ctx: SessionContext, audio_info: dict[str, Any], attempts: int):  # type: ignore[no-untyped-def]
        r = super().result(st, ctx, audio_info, attempts)
        r.raw["realtime"]["gemini_live"] = {
            "late_tool_grace_s": self.late_tool_grace_s,
            "affective_dialog": self.affective_dialog,
            "thinking_level": self.thinking_level,
        }
        return r

    async def setup(self, ws: WebSocketLike, ctx: SessionContext, st: TurnState) -> None:
        await ws.send(_dumps(self.setup_message(ctx)))
        deadline = time.perf_counter() + self.connect_timeout_s
        while True:
            msg = await self.recv_json(ws, deadline)
            if "setupComplete" in msg:
                return
            if "error" in msg:
                raise _error(msg["error"])

    async def send_audio_turn(self, ws: WebSocketLike, pcm: bytes, st: TurnState) -> None:
        await ws.send(_dumps({"realtimeInput": {"activityStart": {}}}))
        mime = f"audio/pcm;rate={self.input_rate}"
        for chunk in self.chunks(pcm):
            await ws.send(
                _dumps({"realtimeInput": {"audio": {"data": b64(chunk), "mimeType": mime}}})
            )
            await self.pace_sleep(chunk)
        await ws.send(_dumps({"realtimeInput": {"activityEnd": {}}}))
        st.t_commit = time.perf_counter()

    async def send_text_turn(self, ws: WebSocketLike, text: str, st: TurnState) -> None:
        if self.text_via_client_content:
            msg: dict[str, Any] = {
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                    "turnComplete": True,
                }
            }
        else:
            msg = {"realtimeInput": {"text": text}}
        await ws.send(_dumps(msg))
        st.t_commit = time.perf_counter()

    def handle(self, msg: dict[str, Any], st: TurnState) -> bool:
        self._grace_state = st
        if msg.get("_grace_expired"):
            st.seen("grace_expired")
            st.grace_until = None
            return True
        if "error" in msg:
            raise _error(msg["error"])
        if "usageMetadata" in msg:
            st.usage = msg["usageMetadata"]
        if "goAway" in msg:
            st.seen("goAway")
        if "toolCallCancellation" in msg:
            st.seen("toolCallCancellation")
        if "voiceActivity" in msg:
            va = msg["voiceActivity"] or {}
            st.seen(f"voiceActivity:{va.get('type', '')}")
            if va.get("type") == "ACTIVITY_END":
                st.server_audio_offset_s = _duration(va.get("audioOffset"))
                st.committed_ack = True
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
                if status == "IN_PROGRESS":
                    # NON_BLOCKING: the tool call (if any) is still coming.
                    return False
                if self._tools_offered and not st.calls and self.late_tool_grace_s > 0:
                    if st.grace_until is None:
                        st.grace_until = time.perf_counter() + self.late_tool_grace_s
                    return False
                return True
            if status == "IDLE" and st.event_counts.get("turnComplete"):
                return True
        tc = msg.get("toolCall")
        if isinstance(tc, dict):
            if st.event_counts.get("turnComplete"):
                st.seen("toolCall:after_turnComplete")
            st.grace_until = None
            for fc in tc.get("functionCalls") or []:
                st.add_call(fc.get("name", ""), fc.get("args") or {}, fc.get("id"))
            # 3.1 blocks on the tool response: nothing more arrives for this turn.
            st.done_reason = "tool_call"
            return True
        return False


def _dumps(obj: Any) -> str:
    return json.dumps(obj)


def _duration(value: Any) -> float | None:
    if isinstance(value, str) and value.endswith("s"):
        try:
            return float(value[:-1])
        except ValueError:
            return None
    return None


def _error(err: Any) -> RealtimeError:
    text = str(err)
    e = classify(RuntimeError(text))
    return e if isinstance(e, RealtimeQuota) else RealtimeError(text[:300])
