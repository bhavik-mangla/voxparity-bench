"""Committed-turn realtime driver family (D008): one WebSocket session per cell.

Each ``respond()`` opens a fresh session, configures it with the SAME system
prompt and tool menu the file drivers receive (the runner builds both through
``system_prompt`` / ``item_tools``; tools are converted with the same
``tool_decl`` builder), streams the stimulus as PCM16 at the provider's input
rate with server VAD OFF, commits the turn explicitly, and collects whatever the
model does until the provider signals the turn is over. The result maps into
the ordinary ``TurnResult``/``ToolCall`` shape, so scoring, resume, dedupe and
report are untouched.

Three outcomes are kept distinct, because conflating them is the D046 failure
class (an absence of measurement recorded as a negative measurement):

- the turn completed with a tool call            -> ``TurnResult`` with calls
- the turn completed WITHOUT a tool call         -> ``TurnResult`` with no calls
  (``acted: false`` in metrics — a genuine decision not to act)
- the turn never completed (timeout, disconnect, provider error) -> an exception,
  which the runner records as an error row and a re-run retries (D074).

Transient disconnects are retried inside ``respond`` with backoff; quota
exhaustion rotates the key pool; everything else raises.

Replay-fidelity tier (D025, docs/PLAN.md): every driver here is tier **B** —
explicit WebSocket turn control, no seed. Tier A is REST + seed; tier C is
VAD-bound replay (e.g. Nova Sonic) and must set ``deterministic_turns=False``.

Protocol details were checked against current vendor docs and against
tau2-bench's MIT-licensed ``src/tau2/voice/audio_native`` adapters
(Copyright (c) 2025 Sierra Research); see docs/REALTIME.md for URLs and dates.
No tau2 code is copied verbatim — tau2 drives these APIs with server VAD in a
tick loop, which is the opposite of what a committed-turn replay needs — but
event names, tool formats and provider quirks were cross-checked against it.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import math
import os
import time
import wave
from abc import abstractmethod
from array import array
from dataclasses import dataclass, field
from typing import Any, Protocol

from voxparity.adapters.base import DriverCapabilities, SessionContext, SessionDriver
from voxparity.schemas.result import ToolCall, TurnResult

FIDELITY_TIER = "B"

EFFORT_ENV = "VOXPARITY_REALTIME_REASONING_EFFORT"


def split_effort(model: str) -> tuple[str, str | None]:
    """``gpt-realtime-2.1@high`` -> (``gpt-realtime-2.1``, ``high``).

    The suffix makes each point of a reasoning-effort sweep a distinct driver
    name (and so a distinct arm) while the model id sent upstream stays bare.
    """
    base, sep, effort = model.partition("@")
    return base, (effort.strip() or None) if sep else None


class RealtimeError(RuntimeError):
    """A provider refused or failed the turn; not retryable."""


class RealtimeTransient(RealtimeError):
    """Disconnect / timeout / 5xx: worth retrying the whole cell."""


class RealtimeQuota(RealtimeError):
    """The current key is rate-limited or out of quota: rotate, then retry."""


class WebSocketLike(Protocol):
    async def send(self, message: str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------


def load_pcm16(path: str, rate: int) -> tuple[bytes, dict[str, Any]]:
    """Read a WAV and return (mono little-endian PCM16 at ``rate``, info).

    Prefers scipy's polyphase resampler (proper anti-aliasing); falls back to
    ``audioop.ratecv`` and then to the pure-python path in providers/ser.py. The
    resampler used is recorded on every row, because it is part of what the
    model actually heard.
    """
    from voxparity.providers.ser import _downmix, _pcm_to_int16, _resample

    with contextlib.closing(wave.open(path, "rb")) as w:
        channels, width, src = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    samples = _downmix(_pcm_to_int16(frames, width), channels)
    resampler = "none"
    if src != rate:
        try:
            import numpy as np
            from scipy.signal import resample_poly  # type: ignore[import-untyped]

            g = math.gcd(src, rate)
            y = resample_poly(np.asarray(samples, dtype=np.float64), rate // g, src // g)
            samples = array("h", np.clip(np.rint(y), -32768, 32767).astype("<i2").tobytes())
            resampler = "scipy.resample_poly"
        except ImportError:
            try:
                import audioop  # type: ignore[import-not-found,unused-ignore]

                out, _ = audioop.ratecv(samples.tobytes(), 2, 1, src, rate, None)
                samples = array("h", out)
                resampler = "audioop.ratecv"
            except ImportError:
                samples = _resample(samples, src, rate)
                resampler = "pure-python-linear"
    pcm = samples.tobytes()
    info = {
        "source_rate": src,
        "input_rate": rate,
        "resampler": resampler,
        "audio_sent_s": round(len(pcm) / 2 / rate, 3),
    }
    return pcm, info


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with contextlib.closing(wave.open(buf, "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# --------------------------------------------------------------------------
# per-turn accumulator
# --------------------------------------------------------------------------


@dataclass
class TurnState:
    """Everything observed during one committed turn."""

    t_open: float = field(default_factory=time.perf_counter)
    t_commit: float | None = None
    t_first_event: float | None = None
    t_first_tool: float | None = None
    t_first_audio: float | None = None
    t_done: float | None = None
    text: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    input_transcript: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    call_ids: set[str] = field(default_factory=set)
    audio_out_bytes: int = 0
    usage: dict[str, Any] | None = None
    done_reason: str = ""
    committed_ack: bool | None = None
    early_response: bool = False
    interrupted: bool = False
    server_vad_events: int = 0
    server_audio_offset_s: float | None = None
    reasoning_echo: Any = None  # what the server says it applied (session echo)
    event_counts: dict[str, int] = field(default_factory=dict)
    # Provider-driven wait past an end-of-turn marker (Gemini Live NON_BLOCKING
    # tool calls can arrive after turnComplete). None = no grace window open.
    grace_until: float | None = None

    def seen(self, kind: str) -> None:
        self.event_counts[kind] = self.event_counts.get(kind, 0) + 1

    def mark_response(self) -> None:
        """Any model output. Output before commit = the model barged in."""
        now = time.perf_counter()
        if self.t_commit is None:
            self.early_response = True
        elif self.t_first_event is None:
            self.t_first_event = now

    def add_audio(self, b64: str) -> None:
        self.mark_response()
        if self.t_first_audio is None and self.t_commit is not None:
            self.t_first_audio = time.perf_counter()
        self.audio_out_bytes += len(b64) * 3 // 4 - b64[-2:].count("=")

    def add_call(self, name: str, args: Any, call_id: str | None) -> None:
        if call_id and call_id in self.call_ids:
            return
        if call_id:
            self.call_ids.add(call_id)
        self.mark_response()
        if self.t_first_tool is None:
            self.t_first_tool = time.perf_counter()
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": args}
        if not isinstance(args, dict):
            args = {}
        self.calls.append({"name": name or "", "args": args, "id": call_id})


def _since(t: float | None, t0: float | None) -> float | None:
    return round(t - t0, 3) if (t is not None and t0 is not None) else None


# --------------------------------------------------------------------------
# base driver
# --------------------------------------------------------------------------


class RealtimeDriver(SessionDriver):
    """Committed-turn WebSocket driver. Subclasses implement the dialect."""

    provider: str = ""
    live_tested: bool = False
    input_rate: int = 24000
    output_rate: int = 24000
    supports_text_input: bool = True
    turn_control_doc: str = ""
    fidelity_tier: str = FIDELITY_TIER
    # Drivers that map ``reasoning_effort`` onto a documented session field set
    # this True. Elsewhere an explicit ``@effort`` is refused (it would not be
    # applied) and the env var is ignored, so a global env cannot break them.
    supports_reasoning_effort: bool = False

    def __init__(
        self,
        model: str,
        reasoning_effort: str | None = None,
        pace: float = 0.0,
        chunk_ms: int = 100,
        turn_timeout_s: float = 90.0,
        connect_timeout_s: float = 20.0,
        max_attempts: int = 3,
        backoff_s: float = 3.0,
    ) -> None:
        model, effort = split_effort(model)
        self.model = model
        explicit = reasoning_effort or effort
        env_effort = os.environ.get(EFFORT_ENV, "").strip()
        if not self.supports_reasoning_effort:
            env_effort = ""
        self.reasoning_effort = explicit or env_effort or None
        if self.reasoning_effort and not self.supports_reasoning_effort:
            raise RealtimeError(
                f"{self.provider} realtime has no documented reasoning-effort setting; "
                f"refusing to record effort={self.reasoning_effort!r} that would not be applied"
            )
        suffix = f"@{self.reasoning_effort}" if self.reasoning_effort else ""
        self.name = f"realtime:{self.provider}:{model}{suffix}"
        self.pace = pace  # 0 = as fast as the socket takes it; 1.0 = real time
        self.chunk_ms = chunk_ms
        self.turn_timeout_s = turn_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s

    @property
    def capabilities(self) -> DriverCapabilities:
        return DriverCapabilities(
            family="committed_turn",
            audio_in=True,
            audio_out=True,
            native_tools=True,
            deterministic_turns=self.fidelity_tier != "C",
            text_twin=self.supports_text_input,
        )

    # -- dialect hooks --------------------------------------------------------
    @abstractmethod
    def url(self) -> str: ...

    def headers(self) -> dict[str, str]:
        return {}

    @abstractmethod
    async def setup(self, ws: WebSocketLike, ctx: SessionContext, st: TurnState) -> None:
        """Send session config and wait for the provider's acknowledgement."""

    @abstractmethod
    async def send_audio_turn(self, ws: WebSocketLike, pcm: bytes, st: TurnState) -> None:
        """Stream ``pcm`` and commit the turn (sets ``st.t_commit``)."""

    @abstractmethod
    async def send_text_turn(self, ws: WebSocketLike, text: str, st: TurnState) -> None: ...

    @abstractmethod
    def handle(self, msg: dict[str, Any], st: TurnState) -> bool:
        """Fold one server message into ``st``; True when the turn is over."""

    def rotate_key(self) -> bool:
        """Advance to the next key after a quota error; False when none left."""
        return False

    def key_index(self) -> int | None:
        return None

    # -- transport (overridden in tests) --------------------------------------
    async def open(self) -> Any:
        from websockets.asyncio.client import connect

        return await connect(
            self.url(),
            additional_headers=self.headers(),
            max_size=None,
            open_timeout=self.connect_timeout_s,
        )

    async def recv_json(self, ws: WebSocketLike, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise RealtimeTransient(f"turn did not complete within {self.turn_timeout_s:.0f}s")
        try:
            raw = await asyncio.wait_for(ws.recv(), remaining)
        except TimeoutError as e:
            raise RealtimeTransient(
                f"turn did not complete within {self.turn_timeout_s:.0f}s"
            ) from e
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"_non_object": data}

    def chunks(self, pcm: bytes) -> list[bytes]:
        step = max(2, int(self.input_rate * self.chunk_ms / 1000) * 2)
        return [pcm[i : i + step] for i in range(0, len(pcm), step)]

    async def pace_sleep(self, chunk: bytes) -> None:
        if self.pace > 0:
            await asyncio.sleep(len(chunk) / 2 / self.input_rate * self.pace)

    # -- the turn ---------------------------------------------------------------
    async def _turn(self, ctx: SessionContext, pcm: bytes | None) -> TurnState:
        st = TurnState()
        try:
            ws = await self.open()
        except Exception as e:
            raise classify(e) from e
        try:
            await self.setup(ws, ctx, st)
            if pcm is not None:
                await self.send_audio_turn(ws, pcm, st)
            else:
                await self.send_text_turn(ws, ctx.text_input or "", st)
            deadline = time.perf_counter() + self.turn_timeout_s
            while True:
                msg = await self.recv_json(ws, deadline)
                if self.handle(msg, st):
                    st.t_done = time.perf_counter()
                    break
        except RealtimeError:
            raise
        except Exception as e:
            raise classify(e) from e
        finally:
            with contextlib.suppress(Exception):
                await ws.close()
        return st

    def respond(self, ctx: SessionContext) -> TurnResult:
        if (ctx.text_input is None) == (ctx.audio_path is None):
            raise ValueError("exactly one of text_input / audio_path must be set")
        if ctx.text_input is not None and not self.supports_text_input:
            raise RealtimeError(
                f"{self.provider} realtime sessions accept no text user turns; "
                "the transcript twin is not applicable (D035)"
            )
        pcm: bytes | None = None
        audio_info: dict[str, Any] = {}
        if ctx.audio_path is not None:
            pcm, audio_info = load_pcm16(ctx.audio_path, self.input_rate)

        attempts = 0
        rotations = 0
        last: Exception | None = None
        while attempts < self.max_attempts:
            attempts += 1
            try:
                st = asyncio.run(self._turn(ctx, pcm))
                return self.result(st, ctx, audio_info, attempts)
            except RealtimeQuota as e:
                last = e
                attempts -= 1  # a quota rotation is not a failed attempt
                rotations += 1
                if rotations > 16 or not self.rotate_key():
                    raise RealtimeError(f"quota exhausted on every usable key: {e}") from e
            except RealtimeTransient as e:
                last = e
                if attempts < self.max_attempts:
                    time.sleep(self.backoff_s * 2 ** (attempts - 1))
        raise RealtimeError(f"gave up after {attempts} attempts: {last}")

    def result(
        self, st: TurnState, ctx: SessionContext, audio_info: dict[str, Any], attempts: int
    ) -> TurnResult:
        text = "".join(st.text).strip() or "".join(st.transcript).strip()
        calls = [ToolCall(tool=c["name"], args=c["args"]) for c in st.calls]
        rt: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "family": "committed_turn",
            "fidelity_tier": self.fidelity_tier,
            # Pinned per D012; None = the provider default (not echoed back).
            "reasoning_effort": self.reasoning_effort,
            "live_tested": self.live_tested,
            "modality_in": "audio" if ctx.audio_path is not None else "text",
            "pace": self.pace,
            **audio_info,
            "done_reason": st.done_reason,
            "latency_first_event_s": _since(st.t_first_event, st.t_commit),
            "latency_first_tool_s": _since(st.t_first_tool, st.t_commit),
            "latency_first_audio_s": _since(st.t_first_audio, st.t_commit),
            "latency_done_s": _since(st.t_done, st.t_commit),
            "response_audio_s": round(st.audio_out_bytes / 2 / self.output_rate, 3),
            "committed_ack": st.committed_ack,
            "early_response": st.early_response,
            "interrupted": st.interrupted,
            "server_vad_events": st.server_vad_events,
            "attempts": attempts,
        }
        if st.reasoning_echo is not None:
            rt["reasoning_effort_echoed"] = st.reasoning_echo
        rt.update(self.extra_metrics())
        if st.server_audio_offset_s is not None:
            rt["server_audio_offset_s"] = st.server_audio_offset_s
        if st.input_transcript:
            rt["input_transcript"] = "".join(st.input_transcript).strip()[:500]
        if st.transcript and st.text:
            rt["output_transcript"] = "".join(st.transcript).strip()[:500]
        if self.key_index() is not None:
            rt["key_index"] = self.key_index()
        rt["events"] = st.event_counts
        latency = _since(st.t_first_tool or st.t_done, st.t_commit)
        return TurnResult(
            text=text,
            tool_calls=calls,
            latency_ms=latency * 1000 if latency is not None else None,
            raw={"function_calls": st.calls, "usage": st.usage, "realtime": rt},
        )

    def extra_metrics(self) -> dict[str, Any]:
        """Driver-specific fields for ``metrics.realtime`` (e.g. serving path)."""
        return {}

    def preflight(self) -> None:
        """Open a session and complete setup — proves key, model id and the
        realtime endpoint once per run, with no input sent (no spend)."""
        ctx = SessionContext(system_prompt="preflight", tools=[], text_input="")

        async def go() -> None:
            st = TurnState()
            ws = await self.open()
            try:
                await self.setup(ws, ctx, st)
            finally:
                with contextlib.suppress(Exception):
                    await ws.close()

        rotations = 0
        while True:
            try:
                asyncio.run(go())
                return
            except RealtimeQuota as e:
                rotations += 1
                if rotations > 16 or not self.rotate_key():
                    raise RealtimeError(f"preflight: quota exhausted on every key: {e}") from e
            except RealtimeError as e:
                raise RealtimeError(f"preflight {self.name}: {e}") from e
            except Exception as e:
                raise RealtimeError(f"preflight {self.name}: {classify(e)}") from e


QUOTA_WORDS = ("quota", "resource_exhausted", "resource exhausted", "rate limit", "rate_limit")


def classify(e: BaseException) -> RealtimeError:
    """Map a transport exception onto transient / quota / fatal."""
    if isinstance(e, RealtimeError):
        return e
    text = str(e)
    low = text.lower()
    if any(w in low for w in QUOTA_WORDS):
        return RealtimeQuota(text[:300])
    try:
        from websockets.exceptions import ConnectionClosed, InvalidStatus
    except ImportError:  # pragma: no cover
        ConnectionClosed = InvalidStatus = ()  # type: ignore[assignment,misc]
    if isinstance(e, InvalidStatus):
        status = e.response.status_code
        if status == 429:
            return RealtimeQuota(f"HTTP 429 on connect: {text[:200]}")
        if status >= 500:
            return RealtimeTransient(f"HTTP {status} on connect")
        return RealtimeError(f"HTTP {status} on connect: {text[:200]}")
    if isinstance(e, ConnectionClosed):
        rcvd = e.rcvd
        code = rcvd.code if rcvd is not None else 1006
        reason = rcvd.reason if rcvd is not None else ""
        # 1008 policy violation / 1007 invalid payload: bad model id, bad config,
        # bad key. Retrying the same request cannot change the answer.
        if code in (1007, 1008, 1003) or 4000 <= code < 5000:
            return RealtimeError(f"closed by server ({code}): {reason[:300]}")
        return RealtimeTransient(f"connection closed ({code}): {reason[:300]}")
    if isinstance(e, (OSError, TimeoutError, asyncio.TimeoutError)):
        return RealtimeTransient(f"{type(e).__name__}: {text[:200]}")
    return RealtimeError(f"{type(e).__name__}: {text[:300]}")


def b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode()
