"""Committed-turn realtime drivers against mock WebSockets (no network).

Server transcripts below follow the shapes observed live on Gemini Live
(2026-09-14) and documented for OpenAI / xAI / Qwen (docs/REALTIME.md).
"""

import asyncio
import base64
import io
import json
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from test_schemas import make_item
from voxparity.adapters.base import SessionContext
from voxparity.adapters.gemini_live import GeminiLiveDriver, free_gemini_pool
from voxparity.adapters.openai_realtime import (
    GrokVoiceDriver,
    OpenAIRealtimeDriver,
    QwenAudioRealtimeDriver,
    QwenOmniRealtimeDriver,
)
from voxparity.adapters.realtime import RealtimeDriver, RealtimeError, load_pcm16
from voxparity.adapters.registry import realtime_driver
from voxparity.harness.report import load_records, summarize
from voxparity.harness.runner import RunWriter, call_metrics, item_tools, run_item, system_prompt
from voxparity.providers.keys import KeyPool
from voxparity.stimuli.store import StimulusRecord, StimulusStore


def wav_bytes(seconds: float = 0.5, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x10\x00" * int(seconds * rate))
    return buf.getvalue()


class FakeWS:
    """Scripted server: ``reply(msg)`` returns the messages to queue after the
    client sends ``msg``. recv() blocks when the queue is empty (-> timeout)."""

    def __init__(self, reply: Callable[[dict[str, Any]], list[Any]]) -> None:
        self.reply = reply
        self.sent: list[dict[str, Any]] = []
        self.queue: list[Any] = []
        self.closed = False

    async def send(self, message: str) -> None:
        msg = json.loads(message)
        self.sent.append(msg)
        self.queue.extend(self.reply(msg))

    async def recv(self) -> str:
        while not self.queue:
            await asyncio.sleep(0.01)
        item = self.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)

    async def close(self) -> None:
        self.closed = True


def attach(driver: RealtimeDriver, sockets: list[FakeWS]) -> list[FakeWS]:
    opened: list[FakeWS] = []

    async def fake_open() -> FakeWS:
        ws = sockets[len(opened)]
        opened.append(ws)
        return ws

    driver.open = fake_open  # type: ignore[method-assign]
    return opened


def gemini_server(tool: str | None = "a", args: dict[str, Any] | None = None) -> FakeWS:
    def reply(msg: dict[str, Any]) -> list[Any]:
        if "setup" in msg:
            reply.tools = bool(msg["setup"].get("tools"))  # type: ignore[attr-defined]
            return [{"setupComplete": {}}]
        rt = msg.get("realtimeInput", {})
        if "activityStart" in rt:
            return [{"serverContent": {}, "voiceActivity": {"type": "ACTIVITY_START"}}]
        ended = "activityEnd" in rt or "text" in rt or "clientContent" in msg
        if not ended:
            return []
        out: list[Any] = []
        if "activityEnd" in rt:
            out.append({"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": "0.500s"}})
            out.append({"serverContent": {"inputTranscription": {"text": "fine whatever"}}})
        if reply.tools and tool:  # type: ignore[attr-defined]
            out.append(
                {"toolCall": {"functionCalls": [{"id": "fc_1", "name": tool, "args": args or {}}]}}
            )
        else:
            pcm = base64.b64encode(b"\x00\x00" * 2400).decode()
            out += [
                {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": pcm}}]}}},
                {"serverContent": {"outputTranscription": {"text": "happy"}}},
                {
                    "serverContent": {"turnComplete": True},
                    "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 20},
                },
            ]
        return out

    reply.tools = False  # type: ignore[attr-defined]
    return FakeWS(reply)


def gemini(**kw: Any) -> GeminiLiveDriver:
    return GeminiLiveDriver(pool=KeyPool("T", keys=["free1", "free2"]), backoff_s=0, **kw)


@pytest.fixture
def clip(tmp_path: Path) -> str:
    p = tmp_path / "clip.wav"
    p.write_bytes(wav_bytes())
    return str(p)


def ctx_audio(clip: str, tools: bool = True) -> SessionContext:
    item = make_item()
    return SessionContext(
        system_prompt=system_prompt(item), tools=item_tools(item) if tools else [], audio_path=clip
    )


# -- Gemini Live ----------------------------------------------------------------


def test_gemini_audio_turn_sequence_and_tool_call(clip: str):
    d = gemini()
    ws = gemini_server("a", {"x": "1"})
    attach(d, [ws])
    r = d.respond(ctx_audio(clip))

    setup = ws.sent[0]["setup"]
    assert setup["realtimeInputConfig"] == {"automaticActivityDetection": {"disabled": True}}
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    names = {f["name"] for f in setup["tools"][0]["functionDeclarations"]}
    assert names == {t.name for t in item_tools(make_item())}
    rt_msgs = [m["realtimeInput"] for m in ws.sent[1:]]
    assert "activityStart" in rt_msgs[0] and "activityEnd" in rt_msgs[-1]
    audio = [m["audio"] for m in rt_msgs[1:-1]]
    assert audio and all(a["mimeType"] == "audio/pcm;rate=16000" for a in audio)
    sent_bytes = sum(len(base64.b64decode(a["data"])) for a in audio)
    assert sent_bytes == 2 * 8000  # 0.5 s at 16 kHz after resampling from 24 kHz

    assert [(c.tool, c.args) for c in r.tool_calls] == [("a", {"x": "1"})]
    rt = r.raw["realtime"]
    assert rt["done_reason"] == "tool_call" and rt["committed_ack"] is True
    assert rt["server_audio_offset_s"] == 0.5 and rt["fidelity_tier"] == "B"
    assert rt["input_transcript"] == "fine whatever" and rt["latency_first_tool_s"] is not None
    assert ws.closed


def test_gemini_no_tool_is_a_clean_outcome_not_an_error(clip: str):
    d = gemini()
    attach(d, [gemini_server(tool=None)])
    r = d.respond(ctx_audio(clip))
    assert r.tool_calls == [] and r.text == "happy"
    assert r.raw["realtime"]["done_reason"] == "turn_complete"
    assert r.raw["realtime"]["response_audio_s"] == pytest.approx(0.1)
    assert r.raw["usage"]["totalTokenCount"] == 20
    m = call_metrics(r, 1.0)
    assert m["acted"] is False and m["realtime"]["provider"] == "gemini-live"


def test_gemini_text_twin_paths():
    item = make_item()
    ctx = SessionContext(system_prompt="s", tools=item_tools(item), text_input=item.transcript)
    d31 = gemini()
    ws = gemini_server("b")
    attach(d31, [ws])
    assert d31.respond(ctx).tool_calls[0].tool == "b"
    assert ws.sent[1] == {"realtimeInput": {"text": item.transcript}}

    d25 = gemini(model="gemini-2.5-flash-native-audio-preview-12-2025")
    ws25 = gemini_server("b")
    attach(d25, [ws25])
    d25.respond(ctx)
    assert ws25.sent[1]["clientContent"]["turnComplete"] is True


def scripted_gemini(after_end: list[Any]) -> FakeWS:
    """Gemini server that answers the end of the user turn with ``after_end``."""

    def reply(msg: dict[str, Any]) -> list[Any]:
        if "setup" in msg:
            return [{"setupComplete": {}}]
        rt = msg.get("realtimeInput", {})
        if "activityEnd" in rt or "text" in rt or "clientContent" in msg:
            return list(after_end)
        return []

    return FakeWS(reply)


def _filler() -> dict[str, Any]:
    return {"serverContent": {"outputTranscription": {"text": "Let me get that for you."}}}


def test_gemini_nonblocking_in_progress_turn_complete_waits_for_late_tool_call(clip: str):
    """Measured on gemini-3.8-live-extended-thinking (2026-09-25): filler speech,
    turnComplete marked IN_PROGRESS, THEN the toolCall. Closing at that
    turnComplete scored every cell as no-call."""
    d = gemini(model="gemini-3.8-live-extended-thinking")
    ws = scripted_gemini(
        [
            _filler(),
            {"serverContent": {"turnComplete": True, "interactionStatus": "IN_PROGRESS"}},
            {"toolCall": {"functionCalls": [{"id": "fc_9", "name": "a", "args": {"x": "1"}}]}},
        ]
    )
    attach(d, [ws])
    r = d.respond(ctx_audio(clip))
    assert [(c.tool, c.args) for c in r.tool_calls] == [("a", {"x": "1"})]
    rt = r.raw["realtime"]
    assert rt["done_reason"] == "tool_call"
    assert rt["events"]["toolCall:after_turnComplete"] == 1
    assert rt["events"]["interactionStatus:IN_PROGRESS"] == 1


def test_gemini_nonblocking_idle_without_call_is_a_clean_no_call(clip: str):
    d = gemini()
    ws = scripted_gemini(
        [
            _filler(),
            {"serverContent": {"turnComplete": True, "interactionStatus": "IN_PROGRESS"}},
            {"serverContent": {"interactionStatus": "IDLE"}},
        ]
    )
    attach(d, [ws])
    r = d.respond(ctx_audio(clip))
    assert r.tool_calls == [] and r.raw["realtime"]["done_reason"] == "turn_complete"


def test_gemini_late_tool_grace_window(clip: str):
    """Unmarked turnComplete, tool call after it: caught only with a grace
    window; the default (0) keeps the frozen behaviour byte-for-byte."""
    script = [
        _filler(),
        {"serverContent": {"turnComplete": True}},
        {"toolCall": {"functionCalls": [{"id": "fc_2", "name": "a", "args": {}}]}},
    ]
    frozen = gemini()
    attach(frozen, [scripted_gemini(script)])
    assert frozen.respond(ctx_audio(clip)).tool_calls == []

    d = gemini(late_tool_grace_s=2.0)
    attach(d, [scripted_gemini(script)])
    r = d.respond(ctx_audio(clip))
    assert [c.tool for c in r.tool_calls] == ["a"]
    assert r.raw["realtime"]["gemini_live"]["late_tool_grace_s"] == 2.0

    quiet = gemini(late_tool_grace_s=0.2)
    attach(quiet, [scripted_gemini(script[:2])])
    r2 = quiet.respond(ctx_audio(clip))
    assert r2.tool_calls == [] and r2.raw["realtime"]["events"]["grace_expired"] == 1


def test_gemini_setup_knobs_affective_and_thinking(clip: str):
    d = gemini(
        model="gemini-2.5-flash-native-audio-latest", affective_dialog=True, thinking_level="low"
    )
    ws = gemini_server("a")
    attach(d, [ws])
    r = d.respond(ctx_audio(clip))
    setup = ws.sent[0]["setup"]
    assert setup["enableAffectiveDialog"] is True
    assert setup["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert r.raw["realtime"]["gemini_live"] == {
        "late_tool_grace_s": 0.0,
        "affective_dialog": True,
        "thinking_level": "low",
    }
    plain = gemini()
    ws2 = gemini_server("a")
    attach(plain, [ws2])
    plain.respond(ctx_audio(clip))
    assert "enableAffectiveDialog" not in ws2.sent[0]["setup"]


def test_gemini_live_never_uses_the_paid_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "f1,f2")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_PAID_KEY", "paid")
    pool = free_gemini_pool()
    assert pool.keys == ["f1", "f2"] and pool.paid_idx is None


def test_gemini_live_paid_only_needs_the_explicit_opt_in(monkeypatch):
    """The D025/PX-003 paid-run shape: free key vars empty, only GEMINI_PAID_KEY
    set. Without the opt-in the driver refuses (SPEND RULE); with
    VOXPARITY_GEMINI_LIVE_ALLOW_PAID=1 the pool holds ONLY the paid key."""
    from voxparity.adapters.gemini_live import gemini_live_pool
    from voxparity.adapters.realtime import RealtimeError

    monkeypatch.setenv("GEMINI_API_KEYS", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_PAID_KEY", "paid")
    monkeypatch.delenv("VOXPARITY_GEMINI_LIVE_ALLOW_PAID", raising=False)
    with pytest.raises(RealtimeError, match="paid key is excluded"):
        gemini_live_pool()
    monkeypatch.setenv("VOXPARITY_GEMINI_LIVE_ALLOW_PAID", "1")
    pool = gemini_live_pool()
    assert pool.keys == ["paid"] and pool.paid_idx == 0
    drv = GeminiLiveDriver("gemini-3.8-live", pool=pool)
    assert drv.model == "gemini-3.8-live" and drv.pool.keys == ["paid"]


def test_transient_disconnect_is_retried(clip: str):
    d = gemini()
    dropped = gemini_server()
    dropped.reply = lambda m: (  # type: ignore[method-assign]
        [{"setupComplete": {}}]
        if "setup" in m
        else (
            [ConnectionClosedError(Close(1011, "internal"), None)]
            if "activityEnd" in m.get("realtimeInput", {})
            else []
        )
    )
    opened = attach(d, [dropped, gemini_server("a", {"x": "1"})])
    r = d.respond(ctx_audio(clip))
    assert len(opened) == 2 and r.raw["realtime"]["attempts"] == 2


def test_policy_close_is_fatal_not_retried(clip: str):
    d = gemini()
    bad = FakeWS(lambda m: [ConnectionClosedError(Close(1008, "models/x is not found"), None)])
    opened = attach(d, [bad, gemini_server()])
    with pytest.raises(RealtimeError, match="not found"):
        d.respond(ctx_audio(clip))
    assert len(opened) == 1


def test_quota_close_rotates_to_next_free_key(clip: str):
    d = gemini()
    exhausted = FakeWS(
        lambda m: [ConnectionClosedError(Close(1011, "You exceeded your current quota"), None)]
    )
    opened = attach(d, [exhausted, gemini_server("a", {"x": "1"})])
    r = d.respond(ctx_audio(clip))
    assert len(opened) == 2 and d.pool.idx == 1 and r.raw["realtime"]["key_index"] == 1


def test_turn_that_never_completes_is_an_error(clip: str):
    d = gemini(turn_timeout_s=0.05, max_attempts=2)
    silent = [FakeWS(lambda m: [{"setupComplete": {}}] if "setup" in m else []) for _ in range(2)]
    attach(d, silent)
    with pytest.raises(RealtimeError, match="did not complete"):
        d.respond(ctx_audio(clip))


# -- OpenAI dialect -------------------------------------------------------------


def openai_server(call: tuple[str, str] | None = ("a", '{"x": "1"}'), vad: bool = False) -> FakeWS:
    def reply(msg: dict[str, Any]) -> list[Any]:
        kind = msg["type"]
        if kind == "session.update":
            return [{"type": "session.updated", "session": msg["session"]}]
        if kind == "input_audio_buffer.commit":
            out: list[Any] = [{"type": "input_audio_buffer.committed", "item_id": "i1"}]
            if vad:
                out.insert(0, {"type": "input_audio_buffer.speech_started"})
            return out
        if kind != "response.create":
            return []
        out = [{"type": "response.created", "response": {"id": "r1"}}]
        output: list[dict[str, Any]] = []
        if call:
            fc = {"type": "function_call", "call_id": "c1", "name": call[0], "arguments": call[1]}
            out.append({"type": "response.function_call_arguments.done", **fc})
            out.append({"type": "response.output_item.done", "item": fc})
            output.append(fc)
        else:
            out.append(
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(b"\0\0" * 2400).decode(),
                }
            )
            out.append(
                {"type": "response.audio.delta", "delta": base64.b64encode(b"\0\0" * 2400).decode()}
            )
            out.append({"type": "response.output_audio_transcript.delta", "delta": "hap"})
            out.append({"type": "response.audio_transcript.delta", "delta": "py"})
        usage = {"input_tokens": 50, "output_tokens": 5}
        out.append(
            {
                "type": "response.done",
                "response": {"status": "completed", "output": output, "usage": usage},
            }
        )
        return out

    return FakeWS(reply)


def test_openai_session_shape_commit_order_and_dedupe(clip: str):
    d = OpenAIRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    ws = openai_server()
    attach(d, [ws])
    r = d.respond(ctx_audio(clip))
    session = ws.sent[0]["session"]
    assert session["type"] == "realtime"
    assert session["audio"]["input"]["turn_detection"] is None
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["tools"][0]["type"] == "function" and "name" in session["tools"][0]
    kinds = [m["type"] for m in ws.sent]
    assert kinds[-2:] == ["input_audio_buffer.commit", "response.create"]
    assert set(kinds[1:-2]) == {"input_audio_buffer.append"}
    # the same call arrives three ways; it must be recorded once
    assert [(c.tool, c.args) for c in r.tool_calls] == [("a", {"x": "1"})]
    assert r.raw["usage"] == {"input_tokens": 50, "output_tokens": 5}
    assert r.raw["realtime"]["committed_ack"] is True
    assert r.raw["realtime"]["resampler"] == "none"


def test_openai_no_tool_collects_audio_and_transcript(clip: str):
    d = OpenAIRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    attach(d, [openai_server(call=None)])
    r = d.respond(ctx_audio(clip))
    assert r.tool_calls == [] and r.text == "happy"
    assert r.raw["realtime"]["response_audio_s"] == pytest.approx(0.2)


def test_server_vad_on_a_replay_is_flagged(clip: str):
    d = OpenAIRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    attach(d, [openai_server(vad=True)])
    assert d.respond(ctx_audio(clip)).raw["realtime"]["server_vad_events"] == 1


def test_openai_text_twin_sends_input_text_item():
    d = OpenAIRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    ws = openai_server(call=("b", "{}"))
    attach(d, [ws])
    item = make_item()
    r = d.respond(SessionContext(system_prompt="s", tools=item_tools(item), text_input="hi"))
    assert ws.sent[1]["item"]["content"] == [{"type": "input_text", "text": "hi"}]
    assert r.tool_calls[0].tool == "b"


def test_openai_error_events(clip: str):
    d = OpenAIRealtimeDriver(pool=KeyPool("T", keys=["k1", "k2"]), backoff_s=0)
    limited = FakeWS(
        lambda m: [{"type": "error", "error": {"code": "rate_limit_exceeded", "message": "slow"}}]
    )
    opened = attach(d, [limited, openai_server()])
    assert d.respond(ctx_audio(clip)).tool_calls and len(opened) == 2 and d.pool.idx == 1

    d2 = OpenAIRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    failed = openai_server()
    base_reply = failed.reply

    def fail(m: dict[str, Any]) -> list[Any]:
        if m["type"] == "response.create":
            err = {"error": {"code": "server_error", "message": "boom"}}
            return [
                {"type": "response.done", "response": {"status": "failed", "status_details": err}}
            ]
        return base_reply(m)

    failed.reply = fail  # type: ignore[method-assign]
    attach(d2, [failed])
    with pytest.raises(RealtimeError, match="boom"):
        d2.respond(ctx_audio(clip))


def test_grok_session_shape(clip: str):
    d = GrokVoiceDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    ws = openai_server()
    attach(d, [ws])
    d.respond(ctx_audio(clip))
    s = ws.sent[0]["session"]
    assert s["turn_detection"] == {"type": None}
    assert s["audio"]["input"]["format"]["rate"] == 24000
    assert "name" in s["tools"][0]
    assert d.url().startswith("wss://api.x.ai/v1/realtime?model=grok-voice")


def test_qwen_nested_tools_and_no_text_twin(clip: str, tmp_path: Path):
    d = QwenOmniRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    ws = openai_server()
    attach(d, [ws])
    d.respond(ctx_audio(clip))
    s = ws.sent[0]["session"]
    assert s["turn_detection"] is None and s["input_audio_format"] == "pcm"
    assert s["tools"][0]["function"]["name"]
    assert d.capabilities.text_twin is False
    with pytest.raises(RealtimeError, match="not applicable"):
        d.respond(SessionContext(system_prompt="s", text_input="hi"))


def test_qwen_audio_voice_required_text_twin_and_done_transcript(clip: str):
    d = QwenAudioRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    assert d.model == "qwen-audio-3.1-realtime-plus"
    assert d.capabilities.text_twin is True
    ws = openai_server()
    attach(d, [ws])
    d.respond(ctx_audio(clip))
    s = ws.sent[0]["session"]
    assert s["voice"] == "beth_v3.1" and s["turn_detection"] is None
    assert s["tools"][0]["function"]["name"]

    def done_only(msg: dict[str, Any]) -> list[Any]:
        kind = msg["type"]
        if kind == "session.update":
            return [{"type": "session.updated"}]
        if kind != "response.create":
            return []
        return [
            {"type": "response.created", "response": {"id": "r1"}},
            {"type": "response.audio_transcript.done", "transcript": "happy"},
            {"type": "response.done", "response": {"status": "completed", "output": []}},
        ]

    d2 = QwenAudioRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    attach(d2, [FakeWS(done_only)])
    r = d2.respond(SessionContext(system_prompt="s", text_input="hi"))
    assert r.text == "happy"


def test_missing_key_names_what_is_needed(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    with pytest.raises(RealtimeError, match="OPENAI_API_KEY"):
        OpenAIRealtimeDriver().headers()


def test_registry_parses_driver_specs(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "f1")
    assert (
        realtime_driver("realtime:gemini-live").name
        == "realtime:gemini-live:gemini-3.1-flash-live-preview"
    )
    assert realtime_driver("realtime:openai:gpt-realtime-2.1-mini").model == "gpt-realtime-2.1-mini"
    assert isinstance(realtime_driver("realtime:qwen"), QwenOmniRealtimeDriver)
    assert isinstance(realtime_driver("realtime:qwen-audio"), QwenAudioRealtimeDriver)
    with pytest.raises(ValueError):
        realtime_driver("realtime:nova")


def test_load_pcm16_resamples_and_reports(clip: str):
    pcm, info = load_pcm16(clip, 16000)
    assert len(pcm) == 16000 and info["audio_sent_s"] == 0.5 and info["resampler"] != "none"


# -- contract: realtime rows flow through runner, scorer and report ----------------


def test_realtime_rows_pass_through_scorer_and_report(tmp_path: Path):
    item = make_item()
    store = StimulusStore(tmp_path / "stimuli")
    for vid in ("happy", "angry"):
        store.put(
            wav_bytes(0.5 + (0.1 if vid == "angry" else 0)),
            StimulusRecord(
                item_id=item.id,
                variant_id=vid,
                sha256="",
                engine="test",
                model="m",
                voice="v",
                prompt="p",
                gates={"cue_check": {"passed": True}},
            ),
        )
    d = gemini()
    attach(d, [gemini_server("a", {"x": "1"}) for _ in range(20)])
    writer = RunWriter(tmp_path / "runs", "rt")
    rows = run_item(d, item, store, "test", writer, "rt")
    assert rows == 5  # twin + 2 x (audio, probe)

    recs = load_records(tmp_path / "runs" / "rt")
    assert not [r for r in recs if r["error"]]
    audio = {r["variant_id"]: r for r in recs if r["condition"] == "audio"}
    assert audio["happy"]["scores"]["passed"] is True
    assert audio["angry"]["scores"]["passed"] is False
    assert audio["happy"]["metrics"]["realtime"]["family"] == "committed_turn"
    probe = {r["variant_id"]: r for r in recs if r["condition"] == "probe"}
    assert probe["happy"]["scores"]["answer"] == "happy"
    assert all(r["driver"] == "realtime:gemini-live:gemini-3.1-flash-live-preview" for r in recs)
    summary = summarize(recs)
    assert summary  # report consumes realtime rows unchanged

    # resume: nothing re-run
    assert run_item(d, item, store, "test", RunWriter(tmp_path / "runs", "rt"), "rt") == 0


def test_qwen_twin_recorded_not_applicable_by_runner(tmp_path: Path):
    item = make_item()
    store = StimulusStore(tmp_path / "stimuli")
    d = QwenOmniRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    writer = RunWriter(tmp_path / "runs", "q")
    run_item(d, item, store, "none", writer, "q")
    twin = next(r for r in load_records(tmp_path / "runs" / "q") if r["condition"] == "text_twin")
    assert twin["scores"]["applicable"] is False and not twin["error"]


def test_openrouter_no_arg_tool_gets_empty_object_schema():
    from voxparity.adapters.openrouter import _with_parameters

    assert _with_parameters({"name": "x", "description": "d"})["parameters"] == {
        "type": "object",
        "properties": {},
    }
    decl = {"name": "y", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}
    assert _with_parameters(decl) is decl
