"""Vercel AI Gateway realtime driver, reasoning-effort plumbing and the Azure
route, against mock sockets (no network).

Server transcripts follow shapes observed live on Sep 25 2026 through
``wss://ai-gateway.vercel.sh/v4/ai/realtime-model`` (normalized events whose
``raw`` is the provider's native event).
"""

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from test_realtime import FakeWS, attach, ctx_audio, wav_bytes
from test_schemas import make_item
from voxparity.adapters.base import SessionContext
from voxparity.adapters.openai_realtime import (
    AzureOpenAIRealtimeDriver,
    OpenAIRealtimeDriver,
    QwenOmniRealtimeDriver,
)
from voxparity.adapters.realtime import EFFORT_ENV, RealtimeError, RealtimeQuota, split_effort
from voxparity.adapters.registry import realtime_driver
from voxparity.adapters.vercel_gateway import (
    VercelGatewayRealtimeDriver,
    gateway_error,
    gateway_pool,
)
from voxparity.harness.runner import item_tools
from voxparity.providers.keys import KeyPool


@pytest.fixture
def clip(tmp_path: Path) -> str:
    p = tmp_path / "clip.wav"
    p.write_bytes(wav_bytes(seconds=5.0))
    return str(p)


def gw(model: str = "openai/gpt-realtime-2.1", **kw: Any) -> VercelGatewayRealtimeDriver:
    kw.setdefault("pace", 0.0)
    return VercelGatewayRealtimeDriver(model, pool=KeyPool("T", keys=["k"]), backoff_s=0, **kw)


def norm(kind: str, raw: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {"type": kind, **extra, **({"raw": raw} if raw is not None else {})}


def gateway_openai_server(effort: str | None = None, call: bool = True) -> FakeWS:
    """OpenAI upstream behind the gateway: native events ride in ``raw``."""

    def reply(msg: dict[str, Any]) -> list[Any]:
        kind = msg["type"]
        if kind == "session-update":
            session: dict[str, Any] = {"type": "realtime", "tools": msg["config"].get("tools")}
            if effort:
                session["reasoning"] = {"effort": effort}
            return [
                norm("session-created", {"type": "session.created", "session": {}}),
                norm("session-updated", {"type": "session.updated", "session": session}),
            ]
        if kind == "input-audio-commit":
            return [norm("audio-committed", {"type": "input_audio_buffer.committed"})]
        if kind != "response-create":
            return []
        pcm = base64.b64encode(b"\0\0" * 2400).decode()
        out = [
            norm("response-created", {"type": "response.created", "response": {"id": "r"}}),
            norm("audio-delta", {"type": "response.output_audio.delta", "delta": pcm}),
            norm(
                "audio-transcript-delta",
                {"type": "response.output_audio_transcript.delta", "delta": "ok"},
            ),
        ]
        output = []
        if call:
            fc = {"type": "function_call", "call_id": "c1", "name": "a", "arguments": '{"x":"1"}'}
            out.append(
                norm(
                    "function-call-arguments-done",
                    {**fc, "type": "response.function_call_arguments.done"},
                )
            )
            output.append(fc)
        usage = {"input_tokens": 9, "output_tokens": 3}
        out.append(
            norm(
                "response-done",
                {
                    "type": "response.done",
                    "response": {"status": "completed", "output": output, "usage": usage},
                },
            )
        )
        return out

    return FakeWS(reply)


def gateway_google_server(late_tool: bool = True) -> FakeWS:
    """Gemini 3.8 Live upstream: filler speech, IN_PROGRESS turnComplete, THEN
    the NON_BLOCKING toolCall — each native message fanned out into several
    normalized events that share the same ``raw``."""

    def reply(msg: dict[str, Any]) -> list[Any]:
        kind = msg["type"]
        if kind == "session-update":
            return [norm("session-created", {"setupComplete": {}})]
        if kind not in ("input-audio-commit", "conversation-item-create"):
            return []
        pcm = base64.b64encode(b"\0\0" * 2400).decode()
        said = {
            "serverContent": {
                "modelTurn": {"parts": [{"inlineData": {"data": pcm}}]},
                "outputTranscription": {"text": "One moment."},
            }
        }
        done = {"serverContent": {"turnComplete": True, "interactionStatus": "IN_PROGRESS"}}
        out = [
            norm("audio-delta", said),
            norm("audio-transcript-delta", said),
            norm("custom", done),
        ]
        if late_tool:
            tc = {"toolCall": {"functionCalls": [{"id": "f1", "name": "a", "args": {"x": "1"}}]}}
            out.append(norm("function-call-arguments-done", tc))
        else:
            idle = {"serverContent": {"turnComplete": True, "interactionStatus": "IDLE"}}
            out.append(norm("custom", idle))
        return out

    return FakeWS(reply)


# -- reasoning effort plumbing ------------------------------------------------------


def test_split_effort():
    assert split_effort("gpt-realtime-2.1@high") == ("gpt-realtime-2.1", "high")
    assert split_effort("openai/gpt-realtime-2.1") == ("openai/gpt-realtime-2.1", None)
    assert split_effort("m@") == ("m", None)


def test_effort_suffix_names_a_distinct_arm_and_is_sent_and_echoed(clip: str):
    d = OpenAIRealtimeDriver("gpt-realtime-2.1@minimal", pool=KeyPool("T", keys=["k"]))
    assert d.model == "gpt-realtime-2.1" and d.name == "realtime:openai:gpt-realtime-2.1@minimal"
    ws = FakeWS(
        lambda m: (
            [{"type": "session.updated", "session": m["session"]}]
            if m["type"] == "session.update"
            else [{"type": "response.done", "response": {"status": "completed"}}]
            if m["type"] == "response.create"
            else []
        )
    )
    attach(d, [ws])
    r = d.respond(ctx_audio(clip))
    assert ws.sent[0]["session"]["reasoning"] == {"effort": "minimal"}
    assert r.raw["realtime"]["reasoning_effort"] == "minimal"
    assert r.raw["realtime"]["reasoning_effort_echoed"] == "minimal"


def test_unset_effort_is_recorded_as_provider_default(clip: str):
    d = OpenAIRealtimeDriver(pool=KeyPool("T", keys=["k"]), backoff_s=0)
    assert "reasoning" not in d.session_config(SessionContext(system_prompt="s"))
    assert d.reasoning_effort is None and "@" not in d.name


def test_effort_env_applies_only_where_supported(monkeypatch):
    monkeypatch.setenv(EFFORT_ENV, "high")
    assert OpenAIRealtimeDriver(pool=KeyPool("T", keys=["k"])).name.endswith("@high")
    # Qwen has no documented effort field: the env is ignored, not half-applied
    assert QwenOmniRealtimeDriver(pool=KeyPool("T", keys=["k"])).reasoning_effort is None
    # ...and an explicit request is refused rather than silently dropped
    with pytest.raises(RealtimeError, match="no documented reasoning-effort"):
        QwenOmniRealtimeDriver("qwen3.5-omni-flash-realtime@high", pool=KeyPool("T", keys=["k"]))


# -- gateway: OpenAI / xAI upstream --------------------------------------------------


def test_gateway_openai_session_shape_and_tool_call(clip: str):
    d = gw("openai/gpt-realtime-2.1@low")
    ws = gateway_openai_server(effort="low")
    attach(d, [ws])
    r = d.respond(ctx_audio(clip))
    cfg = ws.sent[0]["config"]
    assert ws.sent[0]["type"] == "session-update"
    assert cfg["turnDetection"] == {"type": "disabled"}
    assert cfg["providerOptions"] == {"reasoning": {"effort": "low"}}
    assert cfg["inputAudioFormat"] == {"type": "audio/pcm", "rate": 24000}
    # every tool carries a parameters schema, even a no-argument one
    assert all(t["parameters"]["type"] == "object" for t in cfg["tools"])
    kinds = [m["type"] for m in ws.sent]
    assert kinds[-2:] == ["input-audio-commit", "response-create"]
    # 5 s of 24 kHz audio in 2 s frames: 3 appends (xAI bills per client message)
    assert kinds.count("input-audio-append") == 3
    assert max(len(json.dumps(m)) for m in ws.sent) < 256 * 1024
    assert [(c.tool, c.args) for c in r.tool_calls] == [("a", {"x": "1"})]
    rt = r.raw["realtime"]
    assert rt["serving_path"] == "vercel-ai-gateway" and rt["upstream"] == "openai"
    assert rt["reasoning_effort"] == "low" and rt["reasoning_effort_echoed"] == "low"
    assert rt["fidelity_tier"] == "B" and rt["committed_ack"] is True
    assert r.raw["usage"] == {"input_tokens": 9, "output_tokens": 3}
    assert d.url().endswith("realtime-model?ai-model-id=openai/gpt-realtime-2.1")


def test_gateway_no_tool_turn_and_text_twin():
    d = gw("spacexai/grok-voice-think-fast-2.0")
    ws = gateway_openai_server(call=False)
    attach(d, [ws])
    item = make_item()
    r = d.respond(SessionContext(system_prompt="s", tools=item_tools(item), text_input="hi"))
    assert ws.sent[1] == {
        "type": "conversation-item-create",
        "item": {"type": "text-message", "role": "user", "text": "hi"},
    }
    assert r.tool_calls == [] and r.text == "ok"


def test_gateway_setup_error_fails_fast(clip: str):
    d = gw("openai/gpt-realtime-2.1@bogus")
    bad = FakeWS(
        lambda m: [
            norm(
                "error",
                {"type": "error", "error": {"code": "invalid_value"}},
                message="Invalid value: 'bogus'.",
                code="invalid_value",
            )
        ]
    )
    opened = attach(d, [bad])
    with pytest.raises(RealtimeError, match="bogus"):
        d.respond(ctx_audio(clip))
    assert len(opened) == 1  # not retried


def test_gateway_error_classification():
    assert isinstance(gateway_error({"message": "Insufficient credits"}), RealtimeQuota)
    e = gateway_error({"message": "not supported in realtime mode", "code": "invalid_model"})
    assert type(e) is RealtimeError and "invalid_model" in str(e)


def test_gpt_live_is_refused_up_front():
    with pytest.raises(RealtimeError, match="delegation"):
        gw("openai/gpt-live-1")
    with pytest.raises(ValueError):
        gw("gpt-realtime-2.1")  # gateway ids are creator/model


# -- gateway: Google upstream --------------------------------------------------------


def test_gateway_google_waits_out_in_progress_and_dedupes_raw(clip: str):
    d = gw("google/gemini-3.8-live-extended-thinking@low")
    ws = gateway_google_server()
    attach(d, [ws])
    r = d.respond(ctx_audio(clip))
    cfg = ws.sent[0]["config"]
    assert "turnDetection" not in cfg  # the gateway closes the socket on it
    assert cfg["providerOptions"] == {"google": {"thinkingConfig": {"thinkingLevel": "low"}}}
    assert cfg["inputAudioFormat"]["rate"] == 16000
    kinds = [m["type"] for m in ws.sent]
    assert kinds[-1] == "input-audio-commit" and "response-create" not in kinds
    # the tool call arrived AFTER an IN_PROGRESS turnComplete and still counts
    assert [(c.tool, c.args) for c in r.tool_calls] == [("a", {"x": "1"})]
    # one native message fanned out into two normalized events: folded once
    assert r.text == "One moment."
    rt = r.raw["realtime"]
    assert rt["fidelity_tier"] == "C" and rt["trailing_silence_s"] == 1.0
    assert rt["events"]["toolCall:after_turnComplete"] == 1
    assert d.capabilities.deterministic_turns is False
    # trailing silence is appended: 5 s clip + 1 s at 16 kHz, 100 ms frames
    assert kinds.count("input-audio-append") == 60


def test_gateway_google_idle_without_tool_is_a_clean_no_call(clip: str):
    d = gw("google/gemini-3.8-live")
    attach(d, [gateway_google_server(late_tool=False)])
    r = d.respond(ctx_audio(clip))
    assert r.tool_calls == [] and r.raw["realtime"]["done_reason"] == "turn_complete"


def test_gateway_google_defaults_to_realtime_pacing():
    d = VercelGatewayRealtimeDriver("google/gemini-3.8-live", pool=KeyPool("T", keys=["k"]))
    assert d.pace == 1.0 and d.chunk_ms == 100
    assert gw().pace == 0.0 and gw().chunk_ms == 2000


# -- keys, registry, Azure -----------------------------------------------------------


def test_gateway_pool_falls_back_to_repo_env_name(monkeypatch):
    for v in ("AI_GATEWAY_API_KEY", "AI_GATEWAY_API_KEYS", "AI_GATEWAY_PAID_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("VERCEL_GATEWAY", "vck_a, vck_b")
    assert gateway_pool().keys == ["vck_a", "vck_b"]
    monkeypatch.delenv("VERCEL_GATEWAY")
    with pytest.raises(RealtimeError, match="AI_GATEWAY_API_KEY"):
        gateway_pool()


def test_registry_parses_gateway_and_azure_specs(monkeypatch):
    d = realtime_driver("realtime:vercel:openai/gpt-realtime-2.1@high")
    assert isinstance(d, VercelGatewayRealtimeDriver)
    assert d.model == "openai/gpt-realtime-2.1" and d.reasoning_effort == "high"
    assert d.name == "realtime:vercel:openai/gpt-realtime-2.1@high"
    a = realtime_driver("realtime:azure:my-rt21@minimal")
    assert isinstance(a, AzureOpenAIRealtimeDriver) and a.model == "my-rt21"


def test_azure_url_and_header(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("FOUNDRY_API", "az1")
    monkeypatch.setenv("FOUNDRY_AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/openai/v1")
    d = AzureOpenAIRealtimeDriver("rt21")
    assert d.url() == "wss://res.openai.azure.com/openai/v1/realtime?model=rt21"
    assert d.headers() == {"api-key": "az1"}
