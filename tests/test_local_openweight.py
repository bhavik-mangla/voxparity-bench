"""Local open-weight arms added after the frozen matrix: Phi-4-multimodal (MLX
server, its own <|tool|> format), Ultravox v0.5 (llama.cpp, native Llama-3.x
tools) and Qwen2.5-Omni-7B (llama.cpp, prompted JSON). Each label must pick the
tool channel the model was trained for and its own port, so a missing env var
cannot attribute one model's answers to another."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from _held import needs_files
from voxparity.adapters.base import SessionContext
from voxparity.schemas.item import ToolDef, ToolParam


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.delenv("VOXPARITY_LLAMACPP_URL", raising=False)
    import voxparity.adapters.llamacpp_local as m

    return m


class TestLabelSpecs:
    def test_phi_uses_its_tool_tokens_and_port(self, mod):
        drv = mod.LlamaCppDriver("phi-4-multimodal-instruct")
        assert drv.tool_channel == "phi_tool_tokens"
        assert drv.native_tools is False  # no OpenAI `tools` array is sent
        assert drv.capabilities.native_tools is True  # but it is the model's own format
        assert drv.base_url.endswith(":8804")

    def test_ultravox_gets_native_llama_tools(self, mod):
        drv = mod.LlamaCppDriver("ultravox-v0_5-llama-3_1-8b")
        assert drv.tool_channel == "native" and drv.native_tools is True
        assert drv.base_url.endswith(":8805")

    def test_qwen25_omni_is_prompted_json(self, mod):
        drv = mod.LlamaCppDriver("qwen2.5-omni-7b-q4")
        assert drv.tool_channel == "prompt_json"
        assert drv.base_url.endswith(":8806")

    def test_existing_labels_unchanged(self, mod):
        q = mod.LlamaCppDriver("qwen3-omni-30b-a3b-q4")
        g = mod.LlamaCppDriver("gemma-4-12b-q4")
        assert (q.tool_channel, q.base_url) == ("prompt_json", mod.BASE_URL)
        assert (g.tool_channel, g.base_url) == ("native", mod.BASE_URL)

    def test_env_url_still_overrides(self, mod, monkeypatch):
        monkeypatch.setenv("VOXPARITY_LLAMACPP_URL", "http://127.0.0.1:9999")
        assert mod.LlamaCppDriver("phi-4-multimodal-instruct").base_url.endswith(":9999")


class TestPhiToolParse:
    def test_tagged_list(self, mod):
        calls = mod.parse_phi_toolcall(
            '<|tool_call|>[{"name": "book", "arguments": {"slot": "t7"}}]<|/tool_call|>'
        )
        assert calls[0].tool == "book" and calls[0].args == {"slot": "t7"}
        assert calls[0].channel == "phi_tool_tokens"

    def test_tags_stripped_by_detokenizer(self, mod):
        calls = mod.parse_phi_toolcall('[{"name": "a", "parameters": {"x": 1}}]')
        assert calls[0].tool == "a" and calls[0].args == {"x": 1}

    def test_openai_shape_and_string_arguments(self, mod):
        calls = mod.parse_phi_toolcall(
            'Sure. {"type": "function", "function": {"name": "a", "arguments": "{\\"x\\": 2}"}}'
        )
        assert calls[0].tool == "a" and calls[0].args == {"x": 2}

    def test_first_call_only(self, mod):
        calls = mod.parse_phi_toolcall('[{"name": "a", "arguments": {}}, {"name": "b"}]')
        assert [c.tool for c in calls] == ["a"]

    def test_plain_text_and_junk_yield_nothing(self, mod):
        assert mod.parse_phi_toolcall("I'll help you with that.") == []
        assert mod.parse_phi_toolcall('{"x": 1}') == []
        assert mod.parse_phi_toolcall('[{"name": "a", "arguments": "not json"}]') == []
        assert mod.parse_phi_toolcall("[broken") == []


def test_phi_request_embeds_tools_in_system_turn(mod, monkeypatch):
    sent: dict = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '[{"name":"book","arguments":{}}]'}}]}

    monkeypatch.setattr(
        mod.httpx,
        "post",
        lambda url, json=None, timeout=None: (sent.update(json, url=url), FakeResp())[1],
    )
    tools = [ToolDef(name="book", description="d", params=[ToolParam(name="slot", type="string")])]
    drv = mod.LlamaCppDriver("phi-4-multimodal-instruct")
    result = drv.respond(SessionContext(system_prompt="sys", tools=tools, text_input="hello"))

    system = sent["messages"][0]["content"]
    assert system.startswith("sys<|tool|>[") and system.endswith("]<|/tool|>")
    assert '"parameters": {"slot"' in system
    assert "tools" not in sent  # the server rejects an OpenAI tools array
    assert "reply with ONLY a JSON object" not in system
    assert sent["url"].startswith("http://127.0.0.1:8804/")
    assert result.tool_calls[0].tool == "book"
    assert result.raw["tool_channel"] == "phi_tool_tokens"


def test_phi_probe_turn_has_no_tool_block(mod, monkeypatch):
    sent: dict = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "Urgent"}}]}

    monkeypatch.setattr(
        mod.httpx, "post", lambda url, json=None, timeout=None: (sent.update(json), FakeResp())[1]
    )
    drv = mod.LlamaCppDriver("phi-4-multimodal-instruct")
    result = drv.respond(SessionContext(system_prompt="sys", text_input="q"))
    assert sent["messages"][0]["content"] == "sys"
    assert result.tool_calls == [] and result.text == "Urgent"


@needs_files("scripts/serve_phi4mm.py", why="local serving script")
def test_phi_server_renders_model_card_format(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        serve = importlib.import_module("serve_phi4mm")
    finally:
        sys.path.remove(str(scripts))
    import base64

    wav = base64.b64encode(b"RIFF0000WAVE").decode()
    prompt, audios = serve.render(
        [
            {"role": "system", "content": "S<|tool|>[]<|/tool|>"},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": wav}}]},
        ]
    )
    try:
        assert (
            prompt == "<|system|>S<|tool|>[]<|/tool|><|end|><|user|><|audio_1|><|end|><|assistant|>"
        )
        assert len(audios) == 1 and Path(audios[0]).read_bytes() == b"RIFF0000WAVE"
    finally:
        for a in audios:
            Path(a).unlink()
    text_prompt, none = serve.render([{"role": "user", "content": "hi"}])
    assert text_prompt == "<|user|>hi<|end|><|assistant|>" and none == []


class TestNativeParseFallback:
    """llama-server answers HTTP 500 when a native tool call fails its parser.
    The output is deterministic, so the driver must turn it into a measured
    outcome (regenerate unparsed, parse leniently) instead of an error row that
    no resume can clear."""

    def _run(self, mod, monkeypatch, raw_text):
        posts: list[str] = []

        class R:
            def __init__(self, code, payload, text=""):
                self.status_code, self._p, self.text = code, payload, text

            def json(self):
                return self._p

            def raise_for_status(self):
                pass

        def fake_post(url, json=None, timeout=None):
            posts.append(url.rsplit("/", 1)[-1])
            if url.endswith("/chat/completions"):
                msg = "The model produced output that does not match the expected peg-native format"
                return R(500, {}, '{"error":{"message":"' + msg + '"}}')
            if url.endswith("/apply-template"):
                assert json["tools"][0]["function"]["name"] == "book"
                return R(200, {"prompt": "P"})
            assert json["prompt"] == "P" and json["temperature"] == 0.0
            return R(200, {"content": raw_text})

        monkeypatch.setattr(mod.httpx, "post", fake_post)
        drv = mod.LlamaCppDriver("ultravox-v0_5-llama-3_1-8b")
        res = drv.respond(
            SessionContext(
                system_prompt="s", tools=[ToolDef(name="book", description="d")], text_input="x"
            )
        )
        return res, posts

    def test_wellformed_call_is_recovered(self, mod, monkeypatch):
        res, posts = self._run(mod, monkeypatch, '{"name": "book", "parameters": {"a": "b"}}')
        assert posts == ["completions", "apply-template", "completion"]
        assert res.tool_calls[0].tool == "book" and res.tool_calls[0].args == {"a": "b"}
        assert res.tool_calls[0].channel == "native_raw_fallback"
        assert res.raw["tool_parse_fallback"] is True and res.raw["valid_tool_call"] is True

    def test_malformed_call_is_no_call_not_an_error(self, mod, monkeypatch):
        bad = '{"name": "book", "parameters": {"q": "what "they" said"}}'
        res, _ = self._run(mod, monkeypatch, bad)
        assert res.tool_calls == [] and res.text == bad
        assert res.raw["valid_tool_call"] is False

    def test_other_500s_still_raise(self, mod, monkeypatch):
        class R:
            status_code, text = 500, "context overflow"

        monkeypatch.setattr(mod.httpx, "post", lambda url, json=None, timeout=None: R())
        drv = mod.LlamaCppDriver("ultravox-v0_5-llama-3_1-8b")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            drv.respond(SessionContext(system_prompt="s", tools=[], text_input="x"))


class TestVoiceChat:
    """NemotronLabs VoiceChat: full duplex, tool calls on a separate FUNCTION
    channel, no text-input path (so no transcript twin, the D035 case)."""

    def test_label_spec(self, mod):
        drv = mod.LlamaCppDriver("nemotron-voicechat-11b-4bit")
        assert drv.tool_channel == "voicechat" and drv.base_url.endswith(":8807")
        assert drv.capabilities.text_twin is False
        assert mod.LlamaCppDriver("gemma-4-12b").capabilities.text_twin is True

    def test_parse_function_channel_then_text_fallback(self, mod):
        fn = '<TOOLCALL>[{"name": "hold", "arguments": {"why": "x"}}]</TOOLCALL>'
        calls = mod.parse_voicechat_toolcall(fn, "Sure, one moment.")
        assert calls[0].tool == "hold" and calls[0].channel == "voicechat_function"
        calls = mod.parse_voicechat_toolcall("", 'ok <TOOLCALL>[{"name": "a", "arguments": {}}]')
        assert calls[0].tool == "a" and calls[0].channel == "voicechat_text"
        assert mod.parse_voicechat_toolcall("", "Just talking.") == []

    def test_request_is_ascii_with_nvidia_tool_block(self, mod, monkeypatch):
        sent: dict = {}
        FN = '<TOOLCALL>[{"name": "book", "arguments": {}}]</TOOLCALL>'

        class FakeResp:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "Placing a hold now.",
                                "function": FN,
                                "user_transcript": "hello",
                            }
                        }
                    ]
                }

        monkeypatch.setattr(
            mod.httpx,
            "post",
            lambda url, json=None, timeout=None: (sent.update(json), FakeResp())[1],
        )
        drv = mod.LlamaCppDriver("nemotron-voicechat-11b-4bit")
        tools = [ToolDef(name="book", description="Book — now", params=[])]
        res = drv.respond(SessionContext(system_prompt="Agent — calm", tools=tools, text_input="x"))
        system = sent["messages"][0]["content"]
        assert system.isascii() and system.startswith("Agent - calm")
        assert "<AVAILABLE_TOOLS>[" in system and '"name": "book"' in system
        assert "tools" not in sent
        assert res.tool_calls[0].tool == "book" and res.text == ""
        assert res.raw["user_transcript"] == "hello"
