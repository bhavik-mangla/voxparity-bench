"""Contract tests for the StepFun StepAudio 3 chat driver.

The driver reuses the OpenRouter request/retry path pointed at StepFun's
OpenAI-compatible base URL; these tests pin the vendor-specific contract
(data-URI audio part, no OpenRouter headers or routing pin, /models-based
preflight) and guard that the OpenRouter driver's own behavior is untouched.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from voxparity.adapters.base import SessionContext
from voxparity.schemas.item import ToolDef, ToolParam


def _wav(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 2400)
    return p


def _driver(monkeypatch, model: str | None = None):
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    from voxparity.adapters.stepfun import StepFunDriver

    return StepFunDriver(model)


def test_defaults_pin_the_verified_preview_id_and_endpoint(monkeypatch):
    drv = _driver(monkeypatch)
    assert drv.model == "stepaudio-3-chat-preview"
    assert drv.name == "stepfun:stepaudio-3-chat-preview"
    assert drv.base_url == "https://api.stepfun.ai/v1"
    caps = drv.capabilities
    assert caps.audio_in and caps.native_tools and caps.text_twin


def test_sends_data_uri_audio_and_tools_in_one_request(monkeypatch, tmp_path: Path):
    """StepFun's documented input_audio shape is a data URI with no format key."""
    drv = _driver(monkeypatch)
    sent: dict = {}

    def fake_post(body):
        sent.update(body)
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "book", "arguments": '{"slot": "t7"}'}}
                        ],
                    }
                }
            ],
            "usage": {"total_tokens": 7},
        }

    monkeypatch.setattr(drv, "_post", fake_post)
    tools = [
        ToolDef(
            name="book",
            description="d",
            params=[ToolParam(name="slot", type="string", description="d")],
        )
    ]
    result = drv.respond(
        SessionContext(system_prompt="sys", tools=tools, audio_path=str(_wav(tmp_path)))
    )

    part = sent["messages"][1]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["data"].startswith("data:audio/wav;base64,")
    assert "format" not in part["input_audio"]
    assert sent["model"] == "stepaudio-3-chat-preview"
    assert sent["tools"][0]["function"]["name"] == "book"
    assert result.tool_calls[0].tool == "book"
    assert result.tool_calls[0].args == {"slot": "t7"}
    assert result.raw["usage"] == {"total_tokens": 7}


def test_openrouter_provider_pin_never_leaks_into_a_stepfun_request(monkeypatch, tmp_path: Path):
    """VOXPARITY_OPENROUTER_PROVIDER is an OpenRouter routing key; sending it to
    StepFun would be an unknown field at best and a silent misroute at worst."""
    monkeypatch.setenv("VOXPARITY_OPENROUTER_PROVIDER", "Together")
    drv = _driver(monkeypatch)
    sent: dict = {}
    monkeypatch.setattr(
        drv,
        "_post",
        lambda body: (sent.update(body), {"choices": [{"message": {"content": "ok"}}]})[1],
    )
    drv.respond(SessionContext(system_prompt="sys", tools=[], audio_path=str(_wav(tmp_path))))
    assert "provider" not in sent


def test_headers_carry_only_the_bearer_key(monkeypatch):
    drv = _driver(monkeypatch)
    headers = drv._headers()
    assert headers == {"Authorization": "Bearer test-key"}


def test_text_twin_goes_as_a_text_part(monkeypatch):
    drv = _driver(monkeypatch)
    sent: dict = {}
    monkeypatch.setattr(
        drv,
        "_post",
        lambda body: (sent.update(body), {"choices": [{"message": {"content": "ok"}}]})[1],
    )
    drv.respond(SessionContext(system_prompt="sys", tools=[], text_input="hello"))
    assert sent["messages"][1]["content"] == [{"type": "text", "text": "hello"}]


class TestRateLimitPacing:
    """The free tier is 10 RPM (live 429, 2026-09-20). One key means rotation
    is a no-op, so a 429 must be waited out, never written as an error row —
    absence of quota is not a model failure (D046)."""

    def _quiet(self, monkeypatch, mod):
        slept: list[float] = []
        monkeypatch.setattr(mod.time, "sleep", slept.append)
        monkeypatch.setattr(mod, "MIN_INTERVAL_S", 0.0)
        return slept

    def test_waits_out_a_429_and_retries(self, monkeypatch):
        import voxparity.adapters.stepfun as mod
        from voxparity.adapters.openrouter import OpenRouterDriver, OpenRouterError

        slept = self._quiet(monkeypatch, mod)
        drv = _driver(monkeypatch)
        calls = {"n": 0}

        def parent_post(self, body):
            calls["n"] += 1
            if calls["n"] < 3:
                raise OpenRouterError("HTTP 429: rate_limited")
            return {"choices": []}

        monkeypatch.setattr(OpenRouterDriver, "_post", parent_post)
        assert drv._post({"model": "m"}) == {"choices": []}
        assert calls["n"] == 3
        assert slept == [15.0, 30.0]  # growing waits between 429 retries

    def test_non_429_errors_propagate_immediately(self, monkeypatch):
        import voxparity.adapters.stepfun as mod
        from voxparity.adapters.openrouter import OpenRouterDriver, OpenRouterError

        slept = self._quiet(monkeypatch, mod)
        drv = _driver(monkeypatch)

        def parent_post(self, body):
            raise OpenRouterError("HTTP 400: bad request")

        monkeypatch.setattr(OpenRouterDriver, "_post", parent_post)
        with pytest.raises(OpenRouterError, match="HTTP 400"):
            drv._post({"model": "m"})
        assert slept == []

    def test_paces_consecutive_requests_under_the_rpm_cap(self, monkeypatch):
        import voxparity.adapters.stepfun as mod
        from voxparity.adapters.openrouter import OpenRouterDriver

        slept: list[float] = []
        monkeypatch.setattr(mod.time, "sleep", slept.append)
        monkeypatch.setattr(OpenRouterDriver, "_post", lambda self, body: {"choices": []})
        drv = _driver(monkeypatch)
        drv._post({"model": "m"})
        drv._post({"model": "m"})
        # second request waited toward the 6.2s floor (sleep was stubbed, so
        # nearly the whole interval remains)
        assert len(slept) == 1 and 0 < slept[0] <= mod.MIN_INTERVAL_S


class TestPreflight:
    """D032: a dead model id must fail once, not 796 times mid-run. The preview
    id is retired when the free trial ends, so this gate is load-bearing."""

    def test_rejects_a_model_absent_from_the_listing(self, monkeypatch):
        import voxparity.adapters.stepfun as mod

        class FakeResp:
            status_code = 200

            @staticmethod
            def json():
                return {"data": [{"id": "some-other-model"}]}

        monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: FakeResp())
        drv = _driver(monkeypatch)
        with pytest.raises(mod.StepFunError, match="not in the StepFun model listing"):
            drv.preflight()

    def test_passes_when_listed_and_callable(self, monkeypatch):
        import voxparity.adapters.stepfun as mod

        class FakeResp:
            status_code = 200

            @staticmethod
            def json():
                return {"data": [{"id": "stepaudio-3-chat-preview"}]}

        monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: FakeResp())
        drv = _driver(monkeypatch)
        pinged: dict = {}
        monkeypatch.setattr(drv, "_post", lambda body: (pinged.update(body), {"choices": []})[1])
        drv.preflight()
        assert pinged["model"] == "stepaudio-3-chat-preview"
        assert pinged["max_tokens"] == 1


def test_openrouter_driver_is_undisturbed(monkeypatch, tmp_path: Path):
    """Regression guard for the parameterization: the OpenRouter driver keeps
    its own base URL, raw-base64 + format audio part, and provider pin."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("VOXPARITY_OPENROUTER_PROVIDER", "Together")
    from voxparity.adapters.openrouter import OpenRouterDriver

    drv = OpenRouterDriver("vendor/audio")
    assert drv.base_url == "https://openrouter.ai/api/v1"
    sent: dict = {}
    monkeypatch.setattr(
        drv,
        "_post",
        lambda body: (sent.update(body), {"choices": [{"message": {"content": "ok"}}]})[1],
    )
    drv.respond(SessionContext(system_prompt="sys", tools=[], audio_path=str(_wav(tmp_path))))
    part = sent["messages"][1]["content"][0]
    assert part["input_audio"]["format"] == "wav"
    assert not part["input_audio"]["data"].startswith("data:")
    assert sent["provider"] == {"order": ["Together"], "allow_fallbacks": False}
