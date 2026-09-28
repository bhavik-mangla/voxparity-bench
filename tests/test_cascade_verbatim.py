"""Contract tests for the cascade-open-verbatim driver and the dedicated
Gemini transcription path (gemini-3.5-transcribe via the interactions API).

The driver is cascade-open with a different ASR front end: Gemini's dedicated
transcription model in its documented verbatim mode ("preserving raw filler
words ... repetitions, pauses, and false starts"), so the text LLM sees a
transcript that keeps the disfluency channel. Mock HTTP throughout — no
network in tests.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

from voxparity.adapters.base import SessionContext
from voxparity.schemas.item import ToolDef, ToolParam

VERBATIM = "Um... I— I guess so, uh, yeah. Go ahead."


def _wav(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 2400)
    return p


class FakeResp:
    def __init__(self, status: int = 200, json_data: Any = None, headers: dict | None = None):
        self.status_code = status
        self._json = json_data or {}
        self.headers = headers or {}
        self.text = str(json_data)

    def json(self) -> Any:
        return self._json


def _interaction_json(text: str = VERBATIM) -> dict:
    # documented response shape: steps[].content[] with {"type": "text", ...};
    # output_text is the SDK convenience accessor and may be absent on REST.
    return {
        "id": "interactions/abc123",
        "status": "completed",
        "steps": [
            {"id": "step_001", "type": "model_output", "content": [{"type": "text", "text": text}]}
        ],
    }


def _mock_transcribe_http(monkeypatch, calls: list) -> None:
    """Fake the three-step sequence: upload start -> upload finalize ->
    interactions. Records every request for contract assertions."""
    import voxparity.providers.gemini as mod

    def fake_post(url, *, headers, json_body=None, content=None, timeout=0, tries=0, **kw):
        calls.append({"url": url, "headers": headers, "json": json_body, "content": content})
        if url == mod.UPLOAD_URL:
            return FakeResp(headers={"x-goog-upload-url": "https://upload.example/u1"})
        if url == "https://upload.example/u1":
            return FakeResp(json_data={"file": {"uri": "files/f-1", "name": "files/f-1"}})
        if url.endswith("/interactions"):
            return FakeResp(json_data=_interaction_json())
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(mod, "post_with_retry", fake_post)


def _gemini_env(monkeypatch, *, paid_only: bool = True) -> None:
    # the PX-003 billing-guard shape: free vars empty, only the paid key set
    monkeypatch.setenv("GEMINI_API_KEYS", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_PAID_KEY", "paid-key")
    if not paid_only:
        monkeypatch.setenv("GEMINI_API_KEYS", "free-1")


def test_transcribe_verbatim_pins_model_mode_and_upload_contract(monkeypatch):
    from voxparity.providers.gemini import GeminiClient

    _gemini_env(monkeypatch)
    calls: list = []
    _mock_transcribe_http(monkeypatch, calls)
    client = GeminiClient()
    assert client.pool.size == 1 and client.pool.paid_idx == 0  # paid-only proof shape

    out = client.transcribe_verbatim(b"RIFFxxxx")
    assert out == VERBATIM  # fillers, repetition, pause, false start all intact

    start, finalize, interact = calls
    # Files API resumable upload, bound to the same key throughout
    assert start["url"].endswith("/upload/v1beta/files")
    assert start["headers"]["X-Goog-Upload-Protocol"] == "resumable"
    assert start["headers"]["X-Goog-Upload-Header-Content-Type"] == "audio/wav"
    assert finalize["content"] == b"RIFFxxxx"
    assert finalize["headers"]["X-Goog-Upload-Command"] == "upload, finalize"
    assert {c["headers"]["x-goog-api-key"] for c in calls} == {"paid-key"}
    # interactions request: exact model id, uri input, verbatim pinned explicitly
    body = interact["json"]
    assert body["model"] == "gemini-3.5-transcribe"
    assert body["input"] == [{"type": "audio", "uri": "files/f-1", "mime_type": "audio/wav"}]
    assert body["generation_config"]["transcription_config"]["mode"]["type"] == "verbatim"


def test_transcribe_verbatim_prefers_output_text_when_present(monkeypatch):
    import voxparity.providers.gemini as mod

    _gemini_env(monkeypatch)

    def fake_post(url, *, headers, json_body=None, content=None, timeout=0, tries=0, **kw):
        if url == mod.UPLOAD_URL:
            return FakeResp(headers={"x-goog-upload-url": "https://upload.example/u1"})
        if url == "https://upload.example/u1":
            return FakeResp(json_data={"file": {"uri": "files/f-1"}})
        return FakeResp(json_data={"status": "completed", "output_text": "uh, hello"})

    monkeypatch.setattr(mod, "post_with_retry", fake_post)
    assert mod.GeminiClient().transcribe_verbatim(b"x") == "uh, hello"


def test_transcribe_verbatim_rotates_to_paid_and_snaps_back(monkeypatch):
    """Free key exhausted mid-sequence -> the WHOLE upload+interact sequence
    restarts on the next key (files belong to the key that uploaded them), and
    release_paid() routes the next request back to the free keys."""
    import voxparity.providers.gemini as mod

    _gemini_env(monkeypatch, paid_only=False)  # pool: [free-1, paid-key]
    seen_keys: list[str] = []

    def fake_post(url, *, headers, json_body=None, content=None, timeout=0, tries=0, **kw):
        key = headers["x-goog-api-key"]
        seen_keys.append(key)
        if key == "free-1":
            raise TimeoutError("still rate-limited/5xx after 2 attempts")
        if url == mod.UPLOAD_URL:
            return FakeResp(headers={"x-goog-upload-url": "https://upload.example/u1"})
        if url == "https://upload.example/u1":
            return FakeResp(json_data={"file": {"uri": "files/f-2"}})
        return FakeResp(json_data=_interaction_json("okay."))

    monkeypatch.setattr(mod, "post_with_retry", fake_post)
    client = mod.GeminiClient()
    assert client.transcribe_verbatim(b"x") == "okay."
    assert seen_keys[0] == "free-1" and set(seen_keys[1:]) == {"paid-key"}
    assert client.pool.idx == 0  # snap-back: paid key never becomes the default route


def test_transcribe_verbatim_no_transcript_is_an_error(monkeypatch):
    import voxparity.providers.gemini as mod

    _gemini_env(monkeypatch)

    def fake_post(url, *, headers, json_body=None, content=None, timeout=0, tries=0, **kw):
        if url == mod.UPLOAD_URL:
            return FakeResp(headers={"x-goog-upload-url": "https://upload.example/u1"})
        if url == "https://upload.example/u1":
            return FakeResp(json_data={"file": {"uri": "files/f-1"}})
        return FakeResp(json_data={"status": "failed", "steps": []})

    monkeypatch.setattr(mod, "post_with_retry", fake_post)
    with pytest.raises(mod.GeminiError, match="no transcript"):
        mod.GeminiClient().transcribe_verbatim(b"x")


def _verbatim_driver(monkeypatch):
    _gemini_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    from voxparity.adapters.cascade import CascadeDriver

    return CascadeDriver(stack="open-verbatim")


def test_open_verbatim_is_open_cascade_with_gemini_asr(monkeypatch, tmp_path: Path):
    d = _verbatim_driver(monkeypatch)
    assert d.name.startswith("cascade-open-verbatim:gemini-3.5-transcribe+")
    assert d.name.endswith("gpt-oss-120b")
    caps = d.capabilities
    assert caps.audio_in and caps.native_tools and not caps.perception_probe

    sent: dict = {}
    monkeypatch.setattr(d.asr, "transcribe_verbatim", lambda wav: VERBATIM)

    def fake_llm(system, user_text, tool_decls):
        sent.update(system=system, user_text=user_text, tool_decls=tool_decls)
        return "", [{"name": "enroll", "args": {"plan": "basic"}}]

    monkeypatch.setattr(d.groq, "respond_with_tools", fake_llm)
    tools = [
        ToolDef(
            name="enroll",
            description="d",
            params=[ToolParam(name="plan", type="string", description="d")],
        )
    ]
    res = d.respond(
        SessionContext(system_prompt="sys", tools=tools, audio_path=str(_wav(tmp_path)))
    )
    # the verbatim transcript reaches the SAME Groq LLM used by cascade-open
    assert sent["user_text"] == VERBATIM
    assert sent["tool_decls"][0]["name"] == "enroll"
    assert sent["tool_decls"][0]["parameters"]["properties"]["plan"]["type"] == "string"
    assert res.tool_calls[0].tool == "enroll" and res.tool_calls[0].args == {"plan": "basic"}
    assert res.raw["asr_transcript"] == VERBATIM  # provenance for metrics.asr_transcript


def test_open_verbatim_text_twin_never_touches_asr(monkeypatch):
    d = _verbatim_driver(monkeypatch)

    def boom(wav):  # pragma: no cover - the assertion is that it is never hit
        raise AssertionError("text twin must not call ASR")

    monkeypatch.setattr(d.asr, "transcribe_verbatim", boom)
    monkeypatch.setattr(d.groq, "respond_with_tools", lambda s, u, t: ("ok", []))
    res = d.respond(SessionContext(system_prompt="sys", tools=[], text_input="hello"))
    assert res.text == "ok" and res.raw.get("asr_transcript") is None


def test_open_verbatim_refuses_audio_probe(monkeypatch, tmp_path: Path):
    d = _verbatim_driver(monkeypatch)
    with pytest.raises(RuntimeError, match="perception probe"):
        d.respond(SessionContext(system_prompt="q", tools=[], audio_path=str(_wav(tmp_path))))
