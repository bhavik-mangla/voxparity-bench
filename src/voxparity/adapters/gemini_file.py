"""Gemini file-mode driver — the first model-under-test adapter (stateless family).

Free-tier, fully reproducible (audio file in, one response out), and honest about
its family: this is the "closed file-mode" roster group, not realtime. The model
id is pinned per run and recorded in every result row.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from voxparity.adapters.base import DriverCapabilities, SessionContext, SessionDriver
from voxparity.providers.gemini import GeminiClient
from voxparity.schemas.item import ToolDef
from voxparity.schemas.result import ToolCall, TurnResult

MUT_MODEL = os.environ.get("VOXPARITY_GEMINI_MUT_MODEL", "gemini-3.7-flash")

_JSON_TYPES = {"string": "STRING", "number": "NUMBER", "integer": "INTEGER", "boolean": "BOOLEAN"}


def tool_decl(tool: ToolDef) -> dict[str, Any]:
    """ToolDef -> Gemini functionDeclaration."""
    props = {
        p.name: {"type": _JSON_TYPES.get(p.type, "STRING"), "description": p.description}
        for p in tool.params
    }
    decl: dict[str, Any] = {"name": tool.name, "description": tool.description}
    if props:
        decl["parameters"] = {
            "type": "OBJECT",
            "properties": props,
            "required": [p.name for p in tool.params if p.required],
        }
    return decl


class GeminiFileDriver(SessionDriver):
    name = f"gemini-file:{MUT_MODEL}"

    def __init__(self) -> None:
        self.client = GeminiClient()

    @property
    def capabilities(self) -> DriverCapabilities:
        return DriverCapabilities(
            family="stateless", audio_in=True, audio_out=False, native_tools=True
        )

    def respond(self, ctx: SessionContext) -> TurnResult:
        if (ctx.text_input is None) == (ctx.audio_path is None):
            raise ValueError("exactly one of text_input / audio_path must be set")
        parts: list[dict[str, Any]]
        if ctx.audio_path is not None:
            wav = Path(ctx.audio_path).read_bytes()
            parts = [
                {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav).decode()}}
            ]
        else:
            parts = [{"text": ctx.text_input or ""}]
        text, raw_calls = self.client.respond_with_tools(
            MUT_MODEL, ctx.system_prompt, parts, [tool_decl(t) for t in ctx.tools]
        )
        calls = [ToolCall(tool=c.get("name", ""), args=dict(c.get("args", {}))) for c in raw_calls]
        return TurnResult(text=text, tool_calls=calls, raw={"function_calls": raw_calls})
