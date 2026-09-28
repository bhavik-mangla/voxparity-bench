"""Local llama.cpp driver — open models on the Mac (stateless family).

Targets a running ``llama-server`` started with an audio-capable GGUF + its
mmproj (e.g. Qwen3-Omni-30B-A3B Q4). Audio goes in via the OpenAI-compatible
``input_audio`` content part; the tool channel is PER-MODEL (see the class
docstring). Qwen3-Omni uses a prompted-JSON protocol because its chat template
carries no tool support; Gemma 4 uses native tools, and native tools were
VERIFIED to compose with audio input on llama.cpp build 10566 (2026-08-30) --
an earlier version of this note claimed they did not, which was true of an
older build and is no longer true. Whichever channel is used is recorded on
every ToolCall (``channel``) so replay fidelity can be stated per system.

Start the server (see scripts/serve-qwen3-omni.sh):
  llama-server -m models/Qwen3-Omni-...Q4_K_M.gguf \
    --mmproj models/mmproj-...Q8_0.gguf --port 8801 -c 8192
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from voxparity.adapters.base import DriverCapabilities, SessionContext, SessionDriver
from voxparity.schemas.item import ToolDef
from voxparity.schemas.result import ToolCall, TurnResult

BASE_URL = os.environ.get("VOXPARITY_LLAMACPP_URL", "http://127.0.0.1:8801")

# Per-model defaults for labels added after the Qwen/Gemma arms: (tool channel,
# default port). Matched as a lowercase substring of the driver label. Each model
# gets its own port so a missing env var cannot silently point one label at
# another model's server (preflight would refuse anyway). VOXPARITY_LLAMACPP_URL
# still overrides.
#   phi_tool_tokens: Phi-4-multimodal's documented format, tools as a JSON dump
#       between <|tool|> ... <|/tool|> in the system turn. Served by
#       scripts/serve_phi4mm.py (MLX), not llama.cpp — same HTTP surface.
#   native: Ultravox v0.5 wraps Llama-3.1-8B-Instruct, whose chat template
#       llama.cpp parses natively (Llama 3.x tool format).
#   prompt_json: Qwen2.5-Omni's template carries no tool support (as Qwen3-Omni).
MODEL_SPECS: dict[str, tuple[str, int]] = {
    "phi-4-multimodal": ("phi_tool_tokens", 8804),
    "ultravox": ("native", 8805),
    "qwen2.5-omni": ("prompt_json", 8806),
    "nemotron-voicechat": ("voicechat", 8807),
}
#   voicechat: NVIDIA NemotronLabs VoiceChat 11B (full duplex, MLX 4-bit via
#       scripts/serve_voicechat.py). Tools go in the system prompt in NVIDIA's
#       template wording; calls come back on a separate FUNCTION channel as
#       <TOOLCALL>[...]</TOOLCALL>. No text-input path -> no transcript twin.

# Channels whose model has no text-input path (the D035 case).
NO_TEXT_INPUT = ("voicechat",)

# NVIDIA's tool block (examples/speechlm2/function_calling/template.jinja, as
# rendered in the model card). The model card requires ASCII-only prompts.
VOICECHAT_TOOL_PROTOCOL = (
    "\nYou can use the following tools to assist the user if required:\n"
    "<AVAILABLE_TOOLS>{tools}</AVAILABLE_TOOLS>\n"
    "If you decide to call any tool(s), use the following format:\n"
    '<TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}, '
    '{{"name": "tool_name2", "arguments": "tool_args2"}}]</TOOLCALL>\n'
    "The user will execute tool-calls and return responses from tool(s) in this format:\n"
    '<TOOL_RESPONSE>[{{"tool_response1"}}, {{"tool_response2"}}]</TOOL_RESPONSE>\n'
    "Based on the tool responses, you can call additional tools if needed, correct tool "
    "calls if any errors are found, or just respond to the user."
)

_ASCII_MAP = str.maketrans(
    {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def to_ascii(text: str) -> str:
    """VoiceChat's card: system prompts must be ASCII-only."""
    return text.translate(_ASCII_MAP).encode("ascii", "ignore").decode()


_TOOLCALL_RE = re.compile(r"<TOOLCALL>(.*?)(?:</TOOLCALL>|$)", re.DOTALL)


def parse_voicechat_toolcall(function_text: str, text: str = "") -> list[ToolCall]:
    """First call from the FUNCTION channel; the text channel is a flagged fallback."""
    for source, channel in ((function_text, "voicechat_function"), (text, "voicechat_text")):
        m = _TOOLCALL_RE.search(source or "")
        if m:
            calls = parse_phi_toolcall(m.group(1), channel=channel)
            if calls:
                return calls
    return []


TOOL_PROTOCOL = (
    "\n\nAvailable tools (JSON schemas):\n{tools}\n"
    "If an action is warranted, reply with ONLY a JSON object on a single line:\n"
    '{{"tool": "<tool_name>", "args": {{...}}}}\n'
    "If no tool call is warranted, reply with plain text (no JSON object)."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _tool_schema(t: ToolDef) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "params": {
            p.name: {"type": p.type, "description": p.description, "required": p.required}
            for p in t.params
        },
    }


def _openai_decl(t: ToolDef) -> dict[str, Any]:
    from voxparity.adapters.gemini_file import tool_decl
    from voxparity.providers.groq import openai_tool_decl

    decl: dict[str, Any] = openai_tool_decl(tool_decl(t))
    return decl


def _phi_tool_schema(t: ToolDef) -> dict[str, Any]:
    """The tool shape Phi-4-multimodal's model card documents (name, description,
    parameters as {param: {description, type}})."""
    return {
        "name": t.name,
        "description": t.description,
        "parameters": {p.name: {"description": p.description, "type": p.type} for p in t.params},
    }


PHI_TOOL_PROTOCOL = "<|tool|>{tools}<|/tool|>"

_PHI_TAG_RE = re.compile(r"<\|/?tool_call\|>")


def parse_phi_toolcall(text: str, channel: str = "phi_tool_tokens") -> list[ToolCall]:
    """Parse a Phi-4 function call: a JSON object or list, optionally wrapped in
    <|tool_call|> tags (which the detokenizer may already have stripped), with
    ``name`` + ``arguments`` (or ``parameters``). Only the first call is kept:
    our golds are single-call, as for every other driver."""
    body = _PHI_TAG_RE.sub(" ", text)
    start = min((i for i in (body.find("["), body.find("{")) if i >= 0), default=-1)
    if start < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(body[start:])
    except json.JSONDecodeError:
        return []
    if isinstance(obj, list):
        obj = obj[0] if obj else None
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool")
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return []
    if not name or not isinstance(args, dict):
        return []
    return [ToolCall(tool=str(name), args=args, channel=channel)]


def parse_prompted_toolcall(text: str) -> list[ToolCall]:
    """Extract a single prompted-JSON tool call from model text, if present."""
    m = _JSON_RE.search(text)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict) or "tool" not in obj:
        return []
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        return []
    return [ToolCall(tool=str(obj["tool"]), args=args, channel="prompt_json")]


class LlamaCppDriver(SessionDriver):
    """Local llama.cpp driver with a per-model TOOL CHANNEL.

    The prompted-JSON protocol was correct for Qwen3-Omni, whose chat template
    carries no tool support, so llama.cpp has nothing to bind an OpenAI `tools`
    array to. It is NOT correct for Gemma 4, which has a first-class native
    parser and a dedicated GBNF grammar in llama.cpp — one of only two models in
    the binary that does.

    Scoring Gemma through prompted JSON would measure it on a weaker protocol
    than the one it was trained for, and understate it against every cascade arm
    that gets native tools. Since the tool call IS the core judge-free metric
    (D009), that asymmetry would be a benchmark defect, not a detail. The channel
    is therefore per-model and recorded on every result so replay fidelity can be
    stated per system (D025/D050).
    """

    # Models whose llama.cpp chat template exposes real tool calling.
    NATIVE_TOOL_MODELS = ("gemma-4", "gemma4")

    def __init__(self, model_label: str = "qwen3-omni-30b-a3b-q4") -> None:
        self.name = f"llamacpp:{model_label}"
        self.model_label = model_label
        low = model_label.lower()
        spec = next((v for k, v in MODEL_SPECS.items() if k in low), None)
        if any(m in low for m in self.NATIVE_TOOL_MODELS):
            self.tool_channel = "native"
        elif spec is not None:
            self.tool_channel = spec[0]
        else:
            self.tool_channel = "prompt_json"
        self.native_tools = self.tool_channel == "native"
        env_url = os.environ.get("VOXPARITY_LLAMACPP_URL")
        if env_url:
            self.base_url = env_url
        elif spec is not None:
            self.base_url = f"http://127.0.0.1:{spec[1]}"
        else:
            self.base_url = BASE_URL

    @property
    def capabilities(self) -> DriverCapabilities:
        return DriverCapabilities(
            family="stateless",
            audio_in=True,
            audio_out=False,
            # Phi's documented tool format is the model's own, not ours bolted on.
            native_tools=self.tool_channel != "prompt_json",
            text_twin=self.tool_channel not in NO_TEXT_INPUT,
        )

    def preflight(self) -> None:
        """Confirm a server is up AND serving the model this label claims.

        Two servers run on this Mac on different ports (Qwen 8801, Gemma 8802/3)
        and BASE_URL is a single env var, so pointing the Gemma label at the Qwen
        port is a one-typo mistake that would otherwise produce a full run of
        results attributed to the wrong system — worse than a crash, because it
        looks like data. The model id check makes that impossible.
        """
        try:
            resp = httpx.get(f"{self.base_url}/v1/models", timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"no llama-server reachable at {self.base_url} for {self.name}: {e}"
            ) from e
        served = [m.get("id", "") for m in resp.json().get("data", [])]
        stem = self.model_label.split(":")[0].lower()
        key = stem.replace("-", "").replace("_", "")
        if not any(key[:8] in str(s).lower().replace("-", "").replace("_", "") for s in served):
            raise RuntimeError(
                f"{self.base_url} is serving {served!r}, which does not match driver "
                f"label {self.model_label!r} — wrong port, or wrong server. "
                "Refusing to attribute results to the wrong model."
            )

    def respond(self, ctx: SessionContext) -> TurnResult:
        if (ctx.text_input is None) == (ctx.audio_path is None):
            raise ValueError("exactly one of text_input / audio_path must be set")
        system = ctx.system_prompt
        if self.tool_channel == "voicechat":
            system = to_ascii(system)
            if ctx.tools:
                system += VOICECHAT_TOOL_PROTOCOL.format(
                    tools=to_ascii(
                        json.dumps(
                            [{"type": "function", "function": _openai_decl(t)} for t in ctx.tools]
                        )
                    )
                )
        elif ctx.tools and self.tool_channel == "phi_tool_tokens":
            system += PHI_TOOL_PROTOCOL.format(
                tools=json.dumps([_phi_tool_schema(t) for t in ctx.tools])
            )
        elif ctx.tools and not self.native_tools:
            tools_json = json.dumps([_tool_schema(t) for t in ctx.tools], indent=1)
            system += TOOL_PROTOCOL.format(tools=tools_json)

        content: list[dict[str, Any]] | str
        if ctx.audio_path is not None:
            wav_b64 = base64.b64encode(Path(ctx.audio_path).read_bytes()).decode()
            content = [
                {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}},
            ]
        else:
            content = ctx.text_input or ""

        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": 400,
        }
        if ctx.tools and self.native_tools:
            from voxparity.adapters.gemini_file import tool_decl
            from voxparity.providers.groq import openai_tool_decl

            body["tools"] = [
                {"type": "function", "function": openai_tool_decl(tool_decl(t))} for t in ctx.tools
            ]
            body["tool_choice"] = "auto"
            # Our golds are single-call; llama.cpp defaults this off already.
            body["parallel_tool_calls"] = False

        resp = httpx.post(f"{self.base_url}/v1/chat/completions", json=body, timeout=600.0)
        if (
            resp.status_code == 500
            and self.native_tools
            and "does not match the expected" in resp.text
        ):
            return self._raw_fallback(body, ctx)
        if resp.status_code != 200:
            raise RuntimeError(f"llama-server HTTP {resp.status_code}: {resp.text[:300]}")
        message = resp.json()["choices"][0]["message"]
        text = message.get("content") or ""

        if self.native_tools:
            calls = []
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                calls.append(ToolCall(tool=fn.get("name", ""), args=args, channel="native"))
        elif self.tool_channel == "phi_tool_tokens":
            calls = parse_phi_toolcall(text) if ctx.tools else []
        elif self.tool_channel == "voicechat":
            fn_text = message.get("function") or ""
            calls = parse_voicechat_toolcall(fn_text, text) if ctx.tools else []
            return TurnResult(
                text="" if calls else text.strip(),
                tool_calls=calls,
                raw={
                    "content": text,
                    "function": fn_text,
                    "user_transcript": message.get("user_transcript"),
                    "tool_channel": self.tool_channel,
                },
            )
        else:
            calls = parse_prompted_toolcall(text) if ctx.tools else []

        clean_text = "" if calls else text.strip()
        return TurnResult(
            text=clean_text,
            tool_calls=calls,
            raw={"content": text, "tool_channel": self.tool_channel},
        )

    def _raw_fallback(self, body: dict[str, Any], ctx: SessionContext) -> TurnResult:
        """The model emitted a tool call llama-server's native parser rejects
        (seen on Ultravox/Llama-3.1: unescaped quotes inside a string argument).
        That is a deterministic MODEL output, so resuming would repeat the HTTP
        500 forever and leave the cell unmeasured. Instead, regenerate the same
        greedy completion unparsed (/apply-template + /completion, identical
        prompt and media) and parse it leniently: a well-formed call counts, a
        malformed one is recorded as no valid call with the raw text kept —
        what a production tool pipeline would do with it."""
        tmpl = httpx.post(
            f"{self.base_url}/apply-template",
            json={k: body[k] for k in ("messages", "tools", "tool_choice") if k in body},
            timeout=60.0,
        )
        tmpl.raise_for_status()
        prompt: Any = tmpl.json()["prompt"]
        if ctx.audio_path is not None:
            wav_b64 = base64.b64encode(Path(ctx.audio_path).read_bytes()).decode()
            prompt = {"prompt_string": prompt, "multimodal_data": [wav_b64]}
        comp = httpx.post(
            f"{self.base_url}/completion",
            json={
                "prompt": prompt,
                "temperature": body.get("temperature", 0.0),
                "n_predict": body.get("max_tokens", 400),
            },
            timeout=600.0,
        )
        if comp.status_code != 200:
            raise RuntimeError(f"llama-server HTTP {comp.status_code}: {comp.text[:300]}")
        text = comp.json().get("content") or ""
        calls = parse_phi_toolcall(text, channel="native_raw_fallback")
        return TurnResult(
            text="" if calls else text.strip(),
            tool_calls=calls,
            raw={
                "content": text,
                "tool_channel": self.tool_channel,
                "tool_parse_fallback": True,
                "valid_tool_call": bool(calls),
            },
        )
