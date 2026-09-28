"""``cascade-replay:<openrouter-model>[+oracle]`` — the open cascade's own ears,
a frontier text brain.

The open cascade (Groq Whisper -> gpt-oss-120b) is the D026 null arm. A
reviewer can fairly ask whether its null is a WEAK-LLM artifact. This driver
answers that without re-running ASR: it replays the Whisper transcripts the
frozen cascade run already recorded (``metrics.asr_transcript`` in
``runs/20260915-final-cascadeopen-gemini``) into any OpenRouter text model,
with the SAME system prompt and tool schema the harness gives every arm. The
twin gets the gold transcript, exactly as in every arm.

``+oracle`` appends a neutral description of the variant's cue (see
``experiments.oracle_note``) to the transcript: the policy upper bound — what
a words-only brain does when it is TOLD what a perfect listener would hear.

No fallback anywhere (D046): a cell with no cached transcript, a truncated
transcript, or a clip whose sha256 differs from the one the cascade heard
raises, so the cell is recorded as an error and never silently re-transcribed
or filled from the gold words. The single exception is a ladder's second turn:
the frozen cascade transcribed those live and never recorded them, so they are
transcribed live with the SAME Groq Whisper model, and the row says so.

Perception probes are not applicable (no audio reaches the LLM); the runner
records them as such.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from voxparity.adapters.base import DriverCapabilities, SessionContext
from voxparity.adapters.openrouter import OpenRouterDriver, OpenRouterError
from voxparity.schemas.result import ToolCall, TurnResult

DEFAULT_SOURCE = "runs/20260915-final-cascadeopen-gemini"
# runner.call_metrics caps asr_transcript at 500 chars; a transcript at the cap
# may have been cut, and replaying a cut transcript would be a silent defect.
ASR_CAP = 500


class ReplayError(RuntimeError):
    pass


def load_transcripts(source: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """{(item_id, variant_id): (asr_transcript, stimulus_sha256)} from a cascade run.

    Latest clean audio row per cell (a resumed run appends retries). A clean
    row without a transcript, or one at the storage cap, is an error here, not
    a missing key later: better one loud failure at startup than a hole found
    mid-run.
    """
    path = source / "records.jsonl"
    if not path.exists():
        raise ReplayError(f"replay source {path} does not exist")
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("condition") != "audio" or r.get("error"):
            continue
        m = r.get("metrics") or {}
        text = m.get("asr_transcript")
        key = (r["item_id"], r["variant_id"])
        if not isinstance(text, str):
            raise ReplayError(f"{source.name}: clean audio row {key} has no asr_transcript")
        if int(m.get("asr_chars", len(text))) != len(text) or len(text) >= ASR_CAP:
            raise ReplayError(f"{source.name}: asr_transcript for {key} is truncated")
        out[key] = (text, str(r.get("stimulus_sha256") or ""))
    if not out:
        raise ReplayError(f"replay source {source} holds no clean audio rows")
    return out


@dataclass(frozen=True)
class ReplaySpec:
    """``cascade-replay:[groq:]<model>[+gold][+oracle|+sham|+dimension]``.

    - ``groq:`` routes the text brain through Groq's API with the EXACT request
      the frozen open cascade sends (same client, tool conversion, decoding) —
      so ``cascade-replay:groq:openai/gpt-oss-120b`` is the cascade's own LLM.
    - ``+gold`` feeds the item's GOLD transcript on audio cells instead of the
      cached Whisper transcript: a per-variant text twin (the harness twin is
      once per item, so it cannot carry a per-variant note).
    - ``+oracle | +sham | +dimension``: which note (experiments.cell_note)
      is appended to the transcript on audio cells. None = no note.
    """

    model: str
    note: str | None = None
    gold: bool = False
    route: str = "openrouter"

    @property
    def oracle(self) -> bool:
        return self.note == "oracle"


def parse_replay_spec(spec: str) -> ReplaySpec:
    from voxparity.harness.experiments import NOTE_MODES

    family, _, rest = spec.partition(":")
    if family != "cascade-replay" or not rest:
        raise ValueError(
            f"expected cascade-replay:[groq:]<model>[+gold][+oracle|+sham|+dimension], got {spec!r}"
        )
    route = "openrouter"
    if rest.startswith("groq:"):
        route, rest = "groq", rest[len("groq:") :]
    model, *flags = rest.split("+")
    if not model:
        raise ValueError(f"bad cascade-replay spec {spec!r}")
    gold = False
    note: str | None = None
    for f in flags:
        if f == "gold" and not gold and note is None:
            gold = True
        elif f in NOTE_MODES and note is None:
            note = f
        else:
            raise ValueError(f"bad cascade-replay spec {spec!r}: flag {f!r} (+gold, then a note)")
    return ReplaySpec(model=model, note=note, gold=gold, route=route)


def parse_spec(spec: str) -> tuple[str, bool]:
    """Back-compatible ``cascade-replay:<model>[+oracle]`` -> (model, oracle)."""
    p = parse_replay_spec(spec)
    return p.model, p.oracle


class ReplayCascadeDriver(OpenRouterDriver):
    def __init__(
        self,
        model: str,
        oracle: bool = False,
        source: str | Path | None = None,
        transcripts: dict[tuple[str, str], tuple[str, str]] | None = None,
        *,
        note: str | None = None,
        gold: bool = False,
        route: str = "openrouter",
    ) -> None:
        if oracle and note not in (None, "oracle"):
            raise ValueError("oracle=True conflicts with note=" + repr(note))
        # OpenRouterDriver's own note (audio+note) stays None: this driver
        # handles its notes on the transcript path.
        super().__init__(model)
        self.cell_note_mode = "oracle" if oracle else note
        self.oracle = self.cell_note_mode == "oracle"
        self.gold = gold
        if route not in ("openrouter", "groq"):
            raise ValueError(f"unknown replay route {route!r}")
        self.route = route
        self.source = Path(source or os.environ.get("VOXPARITY_REPLAY_SOURCE", DEFAULT_SOURCE))
        self.transcripts = transcripts if transcripts is not None else load_transcripts(self.source)
        flags = ("+gold" if gold else "") + (
            f"+{self.cell_note_mode}" if self.cell_note_mode else ""
        )
        prefix = "groq:" if route == "groq" else ""
        self.name = f"cascade-replay:{prefix}{model}{flags}"
        self._groq: Any = None

    @classmethod
    def from_spec(cls, spec: str, **kw: Any) -> ReplayCascadeDriver:
        p = parse_replay_spec(spec)
        return cls(p.model, note=p.note, gold=p.gold, route=p.route, **kw)

    @property
    def experiment(self) -> dict[str, Any]:
        """Stamped on every row via experiments.experiment_settings. The first
        three keys are exactly the pre-Sep-26 stamp, so existing oracle/replay
        runs resume unchanged; new keys appear only when set."""
        out: dict[str, Any] = {
            "replay_source": self.source.name,
            "replay_cells": len(self.transcripts),
            "oracle": self.oracle,
        }
        if self.cell_note_mode and not self.oracle:
            out["cue_note"] = self.cell_note_mode
        if self.gold:
            out["replay_transcript"] = "gold"
        if self.route != "openrouter":
            out["llm_route"] = self.route
        return out

    @property
    def capabilities(self) -> DriverCapabilities:
        return DriverCapabilities(
            family="stateless",
            audio_in=True,  # it accepts an audio CELL; the audio itself never reaches the LLM
            audio_out=False,
            native_tools=True,
            text_twin=True,
            perception_probe=False,
        )

    def preflight(self) -> None:
        """Model exists, takes tools, answers a 1-token call. Unlike the parent,
        audio input is NOT required: this is a text brain by design."""
        if self.route == "groq":
            from voxparity.providers.groq import LLM_MODEL

            if self.model != LLM_MODEL:
                raise ReplayError(
                    f"groq route replays the cascade's own LLM {LLM_MODEL!r}, not {self.model!r}"
                )
            self._groq_client().respond_with_tools("ping", "ping", [])
            return
        try:
            resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=60.0)
        except httpx.TransportError as e:
            raise OpenRouterError(f"preflight: {e}") from e
        if resp.status_code != 200:
            raise OpenRouterError(f"preflight: HTTP {resp.status_code}: {resp.text[:200]}")
        entry = {m["id"]: m for m in resp.json().get("data", [])}.get(self.model)
        if entry is None:
            raise OpenRouterError(f"model {self.model!r} is not in the OpenRouter catalogue")
        if "tools" not in (entry.get("supported_parameters") or []):
            raise OpenRouterError(f"model {self.model!r} does not support tool calling")
        ping = {"role": "user", "content": "ping"}
        self._post({"model": self.model, "messages": [ping], "max_tokens": 1})

    def _groq_client(self) -> Any:
        if self._groq is None:
            from voxparity.providers.groq import GroqClient

            self._groq = GroqClient()
        return self._groq

    def _live_transcript(self, audio_path: str) -> str:
        return str(self._groq_client().transcribe(Path(audio_path).read_bytes()))

    def _audio_cell_text(self, ctx: SessionContext) -> tuple[str, dict[str, Any]]:
        from voxparity.harness.experiments import cell_note

        item, vid = ctx.item, ctx.variant_id
        if item is None or not vid:
            raise ReplayError("replay needs the cell identity (item, variant_id) on audio turns")
        assert ctx.audio_path is not None
        prov: dict[str, Any]
        if vid.endswith("__followup"):
            base_vid = vid[: -len("__followup")]
            v = next(x for x in item.variants if x.variant_id == base_vid)
            if v.followup is None:
                raise ReplayError(f"{item.id}/{base_vid} has no followup")
            if self.gold:
                transcript = v.followup.caller_reply
                prov = {"asr_source": "gold"}
            else:
                transcript = self._live_transcript(ctx.audio_path)
                prov = {"asr_transcript": transcript, "asr_source": "live-groq"}
        elif self.gold:
            # The per-variant text twin: the gold words the harness twin gets,
            # on an audio cell so a per-variant note can ride along. No ASR.
            transcript = item.transcript
            prov = {"asr_source": "gold"}
        else:
            cached = self.transcripts.get((item.id, vid))
            if cached is None:
                raise ReplayError(
                    f"no cached transcript for {item.id}/{vid} in {self.source.name} "
                    "(no fallback: the cell stays unmeasured)"
                )
            transcript, sha = cached
            actual = hashlib.sha256(Path(ctx.audio_path).read_bytes()).hexdigest()
            if sha and sha != actual:
                raise ReplayError(
                    f"{item.id}/{vid}: clip sha {actual[:12]} differs from the one the "
                    f"cascade transcribed ({sha[:12]}); the cached transcript is not this clip's"
                )
            prov = {"asr_transcript": transcript, "asr_source": f"cached:{self.source.name}"}
        text = transcript
        if self.cell_note_mode:
            note = cell_note(item, vid, self.cell_note_mode)
            text = f"{transcript}\n\n{note}"
            prov["oracle_note"] = note
        return text, prov

    def respond(self, ctx: SessionContext) -> TurnResult:
        if (ctx.text_input is None) == (ctx.audio_path is None):
            raise ValueError("exactly one of text_input / audio_path must be set")
        if not ctx.tools and ctx.audio_path is not None:
            raise ReplayError("a replayed cascade cannot answer an audio perception probe")
        if ctx.audio_path is not None:
            text, prov = self._audio_cell_text(ctx)
        else:
            text, prov = ctx.text_input or "", {}
        if self.route == "groq":
            result = self._complete_groq(ctx.system_prompt, text, ctx.tools)
        else:
            result = self._complete(ctx.system_prompt, [{"type": "text", "text": text}], ctx.tools)
        result.raw.update(prov)
        return result

    def _complete_groq(self, system_prompt: str, text: str, tools: list[Any]) -> TurnResult:
        """The frozen open cascade's exact LLM call (cascade.CascadeDriver,
        stack="open"): same client, tool conversion and temperature 0."""
        from voxparity.adapters.gemini_file import tool_decl
        from voxparity.providers.groq import openai_tool_decl

        decls = [openai_tool_decl(tool_decl(t)) for t in tools]
        out_text, raw_calls = self._groq_client().respond_with_tools(system_prompt, text, decls)
        calls = [ToolCall(tool=c.get("name", ""), args=dict(c.get("args", {}))) for c in raw_calls]
        return TurnResult(text=out_text, tool_calls=calls, raw={"function_calls": raw_calls})
