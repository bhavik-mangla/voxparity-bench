"""Bring-your-own-agent drivers: `--driver python:<module_or_path>:<Class>`."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from typer.testing import CliRunner

from test_schemas import make_item
from voxparity.adapters.base import SessionContext, SessionDriver
from voxparity.adapters.plugin import (
    PluginDriverError,
    load_plugin_driver,
    parse_driver_args,
    run_id_slug,
)
from voxparity.cli import app
from voxparity.harness.report import load_records
from voxparity.schemas.item import ToolDef
from voxparity.stimuli.store import StimulusRecord, StimulusStore

ROOT = Path(__file__).resolve().parents[1]
ECHO = ROOT / "examples" / "byoa_echo_driver.py"
HTTP = ROOT / "examples" / "byoa_http_agent.py"

DUCK_MODULE = '''
from voxparity.adapters.base import DriverCapabilities
from voxparity.schemas.result import TurnResult

class Duck:
    """Implements the contract without subclassing SessionDriver."""
    name = "duck"
    def __init__(self, greeting="hi"):
        self.greeting = greeting
    @property
    def capabilities(self):
        return DriverCapabilities(family="stateless")
    def respond(self, ctx):
        return TurnResult(text=self.greeting)

class NoRespond:
    name = "x"
    capabilities = DriverCapabilities(family="stateless")

class BadCaps:
    name = "bad"
    capabilities = {"family": "stateless"}
    def respond(self, ctx):
        return TurnResult()

class NoName:
    name = ""
    capabilities = DriverCapabilities(family="stateless")
    def respond(self, ctx):
        return TurnResult()

not_a_class = 3
'''


@pytest.fixture
def duck_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    (tmp_path / "byoa_duck_mod.py").write_text(DUCK_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    return "byoa_duck_mod"


def test_load_by_file_path():
    drv = load_plugin_driver(f"python:{ECHO}:EchoFirstToolDriver")
    assert isinstance(drv, SessionDriver)
    assert drv.name == "byoa-echo-first-tool"
    ctx = SessionContext(
        system_prompt="x",
        tools=[ToolDef(name="first", description=""), ToolDef(name="second", description="")],
        text_input="hello",
    )
    assert drv.respond(ctx).tool_calls[0].tool == "first"
    probe = drv.respond(SessionContext(system_prompt="Q?\n- calm\n- angry", text_input="x"))
    assert probe.text == "calm"


def test_load_by_module_path_with_driver_args(duck_module: str):
    drv = load_plugin_driver(f"python:{duck_module}:Duck", {"greeting": "hello"})
    assert isinstance(drv, SessionDriver)  # wrapped by the protocol adapter
    assert drv.name == "duck"
    drv.preflight()  # optional on duck-typed classes
    assert drv.respond(SessionContext(system_prompt="", text_input="x")).text == "hello"
    assert run_id_slug(drv) == "python_duck"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("python:nomodulecolon", "must be python:<module_or_path>:<ClassName>"),
        ("python:definitely_not_a_module_xyz:C", "cannot import driver module"),
        ("python:missing/file.py:C", "driver file not found"),
        ("python:{mod}:Nope", "has no attribute 'Nope'"),
        ("python:{mod}:not_a_class", "is not a class"),
        ("python:{mod}:NoRespond", "missing respond"),
        ("python:{mod}:BadCaps", "must return voxparity.adapters.base.DriverCapabilities"),
        ("python:{mod}:NoName", "name must be a non-empty string"),
        ("python:voxparity.adapters.base:SessionDriver", "is abstract"),
    ],
)
def test_contract_validation_errors(duck_module: str, spec: str, message: str):
    with pytest.raises(PluginDriverError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_plugin_driver(spec.format(mod=duck_module))


def test_bad_constructor_args(duck_module: str):
    with pytest.raises(PluginDriverError, match="cannot construct Duck"):
        load_plugin_driver(f"python:{duck_module}:Duck", {"colour": "red"})


def test_parse_driver_args():
    assert parse_driver_args(["url=http://h/x?a=b", "t="]) == {"url": "http://h/x?a=b", "t": ""}
    with pytest.raises(PluginDriverError):
        parse_driver_args(["novalue"])
    with pytest.raises(PluginDriverError):
        parse_driver_args(["bad-key=1"])


# ---- end-to-end `voxparity run` over two synthetic items ------------------------


def _fixture_bank(tmp_path: Path) -> tuple[Path, Path]:
    items_dir = tmp_path / "items"
    items_dir.mkdir(parents=True)
    store = StimulusStore(tmp_path / "stimuli")
    for n in (1, 2):
        item = make_item(id=f"vxp-byoa-000{n}")
        (items_dir / f"{item.id}.yaml").write_text(
            yaml.safe_dump(item.model_dump(mode="json"), sort_keys=False)
        )
        for vid in ("happy", "angry"):
            store.put(
                b"fakewav" + item.id.encode() + vid.encode(),
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
    return items_dir, tmp_path / "stimuli"


def _run(tmp_path: Path, driver: str, *extra: str):
    items_dir, store_dir = _fixture_bank(tmp_path)
    args = [
        "run",
        str(items_dir),
        "--driver",
        driver,
        "--engine",
        "test",
        "--store-dir",
        str(store_dir),
        "--out-dir",
        str(tmp_path / "runs"),
        "--run-id",
        "byoa",
        *extra,
    ]
    return CliRunner().invoke(app, args, env={"NO_COLOR": "1", "COLUMNS": "200"})


def _plain(output: str) -> str:
    """CLI output without colour codes, box borders or line wrapping."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", text).split())


def test_cli_run_with_offline_echo_driver(tmp_path: Path):
    result = _run(tmp_path, f"python:{ECHO}:EchoFirstToolDriver")
    assert result.exit_code == 0, result.output
    records = load_records(tmp_path / "runs" / "byoa")
    assert {r["item_id"] for r in records} == {"vxp-byoa-0001", "vxp-byoa-0002"}
    assert [r.get("error") for r in records if r.get("error")] == []
    assert all(r["driver"] == "byoa-echo-first-tool" for r in records)
    conditions = {r["condition"] for r in records}
    assert {"audio", "probe"} <= conditions
    scored = [r for r in records if r["condition"] == "audio"]
    assert len(scored) == 4 and all("selection_credit" in r["scores"] for r in scored)
    # Always-first-tool cannot select the gold on both deliveries of an item.
    for item_id in ("vxp-byoa-0001", "vxp-byoa-0002"):
        cells = [r for r in scored if r["item_id"] == item_id]
        assert sum(r["scores"]["selection_credit"] == 1.0 for r in cells) <= 1


def test_cli_rejects_bad_plugin_and_stray_driver_args(tmp_path: Path):
    bad = _run(tmp_path, "python:no_such_module_q:Cls")
    assert bad.exit_code != 0
    assert "cannot import driver module" in _plain(bad.output)
    stray = _run(tmp_path / "b", "gemini-file", "--driver-arg", "a=b")
    assert stray.exit_code != 0
    assert "--driver-arg applies only" in _plain(stray.output)


class _Agent(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        if not body["tools"]:
            reply = {"text": "angry", "tool_calls": []}
        else:
            # Audio -> last tool, text twin -> first tool.
            pick = body["tools"][-1 if body["audio_wav_b64"] else 0]["name"]
            reply = {"text": "ok", "tool_calls": [{"tool": pick, "args": {}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def agent_url():
    _Agent.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/turn"
    server.shutdown()
    server.server_close()


def test_cli_run_with_http_agent_driver(tmp_path: Path, agent_url: str):
    result = _run(
        tmp_path,
        f"python:{HTTP}:HttpAgentDriver",
        "--driver-arg",
        f"url={agent_url}",
        "--driver-arg",
        "name=mock-agent",
    )
    assert result.exit_code == 0, result.output
    records = load_records(tmp_path / "runs" / "byoa")
    assert records and not any(r.get("error") for r in records)
    assert all(r["driver"] == "mock-agent" for r in records)
    seen = _Agent.seen
    audio = [b for b in seen if b["audio_wav_b64"]]
    twin = [b for b in seen if b["text"] is not None and b["tools"]]
    assert audio and twin
    assert all(b["text"] is None for b in audio)
    assert all(b["audio_wav_b64"] is None for b in twin)
    probes = [r for r in records if r["condition"] == "probe"]
    assert probes and all(r["scores"].get("answer") == "angry" for r in probes)


def test_http_agent_rejects_bad_url():
    with pytest.raises(PluginDriverError, match="cannot construct"):
        load_plugin_driver(f"python:{HTTP}:HttpAgentDriver", {"url": "ftp://x"})
