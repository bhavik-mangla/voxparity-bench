"""Closability experiments: prompt conditions, replay/oracle cascade, rollouts,
the cwd-first .env loader, and the exp-vs-baseline analysis."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from _held import needs_full_bank
from test_harness import ScriptedDriver, _seed_store
from test_schemas import make_item
from voxparity.adapters.base import SessionContext
from voxparity.harness import experiments as ex
from voxparity.harness.report import load_records
from voxparity.harness.runner import SYSTEM_TEMPLATE, RunWriter, item_tools, run_item, system_prompt
from voxparity.schemas.item import DeliveryVariant, GoldAction, Item, PerceptionProbe

EXP_VARS = (
    "VOXPARITY_PROMPT_CONDITION",
    "VOXPARITY_TEMPERATURE",
    "VOXPARITY_ROLLOUT_INDEX",
    "VOXPARITY_SKIP_PROBES",
    "VOXPARITY_CUE_BEARING_ONLY",
    "VOXPARITY_SKIP_TWIN",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in EXP_VARS:
        monkeypatch.delenv(v, raising=False)


def _neutral_item() -> Item:
    """One neutral (clean) and one urgent variant with different golds."""
    return make_item(
        variants=[
            DeliveryVariant(
                variant_id="happy",
                emotion="neutral",
                intensity=0.5,
                gold=GoldAction(tool="a", args={"x": ["1"]}, rationale="r"),
            ),
            DeliveryVariant(
                variant_id="angry",
                emotion="urgent",
                intensity=0.5,
                gold=GoldAction(tool="b", rationale="r"),
            ),
        ],
        perception_probe=PerceptionProbe(
            question="q?",
            options=["calm", "urgent"],
            gold_by_variant={"happy": "calm", "angry": "urgent"},
        ),
    )


# --------------------------------------------------------------------------- .env


class TestLoadEnv:
    def test_cwd_env_is_loaded_first(self, tmp_path, monkeypatch):
        from voxparity.cli import load_env

        (tmp_path / ".env").write_text("VXP_TEST_ENV_SOURCE=cwd\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VXP_TEST_ENV_SOURCE", raising=False)
        load_env()
        assert os.environ["VXP_TEST_ENV_SOURCE"] == "cwd"
        monkeypatch.delenv("VXP_TEST_ENV_SOURCE")

    def test_existing_environment_wins(self, tmp_path, monkeypatch):
        from voxparity.cli import load_env

        (tmp_path / ".env").write_text("VXP_TEST_ENV_SOURCE=cwd\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("VXP_TEST_ENV_SOURCE", "shell")
        load_env()
        assert os.environ["VXP_TEST_ENV_SOURCE"] == "shell"

    def test_falls_back_to_package_search_without_a_cwd_env(self, tmp_path, monkeypatch):
        import voxparity.cli as cli

        calls = []
        monkeypatch.chdir(tmp_path)  # no .env here or above (tmp is outside the repo)
        monkeypatch.setattr("dotenv.find_dotenv", lambda usecwd=False: "" if usecwd else "pkg")
        monkeypatch.setattr("dotenv.load_dotenv", lambda p=None: calls.append(p))
        cli.load_env()
        assert calls == ["pkg"]


# --------------------------------------------------------------------------- prompt


class TestPromptConditions:
    def test_none_is_byte_identical_to_the_frozen_prompt(self, monkeypatch):
        item = make_item()
        frozen = SYSTEM_TEMPLATE.format(scenario=item.scenario, policy="")
        assert system_prompt(item) == frozen
        monkeypatch.setenv("VOXPARITY_PROMPT_CONDITION", "none")
        assert system_prompt(item) == frozen

    @pytest.mark.parametrize("cond", ["listen", "sham", "describe_then_act"])
    def test_condition_appends_its_exact_sentence(self, monkeypatch, cond):
        item = make_item()
        monkeypatch.setenv("VOXPARITY_PROMPT_CONDITION", cond)
        frozen = SYSTEM_TEMPLATE.format(scenario=item.scenario, policy="")
        assert system_prompt(item) == frozen + "\n" + ex.PROMPT_CONDITIONS[cond]

    def test_listen_is_the_preregistered_sentence(self):
        assert ex.LISTEN == (
            "How the caller sounds, background sounds and who is speaking can change the "
            "right action."
        )

    def test_sham_is_length_matched_and_says_nothing_about_audio(self):
        assert abs(len(ex.SHAM) - len(ex.LISTEN)) <= 0.15 * len(ex.LISTEN)
        low = ex.SHAM.lower()
        for word in ("sound", "voice", "tone", "background", "speak", "hear", "listen"):
            assert word not in low

    def test_unknown_condition_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("VOXPARITY_PROMPT_CONDITION", "louder")
        with pytest.raises(ex.ExperimentConfigError):
            system_prompt(make_item())


# --------------------------------------------------------------------------- runner stamps


class TestRunnerStamps:
    def test_frozen_run_rows_have_no_experiment_key(self, tmp_path):
        item = make_item()
        store = _seed_store(tmp_path, item.id)
        run_item(ScriptedDriver(), item, store, "test", RunWriter(tmp_path / "runs", "r"), "r")
        for line in (tmp_path / "runs" / "r" / "records.jsonl").read_text().splitlines():
            assert "experiment" not in json.loads(line)

    def test_every_row_carries_the_condition_and_added_text(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOXPARITY_PROMPT_CONDITION", "listen")
        monkeypatch.setenv("VOXPARITY_TEMPERATURE", "1.0")
        monkeypatch.setenv("VOXPARITY_ROLLOUT_INDEX", "2")
        item = make_item()
        store = _seed_store(tmp_path, item.id)
        run_item(ScriptedDriver(), item, store, "test", RunWriter(tmp_path / "runs", "r"), "r")
        rows = load_records(tmp_path / "runs" / "r")
        assert len(rows) == 5
        for r in rows:
            assert r["experiment"]["prompt_condition"] == "listen"
            assert r["experiment"]["prompt_added_text"] == ex.LISTEN
            assert r["experiment"]["temperature"] == 1.0
            assert r["experiment"]["rollout_index"] == 2

    def test_skip_probes_records_not_applicable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOXPARITY_SKIP_PROBES", "1")
        item = make_item()
        store = _seed_store(tmp_path, item.id)
        run_item(ScriptedDriver(), item, store, "test", RunWriter(tmp_path / "runs", "r"), "r")
        probes = [r for r in load_records(tmp_path / "runs" / "r") if r["condition"] == "probe"]
        assert len(probes) == 2
        assert all(p["scores"]["applicable"] is False for p in probes)
        assert all(p["response_text"] == "" for p in probes)

    def test_cue_bearing_only_skips_neutral_variants_visibly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VOXPARITY_CUE_BEARING_ONLY", "1")
        item = _neutral_item()
        store = _seed_store(tmp_path, item.id)
        run_item(ScriptedDriver(), item, store, "test", RunWriter(tmp_path / "runs", "r"), "r")
        rows = load_records(tmp_path / "runs" / "r")
        clean = [r for r in rows if r["variant_id"] == "happy"]
        assert [r["error"] for r in clean] == ["skipped: not cue-bearing (experiment filter)"]
        cue = [r for r in rows if r["variant_id"] == "angry"]
        assert {r["condition"] for r in cue} == {"audio", "probe"}
        assert all(not r["error"] for r in cue)

    def test_consistency_guard_refuses_a_mixed_run_id(self, tmp_path, monkeypatch):
        item = make_item()
        store = _seed_store(tmp_path, item.id)
        writer = RunWriter(tmp_path / "runs", "r")
        run_item(ScriptedDriver(), item, store, "test", writer, "r")
        ex.check_run_consistency(writer.path, ScriptedDriver())  # same condition: fine
        monkeypatch.setenv("VOXPARITY_PROMPT_CONDITION", "sham")
        with pytest.raises(ex.ExperimentConfigError, match="new --run-id"):
            ex.check_run_consistency(writer.path, ScriptedDriver())


# --------------------------------------------------------------------------- temperature


def _fake_completion(sent):
    def fake_post(body):
        sent.append(body)
        return {"choices": [{"message": {"content": "", "tool_calls": []}}], "provider": "X"}

    return fake_post


def test_openrouter_temperature_defaults_to_frozen_zero_and_follows_env(monkeypatch):
    import voxparity.adapters.openrouter as mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    drv = mod.OpenRouterDriver("vendor/model")
    sent: list = []
    monkeypatch.setattr(drv, "_post", _fake_completion(sent))
    drv.respond(SessionContext(system_prompt="s", text_input="hi"))
    monkeypatch.setenv("VOXPARITY_TEMPERATURE", "1")
    drv.respond(SessionContext(system_prompt="s", text_input="hi"))
    assert [b["temperature"] for b in sent] == [0.0, 1.0]


# --------------------------------------------------------------------------- replay


def _replay_source(tmp_path: Path, rows: list[dict]) -> Path:
    src = tmp_path / "20260915-final-cascadeopen-gemini"
    src.mkdir()
    (src / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return src


def _asr_row(item_id, vid, text, sha, error=""):
    return {
        "item_id": item_id,
        "variant_id": vid,
        "condition": "audio",
        "error": error,
        "stimulus_sha256": sha,
        "metrics": {"asr_transcript": text, "asr_chars": len(text)},
    }


class TestReplayCascade:
    def _driver(self, tmp_path, monkeypatch, oracle=False, rows=None):
        from voxparity.adapters.replay import ReplayCascadeDriver

        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        item = _neutral_item()
        store = _seed_store(tmp_path, item.id)
        shas = {v: store.get(item.id, v, "test").sha256 for v in ("happy", "angry")}
        rows = rows or [
            _asr_row(item.id, "happy", "fine whatever book it", shas["happy"]),
            _asr_row(item.id, "angry", "fine whatever just book it", shas["angry"]),
        ]
        src = _replay_source(tmp_path, rows)
        drv = ReplayCascadeDriver("vendor/text-llm", oracle=oracle, source=src)
        sent: list = []
        monkeypatch.setattr(drv, "_post", _fake_completion(sent))
        return drv, item, store, sent

    def test_audio_cells_get_the_cached_transcript_twin_the_gold(self, tmp_path, monkeypatch):
        drv, item, store, sent = self._driver(tmp_path, monkeypatch)
        run_item(drv, item, store, "test", RunWriter(tmp_path / "runs", "r"), "r")
        user = {b["messages"][1]["content"][0]["text"] for b in sent}
        assert user == {item.transcript, "fine whatever book it", "fine whatever just book it"}
        rows = load_records(tmp_path / "runs" / "r")
        probes = [r for r in rows if r["condition"] == "probe"]
        assert probes and all(p["scores"]["applicable"] is False for p in probes)
        audio = [r for r in rows if r["condition"] == "audio"]
        assert all(r["metrics"]["asr_source"].startswith("cached:") for r in audio)
        assert all(r["experiment"]["oracle"] is False for r in rows)
        assert all(not r["error"] for r in rows)

    def test_same_system_prompt_and_tool_schema_as_the_openrouter_arm(self, tmp_path, monkeypatch):
        import voxparity.adapters.openrouter as mod

        drv, item, _store, sent = self._driver(tmp_path, monkeypatch)
        ref = mod.OpenRouterDriver("vendor/text-llm")
        ref_sent: list = []
        monkeypatch.setattr(ref, "_post", _fake_completion(ref_sent))
        ctx = SessionContext(
            system_prompt=system_prompt(item), tools=item_tools(item), text_input=item.transcript
        )
        drv.respond(ctx)
        ref.respond(ctx)
        assert sent[0]["tools"] == ref_sent[0]["tools"]
        assert sent[0]["messages"] == ref_sent[0]["messages"]
        assert sent[0]["temperature"] == 0.0

    def test_missing_transcript_fails_loudly_no_fallback(self, tmp_path, monkeypatch):
        item = _neutral_item()
        store = _seed_store(tmp_path, item.id)
        sha = store.get(item.id, "happy", "test").sha256
        drv, item, store, sent = self._driver(
            tmp_path, monkeypatch, rows=[_asr_row(item.id, "happy", "fine", sha)]
        )
        run_item(drv, item, store, "test", RunWriter(tmp_path / "runs", "r"), "r")
        angry = [
            r
            for r in load_records(tmp_path / "runs" / "r")
            if r["variant_id"] == "angry" and r["condition"] == "audio"
        ]
        assert "no cached transcript" in angry[0]["error"]
        assert len(sent) == 2  # twin + the one cached cell; nothing filled in for "angry"

    def test_clip_hash_mismatch_fails_loudly(self, tmp_path, monkeypatch):
        item = _neutral_item()
        rows = [
            _asr_row(item.id, "happy", "fine", "0" * 64),
            _asr_row(item.id, "angry", "fine", "0" * 64),
        ]
        drv, item, store, _ = self._driver(tmp_path, monkeypatch, rows=rows)
        run_item(drv, item, store, "test", RunWriter(tmp_path / "runs", "r"), "r")
        audio = [r for r in load_records(tmp_path / "runs" / "r") if r["condition"] == "audio"]
        assert all("differs from the one the cascade transcribed" in r["error"] for r in audio)

    def test_source_without_transcript_or_truncated_is_refused(self, tmp_path):
        from voxparity.adapters.replay import ReplayError, load_transcripts

        bad = {"item_id": "i", "variant_id": "v", "condition": "audio", "error": "", "metrics": {}}
        with pytest.raises(ReplayError, match="no asr_transcript"):
            load_transcripts(_replay_source(tmp_path, [bad]))
        cut = tmp_path / "cut"
        cut.mkdir()
        row = _asr_row("i", "v", "x" * 500, "s")
        row["metrics"]["asr_chars"] = 812
        (cut / "records.jsonl").write_text(json.dumps(row) + "\n")
        with pytest.raises(ReplayError, match="truncated"):
            load_transcripts(cut)

    def test_oracle_appends_the_note_on_audio_cells_only(self, tmp_path, monkeypatch):
        drv, item, store, sent = self._driver(tmp_path, monkeypatch, oracle=True)
        assert drv.name == "cascade-replay:vendor/text-llm+oracle"
        run_item(drv, item, store, "test", RunWriter(tmp_path / "runs", "r"), "r")
        texts = [b["messages"][1]["content"][0]["text"] for b in sent]
        assert texts[0] == item.transcript  # twin: bare gold words
        assert texts[1].endswith(ex.oracle_note(item, "happy"))
        assert "sounds urgent" in texts[2]
        audio = [r for r in load_records(tmp_path / "runs" / "r") if r["condition"] == "audio"]
        assert all(r["metrics"]["oracle_note"] for r in audio)

    def test_spec_parsing(self):
        from voxparity.adapters.replay import parse_spec

        assert parse_spec("cascade-replay:a/b") == ("a/b", False)
        assert parse_spec("cascade-replay:a/b+oracle") == ("a/b", True)
        with pytest.raises(ValueError):
            parse_spec("cascade-replay:")


# --------------------------------------------------------------------------- oracle notes


def _bank() -> list[Item]:
    root = Path(__file__).resolve().parents[1] / "items" / "pilot" / "t4"
    return [Item.model_validate(yaml.safe_load(p.read_text())) for p in sorted(root.glob("*.yaml"))]


@needs_full_bank()
def test_oracle_notes_never_echo_a_tool_name_or_gold_rationale():
    items = _bank()
    assert len(items) > 50
    for item in items:
        names = {t.name for t in item_tools(item)}
        for v in item.variants:
            note = ex.oracle_note(item, v.variant_id).lower()
            assert note.startswith("delivery note:")
            for n in names:
                assert n.lower() not in note, (item.id, v.variant_id, n)
                assert n.replace("_", " ").lower() not in note, (item.id, v.variant_id, n)
            assert v.gold.rationale.lower() not in note


def test_oracle_note_is_a_description_on_every_variant_including_neutral():
    item = _neutral_item()
    assert ex.oracle_note(item, "happy") == (
        "Delivery note: the caller sounds calm and neutral. "
        "Background: no other voices or notable sounds."
    )
    assert ex.oracle_note(item, "angry").startswith("Delivery note: the caller sounds urgent.")


# --------------------------------------------------------------------------- analysis


def _write_run(runs: Path, name: str, driver: str, item: Item, audio: dict, twin: dict, exp=None):
    d = runs / name
    d.mkdir(parents=True)
    rows = []
    base = {"run_id": name, "item_id": item.id, "driver": driver, "engine": "gemini"}
    if exp:
        base["experiment"] = exp
    rows.append(
        {
            **base,
            "variant_id": "",
            "condition": "text_twin",
            "error": "",
            "tool_calls": [],
            "scores": {v: {"passed": c == 1.0, "credit": c} for v, c in twin.items()},
        }
    )
    for vid, (tool, credit) in audio.items():
        rows.append(
            {
                **base,
                "variant_id": vid,
                "condition": "audio",
                "error": "",
                "tool_calls": [{"tool": tool, "args": {}}] if tool else [],
                "scores": {"passed": credit == 1.0, "credit": credit},
            }
        )
        rows.append(
            {
                **base,
                "variant_id": vid,
                "condition": "probe",
                "error": "",
                "tool_calls": [],
                "scores": {"answer": "x", "gold": "x", "passed": True},
            }
        )
    (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_analysis_intervention_and_rollouts(tmp_path):
    from voxparity.harness.exp_analysis import analyze, cell_sets, render_markdown

    item = _neutral_item()  # happy=neutral (gold a), angry=urgent (gold b)
    items = {item.id: item}
    cs = cell_sets(items)
    assert cs.cue == {(item.id, "angry")} and cs.neutral == {(item.id, "happy")}
    assert cs.clean_with_cue_gold == {(item.id, "happy"): {"b"}}
    runs = tmp_path / "runs"
    drv = "openrouter:google/gemini-3.7-flash"
    twin = {"happy": 1.0, "angry": 0.0}
    _write_run(
        runs,
        "20260915-final-gemini37or-gemini",
        drv,
        item,
        {"happy": ("a", 1.0), "angry": ("a", 0.0)},
        twin,
    )
    _write_run(
        runs,
        "20260915-final-cascadeopen-gemini",
        "cascade-open:x",
        item,
        {"happy": ("a", 1.0), "angry": ("a", 0.0)},
        twin,
    )
    listen = {"prompt_condition": "listen", "prompt_added_text": ex.LISTEN, "temperature": 0.0}
    _write_run(
        runs,
        "20260915-exp-listen-gemini37or-gemini",
        drv,
        item,
        {"happy": ("b", 0.0), "angry": ("b", 1.0)},
        twin,
        listen,
    )
    for k, passed in ((1, 1.0), (2, 0.0), (3, 1.0)):
        e = {"prompt_condition": "none", "temperature": 1.0, "rollout_index": k}
        _write_run(
            runs,
            f"20260915-exp-t1r{k}-gemini37or-gemini",
            drv,
            item,
            {"happy": ("a", 1.0), "angry": ("b" if passed else "a", passed)},
            twin,
            e,
        )
    data = analyze(runs, items)
    (iv,) = data["intervention"]
    assert iv["audio_delta_cue_bearing"]["mean"] == 1.0
    assert iv["audio_delta_neutral"]["mean"] == -1.0
    assert iv["over_reaction_delta_clean_cells"]["mean"] == 1.0  # paranoia shows up
    (ro,) = data["rollouts"]
    assert ro["k"] == 3 and ro["cells"] == 1
    assert ro["pass_hat_k"]["mean"] == 0.0 and ro["pass_at_k"]["mean"] == 1.0
    assert ro["disagreement_rate"]["mean"] == 1.0
    assert ro["t0_pass_rate"]["mean"] == 0.0
    assert "Prompt intervention" in render_markdown(data)


def test_bootstrap_matches_final_analysis_when_available():
    from voxparity.harness.exp_analysis import cluster_bootstrap

    fa = pytest.importorskip("voxparity.harness.final_analysis")
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 0.5]
    cl = ["a", "a", "b", "c", "c", "d"]
    assert cluster_bootstrap(vals, cl) == fa.cluster_bootstrap(vals, cl)


def test_replay_source_hash_helper_matches_store(tmp_path):
    """The replay driver compares sha256 of the clip bytes, which is exactly the
    store's content address."""
    item = make_item()
    store = _seed_store(tmp_path, item.id)
    rec = store.get(item.id, "happy", "test")
    data = store.audio_path(rec).read_bytes()
    assert hashlib.sha256(data).hexdigest() == rec.sha256
