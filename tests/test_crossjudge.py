"""Cross-judge (independent re-judging of the frozen Gemini-TTS clips) and the
listener audit (spec §12 confirmations, rater-depth targets)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from voxparity.harness.crossjudge import (
    ClipJob,
    agreement,
    done_shas,
    judge_prompt,
    judge_row,
    lead_shift,
    run_xjudge,
    ser_verdict,
)
from voxparity.harness.final_analysis import Arm
from voxparity.harness.listener_audit import recording_validation

OPTIONS = ("Calm", "Angry and demanding", "Sad")


def job(tmp_path: Path, **kw: Any) -> ClipJob:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    base: dict[str, Any] = dict(
        item_id="vxp-x-0001",
        variant_id="angry",
        sha256="s1",
        path=wav,
        question="How does the caller sound?",
        options=OPTIONS,
        gold="Angry and demanding",
        emotion="angry",
        family="delivery:high-arousal-negative",
        axis="delivery",
    )
    base.update(kw)
    return ClipJob(**base)


def reply(text: str, cost: float = 0.0001) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"cost": cost},
        "provider": "OpenAI",
    }


# --------------------------------------------------------------------------- prompt parity


def test_cross_judge_asks_the_cue_judges_exact_prompt(tmp_path: Path) -> None:
    """The independent judge must answer what the Gemini gate answered, word for word."""
    from voxparity.providers.gemini import GeminiClient
    from voxparity.stimuli.validate import cue_check_gate

    seen: dict[str, str] = {}

    class Fake(GeminiClient):
        def __init__(self) -> None:  # no keys needed
            pass

        def _audio_prompt(self, instruction: str, wav_bytes: bytes) -> str:
            seen["prompt"] = instruction
            return "Calm"

    class Probe:
        question = "How does the caller sound?"
        options: ClassVar[list[str]] = list(OPTIONS)
        gold_by_variant: ClassVar[dict[str, str]] = {"angry": "Angry and demanding"}

    class Item:
        perception_probe = Probe()

    cue_check_gate(Fake(), Item(), "angry", b"")  # type: ignore[arg-type]
    assert seen["prompt"] == judge_prompt(job(tmp_path))
    assert "Judge ONLY the vocal delivery" in seen["prompt"]


# --------------------------------------------------------------------------- D046 on the judge


def test_outage_is_unmeasured_not_a_fail(tmp_path: Path) -> None:
    def boom(body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("HTTP 403: Key limit exceeded")

    row = judge_row(job(tmp_path), "openai/gpt-audio-mini", boom)
    assert row["verdict"] is None and row["passed"] is None
    assert "403" in row["error"]


def test_off_menu_answer_is_unmeasured(tmp_path: Path) -> None:
    row = judge_row(job(tmp_path), "m", lambda b: reply("furious, honestly"))
    assert row["verdict"] is None and row["passed"] is None and "off-menu" in row["error"]


def test_verdicts_and_audio_part(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []

    def post(body: dict[str, Any]) -> dict[str, Any]:
        bodies.append(body)
        return reply("angry and demanding.")

    row = judge_row(job(tmp_path), "m", post)
    assert row["passed"] is True and row["verdict"] == "Angry and demanding"
    assert row["cost"] == 0.0001
    parts = bodies[0]["messages"][0]["content"]
    assert parts[0]["text"] == judge_prompt(job(tmp_path))
    assert parts[1]["type"] == "input_audio" and parts[1]["input_audio"]["format"] == "wav"
    assert judge_row(job(tmp_path), "m", lambda b: reply("Calm"))["passed"] is False


def test_run_is_resumable_and_retries_outages(tmp_path: Path) -> None:
    out = tmp_path / "x.jsonl"
    jobs = [job(tmp_path, sha256="a"), job(tmp_path, sha256="b")]
    calls: list[str] = []

    def flaky(body: dict[str, Any]) -> dict[str, Any]:
        calls.append("x")
        if len(calls) == 2:
            raise RuntimeError("HTTP 429")
        return reply("Calm")

    first = run_xjudge(jobs, out, post=flaky, echo=lambda s: None)
    assert first["tally"] == {"fail": 1, "unmeasured": 1}
    assert done_shas(out) == {"a"}
    second = run_xjudge(jobs, out, post=lambda b: reply("Angry and demanding"), echo=lambda s: None)
    assert second["judged"] == 1 and second["tally"] == {"pass": 1}
    assert done_shas(out) == {"a", "b"}
    assert len(out.read_text().splitlines()) == 3


# --------------------------------------------------------------------------- SER limits (D044)


@pytest.mark.parametrize(
    ("kw", "tags", "expected"),
    [
        ({}, {"sensevoice": {"emotion_forced": "angry"}, "emotion2vec": {"top": "neutral"}}, True),
        (
            {},
            {"sensevoice": {"emotion_forced": "neutral"}, "emotion2vec": {"top": "neutral"}},
            False,
        ),
        ({"emotion": "urgent"}, {"sensevoice": {"emotion_forced": "angry"}}, None),
        ({"emotion": "resigned"}, {"sensevoice": {"emotion_forced": "sad"}}, None),
        ({"axis": "scene", "family": "scene:background", "emotion": "neutral"}, {}, None),
        ({"emotion": "anxious"}, {"emotion2vec": {"top": "fearful"}}, True),
        ({}, None, None),
    ],
)
def test_ser_verdict_only_where_the_heads_can_speak(
    tmp_path: Path, kw: dict[str, Any], tags: dict[str, Any] | None, expected: bool | None
) -> None:
    assert ser_verdict(job(tmp_path, **kw), tags)["verdict"] is expected


# --------------------------------------------------------------------------- statistics


def test_agreement_and_kappa() -> None:
    table = {
        ("i", str(n)): {"a": a, "b": b}
        for n, (a, b) in enumerate([(True, True), (True, False), (False, False), (None, True)])
    }
    r = agreement(table, "a", "b")
    assert r["n"] == 3 and r["agreement"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["a_pass_b_fail"] == 1 and r["a_fail_b_pass"] == 0


def _arm(label: str, audio: dict[tuple[str, str], float]) -> Arm:
    return Arm(run=label, label=label, engine="gemini", driver=label, path=Path("."), audio=audio)


def test_lead_shift_detects_selection_that_favours_the_gemini_family() -> None:
    keys = [(f"it{i}", "v") for i in range(40)]
    subset = set(keys[:20])
    # the Gemini arm leads only on cells OUTSIDE the independent judge's admissions
    gem = _arm("g", {k: (0.0 if k in subset else 1.0) for k in keys})
    oth = _arm("o", dict.fromkeys(keys, 0.0))
    casc = _arm("cascadeopen", dict.fromkeys(keys, 0.0))
    r = lead_shift([gem, oth, casc], casc, ["g"], ["o"], subset, set(keys), {}, part="all")
    assert r["lead_subset"]["mean"] == 0.0
    assert r["lead_universe"]["mean"] == 0.5
    assert r["shift"]["mean"] == -0.5 and r["shift"]["hi"] < 0


# --------------------------------------------------------------------------- spec §12


def _probe(player: str, item: str, variant: str, ok: bool, engine: str = "human") -> dict[str, Any]:
    return {
        "_player": player,
        "_session": "s",
        "item_id": item,
        "variant_id": variant,
        "engine": engine,
        "condition": "probe",
        "scores": {"passed": ok},
    }


def test_recording_validation_counts_confirmations_not_listeners() -> None:
    rows = [
        _probe("p1", "it", "a", True),
        _probe("p2", "it", "a", True),
        _probe("p3", "it", "a", False),
        _probe("p1", "it", "a", False),  # a repeat listen by p1 does not add a vote
        _probe("p4", "it", "a", True, engine="gemini"),  # the TTS clip is not the recording
        _probe("p1", "it", "b", True),
        _probe("p2", "it", "b", True),
        _probe("p3", "it", "b", True),
    ]
    out = recording_validation(rows, {("it", "a"), ("it", "b"), ("it2", "c")}, {})
    per = out["per_recording"]
    assert per["it/a"]["listeners"] == 3 and per["it/a"]["confirming"] == 2
    assert not per["it/a"]["passes_s12"] and per["it/b"]["passes_s12"]
    assert out["zero_listeners"] == ["it2/c"]
    assert out["pairs"]["it"] == {"variants": ["a", "b"], "heard_all": 3, "discriminated": 2}
    wo = recording_validation(rows, {("it", "b")}, {}, exclude_players=["p3"])
    assert wo["per_recording"]["it/b"]["confirming"] == 2 and wo["passing_s12"] == 0


def test_reads_real_jsonl_shape(tmp_path: Path) -> None:
    out = tmp_path / "x.jsonl"
    out.write_text(json.dumps({"sha256": "z", "verdict": None}) + "\n")
    assert done_shas(out) == set()
