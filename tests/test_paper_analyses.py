"""Paper analyses: axis mapping, the four-way failure taxonomy, conditional-gap
bootstrap, Spearman/Holm/d' helpers, coverage eligibility, and an end-to-end run
(with figures when matplotlib is installed) on synthetic arms."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from test_final_analysis import rows_for, write_run
from test_human_baseline import human_row
from test_schemas import make_item
from voxparity.harness.final_analysis import headline_row, load_arm
from voxparity.harness.paper_analyses import (
    Context,
    HumanData,
    cell_outcome,
    conditional_gap,
    dprime,
    holm,
    item_family,
    run_all,
    spearman,
    split_half,
    taxonomy_axis,
)
from voxparity.schemas.item import (
    DeliveryVariant,
    GoldAction,
    PerceptionProbe,
    SceneKind,
    SceneSpec,
)

FREEZE = {"run": {"gemini": {"cells_expected": 5}}}


def _scene_item(item_id: str, asset: str, kind: SceneKind = SceneKind.BACKGROUND) -> Any:
    scene = SceneSpec(
        kind=kind,
        asset=asset,
        text="say yes" if asset.startswith("tts:") else None,
        slot="book it" if kind == SceneKind.SLOT_NOISE else None,
    )
    return make_item(
        id=item_id,
        variants=[
            DeliveryVariant(
                variant_id="clean",
                emotion="neutral",
                intensity=0.5,
                gold=GoldAction(tool="a", args={"x": ["1"]}, rationale="r"),
            ),
            DeliveryVariant(
                variant_id="scene",
                emotion="neutral",
                intensity=0.5,
                scene=scene,
                gold=GoldAction(tool="b", rationale="r"),
            ),
        ],
        perception_probe=PerceptionProbe(
            question="q?",
            options=["quiet", "noise"],
            gold_by_variant={"clean": "quiet", "scene": "noise"},
        ),
    )


def test_taxonomy_axis_splits_scene_kinds_and_skips_neutral():
    assert taxonomy_axis(_scene_item("vxp-t-0001", "tts:prompter"), "scene") == "second-speaker"
    assert taxonomy_axis(_scene_item("vxp-t-0002", "synth:co_alarm"), "scene") == (
        "scene (environmental)"
    )
    slot = _scene_item("vxp-t-0003", "synth:white_noise", SceneKind.SLOT_NOISE)
    assert taxonomy_axis(slot, "scene") == "slot-noise"
    assert taxonomy_axis(slot, "clean") is None  # neutral delivery, no scene
    emo = make_item()
    assert taxonomy_axis(emo, "angry") == "delivery emotion"
    assert item_family("vxp-wire-0001") == "wire"


def test_cell_outcome_four_way(tmp_path):
    items = {"vxp-test-0001": make_item()}
    rows = rows_for(
        "vxp-test-0001",
        audio={"happy": True, "angry": False},
        probe={"happy": "happy", "angry": "angry"},
    )
    arm = load_arm(write_run(tmp_path, "20260915-final-m-gemini", rows), items)
    item = items["vxp-test-0001"]
    assert cell_outcome(arm, ("vxp-test-0001", "happy"), item) == "correct"
    # probe right, took tool 'a' = the sibling's gold -> the cue did not move the action
    assert cell_outcome(arm, ("vxp-test-0001", "angry"), item) == "perceived, not acted"
    arm.probe[("vxp-test-0001", "angry")] = False
    assert cell_outcome(arm, ("vxp-test-0001", "angry"), item) == "not perceived"
    arm.probe[("vxp-test-0001", "angry")] = True
    arm.audio_rows[("vxp-test-0001", "angry")]["tool_calls"] = [{"tool": "escalate_to_human"}]
    assert cell_outcome(arm, ("vxp-test-0001", "angry"), item) == "perceived, acted wrong"
    del arm.probe[("vxp-test-0001", "angry")]
    assert cell_outcome(arm, ("vxp-test-0001", "angry"), item) == "wrong (perception n/a)"


def test_conditional_gap_point_and_interval():
    rows = [(f"i{i}", True, 1.0) for i in range(10)] + [(f"i{i}", False, 0.0) for i in range(10)]
    out = conditional_gap(rows)
    assert out["p_given_true"] == 1.0 and out["p_given_false"] == 0.0
    assert out["gap"]["mean"] == 1.0 and out["gap"]["lo"] == 1.0
    assert conditional_gap([("i1", True, 1.0)])["gap"] is None  # no "false" cells


def test_spearman_holm_dprime():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None  # no variance
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}  # step-down, monotone
    d = dprime(50, 100, 50, 100)
    assert d["d_prime"] == pytest.approx(0.0) and d["criterion"] == pytest.approx(0.0)
    assert dprime(0, 0, 1, 2)["d_prime"] is None


def test_split_half_is_none_with_too_few_items():
    items = {"vxp-test-0001": make_item()}
    h = HumanData([human_row("vxp-test-0001", "happy", "a", 1.0)], items)
    assert split_half(h)["r_split"] is None


def _arms(tmp_path: Path, items: dict[str, Any]) -> list[Any]:
    full = []
    part = []
    for i in items:
        full += rows_for(i, audio={"happy": True, "angry": i.endswith("1")})
        part += rows_for(i, audio={"happy": True, "angry": False}) if i.endswith("1") else []
    casc = []
    for i in items:
        casc += rows_for(i, audio={"happy": True, "angry": False}, probe_na=True)
    for r in casc:
        r["driver"] = "cascade-open:x"
    return [
        load_arm(write_run(tmp_path, "20260915-final-good-gemini", full), items),
        load_arm(write_run(tmp_path, "20260915-final-half-gemini", part), items),
        load_arm(write_run(tmp_path, "20260915-final-cascadeopen-gemini", casc), items),
    ]


def test_eligibility_excludes_partial_arms(tmp_path):
    items = {f"vxp-test-000{i}": make_item(id=f"vxp-test-000{i}") for i in (1, 2)}
    freeze = {"run": {"gemini": {"cells_expected": 10}}}
    ctx = Context(_arms(tmp_path, items), items, freeze)
    assert {a.label for a in ctx.eligible} == {"good", "cascadeopen"}
    assert [a.label for a in ctx.partial] == ["half"]
    assert ctx.cascade is not None and ctx.cascade.label == "cascadeopen"


def test_run_all_end_to_end_and_figures(tmp_path):
    items = {f"vxp-test-000{i}": make_item(id=f"vxp-test-000{i}") for i in (1, 2, 3)}
    arms = _arms(tmp_path, items)
    other = []
    for i in items:
        other += rows_for(i, audio={"happy": i.endswith("2"), "angry": True})
    arms.append(load_arm(write_run(tmp_path, "20260915-final-other-gemini", other), items))
    freeze = {"run": {"gemini": {"cells_expected": 15}}}
    headline = [headline_row(a, items, frozenset()) for a in arms]
    human = [
        human_row(i, v, "a", 1.0 if v == "happy" else 0.0, player=p)
        for i in items
        for v in ("happy", "angry")
        for p in ("p1", "p2")
    ]
    data = run_all(arms, items, freeze, human, headline)
    tax = {r["label"]: r for r in data["taxonomy"]}
    good = tax["good"]["all_cue_bearing"]
    assert good["n"] == 6 and good["counts"]["correct"] == 4
    assert tax["cascadeopen"]["perception_measured"] is False
    rb = {r["label"]: r for r in data["robustness"]["arm_vs_cascade"]}
    assert set(rb) == {"good", "other"} and rb["good"]["p_holm"] is not None
    assert data["human"]["agreement"]["tool_choice"]["units"] == 6
    assert data["psychometrics"]["cells"] == 6
    pytest.importorskip("matplotlib")
    from voxparity.harness.paper_figures import render_figures

    files = render_figures(data, tmp_path / "figs")
    assert "fig1_failure_taxonomy.pdf" in files
    first = (tmp_path / "figs" / "fig1_failure_taxonomy.pdf").read_bytes()
    render_figures(data, tmp_path / "figs")
    assert (tmp_path / "figs" / "fig1_failure_taxonomy.pdf").read_bytes() == first  # deterministic
    assert not math.isnan(data["dissociation"]["arms"][0].get("hu", {}).get("mean", 0.0) or 0.0)
