"""Gemini-bias analysis: bootstrap parity with final_analysis, the sign of the
engine x model interaction, and rank-order tau."""

from __future__ import annotations

import numpy as np

from test_final_analysis import items, load_arm, rows_for, write_run  # noqa: F401
from voxparity.harness.final_analysis import cluster_bootstrap
from voxparity.harness.gemini_bias import _col, boot, interaction, kendall_tau_b, rank_order


def test_boot_matches_cluster_bootstrap():
    rng = np.random.default_rng(3)
    ids = [f"i{rng.integers(0, 25)}" for _ in range(120)]
    vals = rng.random(120)
    a = cluster_bootstrap(list(vals), ids)
    b = boot(ids, vals[:, None], _col(0))
    assert b is not None and a is not None
    assert (a["mean"], a["lo"], a["hi"]) == (b["mean"], b["lo"], b["hi"])
    assert b["mde80"] > 0


def test_kendall_tau_b_extremes():
    x = np.array([[1.0, 2.0, 3.0]])
    assert kendall_tau_b(x, x)[0] == 1.0
    assert kendall_tau_b(x, -x)[0] == -1.0


def _arm(tmp_path, items, label, engine, audio):  # noqa: F811
    rows = []
    for item_id, a in audio.items():
        rows += rows_for(item_id, engine=engine, audio=a)
    return load_arm(write_run(tmp_path, f"20260915-final-{label}-{engine}", rows), items)


def test_interaction_positive_when_gemini_arm_loses_more_off_gemini_tts(tmp_path, items):  # noqa: F811
    ok = {"happy": True, "angry": True}
    bad = {"happy": False, "angry": False}
    ids = ("vxp-test-0001", "vxp-test-0002")
    arms = [
        _arm(tmp_path, items, "gemini37or", "gemini", dict.fromkeys(ids, ok)),
        _arm(tmp_path, items, "gemini37or", "kokoro", dict.fromkeys(ids, bad)),
        _arm(tmp_path, items, "gptaudio", "gemini", dict.fromkeys(ids, ok)),
        _arm(tmp_path, items, "gptaudio", "kokoro", dict.fromkeys(ids, ok)),
    ]
    r = interaction(arms, items, ["gemini37or"], ["gptaudio"], ["kokoro"])
    assert r["n"] == 4
    assert r["diff_in_diff"]["mean"] == 1.0
    assert r["per_arm_gain"]["gptaudio"]["mean"] == 0.0


def test_rank_order_reports_tau_on_shared_cells(tmp_path, items):  # noqa: F811
    ids = ("vxp-test-0001", "vxp-test-0002", "vxp-test-0003")
    good, half = {"happy": True, "angry": True}, {"happy": True, "angry": False}
    arms = [
        _arm(tmp_path, items, "gemini37or", "gemini", dict.fromkeys(ids, good)),
        _arm(tmp_path, items, "gemini37or", "kokoro", dict.fromkeys(ids, half)),
        _arm(tmp_path, items, "gptaudio", "gemini", dict.fromkeys(ids, half)),
        _arm(tmp_path, items, "gptaudio", "kokoro", dict.fromkeys(ids, good)),
    ]
    r = rank_order(arms, items, labels=("gemini37or", "gptaudio"), engines=("kokoro",))
    assert r["n"] == 6
    assert r["kendall_tau_b"]["mean"] == -1.0
    assert r["rank_gemini_tts"] == ["gemini37or", "gptaudio"]
    assert r["rank_other_source"] == ["gptaudio", "gemini37or"]
