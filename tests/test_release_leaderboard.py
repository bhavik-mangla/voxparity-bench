"""The README results table and the leaderboard page are generated, fresh, and equal Table A2."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_leaderboard", ROOT / "scripts/release/leaderboard.py"
)
assert _spec and _spec.loader
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)


def test_outputs_are_fresh() -> None:
    for path, text in lb.outputs(ROOT).items():
        assert path.exists(), f"{path} missing: run scripts/release/leaderboard.py"
        assert path.read_text() == text, f"{path} stale: run scripts/release/leaderboard.py"


def test_counts_match_paper() -> None:
    data = lb.load(ROOT)
    main, twinless, ref = lb.split(data)
    assert (len(main), len(twinless), len(ref)) == (23, 5, 3)
    assert sum(r["verdict"] == "passes" for r in main) == 11
    assert sum(r["verdict"] == "below" for r in twinless) == 3


def test_page_has_no_external_requests() -> None:
    page = (ROOT / lb.PAGE).read_text()
    assert "<script src" not in page and "<link" not in page and "@import" not in page
    assert "document.cookie" not in page and "localStorage" not in page


A2 = ROOT / lb.TABLE_A2


@pytest.mark.skipif(not A2.exists(), reason="paper source is not shipped publicly")
def test_rows_equal_table_a2() -> None:
    data = lb.load(ROOT)
    assert lb.check_against_a2(data, A2.read_text()) == []


@pytest.mark.skipif(not A2.exists(), reason="paper source is not shipped publicly")
def test_a2_check_catches_a_changed_number() -> None:
    data = copy.deepcopy(lb.load(ROOT))
    data["rows"][0]["gain"]["lo"] += 0.02
    errs = lb.check_against_a2(data, A2.read_text())
    assert any("vs floor" in e for e in errs)
