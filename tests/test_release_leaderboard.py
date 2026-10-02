"""The README results table and the leaderboard page are generated, fresh, and equal Table A2."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
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
    docs = ROOT / "docs"
    page = (docs / "index.html").read_text()
    # Scripts and stylesheets are local files only; no fonts, analytics or CDNs.
    tag = r"<(?:script|link|img|audio|source|iframe)\b[^>]*\b(?:src|href)"
    refs = re.findall(tag + r'="([^"]+)"', page)
    assert refs and all(not re.match(r"[a-z]+:|//", u) for u in refs), refs
    for name in ("index.html", "app.js", "style.css"):
        text = (docs / name).read_text()
        assert "@import" not in text and "document.cookie" not in text, name
        assert not re.search(r"url\(\s*['\"]?(https?:)?//", text), name
        assert "fetch(" not in text and "XMLHttpRequest" not in text, name
    for src in re.findall(r'data-src="([^"]+)"', page):
        assert (docs / src).exists(), src


def test_page_assets_and_captions() -> None:
    assert lb.check_audio_sources(ROOT) == []
    page = (ROOT / lb.PAGE).read_text()
    assert "@@" not in page
    assert page.count(f'content="{lb.OG_IMAGE}"') == 2
    assert "huggingface.co/spaces/bhavikmangla/voxparity-leaderboard" in page
    caps = lb.pair_captions(lb.pair_counts(ROOT))
    assert set(caps) == {"HERO_FRDCB"} | {
        f"PAIR_{k}" for k in ("REFILL", "CARSVC", "BNKGR", "DRIVE", "ALARMC", "BETPHN", "SEELON")
    }
    for cap in caps.values():
        assert page.count(cap) == 1
    # Every playable clip appears once, each has provenance, and the removed pair is gone.
    srcs = re.findall(r'data-src="audio/([^"]+)"', page)
    sources = json.loads((ROOT / lb.AUDIO_SOURCES).read_text())
    assert len(srcs) == len(set(srcs)) == 16 and set(srcs) == set(sources)
    assert {s["item"] for s in sources.values()} == set(lb.PAIR_ITEMS)
    assert "wire" not in page


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
