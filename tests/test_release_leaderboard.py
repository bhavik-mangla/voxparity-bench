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
    names = ("FRDCB", "REFILL", "CARSVC", "BNKGR", "DRIVE", "ALARMC", "BETPHN", "SEELON")
    assert set(caps) == {f"PAIR_{k}" for k in names}
    for cap in caps.values():
        assert page.count(lb.ascii_html(cap)) == 1
        assert "counts from the public dev-split records" not in cap
    assert caps["PAIR_FRDCB"].startswith("Of the 28 systems, 24 release the deposit")
    assert "13 still release it and 9 hold it" in caps["PAIR_FRDCB"]
    # Eight cards, the credit-union pair first; the hero diagram carries no result line.
    items = re.findall(r'<article class="pair" data-item="([^"]+)"', page)
    assert items == list(lb.PAIR_ITEMS)
    hero = page[page.index('<figure class="xdiag"') : page.index("</figure>")]
    assert 'class="res"' not in hero and 'href="#pairs"' in hero
    # Credit-union clips play in the hero and on their card; every other clip once.
    srcs = re.findall(r'data-src="audio/([^"]+)"', page)
    sources = json.loads((ROOT / lb.AUDIO_SOURCES).read_text())
    assert len(srcs) == 18 and set(srcs) == set(sources) and len(sources) == 16
    assert all(srcs.count(s) == (2 if s.startswith("frdcb") else 1) for s in set(srcs))
    # Findings come before the examples, in the page and in the nav.
    assert page.index('<section id="findings"') < page.index('<section id="how"')
    assert page.index('href="#findings"') < page.index('href="#how"')
    assert {s["item"] for s in sources.values()} == set(lb.PAIR_ITEMS)
    assert "wire" not in page


def test_page_sources_are_ascii() -> None:
    """Typography ships as character references / \\u escapes, immune to UTF-8 mangling."""
    for name in ("index.html", "app.js", "style.css", "leaderboard.data.js", "leaderboard.json"):
        assert (ROOT / "docs" / name).read_bytes().isascii(), name
    assert lb.PAGE_TEMPLATE.isascii()
    page = (ROOT / lb.PAGE).read_text()
    assert re.search(r'<head>\s*<meta charset="utf-8">', page)
    assert "I&rsquo;ve run completely out of my inhaler" in page
    assert lb.ascii_html("child\u2019s \u2192") == "child&rsquo;s &rarr;"


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
