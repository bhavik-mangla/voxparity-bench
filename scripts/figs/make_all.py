"""Regenerate every v3 paper figure into paper/figures/ (PDF + 300-dpi PNG).

    uv run --extra paper python scripts/figs/make_all.py            # figures only
    uv run --extra paper python scripts/figs/make_all.py --derive   # re-derive data/ first

The three derive_*.py scripts need the pinned bank (VXP_BANK; see voxparity.paths)
and the human game runs; their outputs are committed under paper/figures/data/, so
the figures themselves need only committed JSON. Exits non-zero if any two text
labels in any figure overlap.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import style  # noqa: E402


def main() -> int:
    if "--derive" in sys.argv:
        for d in ("derive_heard.py", "derive_harm.py", "derive_sector.py"):
            subprocess.run([sys.executable, str(HERE / d)], check=True)
    style.setup()
    import fig1_glance
    import fig1_overview
    import fig2_leaderboard
    import fig3_heard
    import fig4_asymmetry
    import fig5_facts_feelings
    import fig_appendix

    written = []
    for mod in (
        fig1_overview,
        fig1_glance,
        fig2_leaderboard,
        fig3_heard,
        fig4_asymmetry,
        fig5_facts_feelings,
        fig_appendix,
    ):
        written += mod.main()
    import captions

    captions.main()
    for p in written:
        print("figure", p.relative_to(style.ROOT))
    bad = {k: v for k, v in style.OVERLAPS.items() if v}
    for k, v in bad.items():
        print(f"OVERLAP in {k}: {v}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
