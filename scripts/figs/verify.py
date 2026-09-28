"""Print-size check for every figure PDF in paper/figures/.

For each PDF: its page size (the authored = printed width), the smallest and
largest text size in points (pymupdf spans), and a 300-dpi render of the PDF at
that true size into --out (default: a temp dir) for eyeballing.

    uv run --extra paper --with pymupdf python scripts/figs/verify.py [--out DIR]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pymupdf

FIG = Path(__file__).resolve().parents[2] / "paper" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or Path(tempfile.mkdtemp(prefix="vx-figs-"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'figure':28s} {'width in':>8s} {'height in':>9s} {'min pt':>7s} {'max pt':>7s}")
    for pdf in sorted(FIG.glob("*.pdf")):
        d = pymupdf.open(pdf)
        pg = d[0]
        sizes = [
            s["size"]
            for b in pg.get_text("dict")["blocks"]
            for ln in b.get("lines", [])
            for s in ln["spans"]
            if s["text"].strip()
        ]
        print(
            f"{pdf.stem:28s} {pg.rect.width / 72:8.2f} {pg.rect.height / 72:9.2f} "
            f"{min(sizes):7.2f} {max(sizes):7.2f}"
        )
        pg.get_pixmap(dpi=300).save(out / f"{pdf.stem}_300dpi.png")
    print("renders:", out)


if __name__ == "__main__":
    main()
