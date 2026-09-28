"""Facts vs feelings within grounding strength (review item: grounding confound).

Reads docs/insights/emotion-dive.json (per-cell counts; no model calls) and
recomputes, for protective protocol-grounded cells, P(right | heard) and
P(words' action | heard) for feelings and facts separately within mandate-grounded
cells and within practice/permission-grounded cells, with the same
item-clustered bootstrap as emotion_dive.py (4000 resamples, seed 20260915).

    python3 scripts/insights/grounding_split.py

Writes docs/insights/grounding-split.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from voxparity import private_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emotion_dive import boot, boot_diff

INS = Path(__file__).resolve().parents[2] / "docs/insights"
GROUPS = {
    "mandate": {"mandate"},
    "mandate_incl_mixed": {"mandate", "mandate + permission"},
    "practice_or_permission": {"practice", "permission"},
}
# The paper grounds this US collections call in practice, not a mandate (Tables 2, 4).
# (item, variant) -> obligation; names a held-out cell, so it is private data.
OBLIGATION_OVERRIDE: Any = private_data.load(
    "insights/grounding_split.json",
    lambda d: {(i, v): o for i, v, o in d["obligation_override"]},
)
METRICS = {
    "right_given_heard": ("heard_correct", "heard"),
    "words_given_heard": ("heard_words", "heard"),
}


def per_item(cells: list[dict], pop: str, num: str, den: str) -> dict[str, tuple[float, float]]:
    out: dict[str, list[float]] = {}
    for r in cells:
        k = r["systems"][pop]
        v = out.setdefault(r["item"], [0.0, 0.0])
        v[0] += k[num]
        v[1] += k[den]
    return {i: (a, b) for i, (a, b) in out.items()}


def main() -> None:
    d = json.loads((INS / "emotion-dive.json").read_text())
    cells = [r for r in d["cells"] if not r["legacy"] and r["protective_cell"]]
    for r in cells:
        r["obligation"] = OBLIGATION_OVERRIDE.get((r["item"], r["variant"]), r["obligation"])
    res: dict = {
        "source": "docs/insights/emotion-dive.json",
        "scope": "protective, non-legacy cells",
        "obligation_override": [f"{i}/{v}: {o}" for (i, v), o in OBLIGATION_OVERRIDE.items()],
    }
    for g, obs in GROUPS.items():
        feel = [r for r in cells if r["cue_class"] == "feeling" and r["obligation"] in obs]
        fact = [r for r in cells if r["cue_class"] == "fact" and r["obligation"] in obs]
        res[g] = {"feeling_cells": len(feel), "fact_cells": len(fact)}
        for pop in ("frontier4", "all27"):
            res[g][pop] = {}
            for m, (num, den) in METRICS.items():
                a, b = per_item(feel, pop, num, den), per_item(fact, pop, num, den)
                res[g][pop][m] = {
                    "feelings": boot(a),
                    "facts": boot(b),
                    "feelings_minus_facts": boot_diff(a, b),
                }
    (INS / "grounding-split.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
