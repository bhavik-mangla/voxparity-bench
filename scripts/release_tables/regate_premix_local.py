"""Local-Whisper re-gate for a clip whose frozen ASR gate ran on a pre-mix render.

The Deepgram-free re-gate (scripts/regate_local_asr.py, D116(4)) skipped clips
gated on their caller's PRE-MIX render (verdict ``needs-premix-render``): the
pre-mix render is not retained in the store, and the mixed clip carries a
second voice by design. This script gates the stored MIXED clip with local
Whisper through the frozen gate (``stimuli.validate.asr_roundtrip_gate``) and
records the verdict on the manifest row as ``gates.asr_roundtrip.local_regate``
via ``StimulusStore.set_gate`` (merge-on-write, dirty-row aware). Admission is
NOT changed: ``passed`` on the frozen gate is left as it was; the local verdict
is evidence for the dataset card.

Two WERs are recorded:
  * ``wer_mix``: the mixed clip against the item transcript (the background
    line counts as insertion error);
  * ``wer_bed_stripped``: the same transcript with the scene's own known
    background line removed from the hypothesis (hypothesis-side strip of
    the manipulation, as D097 strips fillers). This is the gate verdict.

    uv run --extra local python scripts/release_tables/regate_premix_local.py \
        --bank "$VXP_BANK" --item vxp-kid911-0001 --variant laughter_bed

No network model calls (local mlx-audio Whisper only).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
from pathlib import Path

import yaml

from voxparity.schemas.item import Item
from voxparity.stimuli.store import StimulusStore
from voxparity.stimuli.validate import WER_GATE, asr_roundtrip_gate

HERE = Path(__file__).resolve().parents[2]


def _regate_module():  # reuse the D116 transcriber + written-form normaliser
    src = HERE / "scripts/regate_local_asr.py"
    spec = importlib.util.spec_from_file_location("regate_local_asr", src)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def strip_bed(hyp: str, bed_text: str) -> str:
    """Remove every occurrence of the bed line's word sequence (stage directions
    in parentheses dropped) from the hypothesis, longest-first, then any
    remaining repeat of its shortest repeated phrase."""
    bed = re.sub(r"\([^)]*\)", " ", bed_text)
    phrases = [p for p in re.split(r"[,.!?;]", bed) if _words(p)]
    out = " ".join(_words(hyp))
    for ph in sorted({" ".join(_words(p)) for p in phrases}, key=len, reverse=True):
        out = re.sub(rf"(?:^|\s){re.escape(ph)}(?=\s|$)", " ", out)
    return " ".join(out.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bank",
        type=Path,
        default=os.environ.get("VXP_BANK"),
        required="VXP_BANK" not in os.environ,
    )
    ap.add_argument("--item", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--engine", default="gemini")
    ap.add_argument("--model", default=None, help="mlx Whisper id (default large-v3-turbo)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    store = StimulusStore(a.bank / "stimuli")
    rec = store.get(a.item, a.variant, a.engine)
    if rec is None:
        raise SystemExit(f"no manifest row for {a.item}/{a.variant}/{a.engine}")
    path = store.audio_path(rec)  # verifies the hash
    item_file = a.bank / "items/pilot/t4" / f"{a.item}.yaml"  # the frozen bank's item
    item = Item.model_validate(yaml.safe_load(item_file.read_text()))
    scene = (rec.scene or {}) if isinstance(rec.scene, dict) else {}
    bed_text = str(scene.get("text") or "")

    mod = _regate_module()
    run = mod.local_transcriber("whisper", a.model)
    hyp = run(path)

    def gate(text: str):
        return min(
            (
                asr_roundtrip_gate(lambda _w, c=c: c, item, b"", None)
                for c in mod.hypothesis_candidates(text)
            ),
            key=lambda r: float(r.detail["wer"]),
        )

    mix = gate(hyp)
    stripped_hyp = strip_bed(hyp, bed_text) if bed_text else hyp
    strip = gate(stripped_hyp)
    result = {
        "asr_engine": f"local:{run.model_id}",  # type: ignore[attr-defined]
        "gated_on": "stored mixed clip (pre-mix render not retained)",
        "wer_mix": mix.detail["wer"],
        "wer_bed_stripped": strip.detail["wer"],
        "threshold": WER_GATE,
        "passed": bool(strip.passed),
        "transcript": hyp,
        "bed_stripped_as": stripped_hyp,
        "rule": "hypothesis-side strip of the scene's own background line (the manipulation)",
        "date": dt.date.today().isoformat(),
        "script": "scripts/release_tables/regate_premix_local.py",
    }
    head = {"item": a.item, "variant": a.variant, "sha256": rec.sha256}
    print(json.dumps({**head, **result}, indent=2))
    if not a.dry_run:
        g = dict(rec.gates.get("asr_roundtrip") or {})
        g["local_regate"] = result
        store.set_gate(a.item, a.variant, a.engine, "asr_roundtrip", g)
        print("recorded gates.asr_roundtrip.local_regate on the manifest row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
