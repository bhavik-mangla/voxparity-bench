"""`voxparity items screen`: promote draft -> screened on a text-only leak screen.

The schema defines ``screened`` as "passed lexical-leak screen". Until now the
screen ran only inside ``items author``; every hand-authored item stayed
``draft`` forever. This module runs it over existing item files.

Two judge calls per item (plus one per followup reply):

1. LEAK (the shared authoring prompt, ``leak_screen``), DECISIVE, matching the
   schema's definition: the shared transcript, or any variant's followup reply,
   must not state or strongly imply an emotion.
2. DISCRIMINATION (advisory, D105). The judge reads the transcript + scenario
   and the item's own delivery conditions (shuffled, lettered) and says whether
   the words favour one. A counterfactual pair always has a natural reading;
   the paired audio-minus-twin design measures what a text reader gets for
   free, so the favoured variant is recorded as ``text_favored`` for analysis,
   not treated as a defect. (An earlier run on 2026-09-14 made this call
   decisive; cached verdicts are re-decided under the current rule.)

Rules: promote only on a clean pass; a judge error is UNMEASURED and leaves the
item draft (D046); items other than draft are never touched (never demoted);
HELD items are never screened or promoted.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from voxparity import private_data
from voxparity.authoring.generate import LEAK_PROMPT, leak_screen
from voxparity.schemas.item import DeliveryVariant, Item, SceneKind

# Items held as defective or pending grounding (id -> reason). Never screened, never
# promoted. The list names held-out items, so it ships with the bank's private data
# (voxparity.private_data); on a public checkout only the in-file markers apply.
HELD: dict[str, str] = private_data.load("src/screen_held.json", default={})
# In-file markers: an uppercase HELD tag or an explicit "held at review: draft".
_HELD_MARKER = re.compile(r"^#.*(\bHELD\b|held at review)", re.MULTILINE)

DISCRIM_PROMPT = """You are auditing one test item for a voice-agent benchmark. The item is only
valid if its WORDS do not give away how the caller sounds.

Context the agent already has: {scenario}

Written transcript of what the caller said: "{transcript}"

Candidate ways this was actually spoken on the call:
{conditions}

The same words are deliberately paired with deliveries that can contradict them:
polite or positive words said sarcastically, reluctantly, or under duress; calm,
coherent words said by an impaired, whispering or frightened caller; an adult errand
voiced by a child. A literal reading of the words is therefore NOT evidence against
the contradicting delivery. Point to a candidate only if the words themselves carry a
MARKER of it: they state or describe the feeling or condition, contain its verbal
signature (fillers, hesitations, insults, "fine, whatever"), or describe a situation
that only one delivery fits.

For EACH candidate rate 1-5 whether a real caller could plausibly say exactly these
words that way (1 = impossible given the words, 5 = entirely plausible). Then answer
with the candidate the words carry a marker of, or "cannot_tell" if none.
Return JSON: {{"fit": {{"<letter>": 1-5, ...}}, "answer": "cannot_tell" or a letter, "why": str}}"""

# Decisive rule: a leak if the judge names a candidate (the words carry its marker)
# OR rates some candidate implausible while another is plausible (gap >= 3).
FIT_GAP_LEAK = 3

PROMPT_HASH = hashlib.sha256((DISCRIM_PROMPT + "\x00" + LEAK_PROMPT).encode()).hexdigest()[:12]

PASS, LEAK, UNMEASURED, HELD_STATUS = "pass", "leak", "unmeasured", "held"


def held_reason(item_id: str, file_text: str = "") -> str | None:
    if item_id in HELD:
        return HELD[item_id]
    m = _HELD_MARKER.search(file_text)
    if m:
        return f"in-file marker: {m.group(0).lstrip('# ').strip()[:160]}"
    return None


def describe_condition(v: DeliveryVariant) -> str:
    """A label-level description of one variant's audio condition. Never includes
    background speech text — that is audio, not the caller's words."""
    emo = v.emotion.value
    parts = [
        "spoken in a neutral, even manner" if emo == "neutral" else f"spoken in a {emo} manner"
    ]
    if v.speaker is not None and v.speaker.value != "adult":
        parts.append(f"by {'a child' if v.speaker.value == 'child' else 'an elderly speaker'}")
    if v.scene is not None:
        s = v.scene
        if s.kind is SceneKind.SLOT_NOISE:
            parts.append(
                f"with a burst of noise over the words '{s.slot}' (the transcript shows "
                "what was said underneath)"
            )
        elif s.kind is SceneKind.TRUNCATION:
            parts.append(
                f"with the line cutting out so the final words '{s.slot}' are never heard "
                "(the transcript shows what the caller was saying)"
            )
        elif s.kind.value == "dtmf":
            parts.append("with telephone keypad tones on the line")
        elif s.asset.startswith("tts:"):
            parts.append("with a second person's voice audible in the background")
        else:
            name = s.asset.split(":", 1)[-1].replace("/", " ").replace("_", " ")
            parts.append(f"with background sound ({name})")
    if v.channel is not None:
        parts.append(f"over a {v.channel.codec} telephone channel")
    return ", ".join(parts)


def item_fingerprint(item: Item) -> str:
    """Hash of everything the judge sees; a stored verdict is reused only if unchanged."""
    blob = json.dumps(
        {
            "t": item.transcript,
            "s": item.scenario,
            "c": [describe_condition(v) for v in item.variants],
            "r": [v.followup.caller_reply if v.followup else None for v in item.variants],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Judge:
    """A JSON judge with an ordered fallback chain of (label, callable). Records which
    route answered each call. A route that fails is skipped for ``cooldown_s`` (a
    dead free bucket otherwise costs minutes of backoff on every call); once every
    route has failed the call raises RuntimeError — an outage, never a verdict."""

    routes: list[tuple[str, Callable[[str], Any]]]
    cooldown_s: float = 900.0
    used: list[str] = field(default_factory=list)
    _dead_until: dict[str, float] = field(default_factory=dict)

    def __call__(self, prompt: str) -> Any:
        import time

        errors = []
        now = time.monotonic()
        live = [r for r in self.routes if self._dead_until.get(r[0], 0.0) <= now]
        for label, fn in live or self.routes:
            try:
                out = fn(prompt)
            except RuntimeError as e:
                errors.append(f"{label}: {str(e)[:120]}")
                self._dead_until[label] = time.monotonic() + self.cooldown_s
                continue
            self.used.append(label)
            return out
        raise RuntimeError(" | ".join(errors) or "no judge routes")


def _letter(raw: Any, letters: dict[str, str]) -> str:
    """Normalize a judge's candidate reference ('A', 'a.', 'Candidate A', 'PATH_A')."""
    text = str(raw).strip()
    if text.lower().replace(" ", "_") in {"cannot_tell", "cant_tell"}:
        return "cannot_tell"
    m = re.fullmatch(r"(?:candidate|path|option)?[\s_:-]*([A-Za-z])\.?", text, re.IGNORECASE)
    return m.group(1).upper() if m and m.group(1).upper() in letters else text


def screen_item(item: Item, judge: Callable[[str], Any]) -> dict[str, Any]:
    """Screen one item. Returns a verdict record (status pass|leak|unmeasured)."""
    variants = list(item.variants)
    random.Random(item.id).shuffle(variants)  # no positional cue from authoring order
    letters = [chr(ord("A") + i) for i in range(len(variants))]
    letter_to_vid = dict(zip(letters, [v.variant_id for v in variants], strict=True))
    conditions = "\n".join(
        f"{L}. {describe_condition(v)}" for L, v in zip(letters, variants, strict=True)
    )
    rec: dict[str, Any] = {
        "item_id": item.id,
        "fingerprint": item_fingerprint(item),
        "prompt_hash": PROMPT_HASH,
        "conditions": {L: letter_to_vid[L] for L in letters},
    }
    try:
        d = judge(
            DISCRIM_PROMPT.format(
                scenario=item.scenario.strip(), transcript=item.transcript, conditions=conditions
            )
        )
        answer = _letter(d["answer"], letter_to_vid)
        if answer.lower().replace(" ", "_") in {"cannot_tell", "cant_tell"}:
            answer = "cannot_tell"
        elif answer not in letter_to_vid:
            raise ValueError(f"unparseable discrimination answer {answer!r}")
        fit = {_letter(k, letter_to_vid): int(v) for k, v in dict(d.get("fit") or {}).items()}
        if set(fit) != set(letter_to_vid):
            raise ValueError(f"fit ratings do not cover every condition: {fit}")
        rec["discrimination"] = {
            "answer": answer,
            "fit": {letter_to_vid[k]: v for k, v in fit.items()},
            "favored_variant": letter_to_vid.get(answer),
            "why": str(d.get("why", ""))[:400],
        }
        t = leak_screen(judge, item.transcript, item.scenario.strip())
        rec["transcript_leak"] = {
            "leaks_emotion": bool(t["leaks_emotion"]),
            "emotion": t.get("emotion"),
            "why": str(t.get("why", ""))[:400],
        }
        replies = []
        for v in item.variants:
            if v.followup is None:
                continue
            r = leak_screen(judge, v.followup.caller_reply, item.scenario.strip())
            replies.append(
                {
                    "variant_id": v.variant_id,
                    "reply": v.followup.caller_reply,
                    "leaks_emotion": bool(r["leaks_emotion"]),
                    "emotion": r.get("emotion"),
                    "why": str(r.get("why", ""))[:300],
                }
            )
        rec["reply_leaks"] = replies
    except (RuntimeError, KeyError, TypeError, ValueError) as e:
        rec["status"] = UNMEASURED
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return rec
    return decide(rec)


def decide(rec: dict[str, Any]) -> dict[str, Any]:
    """Status from stored judge answers (pure: re-scores cached verdicts for free).

    The schema defines ``screened`` as "passed lexical-leak screen", so the LEAK
    answers decide: the shared transcript stating or strongly implying an
    emotion, or any variant's followup reply doing so. The discrimination answer
    (which delivery the words favour) is ADVISORY, recorded as ``text_favored``.
    Every counterfactual pair has a natural reading of its words; the paired
    audio-minus-twin design already measures what a text reader gets for free,
    so a favoured reading is a property to report, not a defect (D105)."""
    if rec.get("status") == UNMEASURED:
        return rec
    reasons = []
    t = rec.get("transcript_leak") or {}
    if t.get("leaks_emotion"):
        reasons.append(f"transcript leaks {t.get('emotion')}")
    reasons += [
        f"reply of '{r['variant_id']}' leaks {r['emotion']}"
        for r in rec.get("reply_leaks") or []
        if r.get("leaks_emotion")
    ]
    d = rec.get("discrimination") or {}
    fit = d.get("fit") or {}
    favored = d.get("favored_variant")
    if not favored and fit and max(fit.values()) - min(fit.values()) >= FIT_GAP_LEAK:
        favored = max(fit, key=lambda k: fit[k])
    rec["text_favored"] = favored
    rec["status"] = LEAK if reasons else PASS
    rec.pop("leak_reasons", None)
    if reasons:
        rec["leak_reasons"] = reasons
    rec["advisory"] = favored is not None
    rec["rule"] = "schema-lexical-leak (D105)"
    return rec


_REVIEW_LINE = re.compile(r"^review: draft$", re.MULTILINE)


def promote_file(path: Path) -> None:
    """draft -> screened, surgically: hand-written YAML comments (grounding notes)
    must survive, so the file is edited as text, never re-dumped."""
    text = path.read_text()
    new, n = _REVIEW_LINE.subn("review: screened", text, count=1)
    if n != 1:
        raise ValueError(f"{path}: no top-level 'review: draft' line")
    new = new.replace("review=draft", "review=screened", 1)
    Item.model_validate(yaml.safe_load(new))  # never write an invalid item
    path.write_text(new)


def load_prior(report: Path) -> dict[str, dict[str, Any]]:
    """Latest terminal (pass/leak) verdict per item from an existing report."""
    prior: dict[str, dict[str, Any]] = {}
    if report.exists():
        for line in report.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") in (PASS, LEAK):
                    prior[r["item_id"]] = r
    return prior


def run_screen(
    files: list[Path],
    judge: Judge,
    report: Path,
    apply: bool = False,
    model_label: str = "",
    log: Callable[[str], None] = print,
    stop_after_outages: int = 3,
) -> dict[str, Any]:
    """Screen every draft item in ``files``; append verdicts to ``report`` (JSONL).

    Resumable: a stored pass/leak with the same fingerprint and prompt hash is
    reused without a judge call. Stops cleanly after ``stop_after_outages``
    consecutive unmeasured items (quota death), leaving the rest untouched."""
    report.parent.mkdir(parents=True, exist_ok=True)
    prior = load_prior(report)
    summary: dict[str, Any] = {
        "promoted": [], "passed": [], "leaked": [], "unmeasured": [], "held": [],
        "not_draft": [], "reused": 0, "stopped_early": False,
    }  # fmt: skip
    streak = 0
    for f in files:
        text = f.read_text()
        item = Item.model_validate(yaml.safe_load(text))
        if item.review != "draft":
            summary["not_draft"].append(item.id)
            continue
        reason = held_reason(item.id, text)
        if reason:
            summary["held"].append((item.id, reason))
            log(f"  HELD   {item.id}: {reason[:90]}")
            continue
        old = prior.get(item.id)
        fresh = old is not None and old["prompt_hash"] == PROMPT_HASH
        if old and fresh and old["fingerprint"] == item_fingerprint(item):
            rec = decide(dict(old))
            summary["reused"] += 1
        else:
            n_before = len(judge.used)
            rec = screen_item(item, judge)
            rec["judge_routes"] = judge.used[n_before:]
            rec["model"] = model_label
            rec["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
            rec["file"] = str(f)
            with report.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = rec["status"]
        if status == UNMEASURED:
            streak += 1
            summary["unmeasured"].append((item.id, rec.get("error", "")))
            log(f"  UNMEAS {item.id}: {rec.get('error', '')[:100]}")
            if streak >= stop_after_outages:
                summary["stopped_early"] = True
                log(f"  {streak} consecutive judge outages — stopping; re-run to resume")
                break
            continue
        streak = 0
        if status == LEAK:
            summary["leaked"].append((item.id, "; ".join(rec.get("leak_reasons", []))))
            log(f"  LEAK   {item.id}: {rec['leak_reasons']}")
            continue
        summary["passed"].append(item.id)
        if apply:
            promote_file(f)
            summary["promoted"].append(item.id)
        log(f"  PASS   {item.id}{' (advisory)' if rec.get('advisory') else ''}"
            f"{' -> screened' if apply else ''}")  # fmt: skip
    return summary
