"""Listener audit of the game data: spec §12 validation of the author's
recordings, a factual profile of the most prolific player, and the rater-depth
target list for the game's coverage-aware sampler.

Spec §12 (blueprint author-as-human rules): an author recording needs >= 3
independent listeners CONFIRMING the intended cue is recognisable. D114 counted
listeners; this module counts CONFIRMATIONS — players who heard the recording
and chose the item's probe gold for it — and reports pair discrimination for
recorded pairs (both halves labelled right by the same listener).

Sources: ``voxparity human`` imports (scored probe/audio rows; player = the
first field of the ``human:<player8>-<session8>`` driver) for correctness, and
the raw trials JSONL for timestamps. Author-batch sessions never reach the
imports (spec §12); ``exclude_players`` removes further browser ids (e.g. to
test sensitivity to one prolific player). No accusation is implied by any
profile field: it is timing and answer data for Bhavik to confirm or deny.
"""

from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from voxparity.harness.final_analysis import classify, cluster_bootstrap, hu_bootstrap, md_table
from voxparity.harness.report import load_records

MIN_CONFIRMING = 3  # spec §12
# repo-relative, so the analysis finds it when run from the pinned bank worktree
DEFAULT_EQUIVALENCES = (
    Path(__file__).resolve().parents[3] / "docs/internal/listener-equivalences.yaml"
)


def load_equivalences(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """``item/variant`` -> {accept: [options], misspecified: bool, ...}."""
    if path is None or not path.exists():
        return {}
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    out = {}
    for k, v in raw.items():
        item, _, variant = str(k).partition("/")
        out[(item, variant)] = {
            "accept": list((v or {}).get("accept") or []),
            "misspecified": bool((v or {}).get("misspecified")),
            "corrected_gold": (v or {}).get("corrected_gold"),
            "reason": (v or {}).get("reason"),
        }
    return out


# --------------------------------------------------------------------------- loading


def player_of(driver: str) -> tuple[str, str]:
    name = driver.split(":", 1)[1] if ":" in driver else driver
    player, _, session = name.rpartition("-")
    return (player or name), session


def load_game_rows(human_glob: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(glob.glob(human_glob)):
        d = Path(p)
        if (d / "records.jsonl").exists():
            for r in load_records(d):
                if r.get("error"):
                    continue
                player, session = player_of(str(r.get("driver", "")))
                rows.append({**r, "_player": player, "_session": session})
    return rows


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- §12


def _confirms(r: dict[str, Any], eq: dict[tuple[str, str], dict[str, Any]]) -> bool:
    s = r.get("scores") or {}
    if s.get("passed"):
        return True
    e = eq.get((r["item_id"], r.get("variant_id") or ""))
    return e is not None and s.get("answer") in e["accept"]


def recording_validation(
    rows: list[dict[str, Any]],
    recordings: Iterable[tuple[str, str]],
    items_by_id: dict[str, Any],
    exclude_players: Iterable[str] = (),
    equivalences: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per author recording: independent listeners, confirmations, verdict.

    A confirmation is the gold or an equivalent description of the intended cue
    (``equivalences``; none given = gold only, the pre-Sep-26 strict count)."""
    excl = set(exclude_players)
    eq = equivalences or {}
    heard: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for r in rows:
        if r.get("engine") != "human" or r.get("condition") != "probe":
            continue
        if r["_player"] in excl:
            continue
        key = (r["item_id"], r.get("variant_id") or "")
        # one listener = one vote: the first answer a player gave on this clip
        heard[key].setdefault(r["_player"], _confirms(r, eq))
    per: dict[str, dict[str, Any]] = {}
    for i, v in sorted(set(recordings)):
        votes = heard.get((i, v), {})
        item = items_by_id.get(i)
        k = len(item.perception_probe.options) if item is not None else None
        n, ok = len(votes), sum(votes.values())
        per[f"{i}/{v}"] = {
            "listeners": n,
            "confirming": ok,
            "share": None if not n else round(ok / n, 3),
            "options": k,
            "chance": None if not k else round(1 / k, 3),
            "family": classify(item, v)[2] if item is not None else "unknown",
            "passes_s12": ok >= MIN_CONFIRMING,
            "passes_s12_and_majority": ok >= MIN_CONFIRMING and ok * 2 > n,
            "confirming_players": sorted(p for p, c in votes.items() if c),
        }
    # pair discrimination: items with >= 2 recorded variants; a listener who heard
    # two halves and labelled both right discriminates that pair
    by_item: dict[str, list[str]] = defaultdict(list)
    for i, v in set(recordings):
        by_item[i].append(v)
    pairs: dict[str, dict[str, Any]] = {}
    for i, vs in sorted(by_item.items()):
        if len(vs) < 2:
            continue
        vs = sorted(vs)
        both = set.intersection(*(set(heard.get((i, v), {})) for v in vs))
        disc = [p for p in both if all(heard[(i, v)][p] for v in vs)]
        pairs[i] = {"variants": vs, "heard_all": len(both), "discriminated": len(disc)}
    vals = list(per.values())
    return {
        "excluded_players": sorted(excl),
        "n": len(per),
        "with_3plus_listeners": sum(x["listeners"] >= MIN_CONFIRMING for x in vals),
        "passing_s12": sum(x["passes_s12"] for x in vals),
        "passing_s12_and_majority": sum(x["passes_s12_and_majority"] for x in vals),
        "zero_listeners": sorted(k for k, x in per.items() if x["listeners"] == 0),
        "below_3_confirming": {
            k: {kk: x[kk] for kk in ("listeners", "confirming", "family")}
            for k, x in per.items()
            if not x["passes_s12"]
        },
        "pairs": pairs,
        "pairs_with_a_discriminating_listener": sum(p["discriminated"] > 0 for p in pairs.values()),
        "per_recording": per,
    }


def recognition(
    rows: list[dict[str, Any]],
    recordings: Iterable[tuple[str, str]],
    items_by_id: dict[str, Any],
    equivalences: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Listener recognition of the author recordings, reported the way affective-
    speech corpora report perceptual validation (RAVDESS, CREMA-D): proportion
    correct against chance, per recording and per cue class, Wagner's unbiased
    hit rate Hu, and a chance-corrected agreement statistic. One listener = the
    first answer a player gave on a clip. Intervals are item-clustered bootstraps.

    ``strict`` counts the gold only; ``lenient`` also accepts an equivalent
    description of the intended cue (and scores mis-specified probes against the
    corrected wording), see ``load_equivalences``."""
    from voxparity.scoring.distribution import krippendorff_alpha
    from voxparity.scoring.stats import cue_class_map

    eq = equivalences or {}
    rec = set(recordings)
    first: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        key = (r["item_id"], r.get("variant_id") or "")
        if r.get("engine") != "human" or r.get("condition") != "probe" or key not in rec:
            continue
        first[key].setdefault(r["_player"], r)
    strict: list[float] = []
    lenient: list[float] = []
    chance: list[float] = []
    clusters: list[str] = []
    fam_of: list[str] = []
    triples_strict: list[tuple[str, str, str]] = []
    triples_lenient: list[tuple[str, str, str]] = []
    per: dict[str, dict[str, Any]] = {}
    labels_by_clip: list[list[str]] = []
    for key in sorted(first):
        item = items_by_id.get(key[0])
        if item is None:
            continue
        cmap = cue_class_map(item)
        k = len(item.perception_probe.options)
        gold = item.perception_probe.gold_by_variant.get(key[1])
        fam = classify(item, key[1])[2]
        n = ok_s = ok_l = 0
        answers = []
        for r in first[key].values():
            ans = (r.get("scores") or {}).get("answer")
            answers.append(str(ans))
            s_ok = bool((r.get("scores") or {}).get("passed"))
            l_ok = _confirms(r, eq)
            n, ok_s, ok_l = n + 1, ok_s + s_ok, ok_l + l_ok
            strict.append(float(s_ok))
            lenient.append(float(l_ok))
            chance.append(1 / k)
            clusters.append(key[0])
            fam_of.append(fam)
            gcls = cmap.get(gold or "", "other")
            triples_strict.append((key[0], gcls, cmap.get(str(ans), "other")))
            triples_lenient.append((key[0], gcls, gcls if l_ok else cmap.get(str(ans), "other")))
        labels_by_clip.append(answers)
        per[f"{key[0]}/{key[1]}"] = {
            "family": fam,
            "listeners": n,
            "strict": round(ok_s / n, 3),
            "lenient": round(ok_l / n, 3),
            "chance": round(1 / k, 3),
            "misspecified": bool((eq.get(key) or {}).get("misspecified")),
        }

    def summ(idx: list[int]) -> dict[str, Any]:
        return {
            "answers": len(idx),
            "recordings": len({clusters[i] + "/" + fam_of[i] for i in idx}),
            "strict": cluster_bootstrap([strict[i] for i in idx], [clusters[i] for i in idx]),
            "lenient": cluster_bootstrap([lenient[i] for i in idx], [clusters[i] for i in idx]),
            "chance": round(sum(chance[i] for i in idx) / len(idx), 3) if idx else None,
        }

    by_family = {f: summ([i for i, x in enumerate(fam_of) if x == f]) for f in sorted(set(fam_of))}
    alpha_rows = [a for a in labels_by_clip if len(a) >= 2]
    vals = list(per.values())
    return {
        "recordings_heard": len(per),
        "listeners_per_recording": {
            "min": min((x["listeners"] for x in vals), default=0),
            "median": median([x["listeners"] for x in vals]) if vals else 0,
            "max": max((x["listeners"] for x in vals), default=0),
        },
        "all_with_3plus_listeners": all(x["listeners"] >= MIN_CONFIRMING for x in vals),
        "pooled": summ(list(range(len(strict)))),
        "hu_strict": hu_bootstrap(triples_strict),
        "hu_lenient": hu_bootstrap(triples_lenient),
        "krippendorff_alpha_labels": round(krippendorff_alpha(alpha_rows), 3)
        if alpha_rows
        else None,
        "recordings_above_chance_strict": sum(x["strict"] > x["chance"] for x in vals),
        "recordings_majority_lenient": sum(x["lenient"] > 0.5 for x in vals),
        "by_family": by_family,
        "misspecified_probes": sorted(f"{i}/{v}" for (i, v), e in eq.items() if e["misspecified"]),
        "per_recording": per,
    }


# --------------------------------------------------------------------------- player profile


def player_profile(
    trials_path: Path,
    rows: list[dict[str, Any]],
    player: str,
    review_path: Path | None = None,
    record_meta: Path | None = None,
    recordings: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Timing, answer and overlap facts about one browser id (no inference)."""
    raw = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
    trials = [r for r in raw if r.get("kind") == "trial" and r.get("player")]
    mine = [r for r in trials if str(r["player"]).startswith(player)]
    others = [r for r in trials if not str(r["player"]).startswith(player)]
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in mine:
        sessions[r["session"]].append(r)
    all_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in trials:
        all_sessions[r["session"]].append(r)
    spans = []
    for sid, rs in all_sessions.items():
        t = sorted(_ts(r["received"]) for r in rs)
        spans.append((t[0], t[-1], str(rs[0]["player"])[:8], sid[:8], len(rs)))
    spans.sort()

    def sess_row(sid: str, rs: list[dict[str, Any]]) -> dict[str, Any]:
        t = sorted(_ts(r["received"]) for r in rs)
        # neighbours: the sessions (any player) that ended/started closest around it
        before = [s for s in spans if s[1] <= t[0] and s[3] != sid[:8]]
        after = [s for s in spans if s[0] >= t[-1] and s[3] != sid[:8]]
        return {
            "session": sid[:8],
            "consent_at": rs[0].get("consent_at"),
            "first_trial": t[0].isoformat(),
            "last_trial": t[-1].isoformat(),
            "minutes": round((t[-1] - t[0]).total_seconds() / 60, 1),
            "trials": len(rs),
            "age_bracket": rs[0].get("age_bracket"),
            "mode": rs[0].get("mode"),
            "batch": rs[0].get("batch"),
            "previous_session": _nb(before[-1], t[0], "before") if before else None,
            "next_session": _nb(after[0], t[-1], "after") if after else None,
        }

    def stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
        def med(key: str) -> float | None:
            v = [float(r[key]) for r in rs if isinstance(r.get(key), int | float)]
            return round(median(v), 1) if v else None

        return {
            "trials": len(rs),
            "median_action_ms": med("action_ms"),
            "median_probe_ms": med("probe_ms"),
            "median_plays": med("plays"),
            "engines": dict(Counter(str(r.get("engine")) for r in rs)),
            "age_brackets": dict(Counter(str(r.get("age_bracket")) for r in rs)),
        }

    # accuracy from the scored imports
    def acc(pred: Any) -> dict[str, Any]:
        probe = [
            bool((r.get("scores") or {}).get("passed"))
            for r in rows
            if r["condition"] == "probe" and pred(r)
        ]
        act = [
            float((r.get("scores") or {}).get("selection_credit") or 0.0)
            for r in rows
            if r["condition"] == "audio" and pred(r)
        ]
        return {
            "probe_acc": None if not probe else round(sum(probe) / len(probe), 3),
            "probe_n": len(probe),
            "action_credit": None if not act else round(sum(act) / len(act), 3),
            "action_n": len(act),
        }

    rec = set(recordings)
    out: dict[str, Any] = {
        "player": player,
        "share_of_trials": round(len(mine) / max(1, len(trials)), 3),
        "sessions": [
            sess_row(s, rs)
            for s, rs in sorted(sessions.items(), key=lambda x: min(r["received"] for r in x[1]))
        ],
        "device_hints": {
            "user_agent": "not collected by the game",
            "ip": "not stored (web/src/routes/api/trials/+server.ts)",
            "age_bracket": sorted({str(r.get("age_bracket")) for r in mine}),
        },
        "answer_style": {"player": stats(mine), "everyone_else": stats(others)},
        "accuracy": {
            "player": acc(lambda r: r["_player"] == player),
            "everyone_else": acc(lambda r: r["_player"] != player),
        },
        "heard_author_recordings": sorted(
            {
                f"{r['item_id']}/{r.get('variant_id')}"
                for r in mine
                if r.get("engine") == "human" and (r["item_id"], r.get("variant_id")) in rec
            }
        ),
    }
    if review_path is not None and review_path.exists():
        out["vs_bhavik_review"] = review_agreement(review_path, mine, others)
    if record_meta is not None and record_meta.exists():
        meta = [json.loads(line) for line in record_meta.read_text().splitlines() if line.strip()]
        stamps = sorted(m["recorded_at"] for m in meta if m.get("recorded_at"))
        out["recording_sessions_local_time"] = {
            "first": stamps[0] if stamps else None,
            "last": stamps[-1] if stamps else None,
            "dates": dict(Counter(s[:10] for s in stamps)),
            "note": "recorded_at is naive local time (the Mac is IST, UTC+05:30); "
            "game times are UTC",
        }
    return out


def _nb(span: tuple[Any, ...], t: datetime, side: str) -> dict[str, Any]:
    gap = (t - span[1]).total_seconds() if side == "before" else (span[0] - t).total_seconds()
    return {"player": span[2], "session": span[3], "trials": span[4], "gap_s": int(gap)}


def review_agreement(
    review_path: Path, mine: list[dict[str, Any]], others: list[dict[str, Any]]
) -> dict[str, Any]:
    """Probe-answer agreement with Bhavik's review-console answers on shared clips.

    The review console and the game share clip ids (cid). Same-person answers
    weeks apart need not agree, so this is evidence in either direction, not a
    test."""
    rev = json.loads(review_path.read_text()).get("answers") or {}
    ref = {
        k.split(":", 1)[1]: str(v.get("answer", "")).strip().lower()
        for k, v in rev.items()
        if k.startswith("listen:")
    }

    def rate(rs: list[dict[str, Any]]) -> dict[str, Any]:
        hits = [
            str(r.get("probe_answer") or "").strip().lower() == ref[r["cid"]]
            for r in rs
            if r.get("cid") in ref
        ]
        return {
            "shared_clips": len(hits),
            "same_answer": sum(hits),
            "rate": None if not hits else round(sum(hits) / len(hits), 3),
        }

    return {"player": rate(mine), "everyone_else": rate(others), "review_clips": len(ref)}


# --------------------------------------------------------------------------- rater depth


def rater_targets(
    rows: list[dict[str, Any]],
    bundle: dict[str, Any],
    items_by_id: dict[str, Any],
    s12: dict[str, Any],
    headline_cells: set[tuple[str, str, str]],
    priority_cells: Iterable[tuple[str, str]] = (),
    target: int = MIN_CONFIRMING,
) -> dict[str, Any]:
    """Cells short of ``target`` independent raters, tiered by what they unblock.

    Tier 1: author recordings that fail spec §12 (need confirming listeners).
    Tier 2: named cells (e.g. negatively discriminating golds) under target.
    Tier 3: cue-bearing headline Gemini-TTS cells no human has ruled on.
    Tier 4: other cue-bearing headline cells. Tier 5: neutral headline cells.
    Tier 6: bundle cells outside the frozen headline."""
    raters: dict[str, set[str]] = defaultdict(set)
    by_cell: dict[tuple[str, str, str], str] = {}
    for item in bundle["items"]:
        for v in item["variants"]:
            by_cell[(item["id"], v["variant_id"], v["engine"])] = v["cid"]
    for r in rows:
        if r["condition"] != "audio":
            continue
        cid = by_cell.get((r["item_id"], r.get("variant_id") or "", r.get("engine") or ""))
        if cid:
            raters[cid].add(r["_player"])
    named = set(priority_cells)
    per_rec = s12["per_recording"]
    out = []
    for item in bundle["items"]:
        for v in item["variants"]:
            key = (item["id"], v["variant_id"], v["engine"])
            n = len(raters.get(v["cid"], set()))
            it = items_by_id.get(item["id"])
            fam = classify(it, v["variant_id"])[2] if it is not None else "unknown"
            cue = fam not in ("delivery:neutral", "sarcasm:sincere")
            rec = per_rec.get(f"{item['id']}/{v['variant_id']}") if v["engine"] == "human" else None
            if rec is not None and not rec["passes_s12"]:
                tier, need, why = (
                    1,
                    max(1, target - rec["confirming"]),
                    "author recording fails §12",
                )
            elif (item["id"], v["variant_id"]) in named and n < target:
                tier, need, why = 2, target - n, "negatively discriminating gold"
            elif key in headline_cells and n < target:
                if cue and v["engine"] == "gemini" and v.get("gate_source") != "human":
                    tier, why = 3, "cue-bearing headline cell, judge-admitted, no human ruling"
                elif cue:
                    tier, why = 4, "cue-bearing headline cell"
                else:
                    tier, why = 5, "neutral headline cell"
                need = target - n
            elif n < target:
                tier, need, why = 6, target - n, "outside the frozen headline"
            else:
                continue
            out.append(
                {
                    "cid": v["cid"],
                    "item_id": item["id"],
                    "variant_id": v["variant_id"],
                    "engine": v["engine"],
                    "raters": n,
                    "need": need,
                    "tier": tier,
                    "reason": why,
                    "family": fam,
                }
            )
    out.sort(key=lambda x: (x["tier"], x["raters"], x["item_id"], x["variant_id"]))
    return {
        "target": target,
        "tiers": dict(sorted(Counter(x["tier"] for x in out).items())),
        "answers_needed": sum(x["need"] for x in out),
        "answers_needed_by_tier": {
            t: sum(x["need"] for x in out if x["tier"] == t)
            for t in sorted({x["tier"] for x in out})
        },
        "cells": out,
    }


# --------------------------------------------------------------------------- driver


def analyze_listeners(
    human_glob: str,
    trials_path: Path,
    bundle_path: Path,
    freeze_path: Path,
    items_root: Path,
    player: str,
    review_path: Path | None,
    record_meta: Path | None,
    priority_cells: Iterable[tuple[str, str]] = (),
    equivalences_path: Path | None = DEFAULT_EQUIVALENCES,
) -> dict[str, Any]:
    from voxparity.harness.crossjudge import load_items

    freeze = json.loads(freeze_path.read_text())
    items_by_id = load_items(freeze, items_root)
    bundle = json.loads(bundle_path.read_text())
    rows = [r for r in load_game_rows(human_glob) if r["item_id"] in items_by_id]
    in_game = {
        (i["id"], v["variant_id"])
        for i in bundle["items"]
        for v in i["variants"]
        if v["engine"] == "human"
    }
    frozen_human = {
        (it["id"], v["variant_id"])
        for it in freeze["items"]
        for v in it["variants"]
        if v.get("status") == "usable" and (v.get("clips") or {}).get("human")
    }
    recordings = in_game | frozen_human
    eq = load_equivalences(equivalences_path)
    s12_strict = recording_validation(rows, recordings, items_by_id)
    s12 = recording_validation(rows, recordings, items_by_id, equivalences=eq)
    s12_wo = recording_validation(
        rows, recordings, items_by_id, exclude_players=[player], equivalences=eq
    )
    headline: set[tuple[str, str, str]] = set()
    for it in freeze["items"]:
        if it.get("design") == "invariant_control":
            continue
        for v in it["variants"]:
            if v.get("status") == "usable":
                for eng in v.get("clips") or {}:
                    headline.add((it["id"], v["variant_id"], eng))
    return {
        "counts": {
            "players": len({r["_player"] for r in rows}),
            "sessions": len({r["_session"] for r in rows}),
            "probe_answers": sum(r["condition"] == "probe" for r in rows),
            "author_recordings": len(recordings),
            "recordings_in_game": len(in_game),
            "frozen_recordings_not_in_game": sorted(f"{i}/{v}" for i, v in frozen_human - in_game),
        },
        "recognition": recognition(rows, recordings, items_by_id, eq),
        "s12": s12,
        "s12_strict_gold_only": {
            k: s12_strict[k]
            for k in (
                "n",
                "passing_s12",
                "passing_s12_and_majority",
                "pairs_with_a_discriminating_listener",
            )
        },
        "s12_without_top_player": s12_wo,
        "top_player": player_profile(
            trials_path, rows, player, review_path, record_meta, recordings
        ),
        "rater_targets": rater_targets(rows, bundle, items_by_id, s12, headline, priority_cells),
    }


def _fmt_ci(e: Any) -> str:
    if not isinstance(e, dict) or e.get("mean") is None:
        return "—"
    if e.get("lo") is None:
        return f"{e['mean']:.2f} (n={e.get('n')})"
    return f"{e['mean']:.2f} [{e['lo']:.2f}, {e['hi']:.2f}] (n={e.get('n')})"


def _recognition_md(rc: dict[str, Any] | None) -> list[str]:
    if not rc:
        return []
    p = rc["pooled"]
    lp = rc["listeners_per_recording"]
    return [
        "## Recognition of the intended cue (main measure)",
        "",
        f"Recordings heard in the game: {rc['recordings_heard']}; listeners per recording "
        f"min {lp['min']}, median {lp['median']}, max {lp['max']}"
        + (
            " (every recording heard by >= 3 independent listeners)."
            if rc["all_with_3plus_listeners"]
            else "."
        ),
        "",
        md_table(
            ["", "estimate"],
            [
                ["recognition, gold only", _fmt_ci(p["strict"])],
                ["recognition, gold or equivalent description", _fmt_ci(p["lenient"])],
                ["chance (1/options, mean)", f"{p['chance']:.2f}"],
                ["Wagner Hu, gold only", _fmt_ci(rc["hu_strict"])],
                ["Wagner Hu, gold or equivalent", _fmt_ci(rc["hu_lenient"])],
                ["Krippendorff alpha on listener labels", str(rc["krippendorff_alpha_labels"])],
                [
                    "recordings recognised above chance (gold only)",
                    f"{rc['recordings_above_chance_strict']}/{rc['recordings_heard']}",
                ],
                [
                    "recordings with a majority confirming (gold or equivalent)",
                    f"{rc['recordings_majority_lenient']}/{rc['recordings_heard']}",
                ],
            ],
        ),
        "",
        "By cue class:",
        "",
        md_table(
            ["cue class", "answers", "gold only", "gold or equivalent", "chance"],
            [
                [
                    f,
                    str(x["answers"]),
                    _fmt_ci(x["strict"]),
                    _fmt_ci(x["lenient"]),
                    f"{x['chance']:.2f}",
                ]
                for f, x in rc["by_family"].items()
            ],
        ),
        "",
        "Mis-specified probes (scored against corrected wording): "
        + (", ".join(rc["misspecified_probes"]) or "none"),
        "",
    ]


def render_markdown(data: dict[str, Any]) -> str:
    s, w = data["s12"], data["s12_without_top_player"]
    c = data["counts"]
    lines = [
        "# Listener audit: spec §12, top player, rater depth",
        "",
        "Generated by `voxparity analyze listeners`. A confirmation = an independent game player "
        "who heard the author recording and chose the item's probe gold or an equivalent "
        "description of the intended cue (docs/internal/listener-equivalences.yaml; mis-specified "
        "probes are scored against the corrected wording). Spec §12 requires >= 3.",
        "",
        f"Players {c['players']}, sessions {c['sessions']}, probe answers {c['probe_answers']}. "
        f"Author recordings {c['author_recordings']} ({c['recordings_in_game']} in the game; "
        "not in "
        f"the game: {', '.join(c['frozen_recordings_not_in_game']) or 'none'}).",
        "",
        *_recognition_md(data.get("recognition")),
        "## Spec §12",
        "",
        md_table(
            ["", "all players", "without the top player"],
            [
                ["recordings", str(s["n"]), str(w["n"])],
                [">= 3 listeners", str(s["with_3plus_listeners"]), str(w["with_3plus_listeners"])],
                [">= 3 confirming (passes §12)", str(s["passing_s12"]), str(w["passing_s12"])],
                [
                    ">= 3 confirming and majority",
                    str(s["passing_s12_and_majority"]),
                    str(w["passing_s12_and_majority"]),
                ],
                [
                    "recorded pairs with a discriminating listener",
                    f"{s['pairs_with_a_discriminating_listener']}/{len(s['pairs'])}",
                    f"{w['pairs_with_a_discriminating_listener']}/{len(w['pairs'])}",
                ],
            ],
        ),
        "",
        "### Recordings failing §12 (all players)",
        "",
        md_table(
            ["recording", "family", "listeners", "confirming"],
            [
                [k, v["family"], str(v["listeners"]), str(v["confirming"])]
                for k, v in sorted(
                    s["below_3_confirming"].items(), key=lambda x: (x[1]["confirming"], x[0])
                )
            ],
        ),
        "",
        "### Per recording",
        "",
        md_table(
            [
                "recording",
                "listeners",
                "confirming",
                "share",
                "chance",
                "confirming w/o top player",
            ],
            [
                [
                    k,
                    str(v["listeners"]),
                    str(v["confirming"]),
                    "—" if v["share"] is None else f"{v['share']:.2f}",
                    "—" if v["chance"] is None else f"{v['chance']:.2f}",
                    str(w["per_recording"][k]["confirming"]),
                ]
                for k, v in sorted(s["per_recording"].items())
            ],
        ),
        "",
        "## Rater-depth targets",
        "",
        f"Target {data['rater_targets']['target']} raters per cell. Cells short by tier: "
        f"{data['rater_targets']['tiers']}; answers needed by tier: "
        f"{data['rater_targets']['answers_needed_by_tier']}. "
        "Full list in the JSON (`rater_targets.cells`).",
        "",
        md_table(
            ["tier", "cid", "cell", "engine", "raters", "need", "reason"],
            [
                [
                    str(x["tier"]),
                    x["cid"],
                    f"{x['item_id']}/{x['variant_id']}",
                    x["engine"],
                    str(x["raters"]),
                    str(x["need"]),
                    x["reason"],
                ]
                for x in data["rater_targets"]["cells"]
                if x["tier"] <= 3
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def render_profile(p: dict[str, Any]) -> str:
    """The top-player profile, kept OUT of the committed tables (it is per-player
    timing data; it belongs under runs/, which is gitignored)."""
    lines = [
        f"## Top player {p['player']} (evidence only)",
        "",
        f"Share of trial answers: {p['share_of_trials']:.0%}. Device hints: {p['device_hints']}.",
        "",
        md_table(
            [
                "session",
                "first trial (UTC)",
                "last",
                "min",
                "trials",
                "prev session (gap s)",
                "next session (gap s)",
            ],
            [
                [
                    x["session"],
                    x["first_trial"][:19],
                    x["last_trial"][11:19],
                    str(x["minutes"]),
                    str(x["trials"]),
                    "—"
                    if not x["previous_session"]
                    else f"{x['previous_session']['player']} ({x['previous_session']['gap_s']})",
                    "—"
                    if not x["next_session"]
                    else f"{x['next_session']['player']} ({x['next_session']['gap_s']})",
                ]
                for x in p["sessions"]
            ],
        ),
        "",
        f"Answer style: {json.dumps(p['answer_style'])}",
        "",
        f"Accuracy: {json.dumps(p['accuracy'])}",
        "",
        f"Author recordings this player heard: {len(p['heard_author_recordings'])}.",
        "",
        f"Agreement with Bhavik's review-console answers: {json.dumps(p.get('vs_bhavik_review'))}",
        "",
        f"Author recording sessions: {json.dumps(p.get('recording_sessions_local_time'))}",
        "",
    ]
    return "\n".join(lines) + "\n"
