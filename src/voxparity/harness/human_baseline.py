"""The human action baseline from the web game (D064), against the frozen matrix.

Game sessions are imported by ``voxparity human`` into ordinary run records
scored by the SAME scorer as the models (D036/D069). This module reads those
records and the ``<date>-final-*`` model runs and writes ``human.{json,md}``
beside the other final tables, with the conventions of ``final_analysis``:
item-clustered percentile bootstrap, 4000 resamples, seed 20260915.

Basis. The game's default SIMPLE mode collects a tool choice and no typed
arguments, so a human row is scored on tool SELECTION (``_selection_only``).
Every human-vs-model contrast therefore uses the models' ``selection_credit``
on the same cells; the models' full (argument-checked) credit is carried
alongside for reference only, never as the comparison.

Cells. A cell here is (item, variant, stimulus engine): a player heard one
specific clip, and each model arm was run on one engine. Contrasts are on
IDENTICAL cells first (the D105 standing rule), clustered by item.

Identity. The importer names each session ``<player8>-<session8>``. A player
(the browser's persistent id) is the unit for "independent listeners"; a
player who played several sessions is still one listener. Sessions from the
``author`` batch never reach the records (spec §12), and the raw trials file,
when given, is scanned so the report states how many such sessions existed.
"""

from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from voxparity.harness.final_analysis import (
    CONF,
    N_BOOT,
    NEUTRAL_FAMILIES,
    SEED,
    Arm,
    Key,
    classify,
    cluster_bootstrap,
    fmt,
    hu_bootstrap,
    md_table,
    paired_bootstrap,
)
from voxparity.harness.report import control_ids, is_control, load_records

NO_CALL = "__none__"
MIN_LISTENERS = 3  # spec §12: a recording needs >= 3 independent listeners
REFERENCE_ARM = "gemini37or"  # D107's best audio-native arm, the D061 Y slot


def selection_credit(scores: dict[str, Any]) -> float:
    """A model row's selection-only credit (the basis a simple-mode human shares).

    Rows recorded before ``selection_credit`` existed reconstruct it exactly as
    ``export_model_attempts`` does: 1.0 on a selection hit, else the acceptable
    alternative's credit.
    """
    if scores.get("selection_credit") is not None:
        return float(scores["selection_credit"])
    return max(float(scores.get("credit") or 0.0), 1.0 if scores.get("selection") else 0.0)


def rater_of(driver: str) -> tuple[str, str]:
    """``human:<player8>-<session8>`` -> (player, session)."""
    name = driver.split(":", 1)[1] if ":" in driver else driver
    player, _, session = name.rpartition("-")
    return (player or name), session


def _cell(item_id: str, variant_id: str, engine: str) -> Key:
    # Key[0] stays the item id so every bootstrap clusters by item.
    return (item_id, f"{variant_id}@{engine}")


def _variant(key: Key) -> str:
    return key[1].rsplit("@", 1)[0]


def _is_cue(key: Key, items_by_id: dict[str, Any]) -> bool:
    item = items_by_id.get(key[0])
    return item is not None and classify(item, _variant(key))[2] not in NEUTRAL_FAMILIES


def _ci(cells: dict[Key, float]) -> dict[str, Any] | None:
    keys = sorted(cells)
    return cluster_bootstrap([cells[k] for k in keys], [k[0] for k in keys])


def load_human_rows(human_glob: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(glob.glob(human_glob)):
        d = Path(p)
        if (d / "records.jsonl").exists():
            rows.extend(load_records(d))
    return rows


def trial_counts(trials_path: Path) -> dict[str, Any]:
    """Raw counts from the game's trials JSONL (what was collected, before import)."""
    players: set[str] = set()
    sessions: set[str] = set()
    completed: set[str] = set()
    author: set[str] = set()
    trial_keys: set[tuple[str, str]] = set()
    session_only = legacy = 0
    batch_sessions: dict[str, set[str]] = defaultdict(set)
    completions: list[dict[str, Any]] = []
    for line in trials_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r.get("session")
        if not sid:
            continue
        if r.get("batch") == "author":
            author.add(sid)
            continue
        if r.get("kind") not in ("trial", "session"):
            legacy += 1  # pre-v3 rows without a clip id; the importer cannot map them
            continue
        sessions.add(sid)
        batch_sessions[str(r.get("batch"))].add(sid)
        if r.get("player"):
            players.add(str(r["player"]))
        if r["kind"] == "trial" and r.get("action") is not None:
            trial_keys.add((sid, str(r.get("cid"))))
        elif r["kind"] == "session":
            completed.add(sid)
            completions.append(r)
    for r in completions:
        for k in r.get("answers") or {}:
            block, _, cid = k.partition(":")
            if block == "act" and (r["session"], cid) not in trial_keys:
                session_only += 1
    return {
        "players": len(players),
        "sessions": len(sessions),
        "completed_sessions": len(completed),
        "action_answers": len(trial_keys) + session_only,
        "trial_rows": len(trial_keys),
        "answers_only_in_completion_row": session_only,
        "author_sessions_excluded": len(author),
        "legacy_rows_unmappable": legacy,
        "sessions_by_batch": {b: len(s) for b, s in sorted(batch_sessions.items())},
    }


def majority_vote(
    answers: dict[Key, list[tuple[str, float]]],
) -> tuple[dict[Key, float], int]:
    """Plurality answer per cell. A tie is scored as the mean credit of the tied
    answers (the expectation of breaking it at random) and counted, never broken
    silently (the D045 consensus rule)."""
    out: dict[Key, float] = {}
    ties = 0
    for k, votes in answers.items():
        counts = Counter(a for a, _ in votes)
        top = max(counts.values())
        modal = sorted(a for a, c in counts.items() if c == top)
        ties += len(modal) > 1
        credit = {a: c for a, c in votes}
        out[k] = sum(credit[a] for a in modal) / len(modal)
    return out, ties


def _model_cells(arms: list[Arm]) -> dict[str, dict[str, Any]]:
    """Per model label, pooled across its stimulus engines."""
    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        m = out.setdefault(
            arm.label, {"cascade": arm.is_cascade, "sel": {}, "full": {}, "engines": []}
        )
        m["engines"].append(arm.engine)
        for (item_id, vid), row in arm.audio_rows.items():
            k = _cell(item_id, vid, arm.engine)
            m["sel"][k] = selection_credit(row.get("scores") or {})
            m["full"][k] = arm.audio[(item_id, vid)]
    return out


def human_baseline(
    human_rows: list[dict[str, Any]],
    arms: list[Arm],
    items_by_id: dict[str, Any],
    recordings: set[tuple[str, str]] | None = None,
    trials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controls = control_ids(items_by_id)
    answers: dict[Key, list[tuple[str, float]]] = defaultdict(list)
    individual: list[tuple[Key, float, str]] = []
    probe_pairs: list[tuple[str, str, str]] = []
    probe_vals: list[tuple[str, float]] = []
    listeners: dict[tuple[str, str], set[str]] = defaultdict(set)
    per_player: Counter[str] = Counter()
    sessions: set[str] = set()
    dropped: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    from voxparity.scoring.stats import cue_class_map

    for r in human_rows:
        if r.get("error"):
            dropped["error"] += 1
            continue
        item = items_by_id.get(r["item_id"])
        if item is None:
            dropped["item_not_in_freeze"] += 1
            continue
        if is_control(r, controls):
            dropped["invariant_control"] += 1
            continue
        player, session = rater_of(str(r.get("driver", "")))
        engine = str(r.get("engine") or "")
        vid = r.get("variant_id") or ""
        scores = r.get("scores") or {}
        if engine == "human" and r.get("condition") in ("audio", "probe"):
            listeners[(r["item_id"], vid)].add(player)
        if r.get("condition") == "audio":
            k = _cell(r["item_id"], vid, engine)
            credit = float(scores.get("credit") or 0.0)
            calls = r.get("tool_calls") or []
            answers[k].append((calls[0]["tool"] if calls else NO_CALL, credit))
            individual.append((k, credit, player))
            per_player[player] += 1
            sessions.add(session)
            modes[str(scores.get("scored_on") or (r.get("metrics") or {}).get("mode"))] += 1
        elif r.get("condition") == "probe" and "answer" in scores:
            ok = bool(scores.get("passed"))
            probe_vals.append((r["item_id"], float(ok)))
            classes = cue_class_map(item)
            probe_pairs.append(
                (
                    r["item_id"],
                    classes.get(str(scores.get("gold")), "other"),
                    classes.get(str(scores.get("answer")), "other"),
                )
            )

    ind_vals = [c for _, c, _ in individual]
    ind_clusters = [k[0] for k, _, _ in individual]
    cue_ind = [(k, c) for k, c, _ in individual if _is_cue(k, items_by_id)]
    neu_ind = [(k, c) for k, c, _ in individual if not _is_cue(k, items_by_id)]
    mean_cell = {k: sum(c for _, c in v) / len(v) for k, v in answers.items()}
    majority, ties = majority_vote(answers)
    rated3 = {k: v for k, v in majority.items() if len(answers[k]) >= MIN_LISTENERS}
    top_player, top_n = per_player.most_common(1)[0] if per_player else ("", 0)
    without_top = [(k, c) for k, c, p in individual if p != top_player]
    ntop = [len(v) for v in answers.values()]

    out: dict[str, Any] = {
        "recruitment": (
            "self-selected volunteers via shared game links; uncompensated; "
            "adults by click-consent; not a paid panel (D064)"
        ),
        "basis": "tool selection (game simple mode); models compared on selection_credit",
        "counts": {
            **(trials or {}),
            "players_imported": len({p for _, _, p in individual}),
            "sessions_imported": len(sessions),
            "audio_answers_scored": len(individual),
            "probe_answers_scored": len(probe_vals),
            "cells": len(answers),
            "items": len({k[0] for k in answers}),
            "cells_by_engine": dict(
                sorted(Counter(k[1].rsplit("@", 1)[1] for k in answers).items())
            ),
            "raters_per_cell": dict(sorted(Counter(ntop).items())),
            "cells_with_3plus": sum(n >= MIN_LISTENERS for n in ntop),
            "dropped_rows": dict(dropped),
            "scored_on": dict(modes),
        },
        "individual": cluster_bootstrap(ind_vals, ind_clusters),
        "individual_cue_bearing": cluster_bootstrap(
            [c for _, c in cue_ind], [k[0] for k, _ in cue_ind]
        ),
        "individual_neutral": cluster_bootstrap(
            [c for _, c in neu_ind], [k[0] for k, _ in neu_ind]
        ),
        "cell_mean": _ci(mean_cell),
        "majority_vote": _ci(majority),
        "majority_vote_ties": ties,
        "majority_vote_3plus": _ci(rated3),
        "top_player_share": round(top_n / len(individual), 4) if individual else None,
        "individual_without_top_player": cluster_bootstrap(
            [c for _, c in without_top], [k[0] for k, _ in without_top]
        ),
        "probe_accuracy": cluster_bootstrap([v for _, v in probe_vals], [i for i, _ in probe_vals]),
        "probe_hu": hu_bootstrap(probe_pairs),
    }

    # ---- same cells: humans vs every model arm (pooled over its engines)
    models = _model_cells(arms)
    rows = []
    for label, m in sorted(models.items()):
        shared = sorted(set(m["sel"]) & set(mean_cell))
        if not shared:
            continue
        h = {k: mean_cell[k] for k in shared}
        s = {k: m["sel"][k] for k in shared}
        cue = [k for k in shared if _is_cue(k, items_by_id)]
        neu = [k for k in shared if k not in set(cue)]
        rows.append(
            {
                "label": label,
                "cascade": m["cascade"],
                "engines": sorted({k[1].rsplit("@", 1)[1] for k in shared}),
                "cells": len(shared),
                "items": len({k[0] for k in shared}),
                "human": _ci(h),
                "model_selection": _ci(s),
                "model_full_credit": _ci({k: m["full"][k] for k in shared}),
                "model_minus_human": paired_bootstrap(s, h),
                "model_minus_human_majority": paired_bootstrap(s, {k: majority[k] for k in shared}),
                "human_cue_bearing": _ci({k: h[k] for k in cue}),
                "model_cue_bearing": _ci({k: s[k] for k in cue}),
                "model_minus_human_cue_bearing": paired_bootstrap(
                    {k: s[k] for k in cue}, {k: h[k] for k in cue}
                ),
                "human_neutral": _ci({k: h[k] for k in neu}),
                "model_neutral": _ci({k: s[k] for k in neu}),
                "model_minus_human_neutral": paired_bootstrap(
                    {k: s[k] for k in neu}, {k: h[k] for k in neu}
                ),
            }
        )
    out["same_cell"] = rows

    # ---- the D061 sentence's human slot, with Y/Z on the SAME cue-bearing cells
    ref = next((r for r in rows if r["label"] == REFERENCE_ARM), None)
    casc = next((r for r in rows if r["cascade"] and r["label"] == "cascadeopen"), None)
    out["d061"] = {
        "human_cue_bearing": out["individual_cue_bearing"],
        "reference_arm": REFERENCE_ARM,
        "same_cells": {
            "human": ref["human_cue_bearing"] if ref else None,
            "reference_arm": ref["model_cue_bearing"] if ref else None,
            "cascade": casc["model_cue_bearing"] if casc else None,
            "reference_minus_human": ref["model_minus_human_cue_bearing"] if ref else None,
            "cascade_minus_human": casc["model_minus_human_cue_bearing"] if casc else None,
        },
    }

    # ---- listener coverage of the author's recordings (spec §12)
    if recordings is not None:
        per_rec = {f"{i}/{v}": len(listeners.get((i, v), set())) for i, v in sorted(recordings)}
        out["recordings"] = {
            "n": len(per_rec),
            "with_3plus_listeners": sum(n >= MIN_LISTENERS for n in per_rec.values()),
            "min_listeners": min(per_rec.values()) if per_rec else None,
            "median_listeners": sorted(per_rec.values())[len(per_rec) // 2] if per_rec else None,
            "below_3": {k: n for k, n in per_rec.items() if n < MIN_LISTENERS},
            "per_recording": per_rec,
        }
    return out


def render_markdown(data: dict[str, Any]) -> str:
    c = data["counts"]
    lines = [
        f"Recruitment: {data['recruitment']}. Basis: {data['basis']}.",
        "",
        f"Players {c.get('players', c['players_imported'])} "
        f"(imported {c['players_imported']}), sessions {c.get('sessions', c['sessions_imported'])} "
        f"({c.get('completed_sessions', '?')} completed), action answers "
        f"{c.get('action_answers', c['audio_answers_scored'])} "
        f"(scored {c['audio_answers_scored']}; rows dropped {c['dropped_rows'] or 'none'}), "
        f"probe answers {c['probe_answers_scored']}, "
        f"cells {c['cells']} on {c['items']} items, cells with >=3 raters "
        f"{c['cells_with_3plus']}. Author-batch sessions excluded: "
        f"{c.get('author_sessions_excluded', 'n/a')}.",
        "",
        md_table(
            ["measure", "estimate"],
            [
                ["individual credit (all cells)", fmt(data["individual"])],
                ["individual, cue-bearing cells", fmt(data["individual_cue_bearing"])],
                ["individual, neutral cells", fmt(data["individual_neutral"])],
                ["per-cell mean", fmt(data["cell_mean"])],
                [
                    f"majority vote ({data['majority_vote_ties']} tied cells at mean credit)",
                    fmt(data["majority_vote"]),
                ],
                ["majority vote, cells with >=3 raters", fmt(data["majority_vote_3plus"])],
                [
                    f"individual without the top player ({_share(data['top_player_share'])} "
                    "of answers)",
                    fmt(data["individual_without_top_player"]),
                ],
                ["perception probe accuracy", fmt(data["probe_accuracy"])],
                ["perception probe Hu (Wagner)", fmt(data["probe_hu"])],
            ],
        ),
        "",
        "Same cells (D105): human per-cell mean vs each model's selection credit on the "
        "identical (item, variant, engine) cells the humans answered.",
        "",
        md_table(
            [
                "arm",
                "engines",
                "cells (items)",
                "human",
                "model (selection)",
                "model - human",
                "model - human, cue-bearing",
                "model - human, neutral",
                "model full credit",
            ],
            [
                [
                    r["label"] + (" (cascade)" if r["cascade"] else ""),
                    "+".join(r["engines"]),
                    f"{r['cells']} ({r['items']})",
                    fmt(r["human"]),
                    fmt(r["model_selection"]),
                    fmt(r["model_minus_human"], signed=True),
                    fmt(r["model_minus_human_cue_bearing"], signed=True),
                    fmt(r["model_minus_human_neutral"], signed=True),
                    fmt(r["model_full_credit"]),
                ]
                for r in data["same_cell"]
            ],
        ),
    ]
    rec = data.get("recordings")
    if rec:
        lines += [
            "",
            f"Author recordings in the game: {rec['n']}; with >= {MIN_LISTENERS} independent "
            f"listeners (distinct players, action or probe): {rec['with_3plus_listeners']}; "
            f"min {rec['min_listeners']}, median {rec['median_listeners']}.",
        ]
        if rec["below_3"]:
            lines.append(
                "Below the quorum: "
                + ", ".join(f"{k} ({n})" for k, n in sorted(rec["below_3"].items()))
                + "."
            )
    return "\n".join(lines) + "\n"


def _share(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def analyze_human(
    runs_glob: str,
    human_glob: str,
    freeze_path: Path,
    out_dir: Path,
    items_root: Path = Path("."),
    trials_path: Path | None = None,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    from voxparity.cli import _iter_item_files, load_item
    from voxparity.harness.final_analysis import load_arms

    freeze = json.loads(freeze_path.read_text())
    items_by_id: dict[str, Any] = {}
    for d in freeze.get("item_dirs", []):
        for f in _iter_item_files(items_root / d):
            it = load_item(f)
            items_by_id[it.id] = it
    arms = load_arms(runs_glob, items_by_id)
    recordings: set[tuple[str, str]] | None = None
    if bundle_path is not None and bundle_path.exists():
        bundle = json.loads(bundle_path.read_text())
        recordings = {
            (i["id"], v["variant_id"])
            for i in bundle["items"]
            for v in i["variants"]
            if v.get("engine") == "human"
        }
    data = human_baseline(
        load_human_rows(human_glob),
        arms,
        items_by_id,
        recordings=recordings,
        trials=trial_counts(trials_path) if trials_path else None,
    )
    data["method"] = {
        "ci": "item-clustered percentile bootstrap",
        "resamples": N_BOOT,
        "seed": SEED,
        "confidence": CONF,
        "freeze": freeze.get("freeze_id"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "human.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    (out_dir / "human.md").write_text(
        "# human baseline (bank-freeze-2026-09-15)\n\n" + render_markdown(data)
    )
    return data
