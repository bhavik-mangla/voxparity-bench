"""Leak scan for a VoxParity public export. Exits 1 on any unreviewed hit.

Run it on the output of ``export_public.py`` (both trees) before anything is
published:

    uv run python scripts/release/leak_scan.py \\
        --bank <private bank checkout> \\
        --dev <export>/repo/docs/release/dev-split.json \\
        --exceptions scripts/release/leak-exceptions.yaml \\
        <export>/repo <export>/data

The held-out lexicon is built from the PRIVATE bank's item files: every item
that is not in the dev split contributes its id, its tool names, and every long
string it contains (transcripts, scenarios, policies, probe options, rationales).
Text that also occurs in a dev item is public anyway and is not a leak.

Hit kinds (one line per file, line and kind; secret values are never printed):

  heldout-id       a non-dev bank item id (incl. legacy, found, screening)
  heldout-text     a 32-char window of any long string of a non-dev item
  heldout-tool     a tool name defined only by non-dev items (standing actions
                   the harness offers on every item are exempt)
  secret           API-key, token or private-key shapes; a non-empty
                   *_KEY/*_TOKEN/*_SECRET assignment in an env-style file
  email            any e-mail address
  local-path       home directories, temp/scratch paths, private worktree names
  forbidden-file   files that must never ship (.env, pickles, game trials, ...)
  audio-not-dev    a .wav whose SHA-256 is not a dev clip

Binary files: PDFs are scanned through ``pdftotext`` when it is installed;
PNGs with a same-stem PDF next to them are covered by that PDF; any other image
is listed as UNSCANNED (a warning, not a failure) so it can be checked by eye.

Reviewed exceptions (``--exceptions``) are YAML entries

    - path: items/pilot/t4/vxp-frdcb-0002.yaml   # glob, relative to the root
      kind: heldout-id
      token_sha256: <sha256 of the matched token>
      reason: why this is accepted

Tokens are referenced by hash so the exceptions file (which ships) names no
held-out content. ``secret`` and ``audio-not-dev`` hits can never be excepted.
``--print-token-hashes`` prints the hash of each hit's token to help write an
entry; the token itself is printed only for ids and tool names (never text).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Offered on every item by the harness (voxparity.harness.runner.STANDING_TOOLS);
# a held-out item that also defines one does not make the name secret.
STANDING_TOOLS = {"ask_clarifying_question", "escalate_to_human"}

MIN_TEXT = 40  # strings shorter than this are too generic to fingerprint an item
WINDOW = 32
STRIDE = 24

SECRET = re.compile(
    r"(sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|sk-or-v1-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}"
    r"|AIza[0-9A-Za-z_-]{30,}|xai-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}"
    r"|-{5}BEGIN [A-Z ]{0,20}PRIVATE KEY|org-[A-Za-z0-9]{16,}|Bearer [A-Za-z0-9._-]{24,})"
)
ENV_ASSIGN = re.compile(
    r"^[ \t]*(?:export[ \t]+)?[A-Z0-9_]*(?:KEY|KEYS|TOKEN|SECRET)[ \t]*=[ \t]*([^\s#]+)", re.M
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# Built from pieces so this file does not match itself.
LOCAL = re.compile(
    "|".join(
        [
            "/" + "Users/",
            "/" + "home/[a-z]",
            "/private/" + "tmp",
            "/" + "tmp/claude",
            "scratch" + "pad",
            "C:" + r"\\\\Users",
            "voxparity" + "-wt-",
            "vx" + "-latex",
            "vx" + "-paper",
            "vx" + "-ins-",
            "voxparity" + "-audio-scout",
            "bhavik" + "mangla",
        ]
    )
)
FORBIDDEN = [
    ".env",
    "*.pkl",
    "*.pickle",
    "web-trials*.jsonl",
    "rows.pkl",
    "stimuli/manifest.yaml",  # private bank manifest (data/audio/ holds the public one)
    "freeze/*",
    "CLAUDE.md",
    "*.local.yaml",
    "ia.html",
]
NEVER_EXCEPTED = {"secret", "audio-not-dev"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svgz"}
SKIP_DIRS = {".git", "__pycache__", ".venv", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


_CANON = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00ad": "",
        "\ufb01": "fi",
        "\ufb02": "fl",
    }
)


def norm(s: object) -> str:
    """Whitespace-collapsed text with typographic quotes, dashes and ligatures
    folded, so a PDF's rendering of a sentence matches its source."""
    return " ".join(str(s or "").translate(_CANON).split())


def strings(o: Any) -> Iterator[str]:
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


@dataclass
class Lexicon:
    held_ids: set[str]
    held_tools: set[str]
    frags: set[str]
    dev_shas: set[str]


def published_text(paths: list[Path]) -> str:
    out = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            t = pdf_text(p)
            if t is None:
                sys.exit(f"--published {p}: pdftotext is required")
            out.append(t)
        else:
            out.append(p.read_text())
    return norm("\n".join(out)).lower()


def build_lexicon(bank: Path, dev_split: Path, published: str = "") -> Lexicon:
    dev = json.loads(dev_split.read_text())
    dev_ids = {it["id"] for it in dev["items"]}
    dev_shas = {v["clip_sha256"] for it in dev["items"] for v in it["variants"]}
    held_ids: set[str] = set()
    held_tools: set[str] = set()
    dev_tools: set[str] = set()
    held_texts: list[str] = []
    dev_texts: list[str] = []
    n_items = 0
    dev_seen: set[str] = set()
    for f in sorted((bank / "items").rglob("*.yaml")):
        try:
            d = yaml.safe_load(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "id" not in d:
            continue
        n_items += 1
        is_dev = d["id"] in dev_ids
        if is_dev:
            dev_seen.add(str(d["id"]))
        if not is_dev:
            held_ids.add(str(d["id"]))
        for tl in d.get("tools") or []:
            if isinstance(tl, dict) and tl.get("name"):
                (dev_tools if is_dev else held_tools).add(str(tl["name"]))
        texts = [norm(s) for s in strings(d) if len(norm(s)) >= MIN_TEXT]
        (dev_texts if is_dev else held_texts).extend(texts)
    if n_items < 100 or not held_ids:
        sys.exit(f"bank at {bank} has {n_items} items: not the full private bank")
    if dev_ids - dev_seen:
        sys.exit(f"{len(dev_ids - dev_seen)} dev-split ids are not in the bank")
    # text already public (dev items, the published paper) is not a leak
    dev_corpus = "\n".join(dev_texts).lower() + "\n" + published
    frags: set[str] = set()
    for t in held_texts:
        t = t.lower()
        for i in range(0, max(1, len(t) - WINDOW + 1), STRIDE):
            w = t[i : i + WINDOW]
            if len(w) == WINDOW and w not in dev_corpus:
                frags.add(w)
    tools = held_tools - dev_tools - STANDING_TOOLS
    if published:
        # hyphenation/line breaks in a PDF can split an id; match on the flat text
        tools = {t for t in tools if t.lower() not in published}
    return Lexicon(held_ids, tools, frags, dev_shas)


@dataclass
class Hit:
    root: str
    rel: str
    kind: str
    line: int
    token: str  # never printed for secret/heldout-text

    @property
    def token_sha256(self) -> str:
        return hashlib.sha256(self.token.encode()).hexdigest()

    def show(self, with_hash: bool) -> str:
        tok = self.token if self.kind in {"heldout-id", "heldout-tool", "local-path"} else ""
        s = f"{self.root}/{self.rel}:{self.line}: {self.kind}"
        if tok:
            s += f" [{tok}]"
        if with_hash and self.kind not in NEVER_EXCEPTED:
            s += f" sha256={self.token_sha256}"
        return s


def line_of(txt: str, pos: int) -> int:
    return txt.count("\n", 0, pos) + 1


def pdf_text(p: Path) -> str | None:
    exe = shutil.which("pdftotext")
    if not exe:
        return None
    r = subprocess.run([exe, "-q", str(p), "-"], capture_output=True, check=False)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def fragment_source(rel: str, txt: str) -> str:
    """JSON escapes (\\u2014, \\") would hide text from a raw search: search the
    decoded strings of JSON/JSONL files instead."""
    if rel.endswith(".json"):
        try:
            return "\n".join(strings(json.loads(txt)))
        except ValueError:
            return txt
    if rel.endswith(".jsonl"):
        out: list[str] = []
        for ln in txt.splitlines():
            try:
                out.extend(strings(json.loads(ln)))
            except ValueError:
                out.append(ln)
        return "\n".join(out)
    return txt


def scan_text(
    root: str,
    rel: str,
    txt: str,
    lex: Lexicon,
    id_re: re.Pattern[str] | None,
    tool_re: re.Pattern[str] | None,
    envlike: bool,
) -> list[Hit]:
    hits: list[Hit] = []
    if id_re:
        for m in id_re.finditer(txt):
            hits.append(Hit(root, rel, "heldout-id", line_of(txt, m.start()), m.group(0)))
    if tool_re:
        for m in tool_re.finditer(txt):
            hits.append(Hit(root, rel, "heldout-tool", line_of(txt, m.start()), m.group(0)))
    for m in SECRET.finditer(txt):
        hits.append(Hit(root, rel, "secret", line_of(txt, m.start()), m.group(0)))
    if envlike:
        for m in ENV_ASSIGN.finditer(txt):
            hits.append(Hit(root, rel, "secret", line_of(txt, m.start()), m.group(1)))
    for m in EMAIL.finditer(txt):
        hits.append(Hit(root, rel, "email", line_of(txt, m.start()), m.group(0)))
    for m in LOCAL.finditer(txt):
        hits.append(Hit(root, rel, "local-path", line_of(txt, m.start()), m.group(0)))
    if lex.frags:
        flat = norm(fragment_source(rel, txt)).lower()
        found: set[str] = set()
        for i in range(0, max(0, len(flat) - WINDOW + 1)):
            w = flat[i : i + WINDOW]
            if w in lex.frags and w not in found:
                found.add(w)
                hits.append(Hit(root, rel, "heldout-text", 0, w))
    return hits


def load_exceptions(path: Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    ex = yaml.safe_load(path.read_text()) or []
    for e in ex:
        if e.get("kind") in NEVER_EXCEPTED:
            sys.exit(f"exceptions file may not except kind {e['kind']!r}")
        for k in ("path", "kind", "token_sha256", "reason"):
            if not e.get(k):
                sys.exit(f"exception entry missing {k!r}: {e}")
    return ex


def excepted(h: Hit, ex: list[dict[str, Any]], used: set[int]) -> bool:
    for i, e in enumerate(ex):
        if (
            e["kind"] == h.kind
            and fnmatch.fnmatch(h.rel, e["path"])
            and e["token_sha256"] == h.token_sha256
        ):
            used.add(i)
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bank", type=Path, required=True, help="private bank checkout (items/)")
    ap.add_argument("--dev", type=Path, required=True, help="dev-split.json")
    ap.add_argument("--exceptions", type=Path, default=None)
    ap.add_argument(
        "--published",
        type=Path,
        action="append",
        default=[],
        help="an already-public document (the arXiv PDF): its text and tool names are not "
        "leaks; held-out ids stay flagged regardless",
    )
    ap.add_argument("--print-token-hashes", action="store_true")
    ap.add_argument("--json", type=Path, default=None, help="also write hits as JSON here")
    ap.add_argument("roots", type=Path, nargs="+")
    a = ap.parse_args()

    lex = build_lexicon(a.bank, a.dev, published_text(a.published))
    ex = load_exceptions(a.exceptions)
    id_re = (
        re.compile(
            r"\b(" + "|".join(sorted(map(re.escape, lex.held_ids), key=len, reverse=True)) + r")\b"
        )
        if lex.held_ids
        else None
    )
    tool_re = (
        re.compile(
            r"\b("
            + "|".join(sorted(map(re.escape, lex.held_tools), key=len, reverse=True))
            + r")\b"
        )
        if lex.held_tools
        else None
    )

    hits: list[Hit] = []
    unscanned: list[str] = []
    n_files = 0
    for root in a.roots:
        files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file() and not SKIP_DIRS & set(p.relative_to(root).parts)
        ]
        for p in files:
            n_files += 1
            rel = str(p.relative_to(root))
            rname = root.name
            if any(fnmatch.fnmatch(p.name, pat) or fnmatch.fnmatch(rel, pat) for pat in FORBIDDEN):
                hits.append(Hit(rname, rel, "forbidden-file", 0, p.name))
            suf = p.suffix.lower()
            if suf == ".wav":
                if hashlib.sha256(p.read_bytes()).hexdigest() not in lex.dev_shas:
                    hits.append(Hit(rname, rel, "audio-not-dev", 0, p.name))
                continue
            if suf == ".pdf":
                txt = pdf_text(p)
                if txt is None:
                    unscanned.append(f"{rname}/{rel} (no pdftotext)")
                    continue
                hits += scan_text(rname, rel, txt, lex, id_re, tool_re, False)
                continue
            if suf in IMAGE_EXT:
                if not p.with_suffix(".pdf").exists():
                    unscanned.append(f"{rname}/{rel}")
                continue
            try:
                txt = p.read_text()
            except (UnicodeDecodeError, OSError):
                unscanned.append(f"{rname}/{rel} (binary)")
                continue
            envlike = (
                p.name.endswith(".env") or p.name.startswith(".env") or p.name == "example.env"
            )
            hits += scan_text(rname, rel, txt, lex, id_re, tool_re, envlike)

    used: set[int] = set()
    live = [h for h in hits if not excepted(h, ex, used)]
    n_exc = len(hits) - len(live)
    by_file: dict[str, list[Hit]] = {}
    for h in live:
        by_file.setdefault(f"{h.root}/{h.rel}", []).append(h)
    for f, hs in sorted(by_file.items()):
        kinds: dict[str, int] = {}
        for h in hs:
            kinds[h.kind] = kinds.get(h.kind, 0) + 1
        print(f"{f}: " + ", ".join(f"{k} x{n}" for k, n in sorted(kinds.items())))
        for h in hs[:12]:
            print("    " + h.show(a.print_token_hashes))
        if len(hs) > 12:
            print(f"    ... {len(hs) - 12} more")
    stale = [ex[i] for i in range(len(ex)) if i not in used]
    for e in stale:
        print(f"STALE EXCEPTION (matched nothing): {e['path']} {e['kind']}")
    if unscanned:
        print(f"\nUNSCANNED (check by eye): {len(unscanned)}")
        for u in unscanned:
            print("    " + u)
    print(
        f"\n{len(by_file)} file(s) flagged, {len(live)} hit(s); "
        f"{n_exc} reviewed exception(s) applied; "
        f"{n_files} files scanned. Lexicon: held-out ids={len(lex.held_ids)} "
        f"text windows={len(lex.frags)} held-only tools={len(lex.held_tools)}"
    )
    if a.json:
        a.json.write_text(
            json.dumps(
                [
                    {
                        "file": f"{h.root}/{h.rel}",
                        "line": h.line,
                        "kind": h.kind,
                        "token_sha256": h.token_sha256 if h.kind not in NEVER_EXCEPTED else None,
                    }
                    for h in live
                ],
                indent=1,
            )
        )
    sys.exit(1 if live else 0)


if __name__ == "__main__":
    main()
