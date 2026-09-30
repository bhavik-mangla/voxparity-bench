# Release process

How the public VoxParity release is built by the organisers, from the full
repository and the private bank (neither ships). The
public code repository and the Zenodo data record are both generated; nothing
is copied by hand, and nothing ships unless an allowlist names it.

| File | Role |
|---|---|
| `export_public.py` | builds `repo/` (public code tree) and `data/` (Zenodo bundle) from a git ref |
| `export.yaml` | the allowlist: what enters the repo, what is excluded, which insight files and templates ship |
| `leak_scan.py` | fails on held-out ids, held-out text, held-out-only tool names, secrets, e-mails, local paths, forbidden files and non-dev audio |
| `leak-exceptions.yaml` | reviewed exceptions, one token each, referenced by hash |
| `../../release/` | templates: public README, LICENSE (Apache-2.0), DATA_LICENSE (CC BY 4.0), NOTICE, CITATION.cff, `.zenodo.json` (code DOI), `zenodo-dataset.json` (data record), dataset card, CANARY.md, SUBMITTING.md |

## Inputs

- this repository at a **tag** (the export reads `git archive <tag>`, so no
  untracked, ignored or uncommitted file can leak);
- the private bank checkout (`--bank`): item files are read with `git show` at
  the bank commit named in `docs/release/dev-split.json`, the stimulus manifest
  likewise (its hash must equal `manifest_sha256` in the held-out register),
  audio from the content-addressed store (every clip hash-verified), run records
  from `runs/20260915-final-*-gemini/`, and the local Whisper re-gate.

## What the exporter enforces

- Dev items only: 40 item files, byte-identical, hash-verified against
  `dev-split.json`; 81 clips, hash-verified.
- The public `dev-split.json` drops the `rejected` map (it names held-out ids).
- Manifest rows go through `voxparity.release.release_manifest` (engine
  allowlist, licence basis per row) with `allowed_items` = dev ids, then
  `public_manifest_row`: every ASR transcript and `slot_aligned_as` string is
  removed (Deepgram, PX-010), the local Whisper re-gate verdict is attached, and
  human raters are pseudonymised.
- Run records go through `release_records(..., allowed_items=dev_ids)`: text-twin
  rows ship only for dev items (they carry the item's golds in `scores`), no
  Deepgram-backed driver ships, and account identifiers are redacted. Rows from
  providers whose terms allow derived fields only (PX-031 Grok, PX-037 nemotron
  `:free`, PX-041/042 StreamLake upstreams) lose response text, tool arguments
  and transcripts (`restrict_record`).
- `example.env` ships with every value blank.
- `data/audio/manifest.yaml` is a harness-ready store manifest, so
  `voxparity run --store-dir data/audio` works on the bundle.
- `SHA256SUMS` and `PROVENANCE.json` (source commit, bank commit, counts) are
  written; the dataset card's counts are filled from the export itself.

## Steps

```bash
# 0. Land every fix through a PR, then tag the export commit (private repo only).
git tag -a v1.0-public-src -m "source of the public export"

# 1. Rehearse on the working tree (output is marked DRY RUN; never publish it).
uv run python scripts/release/export_public.py --src . --worktree-dry-run \
    --bank <bank checkout> --out <scratch>/release-dryrun --force \
    --published paper/arxiv/main.pdf

# 2. Export from the tag.
uv run python scripts/release/export_public.py --src . --ref v1.0-public-src \
    --bank <bank checkout> --out <release-out> --published paper/arxiv/main.pdf
#    exit status 1 = the leak scan found something: read <release-out>/leak-scan.txt

# 3. Verify, all of which must pass.
cd <release-out>/repo && uv sync --extra dev && uv run ruff check . && uv run pytest
uv run voxparity items validate items/                  # 40/40
cd ../data && shasum -a 256 -c SHA256SUMS
gitleaks detect --no-git --source <release-out>         # second opinion on secrets
```

`--published <pdf>` tells the scanner that text and tool names already in the
published paper are not leaks (Figure 1 quotes one held-out item). Held-out ids
are flagged regardless. Use the exact PDF that is on arXiv.

Options: `--with-paper` ships the PDF as `paper/voxparity.pdf`; `--with-web`
ships the game code (`web/data/` never ships); `--config` overrides the
allowlist at the ref (for rehearsals only).

## When the scan flags something

1. Fix it at the source (scrub the file, move per-item tables to the private
   bank) or drop the path from `export.yaml`.
2. Only when the hit is reviewed and harmless, add an entry to
   `leak-exceptions.yaml`. Get the hash with `leak_scan.py --print-token-hashes`.
   Say why in `reason`. Secrets and non-dev audio cannot be excepted.

The scanner reads PDFs with `pdftotext` (poppler). PNGs are covered by a
same-stem PDF; any other image is listed as UNSCANNED and must be checked by
eye.

## Publishing (by the author, after the checks)

1. Copy `data/` into `repo/data/` (the public repository carries the data;
   the quick start expects it in the repository root), re-run the leak scan
   on the combined tree, `git init`, push.
2. Zenodo is linked to the GitHub repository, so publishing the `v1.0.0`
   GitHub release archives code and data together and mints one DOI from
   `.zenodo.json`. `zenodo-dataset.json` is kept only for a separate data
   record, which this release does not use.

## Results table and leaderboard page

`leaderboard.py` generates the README's results section (between the
`leaderboard:start`/`leaderboard:end` markers in `release/README.md`) and the
static leaderboard page `docs/index.html` + `docs/leaderboard.json` +
`docs/.nojekyll`, all from the aggregate result files that ship
(`docs/results/final/paper_leaderboard.json`, `docs/insights/notefull.json`,
`docs/insights/final-analyses.json`). No number is typed by hand.

```bash
uv run python scripts/release/leaderboard.py           # regenerate
uv run python scripts/release/leaderboard.py --check   # stale output or != Table A2 fails
```

`--check` and `tests/test_release_leaderboard.py` also compare every cell with
the paper's Table A2 when `paper/arxiv/` is present. Run it before an export
whenever the result files, the README template or the page template
(`leaderboard_page.html`) change. The page makes no external request (no fonts,
scripts, trackers or cookies). It is served by GitHub Pages from the public
repository: Settings → Pages → Deploy from a branch → `main`, folder `/docs`.
