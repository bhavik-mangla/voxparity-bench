"""Recorded environmental beds: `real:` assets resolved through committed recipes.

Bhavik's directive (Sep 12): real traffic/birds/TV/etc. beds mixed under our
speech instead of synthesized ones. The audio itself may be license-barred from
the repo (docs/CONDITIONS.md pack contract), so what IS committed is a recipe —
`stimuli/packs/<pack>/<class>.json` — recording the source, its license proof,
the download URL, both content hashes, the exact ffmpeg conversion, and the
excerpt window. The audio lives in a gitignored cache
(`stimuli/assets-cache/<pack>/<class>.wav`) rebuilt bit-exactly by
`voxparity packs fetch`.

Recipe shape (produced by the curation lane; code to this contract):

    {
      "id": "real:realbeds/traffic",
      "class": "traffic",
      "source": "...", "source_url": "...", "download_url": "...",
      "original_sha256": "...",
      "license": "CC0-1.0", "license_proof_url": "...", "attribution": "...",
      "convert": "ffmpeg -y -i {in} -ac 1 -ar 24000 -sample_fmt s16 {out}",
      "excerpt": {"offset_s": 12.0, "duration_s": 30.0},
      "converted_sha256": "...",
      "cache_path": "stimuli/assets-cache/realbeds/traffic.wav"
    }

`convert` may be a string (shlex-split; never run through a shell) or an argv
list; either way the tokens `{in}` and `{out}` mark the downloaded original and
the conversion output. The excerpt is cut AFTER conversion with stdlib `wave`
frame arithmetic, so the cache bytes are a pure function of the recipe.

A recipe whose cache is missing FAILS LOUDLY, naming the download_url and the
fetch command. Never a silent fallback to synthesis: a synthesized stand-in for
a recorded bed is a wrong stimulus wearing the right asset id — the D046
phantom class.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

_PREFIX = "real:"


class PackError(RuntimeError):
    """A `real:` asset could not be resolved, verified, or fetched."""


def _fetch_command(pack: str) -> str:
    return f"voxparity packs fetch --pack {pack}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_real_asset_id(asset_id: str) -> tuple[str, str]:
    """Split ``real:<pack>/<class>`` into (pack, class); loud on any other shape."""
    if not asset_id.startswith(_PREFIX):
        raise ValueError(f"not a real: asset id: {asset_id!r}")
    pack, sep, cls = asset_id.removeprefix(_PREFIX).partition("/")
    if not sep or not pack or not cls:
        raise ValueError(f"malformed real: asset id {asset_id!r} (want 'real:<pack>/<class>')")
    return pack, cls


def load_recipe(asset_id: str, packs_dir: Path = Path("stimuli/packs")) -> dict[str, Any]:
    """Load and sanity-check the committed recipe for a ``real:`` asset id."""
    pack, cls = parse_real_asset_id(asset_id)
    path = packs_dir / pack / f"{cls}.json"
    if not path.is_file():
        raise PackError(f"no recipe for {asset_id!r}: expected {path}")
    recipe: dict[str, Any] = json.loads(path.read_text())
    rid = recipe.get("id")
    if rid is not None and rid != asset_id:
        raise PackError(f"recipe {path} declares id {rid!r}, expected {asset_id!r}")
    for key in ("converted_sha256", "cache_path"):
        if not recipe.get(key):
            raise PackError(f"recipe {path} is missing required key {key!r}")
    return recipe


def resolve_real_asset(
    asset_id: str,
    packs_dir: Path = Path("stimuli/packs"),
    repo_root: Path = Path("."),
) -> Path:
    """Return the verified cached WAV for ``real:<pack>/<class>``.

    The cache must exist AND hash to the recipe's ``converted_sha256``; anything
    else raises PackError naming the fix. Never synthesizes a substitute.
    """
    pack, _cls = parse_real_asset_id(asset_id)
    recipe = load_recipe(asset_id, packs_dir=packs_dir)
    cache = repo_root / str(recipe["cache_path"])
    if not cache.is_file():
        raise PackError(
            f"{asset_id!r}: cached audio missing at {cache}.\n"
            f"The recording is not committed (license: {recipe.get('license', 'unknown')}); "
            f"fetch it from {recipe.get('download_url', 'the recipe download_url')} with:\n"
            f"    {_fetch_command(pack)}\n"
            f"Refusing to fall back to synthesis: a substituted bed is a wrong stimulus."
        )
    got = _sha256_file(cache)
    want = recipe["converted_sha256"]
    if got != want:
        raise PackError(
            f"{asset_id!r}: corrupted cache at {cache}: sha256 {got} != "
            f"recipe converted_sha256 {want}. Delete the file and re-run: {_fetch_command(pack)}"
        )
    return cache


def _convert_argv(convert: str | list[str], original: Path, out: Path) -> list[str]:
    """Recipe convert command -> argv with {in}/{out} substituted. shell=False always."""
    tokens = shlex.split(convert) if isinstance(convert, str) else [str(t) for t in convert]
    if not tokens:
        raise PackError("recipe convert command is empty")
    if not any("{in}" in t for t in tokens) or not any("{out}" in t for t in tokens):
        raise PackError(
            f"recipe convert command must reference both {{in}} and {{out}}: {convert!r}"
        )
    return [t.replace("{in}", str(original)).replace("{out}", str(out)) for t in tokens]


def _excerpt_wav(converted: Path, excerpt: dict[str, Any] | None) -> bytes:
    """Cut the recipe's excerpt window out of the converted WAV (stdlib, exact)."""
    if not excerpt:
        return converted.read_bytes()
    offset_s = float(excerpt.get("offset_s", 0.0))
    duration_s = excerpt.get("duration_s")
    with wave.open(str(converted), "rb") as f:
        rate, width, channels = f.getframerate(), f.getsampwidth(), f.getnchannels()
        total = f.getnframes()
        start = min(int(offset_s * rate), total)
        n = (
            total - start
            if duration_s is None
            else min(int(float(duration_s) * rate), total - start)
        )
        f.setpos(start)
        frames = f.readframes(n)
    if n <= 0:
        raise PackError(f"excerpt {excerpt!r} selects no audio (source has {total} frames)")
    buf = BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(frames)
    return buf.getvalue()


def _fetch_one(recipe: dict[str, Any], cache: Path, transport: Any = None) -> None:
    """Download -> verify original -> convert (no shell) -> excerpt -> verify -> cache."""
    import httpx

    url = recipe.get("download_url")
    if not url:
        raise PackError("recipe has no download_url")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        original = tmp / "original.bin"
        with httpx.Client(transport=transport, follow_redirects=True, timeout=300.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            original.write_bytes(resp.content)
        got = _sha256_file(original)
        want = recipe.get("original_sha256")
        if want and got != want:
            raise PackError(
                f"download from {url}: sha256 {got} != recipe original_sha256 {want} "
                f"(source changed upstream or download corrupted; nothing cached)"
            )
        converted = tmp / "converted.wav"
        argv = _convert_argv(recipe["convert"], original, converted)
        proc = subprocess.run(argv, capture_output=True, text=True)  # shell=False, argv only
        if proc.returncode != 0:
            raise PackError(
                f"convert command failed ({proc.returncode}): {argv}\n{proc.stderr[-2000:]}"
            )
        if not converted.is_file():
            raise PackError(f"convert command produced no output file: {argv}")
        data = _excerpt_wav(converted, recipe.get("excerpt"))
        got_c = hashlib.sha256(data).hexdigest()
        want_c = recipe["converted_sha256"]
        if got_c != want_c:
            raise PackError(
                f"converted excerpt sha256 {got_c} != recipe converted_sha256 {want_c} "
                f"(recipe and pipeline disagree; nothing cached)"
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)


def fetch_pack(
    pack: str,
    packs_dir: Path = Path("stimuli/packs"),
    repo_root: Path = Path("."),
    *,
    force: bool = False,
    transport: Any = None,
) -> list[dict[str, Any]]:
    """Materialize every recipe in ``packs_dir/<pack>/`` into the local cache.

    Returns one verdict dict per recipe: {"id", "status": ok|fetched|error,
    "path" | "error"}. Errors are collected, not raised, so one dead URL does
    not block the rest of the pack; the CLI turns any error into exit 1.
    ``transport`` is an httpx transport override for tests (never hit the
    network in tests).
    """
    pack_dir = packs_dir / pack
    recipes = sorted(pack_dir.glob("*.json")) if pack_dir.is_dir() else []
    if not recipes:
        raise PackError(f"no recipes found under {pack_dir}")
    results: list[dict[str, Any]] = []
    for path in recipes:
        recipe: dict[str, Any] = json.loads(path.read_text())
        asset_id = recipe.get("id", f"{_PREFIX}{pack}/{path.stem}")
        try:
            cache = repo_root / str(recipe["cache_path"])
            if cache.is_file() and not force and _sha256_file(cache) == recipe["converted_sha256"]:
                results.append({"id": asset_id, "status": "ok", "path": str(cache)})
                continue
            _fetch_one(recipe, cache, transport=transport)
            results.append({"id": asset_id, "status": "fetched", "path": str(cache)})
        except Exception as exc:  # collected per-file; CLI decides exit code
            results.append({"id": asset_id, "status": "error", "error": str(exc)})
    return results


def wav_duration_s(path: Path) -> float:
    """Duration of a WAV file in seconds (for fetch verdict printing)."""
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


__all__ = [
    "PackError",
    "fetch_pack",
    "load_recipe",
    "parse_real_asset_id",
    "resolve_real_asset",
    "wav_duration_s",
]
