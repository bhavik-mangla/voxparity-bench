"""`real:` asset packs: recipe resolution, loud cache failures, fetch, provenance.

The fixture recipe points at a tiny WAV generated under the test tmp dir; the
fetch tests use an httpx MockTransport — nothing here touches the network, and
the convert step uses /bin/cp so no ffmpeg is required either (the shell=False
argv path is what is under test, not ffmpeg itself).
"""

from __future__ import annotations

import audioop
import hashlib
import json
import math
import struct
import wave
from io import BytesIO
from pathlib import Path

import pytest

from voxparity.schemas.item import SceneKind, SceneSpec
from voxparity.stimuli.packs import (
    PackError,
    fetch_pack,
    load_recipe,
    parse_real_asset_id,
    resolve_real_asset,
)
from voxparity.stimuli.scenes import apply_scene

RATE = 24000


def _tone_wav(dur_s: float = 2.0, freq: float = 220.0, rate: int = RATE) -> bytes:
    n = int(rate * dur_s)
    samples = [int(12000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]
    buf = BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


def _rms(wav: bytes) -> int:
    with wave.open(BytesIO(wav), "rb") as f:
        return audioop.rms(f.readframes(f.getnframes()), 2)


def _write_fixture(
    tmp_path: Path, cls: str = "traffic", **overrides: object
) -> tuple[Path, Path, dict]:
    """A realbeds fixture recipe + cached tiny WAV under tmp. Returns
    (packs_dir, cache_path, recipe)."""
    packs_dir = tmp_path / "packs"
    (packs_dir / "realbeds").mkdir(parents=True)
    cache = tmp_path / "assets-cache" / "realbeds" / f"{cls}.wav"
    cache.parent.mkdir(parents=True)
    wav = _tone_wav(1.5, freq=90.0)
    cache.write_bytes(wav)
    recipe = {
        "id": f"real:realbeds/{cls}",
        "class": cls,
        "source": "Fixture Field Recordings",
        "source_url": "https://example.org/source",
        "download_url": "https://example.org/beds/original.wav",
        "original_sha256": hashlib.sha256(b"original").hexdigest(),
        "license": "CC0-1.0",
        "license_proof_url": "https://example.org/beds/license",
        "attribution": "Fixture bed by Example Recordist (CC0)",
        "convert": "/bin/cp {in} {out}",
        "excerpt": {"offset_s": 0.0, "duration_s": 1.5},
        "converted_sha256": hashlib.sha256(wav).hexdigest(),
        # absolute so the default repo_root=Path(".") resolves it under tmp
        "cache_path": str(cache),
    }
    recipe.update(overrides)
    (packs_dir / "realbeds" / f"{cls}.json").write_text(json.dumps(recipe))
    return packs_dir, cache, recipe


class TestParse:
    def test_parses_pack_and_class(self):
        assert parse_real_asset_id("real:realbeds/traffic") == ("realbeds", "traffic")

    def test_rejects_other_prefixes_and_malformed_ids(self):
        with pytest.raises(ValueError, match="not a real:"):
            parse_real_asset_id("synth:traffic")
        for bad in ("real:realbeds", "real:/traffic", "real:realbeds/"):
            with pytest.raises(ValueError, match="malformed"):
                parse_real_asset_id(bad)


class TestResolve:
    def test_resolves_verified_cache(self, tmp_path):
        packs_dir, cache, _recipe = _write_fixture(tmp_path)
        assert resolve_real_asset("real:realbeds/traffic", packs_dir=packs_dir) == cache

    def test_missing_recipe_is_loud(self, tmp_path):
        packs_dir, _cache, _recipe = _write_fixture(tmp_path)
        with pytest.raises(PackError, match="no recipe for 'real:realbeds/birds'"):
            resolve_real_asset("real:realbeds/birds", packs_dir=packs_dir)

    def test_missing_cache_names_download_url_and_fetch_command(self, tmp_path):
        packs_dir, cache, _recipe = _write_fixture(tmp_path)
        cache.unlink()
        with pytest.raises(PackError) as exc:
            resolve_real_asset("real:realbeds/traffic", packs_dir=packs_dir)
        msg = str(exc.value)
        assert "https://example.org/beds/original.wav" in msg
        assert "voxparity packs fetch --pack realbeds" in msg
        assert "Refusing to fall back to synthesis" in msg

    def test_sha_mismatch_is_corrupted_cache(self, tmp_path):
        packs_dir, cache, _recipe = _write_fixture(tmp_path)
        cache.write_bytes(cache.read_bytes() + b"\x00\x00")
        with pytest.raises(PackError, match="corrupted cache"):
            resolve_real_asset("real:realbeds/traffic", packs_dir=packs_dir)

    def test_recipe_id_must_match_asset_id(self, tmp_path):
        packs_dir, _cache, _recipe = _write_fixture(tmp_path, id="real:realbeds/other")
        with pytest.raises(PackError, match="declares id"):
            resolve_real_asset("real:realbeds/traffic", packs_dir=packs_dir)


class TestApplySceneRealBed:
    def test_background_real_bed_mixes_and_carries_provenance(self, tmp_path):
        packs_dir, _cache, recipe = _write_fixture(tmp_path)
        scene = SceneSpec(
            kind=SceneKind.BACKGROUND, asset="real:realbeds/traffic", snr_db=6.0, seed=5
        )
        primary = _tone_wav(2.0, freq=300.0)
        out, out_recipe = apply_scene(primary, scene, packs_dir=packs_dir)
        assert out_recipe["op"] == "mix_background"
        assert out_recipe["asset"] == "real:realbeds/traffic"
        # the mixed background really is the cached recording, by hash
        assert out_recipe["background_sha256"] == recipe["converted_sha256"]
        assert _rms(out) > _rms(primary)
        # published-dataset attribution requirement: provenance on the manifest
        assert out_recipe["asset_license"] == "CC0-1.0"
        assert out_recipe["asset_attribution"] == "Fixture bed by Example Recordist (CC0)"

    def test_missing_cache_fails_the_scene_loudly(self, tmp_path):
        packs_dir, cache, _recipe = _write_fixture(tmp_path)
        cache.unlink()
        scene = SceneSpec(kind=SceneKind.BACKGROUND, asset="real:realbeds/traffic", seed=1)
        with pytest.raises(PackError, match="voxparity packs fetch --pack realbeds"):
            apply_scene(_tone_wav(), scene, packs_dir=packs_dir)


class TestFetchPack:
    def _transport(self, payload: bytes):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "https://example.org/beds/original.wav"
            return httpx.Response(200, content=payload)

        return httpx.MockTransport(handler)

    def _excerpt(self, wav: bytes, offset_s: float, duration_s: float) -> bytes:
        with wave.open(BytesIO(wav), "rb") as f:
            rate = f.getframerate()
            f.setpos(int(offset_s * rate))
            frames = f.readframes(int(duration_s * rate))
        buf = BytesIO()
        with wave.open(buf, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(frames)
        return buf.getvalue()

    def test_fetch_downloads_converts_excerpts_and_verifies(self, tmp_path):
        original = _tone_wav(3.0, freq=90.0)
        excerpt = self._excerpt(original, 0.5, 1.0)
        packs_dir, cache, _recipe = _write_fixture(
            tmp_path,
            original_sha256=hashlib.sha256(original).hexdigest(),
            excerpt={"offset_s": 0.5, "duration_s": 1.0},
            converted_sha256=hashlib.sha256(excerpt).hexdigest(),
            convert=["/bin/cp", "{in}", "{out}"],  # list form, shell=False argv
        )
        cache.unlink()
        results = fetch_pack("realbeds", packs_dir=packs_dir, transport=self._transport(original))
        assert [r["status"] for r in results] == ["fetched"]
        assert cache.read_bytes() == excerpt
        # end of the loop: the resolved asset is now usable
        assert resolve_real_asset("real:realbeds/traffic", packs_dir=packs_dir) == cache

    def test_fetch_skips_verified_cache_without_network(self, tmp_path):
        packs_dir, cache, _recipe = _write_fixture(tmp_path)
        # no transport: any network attempt would raise inside httpx
        results = fetch_pack("realbeds", packs_dir=packs_dir, transport=None)
        assert [r["status"] for r in results] == ["ok"]
        assert Path(results[0]["path"]) == cache

    def test_fetch_rejects_original_sha_mismatch(self, tmp_path):
        original = _tone_wav(3.0, freq=90.0)
        packs_dir, cache, _recipe = _write_fixture(
            tmp_path, original_sha256=hashlib.sha256(b"not this").hexdigest()
        )
        cache.unlink()
        results = fetch_pack("realbeds", packs_dir=packs_dir, transport=self._transport(original))
        assert results[0]["status"] == "error"
        assert "original_sha256" in results[0]["error"]
        assert not cache.exists()

    def test_fetch_unknown_pack_is_loud(self, tmp_path):
        with pytest.raises(PackError, match="no recipes found"):
            fetch_pack("nosuchpack", packs_dir=tmp_path)


class TestLoadRecipe:
    def test_string_and_list_convert_forms_both_load(self, tmp_path):
        packs_dir, _cache, recipe = _write_fixture(tmp_path)
        loaded = load_recipe("real:realbeds/traffic", packs_dir=packs_dir)
        assert loaded["convert"] == recipe["convert"]
        assert loaded["excerpt"] == {"offset_s": 0.0, "duration_s": 1.5}
        assert loaded["license_proof_url"] == "https://example.org/beds/license"
