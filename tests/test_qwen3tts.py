"""Qwen3TTSBackend contract, with a fake model (no mlx, no weights)."""

from __future__ import annotations

import wave
from io import BytesIO

import pytest

from _held import needs_items
from voxparity.schemas.item import DeliveryVariant, Emotion, GoldAction, SpeakerProfile
from voxparity.stimuli.style import delivery_clause
from voxparity.stimuli.tts import (
    QWEN3TTS_CUSTOMVOICE_MODEL,
    QWEN3TTS_VOICEDESIGN_MODEL,
    Qwen3TTSBackend,
    get_backend,
)


class _Result:
    def __init__(self) -> None:
        self.sample_rate = 24_000
        self.audio = [0.0, 0.5, -0.5, 0.0]


class _FakeModel:
    def __init__(self, rate: int = 24_000) -> None:
        self.calls: list[dict] = []
        self.rate = rate

    def generate(self, **kw):
        self.calls.append(kw)
        r = _Result()
        r.sample_rate = self.rate
        return [r, r]


def _backend(mode: str = "customvoice", rate: int = 24_000) -> Qwen3TTSBackend:
    b = Qwen3TTSBackend(mode=mode, load=False)
    b._model = _FakeModel(rate)
    return b


def _variant(vid: str, emotion: str = "neutral", intensity: float = 0.3, speaker=None):
    return DeliveryVariant(
        variant_id=vid,
        emotion=Emotion(emotion),
        intensity=intensity,
        speaker=SpeakerProfile(speaker) if speaker else None,
        gold=GoldAction(tool="t", args={}, rationale="r"),
    )


def test_one_engine_name_per_mode():
    """Both modes under one name would overwrite each other in the store."""
    from voxparity.stimuli import tts

    cv, vd = tts._BACKENDS["qwen3tts-cv"], tts._BACKENDS["qwen3tts-vd"]
    assert issubclass(cv, Qwen3TTSBackend) and issubclass(vd, Qwen3TTSBackend)
    assert cv(load=False).mode == "customvoice" and cv(load=False).name == "qwen3tts-cv"
    assert vd(load=False).mode == "voicedesign" and vd(load=False).name == "qwen3tts-vd"
    assert Qwen3TTSBackend(mode="voicedesign", load=False).name == "qwen3tts-vd"
    for gone in ("qwen3tts", "qwen3"):
        with pytest.raises(ValueError, match="unknown engine"):
            get_backend(gone)


def test_model_ids_are_pinned_per_mode():
    assert Qwen3TTSBackend(mode="customvoice", load=False).model == QWEN3TTS_CUSTOMVOICE_MODEL
    assert Qwen3TTSBackend(mode="voicedesign", load=False).model == QWEN3TTS_VOICEDESIGN_MODEL
    with pytest.raises(ValueError, match="mode"):
        Qwen3TTSBackend(mode="clone", load=False)


def test_mode_is_not_read_from_env(monkeypatch):
    """The engine name picks the mode; an env switch could label a VD render cv."""
    monkeypatch.setenv("VOXPARITY_QWEN3TTS_MODE", "voicedesign")
    assert Qwen3TTSBackend(load=False).mode == "customvoice"


def test_long_form_wording_is_opt_in_and_recorded():
    from voxparity.stimuli.style import long_form_clause

    v = _variant("b_sad", "sad", 0.7)
    short, long = _backend(), _backend()
    long.long_form = True
    for b in (short, long):
        b.bind_item("vxp-test-0001")
        b.synthesize("I lost my job.", v)
    assert short._model.calls[-1]["instruct"] == f"Say the line {delivery_clause(v)}."
    assert long._model.calls[-1]["instruct"] == f"Say the line {long_form_clause(v)}."
    _, _, prov = long.synthesize("I lost my job.", v)
    assert "wording=long" in prov and long.name == short.name == "qwen3tts-cv"
    # same speaker regardless of wording, so short and long renders stay comparable
    assert (
        short.voice_and_instruct("vxp-test-0001", v)[0]
        == long.voice_and_instruct("vxp-test-0001", v)[0]
    )


def test_instruct_is_the_canonical_delivery_clause():
    b = _backend()
    b.bind_item("vxp-test-0002")
    v = _variant("sarcastic", "sarcastic", 0.7)
    _, _, prompt = b.synthesize("Oh yeah, sign me up.", v)
    sent = b._model.calls[-1]
    assert sent["instruct"] == f"Say the line {delivery_clause(v)}."
    assert "dry, mocking sarcasm" in sent["instruct"]
    # the transcript travels as text, never inside the instruction
    assert sent["text"] == "Oh yeah, sign me up." and "sign me up" not in sent["instruct"]
    assert prompt.startswith(sent["instruct"]) and "mode=customvoice" in prompt


@pytest.mark.parametrize("mode", ["customvoice", "voicedesign"])
def test_speaker_fixed_across_an_items_variants(mode):
    b = _backend(mode)
    b.bind_item("vxp-test-0003")
    _, v1, p1 = b.synthesize("Everything's okay.", _variant("at_ease"))
    _, v2, p2 = b.synthesize("Everything's okay.", _variant("covert", "whispered", 0.85))
    assert v1 == v2
    if mode == "voicedesign":
        persona = v1.removeprefix("persona:")
        assert b._model.calls[0]["instruct"].startswith(persona)
        assert b._model.calls[1]["instruct"].startswith(persona)
        assert "voice" not in b._model.calls[0]
    else:
        assert b._model.calls[0]["voice"] == b._model.calls[1]["voice"] == v1
    assert p1 != p2  # only the delivery differs


def test_speaker_choice_is_deterministic_and_seeded_by_item():
    a, b = _backend(), _backend()
    picks = set()
    for i in range(40):
        item = f"vxp-test-{i:04d}"
        va, _ = a.voice_and_instruct(item, _variant("x"))
        vb, _ = b.voice_and_instruct(item, _variant("x"))
        assert va == vb
        picks.add(va)
    assert picks == set(Qwen3TTSBackend.CUSTOMVOICE_SPEAKERS)
    assert a.seed_for("i", "a") == b.seed_for("i", "a") != a.seed_for("i", "b")


def test_customvoice_pool_is_gender_balanced():
    pool = Qwen3TTSBackend.CUSTOMVOICE_SPEAKERS
    assert pool[0::2] == ("Ryan", "Aiden") and pool[1::2] == ("Vivian", "Sohee")
    b = _backend()
    picks = [b.voice_and_instruct(f"vxp-bal-{i:04d}", _variant("x"))[0] for i in range(400)]
    female = sum(p in pool[1::2] for p in picks) / len(picks)
    assert 0.4 < female < 0.6


def test_unbound_backend_refuses():
    with pytest.raises(ValueError, match="bind_item"):
        _backend().synthesize("Hello.", _variant("v"))


def test_speaker_profiles_refused_in_customvoice_mapped_in_voicedesign():
    cv = _backend("customvoice")
    cv.bind_item("vxp-x-0001")
    for profile in ("child", "elderly"):
        with pytest.raises(ValueError, match="speaker profile"):
            cv.synthesize("Hello.", _variant("v", speaker=profile))
    vd = _backend("voicedesign")
    vd.bind_item("vxp-x-0001")
    _, voice, prompt = vd.synthesize("Hello.", _variant("v", speaker="elderly"))
    assert "eighties" in voice and "frail elderly person" in prompt  # persona + canonical frame
    _, adult_voice, _ = vd.synthesize("Hello.", _variant("w", speaker="adult"))
    assert adult_voice.removeprefix("persona:") in Qwen3TTSBackend.PERSONAS


def test_wav_is_24k_mono_pcm16_and_concatenates_segments():
    b = _backend()
    b.bind_item("vxp-x-0001")
    wav, _, _ = b.synthesize("Hello.", _variant("v"))
    with wave.open(BytesIO(wav), "rb") as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (24000, 1, 2)
        assert f.getnframes() == 8


def test_unexpected_sample_rate_fails_loudly():
    b = _backend(rate=22_050)
    b.bind_item("vxp-x-0001")
    with pytest.raises(RuntimeError, match="sample rate"):
        b.synthesize("Hello.", _variant("v"))


@needs_items("vxp-bnkgr-0001")
def test_cli_generate_binds_item_and_writes_rows_to_the_given_store(monkeypatch, tmp_path):
    """`stimuli generate --engine qwen3tts-cv --store-dir X` must bind each item
    before synthesis and write only into X (adoption renders never touch the bank)."""
    from pathlib import Path

    from typer.testing import CliRunner

    from voxparity.cli import app
    from voxparity.stimuli import tts
    from voxparity.stimuli.store import StimulusStore

    fake = _backend()
    monkeypatch.setattr(tts, "get_backend", lambda name: fake)
    items = Path(__file__).parent.parent / "items" / "pilot" / "t4" / "vxp-bnkgr-0001.yaml"
    store_dir = tmp_path / "qwen-store"
    r = CliRunner().invoke(
        app,
        [
            "stimuli",
            "generate",
            str(items),
            "--engine",
            "qwen3tts-cv",
            "--store-dir",
            str(store_dir),
        ],
    )
    assert r.exit_code == 0, r.output + str(r.exception)
    recs = StimulusStore(store_dir).records()
    assert {x.variant_id for x in recs} == {"grateful", "sarcastic"}
    assert {x.engine for x in recs} == {"qwen3tts-cv"}
    assert len({x.voice for x in recs}) == 1  # one speaker for both deliveries
    assert all(x.model == QWEN3TTS_CUSTOMVOICE_MODEL for x in recs)
