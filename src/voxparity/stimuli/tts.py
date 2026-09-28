"""TTS backend registry: one interface, three engines, three control families.

Control families represented: instruction (gemini style prompt, hume
``description``, camb ``user_instructions``), closed enum (cartesia, retired),
and inline markup tags (elevenlabs, policy-blocked). Multi-engine splits
are methodology commandment 8 — every item class should exist in >=2 engines so
models can't learn one engine's "angry preset".
"""

from __future__ import annotations

from typing import ClassVar, Protocol

from voxparity.schemas.item import DeliveryVariant


class TTSBackend(Protocol):
    name: str
    model: str

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        """Return (wav_bytes, voice_used, prompt_or_markup_used)."""
        ...


class GeminiBackend:
    name = "gemini"

    def __init__(self, voice: str = "Kore") -> None:
        from voxparity.providers.gemini import TTS_MODEL, GeminiClient

        self.client = GeminiClient()
        self.model = TTS_MODEL
        self.voice = voice

    # D084: the speaker axis picks its voice by profile — Google's own catalog
    # descriptors ("Leda: Youthful", "Gacrux: Mature") are the closest documented
    # fit; the style prompt carries the rest of the framing. The default voice
    # is untouched for adult/unspecified variants.
    _PROFILE_VOICE: ClassVar[dict[str, str]] = {"child": "Leda", "elderly": "Gacrux"}

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        from voxparity.stimuli.style import tts_prompt

        prompt = tts_prompt(transcript, variant)
        voice = self.voice
        if variant.speaker is not None:
            voice = self._PROFILE_VOICE.get(variant.speaker.value, self.voice)
        return self.client.synthesize(prompt, voice=voice), voice, prompt


class CartesiaBackend:
    name = "cartesia"

    def __init__(self) -> None:
        from voxparity.providers.cartesia import MODEL, CartesiaClient

        self.client = CartesiaClient()
        self.model = MODEL
        self.voice_id = self.client.default_voice_id()

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        wav = self.client.synthesize(transcript, variant, self.voice_id)
        return wav, self.voice_id, self.client.control_label(variant)


class ElevenLabsBackend:
    name = "elevenlabs"

    def __init__(self) -> None:
        from voxparity.providers.elevenlabs import MODEL, ElevenLabsClient

        self.client = ElevenLabsClient()
        self.model = MODEL
        self.voice_id = self.client.default_voice_id()

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        from voxparity.providers.elevenlabs import _TAG_MAP

        wav = self.client.synthesize(transcript, variant, self.voice_id)
        return wav, self.voice_id, _TAG_MAP[variant.emotion]


class HumeBackend:
    name = "hume"

    def __init__(self) -> None:
        from voxparity.providers.hume import DEFAULT_VOICE, VERSION, HumeClient

        self.client = HumeClient()
        self.model = f"octave-{VERSION}"
        self.voice = DEFAULT_VOICE

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        wav = self.client.synthesize(transcript, variant, self.voice)
        return wav, self.voice, self.client.control_label(variant)


class CambBackend:
    name = "camb"

    def __init__(self) -> None:
        from voxparity.providers.camb import DEFAULT_VOICE_ID, MODEL, CambClient

        self.client = CambClient()
        self.model = MODEL
        self.voice_id = DEFAULT_VOICE_ID

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        wav = self.client.synthesize(transcript, variant, self.voice_id)
        return wav, str(self.voice_id), self.client.control_label(variant)


class FishAudioBackend:
    name = "fishaudio"

    def __init__(self) -> None:
        from voxparity.providers.fishaudio import DEFAULT_VOICE, MODEL, FishAudioClient

        self.client = FishAudioClient()
        self.model = MODEL
        self.voice_id = DEFAULT_VOICE

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        wav = self.client.synthesize(transcript, variant, self.voice_id)
        return wav, self.voice_id, self.client.control_label(variant)


class InworldBackend:
    name = "inworld"

    def __init__(self) -> None:
        from voxparity.providers.inworld import DEFAULT_VOICE, MODEL, InworldClient

        self.client = InworldClient()
        self.model = MODEL
        self.voice = DEFAULT_VOICE

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        wav = self.client.synthesize(transcript, variant, self.voice)
        return wav, self.voice, self.client.control_label(variant)


class AuraBackend:
    """Deepgram Aura-2: the fourth control family — NONE (D087).

    Style-less by design, so it renders ONLY plain-adult neutral variants:
    accepting a styled variant would silently produce neutral audio (the D046
    phantom class), and Aura normalizes written fillers out of the speech
    (probe-verified: it dropped "Um..." entirely), so disfluent transcripts
    are refused too. Clips from this engine must be ASR-gated by Gemini, not
    Nova-3 (same vendor — cross-grading, wired in the CLI).
    """

    name = "aura"

    def __init__(self, voice: str = "thalia") -> None:
        from voxparity.providers.deepgram import DeepgramClient

        self.client = DeepgramClient()
        self.voice = voice
        self.model = f"aura-2-{voice}-en"

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        from voxparity.schemas.item import Emotion
        from voxparity.stimuli.validate import has_disfluencies

        if variant.emotion is not Emotion.NEUTRAL:
            raise ValueError(f"aura is style-less; refuses styled variant {variant.variant_id}")
        if variant.speaker is not None:
            raise ValueError(f"aura has no speaker profiles; refuses {variant.variant_id}")
        if has_disfluencies(transcript):
            raise ValueError("aura normalizes written fillers away; refuses disfluent transcript")
        return self.client.speak(transcript, voice=self.voice), self.voice, "neutral"


class KokoroBackend:
    """Kokoro-82M via mlx-audio: the LOCAL verbatim-disfluency engine (D089).

    Style-less like Aura (voice + speed only), so it refuses styled variants
    and speaker profiles for the same D046-phantom reason. The deliberate,
    probe-verified difference: Kokoro's G2P front end SPEAKS written
    disfluencies — "Um... yes. I— I think so." renders with the "um" voiced at
    0.32-0.72s (Deepgram filler_words=true hears it; default ASR deletes it,
    which is the D083 finding restated) — so disfluent transcripts are
    ACCEPTED here. That is this backend's reason to exist: it gives the
    disfluency axis a second engine where Aura silently drops the fillers.

    Apache-2.0 weights, offline, ~2s/clip warm and ~1.4 GB RSS on Apple
    Silicon. Native 24 kHz matches the project stimulus rate exactly.
    Gate note: Nova-3 shares nothing with Kokoro so it stays the independent
    ASR gate, and disfluent transcripts route through disfluent_wer
    automatically (validate.has_disfluencies), which is what makes a
    filler-deleting ASR a non-issue for these clips.
    """

    name = "kokoro"

    def __init__(self, voice: str = "af_heart") -> None:
        # Lazy import: mlx-audio is a `local` extra (macOS-arm64 wheels only).
        from mlx_audio.tts.utils import load_model

        self.voice = voice
        self.model = "prince-canuma/Kokoro-82M"
        self._model = load_model(self.model)

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        import io
        import wave

        from voxparity.schemas.item import Emotion

        if variant.emotion is not Emotion.NEUTRAL:
            raise ValueError(f"kokoro is style-less; refuses styled variant {variant.variant_id}")
        if variant.speaker is not None:
            raise ValueError(f"kokoro has no speaker profiles; refuses {variant.variant_id}")

        import numpy as np

        segments = self._model.generate(text=transcript, voice=self.voice, speed=1.0, lang_code="a")
        audio = np.concatenate([np.asarray(seg.audio) for seg in segments])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24_000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue(), self.voice, "neutral"


QWEN3TTS_CUSTOMVOICE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
QWEN3TTS_VOICEDESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"


class Qwen3TTSBackend:
    """Qwen3-TTS-12Hz-1.7B via mlx-audio: a LOCAL instruction-driven engine.

    Two modes, because the paired design needs the SAME speaker on both
    deliveries of an item (D081's caveat: a twin that varies speaker AND
    delivery confounds the counterfactual):

    - ``customvoice`` (default): a preset speaker plus ``instruct`` for style.
      The speaker is a fixed voice, so twins cannot drift apart in identity.
    - ``voicedesign``: VoiceDesign invents the voice from the whole description,
      so each item gets ONE fixed persona sentence (seeded by item id) and only
      the delivery clause differs between variants. Drift is still possible
      and is measured, not assumed away.

    The delivery clause is the project's canonical wording (style.delivery_clause),
    never an ad-hoc string, so this engine renders the same authored intent as
    Gemini. Speaker/persona choice is seeded by ITEM id, so it is identical
    across an item's variants and its followups; the sampling seed is per
    (item, variant) so a re-render is reproducible. Call ``bind_item`` before
    ``synthesize`` (the CLI does); an unbound backend refuses rather than
    guessing a speaker.

    Speaker profiles (D084): customvoice presets are fixed adult voices, so
    child/elderly variants are refused; voicedesign maps them to an age-matched
    persona of the item's seeded gender. Neither mapping has passed a human
    2AFC probe on this engine yet.

    Engine names: each mode writes its own engine name (``qwen3tts-cv`` for
    CustomVoice, ``qwen3tts-vd`` for VoiceDesign). The store keys rows by
    (item, variant, engine), so under one shared name the second mode rendered
    would silently replace the first. Both names are human-gated
    (runner.HUMAN_GATED_ENGINES): no clip qualifies for a scored run without a
    passing human_check.

    ``long_form=True`` sends style.long_form_clause (vocal-behaviour prose)
    instead of the short canonical clause. Same engine name: the provenance line
    records ``wording=long``, and a promotion picks one wording per variant.
    """

    MODES = ("customvoice", "voicedesign")
    ENGINE_NAMES: ClassVar[dict[str, str]] = {
        "customvoice": "qwen3tts-cv",
        "voicedesign": "qwen3tts-vd",
    }
    name: str
    SAMPLE_RATE = 24_000
    TEMPERATURE = 0.9
    # Gender-balanced preset pool, interleaved so even index = male, odd = female
    # (the PERSONAS convention). Ryan and Aiden are the card's English-native
    # presets. Vivian and Sohee are not English-native, but passed the round-2
    # probe: Deepgram WER 0.0 on 5/5 plain lines plus airbrv x2, detected language
    # "en" at confidence >= 0.98 on all 6, mean word confidence in the male range.
    # Serena (one line at language confidence 0.78) and Ono_Anna (one render
    # detected as Japanese, empty English transcript) were left out. Accent is an
    # ear question that no machine number settles.
    CUSTOMVOICE_SPEAKERS: tuple[str, ...] = ("Ryan", "Vivian", "Aiden", "Sohee")
    # Affect-free identity sentences: the persona must carry no delivery, or it
    # would leak into the neutral twin. Even index = female, odd = male.
    PERSONAS: tuple[str, ...] = (
        "A woman in her thirties with a clear, mid-pitched General American English voice.",
        "A man in his thirties with a clear, mid-pitched General American English voice.",
        "A woman in her forties with a warm, slightly low General American English voice.",
        "A man in his forties with a steady, slightly low General American English voice.",
    )
    _AGED_PERSONA: ClassVar[dict[tuple[str, int], str]] = {
        ("elderly", 0): "An elderly woman in her eighties with a General American English voice.",
        ("elderly", 1): "An elderly man in his eighties with a General American English voice.",
        ("child", 0): "A young girl about seven years old with an American English voice.",
        ("child", 1): "A young boy about seven years old with an American English voice.",
    }

    def __init__(
        self,
        mode: str | None = None,
        speakers: tuple[str, ...] | None = None,
        load: bool = True,
        long_form: bool = False,
    ) -> None:
        import os

        mode = mode or "customvoice"
        if mode not in self.MODES:
            raise ValueError(f"qwen3tts mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.name = self.ENGINE_NAMES[mode]
        self.long_form = long_form
        env_speakers = os.environ.get("VOXPARITY_QWEN3TTS_SPEAKERS", "")
        self.speakers = speakers or (
            tuple(x.strip() for x in env_speakers.split(",") if x.strip())
            or self.CUSTOMVOICE_SPEAKERS
        )
        self.model = (
            QWEN3TTS_CUSTOMVOICE_MODEL if mode == "customvoice" else QWEN3TTS_VOICEDESIGN_MODEL
        )
        self._item_id: str | None = None
        self._model = None
        if load:
            # Lazy import: mlx-audio is a `local` extra (macOS-arm64 wheels only).
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model)
            if self.mode == "customvoice":
                supported = {s.lower() for s in self._model.get_supported_speakers()}
                missing = [s for s in self.speakers if s.lower() not in supported]
                if missing:
                    raise ValueError(f"qwen3tts presets {missing} not in {sorted(supported)}")

    def bind_item(self, item_id: str) -> None:
        self._item_id = item_id

    @staticmethod
    def _index(key: str, n: int) -> int:
        import hashlib

        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n

    def seed_for(self, item_id: str, variant_id: str) -> int:
        import hashlib

        return int(hashlib.sha256(f"{item_id}/{variant_id}".encode()).hexdigest()[:8], 16)

    def voice_and_instruct(self, item_id: str, variant: DeliveryVariant) -> tuple[str, str]:
        """(speaker-or-persona, instruct) for one variant of one item."""
        from voxparity.stimuli.style import delivery_clause, long_form_clause

        clause = long_form_clause(variant) if self.long_form else delivery_clause(variant)
        if self.mode == "customvoice":
            if variant.speaker is not None and variant.speaker.value != "adult":
                raise ValueError(
                    f"qwen3tts customvoice presets are fixed adult voices; refuses "
                    f"speaker profile {variant.speaker.value!r} on {variant.variant_id} "
                    "(use mode=voicedesign)"
                )
            speaker = self.speakers[self._index(item_id, len(self.speakers))]
            return speaker, f"Say the line {clause}."
        persona_idx = self._index(item_id, len(self.PERSONAS))
        persona = self.PERSONAS[persona_idx]
        if variant.speaker is not None and variant.speaker.value != "adult":
            persona = self._AGED_PERSONA[(variant.speaker.value, persona_idx % 2)]
        return persona, f"{persona} Delivery: say the line {clause}."

    def synthesize(self, transcript: str, variant: DeliveryVariant) -> tuple[bytes, str, str]:
        import io
        import wave

        import numpy as np

        if self._item_id is None:
            raise ValueError(
                "qwen3tts seeds its speaker by item id; call bind_item() before synthesize()"
            )
        if self._model is None:
            raise RuntimeError("qwen3tts model not loaded")
        voice, instruct = self.voice_and_instruct(self._item_id, variant)
        seed = self.seed_for(self._item_id, variant.variant_id)
        try:
            import mlx.core as mx

            mx.random.seed(seed)
        except ImportError:  # fake models in tests
            pass
        kwargs: dict[str, object] = dict(
            text=transcript, instruct=instruct, lang_code="english", temperature=self.TEMPERATURE
        )
        if self.mode == "customvoice":
            kwargs["voice"] = voice
        results = list(self._model.generate(**kwargs))
        if not results:
            raise RuntimeError(f"qwen3tts returned no audio for {variant.variant_id}")
        rate = getattr(results[0], "sample_rate", self.SAMPLE_RATE)
        if rate != self.SAMPLE_RATE:
            raise RuntimeError(f"qwen3tts sample rate {rate} != {self.SAMPLE_RATE}")
        audio = np.concatenate([np.asarray(r.audio, dtype=np.float32).reshape(-1) for r in results])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        provenance = (
            f"{instruct}\n[qwen3tts mode={self.mode} seed={seed} "
            f"temperature={self.TEMPERATURE} lang=english"
            f"{' wording=long' if self.long_form else ''}]"
        )
        voice_label = voice if self.mode == "customvoice" else f"persona:{voice}"
        return buf.getvalue(), voice_label, provenance


class Qwen3TTSCustomVoiceBackend(Qwen3TTSBackend):
    """Engine ``qwen3tts-cv``: preset speaker, the twin-identity-preserving mode."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(mode="customvoice", **kwargs)  # type: ignore[arg-type]


class Qwen3TTSVoiceDesignBackend(Qwen3TTSBackend):
    """Engine ``qwen3tts-vd``: persona prose; re-invents the voice per call (D100)."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(mode="voicedesign", **kwargs)  # type: ignore[arg-type]


_BACKENDS: dict[str, type[TTSBackend]] = {
    "gemini": GeminiBackend,
    "aura": AuraBackend,  # D087 scene-axis second engine; neutral-only, Gemini-gated
    "kokoro": KokoroBackend,  # D089 local; ACCEPTS disfluent transcripts (renders fillers)
    # local candidate instruction engine (D100); human-gated, one name per mode
    "qwen3tts-cv": Qwen3TTSCustomVoiceBackend,
    "qwen3tts-vd": Qwen3TTSVoiceDesignBackend,
    "cartesia": CartesiaBackend,  # retired (D034); refuses without an opt-in flag
    "elevenlabs": ElevenLabsBackend,  # policy-blocked (PX-001)
    "hume": HumeBackend,
    "camb": CambBackend,
    "fishaudio": FishAudioBackend,
    "inworld": InworldBackend,
}


def get_backend(name: str) -> TTSBackend:
    try:
        cls = _BACKENDS[name]
    except KeyError as e:
        raise ValueError(f"unknown engine {name!r}; choose from {sorted(_BACKENDS)}") from e
    return cls()
