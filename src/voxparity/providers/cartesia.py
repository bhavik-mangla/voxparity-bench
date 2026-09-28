"""Cartesia Sonic TTS — RETIRED as a stimulus engine (D034).

Kept in tree for reproducibility of the M2-M4 results, and because the client is
now correct; disabled by default because the engine cannot render the one thing
this benchmark needs.

WHY RETIRED. Cartesia documents that emotion "only work[s] when the emotion is
consistent with the transcript", giving ``<emotion value="sad"/> I'm so excited!``
as an explicit non-working example. Counterfactual delivery — the same sentence
spoken against its own semantics — is the entire VoxParity item design. Measured
on the corrected client: **1/99 pair discrimination** across the bank, with 76
items where the judge labelled BOTH deliveries wrong, and 0/3 on a fresh
re-render. Neutral vs urgent came out at 0.0% duration difference and 10.8 Hz of
median pitch. Gemini-TTS on the same items scores 20/25 pairs.

The client below was fixed first, so the retirement is a finding about the engine
rather than about our code (D031): we had been sending a wrapping
``<emotion ...>text</emotion>`` element (the documented tag is self-closing), the
``2025-04-16`` version header (the enum admits only ``2026-08-14``), and an 8->5
emotion collapse against what is actually a 58-value enum. All corrected here.
The supported mechanism is ``generation_config.emotion``; the inline SSML tag and
the long-gone ``__experimental_controls`` block are deliberately unused.

Set VOXPARITY_ALLOW_CARTESIA=1 to re-enable — e.g. to reproduce a published
number, or to re-test if Cartesia ships incongruent-emotion support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from voxparity.providers.http import post_with_retry
from voxparity.providers.keys import EXHAUSTED_STATUSES, KeyPool
from voxparity.schemas.item import DeliveryVariant, Emotion

BASE = "https://api.cartesia.ai"
# Required header; the API's version enum currently admits exactly this value.
VERSION = os.environ.get("VOXPARITY_CARTESIA_VERSION", "2026-08-14")
MODEL = os.environ.get("VOXPARITY_CARTESIA_MODEL", "sonic-3.6")
# Documented default voice (overridable); chosen once and pinned as a controlled
# variable per methodology commandment 8.
DEFAULT_VOICE = os.environ.get("VOXPARITY_CARTESIA_VOICE", "")

# VoxParity Emotion -> Cartesia `generation_config.emotion` enum value.
# Only URGENT is lossy: the enum has no "urgent"; "alarmed" is its nearest
# neighbour (panicked/scared overshoot into fear). Recorded per clip so the
# substitution is auditable rather than silent.
_EMOTION_MAP: dict[Emotion, str] = {
    Emotion.NEUTRAL: "neutral",
    Emotion.HAPPY: "happy",
    Emotion.FRUSTRATED: "frustrated",
    Emotion.ANGRY: "angry",
    Emotion.SAD: "sad",
    Emotion.ANXIOUS: "anxious",
    Emotion.RESIGNED: "resigned",
    Emotion.URGENT: "alarmed",
    # D065 additions. Nearest enum neighbours only: Cartesia has no true
    # slurred/whispered value, so both are lossy (engine is retired anyway, D034).
    Emotion.SLURRED: "tired",
    Emotion.WHISPERED: "distant",
    # D078 addition: NATIVE value — Cartesia's 58-value enum contains "sarcastic"
    # by name (D031), so this is the rare non-lossy hard-delivery mapping.
    Emotion.SARCASTIC: "sarcastic",
    # D083 additions. Nearest enum neighbours: "confused" is NATIVE; breathless
    # and amused have no value (no panting/laughing register in the 58) — both
    # are lossy — "elated" at least keeps amused's positive valence distinct
    # from the hostile/sarcastic cluster (engine retired anyway, D034).
    Emotion.BREATHLESS: "panicked",
    Emotion.CONFUSED: "confused",
    Emotion.AMUSED: "elated",
}
LOSSY_EMOTIONS = {
    Emotion.URGENT,
    Emotion.SLURRED,
    Emotion.WHISPERED,
    Emotion.BREATHLESS,
    Emotion.AMUSED,
}

RETIRED = (
    "Cartesia is retired as a stimulus engine (D034): 1/99 pair discrimination "
    "across the bank, 76 items with BOTH deliveries mislabelled, and the vendor "
    "documents that emotion control only works when the emotion is consistent "
    "with the transcript — which is the opposite of a counterfactual item. "
    "Set VOXPARITY_ALLOW_CARTESIA=1 to reproduce an existing result anyway."
)


class CartesiaError(RuntimeError):
    pass


@dataclass
class CartesiaClient:
    api_key: str | None = None
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if os.environ.get("VOXPARITY_ALLOW_CARTESIA", "") != "1":
            raise CartesiaError(RETIRED)
        try:
            self.pool = KeyPool("CARTESIA", keys=[self.api_key] if self.api_key else [])
        except RuntimeError as e:
            raise CartesiaError(str(e)) from e
        self.api_key = self.pool.current

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.pool.current, "Cartesia-Version": VERSION}

    def default_voice_id(self) -> str:
        if DEFAULT_VOICE:
            return DEFAULT_VOICE
        resp = httpx.get(f"{BASE}/voices", headers=self._headers(), timeout=self.timeout)
        if resp.status_code != 200:
            raise CartesiaError(f"voices: HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        voices = data if isinstance(data, list) else data.get("data", [])
        english = [v for v in voices if v.get("language", "en") == "en"]
        if not english:
            raise CartesiaError("no English voices available")
        return str(english[0]["id"])

    def control_label(self, variant: DeliveryVariant) -> str:
        """Provenance string recorded in the manifest for this variant."""
        value = _EMOTION_MAP[variant.emotion]
        label = f'generation_config.emotion="{value}"'
        if variant.emotion in LOSSY_EMOTIONS:
            label += f" (lossy: no '{variant.emotion.value}' in enum)"
        return label

    def synthesize(self, transcript: str, variant: DeliveryVariant, voice_id: str) -> bytes:
        body = {
            "model_id": MODEL,
            "transcript": transcript,
            "voice": {"id": voice_id},
            "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
            "language": "en",
            "generation_config": {"emotion": _EMOTION_MAP[variant.emotion]},
        }
        tried = 0
        while True:
            try:
                resp = post_with_retry(
                    f"{BASE}/tts/bytes",
                    headers=self._headers(),
                    json_body=body,
                    timeout=self.timeout,
                    tries=2,
                )
            except (httpx.TransportError, TimeoutError) as e:
                raise CartesiaError(f"tts: {e}") from e
            if resp.status_code in EXHAUSTED_STATUSES:
                tried += 1
                if tried < self.pool.size and self.pool.rotate():
                    continue
            break
        if resp.status_code != 200:
            raise CartesiaError(f"tts: HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.content
