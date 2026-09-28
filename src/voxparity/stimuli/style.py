"""Emotion + intensity -> TTS style instruction.

One canonical mapping so every engine receives semantically identical direction
(per-engine adapters may reformat, never reinterpret). Kept deliberately coarse,
mirroring the Emotion enum (blueprint §3: coarse classes only).
"""

from __future__ import annotations

from voxparity.schemas.item import DeliveryVariant, Emotion, SpeakerProfile

# D084 speaker axis: coarse, obvious-to-a-human framings only. Child renders
# are synthetic by construction (no real child's voice ever enters the
# pipeline) and ship only after the render probe + human 2AFC gate.
_SPEAKER_FRAME: dict[SpeakerProfile, str] = {
    SpeakerProfile.ADULT: "",
    # D095: Bhavik's 2AFC (Sep 13) — round-1 frame rendered borderline
    # (250 Hz, "can't tell"); the excited-6yo variant was ruled "cartoonish
    # fake child"; this frame passed as "a young child" on BOTH Leda and
    # Zephyr (348/358 Hz, WER 0).
    SpeakerProfile.CHILD: (
        "reading carefully like a second-grader, in a small, breathy child's voice, "
    ),
    SpeakerProfile.ELDERLY: (
        "as a frail elderly person in their eighties, slightly slow and breathy, "
    ),
}

_STYLE: dict[Emotion, str] = {
    Emotion.NEUTRAL: "in a calm, even, matter-of-fact tone",
    Emotion.HAPPY: "in a light, warm, upbeat tone, smiling audibly",
    Emotion.FRUSTRATED: "with tense, clipped frustration just under the surface",
    Emotion.ANGRY: "with open anger — hard, loud, confrontational",
    Emotion.SAD: "in a low, heavy, dejected voice close to tears",
    Emotion.ANXIOUS: "in a shaky, worried, hurried voice",
    Emotion.RESIGNED: "in a flat, defeated, given-up sigh of a voice",
    Emotion.URGENT: "fast and pressured, as if something is at stake right now",
    Emotion.SLURRED: "with slow, slurred, drowsy speech — words running together, effortful",
    Emotion.WHISPERED: "in a hushed, tense whisper, as if afraid of being overheard",
    Emotion.SARCASTIC: (
        "with dry, mocking sarcasm — the words say one thing and the voice makes "
        "plain it means the opposite"
    ),
    Emotion.BREATHLESS: (
        "gasping and breathless — only two or three words per breath, audible "
        "effortful breathing between phrases, strained"
    ),
    Emotion.CONFUSED: (
        "disoriented and lost — halting, trailing off mid-thought, uncertain of "
        "things that should be familiar, mildly bewildered"
    ),
    Emotion.AMUSED: (
        "laughing and amused — light, exasperated, joking, clearly not meaning the words literally"
    ),
}

_INTENSITY = [
    (0.34, "only subtly — barely perceptible"),
    (0.67, "clearly and naturally"),
    (1.01, "strongly and unmistakably"),
]


# OPTIONAL long-form wording (D100's open question: does richer description
# rescue an instruction-channel engine?). Used ONLY by engines that take a
# separate instruction and only when a caller asks for it (Qwen3TTSBackend
# long_form=True); tts_prompt never reads it, so Gemini's prompt is unchanged.
# Rules: vocal behaviour (breath, pitch movement, tempo, voice quality), never a
# stereotype label; one description per emotion, identical for every item; and
# restrained, since Bhavik heard Gemini's "strongly and unmistakably" renders as
# over-acted ("too much emotions can be unnatural", "too angry").
_LONG_FORM: dict[Emotion, str] = {
    Emotion.NEUTRAL: (
        "at an ordinary conversational pace in a relaxed, fully voiced tone; pitch "
        "moves only as much as the sentence needs, breathing stays quiet, and nothing "
        "in the voice leans toward any feeling"
    ),
    Emotion.HAPPY: (
        "with a small lift in the voice; pitch sits a little higher and rises easily "
        "at phrase ends, the tempo is relaxed and even, vowels are bright and open as "
        "if half-smiling, and the breath is easy"
    ),
    Emotion.FRUSTRATED: (
        "with held-in tension; consonants are crisp and slightly clipped, pitch stays "
        "fairly level with short firm drops at phrase ends, the pace is a touch quick, "
        "a short breath is pushed out between phrases, and the volume stays at normal "
        "speaking level"
    ),
    Emotion.ANGRY: (
        "with a raised, pressed voice; louder than conversation, hard onsets on stressed "
        "words, pitch pushed up and falling sharply at phrase ends, a fast even tempo "
        "with few pauses, and a tight, forceful voice quality"
    ),
    Emotion.SAD: (
        "with low energy; pitch sits low and moves very little, phrases trail downward, "
        "the tempo is slow with longer pauses, and the voice is soft and slightly breathy "
        "with an occasional unsteady moment, holding back tears rather than crying"
    ),
    Emotion.ANXIOUS: (
        "with nervous unsteadiness; pitch a little higher than usual and wavering, "
        "shallow breaths audible at phrase starts, an uneven tempo that speeds up and "
        "then catches, small hesitations, and a slightly tight, thin voice quality"
    ),
    Emotion.RESIGNED: (
        "with low, flat energy; pitch nearly level, a slow even tempo, a soft exhale "
        "before or after phrases, a relaxed and slightly breathy voice quality, and no "
        "push on any word"
    ),
    Emotion.URGENT: (
        "with focused pressure; a fast pace with very short pauses, stressed words "
        "landing firmly, pitch raised but controlled rather than shouting, quick breaths "
        "taken mid-sentence, and volume a little above normal"
    ),
    Emotion.SLURRED: (
        "with slowed, effortful articulation; consonants soft and smeared, neighbouring "
        "words running together, vowels drawn out, a slow uneven tempo with pauses in odd "
        "places, and a low, heavy, drowsy voice"
    ),
    Emotion.WHISPERED: (
        "as a true whisper with no voiced tone; air-only speech, very quiet, crisp "
        "consonants carrying the words, short quick phrases, and small audible breaths "
        "between them"
    ),
    Emotion.SARCASTIC: (
        "with a dry edge; a slower, deliberate tempo, pitch rises and falls stretched "
        "over the key words, slightly elongated vowels on the words that are not meant, "
        "and a flat, unimpressed voice quality underneath"
    ),
    Emotion.BREATHLESS: (
        "short of breath; two or three words per breath, audible effortful in-breaths "
        "between phrases, and a strained, thin voice that weakens at the end of each "
        "phrase"
    ),
    Emotion.CONFUSED: (
        "unsure and slightly lost; an uneven tempo with pauses mid-phrase, pitch rising "
        "at phrase ends, some words drawn out while searching for the next, and a soft, "
        "hesitant voice quality"
    ),
    Emotion.AMUSED: (
        "with laughter just under the words; a light, bright voice, pitch lifting on "
        "stressed words, a quick relaxed tempo, a breathy chuckle at the edges of "
        "phrases, and the voice never turning serious"
    ),
}

_LONG_FORM_INTENSITY = [
    (0.34, "only slightly"),
    (0.67, "noticeably but naturally"),
    (1.01, "clearly, like a real caller on the phone rather than an actor performing"),
]


def style_instruction(variant: DeliveryVariant) -> str:
    base = _STYLE[variant.emotion]
    for ceiling, qualifier in _INTENSITY:
        if variant.intensity < ceiling:
            return f"{base}, expressed {qualifier}"
    raise ValueError(f"intensity out of range: {variant.intensity}")


def delivery_clause(variant: DeliveryVariant) -> str:
    """Speaker frame + style instruction, with no transcript attached.

    The one authored-intent string every instruction-driven engine receives:
    Gemini wraps it around the line (tts_prompt), engines with a separate
    instruction channel (Qwen3-TTS ``instruct``) take it on its own.
    """
    frame = _SPEAKER_FRAME.get(variant.speaker, "") if variant.speaker else ""
    return f"{frame}{style_instruction(variant)}"


def long_form_clause(variant: DeliveryVariant) -> str:
    """Speaker frame + long-form vocal-behaviour description (opt-in only).

    The long-form counterpart of delivery_clause for instruction-channel engines.
    Never used by tts_prompt.
    """
    frame = _SPEAKER_FRAME.get(variant.speaker, "") if variant.speaker else ""
    for ceiling, qualifier in _LONG_FORM_INTENSITY:
        if variant.intensity < ceiling:
            return f"{frame}{_LONG_FORM[variant.emotion]}, {qualifier}"
    raise ValueError(f"intensity out of range: {variant.intensity}")


def tts_prompt(transcript: str, variant: DeliveryVariant) -> str:
    """Full prompt for style-promptable TTS (Gemini-TTS pattern)."""
    return (
        f"Say the following line {delivery_clause(variant)}. "
        f"Speak only the line itself, exactly as written:\n{transcript}"
    )
