"""Serve NVIDIA NemotronLabs VoiceChat 11B (MLX 4-bit) as a file-replay arm.

VoiceChat is full duplex: it runs on an 80 ms audio timeline and emits three
aligned channels per frame (agent text, a separate FUNCTION channel for tool
calls, agent speech). There is no text-input path — the system prompt is the
only text — so the arm has no transcript twin (DriverCapabilities.text_twin =
False, the D035 case).

File replay follows NVIDIA's own offline recipe (examples/speechlm2/
offline_voicechat_infer.py): the clip is followed by trailing silence "so the
agent has time to respond", and the whole timeline is decoded greedily. The
speech decoder is skipped: in the reference loop the TTS state never feeds back
into the text or function logits (it consumes them), so dropping it changes no
text/function token and saves most of the compute. `--check-tts-parity` runs
one request both ways and asserts identical channels.

HTTP surface (port 8807), used by the llamacpp driver's `voicechat` channel:
  GET  /v1/models
  POST /v1/chat/completions  {messages: [system, user{input_audio}]}
       -> choices[0].message = {content: agent text, function: function channel,
          user_transcript: model's own RNNT transcript}
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

MODEL_ID = "mlx-community/NemotronLabs-VoiceChat-11B-4bit"
TRAILING_SILENCE_S = float(os.environ.get("VOXPARITY_VOICECHAT_TAIL_S", "8.0"))

_state: dict[str, Any] = {"batch_prefill": True}

PARITY_PROMPT = (
    "You are a phone agent for a home security company. Be brief.\n"
    "You can use the following tools to assist the user if required:\n"
    '<AVAILABLE_TOOLS>[{"type": "function", "function": {"name": "check_status", '
    '"description": "Check the status of an account", "parameters": {"type": "object", '
    '"properties": {"account": {"type": "string"}}, "required": []}}}]</AVAILABLE_TOOLS>\n'
    "If you decide to call any tool(s), use the following format:\n"
    '<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}]</TOOLCALL>'
)


def decode_timeline(session: Any, waveform: Any, system_prompt: str, tail_s: float) -> dict:
    """The reference offline loop (session.generate) minus the speech decoder."""
    import mlx.core as mx
    from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram

    model, cfg = session.model, session.model.config
    waveform = mx.pad(
        waveform.astype(mx.float32).squeeze(), (0, round(tail_s * cfg.input_sample_rate))
    )
    mel = log_mel_spectrogram(waveform, cfg.audio_config.preprocessor)
    audio_embeds, lengths, asr_embeds = model.stt_model.perception(
        mel, mx.array([mel.shape[1]], dtype=mx.int32)
    )
    frames = int(lengths[0])
    audio_embeds, asr_embeds = audio_embeds[:, :frames], asr_embeds[:, :frames]
    prompt_ids: list[int] = []
    if system_prompt.strip():
        prompt_ids = [
            cfg.bos_token_id,
            *session.tokenizer.encode(system_prompt, add_special_tokens=False),
            cfg.eos_token_id,
        ]
        prompt_embeds = model.stt_model.embed_tokens(mx.array([prompt_ids], dtype=mx.int32)).astype(
            audio_embeds.dtype
        )
        audio_embeds = mx.concatenate([prompt_embeds, audio_embeds], axis=1)
    n_prompt = len(prompt_ids)
    total = n_prompt + frames
    pad = cfg.pad_token_id
    text = [pad] * total
    func = [pad] * total
    cache = model.stt_model.make_cache()

    def fused(t_prev: int, f_prev: int, emb: Any) -> Any:
        return (
            model.stt_model.embed_tokens(mx.array([[t_prev]], dtype=mx.int32))
            + emb
            + cfg.function_channel_weight
            * model.stt_model.embed_tokens(mx.array([[f_prev]], dtype=mx.int32))
        )

    if n_prompt and _state["batch_prefill"]:
        # Prompt frames: outputs are discarded and both previous-token inputs are
        # PAD by construction, so every input is known up front and one batched
        # prefill is equivalent to the reference's frame-by-frame loop.
        out = model.stt_model(fused(pad, pad, audio_embeds[:, :n_prompt]), cache=cache)
        mx.eval(out.text_logits)
    elif n_prompt:
        for t in range(n_prompt):
            out = model.stt_model(fused(pad, pad, audio_embeds[:, t : t + 1]), cache=cache)
            mx.eval(out.text_logits)
    for t in range(n_prompt, total):
        prev_t = pad if t == 0 else text[t - 1]
        prev_f = pad if t == 0 else func[t - 1]
        out = model.stt_model(fused(prev_t, prev_f, audio_embeds[:, t : t + 1]), cache=cache)
        text[t] = int(mx.argmax(out.text_logits[:, -1]))
        func[t] = int(mx.argmax(out.function_logits[:, -1]))
    text_arr = mx.array(text[n_prompt:], dtype=mx.int32)
    func_arr = mx.array(func[n_prompt:], dtype=mx.int32)
    return {
        "text": session._decode_text(text_arr),
        "function": session._decode_text(func_arr),
        "user_transcript": session._rnnt_decode(asr_embeds, frames),
        "prompt_frames": n_prompt,
        "audio_frames": frames,
    }


def respond(body: dict[str, Any]) -> dict[str, Any]:
    from mlx_audio.stt.utils import load_audio

    session = _state["session"]
    system = ""
    audio_path = None
    for m in body["messages"]:
        c = m.get("content") or ""
        if m["role"] == "system":
            system = c if isinstance(c, str) else ""
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "input_audio":
                    fd, audio_path = tempfile.mkstemp(suffix=".wav")
                    os.write(fd, base64.b64decode(part["input_audio"]["data"]))
                    os.close(fd)
    if audio_path is None:
        raise ValueError("VoiceChat has no text-input path; an input_audio part is required")
    try:
        wav = load_audio(audio_path, sr=session.model.config.input_sample_rate)
        t0 = time.time()
        res = decode_timeline(session, wav, system, TRAILING_SILENCE_S)
        dt = time.time() - t0
    finally:
        os.unlink(audio_path)
    return {
        "id": f"voicechat-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": res["text"],
                    "function": res["function"],
                    "user_transcript": res["user_transcript"],
                },
                "finish_reason": "stop",
            }
        ],
        "timings": {
            "total_s": round(dt, 3),
            "prompt_frames": res["prompt_frames"],
            "audio_frames": res["audio_frames"],
            "tail_s": TRAILING_SILENCE_S,
            "batch_prefill": _state["batch_prefill"],
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(200, respond(body))
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[voicechat] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8807)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--check-tts-parity", metavar="WAV")
    a = ap.parse_args()
    from mlx_vlm import load

    model, processor = load(a.model)
    _state["session"] = model.create_session(processor)
    if a.check_tts_parity:
        from mlx_audio.stt.utils import load_audio

        s = _state["session"]
        wav = load_audio(a.check_tts_parity, sr=model.config.input_sample_rate)
        # A real system prompt, so the batched prefill is covered by the check.
        ref = s.generate(
            wav,
            system_prompt=PARITY_PROMPT,
            extra_decoding_seconds=TRAILING_SILENCE_S,
            use_language_cache=True,
        )
        ref_fn = s._decode_text(ref.function_tokens)
        ours = decode_timeline(s, wav, PARITY_PROMPT, TRAILING_SILENCE_S)
        ok = ref.text == ours["text"] and ref_fn == ours["function"]
        if not ok:  # batched prefill can differ numerically; fall back to per-frame
            _state["batch_prefill"] = False
            ours = decode_timeline(s, wav, PARITY_PROMPT, TRAILING_SILENCE_S)
            ok = ref.text == ours["text"] and ref_fn == ours["function"]
        print(f"[voicechat] batch_prefill={_state['batch_prefill']}", flush=True)
        print(
            f"[voicechat] fn={ours['function']!r} transcript={ours['user_transcript']!r}",
            flush=True,
        )
        print(
            f"[voicechat] tts-parity {'OK' if ok else 'MISMATCH'}: {ref.text!r} vs {ours['text']!r}"
        )
        if not ok:
            raise SystemExit(1)
    print(f"[voicechat] loaded {a.model}; serving on :{a.port}", flush=True)
    # Single-threaded: MLX streams are thread-local (see serve_phi4mm.py).
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
