"""Serve Phi-4-multimodal-instruct (bf16, MLX) behind the llama-server HTTP surface.

Run through scripts/serve-phi4mm.sh, which pins mlx-vlm in an isolated uv env so
the project lockfile is untouched. The driver is the ordinary llamacpp driver
(`--driver llamacpp:phi-4-multimodal-instruct`, port 8804): same
/v1/models + /v1/chat/completions shape, `input_audio` base64 WAV parts.

Why a thin server of our own rather than `mlx_vlm.server`:

1. LoRA modality. Phi-4-MM is one backbone with two LoRAs (speech, vision); the
   reference HF code runs text-only input on the BASE weights. mlx-vlm switches
   LoRA only when an image or audio is present, so a text-only request after an
   audio request silently runs with the speech LoRA still merged (and the first
   one with the vision LoRA merged by default). The transcript twin would then be
   a different network from the one the reference defines, and audio-minus-twin
   would carry a LoRA artifact. We set the modality explicitly on every request.
2. Prompt format. The model card's format is rendered verbatim
   (`<|system|>...<|end|><|user|><|audio_1|><|end|><|assistant|>`); tools arrive
   already embedded by the driver between `<|tool|>` tokens.
3. bf16, not the 4-bit convert: mlx-vlm's quantizer pre-merges BOTH LoRAs into
   the backbone, which is a different system from the released model.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

_lock = threading.Lock()
_state: dict[str, Any] = {}


def render(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Render messages in the model-card format; returns (prompt, audio paths)."""
    prompt = ""
    audios: list[str] = []
    for m in messages:
        role = m["role"]
        content = m.get("content") or ""
        text = ""
        if isinstance(content, str):
            text = content
        else:
            for part in content:
                if part.get("type") == "input_audio":
                    data = base64.b64decode(part["input_audio"]["data"])
                    fd, path = tempfile.mkstemp(suffix=".wav")
                    os.write(fd, data)
                    os.close(fd)
                    audios.append(path)
                    text += f"<|audio_{len(audios)}|>"
                elif part.get("type") == "text":
                    text += part.get("text", "")
        prompt += f"<|{role}|>{text}<|end|>"
    return prompt + "<|assistant|>", audios


def generate(body: dict[str, Any]) -> dict[str, Any]:
    from mlx_vlm import generate as mlx_generate  # type: ignore[import-not-found]

    model, processor = _state["model"], _state["processor"]
    prompt, audios = render(body["messages"])
    try:
        with _lock:
            # Explicit modality every call (see module docstring, point 1).
            model.set_modality(has_image=False, has_audio=bool(audios))
            t0 = time.time()
            res = mlx_generate(
                model,
                processor,
                prompt,
                audio=audios or None,
                max_tokens=int(body.get("max_tokens", 400)),
                temperature=float(body.get("temperature", 0.0)),
                verbose=False,
            )
            dt = time.time() - t0
    finally:
        for a in audios:
            os.unlink(a)
    text = res.text if hasattr(res, "text") else str(res)
    return {
        "id": f"phi4mm-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": getattr(res, "prompt_tokens", None),
            "completion_tokens": getattr(res, "generation_tokens", None),
        },
        "timings": {"total_s": round(dt, 3), "active_lora": getattr(model, "_active_lora", None)},
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
        elif self.path.rstrip("/") == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("tools"):
                # Phi takes tools in the system turn, embedded by the driver.
                self._send(400, {"error": "OpenAI `tools` unsupported; embed <|tool|> in system"})
                return
            self._send(200, generate(body))
        except Exception as e:  # surfaced to the driver as HTTP 500 -> error row
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[phi4mm] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8804)
    ap.add_argument("--model", default=MODEL_ID)
    a = ap.parse_args()
    from mlx_vlm import load

    model, processor = load(a.model, trust_remote_code=True)
    _state.update(model=model, processor=processor)
    print(f"[phi4mm] loaded {a.model}; serving on :{a.port}", flush=True)
    # Single-threaded on purpose: MLX streams are thread-local, so generation
    # must run on the thread that loaded the model. Requests queue in accept().
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
