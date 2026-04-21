# app.py
import argparse
import os
import io
import wave
import hashlib
import tempfile
import threading
from typing import Optional, Tuple, Dict, Any

import torch  # noqa: F401
import torchaudio  # noqa: F401
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from sarashina_tts.generate.generate import SarashinaTTSGenerator

app = FastAPI(title="Sarashina‑TTS API", version="1.0.0")

parser = argparse.ArgumentParser(description="Sarashina‑TTS API server")
parser.add_argument("--model-dir", default=None, help="Path to model directory (default: pretrained_models)")
parser.add_argument("--use-vllm", action="store_true", help="Use vLLM backend for faster inference")
parser.add_argument("--config", default=None, help="Path to JSON config (advanced)")
parser.add_argument("--prompt", default=None, help="Path to default prompt audio file")
parser.add_argument("--prompt-text", default=None, help="Transcription of the default prompt audio")
parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
args, _ = parser.parse_known_args()

if args.config:
    gen = SarashinaTTSGenerator.from_config(args.config)
else:
    gen = SarashinaTTSGenerator(
        model_dir=args.model_dir,
        use_vllm=args.use_vllm,
    )

DEFAULT_PROMPT_PATH = (
    args.prompt
    or os.environ.get("SARASHINA_TTS_PROMPT_FILE")
    or os.path.join(os.path.dirname(__file__), "prompt.wav")
)
DEFAULT_PROMPT_TEXT = (
    args.prompt_text
    or os.environ.get("SARASHINA_TTS_PROMPT_TEXT", "")
)

# prompt cache: (flow_emb, tokens, feat)
_cache: Dict[str, Tuple[Any, Any, Any]] = {}
_cache_lock = threading.Lock()


def _hash_key(audio_bytes: bytes, prompt_text: str) -> str:
    m = hashlib.md5()
    m.update(audio_bytes)
    m.update(prompt_text.encode("utf-8"))
    return m.hexdigest()


def _ensure_tmp_file(upload: UploadFile, audio_bytes: bytes) -> str:
    """Generator expects path: write uploaded audio to temporary file"""
    suffix = os.path.splitext(upload.filename or "")[-1] or ".wav"
    fd, path = tempfile.mkstemp(prefix="prompt_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return path


def _prepare_embeddings(prompt_path: str, cache_key: str):
    """Prepare/read cached flow_emb, tokens, feat (consistent with original demo's private methods)"""
    with _cache_lock:
        hit = _cache.get(cache_key)
    if hit is not None:
        return hit

    flow_emb = gen._extract_zero_shot_embedding(prompt_path)
    tokens = gen._extract_audio_prompt_tokens(prompt_path)
    feat = gen._extract_audio_prompt_feat(prompt_path)

    with _cache_lock:
        _cache[cache_key] = (flow_emb, tokens, feat)
    return flow_emb, tokens, feat


def _detect_sample_rate(default_sr: int = 24000) -> int:
    """Try to get sample rate from generator; otherwise fallback to 24000"""
    for attr in ("sample_rate", "sr", "sampling_rate", "target_sr"):
        if hasattr(gen, attr):
            try:
                val = int(getattr(gen, attr))
                if val > 0:
                    return val
            except Exception:
                pass
    return default_sr


def _extract_audio_and_sr(wavs) -> Tuple[torch.Tensor, int]:
    """
    Compatible with common return formats:
    - [Tensor] or [ (Tensor, sr) ] or [ {'audio': Tensor, 'sampling_rate': sr} ]
    """
    if not wavs:
        raise HTTPException(status_code=500, detail="Generation returned no audio.")

    first = wavs[0]

    # case: (Tensor, sr)
    if isinstance(first, (tuple, list)) and len(first) == 2:
        audio, sr = first
        return torch.as_tensor(audio), int(sr)

    # case: dict
    if isinstance(first, dict):
        audio = first.get("audio")
        sr = first.get("sampling_rate") or first.get("sr") or first.get("sample_rate")
        if audio is None:
            raise HTTPException(status_code=500, detail="Missing 'audio' in generation output.")
        if sr is None:
            sr = _detect_sample_rate()
        return torch.as_tensor(audio), int(sr)

    # case: Tensor only
    if torch.is_tensor(first):
        return first, _detect_sample_rate()

    # Unknown structure
    raise HTTPException(status_code=500, detail=f"Unsupported generation output type: {type(first)}")


def _to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """
    Convert [-1, 1] float Tensor to PCM16 WAV bytes (no disk write)
    Supports shape: (T,) or (C, T)
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # -> (1, T)
    elif audio.ndim == 2:
        pass
    else:
        raise HTTPException(status_code=500, detail=f"Unexpected audio tensor shape: {tuple(audio.shape)}")

    # Clamp to [-1,1] and convert to int16
    audio = torch.clamp(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).round().to(torch.int16)  # (C, T)

    # Write to BytesIO
    buf = io.BytesIO()
    num_channels, num_frames = audio_i16.shape[0], audio_i16.shape[1]

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(sample_rate))
        # wave module requires byte sequence (interleaved channels: C,T -> T,C)
        interleaved = audio_i16.transpose(0, 1).contiguous().numpy().tobytes()
        wf.writeframes(interleaved)

    buf.seek(0)
    return buf.read()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize(
    prompt_file: Optional[UploadFile] = File(
        None, description="Prompt audio file (optional)."
    ),
    prompt_text: Optional[str] = Form(
        None, description="Describe the prompt voice"
    ),
    flow_embedding_only: bool = Form(False),
    text_to_synth: str = Form(..., description="Text to synthesize (single line)"),
    max_length: int = Form(2000),
    repetition_penalty: float = Form(1.0),
    do_sample: bool = Form(True),
    temperature: float = Form(1.0),
    top_p: float = Form(0.95),
):
    """
    Return value: directly return audio/wav (binary stream), no server-side saving, no JSON return.
    """
    try:
        prompt_text = (prompt_text or "").strip()

        if prompt_file is None:
            prompt_path = os.path.abspath(DEFAULT_PROMPT_PATH)
            if not os.path.isfile(prompt_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Default prompt file not found: {prompt_path}",
                )
            try:
                with open(prompt_path, "rb") as f:
                    audio_bytes = f.read()
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to read default prompt file: {exc}",
                )

            if not audio_bytes:
                raise HTTPException(
                    status_code=500, detail="Default prompt file is empty."
                )
            if not prompt_text:
                prompt_text = DEFAULT_PROMPT_TEXT
        else:
            audio_bytes = await prompt_file.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Empty prompt_file.")

            prompt_path = _ensure_tmp_file(prompt_file, audio_bytes)

        if not prompt_text:
            prompt_text = ""

        # Parameter boundaries (consistent with Gradio sliders)
        if not (1 <= max_length <= 4000):
            raise HTTPException(status_code=422, detail="max_length must be in [1, 4000].")
        if not (0.1 <= repetition_penalty <= 5.0):
            raise HTTPException(status_code=422, detail="repetition_penalty must be in [0.1, 5.0].")
        if not (0.1 <= temperature <= 2.0):
            raise HTTPException(status_code=422, detail="temperature must be in [0.1, 2.0].")
        if not (0.0 <= top_p <= 1.0):
            raise HTTPException(status_code=422, detail="top_p must be in [0.0, 1.0].")

        # Cache key and prepare embedding
        key = _hash_key(audio_bytes, prompt_text)
        flow_emb, tokens, feat = _prepare_embeddings(prompt_path, key)

        gen_kwargs = {
            "max_length": int(max_length),
            "repetition_penalty": float(repetition_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

        # Generate
        wavs = gen.generate(
            texts=[text_to_synth],
            flow_embedding=flow_emb,
            audio_prompt_path=prompt_path,
            audio_prompt_text=prompt_text,
            audio_prompt_tokens=tokens,
            audio_prompt_feat=feat,
            flow_embedding_only=flow_embedding_only,
            gen_kwargs=gen_kwargs,
        )

        # Get Tensor and sample rate -> convert to WAV bytes
        audio, sr = _extract_audio_and_sr(wavs)
        wav_bytes = _to_wav_bytes(audio, sr)

        # Direct streaming return, no disk write, no JSON return
        filename = "tts_generated.wav"
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
