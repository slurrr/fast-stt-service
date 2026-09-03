from __future__ import annotations

import contextlib
import io
import math
import os
import struct
import tempfile
import time
import wave
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import BatchedInferencePipeline, WhisperModel

DEFAULT_MODEL = os.getenv("STT_MODEL", "small")
DEFAULT_DEVICE = os.getenv("STT_DEVICE", "cpu")  # cpu | cuda
DEFAULT_DTYPE = os.getenv("STT_DTYPE", os.getenv("STT_COMPUTE_TYPE", "int8" if DEFAULT_DEVICE == "cpu" else "fp16"))
DEFAULT_LANGUAGE = os.getenv("STT_LANGUAGE", "")
DEFAULT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "1"))
DEFAULT_BATCH_SIZE = int(os.getenv("STT_BATCH_SIZE", "1"))
DEFAULT_VAD_FILTER = os.getenv("STT_VAD_FILTER", "1").strip().lower() not in {"0", "false", "no", "off"}
DEFAULT_WARMUP = os.getenv("STT_WARMUP", "1").strip().lower() not in {"0", "false", "no", "off"}


def _dtype_to_compute_type(dtype: str) -> str:
    normalized = str(dtype or "").strip().lower()
    if normalized in {"fp16", "float16"}:
        return "float16"
    if normalized in {"fp32", "float32"}:
        return "float32"
    if normalized in {"int8"}:
        return "int8"
    if normalized in {"int8_float16"}:
        return "int8_float16"
    return "int8" if DEFAULT_DEVICE == "cpu" else "float16"


DEFAULT_COMPUTE_TYPE = _dtype_to_compute_type(DEFAULT_DTYPE)

app = FastAPI(title="Fast STT Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: WhisperModel | None = None
_batched_model: BatchedInferencePipeline | None = None
_model_loaded_at: float | None = None


def _load_model() -> WhisperModel:
    global _model, _model_loaded_at
    if _model is None:
        _model = WhisperModel(DEFAULT_MODEL, device=DEFAULT_DEVICE, compute_type=DEFAULT_COMPUTE_TYPE)
        _model_loaded_at = time.time()
    return _model


def _load_batched_model() -> BatchedInferencePipeline:
    global _batched_model
    if _batched_model is None:
        _batched_model = BatchedInferencePipeline(model=_load_model())
    return _batched_model


def _to_srt(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _segments_to_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_to_srt(float(seg['start']))} --> {_to_srt(float(seg['end']))}")
        lines.append(str(seg["text"]).strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _warmup_audio_path() -> str:
    duration_s = 0.75
    sample_rate = 16000
    frames = int(sample_rate * duration_s)
    with tempfile.NamedTemporaryFile(prefix="stt-warmup-", suffix=".wav", delete=False) as tmp:
        path = tmp.name

    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            buf = bytearray()
            for i in range(frames):
                sample = int(5000 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
                buf.extend(struct.pack("<h", sample))
                if len(buf) >= 4096:
                    wf.writeframesraw(buf)
                    buf.clear()
            if buf:
                wf.writeframesraw(buf)
        return path
    except Exception:
        with contextlib.suppress(Exception):
            os.unlink(path)
        raise


def _warmup_model() -> None:
    if not DEFAULT_WARMUP:
        return

    _load_model()
    if DEFAULT_BATCH_SIZE > 1:
        _load_batched_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "device": DEFAULT_DEVICE,
        "dtype": DEFAULT_DTYPE,
        "compute_type": DEFAULT_COMPUTE_TYPE,
        "language": DEFAULT_LANGUAGE or "auto",
        "beam_size": DEFAULT_BEAM_SIZE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "vad_filter": DEFAULT_VAD_FILTER,
        "loaded": _model is not None,
        "loaded_at": _model_loaded_at,
    }


@app.get("/v1/audio/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": DEFAULT_MODEL, "object": "model", "owned_by": "local"},
        ],
    }


@app.on_event("startup")
def startup() -> None:
    _warmup_model()


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
) -> Any:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing file")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    model_obj = _load_model()

    with tempfile.NamedTemporaryFile(prefix="stt-", suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()

        transcribe_kwargs = dict(
            language=(language or DEFAULT_LANGUAGE or None),
            beam_size=DEFAULT_BEAM_SIZE,
            vad_filter=DEFAULT_VAD_FILTER,
            initial_prompt=(prompt or None),
        )
        if temperature is not None:
            transcribe_kwargs["temperature"] = float(temperature)
        if DEFAULT_BATCH_SIZE > 1:
            batched = _load_batched_model()
            segments_iter, info = batched.transcribe(
                tmp.name,
                batch_size=DEFAULT_BATCH_SIZE,
                **transcribe_kwargs,
            )
        else:
            segments_iter, info = model_obj.transcribe(
                tmp.name,
                **transcribe_kwargs,
            )
        segments = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments_iter
        ]

    text = "".join(seg["text"] for seg in segments).strip()

    if response_format == "text":
        return text
    if response_format == "srt":
        return _segments_to_srt(segments)
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": info.language,
            "duration": info.duration,
            "text": text,
            "segments": segments,
            "model": model,
        }

    return {"text": text}
