# fast-stt-service

A self-hosted, OpenAI-compatible speech-to-text HTTP service built on
faster-whisper. It lets a local voice client post a recording to a familiar
`/v1/audio/transcriptions` endpoint and receive text, while model choice,
language, device, precision, VAD, beam size and batching stay configured on the
service instead of in every client.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Network access on first run: the Whisper checkpoint is downloaded from
  Hugging Face. The default `small.en` model is roughly 500 MB. Later runs reuse
  the local cache.

## Install

```bash
uv sync
```

## Run

```bash
uv run uvicorn service:app --host 127.0.0.1 --port 8891
```

The project is a plain ASGI application, so any ASGI server can host it. With
the default `STT_WARMUP=1` the checkpoint loads at startup; set
`STT_WARMUP=0` to load it on the first request instead.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness and the effective runtime settings |
| `GET` | `/v1/audio/models` | List the model id the service serves |
| `POST` | `/v1/audio/transcriptions` | Transcribe a multipart `file` upload |

`POST /v1/audio/transcriptions` accepts the OpenAI fields `file`, `model`,
`language`, `prompt`, `response_format` and `temperature`. `response_format`
may be `json` (default, returns `{"text": "..."}`), `text` (plain text), `srt`
(subtitles) or `verbose_json` (text plus segment timestamps, language and
duration).

Audio is decoded by PyAV, which ships its own codecs, so WAV, FLAC, MP3, OGG
and M4A input generally work without a system `ffmpeg`.

## Example

```bash
curl -sS http://127.0.0.1:8891/health
```

```json
{"status": "ok", "model": "small.en", "device": "cpu", "dtype": "int8", "compute_type": "int8", "cpu_threads": 8, "language": "en", "beam_size": 1, "batch_size": 1, "vad_filter": true, "loaded": true, "loaded_at": 1767225600.0}
```

```bash
curl -sS \
  -F file=@recording.wav \
  -F response_format=verbose_json \
  http://127.0.0.1:8891/v1/audio/transcriptions
```

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 2.4,
  "text": " Remind me to water the plants.",
  "segments": [
    {"start": 0.0, "end": 2.4, "text": " Remind me to water the plants."}
  ],
  "model": "whisper-1"
}
```

## Configuration

Settings come from environment variables read at startup:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STT_MODEL` | `small.en` | faster-whisper model name or local checkpoint path |
| `STT_LANGUAGE` | `en` | Forced language; set empty for auto-detection |
| `STT_DEVICE` | `cpu` | `cpu` or `cuda` |
| `STT_DTYPE` | `int8` on CPU, `fp16` on CUDA | `int8`, `int8_float16`, `fp16` or `fp32` |
| `STT_CPU_THREADS` | `8` on CPU, `0` on CUDA | Worker threads; `0` lets CTranslate2 choose |
| `STT_BEAM_SIZE` | `1` | Beam width for decoding |
| `STT_BATCH_SIZE` | `1` | Values above 1 enable batched inference |
| `STT_VAD_FILTER` | `1` | Strip silence before transcription |
| `STT_WARMUP` | `1` | Load the model during startup |

The defaults are a balanced CPU profile: `small.en` with int8 weights, beam 1
and eight threads. The English-only checkpoint avoids the language-detection
failure mode of multilingual `small` while keeping short-request latency low.
For multilingual input use `STT_MODEL=small` and an empty `STT_LANGUAGE`; for
higher accuracy at substantially greater CPU time and memory use, try
`STT_MODEL=large-v3-turbo`.

## Verify

Start the service as shown above, then run this from a fresh clone:

```bash
python - <<'PY'
import math, struct, wave

with wave.open("/tmp/stt-check.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes(b"".join(
        struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / 16000)))
        for i in range(16000)
    ))
PY

curl -sS -F file=@/tmp/stt-check.wav \
  http://127.0.0.1:8891/v1/audio/transcriptions
```

A healthy service answers with HTTP 200 and a JSON object containing a `text`
field. A pure tone usually transcribes to an empty string; the 200 response
confirms that upload handling, audio decoding, model inference and response
formatting all ran. `GET /health` should report `"loaded": true` once the model
has been loaded.

## License

MIT. See [LICENSE](LICENSE).
