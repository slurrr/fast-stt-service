# fast-stt-service

OpenAI-compatible local speech-to-text service backed by faster-whisper.
`the-stack` selects `fast-whisper` as the default STT provider and owns only
its endpoint and lifecycle. This service owns the model, language, device,
dtype, VAD, beam, and batching defaults.

## Transport

- `GET /health` reports service readiness and provider-owned runtime details.
- `POST /v1/audio/transcriptions` accepts a multipart `file` and returns
  `{ "text": "..." }` by default. `response_format=verbose_json` additionally
  returns segment metadata.

The shared voice loop sends only the recording and `response_format=verbose_json`;
it does not send model or language settings.

## Provider settings

Settings are configured locally with environment variables or the
`fast-stt-serve` provider wrapper:

- `STT_MODEL` (default `small.en`, English-only)
- `STT_LANGUAGE` (default `en`)
- `STT_DEVICE` (`cpu` by default, or `cuda`)
- `STT_DTYPE`, `STT_CPU_THREADS` (default `8` on CPU), `STT_BEAM_SIZE`, `STT_BATCH_SIZE`, and `STT_VAD_FILTER`

The default is the balanced CPU profile: `small.en` with int8 weights, beam 1,
and eight CPU threads. It avoids the language-detection failure mode of the
multilingual `small` checkpoint while retaining its low latency. Use
`STT_MODEL=small STT_LANGUAGE=` when multilingual auto-detection is required;
use `large-v3-turbo` when maximum accuracy is worth substantially higher CPU
latency and memory use.

Run directly with `uv run --project . uvicorn service:app --host 127.0.0.1 --port 8891`,
or use `fast-stt-serve --bg`. The stack manifest invokes the
provider wrapper and does not copy provider settings into shared runtime state.
