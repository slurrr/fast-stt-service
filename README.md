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

- `STT_MODEL` (default `small`)
- `STT_LANGUAGE` (default automatic detection)
- `STT_DEVICE` (`cpu` by default, or `cuda`)
- `STT_DTYPE`, `STT_BEAM_SIZE`, `STT_BATCH_SIZE`, and `STT_VAD_FILTER`

Run directly with `uv run --project . uvicorn service:app --host 127.0.0.1 --port 8891`,
or use `fast-stt-serve --bg`. The stack manifest invokes the
provider wrapper and does not copy provider settings into shared runtime state.
