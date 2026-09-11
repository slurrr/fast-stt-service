# Current Goal
Find a faster-whisper CPU model/config that materially improves STT quality without sacrificing interactive latency, then leave the service on the best balanced choice.

# Current State
- Benchmarked the current `small` and candidates on 10 curated English clips (167 s total), plus 8 existing user mic clips.
- Live service `127.0.0.1:8891` is running the new balanced config: `small.en`, CPU, int8, beam 1, VAD on, `STT_LANGUAGE=en`, `STT_CPU_THREADS=8`.
- Live `/health` confirms `model=small.en`, `cpu_threads=8`, `language=en`; API benchmark was 9.56 s for 10 clips (~0.90 s/request) and 0.69–0.79 s on five short user clips.
- Repo changes: `service.py` defaults to `small.en`/`en`, passes configurable `STT_CPU_THREADS`, and reports it from `/health`; `README.md` documents the balanced profile and multilingual rollback.
- Command-surface change: `/home/poop/code/dev/dotfiles/.local/bin/fast-stt-serve` now defaults to `small.en`/`en`, defaults CPU threads to 8 (0 for GPU), accepts `--cpu-threads`, and propagates the env. Dotfiles repo already had unrelated user modifications; only this wrapper was changed.

# Decisions
- Choose `small.en` rather than a larger checkpoint as the default balanced upgrade: on the directional caption benchmark it scored 10.5% WER and 16.5x realtime at int8/beam1/8 threads, versus current `small` auto-detect at 21.5% WER and 5.8x realtime. It also avoids the small multilingual checkpoint's catastrophic language-detection miss on one accented English clip.
- Keep `large-v3-turbo` and other candidates cached for optional maximum-quality testing; turbo reached 10.1% WER but only 6.0x realtime even with 24 CPU threads and had ~1.66 GB RSS vs ~0.74 GB for small.en. Larger models did not justify their short-command latency on this machine.
- Force English by default because this voice workflow is English. Roll back multilingual behavior with `STT_MODEL=small STT_LANGUAGE= STT_CPU_THREADS=8` (or pass equivalent wrapper flags).

# Open Problems
- The corpus reference includes auto-generated captions, so WER is directional rather than an absolute accuracy score.
- No ground-truth transcript exists for the user mic clips; quality there was checked by side-by-side transcript inspection only.
- If the user needs multilingual input, the English-only default is not appropriate; consider keeping `small` with an explicit language or accepting turbo's auto-detect latency.

# Resume Instructions
- Read this checkpoint first. Verify with `fast-stt-status` and `curl -sS http://127.0.0.1:8891/health`.
- Relevant files: `/home/poop/code/dev/fast-stt-service/service.py`, `README.md`, and `/home/poop/code/dev/dotfiles/.local/bin/fast-stt-serve`.
- Validation already run: `python -m py_compile service.py`, `uv run --project . ruff check service.py`, `bash -n ~/.local/bin/fast-stt-serve`, live API benchmark.
- If further tuning is requested, start with `STT_MODEL=large-v3-turbo STT_LANGUAGE=en STT_CPU_THREADS=24 STT_DTYPE=int8 STT_BEAM_SIZE=1`; use `fast-stt-stop` before replacing the live process.
