# Foreign Whispers — AI Video Dubbing Pipeline

> An open-source, end-to-end video dubbing system that takes a YouTube video, transcribes it, diarizes speakers, translates it, synthesizes gender-aware dubbed audio, and produces a final dubbed video — all without any paid API keys or GPU required.

> **NYU Spring 2026 — NLP Project**
> **Students:** Tilak Bhansali (`tb3525`) · Advait Jishnani (`aj4700`) 

---

## 🎬 Demo

| | Link |
|---|---|
| **Screen Recording (Pipeline Demo)** | https://youtu.be/IybYzYxw5Mc |
| **Sample Output Video** | https://youtu.be/rJDAIvVWwW0 |

---

## What It Does

```
YouTube URL → Download → Transcribe → Diarize → Translate → TTS → Stitch → Dubbed Video
```

- **Download** — Fetches video and captions from YouTube using `yt-dlp`
- **Transcribe** — Converts speech to text using OpenAI Whisper (`base`, CPU, runs inside API container)
- **Diarize** — Detects speaker turns using `pyannote.audio`, injected into transcript JSON
- **Translate** — Translates English transcript to Spanish using `argostranslate` with MarianMT neural reranking
- **TTS** — Synthesizes gender-aware dubbed audio using Edge TTS (default, free, no GPU) with Chatterbox as an optional GPU upgrade
- **Stitch** — Combines dubbed audio with original video using `ffmpeg` (direct stream copy, no re-encoding)

---

## Architecture

```
┌──────────────────────────┐
│   Frontend (Next.js)      │  :8501 — Dubbing Studio UI
└────────────┬─────────────┘
             │ HTTP (/api/* proxy)
┌────────────▼─────────────┐
│   API (FastAPI, CPU)      │  :8080 — Orchestrates pipeline
│   · Whisper base (CPU)    │         Speech-to-text (in-process)
│   · Edge TTS              │         Gender-aware neural TTS (default)
│   · pyannote diarization  │         Speaker turn detection
└──────────────────────────┘
```

| Layer | Tool | Port |
|-------|------|------|
| Frontend | Next.js + shadcn/ui Dubbing Studio | 8501 |
| API | FastAPI orchestrator (CPU) | 8080 |
| STT | Whisper base (runs inside API container, CPU) | — |
| TTS | Edge TTS (default, CPU); Chatterbox GPU (optional, nvidia profile) | 8020 |

---

## Requirements

- **Docker Desktop** (latest version)
- **Python 3.11+**
- **uv** package manager
- **Git**
- 8GB+ RAM recommended
- Internet access (for Edge TTS and yt-dlp)
- No GPU required — the pipeline runs fully on CPU by default

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/tilak30/foreign-whispers.git
cd foreign-whispers
```

### Step 2 — Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in `HF_TOKEN` (HuggingFace token required for pyannote diarization).

### Step 3 — Install Python dependencies

```bash
uv sync
```

---

## Running the App

### CPU (default — no GPU required)

```bash
docker compose --profile cpu up -d
```

Starts two containers: the FastAPI API (with Whisper base + Edge TTS running in-process) and the Next.js frontend.

### NVIDIA GPU (optional upgrade)

```bash
docker compose --profile nvidia up -d
```

Adds two GPU containers: Whisper STT (Speaches, port 8000) and Chatterbox TTS (port 8020) for higher-quality voice cloning.

### Verify everything is running

```bash
docker compose ps
curl http://localhost:8080/healthz
# → {"status":"ok"}
```

### Open the Dubbing Studio

Go to **http://localhost:8501** in your browser.

---

## TTS Engine

The system automatically selects the best available TTS engine at startup:

1. **Chatterbox GPU** — voice cloning, highest quality; requires a running GPU server at `CHATTERBOX_API_URL` (nvidia profile only)
2. **Edge TTS** *(default on CPU)* — Microsoft neural voices, free, no GPU, gender-aware:
   - Male speakers → `es-ES-AlvaroNeural`
   - Female speakers → `es-ES-ElviraNeural`
3. **Coqui Tacotron2** — fully offline fallback, no internet required

Force a specific engine:

```bash
FW_TTS_ENGINE=edge    # Microsoft neural (default)
FW_TTS_ENGINE=coqui   # Offline fallback
```

---

## Mac-Specific Notes

This project was developed and tested on **Apple M4 (16GB RAM)** running macOS.

Docker's `network_mode: host` does not work on macOS. The repo's `docker-compose.yml` already uses explicit port mappings (`8080:8080`, `8501:8501`) and sets the Next.js API URL to the Docker service name (`http://api:8080`) — so no manual changes are needed on Mac.

---

## Running the Pipeline

### Via the UI (recommended)

1. Open **http://localhost:8501**
2. Select a video from the left sidebar
3. Click **Start Pipeline**
4. Watch each stage complete: Download → Transcribe → Diarize → Translate → Synthesize → Stitch
5. Click the **Baseline** tab to watch the dubbed video

### Via curl (command line)

```bash
# P1 — Download
curl -X POST http://localhost:8080/api/download \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=GYQ5yGV_-Oc"}'

# P2 — Transcribe
curl -X POST http://localhost:8080/api/transcribe/GYQ5yGV_-Oc

# P3 — Translate
curl -X POST http://localhost:8080/api/translate/GYQ5yGV_-Oc

# P4 — TTS (with alignment)
curl -X POST "http://localhost:8080/api/tts/GYQ5yGV_-Oc?alignment=true"

# P5 — Stitch
curl -X POST http://localhost:8080/api/stitch/GYQ5yGV_-Oc
```

### Output location

```
pipeline_data/api/dubbed_videos/{video_title}.mp4
```

---

## Pipeline Stage Latencies (CPU, ~10 min video)

| Stage | Latency |
|-------|---------|
| Download | ~15s |
| Transcribe (Whisper base, CPU) | ~90s |
| Translate + Rerank | ~45s |
| TTS Synthesis (Edge TTS, 3 workers) | ~120s |
| Stitch | ~5s |
| **Total** | **~275s (~4.5 min)** |

---

## Project Structure

```
foreign-whispers/
├── api/src/
│   ├── main.py                  # App factory + lazy model loading
│   ├── core/config.py           # Pydantic settings (FW_ env prefix)
│   ├── routers/                 # Thin route handlers (download, transcribe, etc.)
│   ├── services/
│   │   ├── tts_engine.py        # Edge TTS / Coqui / Chatterbox fallback chain
│   │   └── tts_service.py       # Service wrapper
│   └── schemas/                 # Pydantic request/response models
├── foreign_whispers/
│   ├── reranking.py             # MarianMT neural translation reranking
│   ├── alignment.py             # Ridge regression duration prediction + DP alignment
│   └── diarization.py          # pyannote speaker diarization
├── frontend/                    # Next.js + shadcn/ui Dubbing Studio
├── notebooks/                   # Jupyter notebooks for each pipeline stage
├── pipeline_data/               # Generated artifacts (videos, audio, etc.)
├── tests/                       # Test suite (23/23 targeted tests passing)
├── docker-compose.yml           # Profiles: cpu, nvidia
├── Dockerfile                   # Multi-stage: cpu and gpu targets
└── pyproject.toml               # Python dependencies
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/download` | Download YouTube video + captions |
| POST | `/api/transcribe/{id}` | Whisper speech-to-text |
| POST | `/api/translate/{id}` | EN→ES translation + reranking |
| POST | `/api/tts/{id}` | Time-aligned TTS synthesis |
| POST | `/api/stitch/{id}` | Audio remux (ffmpeg -c:v copy) |
| GET | `/api/video/{id}` | Stream dubbed video |
| GET | `/api/captions/{id}` | Translated WebVTT captions |
| GET | `/healthz` | Health check |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FW_TTS_ENGINE` | *(auto)* | Force `edge`, `coqui`, or `chatterbox` |
| `FW_EDGE_VOICE_MALE` | `es-ES-AlvaroNeural` | Edge TTS male voice |
| `FW_EDGE_VOICE_FEMALE` | `es-ES-ElviraNeural` | Edge TTS female voice |
| `FW_ALIGNMENT` | `on` | Set `off` to disable ridge-regression alignment |
| `FW_TTS_WORKERS` | `3` | Concurrent TTS synthesis threads |
| `FW_WHISPER_MODEL` | `base` | Whisper model size (base runs on CPU) |
| `CHATTERBOX_API_URL` | `http://chatterbox-gpu:8020` | Chatterbox server (nvidia profile only) |
| `HF_TOKEN` | — | HuggingFace token for pyannote diarization |

---

## Development

### Editing without rebuilding

`foreign_whispers/` and `api/` are bind-mounted into the container. Edit on host, then:

```bash
docker compose --profile cpu restart api
```

### Running tests

```bash
uv run pytest tests/ -q -k "not requires_pyannote and not requires_silero"
```

---

## Known Limitations

- **Spanish only** — Translation target is hardcoded to Spanish. Other language pairs require additional argostranslate models and TTS voice configuration.
- **Alignment bounds** — Time-stretching is clamped to [0.75×, 1.25×] to preserve naturalness. Segments outside this range are padded with silence.
- **Diarization requires HF token** — pyannote.audio requires a HuggingFace token and model access agreement on the pyannote/speaker-diarization model page.
- **Edge TTS requires internet** — The default TTS engine makes outbound requests to Microsoft's neural voice API. Use `FW_TTS_ENGINE=coqui` for a fully offline setup.
- **YouTube rate limits** — Heavy usage may trigger rate limiting. Add valid YouTube cookies to `cookies.txt` to mitigate.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Frontend shows blank page | Verify `API_URL` build arg matches service name in `docker-compose.yml` |
| Diarization fails with 401 | Set `HF_TOKEN` in `.env` and accept pyannote model agreements on HuggingFace |
| Edge TTS produces no audio | Check outbound internet access from the API container |
| `cookies.txt is a directory` error | Run `rm -rf cookies.txt && echo '# Netscape HTTP Cookie File' > cookies.txt` |
| Download fails with Internal Server Error | Check `cookies.txt` exists and has the Netscape header |
| Whisper transcription is slow | Expected on CPU with `base` model (~90s for a 10-min video); this is normal |
