# Foreign Whispers — Technical Report

**NYU Spring 2026 | NLP Project**  
**Author:** Tilak Bhansali | **NetID:** `tb3525`

---

## 1. Overview

Foreign Whispers is an end-to-end, open-source video dubbing pipeline. Given a YouTube URL, it automatically downloads the video, transcribes and diarizes the audio, translates each segment into Spanish, synthesizes a gender-aware dubbed audio track with temporal alignment, and stitches the result back into the original video — all accessible from a browser-based Dubbing Studio UI.

The central challenge is not simply translating text — it is ensuring synthesized speech *fits* the original timing. A Spanish translation of an English sentence is typically 20–30% longer in syllable count. Without compensation, dubbed audio continuously drifts, producing a video where speech and lip movement diverge catastrophically. This report describes the architecture, innovations, implementation challenges, and results.

---

## 2. System Architecture

The application uses a distributed, container-based architecture with clear separation of concerns:

```
Docker Compose
├── frontend  (Next.js + shadcn/ui)   :8501
├── api       (FastAPI, CPU)           :8080
├── stt       (Whisper, GPU)           :8000   [nvidia profile]
└── tts       (Chatterbox, GPU)        :8020   [nvidia profile]
```

### 2.1 Frontend — Dubbing Studio

Built with **Next.js 14** and **shadcn/ui**, the frontend presents a pipeline tracker visualizing each stage (Download → Transcribe → Diarize → Translate → Synthesize → Stitch). Each stage button triggers a REST call to the API and displays progress, errors, and results inline. The video player renders the dubbed output with synchronized WebVTT captions through the browser's native `<track>` element — no subtitle burn-in required.

### 2.2 FastAPI Backend

The API container is CPU-only. It orchestrates the pipeline, delegates GPU-heavy work to STT/TTS containers via HTTP, and manages intermediate file storage under `pipeline_data/api/`. All source directories are bind-mounted into the container so edits are visible without rebuilds.

The backend follows a layered architecture:
- **Routers** (`api/src/routers/`): thin HTTP handlers, one per pipeline stage
- **Services** (`api/src/services/`): business logic, HTTP-agnostic
- **Schemas** (`api/src/schemas/`): Pydantic request/response models

### 2.3 Data Flow

```
YouTube URL
  → yt-dlp download (MP4 + rolling captions)
  → Rolling caption deduplication (sentence-level re-segmentation)
  → pyannote diarization (speaker labels injected)
  → argostranslate translation (EN→ES JSON)
  → duration-aware reranking (shorter candidate selection)
  → TTS synthesis (per-segment WAV, time-stretched, gender-aware)
  → ffmpeg remux (-c:v copy, no re-encode)
  → Dubbed MP4 + aligned WebVTT captions
```

---

## 3. Key Technical Innovations

### 3.1 Rolling Caption Deduplication (Core Fix)

**Problem discovered:** YouTube auto-captions are delivered as overlapping rolling windows — typically 5–7 words per caption, sliding forward every ~2 seconds. The original pipeline fed all 170 overlapping windows directly to TTS as independent segments. When assembling audio, each segment placed audio at its start time, but the next segment fired ~2s later and jumped the cursor forward, cutting off the previous segment. This left **~85% of the dubbed video as silence** — only 63 seconds of speech in a 411-second video.

**Solution** (`api/src/routers/transcribe.py`, `_youtube_captions_to_segments()`):

1. **Deduplication**: Extract only the *new* words each caption window introduces by finding the longest suffix of the previous window that matches the current window's prefix. This reconstructs the true word stream without repetition.
2. **Per-word timestamp interpolation**: Each unique word is assigned an approximate timestamp by interpolating its position within the caption window that introduced it.
3. **Monotonicity enforcement**: A second pass ensures timestamps never decrease (rounding errors from interpolation could cause occasional inversions).
4. **Re-segmentation**: Words are regrouped into non-overlapping chunks of ≤15 words, splitting at sentence boundaries (`.`, `!`, `?`) when possible. Each chunk gets a real, unique time window for TTS.
5. **Overlap clamp**: Final pass ensures no segment starts before the previous one ends.

Result: 108 clean, non-overlapping segments (vs 170 overlapping ones), each with 2–6 seconds of window for TTS audio. This is the single highest-impact fix in the project.

### 3.2 Neural Translation Reranking

Rather than truncating over-long translations, a **dual-backend reranking pipeline** (`foreign_whispers/reranking.py`) generates alternatives:

1. **ArgosTranslate** generates a baseline translation
2. **MarianMT** (`Helsinki-NLP/opus-mt-en-es`) generates 5 candidates via beam search
3. Each candidate is scored by predicted TTS duration (syllable-rate heuristic)
4. Gate is on predicted duration, not character count — accepts same-length translations with fewer syllables that would previously be incorrectly rejected
5. `truncate_for_duration_budget()` provides a word-level fallback

### 3.3 Ridge Regression Duration Prediction

To predict TTS segment duration *before* synthesis (enabling smarter scheduling), a Ridge regression model is trained on `(text_features, actual_tts_duration)` pairs from Chatterbox synthesis runs. Features: character count, syllable count (Spanish-aware), word count. Stored as `tts_duration_ridge.json`, loaded in `foreign_whispers/alignment.py`. Achieves ~85ms median absolute error vs ~320ms for the naive syllable-rate heuristic.

### 3.4 Global Alignment with Dynamic Programming

Rather than per-segment greedy processing, `global_align_dp` in `alignment.py`:

1. Computes per-segment stretch factors based on predicted TTS duration vs available time
2. Tags each segment with an `AlignAction` (`MILD_STRETCH`, `STRETCH`, `REQUEST_SHORTER`, `PAD`)
3. Uses DP beam search to minimize maximum stretch factor globally, distributing slack from easy segments to hard ones
4. Uses VAD-detected silence regions to enable `GAP_SHIFT` — borrowing time from natural pauses before triggering reranking

The assembly loop uses `aligned_seg.scheduled_start` (from DP output) rather than raw caption timestamps, so DP decisions actually appear in the output audio.

### 3.5 Gender-Aware Speaker Voice Mapping

Multi-speaker videos require distinct voices for each speaker:

1. **pyannote.audio** detects speaker turns (`SPEAKER_00`, `SPEAKER_01`, etc.)
2. `assign_speakers()` injects speaker labels into transcript segments
3. Translation service preserves all fields (deep copy), so speaker labels carry through to the TTS call
4. Gender inference from speaker label: explicit suffixes (`_F`, `FEM`) → direct; pyannote labels → even-index = male, odd-index = female
5. **Edge TTS** uses `es-ES-AlvaroNeural` (male) or `es-ES-ElviraNeural` (female) accordingly
6. `per_speaker_voices=True` is now always set in the TTS endpoint (previously only activated when an explicit `voice_cloning` param was passed, silently disabling gender differentiation)

### 3.6 Aligned WebVTT Caption Generation

The `GET /api/captions/{id}` endpoint previously cached a VTT file on first access and served it stale forever. If TTS hadn't run yet when captions were first requested, the VTT used original caption timestamps (rolling window times) rather than the actual audio placement times.

Fixed behaviour:
- Always regenerates from the latest `.align.json` (ground truth for TTS placement)
- Uses `scheduled_start_s` + `raw_duration_s` per segment for accurate display timing
- Index-based lookup (not position-based), so segment counts don't need to match between VTT and align data
- Falls back gracefully to translation JSON timestamps when no align data exists yet

### 3.7 TTS Engine Fallback Chain

```
Chatterbox GPU (best quality, voice cloning)
  ↓ not reachable?
Edge TTS — Microsoft neural (free, gender-aware, excellent quality)
  ↓ not installed?
Coqui Tacotron2 (offline, air-gapped, lower quality)
```

### 3.8 MP3 Predelay Stripping

Edge TTS produces MP3 output transcoded to WAV via pydub. pydub's `from_mp3()` preserves ~576-sample encoder predelay — typically 30–50ms of near-silence at the start of each segment. Over 60+ segments this accumulates to ~2–3s of audible late-start drift. Fixed by stripping leading frames below –50dBFS (up to 120ms max) from each segment before time-stretching.

---

## 4. Implementation Challenges

### 4.1 Rolling Caption Format (Biggest Surprise)

The rolling caption deduplication problem was invisible from the code — the pipeline *appeared* to run correctly (all 170 segments synthesized, no errors) but produced 85% silence. The bug only became apparent by extracting the dubbed audio and measuring speech coverage with librosa. This was by far the hardest bug to find.

### 4.2 Async Edge TTS in a Sync Pipeline

`edge-tts` is fully async but TTS synthesis runs from a `ThreadPoolExecutor`. Calling `asyncio.run()` from a thread that may already have a running event loop raises `RuntimeError: This event loop is already running`. Fixed by always dispatching onto a fresh `ThreadPoolExecutor` thread where `asyncio.run()` is safe.

### 4.3 Spanish Syllable Expansion

English is stress-timed with many reduced vowels. Spanish is syllable-timed and retains all vowel sounds. A 3-second English sentence takes ~3.6–4.2 seconds in Spanish. Without proactive duration management, drift accumulates — a 10-minute video can end up 2+ minutes longer in Spanish. Multi-stage mitigation: predict duration before synthesis → select shorter translations → time-stretch within quality bounds → pad/trim to target window.

### 4.4 SPEED_MAX Too Conservative

Original `SPEED_MAX = 1.25` caused widespread hard speech truncation: with Spanish running ~25% longer than English, most segments hit the cap and were trimmed (speech cut off mid-sentence). Raised to `1.35×` — still within pyrubberband's quality range — to accommodate normal-length Spanish segments without clipping.

### 4.5 Stale Caption Cache

The caption endpoint cached the VTT on first access. If called before TTS ran, it permanently cached captions with wrong timestamps. Fixed to always regenerate from the latest `.align.json`.

---

## 5. Results

### 5.1 Audio Quality

| Engine | Quality | Gender-aware | Latency/segment | Cost |
|---|---|---|---|---|
| Chatterbox GPU (Colab T4) | ⭐⭐⭐⭐⭐ | ✅ voice cloning | ~2–4s | Free (Colab) |
| **Edge TTS (default)** | ⭐⭐⭐⭐ | ✅ AlvaroNeural / ElviraNeural | ~0.5–1s | Free |
| Coqui Tacotron2 (CPU) | ⭐⭐ | ❌ | ~5–15s | Free |

### 5.2 Before vs. After Key Fix

| Metric | Before (rolling captions fed directly) | After (dedup + re-segment) |
|---|---|---|
| Speech coverage | 63s / 411s **(15%)** | ~350s / 411s **(~85%)** |
| Silence gaps | 30 gaps, some >50s long | Natural pauses only |
| Segment count | 170 overlapping | 108 non-overlapping |
| Median segment window | ~2.2s | ~3.8s |

### 5.3 Alignment Metrics (Strait of Hormuz test video)

- Mean absolute duration error: ~180ms (vs ~640ms without alignment)
- Segments requiring reranking: ~18%
- Segments using silence padding: ~22%
- Severe stretch (>25%): <5%

### 5.4 Pipeline Latencies (CPU mode, ~7min video)

| Stage | Latency |
|---|---|
| Download | ~15s |
| Transcribe (YouTube captions) | ~2s |
| Translate + Rerank | ~45s |
| TTS Synthesis (Edge TTS, 3 workers) | ~120s |
| Stitch | ~5s |
| **Total** | **~190s (~3 min)** |

---

## 6. Limitations & Future Work

**Prosody Transfer:** Time-stretching preserves pitch only within [0.75, 1.35×]. A neural vocoder (VITS) that synthesizes at a target duration directly would eliminate post-hoc stretching artifacts.

**Lip-Sync:** No attempt at lip-sync beyond temporal alignment. Wav2Lip could animate speaker mouths to match dubbed audio.

**Caption Word-Level Timing:** Current re-segmentation distributes word timestamps by linear interpolation within caption windows. Forced alignment (CTC, Montreal Forced Aligner) would give exact per-word timestamps for tighter caption sync.

**Multi-Language:** Pipeline is hardcoded to EN→ES. Generalizing is architecturally straightforward — swap argostranslate models and TTS voice selections.

**Streaming:** All TTS segments synthesized before assembly. A streaming architecture would reduce perceived latency.

---

## 7. Conclusion

Foreign Whispers successfully implements a fully automated, browser-operated video dubbing pipeline on commodity hardware without proprietary APIs. Key contributions:

1. **Rolling caption deduplication** — the highest-impact fix: converts 170 overlapping 2s windows into 108 real sentence-level segments, raising speech coverage from 15% to ~85%
2. **Aligned WebVTT captions** — always generated from actual TTS audio timeline, never stale
3. **Gender-aware TTS always enabled** — `per_speaker_voices=True` is now unconditional
4. **MP3 predelay stripping** — eliminates accumulated per-segment onset drift
5. **Neural reranking on duration** — duration-gate (not character-count) accepts more valid shorter translations
6. **DP global alignment** — minimizes drift accumulation; wired to use `scheduled_start` in assembly
7. **Fault-tolerant engine fallback chain** — works on any machine, no GPU required

The result is a system that takes a 60 Minutes interview in English and produces a watchable Spanish dub in ~3 minutes on a MacBook.