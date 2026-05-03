# Foreign Whispers - Architectural and Technical Report

## Overview
Foreign Whispers is an open-source, end-to-end video dubbing pipeline. Given a YouTube video URL, it downloads the content, extracts the audio and captions, transcribes the speech, translates the text, synthesizes a dubbed audio track, and finally stitches the audio back into the original video with synchronized subtitles.

## System Architecture
The application follows a distributed, multi-container architecture orchestrated via Docker Compose:

1. **FastAPI Backend (CPU)**: A RESTful orchestrator running in a lightweight CPU container. It exposes endpoints (`/api/download`, `/api/transcribe`, `/api/diarize`, `/api/translate`, `/api/tts`, `/api/stitch`) that manage the pipeline state and delegate intensive tasks to the GPU containers.
2. **Next.js Frontend (CPU)**: A React-based UI providing an interactive Dubbing Studio. It acts as the pipeline control center, visualizing each stage and allowing playback of the final dubbed output.
3. **Whisper STT (GPU)**: A dedicated container running the `faster-whisper-medium` model, handling speech-to-text operations rapidly.
4. **Chatterbox TTS (GPU)**: A specialized TTS container that synthesizes high-quality audio and supports voice cloning through reference WAV files.

## Technical Details

### Stage 1: Download & Extraction
Using `yt-dlp`, the backend fetches the highest quality MP4 stream along with YouTube’s native closed captions. The separation of audio extraction happens via `ffmpeg`, preparing a PCM WAV file required by the subsequent machine learning models.

### Stage 2: Transcription & Diarization
The raw audio is sent to the Whisper API. Since Whisper provides timestamps but lacks speaker identification, we integrated **pyannote.audio** for speaker diarization. By detecting speaker turns, we map individual segments to unique speakers via an intersection-over-union temporal overlap function (`assign_speakers`). The resulting JSON transcript contains both high-accuracy text and precise speaker labels.

### Stage 3: Duration-Aware Translation
Translation is performed locally using the `argostranslate` package. Because target languages (like Spanish) often require more syllables to convey the same meaning as English, a naive translation would exceed the original audio window. We implemented a rule-based re-ranking system (`get_shorter_translations`) that automatically truncates filler words and replaces long idioms with shorter synonyms when segments are tagged with `REQUEST_SHORTER`.

### Stage 4: Voice Cloning & Time-Aligned TTS
For multi-speaker videos, a fallback chain voice resolution strategy (`resolve_speaker_wav`) maps the detected speaker IDs to reference audio clips located in `pipeline_data/speakers`. The TTS engine (`Chatterbox` or local `Coqui`) synthesizes the audio segments.
To fit the new audio exactly into the source video without drift:
- The system employs a greedy alignment algorithm predicting duration using a custom syllable-counting heuristic.
- Short bursts of audio are time-stretched using `pyrubberband` (within a safe 10-40% margin) to preserve audio quality.
- Overflows are shifted into adjacent silence gaps.

### Stage 5: Video Stitching
The final step uses `ffmpeg`'s `-c:v copy` operation to mux the generated time-aligned audio track back onto the original video stream without re-encoding the video track. We convert the synchronized translation segments into WebVTT format for browser playback.

## Limitations & Future Work
- The global alignment strategy uses a greedy left-to-right pass. Future improvements could adopt Dynamic Programming or ILP to globally minimize the maximum stretch factor across all segments.
- TTS generation latency could be reduced by deploying an aggressive streaming chunking approach, streaming audio bytes back to the frontend immediately.

## Conclusion
Foreign Whispers successfully democratizes the video dubbing process, running entirely on self-hosted infrastructure without relying on proprietary APIs. The architectural separation of compute-heavy GPU containers and a lightweight orchestrator ensures stability and seamless orchestration.
