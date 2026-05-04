"""POST /api/transcribe/{video_id} — Whisper transcription with rolling-caption dedup.

Key fix: YouTube auto-captions are delivered as overlapping rolling windows
(each ~5-7 words, sliding every ~2 seconds).  Feeding them directly to TTS
as independent segments means each segment has only ~2s before the next fires
and cuts it off — leaving ~85% of the audio as silence.

_youtube_captions_to_segments() deduplicates the windows into a proper
word stream, then re-segments at sentence boundaries (~15 words max) so every
TTS segment has a real non-overlapping time window.

Gender fix (2025-05): diarize_audio() + assign_speakers() are now called after
every transcription path.  Speaker labels written into each segment dict allow
text_file_to_speech() in tts_engine.py to pick the correct Edge TTS voice
(es-ES-AlvaroNeural for male, es-ES-ElviraNeural for female) per segment.
Without this, seg.get("speaker") was always None → gender defaulted to male
for every segment in the dubbed audio.
"""

import json
import logging
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.main import get_whisper_model
from api.src.schemas.transcribe import TranscribeResponse, TranscribeSegment
from api.src.services.transcription_service import TranscriptionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def _extract_new_words(prev_words: list[str], curr_words: list[str]) -> list[str]:
    """Return words in curr_words that are not a repeat of the tail of prev_words.

    YouTube rolling captions slide forward — each window shares several words
    with the previous one.  This finds the longest suffix of prev_words that
    matches a prefix of curr_words and returns only the novel remainder.
    """
    for overlap in range(min(len(prev_words), len(curr_words), 8), 0, -1):
        if (
            [w.lower().rstrip(".,!?") for w in prev_words[-overlap:]]
            == [w.lower().rstrip(".,!?") for w in curr_words[:overlap]]
        ):
            return curr_words[overlap:]
    return curr_words


def _youtube_captions_to_segments(
    caption_path: pathlib.Path,
    max_segment_words: int = 15,
) -> dict:
    """Convert YouTube rolling captions to non-overlapping Whisper-compatible segments.

    YouTube auto-captions are a rolling 5-7-word window sliding every ~2s.
    Each window OVERLAPS the next, so if fed directly to TTS as individual
    segments each one gets only ~2s before the cursor jumps and cuts it off,
    leaving ~85% of the video as silence.

    This function:
    1. Reconstructs the true word sequence by stripping the repeated prefix
       from each successive caption window.
    2. Tags each unique word with an approximate timestamp (interpolated
       from the caption window it was first introduced in).
    3. Enforces monotonically increasing timestamps.
    4. Re-segments the words into non-overlapping TTS-friendly chunks of
       <= max_segment_words words, splitting at sentence boundaries when possible.
    5. Final pass clamps any segment that still starts before the previous ends.
    """
    raw_caps: list[dict] = []
    for line in caption_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seg = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = seg.get("text", "").strip()
        start = float(seg.get("start", 0))
        duration = float(seg.get("duration", 0))
        if text and duration > 0:
            raw_caps.append({"text": text, "start": start, "duration": duration})

    if not raw_caps:
        return {"language": "en", "text": "", "segments": []}

    # ── Step 1: reconstruct unique word stream with per-word timestamps ──────
    words_with_times: list[tuple[str, float]] = []
    prev_words: list[str] = []

    for cap in raw_caps:
        curr_words = cap["text"].split()
        new_words = _extract_new_words(prev_words, curr_words)
        n_total = len(curr_words)
        n_overlap = n_total - len(new_words)
        for i, word in enumerate(new_words):
            # Place each new word proportionally within the caption window
            frac = (n_overlap + i) / max(n_total, 1)
            t = cap["start"] + cap["duration"] * frac
            words_with_times.append((word, t))
        prev_words = curr_words

    # ── Step 2: enforce monotonically increasing timestamps ──────────────────
    # Interpolation can produce a word that timestamps slightly before the
    # previous word if the rolling window shrinks (short caption after long one).
    for i in range(1, len(words_with_times)):
        word, t = words_with_times[i]
        prev_t = words_with_times[i - 1][1]
        if t < prev_t:
            words_with_times[i] = (word, prev_t + 0.05)

    full_text = " ".join(w for w, _ in words_with_times)

    # ── Step 3: re-segment into non-overlapping TTS-ready chunks ────────────
    segments: list[dict] = []
    buf_words: list[str] = []
    buf_times: list[float] = []

    def flush_buf() -> None:
        if not buf_words:
            return
        seg_text = " ".join(buf_words)
        seg_start = buf_times[0]
        seg_end = buf_times[-1] + 0.3  # small trailing buffer
        if seg_end <= seg_start:
            seg_end = seg_start + 0.5
        segments.append({
            "id": len(segments),
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg_text,
        })

    for word, t in words_with_times:
        buf_words.append(word)
        buf_times.append(t)

        is_sentence_end = word.rstrip().endswith((".", "!", "?"))
        is_at_limit = len(buf_words) >= max_segment_words

        if is_sentence_end or is_at_limit:
            flush_buf()
            buf_words, buf_times = [], []

    flush_buf()  # trailing words

    # ── Step 4: guarantee no segment overlaps the one before it ─────────────
    for i in range(1, len(segments)):
        prev_end = segments[i - 1]["end"]
        if segments[i]["start"] < prev_end:
            gap = segments[i]["end"] - segments[i]["start"]
            segments[i]["start"] = prev_end
            segments[i]["end"] = prev_end + max(gap, 0.3)

    return {
        "language": "en",
        "text": full_text,
        "segments": segments,
    }


def _inject_speaker_labels(result: dict, video_path: pathlib.Path) -> dict:
    """Run pyannote diarization on the source audio and inject speaker labels.

    Extracts a WAV from the video, runs diarize_audio(), then calls
    assign_speakers() to stamp each segment with its speaker label.

    Falls back gracefully:
    - If pyannote is not installed or HF token is absent → uses
      synthetic_diar_segments_from_transcript() to assign alternating
      SPEAKER_00 / SPEAKER_01 labels based on segment index parity.
      This is enough for the gender-parity heuristic in tts_engine.py
      to produce alternating male/female voices, which is far better
      than every segment getting the male default.
    - If ffmpeg extraction fails → returns result unchanged (no speaker field).

    The returned dict has the same structure as the input but each segment
    dict gains a "speaker" key, e.g. "SPEAKER_00" or "SPEAKER_01".
    """
    from foreign_whispers.diarization import (
        assign_speakers,
        diarize_audio,
        synthetic_diar_segments_from_transcript,
    )

    segments = result.get("segments", [])
    if not segments:
        return result

    hf_token: str | None = settings.hf_token if hasattr(settings, "hf_token") else None

    diar_intervals: list[dict] = []

    # ── Try real pyannote diarization first ──────────────────────────────────
    if hf_token:
        # Extract a mono WAV next to the video for pyannote (it needs a plain
        # audio file, not an MP4).  Use ffmpeg if available; skip silently if not.
        audio_wav = video_path.with_suffix(".diar.wav")
        try:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-ac", "1", "-ar", "16000",
                    "-vn", str(audio_wav),
                ],
                check=True,
                capture_output=True,
            )
            diar_intervals = diarize_audio(str(audio_wav), hf_token=hf_token)
        except Exception as exc:
            logger.warning("[transcribe] diarize_audio failed (%s), falling back to synthetic labels", exc)
        finally:
            # Clean up the temporary WAV regardless of success/failure
            if audio_wav.exists():
                audio_wav.unlink(missing_ok=True)
    else:
        logger.info(
            "[transcribe] No HF_TOKEN configured — using synthetic speaker labels "
            "(alternating SPEAKER_00/SPEAKER_01 by segment index). "
            "Set HF_TOKEN in .env and accept pyannote/speaker-diarization-3.1 on "
            "HuggingFace for real diarization."
        )

    # ── Synthetic fallback: alternating labels by segment index ─────────────
    # Even when pyannote runs, diar_intervals may be empty (e.g. single speaker
    # video, diarization failure).  In that case assign_speakers() already
    # defaults all segments to SPEAKER_00, which means male voice everywhere —
    # same as before.  The synthetic fallback at least gives two distinct
    # speakers so the gender parity heuristic fires on odd-indexed segments.
    if not diar_intervals:
        # k=2 → alternates SPEAKER_00 (even, male) / SPEAKER_01 (odd, female)
        diar_intervals = synthetic_diar_segments_from_transcript(segments, k=2)

    result_with_speakers = dict(result)
    result_with_speakers["segments"] = assign_speakers(segments, diar_intervals)
    return result_with_speakers


@router.post("/transcribe/{video_id}", response_model=TranscribeResponse)
async def transcribe_endpoint(
    video_id: str,
    request: Request,
    use_youtube_captions: bool = Query(True, description="Use YouTube captions when available, skipping Whisper"),
    diarize: bool = Query(True, description="Inject speaker labels via pyannote (or synthetic fallback)"),
):
    """Run Whisper transcription on a downloaded video.

    When use_youtube_captions is True (default), YouTube captions are used if
    available, skipping Whisper entirely. When False, Whisper always runs.

    When diarize is True (default), speaker labels are injected into segments
    via pyannote.audio (requires HF_TOKEN) or a synthetic alternating fallback.
    Speaker labels drive gender-aware TTS voice selection downstream.
    """
    videos_dir = settings.videos_dir
    transcriptions_dir = settings.transcriptions_dir
    transcriptions_dir.mkdir(parents=True, exist_ok=True)

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    transcript_path = transcriptions_dir / f"{title}.json"

    # Return cached result if it exists.
    # NOTE: we do NOT short-circuit the cache when diarize=True if the cached
    # transcript already has speaker labels — check for that first.
    if transcript_path.exists() and use_youtube_captions:
        data = json.loads(transcript_path.read_text())
        segs = data.get("segments", [])
        already_diarized = segs and "speaker" in segs[0]
        if already_diarized or not diarize:
            # Cache hit and speaker labels already present (or not wanted).
            return TranscribeResponse(
                video_id=video_id,
                language=data.get("language", "en"),
                text=data.get("text", ""),
                segments=segs,
                skipped=True,
            )
        # Cache hit but missing speaker labels — fall through to re-diarize and
        # overwrite.  This handles the case where the transcript was cached
        # before this fix was deployed.
        result = data

    else:
        result = None

    video_path = videos_dir / f"{title}.mp4"

    if result is None:
        # Prefer YouTube captions (accurate timestamps, free, no GPU)
        if use_youtube_captions:
            yt_caption_path = settings.youtube_captions_dir / f"{title}.txt"
            if yt_caption_path.exists():
                result = _youtube_captions_to_segments(yt_caption_path)

        if result is None:
            # Run Whisper STT
            svc = TranscriptionService(
                ui_dir=settings.data_dir,
                whisper_model=get_whisper_model(request.app),
            )
            result = svc.transcribe(str(video_path))

    # ── Inject speaker labels (diarization) ─────────────────────────────────
    # This is the step that was missing before.  Without it, seg.get("speaker")
    # is always None in tts_engine.py, so gender is never inferred and every
    # segment uses the male Edge TTS voice.
    if diarize:
        result = _inject_speaker_labels(result, video_path)

    transcript_path.write_text(json.dumps(result))

    return TranscribeResponse(
        video_id=video_id,
        language=result.get("language", "en"),
        text=result.get("text", ""),
        segments=result.get("segments", []),
        skipped=False,
    )