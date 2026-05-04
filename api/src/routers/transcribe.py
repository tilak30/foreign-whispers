"""POST /api/transcribe/{video_id} — Whisper transcription with rolling-caption dedup.

Key fix: YouTube auto-captions are delivered as overlapping rolling windows
(each ~5-7 words, sliding every ~2 seconds).  Feeding them directly to TTS
as independent segments means each segment has only ~2s before the next fires
and cuts it off — leaving ~85% of the audio as silence.

_youtube_captions_to_segments() deduplicates the windows into a proper
word stream, then re-segments at sentence boundaries (~15 words max) so every
TTS segment has a real non-overlapping time window.
"""

import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.main import get_whisper_model
from api.src.schemas.transcribe import TranscribeResponse, TranscribeSegment
from api.src.services.transcription_service import TranscriptionService

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


@router.post("/transcribe/{video_id}", response_model=TranscribeResponse)
async def transcribe_endpoint(
    video_id: str,
    request: Request,
    use_youtube_captions: bool = Query(True, description="Use YouTube captions when available, skipping Whisper"),
):
    """Run Whisper transcription on a downloaded video.

    When use_youtube_captions is True (default), YouTube captions are used if
    available, skipping Whisper entirely. When False, Whisper always runs.
    """
    videos_dir = settings.videos_dir
    transcriptions_dir = settings.transcriptions_dir
    transcriptions_dir.mkdir(parents=True, exist_ok=True)

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    transcript_path = transcriptions_dir / f"{title}.json"

    # Return cached result if it exists
    if transcript_path.exists() and use_youtube_captions:
        data = json.loads(transcript_path.read_text())
        return TranscribeResponse(
            video_id=video_id,
            language=data.get("language", "en"),
            text=data.get("text", ""),
            segments=data.get("segments", []),
            skipped=True,
        )

    # Prefer YouTube captions (accurate timestamps, free, no GPU)
    if use_youtube_captions:
        yt_caption_path = settings.youtube_captions_dir / f"{title}.txt"
        if yt_caption_path.exists():
            result = _youtube_captions_to_segments(yt_caption_path)
            transcript_path.write_text(json.dumps(result))
            return TranscribeResponse(
                video_id=video_id,
                language=result["language"],
                text=result["text"],
                segments=result["segments"],
                skipped=True,
            )

    # Run Whisper STT
    svc = TranscriptionService(
        ui_dir=settings.data_dir,
        whisper_model=get_whisper_model(request.app),
    )
    video_path = videos_dir / f"{title}.mp4"
    result = svc.transcribe(str(video_path))

    transcript_path.write_text(json.dumps(result))

    return TranscribeResponse(
        video_id=video_id,
        language=result.get("language", "en"),
        text=result.get("text", ""),
        segments=result.get("segments", []),
    )