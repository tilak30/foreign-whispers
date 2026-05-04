"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


def _stamp_speakers_into_json(json_path, diar_segments: list[dict]) -> None:
    """Re-read a segment JSON, inject speaker labels, write it back.

    Safe no-op if the file doesn't exist or can't be parsed.
    """
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
        labeled = assign_speakers(data.get("segments", []), diar_segments)
        data["segments"] = labeled
        json_path.write_text(json.dumps(data))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "[diarize] failed to stamp speakers into %s: %s", json_path, exc
        )


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str, force: bool = False):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Stamp speaker labels into BOTH transcription AND translation JSONs
    4. Cache and return speaker segments

    Stamping into both files ensures the TTS step always sees speaker labels
    regardless of which path _merge_speaker_labels searches first.
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result — but also re-stamp translation JSON in case it was
    # written after diarize ran (e.g. translate ran before diarize was called).
    # Pass force=true to re-run pyannote even if cache exists (e.g. after
    # adding HF token for the first time, or to pick up a new model).
    if diar_path.exists() and not force:
        data = json.loads(diar_path.read_text())
        diar_segments = data.get("segments", [])

        # Re-stamp translation JSON (idempotent — assign_speakers is a pure merge)
        trans_path = settings.translations_dir / f"{title}.json"
        _stamp_speakers_into_json(trans_path, diar_segments)

        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=diar_segments,
            skipped=True,
        )

    video_path = settings.videos_dir / f"{title}.mp4"
    audio_path = diar_dir / f"{title}.wav"
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-y", str(audio_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    diar_segments = _alignment_service.diarize(str(audio_path))
    speakers = sorted(list(set(s["speaker"] for s in diar_segments)))

    result = {"speakers": speakers, "segments": diar_segments}
    diar_path.write_text(json.dumps(result))

    # ── Stamp speaker labels into transcription JSON ─────────────────────────
    # Try whisper/ subdirectory first, then flat fallback (matches original logic).
    transcript_path = settings.transcriptions_dir / "whisper" / f"{title}.json"
    if not transcript_path.exists():
        transcript_path = settings.transcriptions_dir / f"{title}.json"
    _stamp_speakers_into_json(transcript_path, diar_segments)

    # ── Stamp speaker labels into translation JSON ───────────────────────────
    # This is the key addition: the translate router reads from translations/
    # and writes new segment dicts without the speaker field.  Stamping here
    # means the TTS step will find speaker labels directly in the file it reads,
    # even if _merge_speaker_labels in tts_engine.py fails to locate the
    # transcription JSON.
    trans_path = settings.translations_dir / f"{title}.json"
    _stamp_speakers_into_json(trans_path, diar_segments)

    return DiarizeResponse(video_id=video_id, speakers=speakers, segments=diar_segments)