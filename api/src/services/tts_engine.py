import asyncio
import logging as _logging
import os
import pathlib
import json
import glob
import tempfile

import requests
import librosa
import soundfile as sf
import pyrubberband
from pydub import AudioSegment

# ── Chatterbox API configuration ─────────────────────────────────────
CHATTERBOX_API_URL = os.getenv("CHATTERBOX_API_URL", "http://localhost:8020")
# Path to the default speaker reference WAV, relative to pipeline_data/speakers/
CHATTERBOX_SPEAKER_WAV = os.getenv("CHATTERBOX_SPEAKER_WAV", "")
_CH_HTTP_CONNECT_S = float(os.getenv("FW_CHATTERBOX_HTTP_CONNECT_TIMEOUT", "10"))
_CH_HTTP_READ_S = float(os.getenv("FW_CHATTERBOX_HTTP_READ_TIMEOUT", "120"))
_TTS_CHATTERBOX_ERROR_HINTS_PRINTED: set[str] = set()


def _log_tts_issue_once(kind: str, msg: str) -> None:
    """Print each distinct *kind* of TTS/Chatterbox problem at most once (avoid log spam per segment)."""
    if kind in _TTS_CHATTERBOX_ERROR_HINTS_PRINTED:
        return
    _TTS_CHATTERBOX_ERROR_HINTS_PRINTED.add(kind)
    print(f"[tts] {msg}")

# Set FW_ALIGNMENT=off to use the pre-alignment baseline (legacy unclamped stretch).
# Default is "on" (new clamped path). Useful for A/B comparisons.
_ALIGNMENT_ENABLED = os.getenv("FW_ALIGNMENT", "on").lower() != "off"

SPEED_MIN = 0.75
# 1.35× is the upper bound for pyrubberband quality — beyond this, artifacts
# become perceptible. The previous 1.25 was too conservative: Spanish typically
# runs 20–30% longer than English, so many segments were getting hard-trimmed
# (speech cut off) rather than being slightly sped up.
# If a segment still overflows at 1.35×, REQUEST_SHORTER in alignment will have
# already tried a shorter translation — the trim is the absolute last resort.
SPEED_MAX = 1.35
# When TTS audio is less than this fraction of the target window, skip
# time-stretching entirely — play at natural speed and pad with silence.
# Prevents comically slow speech in windows with long narrator pauses.
_STRETCH_SKIP_RATIO = 0.5
_SPEED_MIN_LEGACY = 0.1
_SPEED_MAX_LEGACY = 10.0


class ChatterboxClient:
    """Thin HTTP client for the Chatterbox TTS API server (OpenAI-compatible).

    Uses /v1/audio/speech for default voice and /v1/audio/speech/upload
    when a speaker reference WAV is provided for voice cloning.
    """

    def __init__(self, base_url: str = CHATTERBOX_API_URL,
                 speaker_wav: str = CHATTERBOX_SPEAKER_WAV):
        self.base_url = base_url.rstrip("/")
        self.speaker_wav = speaker_wav  # path relative to pipeline_data/speakers/

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        """Synthesize *text* via the Chatterbox API and save the WAV to *file_path*.

        If *speaker_wav* is provided (via kwarg or constructor), uses the
        /v1/audio/speech/upload endpoint with the reference WAV for voice cloning.
        Otherwise uses /v1/audio/speech with the server's default voice.
        """
        chunks = self._split_text(text) if len(text) > 200 else [text]
        wav_parts = []

        speaker_wav = kwargs.get("speaker_wav", self.speaker_wav)

        for chunk in chunks:
            if speaker_wav:
                # Voice cloning: upload the reference WAV
                wav_parts.append(self._synthesize_with_voice(chunk, speaker_wav))
            else:
                # Default voice
                wav_parts.append(self._synthesize_default(chunk))

        if len(wav_parts) == 1:
            pathlib.Path(file_path).write_bytes(wav_parts[0])
        else:
            combined = AudioSegment.empty()
            for part in wav_parts:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    tmp.write(part)
                    tmp.flush()
                    combined += AudioSegment.from_wav(tmp.name)
            combined.export(file_path, format="wav")

    def _synthesize_default(self, text: str) -> bytes:
        """Call /v1/audio/speech with the server's default voice."""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/audio/speech",
                json={"input": text, "response_format": "wav"},
                timeout=(_CH_HTTP_CONNECT_S, _CH_HTTP_READ_S),
            )
        except requests.RequestException as exc:
            _log_tts_issue_once(
                "chatterbox_connect",
                f"Cannot reach Chatterbox at {self.base_url!r} ({exc}). "
                "Check SSH tunnel, CHATTERBOX_API_URL inside the API container, and that the GPU host is listening.",
            )
            raise

        if not resp.ok:
            hint = resp.text[:400].replace("\n", " ").strip()
            _log_tts_issue_once(
                f"http_{resp.status_code}",
                f"Chatterbox /v1/audio/speech HTTP {resp.status_code} ({self.base_url}): {hint}",
            )
        resp.raise_for_status()
        data = resp.content
        if not data or len(data) < 100:
            _log_tts_issue_once(
                "empty_wav_response",
                f"Chatterbox returned unusually small payload ({len(data) if data else 0} bytes).",
            )
        return data

    def _synthesize_with_voice(self, text: str, speaker_wav: str) -> bytes:
        """Call /v1/audio/speech/upload with a reference WAV for voice cloning."""
        # Resolve the speaker WAV path — could be relative to speakers dir
        speakers_base = pathlib.Path(__file__).parent.parent.parent.parent / "pipeline_data" / "speakers"
        wav_path = speakers_base / speaker_wav
        if not wav_path.exists():
            # Try as absolute path
            wav_path = pathlib.Path(speaker_wav)
        if not wav_path.exists():
            _logging.getLogger(__name__).warning(
                "[tts] Speaker WAV %s not found, falling back to default voice", speaker_wav
            )
            return self._synthesize_default(text)

        with open(wav_path, "rb") as f:
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/audio/speech/upload",
                    data={"input": text, "response_format": "wav"},
                    files={"voice_file": (wav_path.name, f, "audio/wav")},
                    timeout=(_CH_HTTP_CONNECT_S, _CH_HTTP_READ_S),
                )
            except requests.RequestException as exc:
                _log_tts_issue_once(
                    "chatterbox_upload_connect",
                    f"Cannot reach Chatterbox upload endpoint at {self.base_url!r} ({exc}).",
                )
                raise
        if not resp.ok:
            hint = resp.text[:400].replace("\n", " ").strip()
            _log_tts_issue_once(
                f"upload_http_{resp.status_code}",
                f"Chatterbox /v1/audio/speech/upload HTTP {resp.status_code}: {hint}",
            )
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _split_text(text: str, max_len: int = 200) -> list[str]:
        """Split text at sentence boundaries to stay under max_len chars."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if current and len(current) + len(s) + 1 > max_len:
                chunks.append(current.strip())
                current = s
            else:
                current = f"{current} {s}".strip() if current else s
        if current:
            chunks.append(current.strip())
        return chunks if chunks else [text]


# ── Edge TTS voice mapping ─────────────────────────────────────────────
# Microsoft neural voices — free, no API key, excellent quality.
# Pick gender by checking the diarization label stored on a segment.
_EDGE_VOICE_MALE   = os.getenv("FW_EDGE_VOICE_MALE",   "es-ES-AlvaroNeural")
_EDGE_VOICE_FEMALE = os.getenv("FW_EDGE_VOICE_FEMALE", "es-ES-ElviraNeural")
_EDGE_VOICE_DEFAULT = _EDGE_VOICE_MALE  # fallback when gender unknown


class EdgeTTSClient:
    """Free Microsoft neural TTS via the edge-tts package.

    Requires no API key.  Uses gender-specific Spanish voices:
    - Male:   es-ES-AlvaroNeural  (or FW_EDGE_VOICE_MALE env var)
    - Female: es-ES-ElviraNeural  (or FW_EDGE_VOICE_FEMALE env var)

    The ``gender`` kwarg accepted by ``tts_to_file`` should be
    ``"male"`` / ``"female"`` / ``None``.
    """

    # Silence threshold for stripping MP3 encoder predelay (dBFS).
    # Edge TTS MP3 output typically has 30–50ms of near-silence at the start
    # from the MP3 encoder frame boundary. Stripping it prevents every segment
    # from starting a beat late and accumulating drift over the full clip.
    _PREDELAY_STRIP_DBFS = -50.0
    _PREDELAY_MAX_STRIP_MS = 120  # never strip more than this

    def tts_to_file(self, text: str, file_path: str, **kwargs) -> None:
        import edge_tts

        gender = (kwargs.get("gender") or "").lower()
        if gender == "female":
            voice = _EDGE_VOICE_FEMALE
        elif gender == "male":
            voice = _EDGE_VOICE_MALE
        else:
            voice = _EDGE_VOICE_DEFAULT

        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            mp3_path = file_path + ".mp3"
            await communicate.save(mp3_path)
            audio = AudioSegment.from_mp3(mp3_path)
            # Strip MP3 encoder predelay — pydub's from_mp3 preserves the
            # ~576-sample encoder silence that precedes the first audio frame.
            # This causes every segment to start late, compounding into seconds
            # of drift over a full video.
            audio = self._strip_leading_silence(audio)
            audio.export(file_path, format="wav")
            pathlib.Path(mp3_path).unlink(missing_ok=True)

        # Always run in a fresh thread to avoid event-loop conflicts when called
        # from inside a ThreadPoolExecutor (which may or may not have a loop).
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(asyncio.run, _run()).result()

    def _strip_leading_silence(self, audio: AudioSegment) -> AudioSegment:
        """Remove leading near-silence (MP3 predelay) from a pydub AudioSegment."""
        chunk_ms = 5
        stripped_ms = 0
        while stripped_ms < self._PREDELAY_MAX_STRIP_MS:
            chunk = audio[stripped_ms : stripped_ms + chunk_ms]
            if len(chunk) < chunk_ms:
                break
            if chunk.dBFS > self._PREDELAY_STRIP_DBFS:
                break
            stripped_ms += chunk_ms
        if stripped_ms > 0:
            return audio[stripped_ms:]
        return audio


def _make_coqui_tts_engine():

    """Local Coqui TTS (Spanish tacotron). Uses CUDA if present, otherwise CPU."""

    import functools

    import torch
    from TTS.api import TTS as CoquiTTS

    _original_torch_load = torch.load
    @functools.wraps(_original_torch_load)
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tts] Using local Coqui TTS on {device}")
    return CoquiTTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False).to(device)


def _make_tts_engine():
    """Create TTS engine: Chatterbox → EdgeTTS → Coqui.

    Priority:
      1. Chatterbox GPU server (if reachable and FW_TTS_ENGINE not overridden)
      2. Edge TTS (Microsoft neural, free, gender-aware)  ← default local fallback
      3. Coqui Tacotron2 (offline CPU, robotic but works air-gapped)

    Force a specific engine via FW_TTS_ENGINE env var:
      FW_TTS_ENGINE=edge    → Edge TTS only
      FW_TTS_ENGINE=coqui   → Coqui only
      FW_TTS_ENGINE=local   → Coqui only (alias)
    """

    force = os.getenv("FW_TTS_ENGINE", "").strip().lower()
    if force == "edge":
        print("[tts] FW_TTS_ENGINE=edge — using Edge TTS (Microsoft neural)")
        return EdgeTTSClient()
    if force in ("coqui", "local", "cpu"):
        print(f"[tts] FW_TTS_ENGINE={force!r} — using bundled Coqui")
        return _make_coqui_tts_engine()

    skip_probe = os.getenv("FW_CHATTERBOX_SKIP_HEAVY_PROBE", "").lower() in ("1", "true", "yes")
    if skip_probe:
        client = ChatterboxClient()
        print(f"[tts] Using Chatterbox-compatible server at {CHATTERBOX_API_URL} (probe skipped)")
        return client

    try:
        client = ChatterboxClient()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            client.tts_to_file(text="prueba", file_path=tmp.name)
        print(f"[tts] Using Chatterbox GPU server at {CHATTERBOX_API_URL}")
        return client
    except Exception as exc:
        print(f"[tts] Chatterbox not available ({exc}), falling back to Edge TTS")

    # Try Edge TTS (free Microsoft neural voices, requires internet)
    try:
        import edge_tts  # noqa: F401
        print("[tts] Using Edge TTS (Microsoft neural, gender-aware Spanish voices)")
        return EdgeTTSClient()
    except ImportError:
        print("[tts] edge-tts not installed, falling back to Coqui")

    return _make_coqui_tts_engine()



_tts_engine = None


def _get_tts_engine():
    """Lazy singleton — resolved on first call, not at import time."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = _make_tts_engine()
    return _tts_engine


def text_from_file(file_path) -> str:
    with open(file_path, 'r') as file:
        trans = json.load(file)
    return trans["text"]


def segments_from_file(file_path) -> list[dict]:
    """Load segments with start/end timestamps from a translated JSON file."""
    with open(file_path, 'r') as file:
        trans = json.load(file)
    return trans.get("segments", [])


def files_from_dir(dir_path) -> list:
    SUFFIX = ".json"
    pth = pathlib.Path(dir_path)
    if not pth.exists():
        raise ValueError("provided path does not exist")

    es_files = glob.glob(str(pth) + "/*.json")

    if not es_files:
        raise ValueError(f"no {SUFFIX} files found in {pth}")

    return es_files


def _list_speaker_reference_relpaths(speakers_root: pathlib.Path) -> list[str]:
    """Relative paths under *speakers_root* for all ``*.wav`` files (sorted).

    ``default.wav`` (any depth) is excluded from the list when at least one
    other ``*.wav`` exists so per-speaker round-robin uses distinct reference
    clips instead of burning the first slot on the global fallback.
    """
    if not speakers_root.is_dir():
        return []
    all_rels = sorted(
        str(p.relative_to(speakers_root))
        for p in speakers_root.rglob("*.wav")
    )
    non_default = [r for r in all_rels if pathlib.Path(r).name.lower() != "default.wav"]
    return non_default if non_default else all_rels


def _unique_speaker_order(segments: list[dict]) -> list[str]:
    out: list[str] = []
    for seg in segments:
        sp = seg.get("speaker")
        if sp and sp not in out:
            out.append(str(sp))
    return out


def _speaker_voice_relpath_map(segments: list[dict], speakers_root: pathlib.Path) -> dict[str, str]:
    """Map pyannote speaker id → reference WAV path relative to *speakers_root*."""
    rels = _list_speaker_reference_relpaths(speakers_root)
    if not rels:
        return {}
    order = _unique_speaker_order(segments)
    return {sp: rels[i % len(rels)] for i, sp in enumerate(order)}


def _synthesize_raw(
    tts_engine,
    text: str,
    wav_path: str,
    *,
    speaker_wav: str | None = None,
    gender: str | None = None,
) -> bytes | None:
    """GPU-bound: call TTS engine and return raw WAV bytes, or None on failure."""
    if not text or not text.strip():
        return None
    try:
        if speaker_wav:
            try:
                tts_engine.tts_to_file(text=text, file_path=wav_path, speaker_wav=speaker_wav, gender=gender)
            except TypeError:
                tts_engine.tts_to_file(text=text, file_path=wav_path, gender=gender)
        else:
            tts_engine.tts_to_file(text=text, file_path=wav_path, gender=gender)
        return pathlib.Path(wav_path).read_bytes()
    except Exception as exc:
        print(f"[tts] TTS failed for segment ({exc}), using silence")
        return None


def _postprocess_segment(raw_wav_bytes: bytes | None, target_sec: float,
                         stretch_factor: float, alignment_enabled: bool,
                         work_dir: str) -> tuple:
    """CPU-bound: time-stretch raw TTS audio to match target duration.

    Returns (AudioSegment | None, speed_factor, raw_duration_s).
    """
    if target_sec <= 0:
        return (None, 0.0, 0.0)

    target_ms = int(target_sec * 1000)

    if raw_wav_bytes is None:
        return (AudioSegment.silent(duration=target_ms), 1.0, 0.0)

    work_path = pathlib.Path(work_dir)
    raw_wav = work_path / "raw_segment.wav"
    raw_wav.write_bytes(raw_wav_bytes)

    y, sr = librosa.load(str(raw_wav), sr=None)
    raw_duration = len(y) / sr

    if raw_duration == 0:
        return (AudioSegment.silent(duration=target_ms), 1.0, 0.0)

    duration_ratio = raw_duration / target_sec

    if not alignment_enabled:
        speed_factor = duration_ratio
        speed_factor = max(_SPEED_MIN_LEGACY, min(_SPEED_MAX_LEGACY, speed_factor))
    elif duration_ratio < _STRETCH_SKIP_RATIO:
        # TTS is dramatically shorter than target — narrator was pausing.
        # Play at natural speed; silence padding below handles the gap.
        speed_factor = 1.0
    else:
        effective_target = target_sec * max(stretch_factor, 0.1)
        speed_factor = raw_duration / effective_target
        speed_factor = max(SPEED_MIN, min(SPEED_MAX, speed_factor))

    if abs(speed_factor - 1.0) > 0.01:
        y_stretched = pyrubberband.time_stretch(y, sr, speed_factor)
    else:
        y_stretched = y

    stretched_wav = work_path / "stretched_segment.wav"
    sf.write(str(stretched_wav), y_stretched, sr)

    segment_audio = AudioSegment.from_wav(str(stretched_wav))

    if len(segment_audio) < target_ms:
        segment_audio += AudioSegment.silent(duration=target_ms - len(segment_audio))
    elif len(segment_audio) > target_ms:
        segment_audio = segment_audio[:target_ms]

    return (segment_audio, speed_factor, raw_duration)


def _synced_segment_audio(tts_engine, text: str, target_sec: float, work_dir, stretch_factor: float = 1.0, alignment_enabled: bool = True) -> tuple:
    """Generate TTS audio for *text* and time-stretch it to *target_sec*.

    Convenience wrapper kept for callers that don't use the batch path.
    """
    if target_sec <= 0:
        return (None, 0.0, 0.0)
    raw_wav = str(pathlib.Path(work_dir) / "raw_segment.wav")
    raw_bytes = _synthesize_raw(tts_engine, text, raw_wav)
    return _postprocess_segment(raw_bytes, target_sec, stretch_factor, alignment_enabled, str(work_dir))


def text_to_speech(text, output_file_path):
    _get_tts_engine().tts_to_file(text=text, file_path=str(output_file_path))


def _load_en_transcript(es_source_path: str) -> dict:
    """Locate the source-language transcript that corresponds to the translated file.

    Convention: translated JSON lives at .../translations/{model}/<title>.json
    Source transcript lives at .../transcriptions/{model}/<title>.json
    Returns an empty dict (no segments) if the source file is not found.
    """
    es_path = pathlib.Path(es_source_path)
    # Navigate: translations/{model}/ → data_dir → transcriptions/whisper/
    data_dir = es_path.parent.parent.parent
    en_path = data_dir / "transcriptions" / "whisper" / es_path.name
    if not en_path.exists():
        print(f"[tts] EN transcript not found at {en_path}, alignment skipped")
        return {}
    with open(en_path) as f:
        return json.load(f)


def _find_transcript_path(es_source_path: str) -> "pathlib.Path | None":
    """Locate the source transcription JSON, checking both whisper/ and flat paths."""
    es_path = pathlib.Path(es_source_path)
    data_dir = es_path.parent.parent.parent  # translations/{model}/ -> data_dir

    whisper_path = data_dir / "transcriptions" / "whisper" / es_path.name
    if whisper_path.exists():
        return whisper_path
    flat_path = data_dir / "transcriptions" / es_path.name
    if flat_path.exists():
        return flat_path
    return None


def _merge_speaker_labels(es_segments: list[dict], es_source_path: str) -> list[dict]:
    """Backfill ``speaker`` onto translation segments from the source transcription.

    Matches by start-time proximity (within 0.5s) because the translation and
    transcription pipelines produce different segment counts and id fields do
    NOT correspond 1:1.  For each translation segment, finds the transcription
    segment whose start time is closest and borrows its speaker label.

    Falls back to synthetic parity (even=male, odd=female) if no speaker labels
    exist in the transcription JSON yet (diarize not run / HF token missing).
    """
    transcript_path = _find_transcript_path(es_source_path)
    if transcript_path is None:
        _logging.getLogger(__name__).warning(
            "[tts] Transcription JSON not found at expected path — "
            "using synthetic parity speaker assignment"
        )
        return _synthetic_speaker_fallback(es_segments)

    try:
        with open(transcript_path) as f:
            en_data = json.load(f)
    except Exception:
        return _synthetic_speaker_fallback(es_segments)

    en_segs = en_data.get("segments", [])
    if not en_segs or "speaker" not in en_segs[0]:
        _logging.getLogger(__name__).warning(
            "[tts] No speaker labels in transcription JSON — "
            "diarize step has not run yet. Using synthetic parity assignment."
        )
        return _synthetic_speaker_fallback(es_segments)

    # Build a sorted list of (start_time, speaker) from the transcription
    en_starts: list[tuple[float, str]] = sorted(
        (float(s["start"]), str(s["speaker"]))
        for s in en_segs
        if "start" in s and "speaker" in s
    )

    if not en_starts:
        return _synthetic_speaker_fallback(es_segments)

    _MATCH_TOLERANCE_S = 0.5  # max seconds between translation and transcription segment starts

    n_matched = 0
    out = []
    for seg in es_segments:
        seg_copy = dict(seg)
        if "speaker" not in seg_copy:
            try:
                es_start = float(seg_copy["start"])
            except (KeyError, TypeError, ValueError):
                seg_copy["speaker"] = "SPEAKER_00"
                out.append(seg_copy)
                continue

            # Find nearest transcription segment by start time
            best_speaker = None
            best_dist = float("inf")
            for en_start, sp in en_starts:
                dist = abs(es_start - en_start)
                if dist < best_dist:
                    best_dist = dist
                    best_speaker = sp

            if best_speaker is not None and best_dist <= _MATCH_TOLERANCE_S:
                seg_copy["speaker"] = best_speaker
                n_matched += 1
            else:
                # No close match — fall back to parity by segment index
                idx = len(out)
                seg_copy["speaker"] = f"SPEAKER_{(idx % 2):02d}"

        out.append(seg_copy)

    print(f"[tts] Merged speaker labels: {n_matched}/{len(es_segments)} segments matched by start-time")
    return out


def _synthetic_speaker_fallback(segments: list[dict]) -> list[dict]:
    """Assign alternating SPEAKER_00/SPEAKER_01 by segment index parity.

    Used when no real diarization data is available.  Even index = male
    (SPEAKER_00), odd index = female (SPEAKER_01).  Far better than
    defaulting everything to male.
    """
    print("[tts] Synthetic speaker assignment: even segments=SPEAKER_00 (male), odd=SPEAKER_01 (female)")
    out = []
    for i, seg in enumerate(segments):
        seg_copy = dict(seg)
        if "speaker" not in seg_copy:
            seg_copy["speaker"] = f"SPEAKER_{(i % 2):02d}"
        out.append(seg_copy)
    return out


def _build_alignment(en_transcript: dict, es_transcript: dict, silence_regions: list[dict] | None = None) -> tuple:
    """Run global_align_dp and return (metrics_list, {segment_index: AlignedSegment}).

    Returns ([], {}) if the alignment library is unavailable or fails.
    Uses DP beam search (global_align_dp) over the greedy pass so gap-shift
    decisions are made globally, not just left-to-right.
    """
    try:
        from foreign_whispers.alignment import compute_segment_metrics, global_align_dp
    except ImportError:
        return [], {}
    try:
        metrics = compute_segment_metrics(en_transcript, es_transcript)
        regions = silence_regions or []
        aligned = global_align_dp(metrics, silence_regions=regions)
        return metrics, {seg.index: seg for seg in aligned}
    except Exception as exc:
        print(f"[tts] alignment failed ({exc}), proceeding without alignment")
        return [], {}


def _shorten_segment_text(en_text: str, es_text: str, target_sec: float) -> str:
    """Try to shorten a Spanish translation to fit *target_sec*.

    Delegates to ``get_shorter_translations()`` (student assignment stub).
    Returns the original *es_text* if no shorter candidate is available.
    """
    try:
        from foreign_whispers.reranking import get_shorter_translations
        candidates = get_shorter_translations(
            source_text=en_text,
            baseline_es=es_text,
            target_duration_s=target_sec,
        )
        if candidates:
            return candidates[0].text
    except Exception as exc:
        _logging.getLogger(__name__).warning("[tts] rerank failed: %s", exc)
    return es_text


def _write_align_report(
    output_path: str,
    stem: str,
    metrics: list,
    aligned: list,
    segment_details: list,
    *,
    synth_stats: dict | None = None,
) -> None:
    """Write a {stem}.align.json sidecar with evaluation metrics and per-segment detail.

    segment_details is a list of dicts: [{raw_duration_s, speed_factor, action, text}, ...]
    Written next to the WAV so both baseline and aligned runs produce comparable files.
    """
    try:
        from foreign_whispers.evaluation import clip_evaluation_report
        summary = clip_evaluation_report(metrics, aligned)
    except Exception as exc:
        _logging.getLogger(__name__).warning("clip_evaluation_report failed: %s", exc)
        summary = {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch": 0.0,
            "n_gap_shifts": 0,
            "n_translation_retries": 0,
            "total_cumulative_drift_s": 0.0,
        }

    report = {
        **summary,
        "alignment_enabled": _ALIGNMENT_ENABLED,
        **(synth_stats or {}),
        "segments": segment_details,
    }
    sidecar_path = pathlib.Path(output_path) / f"{stem}.align.json"
    sidecar_path.write_text(json.dumps(report, indent=2))


def _compute_speech_offset(source_path: str) -> float:
    """Compute timing offset between YouTube captions and Whisper segments.

    Returns seconds to add to Whisper timestamps so TTS audio aligns with
    the actual speech start in the original video.
    """
    title = pathlib.Path(source_path).stem
    # source_path: .../translations/{model}/{title}.json → data_dir is 3 levels up
    base_dir = pathlib.Path(source_path).parent.parent.parent

    yt_path = base_dir / "youtube_captions" / f"{title}.txt"
    whisper_path = base_dir / "transcriptions" / "whisper" / f"{title}.json"

    if not yt_path.exists() or not whisper_path.exists():
        return 0.0

    first_line = yt_path.read_text().split("\n", 1)[0].strip()
    if not first_line:
        return 0.0
    yt_start = json.loads(first_line).get("start", 0.0)

    whisper_data = json.loads(whisper_path.read_text())
    segs = whisper_data.get("segments", [])
    whisper_start = segs[0]["start"] if segs else 0.0

    return yt_start - whisper_start


def text_file_to_speech(
    source_path,
    output_path,
    tts_engine=None,
    *,
    alignment=None,
    per_speaker_voices: bool = True,
):
    """Read translated JSON with segment timestamps and produce a time-aligned WAV.

    Each segment is individually synthesized and time-stretched to match its
    original timestamp window.  Gaps between segments are filled with silence.
    Applies the YouTube caption timing offset so TTS audio starts when speech
    actually begins in the original video.

    *tts_engine* overrides the module-level ``tts`` instance (used by the
    FastAPI app which loads the model at startup).

    *alignment* overrides the module-level ``_ALIGNMENT_ENABLED`` flag.
    Pass True for aligned mode, False for baseline, or None to use the env var.

    When *per_speaker_voices* is True and segment dicts contain ``speaker``,
    reference WAVs under ``pipeline_data/speakers/**/*.wav`` are assigned
    round-robin to distinct speaker ids for Chatterbox voice cloning.
    """
    engine = tts_engine if tts_engine is not None else _get_tts_engine()
    use_alignment = alignment if alignment is not None else _ALIGNMENT_ENABLED

    save_name = pathlib.Path(source_path).stem + ".wav"
    print(f"generating {save_name}...", end="")

    segments = segments_from_file(source_path)

    # Backfill speaker labels from transcription JSON using start-time matching.
    # The translate router produces different segment counts than the transcription,
    # so id-based matching fails — we match by nearest start timestamp instead.
    segments = _merge_speaker_labels(segments, source_path)

    if not segments:
        text = text_from_file(source_path)
        save_path = pathlib.Path(output_path) / pathlib.Path(save_name)
        text_to_speech(text, str(save_path))
        print("success!")
        return None

    # Apply YouTube caption timing offset
    offset = _compute_speech_offset(source_path)
    # Clamp to zero — a negative offset (Whisper starts before YouTube caption)
    # would shift all segment start times backwards and corrupt the timeline.
    offset = max(0.0, offset)
    if offset > 0:
        print(f" (applying {offset:.1f}s speech offset)", end="")

    # Pre-compute alignment; also returns flat metrics list for clip_evaluation_report
    with open(source_path) as f:
        es_transcript = json.load(f)
    speakers_root = (
        pathlib.Path(__file__).resolve().parent.parent.parent.parent / "pipeline_data" / "speakers"
    )
    spk_to_wav = (
        _speaker_voice_relpath_map(es_transcript.get("segments", []), speakers_root)
        if per_speaker_voices
        else {}
    )
    en_transcript = _load_en_transcript(source_path)
    # Detect silence regions from the source audio for gap-shift alignment.
    # VAD gives the alignment pass real inter-segment pause information so it
    # can route segments to GAP_SHIFT instead of REQUEST_SHORTER unnecessarily.
    _silence_regions: list[dict] = []
    if use_alignment:
        try:
            from foreign_whispers.vad import detect_silence_regions
            # Source audio: same path as the source video but as WAV in the same data_dir
            data_dir = pathlib.Path(source_path).parent.parent.parent
            audio_candidates = list((data_dir / "audio").glob(f"{pathlib.Path(source_path).stem}*.wav"))
            if audio_candidates:
                _silence_regions = detect_silence_regions(str(audio_candidates[0]))
        except Exception as exc:
            print(f"[tts] VAD silence detection skipped ({exc})")
    if use_alignment:
        _metrics_list, align_map = _build_alignment(en_transcript, es_transcript, _silence_regions)
    else:
        _metrics_list, align_map = [], {}
    _aligned_list = list(align_map.values())

    # ── Prepare per-segment metadata ────────────────────────────────────
    seg_metas = []
    for i, seg in enumerate(segments):
        aligned_seg = align_map.get(i)
        stretch_factor = aligned_seg.stretch_factor if aligned_seg else 1.0
        target_sec = seg["end"] - seg["start"]

        seg_text = seg["text"]
        if aligned_seg is not None:
            from foreign_whispers.alignment import AlignAction
            if aligned_seg.action == AlignAction.REQUEST_SHORTER:
                en_text = ""
                en_segs = en_transcript.get("segments", [])
                if i < len(en_segs):
                    en_text = en_segs[i].get("text", "")
                seg_text = _shorten_segment_text(en_text, seg["text"], target_sec)

        raw_spk = seg.get("speaker") if isinstance(seg, dict) else None
        speaker_wav = spk_to_wav.get(str(raw_spk)) if raw_spk and spk_to_wav else None

        # Infer gender for EdgeTTS voice selection.
        # pyannote labels: SPEAKER_00, SPEAKER_01, ...
        # Some corpora use explicit gender suffixes (e.g. SPK_F0, SPK_M1).
        # Fallback: even-indexed speakers → male, odd-indexed → female.
        seg_gender: str | None = seg.get("gender")  # explicit if diarization set it
        if not seg_gender and raw_spk:
            spk_str = str(raw_spk).upper()
            if any(tag in spk_str for tag in ("_F", "FEM", "WOMAN")):
                seg_gender = "female"
            elif any(tag in spk_str for tag in ("_M", "MALE", "MAN")):
                seg_gender = "male"
            else:
                # Pyannote SPEAKER_00/01 — use speaker index parity
                spk_order = _unique_speaker_order(segments)
                idx = spk_order.index(str(raw_spk)) if str(raw_spk) in spk_order else 0
                seg_gender = "male" if idx % 2 == 0 else "female"

        seg_metas.append({
            "index": i,
            "text": seg_text,
            "start": seg["start"],
            "end": seg["end"],
            "target_sec": target_sec,
            "stretch_factor": stretch_factor,
            "aligned_seg": aligned_seg,
            "speaker_wav": speaker_wav,
            "gender": seg_gender,
        })

    # ── Phase 1: GPU synthesis (concurrent) ───────────────────────────
    # Submit all TTS calls to a thread pool so the GPU stays busy while
    # previous results are being downloaded / decoded.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _TTS_WORKERS = int(os.getenv("FW_TTS_WORKERS", "3"))

    raw_wav_map: dict[int, bytes | None] = {}

    with tempfile.TemporaryDirectory() as synth_dir:
        def _do_synth(meta: dict) -> tuple[int, bytes | None]:
            idx = meta["index"]
            wav_path = str(pathlib.Path(synth_dir) / f"seg_{idx}.wav")
            return idx, _synthesize_raw(
                engine,
                meta["text"],
                wav_path,
                speaker_wav=meta.get("speaker_wav"),
                gender=meta.get("gender"),
            )

        with ThreadPoolExecutor(max_workers=_TTS_WORKERS) as pool:
            futures = {
                pool.submit(_do_synth, m): m["index"]
                for m in seg_metas
            }
            for fut in as_completed(futures):
                idx, raw_bytes = fut.result()
                raw_wav_map[idx] = raw_bytes

    n_total = len(segments)
    n_ok_raw = sum(1 for i in range(n_total) if raw_wav_map.get(i))
    n_fail_raw = n_total - n_ok_raw
    print(
        f" ({n_total} segments; {n_ok_raw} with raw audio, {n_fail_raw} synth failures → silence / raw_duration_s=0)",
        end="",
    )
    _engine_label = type(engine).__name__
    synth_stats = {
        "tts_engine_class": _engine_label,
        "tts_segments_total": n_total,
        "tts_raw_synthesis_ok": n_ok_raw,
        "tts_raw_synthesis_failed": n_fail_raw,
        "chatterbox_api_url": CHATTERBOX_API_URL,
    }

    # ── Phase 2: CPU post-processing (sequential assembly) ────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        combined = AudioSegment.empty()
        cursor_ms = 0
        segment_details = []

        for m in seg_metas:
            i = m["index"]
            aligned_seg = m["aligned_seg"]

            # Use DP-aligned scheduled_start when available so gap-shift decisions
            # made by global_align_dp are actually reflected in the output timeline.
            # Fall back to raw segment start + offset when alignment is disabled.
            if aligned_seg is not None and use_alignment:
                start_ms = int((aligned_seg.scheduled_start + offset) * 1000)
            else:
                start_ms = int((m["start"] + offset) * 1000)
            # Never go backwards — if alignment pushed a segment before cursor,
            # clamp to current position rather than corrupting the timeline.
            start_ms = max(start_ms, cursor_ms)

            if start_ms > cursor_ms:
                combined += AudioSegment.silent(duration=start_ms - cursor_ms)
                cursor_ms = start_ms

            seg_audio, seg_speed_factor, seg_raw_duration = _postprocess_segment(
                raw_wav_map[i], m["target_sec"], m["stretch_factor"],
                use_alignment, tmpdir,
            )

            segment_details.append({
                "index": i,
                "text": m["text"],
                "target_sec": round(m["target_sec"], 3),
                "stretch_factor": round(m["stretch_factor"], 3),
                "raw_duration_s": round(seg_raw_duration, 3),
                "speed_factor": round(seg_speed_factor, 3),
                "action": aligned_seg.action.value if aligned_seg and hasattr(aligned_seg, "action") else "unknown",
                "scheduled_start_s": round(aligned_seg.scheduled_start, 3) if aligned_seg else round(m["start"], 3),
            })

            if seg_audio is not None:
                combined += seg_audio
                cursor_ms += len(seg_audio)
            else:
                # Even when seg_audio is None (silence pad from _postprocess_segment),
                # the cursor must advance by the target window so subsequent segments
                # don't collapse into the same position and overlap.
                target_ms = int(m["target_sec"] * 1000)
                if target_ms > 0:
                    combined += AudioSegment.silent(duration=target_ms)
                    cursor_ms += target_ms

        save_path = pathlib.Path(output_path) / save_name
        combined.export(str(save_path), format="wav")

    stem = pathlib.Path(source_path).stem
    _write_align_report(
        str(output_path), stem, _metrics_list, _aligned_list, segment_details, synth_stats=synth_stats
    )

    print("success!")
    return None


if __name__ == '__main__':
    SOURCE_PATH = "./data/transcriptions/es"
    OUTPUT_PATH = "./audios/"

    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    files = files_from_dir(SOURCE_PATH)
    for file in files:
        text_file_to_speech(file, OUTPUT_PATH)