"""
Audio validation and preprocessing service.

Validates audio files before any API calls are made (fail fast, no wasted cost).
Preprocessing normalises audio to the specification required by Deepgram STT.
"""

import io
import os
import tempfile
from pathlib import Path

import magic
from pydub import AudioSegment

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


class AudioValidationError(Exception):
    """Raised when a submitted audio file fails any validation check."""

    pass


async def validate_and_preprocess(
    file_bytes: bytes,
    filename: str,
    trace_id: str,
) -> bytes:
    """
    Validate then preprocess uploaded audio bytes.

    Validation order (cheap checks first, no API calls):
      1. Extension is .wav or .mp3
      2. Real MIME type matches audio/* (python-magic, checks actual bytes)
      3. File size within bounds
      4. Audio duration within bounds

    Preprocessing (pydub):
      - Convert stereo → mono
      - Resample to 8 kHz

    Returns preprocessed audio bytes ready for STT.
    Temp file cleanup is guaranteed via try/finally.
    """
    _validate_extension(filename, trace_id)

    # Write to temp file for pydub/magic operations
    suffix = Path(filename).suffix
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        _validate_mime_type(tmp_path, trace_id)
        _validate_file_size(len(file_bytes), trace_id)
        processed_bytes = _preprocess_audio(tmp_path, trace_id)
        return processed_bytes
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            log.debug("temp_file_cleaned", trace_id=trace_id, path=tmp_path)


# --- Internal helpers ---


def _validate_extension(filename: str, trace_id: str) -> None:
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in settings.supported_extensions:
        log.warning(
            "audio_validation_failed",
            trace_id=trace_id,
            reason="unsupported_extension",
            extension=ext,
        )
        raise AudioValidationError(
            f"Unsupported file extension '.{ext}'. Supported: {settings.supported_extensions}"
        )


def _validate_mime_type(path: str, trace_id: str) -> None:
    """Use python-magic to check the actual file bytes — not just the extension."""
    mime = magic.from_file(path, mime=True)
    if not mime.startswith("audio/"):
        log.warning(
            "audio_validation_failed",
            trace_id=trace_id,
            reason="invalid_mime_type",
            mime=mime,
        )
        raise AudioValidationError(
            f"File MIME type '{mime}' is not audio. Possible file spoofing detected."
        )


def _validate_file_size(size_bytes: int, trace_id: str) -> None:
    if size_bytes < settings.min_file_size_bytes:
        log.warning(
            "audio_validation_failed",
            trace_id=trace_id,
            reason="file_too_small",
            size_bytes=size_bytes,
        )
        raise AudioValidationError(
            f"File too small ({size_bytes} bytes). "
            f"Minimum is {settings.min_file_size_bytes} bytes."
        )
    if size_bytes > settings.max_file_size_bytes:
        log.warning(
            "audio_validation_failed",
            trace_id=trace_id,
            reason="file_too_large",
            size_bytes=size_bytes,
        )
        raise AudioValidationError(
            f"File too large ({size_bytes} bytes). "
            f"Maximum is {settings.max_file_size_bytes} bytes."
        )


def _preprocess_audio(path: str, trace_id: str) -> bytes:
    """
    Load audio, validate duration, convert to mono 8 kHz WAV.
    Duration check happens here because pydub must load the file first.
    """
    audio = AudioSegment.from_file(path)

    duration_seconds = len(audio) / 1000.0
    if duration_seconds < settings.min_audio_duration_seconds:
        raise AudioValidationError(
            f"Audio duration {duration_seconds:.2f}s too short. "
            f"Minimum is {settings.min_audio_duration_seconds}s."
        )
    if duration_seconds > settings.max_audio_duration_seconds:
        raise AudioValidationError(
            f"Audio duration {duration_seconds:.2f}s too long. "
            f"Maximum is {settings.max_audio_duration_seconds}s."
        )

    # Normalise: mono + 8 kHz (STT standard)
    audio = audio.set_channels(1).set_frame_rate(8000)

    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    processed = buffer.getvalue()

    log.info(
        "audio_preprocessed",
        trace_id=trace_id,
        duration_seconds=round(duration_seconds, 2),
        output_bytes=len(processed),
    )
    return processed
