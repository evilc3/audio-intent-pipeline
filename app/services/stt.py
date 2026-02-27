"""
STT service: Deepgram Nova (primary) + Whisper local (fallback).

Retry strategy: exponential backoff on Deepgram before triggering Whisper.
Confidence below threshold is flagged but not discarded — passes downstream.
"""

import asyncio
import io
import time
from typing import Optional

from app.config import settings
from app.models.stt import STTOutput
from app.utils.logging import get_logger
from pydub import AudioSegment

log = get_logger(__name__)


class STTFailedException(Exception):
    """Raised when both Deepgram and Whisper fail to produce a transcript."""

    pass


async def transcribe(
    audio_bytes: bytes,
    trace_id: str,
    language: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    punctuate: bool = True,
) -> STTOutput:
    """
    Transcribe audio bytes to text.

    Tries Deepgram first with exponential backoff retries.
    Falls back to local Whisper if Deepgram exhausts retries.
    """
    lang = language or settings.default_language
    kw = keywords or []

    output, retry_count = await _try_deepgram(audio_bytes, trace_id, lang, kw, punctuate)
    if output is not None:
        return output

    log.warning("stt_fallback_triggered", trace_id=trace_id, reason="deepgram_exhausted_retries")
    output = await _try_whisper(audio_bytes, trace_id, lang, retry_count)
    if output is not None:
        return output

    raise STTFailedException("Both Deepgram and Whisper failed to produce a transcript")


# --- Deepgram ---


async def _try_deepgram(
    audio_bytes: bytes,
    trace_id: str,
    language: str,
    keywords: list[str],
    punctuate: bool,
) -> tuple[Optional[STTOutput], int]:
    """
    Try Deepgram with exponential backoff.
    Returns (STTOutput, retry_count) on success, (None, retry_count) if all attempts fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(settings.stt_max_retries):
        try:
            t_start = time.monotonic()
            result = await _call_deepgram(audio_bytes, language, keywords, punctuate)
            latency_ms = (time.monotonic() - t_start) * 1000

            transcript = result["transcript"]
            confidence = result["confidence"]
            detected_lang = result["language"]
            duration = result["duration_seconds"]

            status = "ok"
            if confidence < settings.stt_confidence_threshold:
                status = "low_confidence"
                log.warning(
                    "stt_low_confidence",
                    trace_id=trace_id,
                    confidence=confidence,
                    threshold=settings.stt_confidence_threshold,
                )

            log.info(
                "stt_deepgram_success",
                trace_id=trace_id,
                attempt=attempt,
                confidence=confidence,
                language=detected_lang,
                duration_seconds=duration,
                latency_ms=round(latency_ms, 1),
            )
            return (
                STTOutput(
                    transcript=transcript,
                    language=detected_lang,
                    confidence=confidence,
                    duration_seconds=duration,
                    status=status,
                    fallback_triggered=False,
                    retry_count=attempt,
                    latency_ms=round(latency_ms, 1),
                ),
                attempt,
            )

        except Exception as exc:
            last_exc = exc
            delay = settings.stt_retry_base_delay_seconds ** (attempt + 1)
            log.warning(
                "stt_deepgram_retry",
                trace_id=trace_id,
                attempt=attempt + 1,
                error=str(exc),
                retry_delay_seconds=delay,
            )
            await asyncio.sleep(delay)

    log.error(
        "stt_deepgram_failed",
        trace_id=trace_id,
        attempts=settings.stt_max_retries,
        error=str(last_exc),
    )
    return None, settings.stt_max_retries


async def _call_deepgram(
    audio_bytes: bytes,
    language: str,
    keywords: list[str],
    punctuate: bool,
) -> dict:
    """
    Make the actual Deepgram API call.
    Uses the async Deepgram Python SDK.
    """
    from deepgram import DeepgramClient

    client = DeepgramClient(api_key=settings.deepgram_api_key)

    response = client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model=settings.deepgram_model,
                smart_format=True,
                language=language,
                punctuate=punctuate,
                keyterm=keywords if keywords else None, #used for keywords
            )
    
    response = response.dict()

    if not response or not response.get("results"):
        log.error("stt_deepgram_empty_response", language=language)
        return {
            "transcript": "",
            "confidence": 0.0,
            "language": "en",
            "duration_seconds": 0.0,
        }

    channel = response["results"]["channels"][0]
    alt = channel["alternatives"][0]

    transcript = alt["transcript"] or ""
    confidence = alt["confidence"] if alt["confidence"] is not None else 0.0

    # Duration comes from response metadata
    duration = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav").duration_seconds
    
    # Deepgram returns the detected language at the channel level
    detected_lang = "en"
    if hasattr(channel, "detected_language") and channel.detected_language:
        detected_lang = channel.detected_language.split("-")[0]

    return {
        "transcript": transcript,
        "confidence": confidence,
        "language": detected_lang,
        "duration_seconds": duration,
    }


# --- Whisper fallback ---


async def _try_whisper(
    audio_bytes: bytes,
    trace_id: str,
    language: str,
    deepgram_retry_count: int,
) -> Optional[STTOutput]:
    """
    Local Whisper fallback. Runs in a thread pool to avoid blocking the event loop.
    """
    try:
        t_start = time.monotonic()
        result = await asyncio.get_event_loop().run_in_executor(
            None, _call_whisper_sync, audio_bytes, language
        )
        latency_ms = (time.monotonic() - t_start) * 1000

        print(result)

        transcript = result["transcript"]
        # Whisper does not return a numeric confidence — use 0.0 as a proxy
        # (model-level quality, not token-level probability)
        confidence = 0.0
        detected_lang = result.get("language", language.split("-")[0])

        status = "ok"
        if not transcript.strip():
            status = "failed"

        log.info(
            "stt_whisper_success",
            trace_id=trace_id,
            language=detected_lang,
            latency_ms=round(latency_ms, 1),
        )
        return STTOutput(
            transcript=transcript,
            language=detected_lang,
            confidence=confidence,
            duration_seconds=result.get("duration_seconds", 0.0),
            status=status,
            fallback_triggered=True,
            retry_count=deepgram_retry_count,
            latency_ms=round(latency_ms, 1),
        )
    except Exception as exc:
        log.error("stt_whisper_failed", trace_id=trace_id, error=str(exc))
        return None


def _call_whisper_sync(audio_bytes: bytes, language: str) -> dict:
    """
    Synchronous Whisper call — executed in a thread to stay off the event loop.
    Whisper's CPU-bound model loading blocks, so isolation matters.
    """
    import io
    import tempfile
    import os

    import whisper

    model = whisper.load_model("base")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        lang_code = language.split("-")[0]  # "en-US" → "en"
        result = model.transcribe(tmp_path, language=lang_code)

        duration = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav").duration_seconds

        transcript = result.get("text", "")
        if not isinstance(transcript, str):
            transcript = str(transcript)

        return {
            "transcript": transcript.strip(),
            "language": str(result.get("language", lang_code)),
            "duration_seconds": duration,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
