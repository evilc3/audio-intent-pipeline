"""
Orchestration layer: pipeline control flow, state machine, checkpointing,
partial retry, TTL expiry, and final output assembly.

This is the glue that sequences audio → STT → LLM and handles every failure mode.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

UTC = timezone.utc

from fastapi import BackgroundTasks

from app.config import settings
from app.models.pipeline import JobState
from app.models.stt import STTOutput
from app.models.llm import LLMOutput
from app.services.audio import AudioValidationError, validate_and_preprocess
from app.services.llm import LLMFailedException, classify_intent
from app.services.stt import STTFailedException, transcribe
from app.utils.storage import CheckpointError, cleanup_checkpoints, load_checkpoint, save_checkpoint, save_final_result
from app.utils.logging import get_logger

log = get_logger(__name__)

# In-memory job store — production would use Redis
_jobs: dict[str, JobState] = {}


# ---------------------------------------------------------------------------
# Public API (called from routes)
# ---------------------------------------------------------------------------


def create_job() -> JobState:
    """Allocate a new job with a fresh UUID. Stored immediately so /result can poll."""
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    job = JobState(
        job_id=job_id,
        trace_id=job_id,  # same UUID serves as both job ID and trace ID
        status="received",
        created_at=now,
        updated_at=now,
    )
    _jobs[job_id] = job
    log.info("job_created", trace_id=job.trace_id, status="received")
    return job


def get_job(job_id: str) -> Optional[JobState]:
    return _jobs.get(job_id)


def is_expired(job: JobState) -> bool:
    age = (datetime.now(UTC) - job.created_at.replace(tzinfo=UTC)).total_seconds()
    return age > settings.job_ttl_seconds


async def start_pipeline(
    job: JobState,
    file_bytes: bytes,
    filename: str,
    language: Optional[str],
) -> None:
    """
    Full pipeline: validation → STT → LLM → final output.
    Called as a BackgroundTask — runs outside the request lifecycle.
    """
    trace_id = job.trace_id
    t_pipeline_start = _now_ms()

    try:
        # --- Stage: audio validation ---
        _update_state(job, "validating")
        try:
            processed_audio = await validate_and_preprocess(file_bytes, filename, trace_id)
        except AudioValidationError as exc:
            # Validation failure is a caller error — mark failed and surface reason
            _update_state(job, "failed", failed_at_stage="validation", reason=str(exc))
            return

        # --- Stage: STT ---
        _update_state(job, "stt_processing")
        stt_start_ms = _now_ms()
        try:
            stt_output = await transcribe(
                processed_audio,
                trace_id=trace_id,
                language=language,
            )
        except STTFailedException as exc:
            _update_state(job, "failed", failed_at_stage="stt", reason=str(exc))
            return

        stt_latency_ms = _now_ms() - stt_start_ms

        is_valid, reason = _validate_stt_output(stt_output, trace_id)
        if not is_valid:
            _update_state(
                job,
                "failed",
                failed_at_stage="stt",
                reason=reason or "STT output failed intermediate validation",
            )
            return

        await save_checkpoint(trace_id, "stt", stt_output.model_dump())
        job.stt_output = stt_output
        _update_state(job, "stt_complete")

        # --- Stage: LLM ---
        _update_state(job, "llm_processing")
        llm_start_ms = _now_ms()
        try:
            llm_output = await classify_intent(stt_output, trace_id)
        except LLMFailedException as exc:
            _update_state(job, "failed", failed_at_stage="llm", reason=str(exc))
            return

        llm_latency_ms = _now_ms() - llm_start_ms

        is_valid, reason = _validate_llm_output(llm_output, trace_id)
        if not is_valid:
            _update_state(
                job,
                "failed",
                failed_at_stage="llm",
                reason=reason or "LLM output failed intermediate validation",
            )
            return

        await save_checkpoint(trace_id, "llm", llm_output.model_dump())
        job.llm_output = llm_output
        _update_state(job, "llm_complete")

        # --- Stage: assemble final output ---
        total_latency_ms = _now_ms() - t_pipeline_start
        final = _assemble_final_output(
            job=job,
            stt_output=stt_output,
            llm_output=llm_output,
            stt_latency_ms=stt_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
        )

        # print('llm_output:', llm_output)
        # print('stt_output:', stt_output)    
        # print('final:', final)

        # --- Stage: save final output ---
        await save_final_result(trace_id, final)
        _update_state(job, "done")

        log.info(
            "pipeline_complete",
            trace_id=trace_id,
            total_latency_ms=round(total_latency_ms, 1),
            intent=llm_output.intent,
            status="done",
        )

        # TTS stub — shows where real TTS would fit
        await _tts_stub(
            text=llm_output.reasoning or "",
            language=stt_output.language,
            trace_id=trace_id,
        )

    finally:
        # Checkpoint cleanup is best-effort — always runs
        cleanup_checkpoints(trace_id)


async def resume_pipeline(job: JobState, file_bytes: bytes, filename: str) -> None:
    """
    Resume pipeline from last successful checkpoint.
    Called on manual retry — only reruns stages that failed.
    """
    trace_id = job.trace_id
    failed_stage = job.failed_at_stage

    log.info(
        "pipeline_retry_start",
        trace_id=trace_id,
        failed_at_stage=failed_stage,
        retry_count=job.retry_count,
    )

    try:
        stt_output: Optional[STTOutput] = None
        llm_output: Optional[LLMOutput] = None

        if failed_stage in ("llm", None):
            # STT checkpoint exists — load it
            try:
                stt_data = await load_checkpoint(trace_id, "stt")
                stt_output = STTOutput(**stt_data)
            except CheckpointError:
                # No STT checkpoint — must rerun from STT
                failed_stage = "stt"

        if failed_stage == "stt" or stt_output is None:
            # Need to reprocess audio — requires the original bytes
            _update_state(job, "stt_processing")
            stt_start_ms = _now_ms()
            try:
                processed_audio = await validate_and_preprocess(file_bytes, filename, trace_id)
                stt_output = await transcribe(processed_audio, trace_id=trace_id)
            except (AudioValidationError, STTFailedException) as exc:
                _update_state(job, "failed", failed_at_stage="stt", reason=str(exc))
                return

            stt_latency_ms = _now_ms() - stt_start_ms

            is_valid, reason = _validate_stt_output(stt_output, trace_id)
            if not is_valid:
                _update_state(
                    job,
                    "failed",
                    failed_at_stage="stt",
                    reason=reason or "STT output failed intermediate validation",
                )
                return

            await save_checkpoint(trace_id, "stt", stt_output.model_dump())
            job.stt_output = stt_output
            _update_state(job, "stt_complete")
        else:
            stt_latency_ms = stt_output.latency_ms
            # Validate loaded checkpoint just in case
            is_valid, reason = _validate_stt_output(stt_output, trace_id)
            if not is_valid:
                _update_state(job, "failed", failed_at_stage="stt", reason=reason)
                return

        # Run LLM stage
        _update_state(job, "llm_processing")
        llm_start_ms = _now_ms()
        try:
            llm_output = await classify_intent(stt_output, trace_id)
        except LLMFailedException as exc:
            _update_state(job, "failed", failed_at_stage="llm", reason=str(exc))
            return

        llm_latency_ms = _now_ms() - llm_start_ms

        is_valid, reason = _validate_llm_output(llm_output, trace_id)
        if not is_valid:
            _update_state(
                job,
                "failed",
                failed_at_stage="llm",
                reason=reason or "LLM output failed intermediate validation",
            )
            return

        await save_checkpoint(trace_id, "llm", llm_output.model_dump())
        job.llm_output = llm_output
        _update_state(job, "llm_complete")

        total_latency_ms = stt_latency_ms + llm_latency_ms
        final = _assemble_final_output(
            job=job,
            stt_output=stt_output,
            llm_output=llm_output,
            stt_latency_ms=stt_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
        )
        await save_final_result(trace_id, final)
        _update_state(job, "done")

    finally:
        cleanup_checkpoints(trace_id)


# ---------------------------------------------------------------------------
# Intermediate validation — catches silent failures between stages
# ---------------------------------------------------------------------------


def _validate_stt_output(output: STTOutput, trace_id: str) -> tuple[bool, Optional[str]]:
    if output.status == "failed":
        reason = "status_failed"
        log.warning("stt_intermediate_validation_failed", trace_id=trace_id, reason=reason)
        return False, reason
    if not output.transcript.strip():
        reason = "empty_transcript"
        log.warning("stt_intermediate_validation_failed", trace_id=trace_id, reason=reason)
        return False, reason
    if output.language not in settings.supported_languages:
        log.warning(
            "stt_intermediate_validation_failed",
            trace_id=trace_id,
            reason="unsupported_language",
            language=output.language,
        )
        # Allow pipeline to continue with unknown language — LLM handles it
        return True, None
    return True, None


def _validate_llm_output(output: LLMOutput, trace_id: str) -> tuple[bool, Optional[str]]:
    if output.status == "failed":
        reason = "status_failed"
        log.warning("llm_intermediate_validation_failed", trace_id=trace_id, reason=reason)
        return False, reason
    if output.intent not in settings.supported_intents:
        reason = "unsupported_intent"
        log.warning(
            "llm_intermediate_validation_failed",
            trace_id=trace_id,
            reason=reason,
            intent=output.intent,
        )
        return False, reason
    if output.confidence is None or not (0.0 <= output.confidence <= 1.0):
        reason = "invalid_confidence"
        log.warning("llm_intermediate_validation_failed", trace_id=trace_id, reason=reason)
        return False, reason
    if not (output.action or "").strip():
        reason = "empty_action"
        log.warning("llm_intermediate_validation_failed", trace_id=trace_id, reason=reason)
        return False, reason
    return True, None


# ---------------------------------------------------------------------------
# Final output assembly
# ---------------------------------------------------------------------------


def _assemble_final_output(
    job: JobState,
    stt_output: STTOutput,
    llm_output: LLMOutput,
    stt_latency_ms: float,
    llm_latency_ms: float,
    total_latency_ms: float,
) -> dict:
    return {
        "job_id": job.job_id,
        "trace_id": job.trace_id,
        "status": "done",
        "result": {
            "intent": llm_output.intent,
            "confidence": llm_output.confidence,
            "action": llm_output.action,
            "reasoning": llm_output.reasoning,
            "status": llm_output.status,
            "model_used": llm_output.model_used,
            "fallback_triggered": llm_output.fallback_triggered,
            "prompt_version": llm_output.prompt_version,
        },
        "metadata": {
            "stt": {
                "transcript": stt_output.transcript,
                "language": stt_output.language,
                "confidence": stt_output.confidence,
                "duration_seconds": stt_output.duration_seconds,
                "status": stt_output.status,
            },
            "pipeline": {
                "total_latency_ms": round(total_latency_ms, 1),
                "stt_latency_ms": round(stt_latency_ms, 1),
                "llm_latency_ms": round(llm_latency_ms, 1),
                "input_tokens_consumed": llm_output.prompt_tokens,
                "output_tokens_generated": llm_output.completion_tokens,
                "estimated_cost_usd": llm_output.estimated_cost_usd,
                "retries": llm_output.retry_count + stt_output.retry_count,
                "created_at": job.created_at.isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
            },
        },
    }


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _update_state(
    job: JobState,
    status: str,
    failed_at_stage: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    job.status = status
    job.updated_at = datetime.now(UTC)
    if failed_at_stage:
        job.failed_at_stage = failed_at_stage
    if reason:
        job.reason = reason
    log.info(
        "state_transition",
        trace_id=job.trace_id,
        status=status,
        failed_at_stage=failed_at_stage,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# TTS stub
# ---------------------------------------------------------------------------


async def _tts_stub(text: str, language: str, trace_id: str) -> dict:
    """
    Stub for TTS step. Shows where TTS fits in the pipeline.
    Production: integrate Azure TTS, Google TTS, or ElevenLabs.
    """
    result = {
        "status": "stub",
        "text": text,
        "language": language,
        "audio_url": None,
        "note": "TTS not implemented — stub only",
    }
    log.info("tts_stub_called", trace_id=trace_id, language=language)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    import time

    return time.monotonic() * 1000
