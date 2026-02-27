"""
API route handlers — all versioned under /v1/.

Routes:
  POST /v1/analyze-audio   — submit audio, returns immediately with job_id
  GET  /v1/result/{job_id} — poll for pipeline result
  POST /v1/retry/{job_id}  — trigger manual retry from last checkpoint
  GET  /health             — liveness check
"""

import json
from datetime import datetime, timezone
from typing import Optional

UTC = timezone.utc

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.pipeline import (
    HealthResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    RetryResponse,
)
from app.services.orchestrator import (
    create_job,
    get_job,
    is_expired,
    resume_pipeline,
    start_pipeline,
)
from app.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter()


@router.post("/v1/analyze-audio", response_model=JobAcceptedResponse, status_code=202)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
) -> JobAcceptedResponse:
    """
    Accept an audio file, kick off background processing, return job_id immediately.
    Client polls /v1/result/{job_id} for the outcome.
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"

    job = create_job()
    log.info(
        "job_submitted",
        trace_id=job.trace_id,
        filename=filename,
        language=language,
        size_bytes=len(file_bytes),
    )

    background_tasks.add_task(start_pipeline, job, file_bytes, filename, language)

    return JobAcceptedResponse(
        job_id=job.job_id,
        trace_id=job.trace_id,
        status="received",
    )


@router.get("/v1/result/{job_id}", response_model=JobStatusResponse)
async def get_result(job_id: str) -> JobStatusResponse:
    """
    Poll this endpoint for the pipeline result.
    Returns current state while processing; full result when done; failure info if failed.
    """
    job = get_job(job_id)
    if job is None or is_expired(job):
        raise HTTPException(status_code=404, detail="Job not found or expired")

    response = JobStatusResponse(
        job_id=job.job_id,
        trace_id=job.trace_id,
        status=job.status,
        failed_at_stage=job.failed_at_stage,
        reason=job.reason,
    )

    # Attach final result if pipeline completed successfully
    if job.status == "done":
        try:
            from app.utils.storage import load_final_result

            final = await load_final_result(job.trace_id)
            
            # Print for terminal visibility as requested by user
            checkpoint_path = settings.output_dir + f"/final_{job.trace_id}.json"
            print(f"\n--- Result for Job {job_id} ---")
            print(f"Source: {checkpoint_path}")
            print(f"Result: {json.dumps(final.get('result'), indent=2)}")
            print("-------------------------------\n")

            response.result = final.get("result")
            response.metadata = final.get("metadata")
        except Exception:

            log.warning(
                "llm_result_not_found",
                trace_id=job.trace_id,
                job_id=job.job_id,
                status=job.status,
                failed_at_stage=job.failed_at_stage,
                reason=job.reason,
            )

            # Checkpoint may have been cleaned — build from in-memory outputs
            if job.llm_output and job.stt_output:
                response.result = {
                    "intent": job.llm_output.intent,
                    "confidence": job.llm_output.confidence,
                    "action": job.llm_output.action,
                    "reasoning": job.llm_output.reasoning,
                    "status": job.llm_output.status,
                    "model_used": job.llm_output.model_used,
                    "fallback_triggered": job.llm_output.fallback_triggered,
                    "prompt_version": job.llm_output.prompt_version,
                }

    return response


@router.post("/v1/retry/{job_id}", response_model=RetryResponse)
async def retry_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> RetryResponse:
    """
    Trigger manual retry from last successful checkpoint.
    Requires re-upload of the audio file in case STT must be rerun.
    """
    job = get_job(job_id)
    if job is None or is_expired(job):
        raise HTTPException(status_code=404, detail="Job not found or expired")

    # Concurrent retry guard
    if job.status in ("retrying", "stt_processing", "llm_processing", "validating"):
        raise HTTPException(status_code=400, detail="Job is already being processed")

    if job.status != "failed":
        raise HTTPException(
            status_code=400, detail=f"Job is not in failed state (current: {job.status})"
        )

    # Max retry guard
    if job.retry_count >= settings.max_manual_retries:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum retry limit ({settings.max_manual_retries}) reached — job permanently failed",
        )

    resume_stage = job.failed_at_stage or "stt"

    # Mark retrying immediately — before background task starts, prevents concurrent calls
    job.status = "retrying"
    job.retry_count += 1

    file_bytes = await file.read()
    filename = file.filename or "upload"

    log.info(
        "job_retry_accepted",
        trace_id=job.trace_id,
        resume_stage=resume_stage,
        retry_count=job.retry_count,
    )

    background_tasks.add_task(resume_pipeline, job, file_bytes, filename)

    return RetryResponse(
        job_id=job_id,
        status="retrying_from",
        stage=resume_stage,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(UTC).isoformat() + "Z",
    )
