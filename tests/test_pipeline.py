"""
Unit tests for pipeline orchestration:
- Checkpoint save/load/cleanup
- State transitions
- Audio validation rejection
- Concurrent retry guard
- Max retry limit enforcement
- TTL expiry
"""

import os
import json
import tempfile
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.pipeline import JobState
from app.models.stt import STTOutput
from app.models.llm import LLMOutput
from app.services.orchestrator import (
    create_job,
    get_job,
    is_expired,
)
from app.utils.storage import save_checkpoint, load_checkpoint, cleanup_checkpoints, CheckpointError


# ---------------------------------------------------------------------------
# Checkpoint utility tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_save_and_load(tmp_path) -> None:
    """Checkpoint round-trips correctly through save → load."""
    with patch("app.utils.storage.settings") as mock_settings:
        mock_settings.checkpoint_dir = str(tmp_path)

        data = {"transcript": "Hello", "confidence": 0.9}
        await save_checkpoint("trace-1", "stt", data)

        loaded = await load_checkpoint("trace-1", "stt")

    assert loaded["transcript"] == "Hello"
    assert loaded["confidence"] == 0.9


@pytest.mark.asyncio
async def test_checkpoint_load_missing_raises(tmp_path) -> None:
    """Loading a non-existent checkpoint raises CheckpointError."""
    with patch("app.utils.storage.settings") as mock_settings:
        mock_settings.checkpoint_dir = str(tmp_path)

        with pytest.raises(CheckpointError, match="not found"):
            await load_checkpoint("nonexistent-trace", "stt")


@pytest.mark.asyncio
async def test_checkpoint_cleanup_removes_files(tmp_path) -> None:
    """Cleanup deletes all stage checkpoint files for a trace_id."""
    with patch("app.utils.storage.settings") as mock_settings:
        mock_settings.checkpoint_dir = str(tmp_path)
        mock_settings.checkpoint_ttl_seconds = 3600

        await save_checkpoint("trace-clean", "stt", {"data": "x"})
        await save_checkpoint("trace-clean", "llm", {"data": "y"})

        cleanup_checkpoints("trace-clean")

        assert not (tmp_path / "stt_trace-clean.json").exists()
        assert not (tmp_path / "llm_trace-clean.json").exists()


# ---------------------------------------------------------------------------
# Job state tests
# ---------------------------------------------------------------------------


def test_create_job_returns_valid_state() -> None:
    """create_job() allocates a job with received status and valid UUIDs."""
    job = create_job()

    assert job.status == "received"
    assert job.job_id == job.trace_id
    assert len(job.job_id) == 36  # UUID4 length
    assert get_job(job.job_id) is job


def test_get_job_missing_returns_none() -> None:
    result = get_job("nonexistent-uuid")
    assert result is None


def test_is_expired_within_ttl() -> None:
    job = create_job()
    assert is_expired(job) is False


def test_is_expired_past_ttl() -> None:
    job = create_job()
    from datetime import timezone
    # Force creation time to be in the past
    job.created_at = datetime.now(timezone.utc) - timedelta(seconds=7200)
    assert is_expired(job) is True


# ---------------------------------------------------------------------------
# Audio validation rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_audio_validation_failure_sets_failed_state() -> None:
    """AudioValidationError during pipeline marks job as failed."""
    from app.services.audio import AudioValidationError
    from app.services.orchestrator import start_pipeline

    job = create_job()

    with patch(
        "app.services.orchestrator.validate_and_preprocess",
        side_effect=AudioValidationError("File too small"),
    ):
        with patch("app.services.orchestrator.cleanup_checkpoints"):
            await start_pipeline(job, b"bad_audio", "bad.wav", None)

    assert job.status == "failed"
    assert job.failed_at_stage == "validation"


# ---------------------------------------------------------------------------
# Retry guard tests (via route layer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_rejected_when_job_processing() -> None:
    """Retry request rejected with 400 if job is already processing."""
    from fastapi.testclient import TestClient
    from app.main import app

    job = create_job()
    job.status = "stt_processing"
    job.failed_at_stage = "llm"

    client = TestClient(app)
    # Use multipart with a dummy file
    response = client.post(
        f"/v1/retry/{job.job_id}",
        files={"file": ("test.wav", b"x" * 1024, "audio/wav")},
    )
    assert response.status_code == 400
    assert "already being processed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_retry_rejected_when_not_failed() -> None:
    """Retry request rejected if job is not in failed state."""
    from fastapi.testclient import TestClient
    from app.main import app

    job = create_job()
    job.status = "done"

    client = TestClient(app)
    response = client.post(
        f"/v1/retry/{job.job_id}",
        files={"file": ("test.wav", b"x" * 1024, "audio/wav")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_retry_rejected_when_max_retries_exceeded() -> None:
    """Retry blocked when retry_count >= max_manual_retries."""
    from fastapi.testclient import TestClient
    from app.main import app

    job = create_job()
    job.status = "failed"
    job.retry_count = 3  # at max
    job.failed_at_stage = "llm"

    client = TestClient(app)
    response = client.post(
        f"/v1/retry/{job.job_id}",
        files={"file": ("test.wav", b"x" * 1024, "audio/wav")},
    )
    assert response.status_code == 400
    assert "Maximum retry limit" in response.json()["detail"]
