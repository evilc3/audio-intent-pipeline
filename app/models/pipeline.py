"""Pipeline-level Pydantic schemas: job state and final combined output."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.llm import LLMOutput
from app.models.stt import STTOutput


class JobState(BaseModel):
    job_id: str
    trace_id: str
    status: str  # see state machine in docs
    failed_at_stage: Optional[str] = None
    reason: Optional[str] = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime
    # Retained for final output assembly
    stt_output: Optional[STTOutput] = None
    llm_output: Optional[LLMOutput] = None


class PipelineMetadata(BaseModel):
    total_latency_ms: float
    stt_latency_ms: float
    llm_latency_ms: float
    estimated_cost_usd: float
    retries: int
    created_at: datetime
    completed_at: datetime


class STTMetadata(BaseModel):
    transcript: str
    language: str
    confidence: float
    duration_seconds: float
    status: str


class FinalResult(BaseModel):
    intent: Optional[str]
    confidence: Optional[float]
    action: Optional[str]
    reasoning: Optional[str]
    status: str
    model_used: Optional[str]
    fallback_triggered: bool
    prompt_version: str


class FinalOutput(BaseModel):
    job_id: str
    trace_id: str
    status: str
    result: FinalResult
    metadata: dict  # contains "stt" and "pipeline" sub-dicts


# --- API response models ---


class JobAcceptedResponse(BaseModel):
    job_id: str
    trace_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    trace_id: str
    status: str
    failed_at_stage: Optional[str] = None
    reason: Optional[str] = None
    result: Optional[dict] = None
    metadata: Optional[dict] = None


class RetryResponse(BaseModel):
    job_id: str
    status: str
    stage: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
