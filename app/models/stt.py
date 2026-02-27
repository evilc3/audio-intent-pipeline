"""STT layer Pydantic schemas."""

from pydantic import BaseModel, Field


class STTRequest(BaseModel):
    language: str = "en"
    keywords: list[str] = Field(default_factory=list)
    punctuate: bool = True


class STTOutput(BaseModel):
    transcript: str
    language: str
    confidence: float = Field(ge=0.0, le=1.0)
    duration_seconds: float
    status: str  # "ok" | "low_confidence" | "failed"
    fallback_triggered: bool = False
    retry_count: int = 0
    latency_ms: float = 0.0
