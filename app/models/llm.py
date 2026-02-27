"""LLM layer Pydantic schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class IntentResponse(BaseModel):
    """Schema enforced at API level via OpenAI structured outputs."""

    intent: str = Field(description="The classified intent of the user (e.g., billing, account_support, prompt_attack, general_inquiry, out_of_scope, unclear).")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence score between 0.0 and 1.0.")
    action: str = Field(description="The suggested next action for the system based on the intent.")
    reasoning: str = Field(description="Brief explanation or reasoning behind this classification.")


class LLMOutput(BaseModel):
    """Full LLM stage output including operational metadata."""

    intent: Optional[str] = None
    confidence: Optional[float] = None
    action: Optional[str] = None
    reasoning: Optional[str] = None
    status: str  # "ok" | "low_confidence" | "failed"
    model_used: Optional[str] = None
    fallback_triggered: bool = False
    prompt_version: str = "v1.0"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    retry_count: int = 0
    latency_ms: float = 0.0
