"""
LLM reasoning service: GPT-4.1 via Azure OpenAI (primary) + GPT-4.1-mini (fallback).

Input: STT output (transcript + language + confidence).
Output: Structured intent classification with action recommendation.
"""

import asyncio
import re
import time
from typing import Optional, cast

from openai import AsyncAzureOpenAI
from google import genai
from google.genai import types

from app.config import settings
from app.models.llm import IntentResponse, LLMOutput
from app.models.stt import STTOutput
from app.utils.logging import get_logger
from app.prompts.intent_classification import build_user_prompt, build_system_prompt

log = get_logger(__name__)



class LLMFailedException(Exception):
    """Raised when both primary and fallback LLM calls fail."""

    pass


async def classify_intent(stt_output: STTOutput, trace_id: str) -> LLMOutput:
    """
    Classify user intent from a transcribed utterance.

    Validates and sanitises input, then calls GPT-4.1 (primary).
    Falls back to GPT-4.1-mini if the primary exhausts retries.
    """
    _validate_stt_input(stt_output, trace_id)
    transcript = _sanitise_transcript(stt_output.transcript, trace_id)

    system_prompt = build_system_prompt(settings.prompt_version)
    user_prompt = build_user_prompt(
        version=settings.prompt_version,
        transcript=transcript,
        language=stt_output.language,
        stt_confidence=stt_output.confidence,
    )

    # Try models based on backend
    if settings.llm_backend == "gemini":

        log.info("llm_backend_selected", trace_id=trace_id, backend="gemini")

        # Try primary Gemini
        output, retry_count = await _try_gemini_model(
            model_name=settings.gemini_model_primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_id=trace_id,
        )
        if output is not None:
            return output

        # Gemini primary exhausted — fall back
        log.warning("llm_fallback_triggered", trace_id=trace_id, reason="gemini_primary_exhausted", backend="gemini")
        output, _ = await _try_gemini_model(
            model_name=settings.gemini_model_fallback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_id=trace_id,
            is_fallback=True,
            base_retry_count=retry_count,
        )
        if output is not None:
            return output
        
        raise LLMFailedException(f"Gemini models failed after retries: {settings.gemini_model_primary}")

    else:

        log.info("llm_backend_selected", trace_id=trace_id, backend="azure")

        # Try primary Azure OpenAI
        output, retry_count = await _try_model(
            model_name=settings.azure_openai_deployment_primary,
            deployment=settings.azure_openai_deployment_primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_id=trace_id,
            stt_confidence=stt_output.confidence,
        )
        if output is not None:
            return output

        # Primary exhausted — fall back to mini
        log.warning("llm_fallback_triggered", trace_id=trace_id, reason="primary_exhausted_retries", backend="azure")
        output, _ = await _try_model(
            model_name=settings.azure_openai_deployment_fallback,
            deployment=settings.azure_openai_deployment_fallback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_id=trace_id,
            stt_confidence=stt_output.confidence,
            is_fallback=True,
            base_retry_count=retry_count,
        )
        if output is not None:
            return output

        raise LLMFailedException("Both Azure OpenAI primary and fallback failed after retries")


# --- Internal helpers ---


def _validate_stt_input(stt_output: STTOutput, trace_id: str) -> None:
    """Prevent LLM call when STT output is clearly unusable."""
    if stt_output.status == "failed":
        raise LLMFailedException(
            "STT output status is 'failed' — cannot classify intent on empty/failed transcript"
        )
    if not stt_output.transcript.strip():
        raise LLMFailedException("Transcript is empty — cannot classify intent")
    if stt_output.language not in settings.supported_languages:
        log.warning(
            "llm_unsupported_language",
            trace_id=trace_id,
            language=stt_output.language,
        )
        # Do not block — GPT-4.1 handles many languages; flag but continue


def _sanitise_transcript(transcript: str, trace_id: str) -> str:
    """
    Strip content that could manipulate the prompt (injection protection).
    Enforce max length from settings.
    """
    # Remove zero-width chars, unusual unicode control characters
    cleaned = transcript.strip()
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)

    # Truncate at max length to bound token usage
    if len(cleaned) > settings.max_transcript_length:
        log.warning(
            "transcript_truncated",
            trace_id=trace_id,
            original_length=len(cleaned),
            max_length=settings.max_transcript_length,
        )
        cleaned = cleaned[: settings.max_transcript_length]

    return cleaned


async def _try_model(
    model_name: str,
    deployment: str,
    system_prompt: str,
    user_prompt: str,
    trace_id: str,
    stt_confidence: float,
    is_fallback: bool = False,
    base_retry_count: int = 0,
) -> tuple[Optional[LLMOutput], int]:
    """
    Attempt LLM call with exponential backoff.
    Returns (LLMOutput, total_retry_count) on success, (None, retry_count) on failure.
    """

    client = AsyncAzureOpenAI(
        api_key=settings.azure_openai_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version="2024-06-01",
    )

    last_exc: Optional[Exception] = None
    for attempt in range(settings.llm_max_retries):
        try:
            t_start = time.monotonic()
            async with asyncio.timeout(settings.llm_timeout_seconds):
                response = await client.beta.chat.completions.parse(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=IntentResponse,
                )
            latency_ms = (time.monotonic() - t_start) * 1000

            parsed = response.choices[0].message.parsed
            usage = response.usage

            if parsed is None or usage is None:
                log.error(
                    "llm_parse_failed",
                    trace_id=trace_id,
                    model=model_name,
                    reason="Model returned unparseable or empty response",
                )
                continue

            # Per-token cost depends on which model answered
            cost = _estimate_cost(model_name, usage.prompt_tokens, usage.completion_tokens)

            # Flag low LLM confidence
            status = "ok"
            if parsed.confidence < settings.llm_confidence_threshold:
                status = "low_confidence"
                log.warning(
                    "llm_low_confidence",
                    trace_id=trace_id,
                    model=model_name,
                    confidence=parsed.confidence,
                    threshold=settings.llm_confidence_threshold,
                )

            log.info(
                "llm_success",
                trace_id=trace_id,
                model=model_name,
                fallback_triggered=is_fallback,
                attempt=attempt,
                intent=parsed.intent,
                confidence=parsed.confidence,
                latency_ms=round(latency_ms, 1),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                estimated_cost_usd=cost,
                prompt_version=settings.prompt_version,
            )

            return (
                LLMOutput(
                    intent=parsed.intent,
                    confidence=parsed.confidence,
                    action=parsed.action,
                    notes=parsed.notes,
                    status=status,
                    model_used=model_name,
                    fallback_triggered=is_fallback,
                    prompt_version=settings.prompt_version,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    estimated_cost_usd=cost,
                    retry_count=base_retry_count + attempt,
                    latency_ms=round(latency_ms, 1),
                ),
                base_retry_count + attempt,
            )

        except Exception as exc:
            last_exc = exc
            wait = 2**attempt  # 1s, 2s, 4s
            log.warning(
                "llm_retry",
                trace_id=trace_id,
                model=model_name,
                attempt=attempt + 1,
                error=str(exc),
                retry_delay_seconds=wait,
            )
            await asyncio.sleep(wait)

    log.error(
        "llm_model_failed",
        trace_id=trace_id,
        model=model_name,
        attempts=settings.llm_max_retries,
        error=str(last_exc),
    )
    return None, base_retry_count + settings.llm_max_retries


async def _try_gemini_model(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    trace_id: str,
    is_fallback: bool = False,
    base_retry_count: int = 0,
) -> tuple[Optional[LLMOutput], int]:
    """Attempt Gemini call with exponential backoff."""
    client = genai.Client(api_key=settings.gemini_api_key)
    
    last_exc: Optional[Exception] = None
    for attempt in range(settings.llm_max_retries):
        try:
            t_start = time.monotonic()
            # The google-genai SDK generate_content is synchronous by default in many versions, 
            # but we can wrap it or use async if supported. 
            # Using loop.run_in_executor for safety if it's blocking.
            loop = asyncio.get_event_loop()
            
            def call_gemini():
                return client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=IntentResponse,
                        temperature=settings.temperature
                    )   
                )

            async with asyncio.timeout(settings.llm_timeout_seconds):
                response = await loop.run_in_executor(None, call_gemini)
            
            latency_ms = (time.monotonic() - t_start) * 1000
            
            # Use casting to satisfy type checker for structured outputs
            parsed = cast(IntentResponse, response.parsed) if response.parsed else None
            
            # Safely extract usage metadata if available
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens: int = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens: int = getattr(usage, "candidates_token_count", 0) or 0

            if parsed is None:
                log.error("llm_parse_failed", trace_id=trace_id, model=model_name, backend="gemini")
                continue

            cost = _estimate_cost(f"gemini-{model_name}", prompt_tokens, completion_tokens)
            
            status = "ok"
            if parsed.confidence < settings.llm_confidence_threshold:
                status = "low_confidence"

            log.info(
                "llm_success",
                trace_id=trace_id,
                model=model_name,
                backend="gemini",
                fallback_triggered=is_fallback,
                intent=parsed.intent,
                latency_ms=round(latency_ms, 1),
            )

            return (
                LLMOutput(
                    intent=parsed.intent,
                    confidence=parsed.confidence,
                    action=parsed.action,
                    notes=parsed.notes,
                    status=status,
                    model_used=model_name,
                    fallback_triggered=is_fallback,
                    prompt_version=settings.prompt_version,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=cost,
                    retry_count=base_retry_count + attempt,
                    latency_ms=round(latency_ms, 1),
                ),
                base_retry_count + attempt,
            )

        except Exception as exc:
            last_exc = exc
            wait = 2**attempt
            log.warning("llm_retry", trace_id=trace_id, model=model_name, backend="gemini", error=str(exc))
            await asyncio.sleep(wait)

    return None, base_retry_count + settings.llm_max_retries


def _estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    if "gemini" in model_name:
        return (
            prompt_tokens * settings.gemini_cost_per_input_token
            + completion_tokens * settings.gemini_cost_per_output_token
        )
    if "mini" in model_name:
        return (
            prompt_tokens * settings.gpt41_mini_cost_per_input_token
            + completion_tokens * settings.gpt41_mini_cost_per_output_token
        )
    return (
        prompt_tokens * settings.gpt41_cost_per_input_token
        + completion_tokens * settings.gpt41_cost_per_output_token
    )
