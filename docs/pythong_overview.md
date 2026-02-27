# voice-pipeline — Project Design Requirements

## 1. Project Overview

A minimal but production-aware AI pipeline that simulates the core of a voice agent. Accepts an audio file, transcribes it via STT, classifies the user's intent via an LLM, and returns a structured machine-readable response. Designed as a batch pipeline for the homework but architecturally aware of the real-time system it would become in production.

---

## 2. Architecture Overview

```
POST /analyze-audio (audio file)
        ↓
[Validation Layer]         ← reject early, no API calls wasted
        ↓
[Generate trace_id / job_id]
        ↓
[Background Job Starts]
        ↓
Stage 1: STT               ← Deepgram Nova (primary), Whisper (fallback)
        ↓ checkpoint saved: stt_{trace_id}.json
Stage 2: LLM Reasoning     ← GPT-4.1 (primary), GPT-4.1-mini (fallback)
        ↓ checkpoint saved: llm_{trace_id}.json
Stage 3: Orchestration     ← combine, validate, produce final output
        ↓ checkpoint saved: final_{trace_id}.json
        ↓
GET /result/{job_id}       ← client polls for result
```

Each stage is independently retryable. A failed job can be resumed from its last successful checkpoint via `POST /retry/{job_id}` without reprocessing earlier stages.

---

## 3. Tech Stack & Decisions

| Concern | Choice | Reason |
|---|---|---|
| API framework | FastAPI | Async-first, Pydantic native, production standard for AI APIs |
| STT primary | Deepgram Nova | Low latency, conversational accuracy, Italian + English, likely production stack |
| STT fallback | Whisper (local) | Free, independent, no second paid API dependency |
| LLM primary | GPT-4.1 via Azure OpenAI | SOC2 compliant, data security, best accuracy |
| LLM fallback | GPT-4.1-mini via Azure OpenAI | Cheaper, faster, same structured output support |
| Structured output | OpenAI structured outputs + Pydantic | Schema enforced at API level, no fragile JSON parsing |
| Config management | Pydantic Settings | Type-validated config, reads from env vars, FastAPI native |
| Logging | structlog | JSON structured logs, machine readable, trace_id on every line |
| Dependency management | UV | Rust-based, 10-100x faster than pip/poetry, modern standard |
| HTTP client | httpx | Async native, replaces requests for async codebases |
| Testing | pytest + pytest-asyncio | Async test support, regression test suite |
| Linting | ruff | Modern fast linter, replaces flake8 |
| Formatting | black | Opinionated, zero config style enforcement |
| Checkpoint storage | Local disk (homework) → S3 (production) | S3 is purpose-built for file storage, cheaper than Redis at scale |
| Job state storage | In-memory dict (homework) → Redis (production) | Fast ephemeral state, cross-instance sharing |

---

## 4. Project Structure

```
voice-pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py                  ← FastAPI entry point, middleware, global handlers
│   ├── config.py                ← Pydantic Settings, all constants in one place
│   ├── models/
│   │   ├── __init__.py
│   │   ├── stt.py               ← STT input/output Pydantic schemas
│   │   ├── llm.py               ← LLM input/output Pydantic schemas
│   │   └── pipeline.py          ← Job state, final output schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio.py             ← validation, preprocessing
│   │   ├── stt.py               ← Deepgram + Whisper logic
│   │   ├── llm.py               ← GPT-4.1 + fallback logic
│   │   └── orchestrator.py      ← pipeline control flow, checkpointing
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            ← all route handlers
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           ← structlog setup
│       └── checkpoint.py        ← checkpoint read/write/cleanup
├── tests/
│   ├── __init__.py
│   ├── test_stt.py
│   ├── test_llm.py
│   ├── test_pipeline.py
│   └── regression/
│       └── test_cases.py        ← known transcript → expected intent pairs
├── checkpoints/                 ← local disk checkpoint storage
├── .env                         ← never committed
├── .env.example                 ← committed, shows required vars without values
├── .gitignore
├── pyproject.toml               ← UV dependency management
├── README.md
└── REQUIREMENTS.md              ← this document
```

---

## 5. Configuration Management

All constants live in one `config.py`. No magic numbers scattered across the codebase. Tuning the system means changing one file only.

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # Audio validation
    max_file_size_bytes: int = 10 * 1024 * 1024   # 10MB
    min_file_size_bytes: int = 10 * 1024            # 10KB
    max_audio_duration_seconds: int = 300           # 5 minutes
    min_audio_duration_seconds: int = 1
    max_transcript_length: int = 1000
    supported_extensions: list[str] = ["wav", "mp3"]
    supported_languages: list[str] = ["en", "it"]
    default_language: str = "en-US"

    # STT
    stt_confidence_threshold: float = 0.6
    stt_max_retries: int = 3
    stt_retry_base_delay_seconds: float = 1.0

    # LLM
    llm_confidence_threshold: float = 0.5
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 10
    prompt_version: str = "v1.0"
    supported_intents: list[str] = [
        "account_support",
        "billing",
        "technical_support",
        "general_inquiry",
        "out_of_scope",
        "unclear"
    ]

    # Orchestration
    job_ttl_seconds: int = 3600               # 1 hour
    max_manual_retries: int = 3
    checkpoint_dir: str = "./checkpoints"
    checkpoint_ttl_seconds: int = 3600

    # API keys — loaded from environment, never hardcoded
    deepgram_api_key: str
    azure_openai_key: str
    azure_openai_endpoint: str

    # Cost tracking (USD per token)
    gpt41_cost_per_input_token: float = 0.000002
    gpt41_cost_per_output_token: float = 0.000008
    gpt41_mini_cost_per_input_token: float = 0.0000001
    gpt41_mini_cost_per_output_token: float = 0.0000004

    class Config:
        env_file = ".env"

settings = Settings()
```

`.env.example` committed to repo:

```
DEEPGRAM_API_KEY=your_key_here
AZURE_OPENAI_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
```

---

## 6. API Endpoints

All routes versioned under `/v1/`.

### POST /v1/analyze-audio

Accepts audio file. Validates. Generates trace_id. Kicks off background processing. Returns immediately.

**Request:** multipart form data
- `file` — audio file (.wav or .mp3)
- `language` — optional, "en" or "it" (default: auto-detect)

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "received"
}
```

---

### GET /v1/result/{job_id}

Client polls this for result. Returns current pipeline state.

**Response (processing):**
```json
{
  "job_id": "uuid",
  "trace_id": "uuid",
  "status": "llm_processing"
}
```

**Response (complete):**
```json
{
  "job_id": "uuid",
  "trace_id": "uuid",
  "status": "done",
  "result": {
    "intent": "account_support",
    "confidence": 0.87,
    "action": "redirect_to_password_reset",
    "notes": "User is unable to access their account",
    "status": "ok",
    "model_used": "gpt-4.1",
    "fallback_triggered": false,
    "prompt_version": "v1.0"
  },
  "metadata": {
    "stt": {
      "transcript": "I need help resetting my password",
      "language": "en",
      "confidence": 0.92,
      "duration_seconds": 12.4,
      "status": "ok"
    },
    "pipeline": {
      "total_latency_ms": 2100,
      "stt_latency_ms": 1240,
      "llm_latency_ms": 820,
      "estimated_cost_usd": 0.0032,
      "retries": 0,
      "created_at": "2024-01-01T12:00:00Z",
      "completed_at": "2024-01-01T12:00:02Z"
    }
  }
}
```

**Response (failed):**
```json
{
  "job_id": "uuid",
  "trace_id": "uuid",
  "status": "failed",
  "failed_at_stage": "llm",
  "reason": "LLM processing failed after retries"
}
```

---

### POST /v1/retry/{job_id}

Triggers manual retry from last successful checkpoint. Only accepted if job is in `failed` state. Guards against concurrent retries.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "retrying_from",
  "stage": "llm"
}
```

---

### GET /health

```json
{
  "status": "ok",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## 7. Pipeline State Machine

```
received → validating → stt_processing → stt_complete
→ llm_processing → llm_complete → done → failed
```

Every state transition is logged with timestamp and trace_id. Polling endpoint always returns current state so caller knows exactly where their job is.

Failed state includes which stage failed and why, enabling clean partial retry.

---

## 8. Stage 1 — STT Layer

### 8.1 Input Audio Validation

Runs before any API calls. Reject early and cheaply.

Checks in order:
1. File extension is .wav or .mp3
2. Real MIME type matches audio — use `python-magic` to check actual bytes, not just extension
3. File size between `min_file_size_bytes` and `max_file_size_bytes`
4. Audio duration between `min_audio_duration_seconds` and `max_audio_duration_seconds` — use `pydub`

Temp file cleanup via `try/finally` — guaranteed even on exception:

```python
try:
    result = process(temp_path)
finally:
    os.remove(temp_path)
```

### 8.2 Audio Preprocessing

Before sending to Deepgram, normalise via `pydub`:
- Convert stereo to mono
- Standardise sample rate to 16kHz (STT standard)

### 8.3 STT Processing

**Primary — Deepgram Nova**

- Async Python SDK throughout
- Supports auto language detection or explicit language parameter
- Optional hotwords/keywords passed to improve accuracy on domain-specific terms (names, places)
- Punctuation enabled by default
- Dialect support — default `en-US`

**Retry with exponential backoff before fallback:**

```python
for attempt in range(settings.stt_max_retries):
    try:
        result = await call_deepgram(audio)
        break
    except Exception:
        await asyncio.sleep(settings.stt_retry_base_delay_seconds ** attempt)
else:
    # trigger Whisper fallback
```

**Fallback — Whisper (local)**

- Free, no network dependency
- Handles Italian well
- Triggered only after Deepgram exhausts retries

### 8.4 STT Output Contract

```json
{
  "transcript": "I need help resetting my password",
  "language": "en",
  "confidence": 0.92,
  "duration_seconds": 12.4,
  "status": "ok"
}
```

`status` values: `ok`, `low_confidence`, `failed`

If `confidence < stt_confidence_threshold` → `status: low_confidence`. Do not discard — pass downstream with flag.

---

## 9. Stage 2 — LLM Reasoning Layer

### 9.1 Input Validation

- Check STT `status` — if `failed`, do not call LLM, propagate failure
- Check transcript is not empty or whitespace
- Check language is supported

### 9.2 Input Sanitisation

Before injecting transcript into prompt:
- Strip unusual characters
- Enforce `max_transcript_length`
- Protect against prompt injection attacks — a real security concern for voice agents

### 9.3 Prompt Structure

**Prompt version:** tracked as `PROMPT_VERSION = "v1.0"` in config. Logged and included in every output. Enables regression tracking across prompt changes.

**System prompt:**

```
You are an intent classification system for a voice agent.
Your job is to analyse a user transcript and return a structured response.

Rules:
- Supported intents: account_support, billing, technical_support,
  general_inquiry, out_of_scope, unclear
- Use "unclear" if you cannot determine intent with reasonable certainty
- Use "out_of_scope" if the request is clear but outside system capabilities
- Confidence must be a float between 0 and 1
- Always suggest a concrete next action based on the intent
- Input may be English or Italian — handle both
- Factor STT confidence into your own confidence score
```

**User prompt:**

```
Transcript: {transcript}
Language: {language}
STT Confidence: {stt_confidence}

Classify the intent and suggest an action.
```

**Language handling:** System prompt in English regardless of transcript language. GPT-4.1 handles both reliably. Production would evaluate language-specific prompts.

### 9.4 Structured Output Schema

OpenAI structured outputs enforces schema at API level — not a prompt suggestion. Guarantees valid schema every time.

```python
from pydantic import BaseModel, Field

class IntentResponse(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    action: str
    notes: str
    status: str
    prompt_version: str
```

Called via `.parse()`:

```python
response = await client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[...],
    response_format=IntentResponse
)
result = response.choices[0].message.parsed
```

### 9.5 Confidence Score

Self-reported by the LLM in structured output. Pragmatic for homework. Production would use OpenAI logprobs for statistically rigorous token-level confidence scoring.

If `confidence < llm_confidence_threshold` → `status: low_confidence`.

### 9.6 Primary & Fallback Models

**Primary — GPT-4.1 via Azure OpenAI**

Azure OpenAI chosen for SOC2 compliance and data security — important for a voice agent handling user data.

**Fallback — GPT-4.1-mini**

Same prompt, same structured output schema. Cheaper and faster. Only triggered after primary exhausts retries.

### 9.7 Retry with Exponential Backoff

```python
for attempt in range(settings.llm_max_retries):
    try:
        async with asyncio.timeout(settings.llm_timeout_seconds):
            result = await call_llm(model="gpt-4.1")
        break
    except Exception:
        await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
else:
    # trigger gpt-4.1-mini fallback
```

Explicit timeout on every LLM call prevents hanging jobs.

### 9.8 Failure Cases

| Case | Handling |
|---|---|
| Primary fails after retries | Fallback to GPT-4.1-mini with same retry logic |
| Unparseable response | try/except on parse, treat as failed |
| Intent is "unclear" | Valid outcome, status: ok, confidence will be low |
| Confidence below threshold | status: low_confidence, pass downstream with flag |
| Both models fail | Return clean structured failure, never bubble raw exception |

### 9.9 Token Usage & Cost Tracking

OpenAI returns token usage in response object at no extra cost:

```python
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens
estimated_cost = (
    prompt_tokens * settings.gpt41_cost_per_input_token +
    completion_tokens * settings.gpt41_cost_per_output_token
)
```

### 9.10 LLM Output Contract

```json
{
  "intent": "account_support",
  "confidence": 0.87,
  "action": "redirect_to_password_reset",
  "notes": "User is unable to access their account",
  "status": "ok",
  "model_used": "gpt-4.1",
  "fallback_triggered": false,
  "prompt_version": "v1.0"
}
```

---

## 10. Stage 3 — Orchestration Layer

### 10.1 Control Flow

```
Entry point
    ↓
Generate trace_id (uuid4)
    ↓
state → "validating"
    ↓
Run audio validation layer
    ↓ fail → HTTP 400, no job created
    ↓ pass
state → "stt_processing"
    ↓
Run STT (with retry + fallback)
    ↓ fail → state: failed, failed_at_stage: "stt"
    ↓ pass
Save checkpoint: stt_{trace_id}.json
state → "stt_complete"
    ↓
Validate STT output
    ↓ fail → state: failed, failed_at_stage: "stt"
    ↓ pass
state → "llm_processing"
    ↓
Run LLM (with retry + fallback)
    ↓ fail → state: failed, failed_at_stage: "llm"
    ↓ pass
Save checkpoint: llm_{trace_id}.json
state → "llm_complete"
    ↓
Validate LLM output
    ↓ fail → state: failed, failed_at_stage: "llm"
    ↓ pass
Combine STT + LLM → final output
Save checkpoint: final_{trace_id}.json
state → "done"
    ↓
Cleanup checkpoint files (try/finally)
    ↓
TTS stub (see section 10.6)
```

### 10.2 Intermediate Validation

Validate output at each stage boundary before proceeding. Catches silent failures.

```python
def validate_stt_output(output: STTOutput) -> bool:
    if output.status == "failed":
        return False
    if not output.transcript.strip():
        return False
    if output.language not in settings.supported_languages:
        return False
    return True

def validate_llm_output(output: IntentResponse) -> bool:
    if output.status == "failed":
        return False
    if output.intent not in settings.supported_intents:
        return False
    if not 0.0 <= output.confidence <= 1.0:
        return False
    if not output.action.strip():
        return False
    return True
```

### 10.3 Checkpoint Storage

Files named `{stage}_{trace_id}.json`. Written to `./checkpoints/` on local disk.

```python
def save_checkpoint(trace_id: str, stage: str, data: dict) -> None:
    path = f"{settings.checkpoint_dir}/{stage}_{trace_id}.json"
    with open(path, "w") as f:
        json.dump(data, f)

def load_checkpoint(trace_id: str, stage: str) -> dict:
    path = f"{settings.checkpoint_dir}/{stage}_{trace_id}.json"
    with open(path, "r") as f:
        return json.load(f)
```

Cleanup always runs via `try/finally` after job completes or TTL expires.

### 10.4 Partial Retry Logic

`POST /retry/{job_id}` checks `failed_at_stage`, loads last successful checkpoint, resumes from that stage only. Does not reprocess earlier stages.

**Concurrent retry guard:**

```python
if job.status in ["retrying", "stt_processing", "llm_processing"]:
    raise HTTPException(400, "Job is already being processed")
```

State set to `retrying` immediately on acceptance before background task starts.

**Maximum manual retry limit:**

```python
if job.retry_count >= settings.max_manual_retries:
    raise HTTPException(400, "Maximum retry limit reached — job permanently failed")
```

### 10.5 Job Expiry / TTL

Jobs expire after `job_ttl_seconds`. TTL check on every poll:

```python
def is_expired(job: JobState) -> bool:
    age = (datetime.utcnow() - job.created_at).seconds
    return age > settings.job_ttl_seconds
```

Expired jobs return 404.

### 10.6 TTS Stub

Clearly marked stub after final output. No audio generation. Shows where TTS fits in the pipeline:

```python
async def tts_stub(text: str, language: str) -> dict:
    """
    Stub for TTS step.
    Production: integrate Azure TTS, Google TTS, or ElevenLabs.
    Returns structured stub response — no audio generated.
    """
    return {
        "status": "stub",
        "text": text,
        "language": language,
        "audio_url": None,
        "note": "TTS not implemented — stub only"
    }
```

---

## 11. Pydantic Models

All internal data structures use Pydantic. Not just API schemas — every intermediate output is a typed model. Validation throughout the pipeline, not just at the edges.

```python
# models/pipeline.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class JobState(BaseModel):
    job_id: str
    trace_id: str
    status: str
    failed_at_stage: Optional[str] = None
    reason: Optional[str] = None
    retry_count: int = 0
    created_at: datetime
    updated_at: datetime
```

---

## 12. Custom Exceptions

Domain-specific exceptions. Never catch bare `Exception` without re-raising or logging.

```python
class AudioValidationError(Exception):
    pass

class STTFailedException(Exception):
    pass

class LLMFailedException(Exception):
    pass

class CheckpointError(Exception):
    pass
```

---

## 13. Logging & Metrics

### Structured Logging

`structlog` for JSON logs. Every log entry includes `trace_id`.

```python
import structlog
log = structlog.get_logger()

log.info("stage_complete",
    trace_id=trace_id,
    stage="stt",
    latency_ms=1240,
    confidence=0.92,
    status="ok"
)
```

### Metrics Tracked Per Stage

**STT:**
- Latency (ms)
- Confidence score
- Language detection confidence
- Fallback triggered (bool)
- Retry count
- Transcript length vs audio duration

**LLM:**
- Latency (ms)
- Prompt tokens
- Completion tokens
- Estimated cost (USD)
- Model used (primary or fallback)
- Fallback triggered (bool)
- Retry count
- Intent classified
- Confidence score

**Pipeline:**
- Total end-to-end latency (ms)
- Per-stage latency breakdown
- Total estimated cost (USD)
- Final status

### Log Structure

```json
{
  "timestamp": "2024-01-01T12:00:01Z",
  "trace_id": "uuid",
  "job_id": "uuid",
  "stage": "llm",
  "event": "stage_complete",
  "model_used": "gpt-4.1",
  "fallback_triggered": false,
  "status": "ok",
  "latency_ms": 820,
  "prompt_tokens": 210,
  "completion_tokens": 85,
  "estimated_cost_usd": 0.0032,
  "intent": "account_support",
  "confidence": 0.87,
  "retries": 0,
  "prompt_version": "v1.0"
}
```

---

## 14. Testing Strategy

### Unit Tests

Per-service tests covering:
- Audio validation edge cases (too small, too large, wrong MIME type, zero duration)
- STT fallback trigger logic
- LLM prompt sanitisation
- Confidence threshold flagging
- Checkpoint read/write/cleanup

### Integration Tests

Full pipeline run with a real audio file. Assert final output schema and expected intent.

### Regression Test Suite

Known transcript → expected intent pairs. Run after any prompt change. Tag results with `prompt_version` to track which version caused a regression.

```python
# tests/regression/test_cases.py
TEST_CASES = [
    {
        "transcript": "I need help resetting my password",
        "expected_intent": "account_support"
    },
    {
        "transcript": "I want to cancel my subscription",
        "expected_intent": "billing"
    },
    {
        "transcript": "My internet is not working",
        "expected_intent": "technical_support"
    },
    {
        "transcript": "What is the weather today",
        "expected_intent": "out_of_scope"
    },
    {
        "transcript": "asdfgh jkl mmmm",
        "expected_intent": "unclear"
    }
]
```

---

## 15. FastAPI Application Setup

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="voice-pipeline", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global exception handler — never leak stack traces to caller
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
```

---

## 16. Python Standards

| Standard | Tool/Approach |
|---|---|
| Dependency management | UV + pyproject.toml |
| Config management | Pydantic Settings |
| Data validation | Pydantic v2 models everywhere |
| Type hints | Full type hints on all functions |
| Async | async/await throughout, no sync/async mixing |
| HTTP client | httpx (async native) |
| Logging | structlog (JSON structured) |
| Linting | ruff |
| Formatting | black |
| Testing | pytest + pytest-asyncio |
| Secrets | Environment variables via .env, never hardcoded |
| Virtualenv | Managed by UV |

---

## 17. Known Limitations

- BackgroundTasks is in-process — jobs lost if server restarts mid-processing
- In-memory job state — lost on restart, not shared across instances
- Local disk checkpoints — not durable across restarts or deployments
- Self-reported LLM confidence — not statistically rigorous
- Supported intents hardcoded — requires code change to extend
- Rate limiting is basic — no proper gateway-level enforcement
- No idempotency — duplicate submissions create separate jobs
- No graceful shutdown handling for in-flight jobs
- TTS is a stub only

---

## 18. Future Improvements

- **Celery + Redis** — replace BackgroundTasks for job persistence, retries, and independent worker scaling
- **S3 checkpoints** — durable cross-service checkpoint storage
- **Redis job state** — persistent, cross-instance job state
- **Logprobs confidence** — replace self-reported LLM confidence with OpenAI logprobs
- **OpenAI Batch API** — 50% cost reduction for non-realtime bulk processing
- **Real-time streaming** — replace batch file input with real-time audio stream
- **Celery + Redis retry** — proper retry limits, dead letter queues
- **Idempotency keys** — deduplicate concurrent identical submissions
- **Prompt management system** — version, deploy, and rollback prompts independently
- **Config-driven intents** — load supported intents from config file or database
- **Cost alerting** — alert when cost per request exceeds threshold
- **Real TTS integration** — Azure TTS, Google TTS, or ElevenLabs to complete the voice loop
- **Diarisation** — speaker separation for multi-speaker audio
- **Word-level timestamps** — when each word was spoken for downstream use
- **Custom vocabulary** — domain-specific language model for STT accuracy
- **Gateway rate limiting** — proper IP-level rate limiting at infrastructure layer
- **Graceful shutdown** — drain in-flight jobs before process exit