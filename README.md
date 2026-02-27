# audio-intent-pipeline

![Demo](docs/demo.gif)

> Submit an audio file. Get back structured intent. — A production-aware AI voice pipeline that transcribes speech, classifies user intent, and returns a machine-readable JSON response ready for downstream action.

---

## Quick Start

### Option 1 — Docker (Recommended)

No Python setup required. One command runs everything.

```bash
git clone https://github.com/your-username/audio-intent-pipeline
cd audio-intent-pipeline
cp .env.example .env        # add your API keys
docker compose up --build
```

> [!IMPORTANT]
> **Demo API Keys:** For ease of testing, the `.env` includes personal API keys for Deepgram and Gemini. These are provided for demonstration purposes only. Please replace them with your own keys for any extended use.

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Interactive Docs | http://localhost:8000/docs |
| UI Demo | http://localhost:8501 |
 
### Option 2 — Manual

**Prerequisites:** Python 3.13+, [UV](https://docs.astral.sh/uv/), `ffmpeg`, `libmagic`

```bash
git clone https://github.com/your-username/audio-intent-pipeline
cd audio-intent-pipeline
uv sync --dev
cp .env.example .env        # add your API keys
uv run uvicorn app.main:app --reload
```

**Note for `libmagic`:** On macOS, run `brew install libmagic`. On Debian/Ubuntu, run `sudo apt-get install libmagic1`.

### Environment Variables

```bash
# Required
DEEPGRAM_API_KEY=           # Deepgram Nova STT
GEMINI_API_KEY=             # Gemini LLM (default backend)

# LLM Backend selection
LLM_BACKEND=gemini          # 'gemini' or 'azure_openai'

# Optional — Azure OpenAI (required if LLM_BACKEND=azure_openai)
AZURE_OPENAI_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT_PRIMARY=gpt-4.1
AZURE_OPENAI_DEPLOYMENT_FALLBACK=gpt-4.1-mini

# Optional — STT Model overrides
DEEPGRAM_MODEL=nova-3       # default is nova-3

# Optional — Langfuse observability (leave empty to disable)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

See `.env.example` for all optional overrides: confidence thresholds, retry counts, model names, TTLs, cost tracking rates.

### Run Tests

```bash
# Unit + integration (fast, no API calls)
uv run pytest tests/

# Regression suite (hits real LLM, costs money — run before prompt changes)
uv run pytest tests/regression/
```

---

## API Reference

### `POST /v1/analyze-audio`

Submit an audio file. Returns immediately with a `job_id` — processing happens in background.

```bash
curl -X POST http://localhost:8000/v1/analyze-audio \
  -F "file=@recording.wav" \
  -F "language=en"
```

```json
{
  "job_id": "cbc8ef12-9ee7-4bbc-a2ea-25238782a257",
  "trace_id": "cbc8ef12-9ee7-4bbc-a2ea-25238782a257",
  "status": "received"
}
```

### `GET /v1/result/{job_id}`

Poll for the result. Returns current state while processing, full output when done.

```bash
curl http://localhost:8000/v1/result/cbc8ef12-9ee7-4bbc-a2ea-25238782a257
```

```json
{
  "job_id": "cbc8ef12-9ee7-4bbc-a2ea-25238782a257",
  "trace_id": "cbc8ef12-9ee7-4bbc-a2ea-25238782a257",
  "status": "done",
  "result": {
    "intent": "billing",
    "confidence": 0.95,
    "action": "Review recent invoice and credit card charges",
    "notes": "User is asking about their recent invoice and credit card charges.",
    "status": "ok",
    "model_used": "gemini-2.5-flash",
    "fallback_triggered": false,
    "prompt_version": "v1.0"
  },
  "metadata": {
    "stt": {
      "transcript": "I have a question about my recent invoice and the charges on my credit card.",
      "language": "en",
      "confidence": 0.998,
      "duration_seconds": 4.1,
      "status": "ok"
    },
    "pipeline": {
      "total_latency_ms": 5389.4,
      "stt_latency_ms": 2921.4,
      "llm_latency_ms": 2345.7,
      "estimated_cost_usd": 0.000031,
      "retries": 0,
      "created_at": "2026-02-27T07:35:31.348577+00:00",
      "completed_at": "2026-02-27T07:35:36.738321+00:00"
    }
  }
}
```

### `POST /v1/retry/{job_id}`

Manually retry a failed job from its last successful checkpoint. Re-upload the audio file.

```bash
curl -X POST http://localhost:8000/v1/retry/cbc8ef12-9ee7-4bbc-a2ea-25238782a257 \
  -F "file=@recording.wav"
```

```json
{"job_id": "...", "status": "retrying_from", "stage": "llm"}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "timestamp": "2026-02-27T07:35:31Z"}
```

### `Streamlit UI Demo`

Run the interactive UI to upload files and visualize results:

```bash
# Requires the API to be running first
uv run streamlit run scripts/ui_demo.py
```

Defaults to `http://localhost:8501`. If your API is on a different port, set `API_BASE_URL`:
`API_BASE_URL=http://localhost:8001 uv run streamlit run scripts/ui_demo.py`

## Supported intents:
  - `account_support`
  - `billing`
  - `technical_support`
  - `general_inquiry`
  - `out_of_scope`
  - `unclear`

## Supported languages:
  - `en` (tested)
  - `it`

---

# Test Audio Files

Testing a voice pipeline without audio is painful. Pre-generated test audio files are included in `tests/test_audio/` — generated using Deepgram TTS so each file contains a realistic human utterance for a specific intent.

| File | Intent | Phrase |
|---|---|---|
| `intent_account_support.wav` | account_support | "I'm having trouble logging into my account, can you help?" |
| `intent_billing.wav` | billing | "I have a question about my recent invoice and the charges on my credit card." |
| `intent_technical_support.wav` | technical_support | "The application is crashing every time I try to upload a file." |
| `intent_general_inquiry.wav` | general_inquiry | "What are your business hours and where are you located?" |
| `intent_out_of_scope.wav` | out_of_scope | "I'd like to order a large pepperoni pizza for delivery." |
| `intent_unclear.wav` | unclear | "Uh... I... well... maybe... I don't know." |
| `no_transcript.wav` | — | 5 seconds of silence (tests empty transcript handling) |
| `too_long.wav` | — | >300 seconds (tests duration validation rejection) |
| `too_short.wav` | — | 0.5 seconds (tests minimum duration rejection) |

Each intent has both `.wav` and `.mp3` versions. You can regenerate them anytime:

```bash
uv run python scripts/generate_test_audio.py
```

**Try them directly against the API:**

```bash
# Happy path — billing intent
curl -X POST http://localhost:8000/v1/analyze-audio \
  -F "file=@tests/test_audio/intent_billing.wav" \
  -F "language=en"

# Out of scope — system should classify and not route to support
curl -X POST http://localhost:8000/v1/analyze-audio \
  -F "file=@tests/test_audio/intent_out_of_scope.wav" \
  -F "language=en"

# Edge case — should return status: failed at validation stage
curl -X POST http://localhost:8000/v1/analyze-audio \
  -F "file=@tests/test_audio/too_short.wav" \
  -F "language=en"

# Then poll for result (replace with your job_id)
curl http://localhost:8000/v1/result/<job_id>
```

---

## Developer Scripts

Utility scripts for debugging individual service connections without running the full pipeline:

- `uv run python scripts/test_stt.py` — Verify Deepgram connection and transcription.
- `uv run python scripts/test_gemini.py` — Verify Gemini connection and structured output.
- `uv run python scripts/test_llm.py` — Verify Azure OpenAI connection and deployments.
- `uv run python scripts/generate_test_audio.py` — Regenerate the test audio fixtures.

---

## Troubleshooting

### `ImportError: failed to find libmagic`
The pipeline uses `python-magic` to verify audio files. You must install the system-level `libmagic` library:
- **macOS:** `brew install libmagic`
- **Linux:** `sudo apt-get install libmagic1`

### `ffmpeg not found`
Used for audio duration and preprocessing.
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

### `401 Unauthorized` (Deepgram/Gemini)
Ensure your `.env` file has valid API keys and is located in the root directory. If using Docker, remember to `docker compose up --build` after changes to `.env`.

### `ResourceExhausted` (Gemini)
You have hit your API quota. Wait a few minutes or switch to a different model in `app/config.py`.

---

## High Level Architecture

```
POST /v1/analyze-audio
        │
        ▼
┌───────────────────┐
│  Validation Layer │  ← reject bad audio before any API call
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Background Job   │  ← returns immediately, job_id issued
│  (trace_id UUID)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Stage 1: STT    │  ← Deepgram Nova (primary)
│                   │     └─ Whisper local (fallback)
│  checkpoint saved │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Stage 2: LLM    │  ← Gemini 2.5 Flash / GPT-4.1 (primary)
│                   │     └─ smaller model (fallback)
│  checkpoint saved │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 3: Output  │  ← assemble, validate, save final result
└────────┬──────────┘
         │
         ▼
GET /v1/result/{job_id}   ← client polls for result
```

Each stage saves a checkpoint to disk. A failed job resumes from its last successful checkpoint via `POST /v1/retry/{job_id}` — **STT is not rerun if LLM failed.**

### **Pipeline state machine:**

```
received → validating → stt_processing → stt_complete
         → llm_processing → llm_complete → done
                                               ↘ failed (with failed_at_stage)
```

---

## Low Level Architecture

### Tech Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.13 | Target environment |
| Package manager | UV | Rust-based, 10-100x faster than pip/Poetry |
| Server | FastAPI + Uvicorn | Async-first, OpenAPI docs built in |
| Config | Pydantic Settings | Type-validated, reads from `.env`, single source of truth |
| Data models | Pydantic v2 | All pipeline I/O as typed schemas — no raw dicts |
| Logging | structlog | JSON structured logs, `trace_id` on every line |
| HTTP client | httpx | Native async, replaces `requests` |
| Testing | pytest + pytest-asyncio | Standard async testing |
| Linting | ruff + black | Fast, consistent |

### Why Async Throughout

FastAPI and all I/O (Deepgram, Gemini, file reads) use `async/await`. While one request waits for Deepgram to respond (~2-3 seconds), the server handles other requests concurrently. Synchronous blocking would serialize all requests.

Whisper is CPU-bound so it runs in a thread pool via `run_in_executor` to keep the event loop free:

```python
result = await asyncio.get_event_loop().run_in_executor(
    None, _call_whisper_sync, audio_bytes, language
)
```

### Retry & Exponential Backoff

Every external API call retries automatically before triggering fallback:

```
attempt 0 → fail → wait 1s
attempt 1 → fail → wait 2s
attempt 2 → fail → wait 4s
                → trigger fallback model
```

Configurable via `stt_max_retries` and `llm_max_retries` in settings.

### Logging & Observability

All logs are JSON-structured via `structlog`. Every log line carries `trace_id` to correlate all events for a single request:

```json
{
  "timestamp": "2026-02-27T07:35:33Z",
  "trace_id": "cbc8ef12-9ee7-4bbc-a2ea-25238782a257",
  "stage": "llm",
  "event": "llm_success",
  "model_used": "gemini-2.5-flash",
  "intent": "billing",
  "confidence": 0.95,
  "latency_ms": 2345.7,
  "prompt_tokens": 210,
  "completion_tokens": 85,
  "estimated_cost_usd": 0.000031,
  "prompt_version": "v1.0"
}
```

### Latency & Cost Tracking

Every run records per-stage and total latency using `time.monotonic()`. Token usage is captured from the LLM response and cost is calculated against configurable per-token rates in `config.py`:

```python
"pipeline": {
    "total_latency_ms": 5389.4,
    "stt_latency_ms": 2921.4,
    "llm_latency_ms": 2345.7,
    "input_tokens_consumed": 210,
    "output_tokens_generated": 85,
    "estimated_cost_usd": 0.000031
}
```

### Error Handling

Custom domain exceptions raised at each layer, caught at orchestration:

```
AudioValidationError  → raised in audio.py
STTFailedException    → raised in stt.py
LLMFailedException    → raised in llm.py
CheckpointError       → raised in storage.py
```

Global FastAPI exception handler ensures raw stack traces never reach callers. Temp files are cleaned via `try/finally` — cleanup guaranteed even on exception.

### Idempotency

Currently not implemented — duplicate submissions create separate jobs. Production would use an `Idempotency-Key` header to return a cached response for duplicate submissions within TTL (see production roadmap).

### Prompt Management

Prompts are separated from service logic in `app/prompts/intent_classification.py`. Each version is a versioned dict entry. Active version is controlled by `prompt_version` in config and logged on every LLM call for regression tracking.

---

## Fallback Handling

The assignment specifically asks about failure handling. Here is every fallback implemented:

### 1. Audio Input Validation

Validated before any API call — cheapest checks first to avoid wasting credits:

```
① File extension must be .wav or .mp3
② Real MIME type verified via python-magic (catches renamed/spoofed files)
③ File size: 10KB minimum, 10MB maximum
④ Audio duration: 1 second minimum, 300 seconds maximum
```

Any failure raises `AudioValidationError` immediately. Zero API calls made.

### 2. STT Fallback — Deepgram → Whisper

```
Deepgram Nova (3 attempts, exponential backoff)
    └─ if exhausted → Whisper local (free, runs locally, no second paid API)
```

`fallback_triggered: true` is set on the STT output so downstream stages and logs know which model produced the transcript.

### 3. Transcript Quality Validation

After STT, before calling LLM:

```
✓ status must not be "failed"
✓ transcript must not be empty or whitespace
✓ language logged as warning if unsupported (pipeline continues — LLM handles many languages)
```

Prevents wasting LLM tokens on unusable transcripts.

### 4. Prompt Injection Protection

Transcripts are sanitised before prompt insertion:

```python
# Strips zero-width chars and unicode control characters
cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", transcript)
# Truncates at configurable max length to bound token usage
cleaned = cleaned[:settings.max_transcript_length]
```

> **Coming soon:** dedicated `prompt_attack` intent to explicitly classify and flag prompt injection attempts in the transcript itself.

### 5. LLM Fallback — Primary → Smaller Model

```
Gemini 2.5 Flash (3 attempts) → Gemini Flash fallback
GPT-4.1 / Azure (3 attempts)  → GPT-4.1-mini
```

Fallback uses the identical output schema — no downstream changes needed.

### 6. Structured Output — Schema Enforced at API Level

Both Gemini and OpenAI are called with structured output mode. The response schema is enforced by the API, not by prompt instruction — this guarantees the response always matches the `IntentResponse` Pydantic model regardless of what the model "wants" to output:

```python
# OpenAI — schema enforced via response_format
response = await client.beta.chat.completions.parse(
    response_format=IntentResponse,  # API-level enforcement
)

# Gemini — schema enforced via response_schema
config=types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=IntentResponse,  # API-level enforcement
)
```

Eliminates fragile JSON parsing from raw text entirely.

### 7. Intent Design — `out_of_scope` and `unclear`

Two special intents handle edge cases:

`unclear` — audio received but intent cannot be determined (poor audio, garbled speech, meaningless input like "asdfgh jkl"). Correct downstream action: ask user to repeat.

`out_of_scope` — intent understood but outside system capabilities ("What is the weather today?"). Correct downstream action: inform user the system cannot help.

These require different downstream handling — conflating them into a generic "unknown" routes incorrectly.

### 8. Checkpoint-Based Retry

Every stage saves its output before proceeding. On manual retry, the pipeline resumes from the last successful checkpoint:

```
Failed at LLM stage:
  → load STT checkpoint (skip re-transcription, saves cost + latency)
  → rerun LLM only

Failed at STT stage:
  → rerun full pipeline from audio
```

Max 3 manual retries per job. Concurrent retry guard prevents duplicate processing.

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| STT primary | Deepgram Nova | Low latency, Italian+English, async SDK |
| STT fallback | Whisper local | Free, independent, no second paid API |
| LLM primary | Gemini 2.5 Flash | Fast, cheap, structured output support |
| LLM alt primary | GPT-4.1 / Azure | SOC2 compliant for data-sensitive deployments |
| LLM fallback | Smaller model | Same output schema, cheaper, faster |
| Structured output | API-level schema | Eliminates JSON parsing fragility |
| Background jobs | FastAPI BackgroundTasks | Simple for homework scope |
| Job state | In-memory dict | Simple for homework scope |
| Checkpoints | Local disk | Simple for homework scope |
| Config | Pydantic Settings | Type-validated, single source of truth |
| Logging | structlog JSON | Machine-parseable, trace_id on every line |
| LLM confidence | Self-reported | Pragmatic; production → OpenAI logprobs |

---

## Production Roadmap

Deliberate homework simplifications with known production replacements:

1. **Background Processing** — FastAPI `BackgroundTasks` loses in-flight jobs on restart. Production: Celery + Redis. Jobs persist, workers scale horizontally, built-in retry scheduling.

2. **Job State & Metadata Storage** — In-memory dict lost on restart, not shared across instances. Production: **Redis + PostgreSQL**. Redis for fast in-flight job state; PostgreSQL for persistent metadata (latency, cost, intent trends) and audit logs.

3. **Blob & JSON Storage (S3)** — Local disk not durable across container restarts. Production: **S3**. Durable storage for raw JSON checkpoints and final result payloads. Cross-service accessible and cost-effective at scale.

4. **Observability** — stdout JSON logs only. Production: Langfuse for LLM-specific tracing (prompt versions, cost per request, intent distribution over time) + Datadog/Grafana for infrastructure metrics.

5. **LLM Confidence** — Self-reported float in structured output. Production: OpenAI logprobs for token-level probability. More reliable signal for routing low-confidence responses to human review.

6. **LLM Guardrails** — Basic prompt sanitisation. Production: dedicated `prompt_attack` intent, input content moderation layer, PII detection before sending to external APIs, rate limiting per caller.

7. **Intent Configuration** — Hardcoded list in settings, requires code change to extend. Production: config file or database-driven, editable without deployment.

8. **Idempotency** — Duplicate submissions create separate jobs. Production: `Idempotency-Key` header deduplicates within TTL.

9. **Graceful Shutdown** — In-flight jobs abandoned on SIGTERM. Production: drain queue and wait for active jobs before exit.

10. **Real-time Streaming** — Batch file upload. Production: WebSocket or gRPC streaming for sub-2-second voice agent latency.

11. **TTS** — `_tts_stub()` in orchestrator marks the integration point. Production: Azure TTS, Google TTS, or ElevenLabs.

12. **Database Integration** — Currently uses local files and memory. Production: Proper relational (PostgreSQL) or document database (MongoDB) for querying and analyzing final results, allowing for complex reporting and downstream business integration. (Note: Detailed discussion on DB schema and migration to happen at the final stage).

---

## Project Structure

```
audio-intent-pipeline/
├── app/
│   ├── main.py                       ← FastAPI app, CORS, global exception handler
│   ├── config.py                     ← Pydantic Settings — all constants in one place
│   ├── api/
│   │   └── routes.py                 ← route handlers (/v1/analyze-audio, /result, /retry)
│   ├── models/
│   │   ├── stt.py                    ← STTOutput Pydantic schema
│   │   ├── llm.py                    ← LLMOutput, IntentResponse schemas
│   │   └── pipeline.py               ← JobState, API response models
│   ├── prompts/
│   │   └── intent_classification.py  ← versioned prompt management
│   ├── services/
│   │   ├── audio.py                  ← validation + preprocessing (pydub, python-magic)
│   │   ├── stt.py                    ← Deepgram Nova + Whisper fallback
│   │   ├── llm.py                    ← Gemini / Azure OpenAI + fallback + cost tracking
│   │   └── orchestrator.py           ← pipeline state machine + checkpoint control
│   └── utils/
│       ├── logging.py                ← structlog JSON setup
│       └── storage.py                ← checkpoint + final result read/write/cleanup
├── tests/
│   ├── test_stt.py                   ← STT unit + integration tests
│   ├── test_llm.py                   ← LLM unit + integration tests
│   ├── test_pipeline.py              ← orchestrator + route tests
│   ├── test_audio/                   ← pre-recorded fixtures per intent (.wav + .mp3)
│   └── regression/
│       └── test_cases.py             ← known transcript → expected intent pairs
├── scripts/
│   ├── ui_demo.py                    ← Streamlit demo UI
│   └── generate_test_audio.py        ← generate test audio fixtures
├── checkpoints/                      ← temporary stage snapshots (gitignored)
├── final_results/                    ← permanent pipeline outputs (gitignored)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                    ← UV dependency management
└── .env.example
```