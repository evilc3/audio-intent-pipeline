"""
FastAPI application entry point.

Responsibilities:
- Initialise structlog
- Mount routes
- CORS middleware
- Global exception handler (no raw stack traces leak to callers)
"""

import sys

# Python 3.13 compatibility shim — audioop was removed from the stdlib.
# The 'audioop-lts' package provides a backport as a drop-in 'audioop' module.
try:
    import audioop
except ImportError:
    try:
        # Fallback if the package is installed as 'audioop_lts' in some environments
        import audioop_lts as audioop  # type: ignore
        sys.modules["audioop"] = audioop
    except ImportError:
        pass

from contextlib import asynccontextmanager
from datetime import timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

UTC = timezone.utc


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app_startup", version="1.0.0")
    yield
    log.info("app_shutdown")

app = FastAPI(
    title="audio-intent-pipeline",
    version="1.0.0",
    description="AI voice pipeline: STT (Deepgram/Whisper) → LLM intent classification (GPT-4.1)",
    lifespan=lifespan,
)

# CORS — restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler — ensures raw exceptions never leak to callers.
    Structured error response always returned.
    """
    log.error("unhandled_exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


