"""
Disk storage utilities for the pipeline.
- Checkpoints: temporary stage snapshots enabling partial retry
- Final results: permanent output written to final_results/
On retry, the orchestrator loads the last successful checkpoint
instead of reprocessing earlier stages.
"""

import json
import os
from pathlib import Path
from typing import Any

from app.config import settings
from app.utils.logging import get_logger

log = get_logger(__name__)


class CheckpointError(Exception):
    pass

def _final_path(trace_id: str) -> Path:
    return Path(settings.output_dir) / f"final_{trace_id}.json"

def _checkpoint_path(trace_id: str, stage: str) -> Path:
    return Path(settings.checkpoint_dir) / f"{stage}_{trace_id}.json"

async def save_final_result(trace_id: str, data: dict[str, Any]) -> None:
    """Write final result to disk as a JSON checkpoint."""
    path = _final_path(trace_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
        log.info("final_result_saved", trace_id=trace_id, path=str(path))
    except OSError as exc:
        raise CheckpointError(f"Failed to save final result {path}: {exc}") from exc

async def load_final_result(trace_id: str) -> dict[str, Any]:
    """Read a previously saved final result from disk."""
    path = _final_path(trace_id)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        log.info("final_result_loaded", trace_id=trace_id, path=str(path))
        return data
    except FileNotFoundError as exc:
        raise CheckpointError(f"Final result not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Final result corrupt: {path}: {exc}") from exc


async def save_checkpoint(trace_id: str, stage: str, data: dict[str, Any]) -> None:
    """Write stage output to disk as a JSON checkpoint."""
    path = _checkpoint_path(trace_id, stage)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
        log.info("checkpoint_saved", trace_id=trace_id, stage=stage, path=str(path))
    except OSError as exc:
        raise CheckpointError(f"Failed to save checkpoint {path}: {exc}") from exc


async def load_checkpoint(trace_id: str, stage: str) -> dict[str, Any]:
    """Read a previously saved checkpoint from disk."""
    path = _checkpoint_path(trace_id, stage)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        log.info("checkpoint_loaded", trace_id=trace_id, stage=stage, path=str(path))
        return data
    except FileNotFoundError as exc:
        raise CheckpointError(f"Checkpoint not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Checkpoint corrupt: {path}: {exc}") from exc


def cleanup_checkpoints(trace_id: str) -> None:
    """
    Delete all checkpoint files for a given trace_id.
    Called via try/finally — always runs even on exception.
    """
    for stage in ("stt", "llm"):
        path = _checkpoint_path(trace_id, stage)
        try:
            if path.exists():
                os.remove(path)
                log.info("checkpoint_deleted", trace_id=trace_id, stage=stage)
        except OSError as exc:
            # Log but do not raise — cleanup is best-effort
            log.warning("checkpoint_delete_failed", trace_id=trace_id, stage=stage, error=str(exc))
