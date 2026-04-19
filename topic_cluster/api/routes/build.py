"""Routes that build a new topic model from raw messages (inline or file path)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from topic_cluster.api.dependencies import get_state
from topic_cluster.api.schemas import (
    BuildFromPathRequest,
    BuildInlineRequest,
    BuildResponse,
)
from topic_cluster.api.services import (
    normalize_in_memory,
    raw_messages_to_dicts,
    reload_matcher,
    run_build,
    topics_payload_to_summaries,
)
from topic_cluster.api.state import AppState
from topic_cluster.obs import tally_context

router = APIRouter(tags=["build"])


def _build_response(
    state: AppState,
    summary: dict,
    reloaded: bool,
    token_usage: dict | None,
) -> BuildResponse:
    """Combine the pipeline summary with the current topic list into a ``BuildResponse``."""
    payload = state.topics_payload or {}
    return BuildResponse(
        n_threads=summary["n_threads"],
        n_topics=summary["n_topics"],
        n_noise=summary["n_noise"],
        silhouette=summary.get("silhouette"),
        embedder=summary["embedder"],
        artifacts_dir=str(state.settings.artifacts_dir),
        matcher_reloaded=reloaded,
        topics=topics_payload_to_summaries(payload),
        token_usage=token_usage,
    )


@router.post("/build", response_model=BuildResponse)
def build_inline(
    req: BuildInlineRequest,
    state: AppState = Depends(get_state),
) -> BuildResponse:
    """Build a topic model from raw messages sent in the request body."""
    artifacts_dir = state.settings.artifacts_dir
    try:
        with tally_context() as tally:
            raw_dicts = raw_messages_to_dicts(req.raw_messages)
            normalized = normalize_in_memory(
                raw_messages=raw_dicts,
                artifacts_dir=artifacts_dir,
                persist=True,
            )
            with state.lock:
                summary = run_build(normalized, artifacts_dir, req)
                reloaded = reload_matcher(state)
            token_usage = tally.as_dict()
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    return _build_response(state, summary, reloaded, token_usage)


@router.post("/build/from-path", response_model=BuildResponse)
def build_from_path(
    req: BuildFromPathRequest,
    state: AppState = Depends(get_state),
) -> BuildResponse:
    """Build a topic model by reading a JSON array of raw messages from a server-side path."""
    path = Path(req.path)
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"path does not exist or is not a file: {req.path}",
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"file is not valid JSON: {e}",
        )
    if not isinstance(raw, list):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="expected a JSON array of raw messages at top level",
        )

    artifacts_dir = state.settings.artifacts_dir
    try:
        with tally_context() as tally:
            normalized = normalize_in_memory(
                raw_messages=raw,
                artifacts_dir=artifacts_dir,
                persist=True,
            )
            with state.lock:
                summary = run_build(normalized, artifacts_dir, req)
                reloaded = reload_matcher(state)
            token_usage = tally.as_dict()
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    return _build_response(state, summary, reloaded, token_usage)
