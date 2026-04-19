"""Read-only routes: liveness and the current topic list."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from topic_cluster.api.dependencies import get_state
from topic_cluster.api.schemas import HealthResponse, TopicsResponse
from topic_cluster.api.services import topics_payload_to_summaries
from topic_cluster.api.state import AppState

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Return ``ok`` plus whether a model has been loaded and summary info."""
    payload = state.topics_payload
    return HealthResponse(
        status="ok",
        model_loaded=state.matcher is not None,
        embedder=state.embedder_name,
        n_topics=len(payload["topics"]) if payload else None,
        artifacts_dir=str(state.settings.artifacts_dir),
    )


@router.get("/topics", response_model=TopicsResponse)
def topics(state: AppState = Depends(get_state)) -> TopicsResponse:
    """Return the currently loaded topic list, or 503 if no build has run yet."""
    payload = state.topics_payload
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not built — POST /build or POST /build/from-path first",
        )
    return TopicsResponse(
        embedder=payload.get("embedder"),
        silhouette=payload.get("silhouette_score_overall"),
        n_threads=payload.get("n_threads", 0),
        n_noise=payload.get("n_noise", 0),
        topics=topics_payload_to_summaries(payload),
    )
