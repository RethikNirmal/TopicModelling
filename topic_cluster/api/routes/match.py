"""Routes for classifying new raw messages against the loaded topic model."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from topic_cluster.api.dependencies import get_state
from topic_cluster.api.schemas import (
    MatchRequest,
    MatchResponse,
    MatchResultOut,
    RawMessageIn,
)
from topic_cluster.api.state import AppState

router = APIRouter(tags=["match"])


def _match_one(state: AppState, raw_dict: dict) -> MatchResultOut:
    """Run a single match and translate pipeline errors into 4xx/503 responses."""
    if state.matcher is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not built — POST /build or POST /build/from-path first",
        )
    try:
        result = state.matcher.match(raw_dict)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    return MatchResultOut(**result.to_dict())


@router.post("/match", response_model=MatchResponse)
def match_batch(
    req: MatchRequest,
    state: AppState = Depends(get_state),
) -> MatchResponse:
    """Classify a batch of raw messages against the loaded model."""
    with state.lock:
        results = [_match_one(state, m.model_dump()) for m in req.messages]
    return MatchResponse(results=results)


@router.post("/match/single", response_model=MatchResultOut)
def match_single(
    msg: RawMessageIn,
    state: AppState = Depends(get_state),
) -> MatchResultOut:
    """Classify a single raw message; convenience wrapper over ``/match``."""
    with state.lock:
        return _match_one(state, msg.model_dump())
