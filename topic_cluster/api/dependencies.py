"""FastAPI dependency providers for shared app state."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from topic_cluster.api.state import AppState
from topic_cluster.matcher import Matcher


def get_state(request: Request) -> AppState:
    """Return the shared ``AppState`` attached to the FastAPI app."""
    return request.app.state.s


def get_matcher(state: AppState = Depends(get_state)) -> Matcher:
    """Return the live ``Matcher`` or raise 503 if no model has been built yet."""
    if state.matcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not built — POST /build or POST /build/from-path first",
        )
    return state.matcher
