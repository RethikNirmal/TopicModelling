"""Pydantic request/response models for the FastAPI routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class RawMessageIn(BaseModel):
    """A raw inbound message. ``extra='allow'`` keeps app-specific fields intact."""

    model_config = ConfigDict(extra="allow")

    message_id: str
    app: Literal["SLACK", "GMAIL", "OUTLOOK"]
    timestamp: str


class BuildOptions(BaseModel):
    """Embedder / labeler knobs shared by both build entry points."""

    embedder: Literal["st", "openai"] = "st"
    embed_model: str | None = None
    label_with: Literal["keywords", "openai"] = "keywords"
    label_model: str = "gpt-4o-mini"


class BuildInlineRequest(BuildOptions):
    """``POST /build`` body: options plus the raw messages inline."""

    raw_messages: list[RawMessageIn]


class BuildFromPathRequest(BuildOptions):
    """``POST /build/from-path`` body: options plus a server-side JSON file path."""

    path: str


class TopicSummary(BaseModel):
    """One topic as returned by ``/topics`` and ``/build``."""

    topic_id: int
    label: str
    keywords: list[str] = []
    size: int
    representative_thread_keys: list[str] = []


class BuildResponse(BaseModel):
    """Full result of a ``/build`` call — counts, silhouette, topics, and token usage."""

    n_threads: int
    n_topics: int
    n_noise: int
    silhouette: float | None
    embedder: str
    artifacts_dir: str
    matcher_reloaded: bool
    topics: list[TopicSummary]
    token_usage: dict[str, dict[str, int]] | None = None


class TopicsResponse(BaseModel):
    """``GET /topics`` payload: just the persisted topic summary."""

    embedder: str | None
    silhouette: float | None
    n_threads: int
    n_noise: int
    topics: list[TopicSummary]


class HealthResponse(BaseModel):
    """``GET /health`` payload: liveness plus whether a model is loaded."""

    status: str
    model_loaded: bool
    embedder: str | None
    n_topics: int | None
    artifacts_dir: str


class MatchRequest(BaseModel):
    """Batch-match body for ``POST /match``."""

    messages: list[RawMessageIn]


class MatchResultOut(BaseModel):
    """API-facing shape of a single ``MatchResult``."""

    message_id: str
    app: str
    text: str
    top_topic_id: int
    top_topic_label: str | None
    top_probability: float
    second_topic_id: int | None
    second_probability: float | None
    is_new_topic: bool
    is_ambiguous: bool
    suggested_new_topic_label: str | None
    reason: str


class MatchResponse(BaseModel):
    """Batch-match response wrapper."""

    results: list[MatchResultOut]
