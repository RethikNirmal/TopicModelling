"""Observability primitives: structured JSON logging, timing, and token tally.

A single ``topic_cluster`` logger is configured at import time (level and
format controlled by ``LOG_LEVEL`` / ``LOG_FORMAT`` env vars). The
``timed_stage`` context manager wraps any operation and logs start/end
with elapsed-ms; ``record_tokens`` accumulates LLM token usage into a
request-scoped ``Tally``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator


_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def new_correlation_id() -> str:
    """Generate and install a fresh 12-char correlation id for this async context."""
    cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str | None) -> None:
    """Install ``cid`` as the active correlation id (used by incoming ``X-Correlation-Id`` headers)."""
    _correlation_id.set(cid)


def get_correlation_id() -> str | None:
    """Return the active correlation id, or ``None`` if none has been set."""
    return _correlation_id.get()


class _JsonFormatter(logging.Formatter):
    """Logging formatter that emits a single JSON object per record.

    Includes correlation id, any ``extras`` dict the caller passed through
    the adapter, and exception traceback if present.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize one ``LogRecord`` to a JSON string."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        cid = _correlation_id.get()
        if cid:
            payload["correlation_id"] = cid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extras = getattr(record, "extras", None)
        if isinstance(extras, dict):
            payload.update(extras)
        return json.dumps(payload, default=str)


_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_FORMAT = os.environ.get("LOG_FORMAT", "json").lower()


def _build_handler() -> logging.Handler:
    """Create the stdout log handler; picks JSON vs text via the ``LOG_FORMAT`` env var."""
    handler = logging.StreamHandler(sys.stdout)
    if _FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
    return handler


_ROOT_CONFIGURED = False


def _configure_root() -> None:
    """One-shot configuration of the ``topic_cluster`` logger root (idempotent)."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    root = logging.getLogger("topic_cluster")
    root.setLevel(_LEVEL)
    root.handlers.clear()
    root.addHandler(_build_handler())
    root.propagate = False
    _ROOT_CONFIGURED = True


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a logger adapter under the ``topic_cluster`` namespace that accepts ``extras=``."""
    _configure_root()
    base = logging.getLogger(f"topic_cluster.{name}" if not name.startswith("topic_cluster") else name)
    return _ExtrasAdapter(base, {})


class _ExtrasAdapter(logging.LoggerAdapter):
    """``LoggerAdapter`` that passes a caller-provided ``extras=`` dict through to the formatter."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Merge adapter-level and call-level extras into ``record.extras``."""
        extras = kwargs.pop("extras", None) or {}
        merged = {**self.extra, **extras} if self.extra else extras
        if merged:
            kwargs.setdefault("extra", {})["extras"] = merged
        return msg, kwargs


@dataclass
class TokenUsage:
    """Running token counters for a single LLM label (e.g. ``slack_rewrite:gpt-4o-mini``)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        """Accumulate one LLM call's token counts."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.calls += 1

    def as_dict(self) -> dict[str, int]:
        """Return a plain dict suitable for the API response body."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Tally:
    """Per-request token tally keyed by operation label."""

    by_label: dict[str, TokenUsage] = field(default_factory=dict)

    def record(self, label: str, prompt: int, completion: int) -> None:
        """Add token usage under ``label``, creating a ``TokenUsage`` bucket on first use."""
        self.by_label.setdefault(label, TokenUsage()).add(prompt, completion)

    def as_dict(self) -> dict[str, dict[str, int]]:
        """Serialize the tally for inclusion in API responses."""
        return {k: v.as_dict() for k, v in self.by_label.items()}

    def reset(self) -> None:
        """Clear all labels; used between independent request scopes."""
        self.by_label.clear()


_current_tally: ContextVar[Tally | None] = ContextVar("tally", default=None)


@contextmanager
def tally_context() -> Iterator[Tally]:
    """Activate a fresh ``Tally`` for the duration of the block, then pop it."""
    tally = Tally()
    token = _current_tally.set(tally)
    try:
        yield tally
    finally:
        _current_tally.reset(token)


def record_tokens(label: str, prompt: int, completion: int) -> None:
    """Add token counts under ``label`` on the active ``Tally`` (no-op if none is active)."""
    tally = _current_tally.get()
    if tally is not None:
        tally.record(label, prompt, completion)


@contextmanager
def timed_stage(
    logger: logging.LoggerAdapter,
    stage: str,
    **extras: Any,
) -> Iterator[dict[str, Any]]:
    """Emit ``{stage} start`` / ``{stage} ok`` logs and yield a mutable extras dict.

    Callers can mutate the yielded dict to attach additional fields
    (e.g. result counts) that appear in the closing log line.
    """
    ctx: dict[str, Any] = {"stage": stage, **extras}
    logger.info(f"{stage} start", extras=ctx)
    start = time.perf_counter()
    try:
        yield ctx
    except Exception:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            f"{stage} error",
            extras={**ctx, "elapsed_ms": elapsed_ms, "status": "error"},
        )
        raise
    else:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            f"{stage} ok",
            extras={**ctx, "elapsed_ms": elapsed_ms, "status": "ok"},
        )


def timed(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of :func:`timed_stage` for wrapping whole functions."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a wrapped ``fn`` that records timing for ``stage`` on every call."""
        logger = get_logger(fn.__module__.split(".")[-1])

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Invoke ``fn`` inside a ``timed_stage`` block."""
            with timed_stage(logger, stage):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
