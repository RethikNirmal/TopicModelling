"""FastAPI entrypoint: wires lifespan, correlation-id middleware, and routers."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from topic_cluster.api.config import Settings
from topic_cluster.api.routes import build, health, match
from topic_cluster.api.services import reload_matcher
from topic_cluster.api.state import AppState
from topic_cluster.obs import get_logger, new_correlation_id, set_correlation_id

_log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build ``AppState`` on startup and attempt to load a pre-built Matcher."""
    settings = Settings()
    state = AppState(settings=settings)
    try:
        reload_matcher(state)
        _log.info(
            "startup.matcher_loaded",
            extras={"model_loaded": state.matcher is not None},
        )
    except Exception as e:
        _log.warning(
            "startup.matcher_failed",
            extras={"error_type": type(e).__name__, "error": str(e)},
        )
    app.state.s = state
    yield


app = FastAPI(
    title="Topic Cluster API",
    description=(
        "FastAPI wrapper around the cross-app topic-clustering pipeline. "
        "Normalize messages, build BERTopic models, and classify new messages "
        "against current topics or flag them as new topics."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    """Propagate/generate a correlation id and log request start/end with elapsed ms."""
    incoming = request.headers.get("X-Correlation-Id")
    cid = incoming or new_correlation_id()
    set_correlation_id(cid)
    start = time.perf_counter()
    _log.info(
        "request.start",
        extras={"method": request.method, "path": request.url.path},
    )
    try:
        response = await call_next(request)
    except Exception:
        _log.exception(
            "request.error",
            extras={
                "method": request.method,
                "path": request.url.path,
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
            },
        )
        raise
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    _log.info(
        "request.end",
        extras={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
        },
    )
    response.headers["X-Correlation-Id"] = cid
    return response


app.include_router(health.router)
app.include_router(build.router)
app.include_router(match.router)
