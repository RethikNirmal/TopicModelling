"""Process-wide state: the live ``Matcher`` plus a lock that serializes rebuilds."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from topic_cluster.api.config import Settings
from topic_cluster.matcher import Matcher


@dataclass
class AppState:
    """Container for mutable API state; attached to ``app.state.s`` at startup.

    ``lock`` is used to serialize ``/build`` rebuilds against concurrent
    ``/match`` calls — matching while the model is being replaced would
    otherwise hit a torn-read.
    """

    settings: Settings
    lock: threading.Lock = field(default_factory=threading.Lock)
    matcher: Matcher | None = None
    embedder_name: str | None = None
    topics_payload: dict[str, Any] | None = None
