"""Orchestration layer: the single place that sequences normalize → build → reload.

Route handlers call into these functions; everything here is synchronous
and safe to run under the ``AppState.lock``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from topic_cluster.api.schemas import (
    BuildOptions,
    RawMessageIn,
    TopicSummary,
)
from topic_cluster.api.state import AppState
from topic_cluster.cluster import TopicModel, build_openai_representation
from topic_cluster.embedders import build_embedder
from topic_cluster.matcher import Matcher
from topic_cluster.normalizers import NormalizerRegistry
from topic_cluster.obs import get_logger, tally_context, timed_stage
from topic_cluster.person import PersonDirectory
from topic_cluster.schema import NormalizedMessage
from topic_cluster.slack_rewrite import SlackRewriter
from topic_cluster.thread import ThreadBuilder

_log = get_logger("services")


def normalize_in_memory(
    raw_messages: list[dict[str, Any]],
    artifacts_dir: Path,
    persist: bool,
) -> list[NormalizedMessage]:
    """Build the person directory, LLM-rewrite Slack, and run per-app normalization.

    When ``persist`` is ``True``, also writes ``normalized_messages.json``
    and ``person_directory.json`` under ``artifacts_dir`` for later
    debugging / graders.
    """
    with timed_stage(
        _log, "normalize", n_raw=len(raw_messages), persist=persist
    ) as ctx:
        with timed_stage(_log, "normalize.person_directory"):
            directory = PersonDirectory.build(raw_messages)
        rewriter = SlackRewriter(raw_messages=raw_messages)
        rewrites = rewriter.rewrite_all()
        registry = NormalizerRegistry(directory)
        with timed_stage(_log, "normalize.per_message"):
            normalized = [
                registry.for_app(m["app"]).normalize(
                    m,
                    slack_rewrite=(
                        rewrites.get(m["message_id"]) if m["app"] == "SLACK" else None
                    ),
                )
                for m in raw_messages
            ]
        ctx["n_normalized"] = len(normalized)
        if persist:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            out_path = artifacts_dir / "normalized_messages.json"
            out_path.write_text(json.dumps([m.to_dict() for m in normalized], indent=2))
            directory_path = artifacts_dir / "person_directory.json"
            directory_path.write_text(json.dumps(directory.to_dict(), indent=2))
        return normalized


def run_build(
    normalized: list[NormalizedMessage],
    artifacts_dir: Path,
    options: BuildOptions,
) -> dict[str, Any]:
    """Build threads, fit the topic model, and persist artifacts. Returns a summary dict."""
    with timed_stage(
        _log,
        "build",
        n_normalized=len(normalized),
        embedder=options.embedder,
        label_with=options.label_with,
    ) as ctx:
        with timed_stage(_log, "build.threads"):
            threads = ThreadBuilder(normalized).build()
        embedder = build_embedder(options.embedder, options.embed_model)
        representation_model = (
            build_openai_representation(options.label_model)
            if options.label_with == "openai"
            else None
        )
        model = TopicModel(
            artifacts_dir,
            embedder=embedder,
            representation_model=representation_model,
        )
        model.fit(threads)
        with timed_stage(_log, "build.save"):
            summary = model.save()
        summary["n_threads"] = len(threads)
        ctx.update(
            {
                "n_threads": len(threads),
                "n_topics": summary.get("n_topics"),
                "n_noise": summary.get("n_noise"),
                "silhouette": summary.get("silhouette"),
            }
        )
        return summary


def reload_matcher(state: AppState) -> bool:
    """Try to (re)load the Matcher from artifacts. Returns True if loaded."""
    artifacts_dir = state.settings.artifacts_dir
    topics_path = artifacts_dir / "topics.json"
    model_path = artifacts_dir / "bertopic_model"
    if not (topics_path.exists() and model_path.exists()):
        state.matcher = None
        state.embedder_name = None
        state.topics_payload = None
        return False
    payload = json.loads(topics_path.read_text())
    embedder_name = payload.get("embedder") or "st:all-MiniLM-L6-v2"
    if ":" in embedder_name:
        kind, model_name = embedder_name.split(":", 1)
    else:
        kind, model_name = "st", "all-MiniLM-L6-v2"
    embedder = build_embedder(kind, model_name)
    state.matcher = Matcher(artifacts_dir, embedder=embedder)
    state.embedder_name = embedder_name
    state.topics_payload = payload
    return True


def topics_payload_to_summaries(payload: dict[str, Any]) -> list[TopicSummary]:
    """Convert a persisted ``topics.json`` payload into API ``TopicSummary`` models."""
    return [
        TopicSummary(
            topic_id=t["topic_id"],
            label=t["label"],
            keywords=t.get("keywords", []),
            size=t["size"],
            representative_thread_keys=t.get("representative_thread_keys", []),
        )
        for t in payload.get("topics", [])
    ]


def raw_messages_to_dicts(messages: list[RawMessageIn]) -> list[dict[str, Any]]:
    """Dump a list of Pydantic ``RawMessageIn`` models to plain dicts for the pipeline."""
    return [m.model_dump() for m in messages]
