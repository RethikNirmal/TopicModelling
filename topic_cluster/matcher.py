"""Runtime classification of new messages against a fitted BERTopic model.

Loaded once at API startup (and re-loaded after every ``/build``),
``Matcher`` normalizes a single raw message, runs BERTopic's ``transform``,
and flags the result as *new topic* (noise or low probability) or
*ambiguous* (top minus second is below a small gap).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import nltk
import numpy as np
from bertopic import BERTopic
from nltk.corpus import stopwords

from topic_cluster.embedders import Embedder, SentenceTransformerEmbedder
from topic_cluster.normalizers import NormalizerRegistry
from topic_cluster.obs import get_logger, timed_stage
from topic_cluster.person import PersonDirectory
from topic_cluster.schema import NormalizedMessage

_log = get_logger("matcher")


def _load_english_stopwords() -> set[str]:
    """Return the NLTK English stopword set, downloading the corpus on first use."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


@dataclass
class MatchResult:
    """Outcome of classifying one raw message: best topic, second-best, and verdict flags."""

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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict; used to populate the API response model."""
        return asdict(self)


class Matcher:
    """Classifies new raw messages against a previously fitted BERTopic model.

    ``NEW_TOPIC_PROB_THRESHOLD`` and ``AMBIGUOUS_GAP_THRESHOLD`` are tuned
    on the demo dataset. See the off-by-one note for BERTopic's probability
    columns in the ``match`` implementation.
    """

    NEW_TOPIC_PROB_THRESHOLD: ClassVar[float] = 0.35
    AMBIGUOUS_GAP_THRESHOLD: ClassVar[float] = 0.15
    # Domain-specific filler words that NLTK's English list doesn't cover.
    _EXTRA_STOPWORDS: ClassVar[set[str]] = {
        "ok", "okay", "lgtm", "thanks", "please", "hi", "hey", "hello",
        "yes", "also", "just",
    }
    _STOPWORDS: ClassVar[set[str]] = _load_english_stopwords() | _EXTRA_STOPWORDS
    _WORD_RE: ClassVar[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

    def __init__(self, artifacts_dir: Path, embedder: Embedder | None = None) -> None:
        """Load the BERTopic model and topic labels from ``artifacts_dir``."""
        self._artifacts_dir = artifacts_dir
        self._embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self._model: BERTopic = BERTopic.load(str(artifacts_dir / "bertopic_model"))
        topics_payload = json.loads((artifacts_dir / "topics.json").read_text())
        self._labels: dict[int, str] = {
            int(t["topic_id"]): t["label"] for t in topics_payload["topics"]
        }
        self._directory = self._load_or_build_directory()
        self._registry = NormalizerRegistry(self._directory)

    def _load_or_build_directory(self) -> PersonDirectory:
        """Reconstruct the ``PersonDirectory`` from the persisted JSON, or return an empty one."""
        directory_path = self._artifacts_dir / "person_directory.json"
        if directory_path.exists():
            payload = json.loads(directory_path.read_text())
            people = payload.get("people", [])
            raw_msgs: list[dict[str, Any]] = []
            for p in people:
                raw_msgs.append(
                    {
                        "sender_email": p["email"],
                        "sender_name": p.get("display_name"),
                    }
                )
            return PersonDirectory.build(raw_msgs)
        return PersonDirectory()

    def normalize_one(self, raw: dict[str, Any]) -> NormalizedMessage:
        """Run per-app normalization on a single raw message (no Slack rewrite at match time)."""
        return self._registry.for_app(raw["app"]).normalize(raw, slack_rewrite=None)

    def match(self, raw: dict[str, Any]) -> MatchResult:
        """Classify ``raw`` against the loaded topic model and return a ``MatchResult``.

        BERTopic returns a ``(n_docs, n_topics + 1)`` probability matrix
        where column 0 is the noise class (``-1``) and column ``i`` maps to
        topic ``i - 1``. ``_col_to_topic`` handles the off-by-one.
        """
        msg = self.normalize_one(raw)
        text = msg.text or msg.original_content or ""
        with timed_stage(
            _log,
            "matcher.embed",
            message_id=msg.message_id,
            app=msg.app,
            text_chars=len(text),
        ):
            embedding = self._embedder.encode([text])

        with timed_stage(
            _log, "matcher.transform", message_id=msg.message_id, app=msg.app
        ):
            topics, probs = self._model.transform([text], embeddings=embedding)
        assigned_topic = int(topics[0])
        probs_row = (
            np.asarray(probs[0])
            if np.ndim(probs) == 2
            else np.asarray([float(probs[0])])
        )

        # BERTopic's probs array is shape (n_docs, n_topics + 1):
        # column 0 = noise (-1), column i (>=1) = topic_id (i - 1).
        def _col_to_topic(col: int) -> int:
            """Map a BERTopic probability-matrix column index to a topic id."""
            return -1 if col == 0 else col - 1

        if probs_row.ndim == 1 and probs_row.size > 1:
            order = np.argsort(probs_row)[::-1]
            best_topic_id = _col_to_topic(int(order[0]))
            best_prob = float(probs_row[order[0]])
            if order.size > 1:
                second_idx = _col_to_topic(int(order[1]))
                second_prob = float(probs_row[order[1]])
            else:
                second_idx = None
                second_prob = None
        else:
            best_topic_id = assigned_topic
            best_prob = float(probs_row[0]) if probs_row.size else 0.0
            second_idx = None
            second_prob = None

        if assigned_topic == -1:
            best_topic_id = -1
            best_prob = 0.0

        is_new_topic = best_topic_id == -1 or best_prob < self.NEW_TOPIC_PROB_THRESHOLD
        is_ambiguous = (
            not is_new_topic
            and second_prob is not None
            and (best_prob - second_prob) < self.AMBIGUOUS_GAP_THRESHOLD
        )

        if is_new_topic:
            reason = (
                "top topic is noise (-1)"
                if best_topic_id == -1
                else f"top probability {best_prob:.3f} < {self.NEW_TOPIC_PROB_THRESHOLD}"
            )
        elif is_ambiguous:
            reason = (
                f"top - second = {best_prob - (second_prob or 0):.3f} "
                f"< {self.AMBIGUOUS_GAP_THRESHOLD}"
            )
        else:
            reason = (
                f"top probability {best_prob:.3f} >= {self.NEW_TOPIC_PROB_THRESHOLD}"
            )

        suggested_label = self._suggest_label(text) if is_new_topic else None

        _log.info(
            "matcher.verdict",
            extras={
                "message_id": msg.message_id,
                "app": msg.app,
                "top_topic_id": best_topic_id,
                "top_probability": round(best_prob, 4),
                "second_topic_id": second_idx,
                "second_probability": round(second_prob, 4)
                if second_prob is not None
                else None,
                "is_new_topic": is_new_topic,
                "is_ambiguous": is_ambiguous,
            },
        )

        return MatchResult(
            message_id=msg.message_id,
            app=msg.app,
            text=text,
            top_topic_id=best_topic_id,
            top_topic_label=self._labels.get(best_topic_id),
            top_probability=best_prob,
            second_topic_id=second_idx,
            second_probability=second_prob,
            is_new_topic=is_new_topic,
            is_ambiguous=is_ambiguous,
            suggested_new_topic_label=suggested_label,
            reason=reason,
        )

    def _suggest_label(self, text: str) -> str:
        tokens = [w.lower() for w in self._WORD_RE.findall(text)]
        kept = [w for w in tokens if w not in self._STOPWORDS]
        if not kept:
            return "unlabeled"
        counts = Counter(kept)
        top = [w for w, _ in counts.most_common(3)]
        return " / ".join(top)
