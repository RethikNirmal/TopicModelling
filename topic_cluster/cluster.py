"""BERTopic-based clustering over grouped threads.

``TopicModel`` wraps fitting, silhouette scoring, label selection, and
artifact persistence (``bertopic_model/``, ``topics.json``,
``assignments.json``). Labels come from either the OpenAI representation
model (if requested) or the top-3 c-TF-IDF keywords.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from bertopic import BERTopic
from sklearn.metrics import silhouette_score
from umap import UMAP

from topic_cluster.embedders import Embedder, SentenceTransformerEmbedder
from topic_cluster.obs import get_logger, timed_stage
from topic_cluster.thread import Thread

import os

from bertopic.representation import OpenAI as OpenAIRep
from dotenv import load_dotenv
from openai import OpenAI

_log = get_logger("cluster")
def build_openai_representation(model: str = "gpt-4o-mini"):
    """Build a BERTopic ``OpenAI`` representation model for LLM-generated topic labels.

    The prompt asks for a 2-6 word Title Case label. Raises if
    ``OPENAI_API_KEY`` is missing.
    """

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot use OpenAI labels")
    client = OpenAI(api_key=api_key)
    prompt = (
        "I have a topic that contains the following representative documents:\n"
        "[DOCUMENTS]\n\n"
        "The topic is described by these keywords: [KEYWORDS]\n\n"
        "Based on the documents and keywords, give a short topic label "
        "(2-6 words, Title Case, no quotes, no trailing punctuation) that "
        "captures what this topic is about. Reply with the label only."
    )
    return OpenAIRep(
        client=client,
        model=model,
        prompt=prompt,
        nr_docs=4,
        doc_length=300,
        tokenizer="whitespace",
    )


class TopicModel:
    """Thin wrapper around BERTopic with fixed UMAP/clustering hyperparameters.

    The tuned values (``MIN_TOPIC_SIZE=8``, deterministic UMAP with
    ``random_state=42``, cosine metric) produce stable results on the
    ~300-thread dataset; do not change without re-running the notebook
    referenced in ``docs/CODE_GUIDE.md``.
    """

    MIN_TOPIC_SIZE: ClassVar[int] = 8
    RANDOM_STATE: ClassVar[int] = 42
    UMAP_NEIGHBORS: ClassVar[int] = 15
    UMAP_COMPONENTS: ClassVar[int] = 5

    def __init__(
        self,
        artifacts_dir: Path,
        embedder: Embedder | None = None,
        representation_model: Any = None,
    ) -> None:
        """Hold the output dir and optional custom embedder / BERTopic representation model."""
        self._artifacts_dir = artifacts_dir
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self._representation_model = representation_model
        self._model: BERTopic | None = None
        self._threads: list[Thread] = []
        self._embeddings: np.ndarray | None = None
        self._topics: list[int] = []
        self._probs: np.ndarray | None = None
        self._silhouette: float | None = None

    def fit(self, threads: list[Thread]) -> None:
        """Embed every thread, fit BERTopic, and record the overall silhouette score.

        ``silhouette`` is computed only over non-noise points (``topic != -1``)
        and requires at least two non-noise clusters; otherwise it stays ``None``.
        """
        self._threads = threads
        texts = [t.text for t in threads]

        with timed_stage(_log, "cluster.fit", n_threads=len(threads)) as fit_ctx:
            with timed_stage(_log, "cluster.embed", n_threads=len(threads)):
                self._embeddings = self._embedder.encode(texts)

            umap_model = UMAP(
                n_neighbors=self.UMAP_NEIGHBORS,
                n_components=self.UMAP_COMPONENTS,
                random_state=self.RANDOM_STATE,
                metric="cosine",
            )
            bertopic_embedding_model = (
                self._embedder.st_model
                if isinstance(self._embedder, SentenceTransformerEmbedder)
                else None
            )
            self._model = BERTopic(
                min_topic_size=self.MIN_TOPIC_SIZE,
                calculate_probabilities=True,
                umap_model=umap_model,
                embedding_model=bertopic_embedding_model,
                representation_model=self._representation_model,
                verbose=False,
            )
            with timed_stage(_log, "cluster.bertopic_fit"):
                topics, probs = self._model.fit_transform(
                    texts, embeddings=self._embeddings
                )
            self._topics = list(topics)
            self._probs = probs

            labels_arr = np.asarray(self._topics)
            mask = labels_arr != -1
            n_noise = int((~mask).sum())
            n_topics = int(len(set(labels_arr[mask].tolist()))) if mask.any() else 0
            if mask.sum() >= 2 and n_topics >= 2:
                self._silhouette = float(
                    silhouette_score(
                        self._embeddings[mask], labels_arr[mask], metric="cosine"
                    )
                )
            else:
                self._silhouette = None

            fit_ctx.update(
                {
                    "n_topics": n_topics,
                    "n_noise": n_noise,
                    "silhouette": self._silhouette,
                    "embedder": self._embedder.name,
                }
            )
            _log.info(
                "cluster.quality",
                extras={
                    "n_threads": len(threads),
                    "n_topics": n_topics,
                    "n_noise": n_noise,
                    "silhouette": self._silhouette,
                    "embedder": self._embedder.name,
                },
            )

    def save(self) -> dict:
        """Persist the BERTopic model plus ``topics.json`` / ``assignments.json``.

        Returns a summary dict (threads / topics / noise counts, silhouette,
        embedder name) used by the API build response.
        """
        assert self._model is not None and self._embeddings is not None
        model_dir = self._artifacts_dir / "bertopic_model"
        save_embedding_model = (
            self._embedder._model_name  # type: ignore[attr-defined]
            if isinstance(self._embedder, SentenceTransformerEmbedder)
            else None
        )
        self._model.save(
            str(model_dir),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=save_embedding_model,
        )

        topics_payload = self._build_topics_payload()
        (self._artifacts_dir / "topics.json").write_text(
            json.dumps(topics_payload, indent=2)
        )

        assignments_payload = self._build_assignments_payload()
        (self._artifacts_dir / "assignments.json").write_text(
            json.dumps(assignments_payload, indent=2)
        )

        return {
            "n_threads": len(self._threads),
            "n_topics": len(topics_payload["topics"]),
            "n_noise": sum(1 for t in self._topics if t == -1),
            "silhouette": self._silhouette,
            "embedder": self._embedder.name,
        }

    def _build_topics_payload(self) -> dict:
        """Produce the ``topics.json`` shape: one entry per topic with label, keywords, members."""
        assert self._model is not None
        info = self._model.get_topic_info()
        topics_out: list[dict] = []
        for _, row in info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                continue
            words = self._model.get_topic(tid) or []
            keywords = [w for w, _ in words][:10]
            label = self._label_for(tid, keywords)
            members = [
                self._threads[i].thread_key
                for i, t in enumerate(self._topics)
                if t == tid
            ]
            reps = self._representative_threads(tid, members)
            topics_out.append(
                {
                    "topic_id": tid,
                    "label": label,
                    "keywords": keywords,
                    "size": int(row["Count"]),
                    "representative_thread_keys": reps,
                }
            )
        topics_out.sort(key=lambda x: x["topic_id"])
        return {
            "silhouette_score_overall": self._silhouette,
            "n_threads": len(self._threads),
            "n_noise": sum(1 for t in self._topics if t == -1),
            "embedder": self._embedder.name,
            "topics": topics_out,
        }

    def _label_for(self, tid: int, keywords: list[str]) -> str:
        """Pick a label for a topic: OpenAI representation first, else top-3 keywords joined."""
        assert self._model is not None
        if self._representation_model is not None:
            row = self._model.get_topic_info(tid)
            if row is not None and not row.empty and "Representation" in row.columns:
                rep = row["Representation"].iloc[0]
                if isinstance(rep, list) and rep:
                    cand = str(rep[0]).strip().strip('"').strip("'").rstrip(".")
                    if cand:
                        return cand
        if keywords:
            return " / ".join(keywords[:3])
        return f"topic_{tid}"

    def _representative_threads(self, tid: int, members: list[str]) -> list[str]:
        """Return up to 3 thread keys most strongly assigned to ``tid`` by probability."""
        assert self._probs is not None
        if not members:
            return []
        member_idx = [i for i, t in enumerate(self._topics) if t == tid]
        if self._probs.ndim == 2:
            scored = sorted(
                member_idx,
                key=lambda i: float(self._probs[i, tid]) if tid < self._probs.shape[1] else 0.0,
                reverse=True,
            )
        else:
            scored = sorted(member_idx, key=lambda i: float(self._probs[i]), reverse=True)
        return [self._threads[i].thread_key for i in scored[:3]]

    def _build_assignments_payload(self) -> dict:
        """Produce the ``assignments.json`` shape: ``{thread_key: {topic_id, probability, app}}``."""
        assert self._probs is not None
        out: dict[str, dict] = {}
        for i, thread in enumerate(self._threads):
            tid = int(self._topics[i])
            if self._probs.ndim == 2:
                if tid == -1:
                    prob = float(self._probs[i].max()) if self._probs.shape[1] else 0.0
                else:
                    prob = (
                        float(self._probs[i, tid])
                        if tid < self._probs.shape[1]
                        else 0.0
                    )
            else:
                prob = float(self._probs[i])
            out[thread.thread_key] = {
                "topic_id": tid,
                "probability": prob,
                "app": thread.app,
            }
        return out
