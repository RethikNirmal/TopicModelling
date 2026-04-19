"""Embedding backends behind a single ``Embedder`` protocol.

Two concrete backends are supported: a local sentence-transformers model
(default ``all-MiniLM-L6-v2``) and OpenAI's ``text-embedding-3-small``.
``name`` is ``"{kind}:{model_name}"`` so it can be persisted into
``topics.json`` and used to invalidate the embedding cache.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from topic_cluster.obs import get_logger, record_tokens, timed_stage

_log = get_logger("embedders")


class Embedder(Protocol):
    """Structural type for any embedding backend used by the pipeline."""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a 2-D float array of shape ``(len(texts), dim)``."""
        ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder (default: ``all-MiniLM-L6-v2``)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the named sentence-transformers model eagerly."""
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model_name}"
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def st_model(self):
        """Expose the underlying ``SentenceTransformer`` (BERTopic wants it directly)."""
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode ``texts`` to a numpy array. Embeddings are not normalized here."""
        with timed_stage(_log, "embed.st", embedder=self.name, n_texts=len(texts)):
            return self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )


class OpenAIEmbedder:
    """OpenAI-hosted embedder (default: ``text-embedding-3-small``), batched at 100."""

    BATCH_SIZE = 100

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        """Load ``OPENAI_API_KEY`` from env/.env; raises if the key is absent."""
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use OpenAI embedder")
        self.name = f"openai:{model_name}"
        self._model_name = model_name
        self._client = OpenAI(api_key=api_key)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode ``texts`` in batches; blank strings are replaced with a space to appease the API."""
        sanitized = [t if t.strip() else " " for t in texts]
        with timed_stage(
            _log, "embed.openai", embedder=self.name, n_texts=len(sanitized)
        ) as ctx:
            vectors: list[list[float]] = []
            total_prompt = 0
            for start in range(0, len(sanitized), self.BATCH_SIZE):
                batch = sanitized[start : start + self.BATCH_SIZE]
                resp = self._client.embeddings.create(
                    model=self._model_name, input=batch
                )
                vectors.extend(d.embedding for d in resp.data)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    total_prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
            if total_prompt:
                record_tokens(f"embed:{self._model_name}", total_prompt, 0)
                ctx["prompt_tokens"] = total_prompt
            return np.asarray(vectors, dtype=np.float32)


def build_embedder(kind: str, model_name: str | None) -> Embedder:
    """Factory: ``kind`` is ``"st"`` or ``"openai"``; ``model_name`` is optional."""
    if kind == "st":
        return SentenceTransformerEmbedder(model_name or "all-MiniLM-L6-v2")
    if kind == "openai":
        return OpenAIEmbedder(model_name or "text-embedding-3-small")
    raise ValueError(f"unknown embedder kind: {kind!r} (expected 'st' or 'openai')")
