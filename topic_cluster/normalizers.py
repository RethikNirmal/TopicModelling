"""Per-app normalizers that convert raw messages to ``NormalizedMessage``.

This module is the correctness surface of the pipeline: downstream
clustering assumes ``thread_key``, ``text``, and ``participants`` are
set consistently. See ``CLAUDE.md`` for the non-obvious design choices
(never merging by canonical subject, Slack participants being sender-only,
``parent_message_id`` being ``None`` for Outlook, etc.).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, ClassVar, Iterable

from topic_cluster.person import PersonDirectory
from topic_cluster.schema import NormalizedMessage


class Normalizer:
    """Abstract base for per-app normalizers.

    Subclasses implement :meth:`normalize`. Shared helpers (timestamp
    parsing, subject canonicalization, email de-duplication, mention
    resolution) live here.
    """

    _RE_SUBJECT_PREFIX: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:(?:re|fwd|fw)\s*:\s*)+", re.IGNORECASE
    )

    def __init__(self, directory: PersonDirectory) -> None:
        """Bind the shared ``PersonDirectory`` used for mention resolution."""
        self._directory = directory

    def normalize(
        self, raw: dict[str, Any], slack_rewrite: str | None = None
    ) -> NormalizedMessage:
        """Produce a ``NormalizedMessage`` from one raw message dict.

        ``slack_rewrite`` is only meaningful for ``SlackNormalizer``;
        other subclasses ignore it.
        """
        raise NotImplementedError

    @staticmethod
    def _parse_timestamp(s: str) -> datetime:
        """Parse an ISO-8601 timestamp, tolerating the trailing ``Z`` form."""
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    @classmethod
    def _canonical_subject(cls, subject: str | None) -> str | None:
        """Strip leading ``Re:``/``Fwd:``/``Fw:`` chains from an email subject."""
        if subject is None:
            return None
        cleaned = cls._RE_SUBJECT_PREFIX.sub("", subject).strip()
        return cleaned or None

    @staticmethod
    def _clean_emails(values: Iterable[str | None]) -> list[str]:
        """Lowercase, strip, and de-dupe an iterable of emails, preserving order."""
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            if not v:
                continue
            key = v.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _resolve_mentions(self, raw_mentions: list[str]) -> tuple[list[str], list[str]]:
        """Split raw mention tokens into ``(resolved_person_ids, unresolved_raw)``."""
        resolved: list[str] = []
        unresolved: list[str] = []
        for m in raw_mentions:
            pid = self._directory.resolve_mention(m)
            if pid:
                resolved.append(pid)
            else:
                unresolved.append(m)
        return resolved, unresolved


class SlackNormalizer(Normalizer):
    """Slack-specific normalizer.

    Uses ``thread_id`` (falling back to the message id) for thread grouping,
    fills ``text`` from the LLM rewrite when available (or the raw content
    otherwise), and treats the channel string as ``subject`` /
    ``canonical_subject``. ``participants`` is sender-only — Slack raw
    messages do not carry a recipient list for channels.
    """

    def normalize(
        self, raw: dict[str, Any], slack_rewrite: str | None = None
    ) -> NormalizedMessage:
        """Normalize one Slack raw message, optionally swapping in the LLM rewrite."""
        message_id = raw["message_id"]
        app = "SLACK"
        channel = raw.get("channel")
        thread_native = raw.get("thread_id") or message_id
        thread_key = f"{app}:{thread_native}"

        parent_id = raw.get("reply_to_message_id") if raw.get("is_thread_reply") else None
        timestamp = self._parse_timestamp(raw["timestamp"])
        sender_id = (raw.get("sender_email") or "").strip().lower()
        participants = self._clean_emails([raw.get("sender_email")])

        content = raw.get("content", "") or ""
        text = (slack_rewrite or "").strip() or f"[{channel}] {content}".strip()

        mentions_raw = raw.get("mentions") or []
        resolved, unresolved = self._resolve_mentions(mentions_raw)

        return NormalizedMessage(
            message_id=message_id,
            app=app,
            thread_key=thread_key,
            parent_message_id=parent_id,
            timestamp=timestamp,
            sender_id=sender_id,
            participants=participants,
            subject=channel,
            canonical_subject=channel,
            original_content=content,
            text=text,
            mentions=resolved,
            raw_mentions=unresolved,
            raw=raw,
        )


class GmailNormalizer(Normalizer):
    """Gmail-specific normalizer.

    Uses ``thread_id`` for grouping, keeps the native ``in_reply_to`` as
    ``parent_message_id``, and assembles ``text`` as
    ``canonical_subject\\nbody`` when both are present.
    """

    def normalize(
        self, raw: dict[str, Any], slack_rewrite: str | None = None
    ) -> NormalizedMessage:
        """Normalize one Gmail raw message. ``slack_rewrite`` is ignored."""
        message_id = raw["message_id"]
        app = "GMAIL"
        thread_native = raw.get("thread_id") or message_id
        thread_key = f"{app}:{thread_native}"

        parent_id = raw.get("in_reply_to")
        timestamp = self._parse_timestamp(raw["timestamp"])
        sender_id = (raw.get("from_email") or "").strip().lower()

        participants = self._clean_emails(
            [raw.get("from_email"), *(raw.get("to") or []), *(raw.get("cc") or [])]
        )

        subject = raw.get("subject")
        canonical = self._canonical_subject(subject)
        body = raw.get("body", "") or ""
        text_parts = [canonical, body] if canonical else [body]
        text = "\n".join(p for p in text_parts if p).strip()

        return NormalizedMessage(
            message_id=message_id,
            app=app,
            thread_key=thread_key,
            parent_message_id=parent_id,
            timestamp=timestamp,
            sender_id=sender_id,
            participants=participants,
            subject=subject,
            canonical_subject=canonical,
            original_content=body,
            text=text,
            mentions=[],
            raw_mentions=[],
            raw=raw,
        )


class OutlookNormalizer(Normalizer):
    """Outlook-specific normalizer.

    Uses ``conversation_id`` for grouping. ``parent_message_id`` is always
    ``None`` — Outlook does not provide a reliable in-reply-to field, and
    reconstructing parent chains from timestamps/subjects is explicitly
    out of scope.
    """

    def normalize(
        self, raw: dict[str, Any], slack_rewrite: str | None = None
    ) -> NormalizedMessage:
        """Normalize one Outlook raw message. ``slack_rewrite`` is ignored."""
        message_id = raw["message_id"]
        app = "OUTLOOK"
        thread_native = raw.get("conversation_id") or message_id
        thread_key = f"{app}:{thread_native}"

        parent_id = None
        timestamp = self._parse_timestamp(raw["timestamp"])
        sender_id = (raw.get("sender_email") or "").strip().lower()

        participants = self._clean_emails(
            [raw.get("sender_email"), *(raw.get("recipients") or [])]
        )

        subject = raw.get("subject")
        canonical = self._canonical_subject(subject)
        body = raw.get("body_full", "") or ""
        text_parts = [canonical, body] if canonical else [body]
        text = "\n".join(p for p in text_parts if p).strip()

        return NormalizedMessage(
            message_id=message_id,
            app=app,
            thread_key=thread_key,
            parent_message_id=parent_id,
            timestamp=timestamp,
            sender_id=sender_id,
            participants=participants,
            subject=subject,
            canonical_subject=canonical,
            original_content=body,
            text=text,
            mentions=[],
            raw_mentions=[],
            raw=raw,
        )


class NormalizerRegistry:
    """Dispatch table mapping an ``app`` string to its ``Normalizer`` instance."""

    def __init__(self, directory: PersonDirectory) -> None:
        """Instantiate one normalizer per supported app, all sharing ``directory``."""
        self._by_app: dict[str, Normalizer] = {
            "SLACK": SlackNormalizer(directory),
            "GMAIL": GmailNormalizer(directory),
            "OUTLOOK": OutlookNormalizer(directory),
        }

    def for_app(self, app: str) -> Normalizer:
        """Return the normalizer for ``app``. Raises ``ValueError`` if unknown."""
        try:
            return self._by_app[app]
        except KeyError:
            raise ValueError(f"Unknown app: {app!r}")
