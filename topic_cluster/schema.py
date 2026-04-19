"""Core dataclasses shared by the normalization and clustering pipeline.

Defines the two shapes that every downstream component depends on:
``Person`` (entries in the person directory) and ``NormalizedMessage``
(the common representation after per-app normalization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Person:
    """A single identity in the person directory.

    ``person_id`` is the lowercased email and acts as the primary key.
    ``is_role_mailbox`` flags shared mailboxes (hr@, cto@, ...) and
    ``is_external`` flags addresses outside the company domain.
    """

    person_id: str
    email: str
    display_name: str | None
    first_name: str | None
    is_role_mailbox: bool
    is_external: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "person_id": self.person_id,
            "email": self.email,
            "display_name": self.display_name,
            "first_name": self.first_name,
            "is_role_mailbox": self.is_role_mailbox,
            "is_external": self.is_external,
        }


@dataclass
class NormalizedMessage:
    """The single per-message shape produced by all three normalizers.

    ``original_content`` is the verbatim body (audit trail); ``text`` is the
    clustering-ready form (e.g. Slack's LLM-rewritten sentence or Gmail's
    ``canonical_subject + body``). ``thread_key`` is namespaced as
    ``"{app}:{native_id}"``. ``parent_message_id`` is intentionally ``None``
    for Outlook — see ``CLAUDE.md``.
    """

    message_id: str
    app: str
    thread_key: str
    parent_message_id: str | None
    timestamp: datetime
    sender_id: str
    participants: list[str]
    subject: str | None
    canonical_subject: str | None
    original_content: str
    text: str
    mentions: list[str] = field(default_factory=list)
    raw_mentions: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict; timestamp is emitted as ISO-8601 Z form."""
        return {
            "message_id": self.message_id,
            "app": self.app,
            "thread_key": self.thread_key,
            "parent_message_id": self.parent_message_id,
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            "sender_id": self.sender_id,
            "participants": list(self.participants),
            "subject": self.subject,
            "canonical_subject": self.canonical_subject,
            "original_content": self.original_content,
            "text": self.text,
            "mentions": list(self.mentions),
            "raw_mentions": list(self.raw_mentions),
            "raw": self.raw,
        }
