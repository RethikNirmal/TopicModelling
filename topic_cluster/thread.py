"""Thread grouping: collapses ``NormalizedMessage`` objects into ``Thread`` records.

The clustering step runs on threads (~300 in this dataset), not individual
messages — single short Slack messages make BERTopic noisy, and same-thread
replies are strongly topically coherent. Per-app text assembly differs;
see ``_text_for_*``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from topic_cluster.schema import NormalizedMessage


@dataclass
class Thread:
    """A single grouped thread, already flattened to one ``text`` blob for clustering.

    ``start_ts`` / ``end_ts`` come from the first/last message after
    chronological sort. ``participants`` is the union across all messages.
    """

    thread_key: str
    app: str
    message_ids: list[str]
    participants: set[str]
    start_ts: datetime
    end_ts: datetime
    text: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict; participants are sorted for stability."""
        return {
            "thread_key": self.thread_key,
            "app": self.app,
            "message_ids": list(self.message_ids),
            "participants": sorted(self.participants),
            "start_ts": self.start_ts.isoformat().replace("+00:00", "Z"),
            "end_ts": self.end_ts.isoformat().replace("+00:00", "Z"),
            "text": self.text,
        }


class ThreadBuilder:
    """Groups normalized messages by ``thread_key`` and emits one ``Thread`` per group."""

    MAX_TEXT_CHARS: ClassVar[int] = 1500

    def __init__(self, messages: list[NormalizedMessage]) -> None:
        """Hold the messages to group. Call :meth:`build` to produce threads."""
        self._messages = messages

    def build(self) -> list[Thread]:
        """Produce one chronologically-sorted ``Thread`` per distinct ``thread_key``.

        Empty-text threads are dropped (nothing for BERTopic to cluster on).
        Returned list is sorted by ``start_ts`` ascending.
        """
        grouped: dict[str, list[NormalizedMessage]] = defaultdict(list)
        for m in self._messages:
            grouped[m.thread_key].append(m)

        threads: list[Thread] = []
        for thread_key, msgs in grouped.items():
            msgs.sort(key=lambda m: m.timestamp)
            app = msgs[0].app
            text = self._build_text(app, msgs)
            if not text.strip():
                continue
            # if len(text) > self.MAX_TEXT_CHARS:
            #     text = text[: self.MAX_TEXT_CHARS].rstrip()
            participants: set[str] = set()
            for m in msgs:
                participants.update(m.participants)
            threads.append(
                Thread(
                    thread_key=thread_key,
                    app=app,
                    message_ids=[m.message_id for m in msgs],
                    participants=participants,
                    start_ts=msgs[0].timestamp,
                    end_ts=msgs[-1].timestamp,
                    text=text,
                )
            )
        threads.sort(key=lambda t: t.start_ts)
        return threads

    def _build_text(self, app: str, msgs: list[NormalizedMessage]) -> str:
        """Dispatch to the per-app text assembler, or fall back to joined originals."""
        if app == "SLACK":
            return self._text_for_slack(msgs)
        if app == "GMAIL":
            return self._text_for_gmail(msgs)
        if app == "OUTLOOK":
            return self._text_for_outlook(msgs)
        return "\n".join(m.original_content for m in msgs if m.original_content)

    def _text_for_slack(self, msgs: list[NormalizedMessage]) -> str:
        """Slack: prepend ``[#channel]`` then join the LLM-rewritten message texts."""
        channel = (msgs[0].raw.get("channel", "") if msgs[0].raw else "").lstrip("#")
        prefix = f"[#{channel}] " if channel else ""
        body = "\n".join(m.text for m in msgs if m.text)
        return f"{prefix}{body}".strip()

    def _text_for_gmail(self, msgs: list[NormalizedMessage]) -> str:
        """Gmail: canonical subject followed by concatenated bodies."""
        return self._subject_plus_bodies(msgs)

    def _text_for_outlook(self, msgs: list[NormalizedMessage]) -> str:
        """Outlook: canonical subject followed by concatenated bodies."""
        return self._subject_plus_bodies(msgs)

    @staticmethod
    def _subject_plus_bodies(msgs: list[NormalizedMessage]) -> str:
        """Shared helper used by both email assemblers; takes the first message's subject."""
        subject = msgs[0].canonical_subject or msgs[0].subject or ""
        bodies = "\n".join(m.original_content for m in msgs if m.original_content)
        if subject and bodies:
            return f"{subject}\n{bodies}"
        return subject or bodies
