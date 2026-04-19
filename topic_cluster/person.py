"""Builds and queries the shared person directory.

Collects identities from every raw message (senders, recipients, cc),
unifies them by lowercased email, and provides mention-resolution used
when normalizing Slack ``@mentions``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, ClassVar, Iterable

from topic_cluster.schema import Person


class PersonDirectory:
    """Lookup table of ``Person`` objects keyed by lowercased email.

    Uses an explicit allow-list (``ROLE_LOCAL_PARTS``) for role mailboxes
    rather than heuristics — heuristic detection produced false positives
    on real names. Mentions resolve by first-name (e.g. ``@tom``) with an
    email local-part fallback (e.g. ``@bob``).
    """

    ROLE_LOCAL_PARTS: ClassVar[set[str]] = {
        "hr", "finance", "security", "devops", "admin", "accounts",
        "cto", "cfo", "recruiter", "hiring.manager", "tech.lead", "support.lead",
    }
    COMPANY_DOMAIN: ClassVar[str] = "company.com"

    _SENDER_FIELDS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("sender_email", "sender_name"),
        ("from_email", "from_name"),
    )
    _RECIPIENT_FIELDS: ClassVar[tuple[str, ...]] = ("to", "cc", "recipients")

    def __init__(self) -> None:
        """Create an empty directory; use :meth:`build` for the normal entry point."""
        self._people: dict[str, Person] = {}

    @classmethod
    def build(cls, raw_messages: Iterable[dict[str, Any]]) -> "PersonDirectory":
        """Construct a directory by scanning every sender/recipient across all apps.

        When the same email appears with multiple display names, the most
        frequently seen name wins (ties broken by ``Counter.most_common``).
        """
        directory = cls()
        name_counts: dict[str, Counter[str]] = {}

        for msg in raw_messages:
            for email_key, name_key in cls._SENDER_FIELDS:
                email = msg.get(email_key)
                if not email:
                    continue
                key = email.strip().lower()
                if not key:
                    continue
                name_counts.setdefault(key, Counter())
                name = msg.get(name_key)
                if name and name.strip():
                    name_counts[key][name.strip()] += 1

            for recipient_key in cls._RECIPIENT_FIELDS:
                values = msg.get(recipient_key)
                if not values:
                    continue
                for email in values:
                    if not email:
                        continue
                    key = email.strip().lower()
                    if not key:
                        continue
                    name_counts.setdefault(key, Counter())

        for email_key, counts in name_counts.items():
            display_name = counts.most_common(1)[0][0] if counts else None
            directory._add(email_key, display_name)

        return directory

    def _add(self, email: str, display_name: str | None) -> None:
        """Insert (or overwrite) one ``Person`` record, computing the derived flags."""
        key = email.strip().lower()
        local_part, _, domain = key.partition("@")
        first_name = self._infer_first_name(local_part, display_name)
        person = Person(
            person_id=key,
            email=key,
            display_name=display_name,
            first_name=first_name,
            is_role_mailbox=local_part in self.ROLE_LOCAL_PARTS,
            is_external=bool(domain) and domain != self.COMPANY_DOMAIN,
        )
        self._people[key] = person

    @staticmethod
    def _infer_first_name(local_part: str, display_name: str | None) -> str | None:
        """Pick a best-effort first name: display name token wins, else email local part."""
        if display_name:
            first_token = display_name.strip().split()[0] if display_name.strip() else ""
            if first_token:
                return first_token.lower()
        if local_part:
            return local_part.split(".")[0].lower() or None
        return None

    def get(self, email: str) -> Person | None:
        """Return the ``Person`` for ``email`` (case-insensitive) or ``None``."""
        if not email:
            return None
        return self._people.get(email.strip().lower())

    def all(self) -> list[Person]:
        """Return every ``Person`` in the directory as a list."""
        return list(self._people.values())

    def resolve_mention(self, raw: str) -> str | None:
        """Resolve a raw mention token (e.g. ``@tom``) to a ``person_id``.

        Strategy: first-name match wins if unique, then email local-part.
        Returns ``None`` if the token is ambiguous or unknown — callers
        keep the raw string in ``NormalizedMessage.raw_mentions``.
        """
        if not raw:
            return None
        token = raw.lstrip("@").strip().lower()
        if not token:
            return None

        hits = [p for p in self._people.values() if p.first_name == token]
        if len(hits) == 1:
            return hits[0].person_id

        hits = [p for p in self._people.values() if p.email.split("@", 1)[0] == token]
        if len(hits) == 1:
            return hits[0].person_id

        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the directory for persisting to ``person_directory.json``."""
        return {
            "company_domain": self.COMPANY_DOMAIN,
            "people": [p.to_dict() for p in self._people.values()],
        }
