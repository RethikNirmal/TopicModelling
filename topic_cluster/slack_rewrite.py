"""LLM preprocessor that rewrites short Slack messages for clustering.

Many Slack messages ("LGTM", "ok") are too short to cluster by themselves,
so each message is rewritten into a one-sentence standalone topic summary
using nearby thread siblings and recent channel context. On any LLM error
the rewriter falls back to the original content and logs a warning.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, ClassVar

from dotenv import load_dotenv

from topic_cluster.obs import get_logger, record_tokens, timed_stage

_log = get_logger("slack_rewrite")


class SlackRewriter:
    """Batch and single-message LLM rewriter for Slack content.

    ``rewrite_all`` drives the full-dataset pass; ``rewrite_one`` is for
    live ``/match`` flows. Both go through ``_rewrite_with_status`` so the
    output-length cap and fallback behavior are identical.
    """

    MODEL: ClassVar[str] = "gpt-4o-mini"
    CONTEXT_CHANNEL_WINDOW_HOURS: ClassVar[int] = 48
    CONTEXT_CHANNEL_MAX_MSGS: ClassVar[int] = 20
    OUTPUT_MAX_CHARS: ClassVar[int] = 200
    TEMPERATURE: ClassVar[float] = 0.0

    _SYSTEM_PROMPT: ClassVar[str] = (
        "You are a preprocessing step for a topic-clustering system. "
        "The target Slack message is typically too short to cluster on its own. "
        "Given the context below, rewrite the target as a single standalone sentence "
        "or short paragraph (<=200 characters) that captures its topic.\n\n"
        "Rules:\n"
        "1. Use context only if it is clearly about the same subject as the target. "
        "Ignore unrelated messages.\n"
        "2. Remove filler like \"LGTM\", \"Sounds good\", \"On it\". Instead describe "
        "what is being agreed to, if context makes that clear.\n"
        "3. If no context is relevant, rewrite using only the target.\n"
        "4. If the target has no extractable topic even with context, return the "
        "original verbatim.\n"
        "5. Do not invent facts not present in target or context.\n"
        "Return only the rewritten text -- no preamble."
    )

    def __init__(self, raw_messages: list[dict[str, Any]]) -> None:
        """Filter to Slack messages and pre-build thread/channel indexes.

        Non-Slack messages are ignored here; the caller passes the full
        raw list and this class picks what it can use.
        """
        self._raw = [m for m in raw_messages if m.get("app") == "SLACK"]
        self._by_thread, self._by_channel_sorted = self._index()
        self._client = None  # lazy

    def _index(
        self,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
        """Build ``(by_thread, by_channel)`` maps with messages sorted by timestamp."""
        by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for msg in self._raw:
            if msg.get("thread_id"):
                by_thread[msg["thread_id"]].append(msg)
            if msg.get("channel"):
                by_channel[msg["channel"]].append(msg)

        key = lambda m: m.get("timestamp", "")
        for v in by_thread.values():
            v.sort(key=key)
        for v in by_channel.values():
            v.sort(key=key)
        return by_thread, by_channel

    def rewrite_all(self) -> dict[str, str]:
        """Rewrite every Slack message and return a ``{message_id: text}`` map.

        Failures are counted and logged but do not raise — the final value
        is the original content for any message whose LLM call failed.
        """
        with timed_stage(_log, "slack_rewrite.all", n_messages=len(self._raw)) as ctx:
            out: dict[str, str] = {}
            failures = 0
            for msg in self._raw:
                text, ok = self._rewrite_with_status(msg)
                out[msg["message_id"]] = text
                if not ok:
                    failures += 1
            ctx["n_failed"] = failures
            return out

    def rewrite_one(self, msg: dict[str, Any]) -> str:
        """Rewrite a single Slack message; returns the original on LLM failure."""
        text, _ = self._rewrite_with_status(msg)
        return text

    def _rewrite_with_status(self, msg: dict[str, Any]) -> tuple[str, bool]:
        """Core rewrite driver: returns ``(text, ok)`` so callers can count failures."""
        mid = msg["message_id"]
        context = self._build_context(msg)
        target_content = (msg.get("content") or "").strip()

        start = time.perf_counter()
        ok = True
        try:
            rewritten = self._call_llm(msg, context)
        except Exception as e:
            ok = False
            _log.warning(
                "slack_rewrite.llm_failed",
                extras={
                    "message_id": mid,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "fallback": "original_content",
                },
            )
            rewritten = target_content
        else:
            _log.debug(
                "slack_rewrite.llm_ok",
                extras={
                    "message_id": mid,
                    "elapsed_ms": int((time.perf_counter() - start) * 1000),
                    "context_chars": len(context),
                    "target_chars": len(target_content),
                },
            )

        rewritten = (rewritten or "").strip() or target_content
        if len(rewritten) > self.OUTPUT_MAX_CHARS:
            rewritten = rewritten[: self.OUTPUT_MAX_CHARS].rstrip()
        return rewritten, ok

    def _build_context(self, msg: dict[str, Any]) -> str:
        """Assemble the context string (thread siblings + prior channel window) for the LLM."""
        thread_siblings = [
            m for m in self._by_thread.get(msg.get("thread_id") or "", [])
            if m["message_id"] != msg["message_id"]
        ]

        target_ts = self._parse_ts(msg.get("timestamp"))
        channel_msgs = self._by_channel_sorted.get(msg.get("channel") or "", [])
        sibling_ids = {m["message_id"] for m in thread_siblings}
        sibling_ids.add(msg["message_id"])

        channel_window: list[dict[str, Any]] = []
        if target_ts is not None:
            cutoff = target_ts - timedelta(hours=self.CONTEXT_CHANNEL_WINDOW_HOURS)
            for m in channel_msgs:
                ts = self._parse_ts(m.get("timestamp"))
                if ts is None or ts >= target_ts:
                    continue
                if ts < cutoff:
                    continue
                if m["message_id"] in sibling_ids:
                    continue
                channel_window.append(m)
            channel_window = channel_window[-self.CONTEXT_CHANNEL_MAX_MSGS :]

        lines: list[str] = []
        if thread_siblings:
            lines.append("THREAD SIBLINGS (same thread as target):")
            for m in thread_siblings:
                lines.append(self._fmt_ctx_msg(m))
            lines.append("")
        if channel_window:
            lines.append(
                f"CHANNEL WINDOW (prior {self.CONTEXT_CHANNEL_WINDOW_HOURS}h in "
                f"{msg.get('channel')}):"
            )
            for m in channel_window:
                lines.append(self._fmt_ctx_msg(m))
            lines.append("")
        return "\n".join(lines)

    def _fmt_ctx_msg(self, m: dict[str, Any]) -> str:
        """Format a single context message as ``[ts] First: content`` for the LLM prompt."""
        sender = m.get("sender_name") or m.get("sender_email") or "?"
        first = sender.split()[0] if sender else "?"
        ts = m.get("timestamp", "")
        content = (m.get("content") or "").replace("\n", " ").strip()
        return f"[{ts}] {first}: {content}"

    def _call_llm(self, msg: dict[str, Any], context: str) -> str:
        """Invoke the OpenAI chat completion call and record token usage."""
        client = self._get_client()
        user_prompt = (
            f"TARGET MESSAGE:\n"
            f"- channel: {msg.get('channel')}\n"
            f"- sender: {msg.get('sender_name') or msg.get('sender_email')}\n"
            f"- timestamp: {msg.get('timestamp')}\n"
            f"- thread_id: {msg.get('thread_id')}\n"
            f"- content: {msg.get('content', '')}\n\n"
            f"CONTEXT:\n{context or '(none)'}"
        )
        resp = client.chat.completions.create(
            model=self.MODEL,
            temperature=self.TEMPERATURE,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            record_tokens(
                f"slack_rewrite:{self.MODEL}",
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
        return resp.choices[0].message.content or ""

    def _get_client(self):
        """Lazily construct the OpenAI client; raises if ``OPENAI_API_KEY`` is missing."""
        if self._client is not None:
            return self._client
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot call LLM")
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        return self._client

    @staticmethod
    def _parse_ts(s: str | None) -> datetime | None:
        """Parse an ISO-8601 timestamp; return ``None`` for missing or malformed input."""
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
