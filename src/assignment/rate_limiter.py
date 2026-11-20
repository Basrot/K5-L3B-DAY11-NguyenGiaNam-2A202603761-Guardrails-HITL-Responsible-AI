"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(
        self, max_requests: int = 10, window_seconds: int = 60, max_users: int = 512
    ):
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("max_requests and window_seconds must be positive")
        if max_users < 1:
            raise ValueError("max_users must be positive")
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_users = max_users
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.user_last_seen: dict[str, float] = {}
        self.blocked_count = 0
        self.total_count = 0

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        stale_users = [
            uid
            for uid, last_seen in self.user_last_seen.items()
            if now - last_seen > self.window_seconds
        ]
        for stale_user in stale_users:
            self.user_last_seen.pop(stale_user, None)
            self.user_windows.pop(stale_user, None)
        if user_id not in self.user_windows and len(self.user_windows) >= self.max_users:
            oldest_user = min(
                self.user_last_seen, key=self.user_last_seen.get, default=None
            )
            if oldest_user is None:
                oldest_user = next(iter(self.user_windows))
            self.user_last_seen.pop(oldest_user, None)
            self.user_windows.pop(oldest_user, None)
        window = self.user_windows[user_id]
        self.user_last_seen[user_id] = now

        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self.max_requests:
            wait = max(1, int(self.window_seconds - (now - window[0]) + 0.999))
            self.blocked_count += 1
            return self._block_response(f"Rate limit exceeded. Try again in {wait}s.")
        window.append(now)
        return None
