"""In-memory service layer for the temporary Blue classroom demo."""
from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from collections import deque

from core.utils import chat_with_agent


class DemoError(Exception):
    status_code = 400


class DemoInputError(DemoError):
    status_code = 400


class DemoRateLimitError(DemoError):
    status_code = 429


class DemoProviderError(DemoError):
    status_code = 503


class BlueDemoService:
    """Run prompts through the existing Blue pipeline, serially and in memory."""

    def __init__(
        self,
        *,
        agent,
        runner,
        limiter,
        input_guard,
        output_guard,
        audit,
        monitor,
        global_max_requests: int = 10,
        global_window_seconds: int = 60,
        max_pending_requests: int = 20,
        max_prompt_chars: int = 2000,
        audit_limit: int = 200,
        event_limit: int = 30,
    ):
        if global_max_requests < 1 or global_window_seconds < 1:
            raise ValueError("global request limits must be positive")
        if max_pending_requests < 1:
            raise ValueError("max_pending_requests must be positive")
        if max_prompt_chars < 1 or audit_limit < 1 or event_limit < 1:
            raise ValueError("prompt and memory limits must be positive")
        self.agent = agent
        self.runner = runner
        self.limiter = limiter
        self.input_guard = input_guard
        self.output_guard = output_guard
        self.audit = audit
        self.monitor = monitor
        self.global_max_requests = global_max_requests
        self.global_window_seconds = global_window_seconds
        self.max_pending_requests = max_pending_requests
        self.max_prompt_chars = max_prompt_chars
        self.audit_limit = audit_limit
        self._global_requests: deque[float] = deque()
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._pending = 0
        self._processing = False
        self._request_number = 0
        self._events: deque[dict] = deque(maxlen=event_limit)

    @classmethod
    def from_environment(cls) -> "BlueDemoService":
        """Create the same guarded Blue used by Checkpoint 3, without Red setup."""
        from core.config import get_openrouter_api_key

        if not get_openrouter_api_key():
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Add it to .env before starting the demo."
            )

        from agents.agent import create_blue_agent
        from assignment.pipeline import build_observability, build_production_plugins
        from assignment.rate_limiter import RateLimitPlugin
        from guardrails.input_guardrails import InputGuardrailPlugin
        from guardrails.output_guardrails import OutputGuardrailPlugin

        plugins = build_production_plugins(use_llm_judge=False)
        agent, runner = create_blue_agent(plugins)
        audit, monitor = build_observability()
        limiter = next(p for p in plugins if isinstance(p, RateLimitPlugin))
        input_guard = next(p for p in plugins if isinstance(p, InputGuardrailPlugin))
        output_guard = next(p for p in plugins if isinstance(p, OutputGuardrailPlugin))
        return cls(
            agent=agent,
            runner=runner,
            limiter=limiter,
            input_guard=input_guard,
            output_guard=output_guard,
            audit=audit,
            monitor=monitor,
        )

    @staticmethod
    def _clean_alias(alias: str) -> str:
        alias = re.sub(r"[^\w .-]", "", (alias or "").strip(), flags=re.UNICODE)
        alias = re.sub(r"\s+", " ", alias)[:24].strip()
        return alias or "Guest"

    def process_prompt(self, prompt: str, *, user_id: str, alias: str) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise DemoInputError("Enter a prompt before submitting.")
        if len(prompt) > self.max_prompt_chars:
            raise DemoInputError(
                f"Prompt is too long. Limit: {self.max_prompt_chars} characters."
            )
        if not isinstance(user_id, str) or not user_id or len(user_id) > 128:
            raise DemoInputError("Refresh the page to start a valid demo session.")

        with self._condition:
            if self._pending >= self.max_pending_requests:
                raise DemoRateLimitError(
                    "The demo is busy. Please wait for a prompt to finish and retry."
                )
            ticket = self._next_ticket
            self._next_ticket += 1
            self._pending += 1
            while ticket != self._serving_ticket:
                self._condition.wait()
            self._pending -= 1
            self._processing = True

        try:
            return self._process_in_order(prompt, user_id, self._clean_alias(alias))
        finally:
            with self._condition:
                self._processing = False
                self._serving_ticket += 1
                self._condition.notify_all()

    def _process_in_order(self, prompt: str, user_id: str, alias: str) -> dict:
        now = time.monotonic()
        cutoff = now - self.global_window_seconds
        while self._global_requests and self._global_requests[0] <= cutoff:
            self._global_requests.popleft()

        self._request_number += 1
        request_id = f"demo-{self._request_number}-{uuid.uuid4().hex[:8]}"
        self.monitor.total_requests += 1
        self.audit.record_input(user_id=user_id, text=prompt, request_id=request_id)

        if len(self._global_requests) >= self.global_max_requests:
            self.monitor.blocked_requests += 1
            self.monitor.rate_limit_hits += 1
            message = "The classroom request limit was reached. Please wait a minute and retry."
            self.audit.record_output(
                user_id=user_id,
                text=message,
                blocked=True,
                layer="demo_rate_limit",
                request_id=request_id,
            )
            self._add_event(alias, state="rate_limited", layer="demo_rate_limit")
            self._trim_audit()
            raise DemoRateLimitError(message)

        self._global_requests.append(now)
        before_rate = self.limiter.blocked_count
        before_input = self.input_guard.blocked_count
        before_output = self.output_guard.blocked_count
        before_redacted = self.output_guard.redacted_count

        try:
            response, _ = asyncio.run(
                chat_with_agent(
                    self.agent, self.runner, prompt, user_id=user_id
                )
            )
        except Exception as exc:
            error_name = type(exc).__name__.casefold()
            status_code = getattr(exc, "status_code", None)
            if status_code == 429 or "ratelimit" in error_name:
                message = "Blue is rate-limited right now. Please wait a moment and retry."
                error_layer = "provider_rate_limit"
                event_state = "rate_limited"
                self.monitor.rate_limit_hits += 1
            elif "timeout" in error_name or status_code in {408, 504}:
                message = "Blue took too long to respond. Please wait a moment and retry."
                error_layer = "provider_timeout"
                event_state = "unavailable"
            else:
                message = "Blue is temporarily unavailable. Please wait a moment and retry."
                error_layer = "provider_error"
                event_state = "unavailable"
            self.audit.record_output(
                user_id=user_id,
                text=message,
                blocked=False,
                layer=error_layer,
                request_id=request_id,
            )
            self._add_event(alias, state=event_state, layer=error_layer)
            self._trim_audit()
            if error_layer == "provider_rate_limit":
                raise DemoRateLimitError(message) from None
            raise DemoProviderError(message) from None

        rate_blocked = self.limiter.blocked_count > before_rate
        input_blocked = self.input_guard.blocked_count > before_input
        output_blocked = self.output_guard.blocked_count > before_output
        redacted = self.output_guard.redacted_count > before_redacted
        blocked = rate_blocked or input_blocked or output_blocked
        layer = (
            "rate_limit" if rate_blocked else
            "input_guardrail" if input_blocked else
            "output_guardrail" if output_blocked or redacted else None
        )
        self.monitor.blocked_requests += int(blocked)
        self.monitor.rate_limit_hits += int(rate_blocked)
        self.audit.record_output(
            user_id=user_id,
            text=response or "",
            blocked=blocked,
            layer=layer,
            request_id=request_id,
        )
        state = "blocked" if blocked else "redacted" if redacted else "answered"
        self._add_event(alias, state=state, layer=layer)
        self._trim_audit()
        return {
            "request_id": request_id,
            "response": response or "",
            "blocked": blocked,
            "redacted": redacted,
            "layer": layer,
        }

    def _trim_audit(self) -> None:
        logs = getattr(self.audit, "logs", None)
        if isinstance(logs, list) and len(logs) > self.audit_limit:
            del logs[:-self.audit_limit]

    def _add_event(self, alias: str, *, state: str, layer: str | None) -> None:
        with self._condition:
            self._events.append({"alias": alias, "state": state, "layer": layer})

    def status(self) -> dict:
        with self._condition:
            return {
                "pending": self._pending,
                "processing": self._processing,
                "recent": list(self._events),
                "total_requests": self.monitor.total_requests,
                "blocked_requests": self.monitor.blocked_requests,
            }
