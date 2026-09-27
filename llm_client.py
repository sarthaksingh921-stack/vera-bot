"""
llm_client.py
=============
Minimal stdlib-only client for the Anthropic Messages API. Deliberately
avoids the `anthropic` SDK so the deployed bot has one fewer dependency to
break, matching the style of the judge's own reference AnthropicProvider in
judge_simulator.py.

Set ANTHROPIC_API_KEY in the environment before deploying. If it's not set,
`available()` returns False and callers (composer.py) fall back to a
deterministic template so /v1/tick and /v1/reply never hang or 500 — they
just degrade gracefully, which matters because both endpoints have a hard
30s budget (challenge-testing-brief.md §5).
"""

from __future__ import annotations

import json
import os
from urllib import error as urlerror
from urllib import request as urlrequest

DEFAULT_MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"


class LLMError(Exception):
    pass


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 25,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> str:
        """Single-turn completion. Raises LLMError on any failure so
        callers can fall back cleanly."""
        if not self.api_key:
            raise LLMError("no_api_key")

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        req = urlrequest.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            resp = urlrequest.urlopen(req, timeout=self.timeout)
            data = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:400]
            raise LLMError(f"http_{e.code}: {body}") from e
        except Exception as e:  # noqa: BLE001 - want a single failure mode
            raise LLMError(str(e)) from e

        parts = data.get("content", [])
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not text:
            raise LLMError("empty_response")
        return text
