"""
context_store.py
=================
Thread-safe, in-memory, versioned context store.

Keyed by (scope, context_id). A push with version <= current version is a
no-op (idempotent). A push with a higher version atomically replaces the
prior payload. This is exactly the contract described in
challenge-testing-brief.md §2.1.

No third-party dependencies — pure stdlib so it's trivially unit-testable
without spinning up the HTTP layer.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


class StaleVersionError(Exception):
    def __init__(self, current_version: int):
        self.current_version = current_version
        super().__init__(f"stale_version (current={current_version})")


class ContextStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        # (scope, context_id) -> {"version": int, "payload": dict}
        self._store: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # -- writes -----------------------------------------------------------

    def push(self, scope: str, context_id: str, version: int, payload: dict) -> int:
        """Returns the version now stored. Raises StaleVersionError if the
        incoming version is <= what's already stored (caller should treat
        that as a no-op ack, per the idempotency contract — NOT an error
        for equal versions, only for strictly-lower-or-equal pushes of the
        same version repeated, which the brief calls a no-op)."""
        if scope not in VALID_SCOPES:
            raise ValueError(f"invalid_scope: {scope}")
        with self._lock:
            key = (scope, context_id)
            cur = self._store.get(key)
            if cur is not None and version <= cur["version"]:
                if version == cur["version"]:
                    # Re-posting the same version: no-op, not an error.
                    return cur["version"]
                raise StaleVersionError(cur["version"])
            self._store[key] = {"version": version, "payload": payload}
            return version

    def wipe(self) -> None:
        with self._lock:
            self._store.clear()

    # -- reads --------------------------------------------------------------

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._store.get((scope, context_id))
            return entry["payload"] if entry else None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        with self._lock:
            entry = self._store.get((scope, context_id))
            return entry["version"] if entry else None

    def all_of(self, scope: str) -> Dict[str, dict]:
        with self._lock:
            return {
                cid: entry["payload"]
                for (s, cid), entry in self._store.items()
                if s == scope
            }

    def counts(self) -> Dict[str, int]:
        with self._lock:
            counts = {s: 0 for s in VALID_SCOPES}
            for (scope, _cid) in self._store.keys():
                counts[scope] = counts.get(scope, 0) + 1
            return counts

    def category_for_merchant(self, merchant_id: str) -> Optional[dict]:
        m = self.get("merchant", merchant_id)
        if not m:
            return None
        slug = m.get("category_slug")
        return self.get("category", slug) if slug else None
