"""
validators.py
=============
Post-LLM guardrails. These are cheap, deterministic checks that catch the
anti-patterns the brief explicitly says the judge penalizes
(challenge-brief.md §11) and the malformed-output failure modes
(challenge-testing-brief.md §10). Nothing here calls an LLM — it's pure
string/logic checks so it's fast and testable.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

VALID_CTA = {"binary", "open_ended", "none"}

# Multiple "reply X for Y" style options beyond a simple binary pair is the
# "Reply YES for X, NO for Y, MAYBE for Z" anti-pattern called out in the
# brief. Two options (e.g. slot A / slot B) is fine for booking flows.
_OPTION_PATTERN = re.compile(r"\breply\s+\d+\b|\breply\s+(yes|no|stop)\b", re.IGNORECASE)

_PREAMBLE_PATTERNS = [
    re.compile(r"^\s*i hope you('| a)re doing well", re.IGNORECASE),
    re.compile(r"^\s*i('m| am) reaching out today", re.IGNORECASE),
]

_REINTRO_PATTERNS = [
    re.compile(r"\bhi[,!]? this is vera\b", re.IGNORECASE),
    re.compile(r"\bvera se bol rahi hoon\b", re.IGNORECASE),
]


def validate_action(
    body: str,
    cta: str,
    taboo_words: Iterable[str],
    prior_bodies: Iterable[str] = (),
    is_first_message: bool = True,
) -> Tuple[bool, List[str]]:
    """Returns (ok, issues). ok=False means the caller should re-prompt or
    fall back rather than send this body as-is."""
    issues: List[str] = []

    if not body or not body.strip():
        issues.append("empty_body")
        return False, issues

    body_lower = body.lower()

    for word in taboo_words:
        if word and word.lower() in body_lower:
            issues.append(f"taboo_word:{word}")

    if cta not in VALID_CTA:
        issues.append(f"invalid_cta:{cta}")

    option_hits = _OPTION_PATTERN.findall(body)
    if len(option_hits) > 2:
        issues.append("multi_cta_suspected")

    for pat in _PREAMBLE_PATTERNS:
        if pat.search(body):
            issues.append("long_preamble")
            break

    if not is_first_message:
        for pat in _REINTRO_PATTERNS:
            if pat.search(body):
                issues.append("reintroduced_self")
                break

    normalized = body.strip().lower()
    for prior in prior_bodies:
        if prior and prior.strip().lower() == normalized:
            issues.append("exact_repeat")
            break

    ok = not any(i for i in issues if not i.startswith("taboo_word") or True)
    # Taboo words and repeats are hard failures; multi_cta/preamble are soft
    # (logged in rationale, don't block sending) — hard-fail only on the
    # things the judge scores as outright malformed/penalized-severely.
    hard_fail = any(
        i == "empty_body" or i == "exact_repeat" or i.startswith("taboo_word")
        for i in issues
    )
    return (not hard_fail), issues
