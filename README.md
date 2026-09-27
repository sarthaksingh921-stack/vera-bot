# Vera Bot — magicpin AI Challenge submission

**Team:** Sarthak Singh Meena · B.Tech Computer Engineering, DTU
**Model:** Claude (`claude-sonnet-4-6` by default, configurable via `ANTHROPIC_MODEL`)

## Approach

One composer, no per-trigger-kind branching. Instead of writing a separate prompt
template for each of the ~25 trigger `kind`s in the dataset, `composer.py` serializes
the full 4-layer context (category, merchant, trigger, optional customer) into one
structured bundle and asks Claude to pick the single driving signal, ground every claim
in the given data, match category voice, and use exactly one compulsion lever with a
single CTA. A small `TRIGGER_FRAMING` table gives light per-kind hints (mostly: is this
external news or an internal number, does it concern the merchant or their customer),
not hard branches — unknown kinds fall back to a generic framing.

This is a deliberate bet on the FAQ in `challenge-testing-brief.md`: *"the real judge
harness injects fresh scenarios you haven't seen... bots that pattern-match the
simulator will fail."* A generic, context-grounded prompt should degrade gracefully on
trigger kinds and merchant states that never appeared in the seed dataset, where a
per-kind template bank would silently break.

**Reply handling** (`conversation.py`) layers cheap, deterministic heuristics *before*
any LLM call, to save latency budget and to reliably satisfy the three Phase-4 replay
scenarios:

- **Auto-reply detection** — verbatim repeats or known canned-reply phrases (English +
  Hindi) trigger one more low-friction attempt, then a graceful exit on the second hit.
- **Intent-transition handling** — explicit commitment phrases ("let's do it", "chalo",
  "sign me up") route straight to an action-mode LLM call instructed *not* to re-qualify.
- **Hostile/opt-out handling** — ends the conversation immediately with a short apology,
  no further nudges.

Only the free-form "what do I actually say next" step touches the LLM; routing is
regex/word-list based and instant.

**Resilience:** every LLM call path has a deterministic fallback (`_fallback_body` in
`composer.py`, and a canned reply in `conversation.py`) so `/v1/tick` and `/v1/reply`
never time out or return malformed/empty bodies even if the API key is missing, the
network blips, or Claude's output fails validation twice. This trades a worse message
for a valid one — per `challenge-testing-brief.md` §10, malformed/empty output is
penalized harder than a plain-but-valid message.

**Determinism:** `temperature=0` on every call, plus an in-process compose cache keyed
by a hash of the full input bundle, so repeated ticks for the same (category, merchant,
trigger, customer) state return the identical message rather than re-rolling the LLM.

## Architecture

```
context_store.py   In-memory, versioned, thread-safe (scope, context_id) -> payload store.
llm_client.py       Stdlib-only Anthropic Messages API client (no SDK dependency).
validators.py        Post-LLM checks: taboo words, CTA shape, anti-repetition, empty body.
composer.py          Builds the 4-layer prompt, calls Claude, parses/validates, falls back.
conversation.py       /v1/reply state machine: auto-reply / intent / hostile detection + LLM reply.
main.py               FastAPI wiring for the 5 required endpoints.
generate_submission.py  Offline script producing submission.jsonl for the 30 canonical pairs.
self_test.py           Stdlib-only test suite (mock LLM) — no network needed to run.
```

## Running locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --host 0.0.0.0 --port 8080
```

Sanity-check the core logic without any network or API key:

```bash
python3 self_test.py   # 38 checks, pure stdlib + mock LLM
```

Generate `submission.jsonl` for the 30 canonical test pairs:

```bash
python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
export ANTHROPIC_API_KEY=sk-ant-...
python3 generate_submission.py --expanded-dir expanded --out submission.jsonl
```

Run the official local judge simulator against the running server:

```bash
export BOT_URL=http://localhost:8080
python3 judge_simulator.py   # from the original challenge zip
```

## Deploying

Three ready-made options are included:

- **Render** — `render.yaml` (set `ANTHROPIC_API_KEY` as a secret env var in the dashboard)
- **Railway / Heroku-style** — `Procfile`
- **Any container host** — `Dockerfile`

After deploying, set `ANTHROPIC_API_KEY` (and optionally `VERA_TEAM_NAME`,
`VERA_TEAM_MEMBERS`, `VERA_CONTACT_EMAIL`) in the host's environment variables, then
submit the public base URL.

## Tradeoffs / what would help most

- **No embeddings/retrieval** — the full category digest is passed in the prompt
  directly rather than retrieved by relevance. At 5 categories with a handful of digest
  items each this fits comfortably in context; wouldn't scale past a few dozen items
  per category without retrieval.
- **Suppression is tick-scoped, not calendar-scoped** — `main.py` dedups a given
  `(merchant, trigger)` pair within the process lifetime via `_seen_trigger_merchant_pairs`,
  not by parsing `suppression_key`'s implied time window (e.g. `2026-W17`). Good enough
  for a 60-minute test window; a production version would parse and respect the window.
- **Auto-reply detection is heuristic, not learned** — a phrase list + verbatim-repeat
  check. Works for the brief's own examples; a merchant with an unusual auto-reply
  wording could slip through once before the verbatim-repeat check catches it on the
  second occurrence.
- **What would help most:** a stream of *real* (anonymized) merchant reply transcripts
  per category would let the auto-reply/hostile/intent phrase lists be learned rather
  than hand-written, which is the single most template-y part of this submission.
