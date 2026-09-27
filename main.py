"""
main.py
=======
The 5-endpoint HTTP surface the judge harness calls, per
challenge-testing-brief.md §2. This module is intentionally thin — all real
logic lives in context_store.py, composer.py and conversation.py so those
stay testable without an HTTP server.

Run: uvicorn main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from composer import Composer
from context_store import ContextStore, StaleVersionError
from conversation import ConversationManager
from llm_client import AnthropicClient

START_TIME = time.time()
MAX_ACTIONS_PER_TICK = 20
MAX_UNANSWERED_SENDS = 3  # graceful stop per brief §12.5

TEAM_NAME = os.environ.get("VERA_TEAM_NAME", "Team Sarthak")
TEAM_MEMBERS = [m.strip() for m in os.environ.get("VERA_TEAM_MEMBERS", "Sarthak Singh Meena").split(",") if m.strip()]
CONTACT_EMAIL = os.environ.get("VERA_CONTACT_EMAIL", "")
BOT_VERSION = os.environ.get("VERA_VERSION", "1.0.0")

app = FastAPI(title="Vera AI Challenge Bot")

store = ContextStore()
llm = AnthropicClient()
composer = Composer(llm=llm)
conversations = ConversationManager(llm=llm)

# suppression_key -> last sent iso timestamp, to avoid re-sending the same
# logical message every tick while its trigger remains "active"
_sent_suppression_keys: Dict[str, str] = {}
# conversation_id -> merchant_id/customer_id/trigger_id, for suppression-key
# based dedup at the (merchant, trigger) level regardless of conversation id
_seen_trigger_merchant_pairs: set[str] = set()


# ---------------------------------------------------------------------------
# /v1/healthz, /v1/metadata
# ---------------------------------------------------------------------------


@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": store.counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": llm.model,
        "approach": "single-prompt Claude composer, generic across trigger kinds; "
        "auto-reply/intent/hostile detection layer on /v1/reply; deterministic "
        "fallback templates if the LLM is unavailable or output fails validation twice.",
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# /v1/context
# ---------------------------------------------------------------------------


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody, response: Response):
    try:
        stored_version = store.push(body.scope, body.context_id, body.version, body.payload)
    except StaleVersionError as e:
        response.status_code = 409
        return {"accepted": False, "reason": "stale_version", "current_version": e.current_version}
    except ValueError:
        response.status_code = 400
        return {"accepted": False, "reason": "invalid_scope", "details": f"unknown scope '{body.scope}'"}

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{stored_version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/teardown")
async def teardown():
    store.wipe()
    _sent_suppression_keys.clear()
    _seen_trigger_merchant_pairs.clear()
    return {"status": "wiped"}


# ---------------------------------------------------------------------------
# /v1/tick
# ---------------------------------------------------------------------------


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions: List[Dict[str, Any]] = []

    for trigger_id in body.available_triggers:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break

        trigger = store.get("trigger", trigger_id)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        if not merchant_id:
            continue

        merchant = store.get("merchant", merchant_id)
        if not merchant:
            continue

        category = store.get("category", merchant.get("category_slug", ""))
        if not category:
            continue

        customer = store.get("customer", customer_id) if customer_id else None

        suppression_key = trigger.get("suppression_key", f"gen:{trigger_id}")
        pair_key = f"{merchant_id}:{trigger_id}"
        if pair_key in _seen_trigger_merchant_pairs:
            # already sent this exact (merchant, trigger); don't resend
            # unless the trigger's own version changed (a version bump
            # implicitly changes the trigger payload, which changes the
            # composer's cache key, so we allow it through if that's why
            # we're here — but we still dedup a plain repeat tick).
            continue

        msg = composer.compose(category, merchant, trigger, customer)

        conversation_id = f"conv_{merchant_id}_{trigger_id}_{uuid.uuid4().hex[:6]}"
        conversations.get_or_create(conversation_id, merchant_id, customer_id, trigger_id)
        conversations.record_bot_send(conversation_id, msg.body)

        _seen_trigger_merchant_pairs.add(pair_key)
        _sent_suppression_keys[suppression_key] = body.now

        actions.append(
            {
                "conversation_id": conversation_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": msg.send_as,
                "trigger_id": trigger_id,
                "template_name": msg.template_name,
                "template_params": msg.template_params,
                "body": msg.body,
                "cta": msg.cta,
                "suppression_key": msg.suppression_key,
                "rationale": msg.rationale,
            }
        )

    return {"actions": actions}


# ---------------------------------------------------------------------------
# /v1/reply
# ---------------------------------------------------------------------------


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    merchant_id = body.merchant_id or ""
    merchant = store.get("merchant", merchant_id) if merchant_id else None
    category = store.get("category", merchant.get("category_slug", "")) if merchant else None

    state = conversations.get_or_create(body.conversation_id, merchant_id, body.customer_id)

    if state.ended:
        return {"action": "end", "rationale": "Conversation already ended."}

    if not merchant or not category:
        # We don't have enough context to respond meaningfully — still must
        # answer within 30s with a valid shape.
        return {
            "action": "end",
            "rationale": "No merchant/category context on file for this conversation; cannot respond meaningfully.",
        }

    result = conversations.decide_reply(state, category, merchant, body.message, body.from_role)
    return result


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
