"""
composer.py
===========
The heart of the bot: compose(category, merchant, trigger, customer=None)
-> ComposedMessage.

Design choice: rather than hardcoding a prompt template per trigger `kind`,
the composer is generic — it serializes the full 4-layer context and asks
Claude to (a) pick the ONE signal that should drive the message, (b) ground
every claim in the given data, (c) match category voice, and (d) use one
compulsion lever with a single low-friction CTA. This matters because the
challenge's own FAQ is explicit that the real judge harness injects trigger
kinds and scenarios candidates haven't seen — a bot with per-kind templates
would silently degrade on those; a bot that reasons generically from
whatever context it's given does not.

A small `TRIGGER_FRAMING` table still gives light-touch hints for known
kinds (mostly: "is this external news or an internal number?", "does this
concern the merchant or one of their customers?") — think of it as
few-shot flavoring, not branching logic. Unknown kinds fall back to a
generic framing so new trigger kinds degrade gracefully rather than
breaking.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_client import AnthropicClient, LLMError
from validators import validate_action

# --------------------------------------------------------------------------
# Light-touch per-kind framing hints (not required, just quality boosters)
# --------------------------------------------------------------------------

TRIGGER_FRAMING: Dict[str, str] = {
    "research_digest": "External research/knowledge item. Anchor on the specific finding + who it's relevant to in this merchant's patient/customer mix.",
    "regulation_change": "A compliance deadline. Lead with the deadline and what changes, not hype.",
    "recall_due": "A specific customer's recall window has opened. This is a booking nudge sent on the merchant's behalf — offer the real available slots, nothing invented.",
    "perf_dip": "An internal metric dropped. Frame as loss aversion — a concrete number, not vague concern.",
    "perf_spike": "An internal metric rose. Frame as a moment to double down (repost, extend an offer) — congratulatory but brief, not hype.",
    "renewal_due": "Subscription renewal approaching. Be direct about days remaining and plan; no scare tactics.",
    "festival_upcoming": "A festival is coming. Relevant only if it fits this category — suggest a concrete, category-appropriate offer angle.",
    "milestone_reached": "Close to a milestone (e.g. review count). Light, celebratory nudge with the exact number.",
    "dormant_with_vera": "Merchant hasn't engaged in a while. Re-open low-friction, referencing what was last discussed if known.",
    "review_theme_emerged": "A recurring review theme (positive or negative) has emerged. Surface it factually and ask if they want to act on it.",
    "competitor_opened": "A nearby competitor opened with a stated offer. Inform, don't panic — give the merchant a clear next move.",
    "customer_lapsed_soft": "A customer's engagement is softly lapsing. Gentle winback, no hard sell.",
    "customer_lapsed_hard": "A customer has been gone a long time. Winback framed around what they valued before (their previous focus/service), not generic discounting.",
    "appointment_tomorrow": "A booking exists for tomorrow. Simple, useful confirmation/reminder.",
    "chronic_refill_due": "A pharmacy customer's regular medication is about to run out. Practical, no medical claims, offer the concrete next step (refill/delivery).",
    "trial_followup": "Following up after a trial session/visit. Offer the next concrete session slot.",
    "curious_ask_due": "A scheduled, low-stakes curiosity check-in with the merchant (e.g. 'what's in demand this week'). Keep it short and genuinely conversational.",
    "wedding_package_followup": "A bridal/event package customer is approaching their event date. Suggest the next relevant step in their prep program.",
    "winback_eligible": "The merchant's subscription lapsed a while ago and they're eligible for winback. Be direct about what they're missing (with real numbers) and the simple path back.",
    "ipl_match_today": "A live sports event is happening today near this merchant. Only relevant for merchants where match-day footfall matters (e.g. restaurants) — suggest a timely, specific angle.",
    "active_planning_intent": "The merchant is mid-conversation about something specific they asked for — continue that exact thread, don't restart with qualifying questions.",
    "seasonal_perf_dip": "An expected seasonal dip, not a red flag. Frame reassuringly with a concrete counter-move if one is available.",
    "supply_alert": "A regulatory/supply recall affecting specific batches or a specific molecule. Highest urgency — lead with the actionable fact (which batches/what to do).",
    "category_seasonal": "A seasonal demand shift for the whole category. Suggest a concrete shelf/stock/service action tied to the specific trend numbers given.",
    "gbp_unverified": "Merchant's Google Business Profile isn't verified yet. Concrete ask: what verification path, what uplift they're leaving on the table.",
    "cde_opportunity": "A professional development / CE opportunity (e.g. webinar). Light, informational, easy yes/no.",
    "scheduled_recurring": "A recurring, low-stakes cadence touchpoint. Keep it genuinely light — this should feel like a check-in, not a sales push.",
}

DEFAULT_FRAMING = (
    "Use the trigger's kind, payload, urgency and source (external/internal) to infer "
    "why this message should go out now. Do not invent meaning the payload doesn't support."
)


@dataclass
class ComposedMessage:
    body: str
    cta: str  # "binary" | "open_ended" | "none"
    send_as: str  # "vera" | "merchant_on_behalf"
    suppression_key: str
    rationale: str
    template_name: Optional[str] = None
    template_params: Optional[List[str]] = None
    issues: List[str] = field(default_factory=list)  # validator flags, for debugging/logs


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the message-composition engine behind Vera, magicpin's WhatsApp merchant-growth \
assistant for Indian local businesses. You write ONE outbound WhatsApp message from a structured \
context bundle (category + merchant + trigger + optional customer).

HARD RULES (violating any of these is a severe scoring penalty):
1. Ground every specific claim (number, date, name, offer, source) ONLY in the context given. \
Never invent a statistic, offer, competitor name, or citation that isn't present in the input. \
If you don't have a number to anchor on, don't invent one — use a qualitative but still specific hook instead.
2. Exactly ONE call-to-action, and it should be the last sentence of the message, not buried. \
Binary yes/no ("Reply 1 for X, 2 for Y" for booking/slot choices is fine) or one clear open-ended ask. \
Never stack multiple asks.
3. Match category voice exactly: dentists/doctors = clinical peer tone, technical vocabulary OK; \
salons = warm and practical; restaurants = operator-to-operator; gyms = coaching/motivational; \
pharmacies = precise and trustworthy. Avoid every word in the category's taboo/vocab_taboo list.
4. Prefer service+price specificity ("Haircut @ ₹99") over generic discount framing ("10% off"). \
Use the merchant's actual offer_catalog / active offers — never a price that isn't in the data.
5. Match the audience's language preference. If Hindi is in scope (merchant identity.languages includes \
"hi", or a customer's language_pref says "hi" or "hi-en mix"), write natural Hindi-English code-mix \
(Devanagari not required — Roman Hindi is fine and common on WhatsApp). Otherwise write in English.
6. No promotional hype ("AMAZING DEAL!!!"), no long preamble ("I hope you're doing well..."), no \
re-introducing yourself if this isn't the first message in the conversation.
7. Use at least one compulsion lever explicitly: specificity/verifiability, loss aversion, social proof, \
effort externalization ("I've already drafted X"), curiosity, reciprocity, or asking the merchant a real \
question. The two most underused (per production data) are social proof and asking the merchant — prefer \
these when the context supports them.
8. If a customer context is present, you are drafting a message the MERCHANT sends to their OWN customer \
(send_as = merchant_on_behalf) — write in the merchant's voice to their patient/client, not Vera's voice \
to the merchant. Respect the customer's consent scope; don't reference anything outside what they \
consented to.
9. Keep it WhatsApp-length: concise, scannable, no walls of text. No hard character cap, but every \
sentence should earn its place.

You must respond with ONLY a single JSON object, no markdown fences, no commentary, in exactly this shape:
{"body": "<the message text>", "cta": "binary" | "open_ended" | "none", "rationale": "<1-2 sentences: \
which single signal you chose to drive this message, and why, and which compulsion lever(s) you used>"}
"""


def _find_digest_item(category: dict, trigger_payload: dict) -> Optional[dict]:
    digest = category.get("digest", []) if category else []
    for key in ("top_item_id", "digest_item_id", "alert_id"):
        item_id = trigger_payload.get(key)
        if not item_id:
            continue
        for item in digest:
            if item.get("id") == item_id:
                return item
    return None


def _active_offers(merchant: dict) -> List[dict]:
    return [o for o in merchant.get("offers", []) if o.get("status") == "active"]


def _conversation_tail(merchant: dict, n: int = 4) -> List[dict]:
    hist = merchant.get("conversation_history", []) or []
    return hist[-n:]


def _language_hint(merchant: dict, customer: Optional[dict]) -> str:
    if customer:
        return customer.get("identity", {}).get("language_pref", "english")
    langs = merchant.get("identity", {}).get("languages", [])
    return "hi-en mix" if "hi" in langs else "english"


def build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    conversation_tail: Optional[List[dict]] = None,
) -> str:
    payload = trigger.get("payload", {}) or {}
    digest_item = _find_digest_item(category, payload)
    framing = TRIGGER_FRAMING.get(trigger.get("kind", ""), DEFAULT_FRAMING)

    bundle: Dict[str, Any] = {
        "category": {
            "slug": category.get("slug"),
            "voice": category.get("voice", {}),
            "offer_catalog": category.get("offer_catalog", []),
            "peer_stats": category.get("peer_stats", {}),
            "seasonal_beats": category.get("seasonal_beats", []),
            "trend_signals": category.get("trend_signals", []),
        },
        "merchant": {
            "identity": merchant.get("identity", {}),
            "subscription": merchant.get("subscription", {}),
            "performance": merchant.get("performance", {}),
            "active_offers": _active_offers(merchant),
            "customer_aggregate": merchant.get("customer_aggregate", {}),
            "signals": merchant.get("signals", []),
            "review_themes": merchant.get("review_themes", []),
        },
        "trigger": {
            "kind": trigger.get("kind"),
            "source": trigger.get("source"),
            "urgency": trigger.get("urgency"),
            "payload": payload,
        },
        "digest_item_referenced_by_trigger": digest_item,
        "customer": customer,
        "recent_conversation": conversation_tail or _conversation_tail(merchant),
        "language_hint": _language_hint(merchant, customer),
        "is_first_message_in_conversation": not bool(conversation_tail or _conversation_tail(merchant)),
    }

    return (
        f"TRIGGER FRAMING HINT: {framing}\n\n"
        f"CONTEXT BUNDLE (JSON):\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Compose the single best message now. Respond with only the JSON object described "
        "in the system prompt."
    )


# --------------------------------------------------------------------------
# Parsing + fallback
# --------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    match = _JSON_BLOCK.search(raw)
    if not match:
        raise ValueError("no_json_found")
    return json.loads(match.group())


def _fallback_body(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    """Deterministic, non-LLM template used only if the LLM is unavailable
    or its output fails validation twice. Keeps the endpoint from ever
    returning an empty/malformed action, which the judge penalizes harder
    than a plain-but-valid message."""
    name = merchant.get("identity", {}).get("name", "there")
    owner = merchant.get("identity", {}).get("owner_first_name")
    salutation = f"Dr. {owner}" if owner and category.get("slug") == "dentists" else (owner or name)
    kind = trigger.get("kind", "update")
    offers = _active_offers(merchant)
    offer_line = f" You currently have {offers[0]['title']} live." if offers else ""
    if customer:
        cust_name = customer.get("identity", {}).get("name", "there")
        return (
            f"Hi {cust_name}, this is {name} — following up on a {kind.replace('_', ' ')}."
            f"{offer_line} Let us know if you'd like to book a slot."
        )
    return (
        f"{salutation}, quick update: a {kind.replace('_', ' ')} signal just came in for your account."
        f"{offer_line} Want me to put together next steps?"
    )


# --------------------------------------------------------------------------
# Composer
# --------------------------------------------------------------------------


class Composer:
    def __init__(self, llm: Optional[AnthropicClient] = None) -> None:
        self.llm = llm or AnthropicClient()
        self._cache: Dict[str, ComposedMessage] = {}

    def _cache_key(self, category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
        raw = json.dumps(
            {"c": category, "m": merchant, "t": trigger, "cu": customer},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def compose(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: Optional[dict] = None,
        conversation_tail: Optional[List[dict]] = None,
        prior_bodies: Optional[List[str]] = None,
    ) -> ComposedMessage:
        cache_key = self._cache_key(category, merchant, trigger, customer)
        if cache_key in self._cache:
            return self._cache[cache_key]

        taboo = category.get("voice", {}).get("vocab_taboo", [])
        suppression_key = trigger.get("suppression_key", f"gen:{trigger.get('id', '')}")
        send_as = "merchant_on_behalf" if customer else "vera"
        is_first = not bool(conversation_tail or _conversation_tail(merchant))

        body_text, cta, rationale, issues = self._compose_via_llm(
            category, merchant, trigger, customer, conversation_tail, taboo, prior_bodies or [], is_first
        )

        msg = ComposedMessage(
            body=body_text,
            cta=cta,
            send_as=send_as,
            suppression_key=suppression_key,
            rationale=rationale,
            issues=issues,
        )

        if is_first:
            msg.template_name = f"vera_{trigger.get('kind', 'generic')}_v1"
            msg.template_params = [
                merchant.get("identity", {}).get("name", ""),
                trigger.get("kind", ""),
                cta,
            ]

        self._cache[cache_key] = msg
        return msg

    def _compose_via_llm(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: Optional[dict],
        conversation_tail: Optional[List[dict]],
        taboo: List[str],
        prior_bodies: List[str],
        is_first: bool,
    ) -> tuple[str, str, str, List[str]]:
        if not self.llm.available():
            return (
                _fallback_body(category, merchant, trigger, customer),
                "open_ended",
                "LLM unavailable — deterministic fallback template used.",
                ["llm_unavailable"],
            )

        user_prompt = build_user_prompt(category, merchant, trigger, customer, conversation_tail)

        for attempt in range(2):  # one retry if validation fails
            try:
                raw = self.llm.complete(SYSTEM_PROMPT, user_prompt, max_tokens=700, temperature=0.0)
                parsed = _parse_llm_json(raw)
                body = str(parsed.get("body", "")).strip()
                cta = str(parsed.get("cta", "open_ended")).strip()
                rationale = str(parsed.get("rationale", "")).strip()
            except (LLMError, ValueError, json.JSONDecodeError) as e:
                if attempt == 0:
                    continue
                return (
                    _fallback_body(category, merchant, trigger, customer),
                    "open_ended",
                    f"LLM call/parse failed ({e}); deterministic fallback used.",
                    ["llm_failed"],
                )

            ok, issues = validate_action(body, cta, taboo, prior_bodies, is_first)
            if ok:
                return body, cta if cta in {"binary", "open_ended", "none"} else "open_ended", rationale, issues
            if attempt == 0:
                # one retry, nudging the model away from the specific issue
                user_prompt += (
                    f"\n\nYour previous attempt had issues: {issues}. "
                    "Regenerate, fixing these specifically. Same JSON format only."
                )
                continue
            # second failure: fall back rather than send something bad
            return (
                _fallback_body(category, merchant, trigger, customer),
                "open_ended",
                f"LLM output failed validation twice ({issues}); fallback used.",
                issues,
            )

        # unreachable, but keeps type-checkers happy
        return (
            _fallback_body(category, merchant, trigger, customer),
            "open_ended",
            "Unexpected composer fallthrough.",
            ["fallthrough"],
        )
