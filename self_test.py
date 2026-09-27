#!/usr/bin/env python3
"""
self_test.py
============
Pure-stdlib sanity tests for context_store.py, composer.py, validators.py
and conversation.py — no fastapi, no real network call, no anthropic SDK
needed. Uses a MockLLM standing in for AnthropicClient so the full
compose()/decide_reply() pipeline can be exercised offline.

Run: python3 self_test.py
"""

from __future__ import annotations

import json
import sys

from composer import Composer
from context_store import ContextStore, StaleVersionError
from conversation import ConversationManager, is_auto_reply, is_hostile, is_intent_commit
from validators import validate_action

PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name: str, cond: bool, detail: str = ""):
    status = PASS if cond else FAIL
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == FAIL else ""))


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Stands in for AnthropicClient. Returns a canned, well-formed JSON
    response so we can test the pipeline around it without hitting the
    network (this sandbox has no internet access)."""

    def __init__(self, script=None):
        self.model = "mock-model"
        self.calls = 0
        self.script = script or []  # optional list of canned responses

    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, max_tokens=700, temperature=0.0) -> str:
        self.calls += 1
        if self.script:
            idx = min(self.calls - 1, len(self.script) - 1)
            return self.script[idx]
        # Default canned compose-style response
        payload = {
            "body": "Dr. Meera, JIDA's Oct issue landed — 2,100-patient trial shows 3-month "
            "fluoride recall cuts caries 38% better for high-risk adults. Want the abstract "
            "+ a patient-ed draft?",
            "cta": "open_ended",
            "rationale": "Chose the research_digest trigger; used specificity + effort externalization.",
        }
        return json.dumps(payload)


class UnavailableLLM:
    def __init__(self):
        self.model = "none"

    def available(self) -> bool:
        return False

    def complete(self, *a, **kw):
        raise AssertionError("should not be called when unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CATEGORY = {
    "slug": "dentists",
    "voice": {
        "tone": "peer_clinical",
        "vocab_taboo": ["guaranteed", "100% safe", "completely cure", "miracle"],
    },
    "offer_catalog": [{"id": "den_001", "title": "Dental Cleaning @ ₹299"}],
    "digest": [
        {
            "id": "d_2026W17_jida_fluoride",
            "title": "3-month fluoride varnish recall outperforms 6-month",
            "source": "JIDA Oct 2026, p.14",
        }
    ],
}

MERCHANT = {
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "category_slug": "dentists",
    "identity": {"name": "Dr. Meera's Dental Clinic", "owner_first_name": "Meera", "languages": ["en", "hi"]},
    "performance": {"views": 2410, "calls": 18, "ctr": 0.021},
    "offers": [{"id": "o_meera_001", "title": "Dental Cleaning @ ₹299", "status": "active"}],
    "conversation_history": [],
    "signals": ["ctr_below_peer_median"],
}

TRIGGER = {
    "id": "trg_001_research_digest_dentists",
    "kind": "research_digest",
    "source": "external",
    "urgency": 2,
    "suppression_key": "research:dentists:2026-W17",
    "payload": {"category": "dentists", "top_item_id": "d_2026W17_jida_fluoride"},
}


def test_context_store():
    store = ContextStore()
    v = store.push("merchant", "m_001", 1, {"x": 1})
    check("context_store: initial push accepted", v == 1)

    v2 = store.push("merchant", "m_001", 1, {"x": 1})
    check("context_store: re-post same version is no-op", v2 == 1)

    v3 = store.push("merchant", "m_001", 2, {"x": 2})
    check("context_store: higher version replaces", v3 == 2 and store.get("merchant", "m_001")["x"] == 2)

    try:
        store.push("merchant", "m_001", 1, {"x": 99})  # now genuinely stale (current=2)
        check("context_store: stale lower version raises", False, "did not raise")
    except StaleVersionError as e:
        check("context_store: stale lower version raises", e.current_version == 2)

    try:
        store.push("bogus_scope", "id", 1, {})
        check("context_store: invalid scope rejected", False)
    except ValueError:
        check("context_store: invalid scope rejected", True)

    counts = store.counts()
    check("context_store: counts reflect pushes", counts["merchant"] == 1, str(counts))


def test_validators():
    ok, issues = validate_action("Great! Guaranteed results every time.", "open_ended", ["guaranteed"])
    check("validators: taboo word is hard fail", not ok, str(issues))

    ok2, issues2 = validate_action("Hi Dr. Meera, quick one — want the abstract?", "open_ended", ["guaranteed"])
    check("validators: clean message passes", ok2, str(issues2))

    ok3, issues3 = validate_action("", "open_ended", [])
    check("validators: empty body fails", not ok3)

    ok4, issues4 = validate_action("Same message", "open_ended", [], prior_bodies=["Same message"])
    check("validators: exact repeat fails", not ok4, str(issues4))


def test_composer_with_llm():
    composer = Composer(llm=MockLLM())
    msg = composer.compose(CATEGORY, MERCHANT, TRIGGER, customer=None)
    check("composer: body non-empty", bool(msg.body.strip()))
    check("composer: cta valid", msg.cta in {"binary", "open_ended", "none"}, msg.cta)
    check("composer: send_as=vera with no customer", msg.send_as == "vera")
    check("composer: suppression_key passthrough", msg.suppression_key == TRIGGER["suppression_key"])
    check("composer: specificity present (2,100 / 38%)", "2,100" in msg.body and "38%" in msg.body, msg.body)
    check("composer: no taboo words leaked", "guaranteed" not in msg.body.lower())
    check("composer: rationale non-empty", bool(msg.rationale.strip()))
    check("composer: first message gets template fields", msg.template_name is not None)

    # cache hit — identical inputs should not trigger a second LLM call
    llm = MockLLM()
    composer2 = Composer(llm=llm)
    composer2.compose(CATEGORY, MERCHANT, TRIGGER, customer=None)
    composer2.compose(CATEGORY, MERCHANT, TRIGGER, customer=None)
    check("composer: identical inputs are cached (1 LLM call for 2 composes)", llm.calls == 1, str(llm.calls))


def test_composer_customer_facing():
    customer = {
        "customer_id": "c_001_priya",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "identity": {"name": "Priya", "language_pref": "hi-en mix"},
        "state": "lapsed_soft",
    }
    recall_trigger = {
        "id": "trg_003_recall_due_priya",
        "kind": "recall_due",
        "source": "internal",
        "urgency": 3,
        "suppression_key": "recall:c_001_priya:6mo",
        "payload": {
            "service_due": "6_month_cleaning",
            "available_slots": [{"label": "Wed 5 Nov, 6pm"}, {"label": "Thu 6 Nov, 5pm"}],
        },
    }
    script = [json.dumps({
        "body": "Hi Priya, Dr. Meera's clinic here. Your 6-month cleaning recall is due — "
        "Wed 5 Nov 6pm ya Thu 6 Nov 5pm, dono slots free hain. Reply 1 for Wed, 2 for Thu.",
        "cta": "binary",
        "rationale": "Customer-facing recall; used real slots, binary CTA.",
    })]
    composer = Composer(llm=MockLLM(script=script))
    msg = composer.compose(CATEGORY, MERCHANT, recall_trigger, customer=customer)
    check("composer: send_as=merchant_on_behalf with customer present", msg.send_as == "merchant_on_behalf")
    check("composer: customer-facing message references real slot", "5 Nov" in msg.body or "6 Nov" in msg.body, msg.body)


def test_composer_fallback_no_llm():
    composer = Composer(llm=UnavailableLLM())
    msg = composer.compose(CATEGORY, MERCHANT, TRIGGER, customer=None)
    check("composer: fallback body non-empty when LLM unavailable", bool(msg.body.strip()))
    check("composer: fallback flags llm_unavailable", "llm_unavailable" in msg.issues, str(msg.issues))


def test_conversation_detectors():
    check("conversation: detects auto-reply phrase", is_auto_reply([], "Thank you for contacting us, our team will get back to you."))
    check("conversation: detects repeated verbatim as auto-reply", is_auto_reply(["Hello there"], "Hello there"))
    check("conversation: does not flag normal reply as auto-reply", not is_auto_reply([], "Yes, send me the abstract"))

    check("conversation: detects intent commit (english)", is_intent_commit("Ok let's do it, sign me up"))
    check("conversation: detects intent commit (hinglish)", is_intent_commit("Haan theek hai, chalo start karo"))
    check("conversation: does not flag question as intent commit", not is_intent_commit("What would that look like?"))

    check("conversation: detects hostile", is_hostile("Stop messaging me. This is useless spam."))
    check("conversation: does not flag neutral msg as hostile", not is_hostile("Not interested right now, thanks"))


def test_conversation_manager_flow():
    mgr = ConversationManager(llm=MockLLM())
    state = mgr.get_or_create("conv_1", "m_001_drmeera_dentist_delhi", None, "trg_001")

    # first auto-reply -> one more attempt (send), not end
    r1 = mgr.decide_reply(state, CATEGORY, MERCHANT, "Thank you for contacting us, team will get back to you.")
    check("conv flow: first auto-reply -> send (one more attempt)", r1["action"] == "send", str(r1))

    # second auto-reply (verbatim repeat) -> end
    r2 = mgr.decide_reply(state, CATEGORY, MERCHANT, "Thank you for contacting us, team will get back to you.")
    check("conv flow: second auto-reply -> end", r2["action"] == "end", str(r2))
    check("conv flow: state marked ended", state.ended)

    # fresh conversation: intent commit -> send, action-mode instruction passed
    mgr2 = ConversationManager(llm=MockLLM(script=[json.dumps({
        "body": "Great — I'm setting up your listing now. Just confirm your business hours and I'll publish it.",
        "cta": "open_ended",
        "rationale": "Merchant committed; moved straight to action.",
    })]))
    state2 = mgr2.get_or_create("conv_2", "m_002", None, "trg_x")
    r3 = mgr2.decide_reply(state2, CATEGORY, MERCHANT, "Yes let's do it, sign me up")
    check("conv flow: intent commit -> send", r3["action"] == "send", str(r3))
    check("conv flow: intent commit reply doesn't re-qualify", "would you" not in r3["body"].lower())

    # hostile -> end
    mgr3 = ConversationManager(llm=MockLLM())
    state3 = mgr3.get_or_create("conv_3", "m_003", None, "trg_y")
    r4 = mgr3.decide_reply(state3, CATEGORY, MERCHANT, "Stop messaging me. This is useless spam.")
    check("conv flow: hostile -> end", r4["action"] == "end", str(r4))
    check("conv flow: hostile end has apology/stop language", "sorry" in r4["body"].lower() or "won't" in r4["body"].lower())


def main():
    test_context_store()
    test_validators()
    test_composer_with_llm()
    test_composer_customer_facing()
    test_composer_fallback_no_llm()
    test_conversation_detectors()
    test_conversation_manager_flow()

    n_fail = sum(1 for _, status, _ in _results if status == FAIL)
    n_total = len(_results)
    print(f"\n{n_total - n_fail}/{n_total} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
