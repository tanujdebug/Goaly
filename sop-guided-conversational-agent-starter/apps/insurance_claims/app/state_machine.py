"""The SOP state machine. This module owns every phase transition and every
gating decision -- the LLM (via extraction.py for understanding and
prompts.py + llm_client for phrasing) never decides on its own whether to
advance a phase or disclose claim data. See memory `sop-agent-architecture-
plan` for the full design rationale.
"""

from __future__ import annotations

import re

from . import extraction, grounding, prompts

OFF_TOPIC_ESCALATION_THRESHOLD = 3
FRUSTRATION_ESCALATION_THRESHOLD = 4
CONSENT_TIMEOUT_THRESHOLD = 4

# Code-level backstop for VERIFY_ID: some models don't reliably follow the
# prompt's instructions not to claim premature verification, reference a
# claim, or invent contact/submission details. If a free-text reply from the
# LLM matches any of these while the caller is still unverified, it is
# discarded in favor of prompts.verify_id_safe_fallback() -- a deterministic
# message that cannot hallucinate. See memory `sop-agent-architecture-plan`.
_VERIFY_GATE_BREACH_PATTERNS = [
    re.compile(r"\bCL-\d+\b", re.I),                       # referencing a specific claim id
    re.compile(r"claim number|policy number", re.I),        # soliciting/acting on case identifiers this early
    re.compile(r"pull(?:ed|ing)? up|pull that up|look(?:ing)? into (?:that|this|your) claim|"
               r"review(?:ing)? (?:that|this|your) claim|found (?:your|the) claim", re.I),
    re.compile(r"i(?:'|’)ll send|i will send|you should receive|"
               r"forward(?:ed)? your request|claims team|sent to (?:your|the) email", re.I),
    re.compile(r"you(?:'|’)re verified|your identity is verified|verification (?:is )?complete|"
               r"confirmed your identity|confirming your identity", re.I),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                # any email address in the reply
    re.compile(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"),        # any phone/fax number in the reply
    re.compile(r"member portal|upload the (?:pdf|document|file|report)|"
               r"fax (?:the|it|to)|mail a hard copy", re.I),
]


def _breaches_verify_gate(reply: str) -> bool:
    return any(p.search(reply) for p in _VERIFY_GATE_BREACH_PATTERNS)


def process_message(session, user_message: str, llm) -> str:
    session.history.append({"role": "user", "content": user_message})

    if session.phase == "ENDED":
        reply = prompts.ended_message()
        session.history.append({"role": "assistant", "content": reply})
        return reply

    if session.phase == "HUMAN_ESCALATION":
        reply = prompts.human_escalation_message(session)
        session.history.append({"role": "assistant", "content": reply})
        return reply

    data = extraction.extract_turn(llm, session, user_message)
    _apply_extracted_slots(session, data)

    if data.get("wants_human"):
        session.phase = "HUMAN_ESCALATION"
        session.escalated = True
        session.escalation_reason = "caller requested a human"
        reply = prompts.human_escalation_message(session, requested=True)
        session.history.append({"role": "assistant", "content": reply})
        return reply

    # A caller can announce mid-conversation that they're actually someone
    # else calling on the policyholder's behalf -- even in a session that
    # was already verified as the policyholder (or as a different rep). Any
    # prior verification was for a *different speaker* and must not carry
    # over: re-gate on consent immediately, regardless of current phase,
    # before any further case data is disclosed. This check only lives here
    # (not inside _handle_verify_id) because _handle_verify_id is never
    # reached again once phase has moved past VERIFY_ID.
    if data.get("is_representative") and not session.is_representative:
        reply = _handle_new_representative_claim(session, data, llm)
        session.history.append({"role": "assistant", "content": reply})
        return reply

    if not data.get("in_scope", True):
        session.off_topic_count += 1
        if session.off_topic_count >= OFF_TOPIC_ESCALATION_THRESHOLD:
            session.phase = "HUMAN_ESCALATION"
            session.escalated = True
            session.escalation_reason = "repeated out-of-scope questions"
            reply = prompts.human_escalation_message(session, off_topic=True)
        else:
            reply = llm.chat(_build_messages(
                session, prompts.out_of_scope_system(session, session.off_topic_count)))
        session.history.append({"role": "assistant", "content": reply})
        return reply

    session.off_topic_count = 0

    emotion = data.get("emotion")
    session.last_emotion = emotion
    if emotion in ("frustrated", "angry", "anxious", "confused"):
        session.frustration_count += 1
    else:
        session.frustration_count = max(0, session.frustration_count - 1)

    handler = _PHASE_HANDLERS[session.phase]
    reply = handler(session, user_message, data, llm)
    session.history.append({"role": "assistant", "content": reply})
    return reply


def _apply_extracted_slots(session, data: dict) -> None:
    pii = data.get("pii") or {}
    for field in grounding.PII_FIELDS:
        value = pii.get(field)
        if value:
            session.pii_slots[field] = value

    if data.get("case_type_hint"):
        session.case_type_hint = data["case_type_hint"]
    if data.get("status_hint"):
        session.status_hint = data["status_hint"]
    if data.get("month_hint"):
        session.month_hint = data["month_hint"]
    if data.get("free_text_hint"):
        session.free_text_hint = data["free_text_hint"]
    if data.get("document_mentioned"):
        session.active_document = data["document_mentioned"]


def _build_messages(session, system: str) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    messages.extend(session.history[-10:])
    return messages


# ---------------------------------------------------------------------------
# VERIFY_ID
# ---------------------------------------------------------------------------

def _handle_new_representative_claim(session, data: dict, llm) -> str:
    """A caller has just announced (for the first time in this session) that
    they're calling on someone else's behalf. Resets any prior verification
    for this session -- it applied to a different speaker -- and routes
    through the same recognized-rep + consent gate used pre-verification, no
    matter what phase the session was previously in."""
    rep = grounding.match_representative(
        rep_name=data.get("rep_name"), buyer_name=data.get("buyer_name"),
    )
    if not rep:
        return llm.chat(_build_messages(session, prompts.representative_unrecognized_system(
            session, data.get("buyer_name"), session.last_emotion,
        )))

    session.is_representative = True
    session.rep_name = rep["rep_name"]
    session.rep_buyer_party_id = rep["buyer_party_id"]
    session.consent_status = "pending"
    session.consent_checks = 0
    session.verified = False
    session.party_id = None
    session.active_case_id = None
    session.phase = "VERIFY_ID"
    return _handle_verify_id(session, "", data, llm)


def _handle_verify_id(session, user_message: str, data: dict, llm) -> str:
    session.verification_attempts += 1

    if session.frustration_count >= FRUSTRATION_ESCALATION_THRESHOLD:
        session.phase = "HUMAN_ESCALATION"
        session.escalated = True
        session.escalation_reason = "sustained frustration during identity verification"
        return prompts.human_escalation_message(session)

    # Representative/consent sub-flow: a non-policyholder caller acting on
    # someone else's behalf is gated on that policyholder's consent, not on
    # PII match -- a family member won't necessarily know the policyholder's
    # SSN or DOB, so the standard verify_identity() path doesn't apply here.
    # (The initial detection -- first time a caller claims to be a rep --
    # happens in process_message via _handle_new_representative_claim, since
    # that must intercept regardless of phase, not just while phase ==
    # VERIFY_ID. This block only continues an already-started consent poll.)
    if session.is_representative and session.consent_status != "approved":
        session.consent_checks += 1
        status = grounding.next_consent_status(session.consent_scenario, session.consent_checks)
        session.consent_status = status

        if status == "approved":
            session.verified = True
            session.party_id = session.rep_buyer_party_id
            session.phase = "RESOLVE_INTENT"
            holder = grounding.get_policyholder(session.party_id)
            note = (f"The policyholder ({holder['name'].split()[0]}) just granted "
                    f"consent for {session.rep_name} to discuss the account.")
            return _handle_resolve_intent(session, user_message, data, llm, transition_note=note)

        if session.consent_checks >= CONSENT_TIMEOUT_THRESHOLD:
            session.phase = "HUMAN_ESCALATION"
            session.escalated = True
            session.escalation_reason = "consent request timed out"
            return prompts.human_escalation_message(session)

        return llm.chat(_build_messages(session, prompts.consent_pending_system(
            session, session.last_emotion, session.consent_checks,
        )))

    party_id, matched, ambiguous = grounding.verify_identity(session.pii_slots)

    if party_id:
        session.verified = True
        session.party_id = party_id
        session.phase = "RESOLVE_INTENT"
        holder = grounding.get_policyholder(party_id)
        note = f"You just confirmed the caller's identity ({holder['name'].split()[0]})."
        return _handle_resolve_intent(session, user_message, data, llm, transition_note=note)

    missing = [f for f in grounding.PII_FIELDS if f not in matched]
    system = prompts.verify_id_system(
        session, matched_fields=matched, missing_fields=missing,
        ambiguous=ambiguous, emotion=session.last_emotion,
        attempts=session.verification_attempts,
    )
    reply = llm.chat(_build_messages(session, system))
    if _breaches_verify_gate(reply):
        reply = prompts.verify_id_safe_fallback(matched, missing, ambiguous)
    return reply


# ---------------------------------------------------------------------------
# RESOLVE_INTENT
# ---------------------------------------------------------------------------

def _handle_resolve_intent(session, user_message: str, data: dict, llm,
                            transition_note: str | None = None) -> str:
    if session.pending_case_confirmation:
        affirmation = data.get("affirmation")
        if affirmation is True:
            session.active_case_id = session.pending_case_confirmation
            session.pending_case_confirmation = None
            session.resolved_intent = session.resolved_intent or "general_claim_question"
            session.phase = "PROCESS_CASE"
            return _handle_process_case(session, user_message, data, llm,
                                         transition_note=transition_note)
        if affirmation is False:
            session.pending_case_confirmation = None
            session.case_type_hint = None
            session.status_hint = None
            session.month_hint = None
            # fall through to re-resolve from the full claim list below

    if not session.active_case_id and not session.pending_case_confirmation:
        case_id, candidates = grounding.match_case_by_hints(
            session.party_id, session.case_type_hint, session.status_hint, session.month_hint,
        )
        if case_id:
            session.pending_case_confirmation = case_id
            case = grounding.get_claim(case_id)
            system = prompts.resolve_intent_confirm_system(session, case, transition_note)
            return llm.chat(_build_messages(session, system))
        if len(candidates) > 1:
            system = prompts.resolve_intent_disambiguate_system(session, candidates, transition_note)
            return llm.chat(_build_messages(session, system))

    all_claims = grounding.get_claims_for_party(session.party_id)
    system = prompts.resolve_intent_open_system(session, all_claims, transition_note)
    return llm.chat(_build_messages(session, system))


# ---------------------------------------------------------------------------
# PROCESS_CASE
# ---------------------------------------------------------------------------

def _handle_process_case(session, user_message: str, data: dict, llm,
                          transition_note: str | None = None) -> str:
    case = grounding.get_claim(session.active_case_id) if session.active_case_id else None
    if not case:
        session.phase = "RESOLVE_INTENT"
        session.active_case_id = None
        return _handle_resolve_intent(session, user_message, data, llm)

    if data.get("wants_to_end"):
        session.phase = "POST_PROCESS"
        return _handle_post_process(session, user_message, data, llm,
                                     transition_note="The caller is ready to wrap up.")

    topic = data.get("topic_guess")
    doc = session.active_document
    grounded_snippets: list[str] = []

    if topic == "necessity" and doc:
        required = grounding.is_document_required(case, doc)
        grounded_snippets.append(
            f"Is '{doc}' required for claim {case['case_id']}? {required}. "
            f"Denial reason on file: {case.get('denial_reason', 'n/a')}. "
            f"All outstanding documents: {', '.join(case.get('documents_needed', [])) or 'none currently outstanding'}."
        )
        grounded_snippets.append(grounding.get_document_guidance(doc, case.get("case_type")))
    elif topic in grounding.TOPICS:
        answer = grounding.get_followup_answer(topic, case)
        if answer:
            grounded_snippets.append(answer)
        if doc:
            grounded_snippets.append(grounding.get_document_guidance(doc, case.get("case_type")))
    elif topic == "document_alternative" and doc:
        grounded_snippets.append(grounding.get_document_alternative(doc))
    else:
        grounded_snippets.append(grounding.case_fact_sheet(case))
        if doc:
            grounded_snippets.append(grounding.get_document_guidance(doc, case.get("case_type")))

    if not grounded_snippets:
        grounded_snippets.append(grounding.get_followup_fallback())

    system = prompts.process_case_system(session, case, grounded_snippets, transition_note)
    return llm.chat(_build_messages(session, system))


# ---------------------------------------------------------------------------
# POST_PROCESS
# ---------------------------------------------------------------------------

def _build_summary(session, llm) -> str:
    case = grounding.get_claim(session.active_case_id) if session.active_case_id else None
    facts = grounding.case_fact_sheet(case) if case else "No specific claim was resolved."
    system = (
        "Summarize this insurance claims support call in 3-5 short bullet "
        "points: what was discussed, the claim status/outcome, and the "
        "major follow-up items or next steps. Use ONLY these known facts "
        f"for the claim, do not invent anything:\n{facts}"
    )
    return llm.chat(_build_messages(session, system), temperature=0.2)


def _handle_post_process(session, user_message: str, data: dict, llm,
                          transition_note: str | None = None) -> str:
    holder = grounding.get_policyholder(session.party_id) if session.party_id else None
    email = holder.get("email") if holder else "your email on file"

    if not session.email_offered:
        session.email_offered = True
        session.summary_text = _build_summary(session, llm)
        system = prompts.post_process_offer_system(session, session.summary_text, email, transition_note)
        return llm.chat(_build_messages(session, system))

    affirmation = data.get("affirmation")
    if affirmation is True and not session.email_sent:
        session.email_sent = True
        session.phase = "ENDED"
        system = prompts.post_process_sent_system(session, email)
        return llm.chat(_build_messages(session, system))
    if affirmation is False:
        session.phase = "ENDED"
        system = prompts.post_process_skip_system(session)
        return llm.chat(_build_messages(session, system))

    system = prompts.post_process_clarify_system(session)
    return llm.chat(_build_messages(session, system))


_PHASE_HANDLERS = {
    "VERIFY_ID": _handle_verify_id,
    "RESOLVE_INTENT": _handle_resolve_intent,
    "PROCESS_CASE": _handle_process_case,
    "POST_PROCESS": _handle_post_process,
}


def debug_snapshot(session) -> dict:
    return {
        "phase": session.phase,
        "verified": session.verified,
        "party_id": session.party_id,
        "matched_pii_fields": sorted(grounding.verify_identity(session.pii_slots)[1]),
        "case_type_hint": session.case_type_hint,
        "status_hint": session.status_hint,
        "month_hint": session.month_hint,
        "active_case_id": session.active_case_id,
        "active_document": session.active_document,
        "off_topic_count": session.off_topic_count,
        "frustration_count": session.frustration_count,
        "last_emotion": session.last_emotion,
        "is_representative": session.is_representative,
        "rep_name": session.rep_name,
        "consent_status": session.consent_status,
        "escalated": session.escalated,
        "email_offered": session.email_offered,
        "email_sent": session.email_sent,
    }
