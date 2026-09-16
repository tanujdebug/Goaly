"""The always-on extractor: runs on every user turn regardless of the
current phase. It is the ONLY place doing free-form language understanding
-- it never decides SOP gating or answer content, it just turns messy text
into structured slots that the deterministic state machine (state_machine.py)
reads. See memory `sop-agent-architecture-plan` decisions #4 and #6."""

from __future__ import annotations

from . import grounding

_SCHEMA_HINT = """
Return ONLY a JSON object with this exact shape (use null for anything not
present in the message):

{
  "pii": {
    "full_name": string|null,
    "dob": string|null,            // normalize to YYYY-MM-DD if possible
    "phone": string|null,          // digits only if possible
    "email": string|null,
    "ssn_last4": string|null       // exactly 4 digits (SSN or national ID last 4)
  },
  "case_type_hint": "healthcare"|"dental"|"auto"|null,
  "status_hint": "denied"|"closed"|"open"|null,
  "month_hint": string|null,       // English month name, e.g. "January"
  "free_text_hint": string|null,   // short paraphrase of what they want, if any
  "rewritten_question": string|null,  // the user's question rewritten as a
                                        // fully explicit standalone question,
                                        // using the conversation context and
                                        // the "current focus" given below
                                        // (e.g. "should I?" + focus on the
                                        // office note -> "Is the treating
                                        // provider office note required for
                                        // my claim?"). null if not a question.
  "topic_guess": one of {topics} | "necessity" | "status_or_denial" | null,
  "document_mentioned": string|null,   // one of {documents}, else raw phrase
  "emotion": "frustrated"|"angry"|"anxious"|"confused"|null,
  "is_representative": boolean,  // true if the caller says or implies they
                                   // are NOT the policyholder and are calling
                                   // on that person's behalf (e.g. a family
                                   // member handling their claim)
  "rep_name": string|null,       // the caller's own name, if they gave it
                                   // while identifying as a representative
  "buyer_name": string|null,     // the policyholder's name they say they're
                                   // calling on behalf of
  "in_scope": boolean,   // false for questions unrelated to this caller's
                          // own insurance claim / policy / this conversation
  "wants_human": boolean,  // true ONLY if the caller explicitly asks to
                             // speak with a human, agent, person, or
                             // representative, or asks to be transferred
                             // (e.g. "let me talk to a person", "transfer
                             // me", "get me a human"). Do NOT set this true
                             // just because the caller states a policy or
                             // claim number, describes their situation,
                             // sounds frustrated, or asks an in-scope
                             // question -- those are normal turns, not a
                             // request for a human.
  "wants_to_end": boolean,
  "affirmation": true|false|null   // true/false if this message is a yes/no
                                     // answer to a prior yes/no question,
                                     // else null
}
""".strip()


def build_extractor_messages(session, user_message: str) -> list[dict]:
    focus = []
    if session.active_case_id:
        focus.append(f"active_case_id={session.active_case_id}")
    if session.active_document:
        focus.append(f"active_document={session.active_document}")
    focus_text = ", ".join(focus) or "none yet"

    schema = (
        _SCHEMA_HINT
        .replace("{topics}", str(grounding.TOPICS))
        .replace("{documents}", str(grounding.DOCUMENT_NAMES))
    )

    system = (
        "You are the extraction layer for an insurance claims support agent. "
        "Read the caller's latest message in context and extract structured "
        "information. Do not answer the caller and do not invent facts -- "
        f"only extract what is actually present or clearly implied.\n\n"
        f"Current phase: {session.phase}\n"
        f"Current focus: {focus_text}\n\n"
        f"{schema}"
    )

    messages = [{"role": "system", "content": system}]
    for turn in session.history[-6:]:
        messages.append(turn)
    if not session.history or session.history[-1].get("content") != user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def extract_turn(llm, session, user_message: str) -> dict:
    messages = build_extractor_messages(session, user_message)
    data = llm.chat_json(messages, temperature=0.1)
    data.setdefault("pii", {})
    data.setdefault("in_scope", True)
    data.setdefault("wants_human", False)
    data.setdefault("wants_to_end", False)
    data.setdefault("affirmation", None)
    data.setdefault("is_representative", False)
    data.setdefault("rep_name", None)
    data.setdefault("buyer_name", None)
    return data
