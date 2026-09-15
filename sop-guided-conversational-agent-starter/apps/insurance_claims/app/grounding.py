"""Structured retrieval over the fixture data. No embeddings, no vector
store -- claims/policyholders are exact key lookups, case hints are matched
by attribute filtering, and guideline questions are matched against a
closed set of topics whose answer text is always the pre-written template,
never LLM-synthesized prose. See memory `sop-agent-architecture-plan` for
the rationale."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

PII_FIELDS = ["full_name", "dob", "phone", "email", "ssn_last4"]

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def _load(name: str):
    with open(FIXTURES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


_CLAIMS = _load("claims.json")
_POLICYHOLDERS = _load("policyholders.json")
_REPRESENTATIVES = _load("representatives.json")
_GUIDELINE = _load("required_document_guideline.json")

TOPICS = [e["topic"] for e in _GUIDELINE.get("claim_followup_guidance", [])]
DOCUMENT_NAMES = list(_GUIDELINE.get("document_guidance", {}).keys())


# ---------------------------------------------------------------------------
# Identity verification
# ---------------------------------------------------------------------------

def _norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return unicodedata.normalize("NFKC", s).strip().lower()


def _norm_phone(s: Optional[str]) -> str:
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    return digits[-10:]


def _norm_digits(s: Optional[str], length: int = 4) -> str:
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    return digits[-length:] if len(digits) >= length else digits


def _match_fields(pii_slots: dict, record: dict) -> set[str]:
    matched: set[str] = set()

    if pii_slots.get("full_name"):
        names = [record.get("name", "")] + record.get("name_aliases", [])
        if _norm_text(pii_slots["full_name"]) in {_norm_text(n) for n in names}:
            matched.add("full_name")

    if pii_slots.get("dob") and _norm_text(pii_slots["dob"]) == _norm_text(record.get("dob")):
        matched.add("dob")

    if pii_slots.get("phone"):
        phones = [record.get("phone", "")] + record.get("phone_aliases", [])
        if _norm_phone(pii_slots["phone"]) and _norm_phone(pii_slots["phone"]) in {_norm_phone(p) for p in phones}:
            matched.add("phone")

    if pii_slots.get("email"):
        emails = [record.get("email", "")] + record.get("email_aliases", [])
        if _norm_text(pii_slots["email"]) in {_norm_text(e) for e in emails}:
            matched.add("email")

    if pii_slots.get("ssn_last4"):
        claimed = _norm_digits(pii_slots["ssn_last4"])
        if claimed and claimed == _norm_digits(record.get("id_last4")):
            matched.add("ssn_last4")

    return matched


def verify_identity(pii_slots: dict) -> tuple[Optional[str], set[str], bool]:
    """Returns (party_id, matched_fields, ambiguous).

    party_id is set only if a single record uniquely matches >=3 of the 5
    official PII fields. `matched_fields` is always the best (highest-count)
    match found so far, useful for telling the caller what's still needed.
    `ambiguous` is True if two or more different records tie for the best
    score at >=3 fields (caller should be asked for a differentiator, never
    told which records matched).
    """
    scored: list[tuple[str, set[str]]] = []
    for record in _POLICYHOLDERS:
        matched = _match_fields(pii_slots, record)
        if matched:
            scored.append((record["party_id"], matched))

    if not scored:
        return None, set(), False

    scored.sort(key=lambda t: len(t[1]), reverse=True)
    best_count = len(scored[0][1])
    top = [s for s in scored if len(s[1]) == best_count]

    if best_count >= 3:
        if len({party for party, _ in top}) == 1:
            return top[0][0], top[0][1], False
        return None, scored[0][1], True

    return None, scored[0][1], False


def get_policyholder(party_id: str) -> Optional[dict]:
    for p in _POLICYHOLDERS:
        if p["party_id"] == party_id:
            return p
    return None


def get_representative_context(name: str) -> Optional[dict]:
    """Looks up whether `name` is a known authorized representative calling
    on behalf of a policyholder (stretch: representative/consent flow)."""
    for rep in _REPRESENTATIVES:
        if _norm_text(rep.get("rep_name")) == _norm_text(name):
            return rep
    return None


# ---------------------------------------------------------------------------
# Claims lookup + hint-based case matching
# ---------------------------------------------------------------------------

def get_claims_for_party(party_id: str) -> list[dict]:
    return [c for c in _CLAIMS if c["party_id"] == party_id]


def get_claim(case_id: str) -> Optional[dict]:
    for c in _CLAIMS:
        if c["case_id"] == case_id:
            return c
    return None


def match_case_by_hints(party_id: str, case_type_hint: Optional[str] = None,
                         status_hint: Optional[str] = None,
                         month_hint: Optional[str] = None
                         ) -> tuple[Optional[str], list[dict]]:
    """Attribute-filters this party's claims by the structured hints
    extracted from freeform speech. Returns (case_id, candidates):
    case_id is set only if exactly one claim matches."""
    candidates = get_claims_for_party(party_id)

    def matches(c: dict) -> bool:
        if case_type_hint and c.get("case_type") != case_type_hint:
            return False
        if status_hint and c.get("status") != status_hint:
            return False
        if month_hint:
            month_num = _MONTHS.get(month_hint.lower())
            created = c.get("created_at", "")
            if month_num and len(created.split("-")) >= 2:
                try:
                    if int(created.split("-")[1]) != month_num:
                        return False
                except ValueError:
                    pass
        return True

    if not (case_type_hint or status_hint or month_hint):
        return (candidates[0]["case_id"], candidates) if len(candidates) == 1 else (None, candidates)

    filtered = [c for c in candidates if matches(c)]
    if len(filtered) == 1:
        return filtered[0]["case_id"], filtered
    return None, filtered


def case_fact_sheet(case: dict) -> str:
    """Plain-text dump of a case's known fields -- the single grounded
    source of truth for general/status/denial questions, so the LLM can't
    invent numbers or outcomes."""
    lines = [
        f"case_id: {case['case_id']}",
        f"case_type: {case.get('case_type')}",
        f"status: {case.get('status')}",
        f"created_at: {case.get('created_at')}",
        f"summary: {case.get('summary')}",
    ]
    if case.get("denial_reason"):
        lines.append(f"denial_reason: {case['denial_reason']}")
    if case.get("documents_needed"):
        lines.append(f"documents_needed: {', '.join(case['documents_needed'])}")
    if case.get("appeal_deadline"):
        lines.append(f"appeal_deadline: {case['appeal_deadline']}")
    for field in ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee"):
        if case.get(field) is not None:
            lines.append(f"{field}: {case[field]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guideline templates (closed-set topics, document guidance, alternatives)
#
# claims.json uses short document names ("office note", "pathology report")
# while required_document_guideline.json's guidance dicts use long
# descriptive keys ("treating provider office note", "original pathology
# report"). _same_document() bridges the two forms via case-insensitive
# substring containment so a mention in either form resolves consistently,
# instead of requiring an exact string match that would silently miss.
# ---------------------------------------------------------------------------

def _same_document(a: str, b: str) -> bool:
    a, b = _norm_text(a), _norm_text(b)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def is_document_required(case: dict, document_name: str) -> bool:
    return any(_same_document(document_name, d) for d in case.get("documents_needed", []))


def get_followup_answer(topic: str, case: dict) -> Optional[str]:
    for entry in _GUIDELINE.get("claim_followup_guidance", []):
        if entry["topic"] == topic:
            documents = ", ".join(case.get("documents_needed", [])) or "the requested items"
            return entry["en"].format(
                case_id=case.get("case_id", ""),
                documents=documents,
                average_processing_time_after_submission=(
                    _GUIDELINE["claim_followup_settings"]
                    ["average_processing_time_after_submission"]["en"]
                ),
            )
    return None


def get_followup_fallback() -> str:
    return _GUIDELINE["claim_followup_fallback"]["en"]


def get_human_review_note() -> str:
    return _GUIDELINE["claim_followup_settings"]["human_review_after_document_alternatives_exhausted"]["en"]


def _lookup_by_document(guidance_dict: dict, document_name: Optional[str]) -> Optional[dict]:
    if not document_name:
        return None
    if document_name in guidance_dict:
        return guidance_dict[document_name]
    for key, value in guidance_dict.items():
        if key != "default" and _same_document(document_name, key):
            return value
    return None


def get_document_guidance(document_name: Optional[str], case_type: Optional[str] = None) -> str:
    match = _lookup_by_document(_GUIDELINE.get("document_guidance", {}), document_name)
    if match:
        return match["en"]
    if case_type and case_type in _GUIDELINE.get("case_type_guidance", {}):
        return _GUIDELINE["case_type_guidance"][case_type]["en"]
    return _GUIDELINE["default_guidance"]["en"]


def get_document_alternative(document_name: Optional[str]) -> str:
    alt = _GUIDELINE.get("document_alternative_guidance", {})
    match = _lookup_by_document(alt, document_name)
    if match:
        return match["en"]
    return alt["default"]["en"]
