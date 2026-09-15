from app import grounding


def test_verify_identity_needs_three_fields():
    party_id, matched, ambiguous = grounding.verify_identity({
        "full_name": "Margaret Chen",
        "dob": "1985-03-15",
    })
    assert party_id is None
    assert matched == {"full_name", "dob"}
    assert ambiguous is False


def test_verify_identity_succeeds_with_three_matching_fields():
    party_id, matched, ambiguous = grounding.verify_identity({
        "full_name": "Margaret Chen",
        "dob": "1985-03-15",
        "ssn_last4": "4472",
    })
    assert party_id == "P9"
    assert matched == {"full_name", "dob", "ssn_last4"}
    assert ambiguous is False


def test_verify_identity_rejects_mixed_identity_stitching():
    # name from one real person + ssn from another must not verify anyone
    party_id, matched, ambiguous = grounding.verify_identity({
        "full_name": "Margaret Chen",
        "ssn_last4": "9180",  # actually Ava Lopez's
        "phone": "+16503882920",  # actually Ava Lopez's
    })
    assert party_id is None


def test_verify_identity_supports_aliases():
    party_id, matched, ambiguous = grounding.verify_identity({
        "full_name": "Yaven Li",
        "email": "yawen.li@example.com",
        "phone": "+16505212830",
    })
    assert party_id == "P13"


def test_match_case_by_hints_finds_unique_claim():
    case_id, candidates = grounding.match_case_by_hints(
        "P9", case_type_hint="healthcare", status_hint="denied", month_hint="January",
    )
    assert case_id == "CL-2048"


def test_match_case_by_hints_ambiguous_without_enough_hints():
    case_id, candidates = grounding.match_case_by_hints("P9", case_type_hint="healthcare")
    assert case_id is None
    assert len(candidates) == 2  # CL-2048 and CL-2011


def test_is_document_required_bridges_short_and_long_names():
    case = grounding.get_claim("CL-2048")
    assert grounding.is_document_required(case, "treating provider office note") is True
    assert grounding.is_document_required(case, "office note") is True
    assert grounding.is_document_required(case, "repair estimate") is False


def test_get_document_guidance_bridges_short_and_long_names():
    text_short = grounding.get_document_guidance("office note")
    text_long = grounding.get_document_guidance("treating provider office note")
    assert text_short == text_long
    assert "patient name" in text_short


def test_get_document_guidance_falls_back_to_case_type_then_default():
    assert "diagnosis report" not in grounding.DOCUMENT_NAMES
    text = grounding.get_document_guidance("diagnosis report", case_type="healthcare")
    assert "treating provider or facility name" in text


def test_followup_answer_fills_template():
    case = grounding.get_claim("CL-2048")
    text = grounding.get_followup_answer("submission_timing", case)
    assert "CL-2048" in text
    assert "pathology report" in text
    assert "office note" in text


def test_followup_answer_unknown_topic_returns_none():
    case = grounding.get_claim("CL-2048")
    assert grounding.get_followup_answer("not_a_real_topic", case) is None
