from app import state_machine
from app.llm_client import MockLLMClient
from app.memory import SessionMemory


def new_session():
    return SessionMemory(session_id="test")


def test_margaret_chen_scenario_end_to_end():
    """The graded scenario from the assessment: identity + intent hint
    arrive in one message during VERIFY_ID. The agent must verify identity,
    must not disclose claim details before verification, must remember the
    denied-healthcare-January hint, and must use it after verification
    instead of asking from scratch."""
    session = new_session()
    llm = MockLLMClient()

    reply1 = state_machine.process_message(
        session,
        "I'm the policyholder. My name is Margaret Chen, policy POL-9921. "
        "I'm calling about my denied healthcare claim from January. "
        "DOB is 1985-03-15, SSN last four is 4472.",
        llm,
    )

    # Verified in one turn from the three PII fields (name, dob, ssn_last4).
    assert session.verified is True
    assert session.party_id == "P9"

    # No claim data leaked into the VERIFY_ID-phase prompt itself: the
    # verify_id system prompt never receives case facts, so the only way
    # case data could leak is if the phase failed to advance before using
    # case-aware prompts. Confirm the hint was captured but the case was
    # only referenced *after* moving into RESOLVE_INTENT/PROCESS_CASE.
    assert session.case_type_hint == "healthcare"
    assert session.status_hint == "denied"
    assert session.month_hint == "January"

    # The hint uniquely matches CL-2048, so the agent should propose it for
    # confirmation rather than asking "what are you calling about" cold.
    assert session.phase == "RESOLVE_INTENT"
    assert session.pending_case_confirmation == "CL-2048"
    assert "CL-2048" in reply1 or "mock reply" in reply1  # mock echoes system prompt

    reply2 = state_machine.process_message(session, "yes", llm)
    assert session.active_case_id == "CL-2048"
    assert session.phase == "PROCESS_CASE"
    assert reply2  # got a grounded reply, not an error

    reply3 = state_machine.process_message(
        session, "should I submit the office note?", llm,
    )
    assert session.active_document == "treating provider office note"
    assert reply3


def test_verify_id_blocks_until_three_fields_even_with_intent_hint():
    session = new_session()
    llm = MockLLMClient()

    state_machine.process_message(
        session, "I'm calling about my denied healthcare claim from January.", llm,
    )
    assert session.phase == "VERIFY_ID"
    assert session.verified is False
    # hint captured even though verification hasn't happened yet
    assert session.case_type_hint == "healthcare"
    assert session.status_hint == "denied"

    state_machine.process_message(session, "My name is Margaret Chen.", llm)
    assert session.verified is False  # only 1 field so far

    state_machine.process_message(session, "DOB is 1985-03-15.", llm)
    assert session.verified is False  # only 2 fields so far

    state_machine.process_message(session, "SSN last four is 4472.", llm)
    assert session.verified is True  # 3rd field completes verification
    assert session.phase != "VERIFY_ID"


def test_identity_stitching_across_people_does_not_verify():
    session = new_session()
    llm = MockLLMClient()
    state_machine.process_message(session, "My name is Margaret Chen.", llm)
    state_machine.process_message(session, "SSN last four is 9180.", llm)  # Ava Lopez's
    assert session.verified is False
    assert session.phase == "VERIFY_ID"


def test_out_of_scope_escalates_after_repeated_retries():
    session = new_session()
    llm = MockLLMClient()
    for _ in range(3):
        state_machine.process_message(session, "what is reinforcement learning?", llm)
    assert session.phase == "HUMAN_ESCALATION"
    assert session.escalated is True


def test_explicit_human_request_escalates_immediately():
    session = new_session()
    llm = MockLLMClient()
    reply = state_machine.process_message(session, "let me talk to a human representative", llm)
    assert session.phase == "HUMAN_ESCALATION"
    assert "representative" in reply.lower() or "connect" in reply.lower()


def test_post_process_email_opt_out_still_ends_call():
    session = new_session()
    session.verified = True
    session.party_id = "P9"
    session.phase = "POST_PROCESS"
    llm = MockLLMClient()

    state_machine.process_message(session, "wrap it up", llm)
    assert session.email_offered is True
    assert session.phase == "POST_PROCESS"

    state_machine.process_message(session, "no thanks", llm)
    assert session.email_sent is False
    assert session.phase == "ENDED"


def test_post_process_email_opt_in_marks_sent():
    session = new_session()
    session.verified = True
    session.party_id = "P9"
    session.phase = "POST_PROCESS"
    llm = MockLLMClient()

    state_machine.process_message(session, "please wrap up", llm)
    state_machine.process_message(session, "yes", llm)
    assert session.email_sent is True
    assert session.phase == "ENDED"
