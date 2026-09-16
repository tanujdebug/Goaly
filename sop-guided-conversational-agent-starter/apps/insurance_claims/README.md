# SOP-Guided Insurance Claims Agent

An insurance claims support agent that follows a fixed 4-phase SOP
(`VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS`) while still
conversing naturally. Phase order and safety gates (identity verification,
grounded-only claim answers) are enforced deterministically in code; the LLM
handles language understanding and phrasing within whatever a given phase
allows.

## How it works

- **`app/state_machine.py`** — the only place that decides phase transitions
  and what's allowed to be disclosed. Deterministic, not LLM-controlled.
- **`app/extraction.py`** — an always-on extractor that runs on every turn,
  regardless of phase, pulling PII candidates, intent/case hints, emotion,
  and scope out of the caller's message into session memory. This is how a
  hint mentioned during `VERIFY_ID` (e.g. "my denied healthcare claim from
  January") is remembered and used once verification completes, without
  ever unlocking anything early.
- **`app/grounding.py`** — structured retrieval over `fixtures/*.json`.
  Claims/policyholders are exact key lookups; case hints are matched by
  attribute filtering; document/topic questions are matched against a
  closed set of topics whose answer text is always the pre-written
  template, never LLM-generated prose. No vector database or embeddings —
  the fixture data is small and structured enough that this is more
  reliable and auditable than similarity search.
- **`app/prompts.py`** — phase-scoped system prompts carrying only the facts
  the state machine has decided are groundable at that point.
- **`app/llm_client.py`** — a thin OpenAI-compatible chat client. A
  `MockLLMClient` (rule-based, no network) is also included for offline
  testing/demoing without an API key.

## Setup

Requires Python 3.11+.

```bash
cd apps/insurance_claims
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the server:

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

Then open `http://localhost:8000` — a chat UI with a session state panel is
served there. Paste your API key into the "API key" field in the sidebar
(it's sent only with each chat request from your browser, never stored
server-side); it defaults to an OpenAI-compatible chat-completions endpoint,
model `gpt-4o-mini`. You can point `Base URL` at any OpenAI-compatible
endpoint and change the model — the "Groq" provider preset in the dropdown
fills in Groq's endpoint and a free-tier model (`openai/gpt-oss-20b`) for a
quick no-cost test; it was also the model most used to stress-test the
`VERIFY_ID` gate during development, since smaller/faster models are more
likely to ignore prompt instructions than GPT-4o-mini, which is exactly
what the deterministic backstops in `state_machine.py` are there to catch.

You can instead supply the key via environment variables and skip the UI
field:

```bash
MODEL_API_KEY=sk-... .venv/bin/uvicorn app.main:app --port 8000
# or: OPENAI_API_KEY=sk-...
# optional: MODEL_BASE_URL=..., MODEL_NAME=gpt-4o-mini
```

### Local mode (Ollama, no API key/cost)

Point the agent at a local [Ollama](https://ollama.com) install instead of a
paid API. Ollama exposes an OpenAI-compatible endpoint out of the box, so
`llm_client.py` talks to it the same way it talks to OpenAI.

```bash
ollama pull llama3.1     # or any other chat-capable model you have pulled
ollama serve             # usually already running as a background service
```

Then either select "Ollama (local, no key needed)" as the provider in the
UI (base URL and model default to `http://localhost:11434/v1` and
`llama3.1` if left blank), or run the server with:

```bash
MODEL_PROVIDER=ollama .venv/bin/uvicorn app.main:app --port 8000
# optional: MODEL_NAME=llama3.1, MODEL_BASE_URL=http://localhost:11434/v1
# (no API key needed)
```

Smaller/older local models are noticeably less reliable than GPT-4o-mini at
following the JSON-extraction contract in `app/extraction.py` and the
phase-scoped system prompts in `app/prompts.py` — if replies look off,
try a larger instruction-tuned model.

Observed concretely: `llama3.1:8b` (Q4_K_M) correctly extracts PII fields
but sometimes hallucinates `wants_human: true` on the canonical
verification message, causing an immediate (harmless, but incorrect)
escalation to a human representative instead of proceeding through
verification — it never discloses anything or skips a gate, it just
escalates a turn too early. Cloud models (GPT-4o-mini, Gemini) tested
clean on this exact message. If you hit this, either try a stronger
instruction-following local model, or use a cloud provider for reliable
testing.

### Offline / no-key mode

Select "Mock (offline, no key needed)" as the provider in the UI, or set
`MODEL_PROVIDER=mock`. This runs the same state machine against a
rule-based stand-in for the LLM (regex-based extraction, template-only
replies) so the full phase-transition logic can be exercised without an API
key. It is for quick sanity checks only — replies won't read naturally;
use a real provider to see the agent actually converse.

## Docker

```bash
cd apps/insurance_claims
docker build -t sop-claims-agent .
docker run -p 8000:8000 -e MODEL_API_KEY=sk-... sop-claims-agent
```

Then open `http://localhost:8000`.

## Tests

```bash
cd apps/insurance_claims
.venv/bin/pip install -r requirements.txt   # includes pytest
.venv/bin/pytest tests/ -v
```

Covers: identity verification (including alias matching and rejecting
mixed-identity "stitching" across different people), case-hint matching,
document-name normalization between the two fixture files' naming
conventions, guideline template filling, and a full scripted run of the
assessment's graded scenario (Margaret Chen) plus off-topic escalation and
post-process email opt-in/opt-out.

## Try it: the graded scenario

In the chat UI (or via the "Margaret Chen" sample button):

> I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm
> calling about my denied healthcare claim from January. DOB is
> 1985-03-15, SSN last four is 4472.

Expected: identity verifies immediately (name + DOB + SSN last 4 = 3 of the
5 official fields), no claim detail is disclosed before that point, the
denied-healthcare-January hint is remembered, and once verified the agent
proposes the matching claim (`CL-2048`) for confirmation instead of asking
"what are you calling about" from scratch.

Other things to try:
- **Partial/refused verification**: give only a name, or refuse a field —
  the agent should ask for an acceptable alternative, not disclose anything.
- **Out-of-scope**: ask "what is RL?" a few times in a row — after repeated
  off-topic questions, the agent offers a human representative.
- **Frustration (bonus)**: "I already told you who I am. This is
  ridiculous. Just tell me why my claim was denied." — the agent should
  acknowledge the frustration, explain why verification is required, and
  keep offering the allowed verification paths rather than disclosing
  anything or dropping the gate.
- **Post-process**: once a claim question is answered, say "that's all" —
  the agent offers an email summary and respects a yes/no answer either way.

## Known limitations

- Session state is in-memory per process (fine for this demo; not
  persisted across restarts, not safe with multiple uvicorn workers).
- The "send email" step in `POST_PROCESS` logs/marks the summary as sent
  rather than dispatching a real email — no SMTP credentials are part of
  this assessment's scope.
- The `VERIFY_ID`-phase hallucination guard (`state_machine._breaches_verify_gate`)
  is a pattern-based backstop, not a semantic check — it catches the
  disclosure/false-confirmation patterns observed in testing (claim IDs,
  "I've pulled up...", fabricated emails/phone numbers, "you're verified"),
  but a sufficiently different phrasing could in principle slip past it.
  The deterministic gate underneath (`grounding.verify_identity`) is the
  real source of truth and cannot be talked around regardless.

## Representative / consent flow

A caller who isn't the policyholder (e.g. "I'm calling on behalf of my
mother Margaret Chen") is matched against `fixtures/representatives.json`
by their own name or the policyholder's name, then gated on the
policyholder's consent rather than a PII match — simulated as an async
status check (`fixtures/consent_scenarios.json`) that's polled once per
turn regardless of what the caller says. This re-gates on every turn, not
just during initial verification: if a caller announces mid-conversation
(even in an already-verified session) that they're actually calling on
someone else's behalf, any prior verification for that session is reset
and consent is required before anything further is disclosed. Try it with
the "Representative + consent" sample button.
