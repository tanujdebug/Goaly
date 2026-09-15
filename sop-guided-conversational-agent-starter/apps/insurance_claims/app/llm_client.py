"""Provider-agnostic LLM client. Default implementation talks to any
OpenAI-compatible chat-completions endpoint using an API key supplied at
request time (or via env vars as a fallback). A MockLLMClient is provided
for offline testing / demoing the state machine without spending real API
credits -- see README for how to enable it (MODEL_PROVIDER=mock). Ollama's
built-in OpenAI-compatible endpoint (`/v1`) is supported the same way via
MODEL_PROVIDER=ollama -- see README for local setup."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


class LLMError(RuntimeError):
    pass


class OpenAICompatClient:
    def __init__(self, api_key: Optional[str], base_url: Optional[str] = None,
                 model: str = "gpt-4o-mini"):
        if not api_key:
            raise LLMError(
                "No API key provided. Pass one in the UI, or set "
                "MODEL_API_KEY / OPENAI_API_KEY in the environment."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'openai' package is required: pip install openai") from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model

    # Some OpenAI-compatible servers (older Ollama builds in particular)
    # reject the `response_format` param outright instead of ignoring it.
    # The extractor prompt already asks for JSON explicitly, so on that
    # specific failure we retry once without it rather than hard-failing.
    _supports_response_format = True

    def chat(self, messages: list[dict], temperature: float = 0.4,
              json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {}
        if json_mode and self._supports_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - surface as LLMError uniformly
            if json_mode and kwargs.get("response_format"):
                self._supports_response_format = False
                return self.chat(messages, temperature=temperature, json_mode=json_mode)
            raise LLMError(f"LLM call failed: {exc}") from exc
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict], temperature: float = 0.2) -> dict:
        raw = self.chat(messages, temperature=temperature, json_mode=True)
        return _parse_json(raw)


class OllamaClient(OpenAICompatClient):
    """Talks to a local Ollama server via its built-in OpenAI-compatible
    endpoint. No real API key is needed -- Ollama ignores it -- and the
    base URL/model default to a stock local install."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: str = "llama3.1"):
        super().__init__(
            api_key=api_key or "ollama",
            base_url=base_url or "http://localhost:11434/v1",
            model=model,
        )


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"Could not parse JSON from model output: {raw[:300]!r}")


# ---------------------------------------------------------------------------
# Mock client: rule-based, no network calls. Used for automated tests and
# for MODEL_PROVIDER=mock so the flow can be exercised without an API key.
# It is intentionally simple -- it is not a substitute for the real model,
# only a deterministic stand-in for the extractor JSON contract and for
# producing a readable (if plain) reply from the grounded system prompt.
# ---------------------------------------------------------------------------

_EMOTION_WORDS = {
    "frustrated": ["frustrat", "ridiculous", "come on", "annoyed", "sick of"],
    "angry": ["angry", "furious", "pissed", "unacceptable"],
    "anxious": ["worried", "anxious", "scared", "nervous"],
    "confused": ["confused", "don't understand", "not sure what"],
}
_MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
_CASE_TYPES = ["healthcare", "dental", "auto"]
_STATUSES = ["denied", "closed", "open"]
_IN_SCOPE_HINTS = [
    "claim", "policy", "denied", "denial", "reimburs", "document", "submit",
    "status", "appeal", "email", "verify", "identity", "human", "represent",
    "dob", "ssn", "ss n", "phone", "email", "name",
]


class MockLLMClient:
    model = "mock"

    def chat(self, messages: list[dict], temperature: float = 0.4,
              json_mode: bool = False) -> str:
        if json_mode:
            return json.dumps(self._extract(messages))
        return self._reply(messages)

    def chat_json(self, messages: list[dict], temperature: float = 0.2) -> dict:
        return self._extract(messages)

    @staticmethod
    def _last_user_message(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def _extract(self, messages: list[dict]) -> dict:
        text = self._last_user_message(messages)
        low = text.lower()

        out: dict[str, Any] = {
            "pii": {},
            "case_type_hint": None,
            "status_hint": None,
            "month_hint": None,
            "free_text_hint": None,
            "rewritten_question": text if "?" in text else None,
            "topic_guess": None,
            "document_mentioned": None,
            "emotion": None,
            "in_scope": True,
            "wants_human": "human" in low or "representative" in low,
            "wants_to_end": any(p in low for p in ["that's all", "nothing else", "bye", "goodbye"]),
            "affirmation": None,
        }

        name_match = re.search(r"name is ([A-Za-z][A-Za-z' -]*?)(?=[,.\d]|$)", text)
        if name_match:
            out["pii"]["full_name"] = name_match.group(1).strip()

        dob_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if dob_match:
            out["pii"]["dob"] = dob_match.group(1)

        email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        if email_match:
            out["pii"]["email"] = email_match.group(0)

        ssn_match = re.search(r"(?:ssn|last four|last 4)[^\d]*(\d{4})\b", low)
        if ssn_match:
            out["pii"]["ssn_last4"] = ssn_match.group(1)

        phone_match = re.search(r"(\+?\d[\d\-\s]{8,}\d)", text)
        if phone_match:
            digits = re.sub(r"\D", "", phone_match.group(0))
            # require a real phone-length digit run so this doesn't swallow
            # an 8-digit DOB (YYYY-MM-DD minus separators) as a phone number
            if len(digits) >= 10:
                out["pii"]["phone"] = digits

        for ct in _CASE_TYPES:
            if ct in low:
                out["case_type_hint"] = ct
                break
        for st in _STATUSES:
            if st in low:
                out["status_hint"] = st
                break
        for m in _MONTHS:
            if m in low:
                out["month_hint"] = m.capitalize()
                break

        if any(w in low for w in ["how do i submit", "where do i submit", "portal", "upload"]):
            out["topic_guess"] = "submission_method"
        elif "should i" in low or "do i need" in low or "is it required" in low:
            out["topic_guess"] = "necessity"
        elif any(w in low for w in ["how soon", "when do i need", "when should i submit"]):
            out["topic_guess"] = "submission_timing"
        elif any(w in low for w in ["how long", "processing time", "review time"]):
            out["topic_guess"] = "processing_time_after_submission"
        elif any(w in low for w in ["format", "file type", "pdf", "scan", "photo"]):
            out["topic_guess"] = "file_format_requirements"
        elif any(w in low for w in ["confirm receipt", "did you get", "show up in status"]):
            out["topic_guess"] = "receipt_confirmation"
        elif any(w in low for w in ["why was", "denied", "denial reason", "status"]):
            out["topic_guess"] = "status_or_denial"

        if "office note" in low:
            out["document_mentioned"] = "treating provider office note"
        elif "pathology" in low:
            out["document_mentioned"] = "original pathology report"
        elif "estimate" in low:
            out["document_mentioned"] = "repair estimate"
        elif "photo" in low:
            out["document_mentioned"] = "supplemental accident scene photos"

        for emotion, words in _EMOTION_WORDS.items():
            if any(w in low for w in words):
                out["emotion"] = emotion
                break

        if low.strip() in ("yes", "yes please", "yep", "sure", "correct", "that's right"):
            out["affirmation"] = True
        elif low.strip() in ("no", "nope", "no thanks", "not now", "skip"):
            out["affirmation"] = False

        out["in_scope"] = (
            any(h in low for h in _IN_SCOPE_HINTS)
            or len(low.split()) < 4
            or out["wants_to_end"]
            or out["affirmation"] is not None
        )

        return out

    def _reply(self, messages: list[dict]) -> str:
        system = messages[0]["content"] if messages else ""
        # The mock just surfaces the grounded/system instructions plainly so
        # automated tests can assert on phase behavior; it is not meant to
        # read naturally. Real phrasing comes from the actual LLM provider.
        return f"[mock reply based on: {system[:400].strip()}]"


def build_llm_client(provider: Optional[str], api_key: Optional[str],
                      base_url: Optional[str], model: Optional[str]):
    provider = (provider or "openai").lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "ollama":
        return OllamaClient(api_key=api_key, base_url=base_url,
                             model=model or "llama3.1")
    return OpenAICompatClient(api_key=api_key, base_url=base_url,
                               model=model or "gpt-4o-mini")
