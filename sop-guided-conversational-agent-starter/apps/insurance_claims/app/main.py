from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import state_machine
from .llm_client import LLMError, build_llm_client
from .memory import SessionStore

app = FastAPI(title="SOP-Guided Insurance Claims Agent")

STORE = SessionStore()
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    phase: str
    debug: dict


class ResetRequest(BaseModel):
    session_id: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = STORE.get_or_create(req.session_id)

    provider = req.provider or os.environ.get("MODEL_PROVIDER", "openai")
    api_key = req.api_key or os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = req.base_url or os.environ.get("MODEL_BASE_URL")
    default_model = "llama3.1" if provider.lower() == "ollama" else "gpt-4o-mini"
    model = req.model or os.environ.get("MODEL_NAME", default_model)

    try:
        llm = build_llm_client(provider=provider, api_key=api_key, base_url=base_url, model=model)
    except LLMError as exc:
        return ChatResponse(
            session_id=session.session_id,
            reply=f"Setup error: {exc}",
            phase=session.phase,
            debug=state_machine.debug_snapshot(session),
        )

    try:
        reply = state_machine.process_message(session, req.message, llm)
    except LLMError as exc:
        reply = f"Sorry, I hit an error talking to the model: {exc}"

    return ChatResponse(
        session_id=session.session_id,
        reply=reply,
        phase=session.phase,
        debug=state_machine.debug_snapshot(session),
    )


@app.post("/api/reset")
def reset(req: ResetRequest):
    STORE.reset(req.session_id)
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"ok": True}
