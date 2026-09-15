from app.llm_client import (
    LLMError,
    MockLLMClient,
    OllamaClient,
    OpenAICompatClient,
    build_llm_client,
)


def test_build_llm_client_routes_mock():
    assert isinstance(build_llm_client("mock", None, None, None), MockLLMClient)


def test_build_llm_client_routes_ollama_with_local_defaults():
    llm = build_llm_client("ollama", api_key=None, base_url=None, model=None)
    assert isinstance(llm, OllamaClient)
    assert llm.model == "llama3.1"
    assert str(llm._client.base_url) == "http://localhost:11434/v1/"
    # Ollama doesn't check the key, but the OpenAI SDK requires a non-empty
    # string -- OllamaClient must supply a dummy value rather than None.
    assert llm._client.api_key


def test_build_llm_client_routes_ollama_respects_overrides():
    llm = build_llm_client(
        "ollama", api_key=None, base_url="http://example.internal:11434/v1",
        model="mistral",
    )
    assert llm.model == "mistral"
    assert str(llm._client.base_url) == "http://example.internal:11434/v1/"


def test_build_llm_client_defaults_to_openai_compat():
    llm = build_llm_client("openai", api_key="sk-test", base_url=None, model=None)
    assert isinstance(llm, OpenAICompatClient)
    assert llm.model == "gpt-4o-mini"


def test_ollama_client_fails_gracefully_with_no_server_running():
    llm = build_llm_client("ollama", api_key=None, base_url=None, model=None)
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        assert False, "expected LLMError when no Ollama server is reachable"
    except LLMError:
        pass
