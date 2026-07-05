from fastapi import HTTPException


def llm_error_detail(llm, exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)

    provider = _provider_name(llm)
    provider_detail = _provider_detail(provider, exc)
    if provider_detail:
        return provider_detail

    message = _clean_message(str(exc))

    if provider == "Ollama":
        return "Local LLM is unavailable or timed out."
    if provider:
        return f"{provider} request failed: {message or 'provider is unavailable or timed out.'}"
    return f"LLM request failed: {message or 'provider is unavailable or timed out.'}"


def _provider_name(llm) -> str | None:
    name = llm.__class__.__name__.removesuffix("Client")
    if name in {"OpenAI", "OpenRouter", "Anthropic", "Gemini", "Ollama"}:
        return name
    return None


def _clean_message(message: str) -> str:
    cleaned = " ".join(message.split())
    if "Incorrect API key provided" in cleaned or "invalid_api_key" in cleaned:
        return "invalid API key"
    return cleaned[:500]


def _provider_detail(provider: str | None, exc: Exception) -> str | None:
    if provider == "OpenAI":
        try:
            from openai import AuthenticationError
        except Exception:
            AuthenticationError = ()  # type: ignore[assignment]
        if isinstance(exc, AuthenticationError):
            return "OpenAI request failed: invalid API key"
    return None
