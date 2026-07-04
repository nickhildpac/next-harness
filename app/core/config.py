from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ToneDefinition(BaseModel):
    system_template: str
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(ge=0.0, le=1.0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "conversational-ai-backend"
    environment: str = "dev"
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///./var/app.db"

    llm_provider: Literal[
        "openrouter", "ollama", "auto", "openai", "anthropic", "gemini"
    ] = "openrouter"
    ollama_base_url: AnyHttpUrl = "http://localhost:11434"
    openrouter_base_url: AnyHttpUrl = "https://openrouter.ai/api/v1"
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_max_tokens: int = 2048
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    default_model: str = "llama3.1"
    request_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    context_token_budget: int = 6000
    summary_trigger_tokens: int = 4500
    window_turn_count: int = 12
    custom_persona_max_chars: int = 800

    tones: dict[str, ToneDefinition] = {
        "professional": ToneDefinition(
            system_template=(
                "You are a professional assistant. Be clear, respectful, accurate, and action-oriented."
            ),
            temperature=0.35,
            top_p=0.9,
        ),
        "friendly": ToneDefinition(
            system_template="You are a friendly assistant. Be warm, helpful, and conversational.",
            temperature=0.7,
            top_p=0.95,
        ),
        "concise": ToneDefinition(
            system_template="You are a concise assistant. Prefer brief, direct answers.",
            temperature=0.25,
            top_p=0.85,
        ),
        "empathetic": ToneDefinition(
            system_template="You are an empathetic assistant. Be validating, careful, and practical.",
            temperature=0.6,
            top_p=0.92,
        ),
        "technical": ToneDefinition(
            system_template=(
                "You are a technical assistant. Be precise, structured, and explicit about assumptions."
            ),
            temperature=0.3,
            top_p=0.9,
        ),
        "humorous": ToneDefinition(
            system_template=(
                "You are a humorous assistant. Be useful first, with light humor when appropriate."
            ),
            temperature=0.8,
            top_p=0.95,
        ),
    }

    note_styles: dict[str, ToneDefinition] = {
        "default": ToneDefinition(
            system_template=(
                "You produce clean, well-structured markdown notes with headings, lists, and short paragraphs."
            ),
            temperature=0.4,
            top_p=0.9,
        ),
        "academic": ToneDefinition(
            system_template=(
                "You produce academic markdown notes: precise, formal, and taxonomic. "
                "Use nested lists for structure."
            ),
            temperature=0.25,
            top_p=0.85,
        ),
        "meeting": ToneDefinition(
            system_template=(
                "You produce meeting notes as markdown with 'Attendees', 'Agenda', 'Decisions', "
                "'Action Items' (owner + due), and 'Notes' sections. Bullet-first, terse."
            ),
            temperature=0.3,
            top_p=0.88,
        ),
        "blog": ToneDefinition(
            system_template=(
                "You produce engaging blog-style markdown: catchy H1, hook paragraph, "
                "scannable H2 sections, closing takeaway."
            ),
            temperature=0.7,
            top_p=0.95,
        ),
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
