# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (Python >= 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the API (SQLite by default, tables auto-created on startup)
uvicorn app.main:app --reload         # UI served at /app/, redirect from /

# Tests (pytest-asyncio in auto mode; no config needed for async tests)
pytest                                 # full suite
pytest tests/test_api.py              # single file
pytest tests/test_api.py::test_compatibility_payload_aliases   # single test

# Lint (line-length 100)
ruff check .
ruff format .

# Full stack: API + Postgres + Ollama
docker compose up --build
```

## Architecture

Async FastAPI backend for durable, multi-conversation chat. Strict ports/adapters layering, wired
top-to-bottom through FastAPI dependency injection (`app/api/dependencies.py`):

```
routes/ → services/ → orchestration/ (LangGraph) → ports/ ← adapters/
                    ↘ repositories/ → db/ (SQLAlchemy async)
```

- **`app/ports/llm.py`** — the `LLMClient` Protocol (`chat`, `stream_chat`, `health`) plus frozen
  dataclasses (`ChatMessage`, `GenerationParams`, `LLMResult`). This is the single seam between the
  app and any LLM. Everything above depends on the Protocol, never a concrete client.
- **`app/adapters/`** — `OpenRouterClient` and `OllamaClient` both implement `LLMClient`.
- **`app/orchestration/chat_graph.py`** — LangGraph wrapper (`ChatGraph`) around the LLM call.
  Currently a single `generate` node; it exists so multi-step stateful flows can be added as nodes
  rather than growing inside a service method.
- **`app/services/conversations.py`** — `ConversationService` is the orchestrator: persists the user
  message, resolves tone, builds context, runs the graph (or streams), persists the assistant reply,
  triggers summarization, then commits. Holds both the sync (`send_message`) and SSE streaming
  (`stream_message`) paths.
- **`app/services/memory.py`** — `MemoryService` assembles LLM context: system prompt + rolling
  summary + recent turn window, trimmed to `context_token_budget`. Rolls older turns into an
  LLM-generated summary once unsummarized tokens exceed `summary_trigger_tokens`.
- **`app/repositories/conversations.py`** — all DB access. Services never touch the session directly
  for queries; they go through the repository and own the `commit`.

### LLM provider selection

Provider resolution happens per-request in `build_llm_client`. Precedence:
1. Per-request override: `X-LLM-Provider` header or `?llm_provider=` query param (`openrouter` | `ollama` | `auto`).
2. `LLM_PROVIDER` setting (default `openrouter`).
3. Automatic fallback: `openrouter`/`auto` **falls back to Ollama when `OPENROUTER_API_KEY` is unset**.

Note: generation currently uses `settings.default_model` (Ollama's `llama3.1`) for `GenerationParams`;
the OpenRouter model is set inside its adapter. Ollama is expected at `localhost:11434`.

### Persistence

- SQLAlchemy 2.0 async. SQLite (`sqlite+aiosqlite:///./var/app.db`) for local dev; Compose overrides
  `DATABASE_URL` to Postgres (`asyncpg`). Tables are created via `Base.metadata.create_all` on
  startup — **there are no migrations**, so model changes need a fresh DB locally.
- Models: `Conversation` 1:N `Message`, 1:1 `ConversationSummary`. `IdMixin`/`TimestampMixin` in
  `app/db/base.py`.

### Conversation kinds

`Conversation.kind` is `assistant` (default; one user + LLM replies) or `duo` (two human
participants: `user_id` + `second_user_id`, created by passing `participants: [a, b]` to
`POST /conversations`). In duo conversations `POST .../messages` only persists the message
(`assistant_message` is null; the sender must be a participant) — the LLM only speaks through
`POST /conversations/{id}/suggest` (`for_user`, aliases `as_user`/`user_id`; optional
`tone_override` and `persist`). Suggestion context maps the target user's past messages to the
assistant role and the other participant's to the user role
(`MemoryService.duo_context_messages`); persisted suggestions are `role=user` rows with `model`
set, marking them AI-drafted.

### Tones & request compatibility

- Six built-in tones (each a system template + temperature/top_p) live in `Settings.tones`
  (`app/core/config.py`); plus a `custom` tone that wraps a user persona. `ToneService.resolve`
  applies a per-message `tone_override` over the conversation default.
- The API accepts lenient/aliased payloads for compatibility (`app/schemas/conversation.py`):
  `tone` as a bare string, tone aliases (`formal`→`professional`, `direct`→`concise`,
  `playful`→`humorous`), `text` as an alias for message `content`, and `user_id` defaulting to
  `"anonymous"` (rebound to the conversation owner in the service). Preserve these aliases when
  editing schemas.

## Config

Settings are loaded from `.env` via pydantic-settings (`Settings`, cached by `get_settings`). Key
knobs: `context_token_budget` (6000), `summary_trigger_tokens` (4500), `window_turn_count` (12),
`custom_persona_max_chars` (800). Copy `.env.example` to `.env` to start.

## API surface

`POST /conversations`, `GET /conversations/{id}`, `PATCH /conversations/{id}/tone`,
`POST /conversations/{id}/messages?stream=true` (SSE), `POST /conversations/{id}/suggest`
(duo only: LLM drafts a reply for a participant), `GET /conversations/{id}/messages`,
`DELETE /conversations/{id}` (archive), `GET /tones`, `GET /health`, `GET /health/llm`.
