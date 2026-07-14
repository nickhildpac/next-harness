# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

Async FastAPI backend framed around **agentic task completion**: users submit goals, an agent
loop reasons and calls tools until it reaches a `finish` step. Chat/tone/memory still exist as a
secondary surface. Strict ports/adapters layering, wired top-to-bottom through FastAPI dependency
injection (`app/api/dependencies.py`):

```
routes/ → services/ → orchestration/ (LangGraph) → ports/ ← adapters/
                    ↘ repositories/ → db/ (SQLAlchemy async)
                    ↘ tools/ (registry + built-ins)
```

- **`app/ports/llm.py`** — the `LLMClient` Protocol (`chat`, `stream_chat`, `health`) plus frozen
  dataclasses (`ChatMessage`, `GenerationParams`, `LLMResult`, `ToolSpec`, `ToolCall`). This is the
  single seam between the app and any LLM. Everything above depends on the Protocol.
- **`app/adapters/`** — `OpenRouterClient`, `OllamaClient`, `OpenAIClient`, `AnthropicClient`,
  `GeminiClient` all implement `LLMClient`.
- **`app/tools/`** — provider-neutral tool layer. `registry.py` holds the `Tool`/`ToolRegistry`
  types; `builtins.py` registers the default tools (`now`, `list_notes`, `get_note`, `create_note`,
  `update_note`, `list_translations`, `translate_text`, `ingest_task_document`,
  `list_task_documents`, `search_task_documents`, `http_fetch`, `finish`); `protocol.py` renders the
  tool manifest into a system prompt and parses `<tool_call>{...}</tool_call>` JSON blocks out of
  assistant text — that fenced-JSON contract is how tools work with every adapter, no
  provider-native tool API required.
- **`app/orchestration/agent_graph.py`** — LangGraph loop with two nodes: `reason` (LLM turn) →
  `act` (dispatch tool calls) → `reason` again. Exits when the model emits `finish`, produces a
  reply with no tool calls, or hits `max_steps`.
- **`app/orchestration/chat_graph.py`** — the older single-node LLM wrapper, retained for the chat
  surface.
- **`app/services/tasks.py`** — `TaskService` creates an `AgentTask`, runs the loop against a
  (optionally scoped) tool registry, persists every reason/act step as an `AgentTaskStep`, and
  finalizes status (`completed`/`failed`).
- **`app/services/conversations.py`** — chat orchestrator (persist user message, resolve tone,
  build context, run `ChatGraph` or stream, persist reply, summarize, commit). Still handles the
  `assistant` and `duo` conversation kinds.
- **`app/services/memory.py`** — `MemoryService` builds chat context: system prompt + rolling
  summary + recent turn window trimmed to `context_token_budget`, with summarization when
  unsummarized tokens exceed `summary_trigger_tokens`.
- **`app/repositories/{conversations,notes,translations,tasks}.py`** — all DB access. Services never
  touch the session directly for queries; they go through the repository and own the `commit`.

### LLM provider selection

Provider resolution for chat/notes/translations happens per-request in `build_llm_client`. Precedence:
1. Per-request override: `X-LLM-Provider` header or `?llm_provider=` query param (`openrouter` | `ollama` | `auto` | `openai` | `anthropic` | `gemini`).
2. `LLM_PROVIDER` setting (default `openrouter`).
3. Automatic fallback: `openrouter`/`auto` **falls back to Ollama when `OPENROUTER_API_KEY` is unset**.

Agent tasks use `TASK_LLM_PROVIDER` (default `openai`) so task tool-following can be tuned
independently from chat. Set `TASK_OPENAI_MODEL` to override `OPENAI_MODEL` for task agents only.

Note: generation currently uses `settings.default_model` (Ollama's `llama3.1`) for `GenerationParams`;
the OpenRouter model is set inside its adapter. Ollama is expected at `localhost:11434`.

### Persistence

- SQLAlchemy 2.0 async. SQLite (`sqlite+aiosqlite:///./var/app.db`) for local dev; Compose overrides
  `DATABASE_URL` to Postgres (`asyncpg`). Tables are created via `Base.metadata.create_all` on
  startup — **there are no migrations**, so model changes need a fresh DB locally.
- Models: `Conversation` 1:N `Message`, 1:1 `ConversationSummary`; `AgentTask` 1:N `AgentTaskStep`;
  `Note`, `Translation`. `IdMixin`/`TimestampMixin` in `app/db/base.py`.

### Agent loop contract

- The system prompt for a task run = agent preamble + tool manifest from `render_tool_manifest`. The
  model is instructed to emit tool calls as `<tool_call>{"name":..., "arguments":{...}}</tool_call>`
  blocks in its content. `parse_tool_calls` extracts them and treats the remaining text as visible
  reasoning (persisted as a `thought` step).
- One loop iteration = one `reason` LLM turn plus zero-or-more tool calls executed by `act`. If the
  model returns *no* tool calls, that turn's text becomes the final answer (`final` step). If it
  calls the built-in `finish` tool, the run completes with `finish.summary` as `result_summary`.
- `max_steps` (default 8, capped at 32) counts reason turns; hitting it marks the task `failed`
  with a step-limit error. Tool failures are surfaced back to the model as JSON error observations
  so it can recover on the next turn.
- Tools receive a `ToolContext(session, http_client, user_id, metadata)`. Handlers that touch the DB
  use the same session as the request, so their writes commit atomically with the run's step log.
- `allowed_tools` on the task request narrows the registry for that run (the `finish` tool is
  always kept in scope). Omit it to expose everything the server registered.

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

Task/agent surface (primary):
- `POST /tasks` — create + run an agent task (`goal`, optional `user_id`, `max_steps`,
  `allowed_tools`, `run=false` to just persist). Response is the full `TaskDetail` including every
  step. `prompt`/`objective`/`task` are accepted as aliases for `goal`.
- `POST /tasks/{task_id}/documents` — attach `.pdf`, `.txt`, or `.md` uploads to a pending task as
  task-scoped RAG context.
- `POST /tasks/{task_id}/run` — run a pending task after optional document uploads.
- `GET /tasks?user_id=...`, `GET /tasks/{task_id}` — list and inspect runs.
- `GET /tools` — introspect the registered tool set (name, description, JSON schema).
- MCP stdio server: `python -m app.mcp` (Cursor: `.cursor/mcp.json`); tools mirror
  `build_default_registry()` via `Tool.spec()`, excluding `finish`. Identity via
  `MCP_USER_ID` / `MCP_TASK_ID` or per-call `user_id` / `task_id` args.

Chat/notes/translations surface (secondary): `POST /conversations`, `GET /conversations/{id}`,
`PATCH /conversations/{id}/tone`, `POST /conversations/{id}/messages?stream=true` (SSE),
`POST /conversations/{id}/suggest` (duo-only reply drafting), `GET /conversations/{id}/messages`,
`DELETE /conversations/{id}` (archive), `POST /notes`, `POST /translations`, `GET /tones`,
`GET /providers`, `GET /health`, `GET /health/llm`.

The static UI at `/` now lands on `/app/tasks.html` (goal + tool picker + step trace); the chat UI
is still available at `/app/`.
