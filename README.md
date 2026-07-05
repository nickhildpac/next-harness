# Cue Task & Agent Backend

Async FastAPI backend for **agentic task completion** with a local-first LLM. Users hand the app
a goal; an agent loop reasons and calls tools until the goal is met (or the step budget runs out).

Chat, notes, and translations still ship as secondary surfaces built on the same LLM port.

## Architecture

Built on **LangGraph** so the agent's reason → act → observe loop lives as explicit nodes rather
than a tangled service method.

Layers:

- API layer: FastAPI routers and dependency injection (`/tasks`, `/tools`, `/conversations`, ...).
- Service layer: `TaskService` runs an agent task end-to-end; `ConversationService` handles chat.
- Orchestration layer: `AgentGraph` (task loop) and `ChatGraph` (chat turn).
- Tool layer: `ToolRegistry` + built-in tools (`app/tools/builtins.py`) and a provider-neutral
  `<tool_call>` protocol (`app/tools/protocol.py`) that works across every LLM adapter.
- Ports/adapters: LLM client, repositories, token counting.

## Quickstart

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open `http://localhost:8000/` — you'll land on the task console. Enter a goal, keep or narrow the
tool set, and press **Run task** to watch the agent's reasoning, tool calls, and results.

## LLM provider

Providers wired through the same `LLMClient` port: OpenRouter, OpenAI, Anthropic, Gemini, Ollama.
`LLM_PROVIDER=openrouter` is the default; when `OPENROUTER_API_KEY` is unset, the app falls back to
local Ollama automatically. Per-request override: `X-LLM-Provider: openai` (or
`?llm_provider=openai`). Ollama is expected at `http://localhost:11434` with `llama3.1` pulled.

```bash
ollama pull llama3.1
```

Docker Compose starts the API, Postgres, and Ollama:

```bash
docker compose up --build
```

SQLite is the default for local development. Compose overrides `DATABASE_URL` to Postgres.

## API

Task/agent surface (primary):

- `POST /tasks` — create + run an agent task. Body: `{"goal": "...", "user_id": "...", "max_steps": 8, "allowed_tools": ["list_notes", "create_note"]}`. Also accepts `prompt`/`objective`/`task` as goal aliases. Set `run: false` to persist without running.
- `GET /tasks?user_id=...` — list past runs.
- `GET /tasks/{task_id}` — inspect a run (status, `result_summary`, per-step trace).
- `GET /tools` — introspect registered tools and their JSON parameter schemas.

Chat/notes/translations surface (secondary):

- `POST /conversations`, `GET /conversations/{id}`, `PATCH /conversations/{id}/tone`
- `POST /conversations/{id}/messages?stream=true` (SSE)
- `POST /conversations/{id}/suggest` (duo-only reply drafting)
- `GET /conversations/{id}/messages`, `DELETE /conversations/{id}`
- `POST /notes`, `POST /translations`
- `GET /tones`, `GET /providers`, `GET /health`, `GET /health/llm`

## Built-in tools

`now`, `list_notes`, `get_note`, `create_note`, `list_translations`, `http_fetch`, `finish`. The
`finish` tool is how the agent signals task completion — it always stays in scope even when
`allowed_tools` is narrowed. Add your own by dropping a `Tool` into `app/tools/builtins.py:all_tools`.
