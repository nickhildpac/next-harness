# Conversational AI Backend

Async FastAPI backend for durable, multi-conversation chat with a local-first LLM provider.

## Architecture

This scaffold uses **LangGraph** rather than classic LangChain memory. The current graph is intentionally compact, but LangGraph is a better fit for stateful multi-turn flows because conversation state, prompt assembly, generation, and persistence can grow into explicit nodes without burying control flow inside one service method.

Layering:

- API layer: FastAPI routers and dependency injection.
- Service layer: conversation lifecycle, tone updates, message orchestration.
- Orchestration layer: LangGraph chat workflow.
- Ports/adapters: LLM client, memory store, repositories, token counting.

## Quickstart

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

The backend can use either OpenRouter or Ollama through the same internal LLM port.
`LLM_PROVIDER=openrouter` is the default; when `OPENROUTER_API_KEY` is not set, the
app falls back to local Ollama automatically.

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Per request, use `X-LLM-Provider: openrouter` or `X-LLM-Provider: ollama`; the
same value is also accepted as `?llm_provider=openrouter`.

Ollama is expected at `http://localhost:11434` with `llama3.1` available:

```bash
ollama pull llama3.1
```

Docker Compose starts the API, Postgres, and Ollama:

```bash
docker compose up --build
```

SQLite is the default for local development. Compose overrides `DATABASE_URL` to Postgres.

## API

- `POST /conversations`
- `GET /conversations/{id}`
- `PATCH /conversations/{id}/tone`
- `POST /conversations/{id}/messages?stream=true`
- `GET /conversations/{id}/messages`
- `DELETE /conversations/{id}`
- `GET /health`
- `GET /health/llm`

Streaming uses Server-Sent Events.
