# Architecture Decision Records

This file records the current task-agent decisions that are not obvious from the code alone.

## ADR-001: Provider-Neutral Task Tool Protocol

Status: Accepted

Date: 2026-07-07

### Context

The backend supports OpenRouter, OpenAI, Anthropic, Gemini, and Ollama through the shared
`LLMClient` port in `app/ports/llm.py`. These providers do not expose identical native tool-calling
APIs, and Ollama/local models are useful during development. Task runs also need a provider-agnostic
trace of reasoning, tool calls, tool observations, and final output.

### Decision

Task agents use the provider-neutral `<tool_call>{...}</tool_call>` text protocol as the primary
tool-calling contract. `app/tools/protocol.py` renders the tool manifest into the system prompt and
parses assistant output. `app/orchestration/agent_graph.py` drives the reason/act loop and dispatches
parsed calls through a `ToolInvoker` port.

In production, `TaskService` opens an MCP stdio client against `python -m app.mcp` and uses
`HybridToolInvoker`: non-`finish` tools run in the MCP subprocess (own DB session via
`McpRuntime`); `finish` stays local so the agent can complete without exposing control flow over
MCP. `ToolRegistry` remains the MCP server backend and the fast local invoker used in unit tests
(`use_mcp_tools=False`).

Native provider tool APIs can be added later as adapter-level optimizations, but they should preserve
the same invoker contract, task step trace, and error observation behavior.

### Consequences

- The same task loop works across hosted providers and local Ollama.
- Tool calls and observations are easy to persist in `AgentTaskStep` rows.
- The model can produce malformed text or no tool call. Current mitigations live in
  `AgentGraph`: explicit tool-use prompting, empty-response retries, false-refusal retry text, and
  structured tool error observations.
- Future native-tool support must not bypass task step persistence or the allowed-tools permission
  model.
- MCP process boundary means tool writes no longer share the request session/transaction with task
  step persistence.

## ADR-002: Task Documents Are First-Class Attachments

Status: Accepted

Date: 2026-07-07

### Context

Task creation and document-backed task creation previously used different request shapes. That made
callers choose between JSON task creation and a separate multipart creation flow for the same task
concept.

### Decision

Use one task lifecycle:

1. `POST /tasks` creates a task from JSON. `run: false` creates a pending task.
2. `POST /tasks/{task_id}/documents` attaches `.pdf`, `.txt`, or `.md` files to a pending task.
3. `POST /tasks/{task_id}/run` runs the pending task.

Uploaded documents are task-scoped RAG context. The agent can inspect them with
`list_task_documents` and `search_task_documents`.

### Consequences

- Clients have one task creation contract and add documents as attachments.
- Upload/ingestion failures happen before the task run starts.
- The frontend can still expose a single "Run task" action by creating the task, uploading staged
  files, and then running it.
- Task document behavior is explicit in `app/api/routes/tasks.py` and `app/services/tasks.py`.

## ADR-003: Task Agents Use Explicit LLM Provider Configuration

Status: Accepted

Date: 2026-07-07

### Context

Chat, notes, and translations can tolerate a broader provider policy. Agent tasks are more sensitive
to tool-following behavior because the model must emit valid tool calls and recover from tool
observations.

### Decision

General LLM traffic uses `LLM_PROVIDER` and request overrides through `build_llm_client`. Agent tasks
use `TASK_LLM_PROVIDER` instead, with `TASK_OPENAI_MODEL` available as a task-only OpenAI model
override.

### Consequences

- Task reliability can be tuned independently from chat.
- The default task policy is visible in config instead of being hidden in dependency wiring.
- Provider drift should be documented in `.env.example`, `README.md`, and `AGENTS.md` whenever these
  settings change.

## ADR-004: Tool Side Effects Follow Task Commit Semantics

Status: Accepted

Date: 2026-07-07

### Context

Agent tools can create notes, translations, and task documents. If those services commit internally,
a tool side effect can persist even when the task later fails or the task trace is not committed.

### Decision

Direct API service calls keep their default commit behavior. Task tool handlers call side-effecting
services with commit deferral where supported, so database writes commit with the final task trace.
First-class document attachments are different: `POST /tasks/{task_id}/documents` is a user action
before the run, so ingestion commits before the agent starts.

### Consequences

- Agent-created notes/translations and task trace rows share the same database transaction.
- Task attachment ingestion is durable before execution, which gives the agent stable RAG context.
- Vector-store writes are still external to the SQL transaction. Retrieval should continue to ignore
  orphaned vector hits that no longer have matching SQL document rows.

## Current Known Tradeoffs

- Task execution still runs inline in the request/response path. Long-running tasks would be more
  robust behind a worker with persisted incremental step updates and cancellation.
- `Tool.parameters` schemas are advertised to the model, but argument validation is still enforced
  mostly inside handlers. Central JSON Schema validation in `ToolRegistry.invoke` would make tool
  failures more consistent.
- `http_fetch` remains broad for an autonomous agent. Before exposing it widely, add domain
  allowlists, private-network blocking after DNS resolution, redirect validation, and tighter
  response limits.
- Database tables are created with `Base.metadata.create_all` on startup. Production use needs
  migrations.
