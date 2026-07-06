# Backend Agent Task Review

## Current Shape

The backend exposes a task-running agent via `POST /tasks`. `TaskService` creates an `AgentTask`, builds a scoped `ToolRegistry`, runs `AgentGraph`, persists step traces, and then returns `TaskDetail`.

The core loop is simple:

1. LLM reason turn
2. Parse `<tool_call>{...}</tool_call>` blocks
3. Invoke tools
4. Feed tool observations back
5. Stop on finish, no tool call, error, or step limit.

## What’s Good

- **Clean Service Boundary**: `app/api/routes/tasks.py` only handles auth/user scoping, while `TaskService` (line 22) owns task creation/run/persistence.
- **Compact and Testable Agent Loop**: `AgentGraph` (line 57) has clear reason and act nodes.
- **Tool Scoping Supported**: `allowed_tools` supports tool scoping, and `finish` is always available (tasks.py line 104).
- **Useful Step Tracing**: The system records `thought`, `tool_call`, `tool_result`, `final`, and `error` steps through `TaskStepResponse` (line 34).



## Main Issues



### Task execution is synchronous and fragile for long agent runs

`POST /tasks` runs the full agent inline before returning (tasks.py line 47). If the server dies mid-run, the task may remain running, and intermediate steps are only persisted after `graph.run` completes.

**Suggested fix**: Move execution to a background worker/job queue. `POST /tasks` should create a pending task and return immediately; a worker should append steps as they happen. Add `POST /tasks/{id}/run`, `POST /tasks/{id}/cancel`, and streaming/polling for step updates.

### Tool writes are not transactionally consistent

Some tools write via the shared task session and commit at the end, but others commit internally. `translate_text` calls `TranslationService.translate`, which commits inside the service (translations.py line 87). RAG ingestion also commits inside `rag.py` (line 130). This means a tool can persist side effects even if the task later fails or the trace is never persisted.

**Suggested fix**: Define a consistent contract: either all tools are transactional under the task run, or side-effecting tools are explicitly “committed actions” with compensating metadata. For agents, explicit side-effect logs/outbox records are preferred.

### `allowed_tools` silently accepts invalid tool names

`_scoped_registry` always adds `finish`, so `allowed_tools=["bad_tool"]` still creates a registry containing `finish` and does not hit the “did not match” error (tasks.py line 107).

**Suggested fix**: Validate requested tools before adding `finish`. Return 400 with unknown tool names.

### Tool schemas are advertised but not centrally enforced

`Tool.parameters` is included in the manifest, but `ToolRegistry.invoke` passes arguments straight to handlers (registry.py line 88). Validation is scattered in each handler.

**Suggested fix**: Validate arguments against each tool’s JSON Schema before invocation. Return structured `ToolError` for schema failures.

### `http_fetch` is too powerful for an autonomous agent

The tool allows arbitrary HTTP(S) fetches (builtins.py line 293). That creates SSRF/internal network risk if exposed broadly.

**Suggested fix**: Do not include `http_fetch` by default. Require allowlisted domains, block localhost/private IP ranges after DNS resolution, cap response streaming before reading full text, and disable redirects or validate redirect targets.

### Custom tool-call parsing is pragmatic but brittle

`parse_tool_calls` uses regex against assistant text (protocol.py line 23). It silently ignores malformed JSON (protocol.py line 64).

**Suggested fix**: Keep this provider-neutral protocol for now, but add a repair/retry path when malformed calls are detected. For providers that support native tools, consider optional adapter-native tool calling later.

## How To Utilize Agents

Use agents for bounded, multi-step workflows where the model needs to decide which backend capability to call next. Do not use them for simple CRUD, direct chat, or deterministic API calls.

### Good current use cases:

- **Personal knowledge/RAG task**
  - **Use tools**: `ingest_task_document`, `search_task_documents`, `create_note`, `finish`.
  - **Example goal**: “Read this pasted policy text, extract key obligations, and save a checklist note.”
- **Note organization**
  - **Use tools**: `list_notes`, `get_note`, `create_note`, `update_note`, `finish`.
  - **Example goal**: “Find my rough notes about LangGraph and consolidate them into a clean technical summary.”
- **Translation workflow**
  - **Use tools**: `translate_text`, `list_translations`, `finish`.
  - **Example goal**: “Translate this text into Spanish and save the result.”
- **Inspection/research workflow**
  - **Use tools**: `http_fetch` (only if domain-restricted).
  - **Example goal**: “Fetch this public API response and summarize the fields.”



### Recommended request pattern:

For general note summarization:

```json
{
  "goal": "Summarize my notes about RAG and save a concise implementation checklist.",
  "max_steps": 8,
  "allowed_tools": ["list_notes", "get_note", "create_note"]
}
```

For RAG:

```json
{
  "goal": "Ingest this document text, search it for deployment risks, and produce a cited summary.",
  "max_steps": 10,
  "allowed_tools": ["ingest_task_document", "search_task_documents"]
}
```



## Best Next Backend Improvements

- Add agent profiles/tool presets: `note_agent`, `rag_agent`, `translation_agent`, `research_agent`. Each profile maps to a safe `allowed_tools` set and system prompt extension.
- Add background execution with persisted incremental steps.
- Add tool argument validation in `ToolRegistry`.
- Add side-effect policy: dry-run, confirm-before-write, or committed-action logs.
- Harden or remove default `http_fetch`.

The existing backend is a good prototype foundation. The next step is turning the current “general tool-using loop” into explicit, safe, domain-specific agent workflows.