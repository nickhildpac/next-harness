"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch, responseErrorMessage } from "@/lib/api";
import type { ApiOptions } from "@/lib/api";
import type {
  AgentThread,
  AgentThreadDetail,
  TaskDetail,
  TaskStep,
  ToolInfo
} from "@/lib/types";
import { requiredToolsForTaskGoal } from "../CueApp.helpers";

export type BackendApi = <T>(path: string, options?: ApiOptions) => Promise<T>;

type UseTasksOptions = {
  api: BackendApi;
  token: string | null;
};

async function streamTaskRun(
  path: string,
  options: {
    token: string | null;
    method?: string;
    json?: unknown;
    onTask: (task: TaskDetail) => void;
    onStep: (step: TaskStep) => void;
    onDone: (task: TaskDetail) => void;
  }
) {
  const headers = new Headers();
  if (options.token) headers.set("Authorization", `Bearer ${options.token}`);
  let body: BodyInit | undefined;
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  }
  const response = await fetch(`/api/backend${path}`, {
    method: options.method || "POST",
    headers,
    body,
    cache: "no-store"
  });
  if (!response.ok) throw new Error(await responseErrorMessage(response));
  if (!response.body) throw new Error(`${response.status} ${response.statusText}`);

  let streamError: string | null = null;
  await readNdjson(response.body, (event) => {
    if (event.event === "error") {
      const payload = event.data as { error?: unknown };
      streamError = String(payload.error || "Stream failed");
      return;
    }
    if (event.event === "task") {
      options.onTask(event.data as TaskDetail);
      return;
    }
    if (event.event === "step") {
      options.onStep(event.data as TaskStep);
      return;
    }
    if (event.event === "done") {
      options.onDone(event.data as TaskDetail);
    }
  });
  if (streamError) throw new Error(streamError);
}

async function readNdjson(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: { event: string; data: unknown }) => void
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const parsed = JSON.parse(trimmed) as { event?: string; data?: unknown };
      onEvent({ event: parsed.event || "message", data: parsed.data });
    }
    if (done) break;
  }

  if (buffer.trim()) {
    const parsed = JSON.parse(buffer.trim()) as { event?: string; data?: unknown };
    onEvent({ event: parsed.event || "message", data: parsed.data });
  }
}

export function useTasks({ api, token }: UseTasksOptions) {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [selectedTools, setSelectedTools] = useState<Set<string>>(new Set());
  const [threads, setThreads] = useState<AgentThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [activeThread, setActiveThread] = useState<AgentThreadDetail | null>(null);
  const [taskGoal, setTaskGoal] = useState("");
  const [taskFiles, setTaskFiles] = useState<File[]>([]);
  const [maxSteps, setMaxSteps] = useState(8);
  const [taskRunning, setTaskRunning] = useState(false);
  const [taskError, setTaskError] = useState("");

  const resetTasks = useCallback(() => {
    setTools([]);
    setSelectedTools(new Set());
    setThreads([]);
    setActiveThreadId(null);
    setActiveThread(null);
    setTaskGoal("");
    setTaskFiles([]);
    setMaxSteps(8);
    setTaskRunning(false);
    setTaskError("");
  }, []);

  useEffect(() => {
    if (!token) resetTasks();
  }, [resetTasks, token]);

  const loadThreads = useCallback(async () => {
    const [toolRows, threadRows] = await Promise.all([
      api<ToolInfo[]>("/tools"),
      api<AgentThread[]>("/threads")
    ]);
    setTools(toolRows);
    setSelectedTools((current) => (current.size ? current : new Set(toolRows.map((tool) => tool.name))));
    setThreads(threadRows);
  }, [api]);

  const selectThread = useCallback(
    async (id: string) => {
      setActiveThreadId(id);
      const thread = await api<AgentThreadDetail>(`/threads/${id}`);
      setActiveThread(thread);
    },
    [api]
  );

  const deleteThread = useCallback(
    async (id: string) => {
      if (!window.confirm("Delete this agent thread? This cannot be undone.")) return;
      await api<void>(`/threads/${id}`, { method: "DELETE" });
      setThreads((current) => current.filter((thread) => thread.id !== id));
      if (activeThreadId === id) {
        setActiveThreadId(null);
        setActiveThread(null);
        setTaskGoal("");
        setTaskFiles([]);
      }
    },
    [activeThreadId, api]
  );

  const startNewThread = useCallback(() => {
    setActiveThreadId(null);
    setActiveThread(null);
    setTaskGoal("");
    setTaskFiles([]);
    setTaskError("");
  }, []);

  const selectAllTools = useCallback(() => {
    setSelectedTools(new Set(tools.map((tool) => tool.name)));
  }, [tools]);

  const clearTools = useCallback(() => {
    setSelectedTools(new Set());
  }, []);

  const chooseTools = useCallback((toolNames: string[]) => {
    setSelectedTools(new Set(toolNames));
  }, []);

  const toggleTool = useCallback((toolName: string) => {
    setSelectedTools((current) => {
      const next = new Set(current);
      if (next.has(toolName)) next.delete(toolName);
      else next.add(toolName);
      return next;
    });
  }, []);

  const addTaskFiles = useCallback((files: File[]) => {
    setTaskFiles((current) => [...current, ...files]);
  }, []);

  const removeTaskFile = useCallback((index: number) => {
    setTaskFiles((current) => current.filter((_, currentIndex) => currentIndex !== index));
  }, []);

  const applyStreamHandlers = useCallback(() => {
    let streamedTaskId: string | null = null;
    let streamedThreadId: string | null = null;

    const upsertTask = (task: TaskDetail) => {
      if (!task.thread_id) return;
      streamedTaskId = task.id;
      streamedThreadId = task.thread_id;
      setActiveThreadId(task.thread_id);
      setActiveThread((current) => {
        if (!current || current.id !== task.thread_id) {
          return {
            id: task.thread_id!,
            user_id: task.user_id,
            title: task.goal,
            created_at: task.created_at,
            updated_at: task.updated_at,
            tasks: [task]
          };
        }
        const exists = current.tasks.some((item) => item.id === task.id);
        return {
          ...current,
          updated_at: task.updated_at,
          tasks: exists
            ? current.tasks.map((item) => (item.id === task.id ? task : item))
            : [...current.tasks, task]
        };
      });
    };

    return {
      onTask: (task: TaskDetail) => {
        upsertTask({ ...task, steps: task.steps || [] });
      },
      onStep: (step: TaskStep) => {
        setActiveThread((current) => {
          if (!current || !streamedTaskId) return current;
          return {
            ...current,
            tasks: current.tasks.map((task) =>
              task.id === streamedTaskId
                ? { ...task, steps: [...task.steps, step] }
                : task
            )
          };
        });
      },
      onDone: (task: TaskDetail) => {
        upsertTask(task);
      },
      threadId: () => streamedThreadId
    };
  }, []);

  const runTask = useCallback(async () => {
    if (!taskGoal.trim()) return;

    const availableToolNames = new Set(tools.map((tool) => tool.name));
    const selected = new Set(selectedTools);
    const taskDocumentTools = taskFiles.length ? ["list_task_documents", "search_task_documents"] : [];
    for (const toolName of [...requiredToolsForTaskGoal(taskGoal), ...taskDocumentTools]) {
      if (availableToolNames.has(toolName)) selected.add(toolName);
    }

    if (tools.length > 0 && selected.size === 0) return;
    if (selected.size !== selectedTools.size) setSelectedTools(selected);

    setTaskRunning(true);
    setTaskError("");
    try {
      const allowed = Array.from(selected);
      const allowedTools = tools.length === 0 || allowed.length === tools.length ? null : allowed;
      const handlers = applyStreamHandlers();
      const payload = {
        goal: taskGoal,
        max_steps: maxSteps,
        allowed_tools: allowedTools
      };

      if (taskFiles.length) {
        let pendingTask: TaskDetail;
        if (activeThreadId) {
          pendingTask = await api<TaskDetail>(`/threads/${activeThreadId}/tasks`, {
            method: "POST",
            json: { ...payload, run: false }
          });
        } else {
          const pendingThread = await api<AgentThreadDetail>("/threads", {
            method: "POST",
            json: { ...payload, run: false }
          });
          pendingTask = pendingThread.tasks[pendingThread.tasks.length - 1];
          if (!pendingTask) throw new Error("Thread was created without a task.");
          setActiveThreadId(pendingThread.id);
          setActiveThread(pendingThread);
        }

        for (const file of taskFiles) {
          const form = new FormData();
          form.append("file", file);
          await apiFetch(`/tasks/${pendingTask.id}/documents`, {
            method: "POST",
            token,
            body: form
          });
        }

        await streamTaskRun(`/tasks/${pendingTask.id}/run?stream=true`, {
          token,
          method: "POST",
          ...handlers
        });
      } else {
        const path = activeThreadId
          ? `/threads/${activeThreadId}/tasks?stream=true`
          : "/threads?stream=true";
        await streamTaskRun(path, {
          token,
          method: "POST",
          json: payload,
          ...handlers
        });
      }

      setTaskGoal("");
      setTaskFiles([]);
      await loadThreads();
      const completedThreadId = handlers.threadId() || activeThreadId;
      if (completedThreadId) await selectThread(completedThreadId);
    } catch (error) {
      setTaskError(error instanceof Error ? error.message : "Agent task failed");
    } finally {
      setTaskRunning(false);
    }
  }, [
    api,
    activeThreadId,
    applyStreamHandlers,
    loadThreads,
    maxSteps,
    selectThread,
    selectedTools,
    taskFiles,
    taskGoal,
    token,
    tools
  ]);

  return {
    tools,
    selectedTools,
    threads,
    activeThreadId,
    activeThread,
    taskGoal,
    taskFiles,
    maxSteps,
    taskRunning,
    taskError,
    setTaskGoal,
    setMaxSteps,
    loadThreads,
    selectThread,
    deleteThread,
    startNewThread,
    runTask,
    selectAllTools,
    clearTools,
    chooseTools,
    toggleTool,
    addTaskFiles,
    removeTaskFile
  };
}
