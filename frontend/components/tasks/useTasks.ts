"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import type { ApiOptions } from "@/lib/api";
import type { Task, TaskDetail, ToolInfo } from "@/lib/types";
import { requiredToolsForTaskGoal } from "../CueApp.helpers";

export type BackendApi = <T>(path: string, options?: ApiOptions) => Promise<T>;

type UseTasksOptions = {
  api: BackendApi;
  token: string | null;
};

export function useTasks({ api, token }: UseTasksOptions) {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [selectedTools, setSelectedTools] = useState<Set<string>>(new Set());
  const [tasks, setTasks] = useState<Task[]>([]);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [activeTask, setActiveTask] = useState<TaskDetail | null>(null);
  const [taskGoal, setTaskGoal] = useState("");
  const [taskFiles, setTaskFiles] = useState<File[]>([]);
  const [maxSteps, setMaxSteps] = useState(8);
  const [taskRunning, setTaskRunning] = useState(false);

  const resetTasks = useCallback(() => {
    setTools([]);
    setSelectedTools(new Set());
    setTasks([]);
    setActiveTaskId(null);
    setActiveTask(null);
    setTaskGoal("");
    setTaskFiles([]);
    setMaxSteps(8);
    setTaskRunning(false);
  }, []);

  useEffect(() => {
    if (!token) resetTasks();
  }, [resetTasks, token]);

  const loadTasks = useCallback(async () => {
    const [toolRows, taskRows] = await Promise.all([api<ToolInfo[]>("/tools"), api<Task[]>("/tasks")]);
    setTools(toolRows);
    setSelectedTools((current) => (current.size ? current : new Set(toolRows.map((tool) => tool.name))));
    setTasks(taskRows);
  }, [api]);

  const selectTask = useCallback(
    async (id: string) => {
      setActiveTaskId(id);
      const task = await api<TaskDetail>(`/tasks/${id}`);
      setActiveTask(task);
    },
    [api]
  );

  const startNewTask = useCallback(() => {
    setActiveTaskId(null);
    setActiveTask(null);
    setTaskFiles([]);
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
    try {
      const allowed = Array.from(selected);
      const allowedTools = tools.length === 0 || allowed.length === tools.length ? null : allowed;
      let task: TaskDetail;

      if (taskFiles.length) {
        const pendingTask = await api<TaskDetail>("/tasks", {
          method: "POST",
          json: {
            goal: taskGoal,
            max_steps: maxSteps,
            allowed_tools: allowedTools,
            run: false
          }
        });

        for (const file of taskFiles) {
          const form = new FormData();
          form.append("file", file);
          await apiFetch(`/tasks/${pendingTask.id}/documents`, {
            method: "POST",
            token,
            body: form
          });
        }

        task = await api<TaskDetail>(`/tasks/${pendingTask.id}/run`, { method: "POST" });
      } else {
        task = await api<TaskDetail>("/tasks", {
          method: "POST",
          json: {
            goal: taskGoal,
            max_steps: maxSteps,
            allowed_tools: allowedTools
          }
        });
      }

      setTaskGoal("");
      setTaskFiles([]);
      setActiveTaskId(task.id);
      setActiveTask(task);
      await loadTasks();
    } finally {
      setTaskRunning(false);
    }
  }, [api, loadTasks, maxSteps, selectedTools, taskFiles, taskGoal, token, tools]);

  return {
    tools,
    selectedTools,
    tasks,
    activeTaskId,
    activeTask,
    taskGoal,
    taskFiles,
    maxSteps,
    taskRunning,
    setTaskGoal,
    setMaxSteps,
    loadTasks,
    selectTask,
    startNewTask,
    runTask,
    selectAllTools,
    clearTools,
    chooseTools,
    toggleTool,
    addTaskFiles,
    removeTaskFile
  };
}
