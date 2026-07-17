"use client";

import { KeyboardEvent as ReactKeyboardEvent, useEffect, useRef, useState } from "react";
import type { ToolInfo } from "@/lib/types";
import {
  formatFileSize,
  requiredToolsForTaskGoal,
  taskToolPresets
} from "../CueApp.helpers";
import styles from "../CueApp.module.css";

type TaskComposerProps = {
  tools: ToolInfo[];
  selectedTools: Set<string>;
  taskGoal: string;
  taskFiles: File[];
  maxSteps: number;
  taskRunning: boolean;
  onGoalChange: (goal: string) => void;
  onMaxStepsChange: (maxSteps: number) => void;
  onRunTask: () => void;
  onSelectAllTools: () => void;
  onClearTools: () => void;
  onChooseTools: (toolNames: string[]) => void;
  onToggleTool: (toolName: string) => void;
  onAddFiles: (files: File[]) => void;
  onRemoveFile: (index: number) => void;
};

export function TaskComposer({
  tools,
  selectedTools,
  taskGoal,
  taskFiles,
  maxSteps,
  taskRunning,
  onGoalChange,
  onMaxStepsChange,
  onRunTask,
  onSelectAllTools,
  onClearTools,
  onChooseTools,
  onToggleTool,
  onAddFiles,
  onRemoveFile
}: TaskComposerProps) {
  const [toolsOpen, setToolsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!toolsOpen) return;
    function handlePointer(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setToolsOpen(false);
      }
    }
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") setToolsOpen(false);
    }
    document.addEventListener("mousedown", handlePointer);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handlePointer);
      document.removeEventListener("keydown", handleKey);
    };
  }, [toolsOpen]);

  const availableToolNames = new Set(tools.map((tool) => tool.name));
  const requiredToolNames = [
    ...new Set([
      ...requiredToolsForTaskGoal(taskGoal),
      ...(taskFiles.length ? ["list_task_documents", "search_task_documents"] : [])
    ])
  ].filter((toolName) => availableToolNames.has(toolName));
  const missingRequiredToolNames = requiredToolNames.filter((toolName) => !selectedTools.has(toolName));
  const effectiveSelectedTools = new Set([...selectedTools, ...requiredToolNames]);
  const availablePresets = taskToolPresets
    .map((preset) => ({
      ...preset,
      tools: preset.tools.filter((toolName) => availableToolNames.has(toolName))
    }))
    .filter((preset) => preset.tools.length > 0);
  const taskRunDisabled = taskRunning || !taskGoal.trim() || (tools.length > 0 && effectiveSelectedTools.size === 0);

  return (
    <>
      {requiredToolNames.length ? (
        <div className={styles.taskHint}>
          <strong>Detected workflow:</strong> this goal needs {requiredToolNames.join(", ")}.
          {missingRequiredToolNames.length
            ? ` Missing tools will be added when the task runs: ${missingRequiredToolNames.join(", ")}.`
            : " Required tools are selected."}
        </div>
      ) : null}
      {taskFiles.length ? (
        <div className={styles.taskFileList}>
          {taskFiles.map((file, index) => (
            <span key={`${file.name}-${file.size}-${index}`} className={styles.taskFilePill}>
              {file.name} · {formatFileSize(file.size)}
              <button className={styles.smallButton} onClick={() => onRemoveFile(index)}>
                Remove
              </button>
            </span>
          ))}
        </div>
      ) : null}
      <div className={styles.composerInputWrap}>
        <textarea
          className={styles.textarea}
          rows={3}
          placeholder="e.g. Summarize my last 3 notes and save the summary as a new note."
          value={taskGoal}
          onChange={(event) => onGoalChange(event.target.value)}
          onKeyDown={(event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (!taskRunDisabled) onRunTask();
            }
          }}
        />
        <div className={styles.composerActionsOverlay}>
          <div className={styles.toolsDropdown} ref={dropdownRef}>
            <button className={styles.ghostButton} onClick={() => setToolsOpen((open) => !open)}>
              Tools · {effectiveSelectedTools.size}/{tools.length}
            </button>
            {toolsOpen ? (
              <div className={styles.toolsDropdownPanel}>
                <div className={styles.taskToolbar}>
                  <button className={styles.ghostButton} onClick={onSelectAllTools}>
                    All tools
                  </button>
                  <button className={styles.ghostButton} onClick={onClearTools}>
                    Clear
                  </button>
                  {availablePresets.map((preset) => (
                    <button
                      key={preset.id}
                      className={styles.ghostButton}
                      title={preset.description}
                      onClick={() => onChooseTools(preset.tools)}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
                {tools.map((tool) => (
                  <label key={tool.name} className={styles.toolCheckRow} title={tool.description}>
                    <input
                      type="checkbox"
                      checked={selectedTools.has(tool.name)}
                      onChange={() => onToggleTool(tool.name)}
                    />
                    {tool.name}
                  </label>
                ))}
                <label className={styles.toolCheckRow}>
                  Max steps
                  <input
                    className={styles.input}
                    type="number"
                    min={1}
                    max={32}
                    value={maxSteps}
                    onChange={(event) => onMaxStepsChange(Number(event.target.value) || 8)}
                  />
                </label>
              </div>
            ) : null}
          </div>
          <label className={styles.iconButton} title="Attach .pdf, .txt, or .md files">
            📎
            <input
              type="file"
              multiple
              accept=".pdf,.txt,.md"
              style={{ display: "none" }}
              onChange={(event) => {
                const files = Array.from(event.target.files || []);
                onAddFiles(files);
                event.target.value = "";
              }}
            />
          </label>
          <button
            className={styles.sendButton}
            disabled={taskRunDisabled}
            title={taskRunning ? "Running..." : "Run agent task"}
            onClick={onRunTask}
          >
            ↑
          </button>
        </div>
      </div>
      {tools.length > 0 && selectedTools.size === 0 ? (
        <div className={styles.taskHint}>
          {requiredToolNames.length
            ? `Required tools will be added when the task runs: ${requiredToolNames.join(", ")}.`
            : "Select at least one tool, or choose All tools."}
        </div>
      ) : (
        <div className={styles.hint}>
          {taskRunning
            ? "Running agent task..."
            : effectiveSelectedTools.size === tools.length
              ? "Enter to run · Shift+Enter for new line · all registered tools are available"
              : `Enter to run · Shift+Enter for new line · ${effectiveSelectedTools.size} of ${tools.length} tools selected`}
        </div>
      )}
    </>
  );
}
