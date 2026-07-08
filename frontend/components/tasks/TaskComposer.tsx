"use client";

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
      <textarea
        className={styles.textarea}
        rows={4}
        placeholder="e.g. Summarize my last 3 notes and save the summary as a new note."
        value={taskGoal}
        onChange={(event) => onGoalChange(event.target.value)}
      />
      {requiredToolNames.length ? (
        <div className={styles.taskHint}>
          <strong>Detected workflow:</strong> this goal needs {requiredToolNames.join(", ")}.
          {missingRequiredToolNames.length
            ? ` Missing tools will be added when the task runs: ${missingRequiredToolNames.join(", ")}.`
            : " Required tools are selected."}
        </div>
      ) : null}
      <div className={styles.row}>
        <input
          className={styles.input}
          type="number"
          min={1}
          max={32}
          value={maxSteps}
          onChange={(event) => onMaxStepsChange(Number(event.target.value) || 8)}
        />
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
        <button className={styles.primaryButton} disabled={taskRunDisabled} onClick={onRunTask}>
          {taskRunning ? "Running..." : "Run task"}
        </button>
      </div>
      {taskFiles.length ? (
        <div className={styles.taskFileList} style={{ marginTop: 8 }}>
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
      <div className={styles.row} style={{ flexWrap: "wrap" }}>
        {tools.map((tool) => (
          <button
            key={tool.name}
            className={styles.chip}
            onClick={() => onToggleTool(tool.name)}
            style={{
              borderColor: selectedTools.has(tool.name) ? "var(--accent)" : "var(--border-input)"
            }}
            title={tool.description}
          >
            {tool.name}
          </button>
        ))}
      </div>
      {tools.length > 0 && selectedTools.size === 0 ? (
        <div className={styles.taskHint}>
          {requiredToolNames.length
            ? `Required tools will be added when the task runs: ${requiredToolNames.join(", ")}.`
            : "Select at least one tool, or choose All tools."}
        </div>
      ) : (
        <div className={styles.hint}>
          {effectiveSelectedTools.size === tools.length
            ? "All registered tools are available. The backend keeps finish in scope automatically."
            : `${effectiveSelectedTools.size} of ${tools.length} tools selected. The backend keeps finish in scope automatically.`}
        </div>
      )}
    </>
  );
}
