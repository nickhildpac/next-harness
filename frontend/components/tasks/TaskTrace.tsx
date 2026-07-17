"use client";

import { useMemo } from "react";
import { marked } from "marked";
import type { TaskDetail } from "@/lib/types";
import { escapeHtml, sanitizeMarkdownHtml, taskStepDetail, taskStepLabel } from "../CueApp.helpers";
import styles from "../CueApp.module.css";

type TaskTraceProps = {
  activeTask: TaskDetail | null;
};

export function TaskTrace({ activeTask }: TaskTraceProps) {
  const summaryText = activeTask?.result_summary || activeTask?.error || activeTask?.goal || "";

  const summaryHtml = useMemo(() => {
    if (!summaryText) return "";
    try {
      const result = marked.parse(summaryText);
      return typeof result === "string" ? sanitizeMarkdownHtml(result) : escapeHtml(summaryText);
    } catch {
      return escapeHtml(summaryText);
    }
  }, [summaryText]);

  if (!activeTask) {
    return <div className={styles.empty}>Enter a goal and press Run task, or select a past task.</div>;
  }

  const lastStep = activeTask.steps.at(-1) ?? null;
  const latestStep = lastStep?.kind === "final" ? null : lastStep;

  return (
    <div className={styles.formStack}>
      <div className={styles.card}>
        <strong>{activeTask.status}</strong>
        <div className={styles.taskMeta}>
          <span>{activeTask.steps_taken} reason turn(s)</span>
          <span>max {activeTask.max_steps}</span>
          <span>{activeTask.model || "model unknown"}</span>
          <span>tools: {activeTask.allowed_tools?.length ? activeTask.allowed_tools.join(", ") : "all registered"}</span>
        </div>
        <div className={styles.markdownBody} dangerouslySetInnerHTML={{ __html: summaryHtml }} />
      </div>
      {latestStep ? (
        <div
          key={latestStep.id}
          className={`${styles.taskStep} ${
            latestStep.kind === "final"
              ? styles.taskStepFinal
              : latestStep.kind === "error" || latestStep.ok === false
                ? styles.taskStepError
                : ""
          }`}
        >
          <div className={styles.taskStepHeader}>
            <span className={styles.taskStepIndex}>#{latestStep.step_index}</span>
            <strong>{taskStepLabel(latestStep)}</strong>
            {taskStepDetail(latestStep) ? <span>{taskStepDetail(latestStep)}</span> : null}
            {latestStep.ok !== null ? (
              <span className={latestStep.ok ? styles.taskStepOk : styles.taskStepFailed}>
                {latestStep.ok ? "ok" : "failed"}
              </span>
            ) : null}
          </div>
          {latestStep.content ? <p>{latestStep.content}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
