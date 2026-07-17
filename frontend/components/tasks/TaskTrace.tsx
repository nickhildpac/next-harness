"use client";

import { useMemo } from "react";
import { marked } from "marked";
import type { AgentThreadDetail, TaskDetail } from "@/lib/types";
import { escapeHtml, sanitizeMarkdownHtml, taskStepDetail, taskStepLabel } from "../CueApp.helpers";
import styles from "../CueApp.module.css";

type TaskTraceProps = {
  activeThread: AgentThreadDetail | null;
};

export function TaskTrace({ activeThread }: TaskTraceProps) {
  if (!activeThread) {
    return <div className={styles.empty}>Enter a goal to start a thread, or select a past thread.</div>;
  }

  return (
    <div className={styles.messages}>
      {activeThread.tasks.map((task) => (
        <TaskRunTrace key={task.id} task={task} />
      ))}
    </div>
  );
}

function TaskRunTrace({ task }: { task: TaskDetail }) {
  const isRunning = task.status === "running" || task.status === "pending";
  const resultText = task.result_summary || task.error || "";
  const resultHtml = useMemo(() => {
    if (!resultText) return "";
    try {
      const result = marked.parse(resultText);
      return typeof result === "string" ? sanitizeMarkdownHtml(result) : escapeHtml(resultText);
    } catch {
      return escapeHtml(resultText);
    }
  }, [resultText]);

  const activitySteps = task.steps.filter(
    (step) => step.kind === "tool_call" || step.kind === "tool_result"
  );

  return (
    <>
      <div className={`${styles.messageRow} ${styles.messageMine}`}>
        <div className={`${styles.messageStack} ${styles.messageStackMine}`}>
          <div className={styles.sender}>You</div>
          <div className={`${styles.bubble} ${styles.bubbleMine}`}>{task.goal}</div>
        </div>
      </div>
      <div className={`${styles.messageRow} ${styles.messageOther}`}>
        <div className={`${styles.messageStack} ${styles.messageStackOther}`}>
          <div className={styles.sender}>Agent</div>
          {isRunning ? (
            <div className={styles.taskActivity}>
              {activitySteps.map((step) => (
                <div
                  key={step.id}
                  className={`${styles.taskActivityRow} ${
                    step.ok === false ? styles.taskActivityFailed : ""
                  }`}
                >
                  <span className={styles.taskActivityLabel}>{taskStepLabel(step)}</span>
                  <span>{taskStepDetail(step) || step.tool_name || ""}</span>
                </div>
              ))}
              <div className={styles.taskActivityRow}>
                <span>Working...</span>
                <span className={styles.cursor} />
              </div>
            </div>
          ) : resultHtml ? (
            <div
              className={`${styles.bubble} ${styles.markdownBody}`}
              dangerouslySetInnerHTML={{ __html: resultHtml }}
            />
          ) : (
            <div className={styles.bubble}>No result produced.</div>
          )}
        </div>
      </div>
    </>
  );
}
