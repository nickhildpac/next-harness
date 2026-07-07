"use client";

import type { TaskDetail } from "@/lib/types";
import { taskStepDetail, taskStepLabel } from "../CueApp.helpers";
import styles from "../CueApp.module.css";

type TaskTraceProps = {
  activeTask: TaskDetail | null;
};

export function TaskTrace({ activeTask }: TaskTraceProps) {
  if (!activeTask) {
    return <div className={styles.empty}>Enter a goal and press Run task, or select a past task.</div>;
  }

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
        <p>{activeTask.result_summary || activeTask.error || activeTask.goal}</p>
      </div>
      {activeTask.steps.map((step) => (
        <div
          key={step.id}
          className={`${styles.taskStep} ${
            step.kind === "final" ? styles.taskStepFinal : step.kind === "error" || step.ok === false ? styles.taskStepError : ""
          }`}
        >
          <div className={styles.taskStepHeader}>
            <span className={styles.taskStepIndex}>#{step.step_index}</span>
            <strong>{taskStepLabel(step)}</strong>
            {taskStepDetail(step) ? <span>{taskStepDetail(step)}</span> : null}
            {step.ok !== null ? (
              <span className={step.ok ? styles.taskStepOk : styles.taskStepFailed}>
                {step.ok ? "ok" : "failed"}
              </span>
            ) : null}
          </div>
          {step.content ? <p>{step.content}</p> : null}
          {step.payload ? <pre className={styles.pre}>{JSON.stringify(step.payload, null, 2)}</pre> : null}
        </div>
      ))}
    </div>
  );
}
