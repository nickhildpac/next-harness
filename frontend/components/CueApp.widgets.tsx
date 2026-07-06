import { FormEvent, useRef, useState } from "react";

import type { Message, ProviderInfo, ToneId, ToneInfo } from "@/lib/types";
import { defaultProvider, providerLabel } from "./CueApp.helpers";
import type { StreamMessage } from "./CueApp.types";
import styles from "./CueApp.module.css";

export function AuthOverlay({
  onSubmit,
  error
}: {
  onSubmit: (mode: "login" | "register", event: FormEvent<HTMLFormElement>) => void;
  error: string;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const modeRef = useRef(mode);
  return (
    <div className={styles.modal}>
      <form className={styles.authCard} onSubmit={(event) => onSubmit(modeRef.current, event)}>
        <div className={styles.brandName}>Cue</div>
        <div className={styles.preview}>{error}</div>
        <input className={styles.input} name="email" type="email" autoComplete="email" placeholder="Email" required />
        <input
          className={styles.input}
          name="password"
          type="password"
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          placeholder="Password"
          required
        />
        <div className={styles.split}>
          <button className={styles.primaryButton} type="submit" onClick={() => { setMode("login"); modeRef.current = "login"; }}>
            Login
          </button>
          <button className={styles.ghostButton} type="submit" onClick={() => { setMode("register"); modeRef.current = "register"; }}>
            Register
          </button>
        </div>
      </form>
    </div>
  );
}

export function ProviderSelect({
  providers,
  value,
  onChange
}: {
  providers: ProviderInfo[];
  value: string;
  onChange: (value: string) => void;
}) {
  const selected = value || defaultProvider(providers)?.id || "";
  return (
    <select className={styles.select} value={selected} onChange={(event) => onChange(event.target.value)}>
      {providers.map((provider) => (
        <option key={provider.id} value={provider.id} disabled={provider.available === false}>
          {providerLabel(provider)}
        </option>
      ))}
    </select>
  );
}

export function TranslationBubble({
  mine,
  label,
  text
}: {
  mine?: boolean;
  label: string;
  text: string;
}) {
  return (
    <div className={`${styles.messageRow} ${mine ? styles.messageMine : styles.messageOther}`}>
      <div className={`${styles.messageStack} ${mine ? styles.messageStackMine : styles.messageStackOther}`}>
        <div className={styles.sender}>{label}</div>
        <div className={`${styles.bubble} ${mine ? styles.bubbleMine : ""}`}>{text}</div>
      </div>
    </div>
  );
}

export function MessageBubble({
  message,
  mine,
  isDuo,
  onDelete
}: {
  message: Message | StreamMessage;
  mine: boolean;
  isDuo: boolean;
  onDelete?: () => void;
}) {
  const sender = isDuo && message.user_id ? `${message.user_id}${message.model ? " · AI draft" : ""}` : message.role === "user" ? "You" : "Assistant";
  return (
    <div className={`${styles.messageRow} ${mine ? styles.messageMine : styles.messageOther}`}>
      {mine && onDelete ? (
        <button className={styles.deleteInline} onClick={onDelete}>
          x
        </button>
      ) : null}
      <div className={`${styles.messageStack} ${mine ? styles.messageStackMine : styles.messageStackOther}`}>
        <div className={styles.sender}>{sender}</div>
        <div className={`${styles.bubble} ${mine ? styles.bubbleMine : ""}`}>
          {message.content}
          {(message as StreamMessage).streaming ? <span className={styles.cursor} /> : null}
        </div>
        {!mine && message.citations?.length ? (
          <div className={styles.sources}>
            Sources:
            {message.citations.map((citation) => (
              <span key={`${citation.document_id}-${citation.marker}`} title={citation.snippet}>
                [{citation.marker}] {citation.filename}
                {citation.page ? ` p.${citation.page}` : ""}
              </span>
            ))}
          </div>
        ) : null}
      </div>
      {!mine && onDelete ? (
        <button className={styles.deleteInline} onClick={onDelete}>
          x
        </button>
      ) : null}
    </div>
  );
}

export function NewConversationModal({
  tones,
  providers,
  onClose,
  onCreate
}: {
  tones: ToneInfo[];
  providers: ProviderInfo[];
  onClose: () => void;
  onCreate: (payload: {
    title?: string;
    tone_name: string;
    custom_persona?: string;
    participants?: string[];
    provider?: string;
  }) => Promise<void>;
}) {
  const [kind, setKind] = useState("assistant");
  const [tone, setTone] = useState<ToneId>("friendly");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const first = String(form.get("first") || "").trim();
    const second = String(form.get("second") || "").trim();
    if (kind === "duo" && (!first || !second || first === second)) return;
    setBusy(true);
    try {
      await onCreate({
        title: String(form.get("title") || "").trim() || undefined,
        tone_name: tone,
        custom_persona: tone === "custom" ? String(form.get("persona") || "").trim() || undefined : undefined,
        participants: kind === "duo" ? [first, second] : undefined,
        provider: String(form.get("provider") || "")
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.modal}>
      <form className={styles.modalCard} onSubmit={submit}>
        <div className={styles.title}>New conversation</div>
        <label className={styles.formStack}>
          <span className={styles.label}>Type</span>
          <select className={styles.select} value={kind} onChange={(event) => setKind(event.target.value)}>
            <option value="assistant">Assistant chat</option>
            <option value="duo">Two people</option>
          </select>
        </label>
        <label className={styles.formStack}>
          <span className={styles.label}>Title</span>
          <input className={styles.input} name="title" maxLength={255} placeholder="New conversation" />
        </label>
        {kind === "duo" ? (
          <div className={styles.split}>
            <label className={styles.formStack}>
              <span className={styles.label}>Participant 1</span>
              <input className={styles.input} name="first" maxLength={128} placeholder="alice" />
            </label>
            <label className={styles.formStack}>
              <span className={styles.label}>Participant 2</span>
              <input className={styles.input} name="second" maxLength={128} placeholder="bob" />
            </label>
          </div>
        ) : null}
        <label className={styles.formStack}>
          <span className={styles.label}>Tone</span>
          <select className={styles.select} value={tone} onChange={(event) => setTone(event.target.value as ToneId)}>
            {tones.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
            <option value="custom">Custom</option>
          </select>
        </label>
        <label className={styles.formStack}>
          <span className={styles.label}>Model</span>
          <select className={styles.select} name="provider" defaultValue={defaultProvider(providers)?.id}>
            {providers.map((provider) => (
              <option key={provider.id} value={provider.id} disabled={provider.available === false}>
                {providerLabel(provider)}
              </option>
            ))}
          </select>
        </label>
        {tone === "custom" ? (
          <label className={styles.formStack}>
            <span className={styles.label}>Custom persona</span>
            <textarea className={styles.textarea} name="persona" maxLength={800} rows={3} />
          </label>
        ) : null}
        <div className={styles.row} style={{ justifyContent: "flex-end" }}>
          <button type="button" className={styles.ghostButton} onClick={onClose}>
            Cancel
          </button>
          <button className={styles.primaryButton} disabled={busy}>
            Create
          </button>
        </div>
      </form>
    </div>
  );
}
