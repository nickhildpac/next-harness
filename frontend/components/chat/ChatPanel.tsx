"use client";

import { KeyboardEvent, RefObject } from "react";
import type { ProviderInfo, ToneId, ToneInfo, User } from "@/lib/types";
import type { ConversationView, StreamMessage } from "../CueApp.types";
import { MessageBubble, ProviderSelect } from "../CueApp.widgets";
import {
  PROVIDER_CONV_PREFIX,
  PROVIDER_DEFAULT_KEY,
  participants,
  providerForConversation,
  store,
  toneLabel
} from "../CueApp.helpers";
import styles from "../CueApp.module.css";

type ChatPanelProps = {
  activeConversation: ConversationView | null;
  draft: string;
  streaming: boolean;
  suggesting: boolean;
  uploadingDoc: boolean;
  loadingMessages: boolean;
  messagesRef: RefObject<HTMLDivElement>;
  tones: ToneInfo[];
  providers: ProviderInfo[];
  llmAvailable: boolean;
  healthChecked: boolean;
  user: User | null;
  setDraft: (draft: string) => void;
  sendMessage: (textOverride?: string) => void;
  suggestReply: (conversationId: string, sendAs: string) => void;
  deleteMessage: (conversationId: string, messageId: string) => void;
  deleteConversation: (id: string) => void;
  toggleDocuments: (conversationId: string, next: boolean) => void;
  uploadDocument: (conversationId: string, file: File) => void;
  deleteDocument: (conversationId: string, documentId: string) => void;
  changeTone: (label: string, conversationId: string, currentToneName: ToneId | undefined) => void;
  updateConversation: (id: string, updater: (c: ConversationView) => ConversationView) => void;
};

export function ChatPanel({
  activeConversation,
  draft,
  streaming,
  suggesting,
  uploadingDoc,
  loadingMessages,
  messagesRef,
  tones,
  providers,
  llmAvailable,
  healthChecked,
  user,
  setDraft,
  sendMessage,
  suggestReply,
  deleteMessage,
  deleteConversation,
  toggleDocuments,
  uploadDocument,
  deleteDocument,
  changeTone,
  updateConversation
}: ChatPanelProps) {
  if (!activeConversation) {
    return <div className={styles.empty}>Create a conversation to begin.</div>;
  }

  const isDuo = activeConversation.kind === "duo";
  const people = participants(activeConversation);
  const composerLocked = streaming || suggesting || (!isDuo && !llmAvailable);
  const sendDisabled = !draft.trim() || composerLocked;

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          <div className={styles.title}>
            {isDuo ? "👥 " : ""}
            {activeConversation.title}
          </div>
          <div className={styles.subtitle}>
            {isDuo ? `${people.join(" & ")} · ` : ""}
            {activeConversation.messages.length} messages
          </div>
        </div>
        <div className={styles.controls}>
          <button
            className={styles.dangerButton}
            title="Delete conversation"
            onClick={() => void deleteConversation(activeConversation.id)}
          >
            🗑
          </button>
        </div>
      </div>
      {healthChecked && !llmAvailable && !isDuo ? (
        <div className={styles.statusBar}>connecting...</div>
      ) : null}
      <div className={styles.messages} ref={messagesRef}>
        {loadingMessages ? <div className={styles.empty}>Loading conversation...</div> : null}
        {activeConversation.messages.map((message) => (
          <MessageBubble
            key={message.id || (message as StreamMessage).localId}
            message={message}
            mine={isDuo ? message.user_id === activeConversation.sendAs : message.role === "user"}
            isDuo={isDuo}
            onDelete={
              message.id ? () => void deleteMessage(activeConversation.id, message.id) : undefined
            }
          />
        ))}
      </div>
      <div className={styles.composerWrap}>
        <div className={styles.composer}>
          <div className={styles.composerInputWrap}>
            <textarea
              className={styles.draft}
              rows={3}
              placeholder={
                isDuo
                  ? `Message as ${activeConversation.sendAs || ""}...`
                  : "Message the assistant..."
              }
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
            />
            <div className={styles.composerActionsOverlay}>
              {!isDuo && activeConversation.use_documents ? (
                <label
                  className={styles.iconButton}
                  title={uploadingDoc ? "Uploading..." : "Upload .pdf, .txt, or .md"}
                  style={{ opacity: uploadingDoc ? 0.5 : 1, cursor: uploadingDoc ? "default" : "pointer" }}
                >
                  📎
                  <input
                    hidden
                    type="file"
                    accept=".pdf,.txt,.md"
                    disabled={uploadingDoc}
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) void uploadDocument(activeConversation.id, file);
                    }}
                  />
                </label>
              ) : null}
              {isDuo ? (
                <button
                  className={styles.suggestButton}
                  disabled={suggesting || streaming}
                  onClick={() =>
                    void suggestReply(activeConversation.id, activeConversation.sendAs || "")
                  }
                >
                  ✨
                </button>
              ) : null}
              <button
                className={styles.sendButton}
                disabled={sendDisabled}
                onClick={() => void sendMessage()}
              >
                ↑
              </button>
            </div>
          </div>
          {!isDuo && activeConversation.use_documents &&
           (activeConversation.documents.length > 0 || activeConversation.docsLoaded) ? (
            <div className={styles.composerDocs}>
              {activeConversation.documents.map((document) => (
                <span className={styles.chip} key={document.id}>
                  📄 {document.filename}
                  <button
                    className={styles.deleteInline}
                    onClick={() => void deleteDocument(activeConversation.id, document.id)}
                  >
                    x
                  </button>
                </span>
              ))}
              {activeConversation.docsLoaded && !activeConversation.documents.length ? (
                <span className={styles.preview}>
                  No documents yet — upload a .pdf, .txt, or .md
                </span>
              ) : null}
            </div>
          ) : null}
          <div className={styles.composerActions}>
            {isDuo ? (
              <>
                <span className={styles.label}>Acting as</span>
                <select
                  className={styles.select}
                  value={activeConversation.sendAs || ""}
                  onChange={(event) =>
                    updateConversation(activeConversation.id, (c) => ({
                      ...c,
                      sendAs: event.target.value
                    }))
                  }
                >
                  {people.map((person) => (
                    <option key={person} value={person}>
                      {person}
                    </option>
                  ))}
                </select>
              </>
            ) : (
              <label className={styles.label}>
                <input
                  type="checkbox"
                  checked={activeConversation.use_documents}
                  onChange={(event) =>
                    void toggleDocuments(activeConversation.id, event.target.checked)
                  }
                />{" "}
                Docs
              </label>
            )}
            <span className={styles.label}>Tone</span>
            <select
              className={styles.select}
              value={toneLabel(tones, activeConversation.tone_name)}
              onChange={(event) =>
                void changeTone(event.target.value, activeConversation.id, activeConversation.tone_name)
              }
            >
              {tones.map((tone) => (
                <option key={tone.id} value={tone.label}>
                  {tone.label}
                </option>
              ))}
            </select>
            <span className={styles.label}>Model</span>
            <ProviderSelect
              providers={providers}
              value={providerForConversation(activeConversation.id, providers)}
              onChange={(value) => {
                store(PROVIDER_CONV_PREFIX + activeConversation.id, value);
                store(PROVIDER_DEFAULT_KEY, value);
              }}
            />
          </div>
        </div>
        <div className={styles.hint}>
          {suggesting
            ? `Drafting a reply for ${activeConversation.sendAs || ""}...`
            : streaming
              ? "Assistant is responding..."
              : isDuo
                ? `Enter to send as ${activeConversation.sendAs || ""} · AI drafts their reply with ✨`
                : "Enter to send · Shift+Enter for new line"}
        </div>
      </div>
    </>
  );
}
