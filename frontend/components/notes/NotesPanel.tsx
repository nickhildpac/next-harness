"use client";

import { RefObject } from "react";
import type { Note, NoteStyle, ProviderInfo } from "@/lib/types";
import { providerForNote, providerLabel } from "../CueApp.helpers";
import styles from "../CueApp.module.css";

type NotesPanelProps = {
  activeNote: Note | null;
  noteStyles: NoteStyle[];
  notePreview: boolean;
  regenPrompt: string;
  regenPending: boolean;
  noteHtml: string;
  noteEditorRef: RefObject<HTMLTextAreaElement>;
  providers: ProviderInfo[];
  setNotePreview: (value: (prev: boolean) => boolean) => void;
  setRegenPrompt: (value: string) => void;
  saveNote: (note: Note, patch: Partial<Pick<Note, "title" | "content">>) => void;
  deleteNote: (id: string) => void;
  regenerateNote: (noteId: string, prompt: string) => void;
  changeNoteProvider: (noteId: string, providerId: string) => void;
};

export function NotesPanel({
  activeNote,
  noteStyles,
  notePreview,
  regenPrompt,
  regenPending,
  noteHtml,
  noteEditorRef,
  providers,
  setNotePreview,
  setRegenPrompt,
  saveNote,
  deleteNote,
  regenerateNote,
  changeNoteProvider
}: NotesPanelProps) {
  if (!activeNote) {
    return <div className={styles.empty}>Create a note to begin.</div>;
  }

  return (
    <>
      <div className={styles.header}>
        <div className={styles.headerTitle}>
          <div className={styles.title}>{activeNote.title || "Untitled note"}</div>
          <div className={styles.subtitle}>{activeNote.style_name}</div>
        </div>
        <div className={styles.controls}>
          <button
            className={styles.smallButton}
            onClick={() => setNotePreview((v) => !v)}
            style={{ fontWeight: 600 }}
          >
            {notePreview ? "Edit" : "Preview"}
          </button>
          <button className={styles.dangerButton} onClick={() => void deleteNote(activeNote.id)}>
            🗑
          </button>
        </div>
      </div>
      <div className={styles.panel}>
        <div className={styles.formStack}>
          <div className={`${styles.noteBodyWrap} ${regenPending ? styles.noteBodyWrapPending : ""}`}>
            {regenPending ? (
              <div className={styles.noteLoadingOverlay}>
                <span className={styles.noteSpinner} aria-hidden="true" />
                Regenerating note...
              </div>
            ) : null}
            <input
              className={styles.input}
              value={activeNote.title || ""}
              placeholder="Untitled note"
              disabled={regenPending}
              onChange={(event) => void saveNote(activeNote, { title: event.target.value })}
            />
            {notePreview ? (
              <div
                className={`${styles.textarea} ${styles.notePreview}`}
                dangerouslySetInnerHTML={{ __html: noteHtml }}
              />
            ) : (
              <textarea
                ref={noteEditorRef}
                className={`${styles.textarea} ${styles.noteEditor}`}
                value={activeNote.content}
                disabled={regenPending}
                onChange={(event) => void saveNote(activeNote, { content: event.target.value })}
              />
            )}
          </div>
          <div className={`${styles.row} ${regenPending ? styles.noteBodyWrapPending : ""}`}>
            <select
              className={styles.select}
              value={providerForNote(activeNote.id, providers)}
              disabled={regenPending}
              onChange={(event) => changeNoteProvider(activeNote.id, event.target.value)}
            >
              {providers.map((provider) => (
                <option
                  key={provider.id}
                  value={provider.id}
                  disabled={provider.available === false}
                >
                  {providerLabel(provider)}
                </option>
              ))}
            </select>
            <select
              className={styles.select}
              defaultValue={activeNote.style_name}
              disabled={regenPending}
            >
              {noteStyles.map((style) => (
                <option key={style.id} value={style.id}>
                  {style.label}
                </option>
              ))}
            </select>
            <input
              className={styles.input}
              value={regenPrompt}
              disabled={regenPending}
              onChange={(event) => setRegenPrompt(event.target.value)}
              placeholder="Regenerate instruction"
              style={{ flex: 1 }}
            />
            <button
              className={styles.primaryButton}
              disabled={regenPending || !regenPrompt.trim()}
              onClick={() => void regenerateNote(activeNote.id, regenPrompt)}
            >
              {regenPending ? "Regenerating..." : "Regenerate"}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
