"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { marked } from "marked";
import type { ApiOptions } from "@/lib/api";
import type { Note, NoteStyle, ProviderInfo } from "@/lib/types";
import {
  PROVIDER_DEFAULT_KEY,
  PROVIDER_NOTE_PREFIX,
  escapeHtml,
  providerForNote,
  sanitizeMarkdownHtml,
  store
} from "../CueApp.helpers";

type UseNotesParams = {
  api: <T>(path: string, options?: ApiOptions) => Promise<T>;
  token: string | null;
  providers: ProviderInfo[];
};

export function useNotes({ api, token, providers }: UseNotesParams) {
  const [notes, setNotes] = useState<Note[]>([]);
  const [activeNoteId, setActiveNoteId] = useState<string | null>(null);
  const [noteStyles, setNoteStyles] = useState<NoteStyle[]>([
    { id: "default", label: "Default" },
    { id: "academic", label: "Academic" },
    { id: "meeting", label: "Meeting" },
    { id: "blog", label: "Blog" }
  ]);
  const [notePreview, setNotePreview] = useState(false);
  const [regenPrompt, setRegenPrompt] = useState("");
  const [regenPending, setRegenPending] = useState(false);

  const noteEditorRef = useRef<HTMLTextAreaElement>(null);

  const activeNote = notes.find((n) => n.id === activeNoteId) || null;

  const noteHtml = useMemo(() => {
    if (!notePreview || !activeNote?.content) return "";
    try {
      const result = marked.parse(activeNote.content);
      return typeof result === "string" ? sanitizeMarkdownHtml(result) : "";
    } catch {
      return escapeHtml(activeNote.content);
    }
  }, [activeNote?.content, notePreview]);

  useEffect(() => {
    if (!token) {
      setNotes([]);
      setActiveNoteId(null);
      setNotePreview(false);
      setRegenPrompt("");
      setRegenPending(false);
    }
  }, [token]);

  useEffect(() => {
    noteEditorRef.current?.focus();
  }, [activeNoteId]);

  const loadNotes = useCallback(async () => {
    const rows = await api<Note[]>("/notes");
    setNotes(rows);
    setActiveNoteId((current) =>
      current && rows.some((n) => n.id === current) ? current : rows[0]?.id || null
    );
  }, [api]);

  const loadNoteStyles = useCallback(async () => {
    try {
      const rows = await api<NoteStyle[]>("/note-styles");
      setNoteStyles(rows);
    } catch {
      // keep fallback styles
    }
  }, [api]);

  const createNote = useCallback(async () => {
    const note = await api<Note>("/notes", {
      method: "POST",
      json: { title: "Untitled note", content: "", style: { style_name: "default" } }
    });
    setNotes((current) => [note, ...current]);
    setActiveNoteId(note.id);
    setNotePreview(false);
  }, [api]);

  const saveNote = useCallback(
    async (note: Note, patch: Partial<Pick<Note, "title" | "content">>) => {
      setNotes((current) => current.map((n) => (n.id === note.id ? { ...n, ...patch } : n)));
      const updated = await api<Note>(`/notes/${note.id}`, { method: "PATCH", json: patch });
      setNotes((current) => current.map((n) => (n.id === note.id ? updated : n)));
    },
    [api]
  );

  const deleteNote = useCallback(
    async (id: string) => {
      await api<void>(`/notes/${id}`, { method: "DELETE" });
      setNotes((current) => {
        const remaining = current.filter((n) => n.id !== id);
        setActiveNoteId((cur) => (cur === id ? remaining[0]?.id || null : cur));
        return remaining;
      });
    },
    [api]
  );

  const regenerateNote = useCallback(
    async (noteId: string, prompt: string) => {
      if (!prompt.trim()) return;
      setRegenPending(true);
      try {
        const data = await api<{ content: string }>(`/notes/${noteId}/regenerate`, {
          method: "POST",
          provider: providerForNote(noteId, providers),
          json: { prompt }
        });
        setNotes((current) =>
          current.map((n) => (n.id === noteId ? { ...n, content: data.content } : n))
        );
        setRegenPrompt("");
      } finally {
        setRegenPending(false);
      }
    },
    [api, providers]
  );

  const changeNoteProvider = useCallback(
    (noteId: string, providerId: string) => {
      store(PROVIDER_NOTE_PREFIX + noteId, providerId);
      store(PROVIDER_DEFAULT_KEY, providerId);
    },
    []
  );

  return {
    notes,
    activeNoteId,
    activeNote,
    noteStyles,
    notePreview,
    regenPrompt,
    regenPending,
    noteHtml,
    noteEditorRef,
    setActiveNoteId,
    setNotePreview,
    setRegenPrompt,
    loadNotes,
    loadNoteStyles,
    createNote,
    saveNote,
    deleteNote,
    regenerateNote,
    changeNoteProvider
  };
}
