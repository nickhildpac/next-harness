"use client";

import { FormEvent, KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { marked } from "marked";
import { ApiError, apiFetch, apiJson, readSse, responseErrorMessage } from "@/lib/api";
import type { ApiOptions } from "@/lib/api";
import type {
  Conversation,
  DocumentInfo,
  Language,
  Message,
  Note,
  NoteStyle,
  PaginatedMessages,
  ProviderInfo,
  ToneInfo,
  TokenResponse,
  TranslationSession,
  TranslationSessionDetail,
  TranslationTurn,
  User
} from "@/lib/types";
import {
  AUTH_TOKEN_KEY,
  PROVIDER_CONV_PREFIX,
  PROVIDER_DEFAULT_KEY,
  PROVIDER_NOTE_PREFIX,
  PROVIDER_TRANSLATE_KEY,
  THEME_KEY,
  defaultProvider,
  escapeHtml,
  fallbackProviders,
  fallbackTones,
  newestPreview,
  normalizeConversation,
  participants,
  providerForConversation,
  providerForNote,
  providerForTranslate,
  providerLabel,
  sanitizeMarkdownHtml,
  store,
  stored,
  timeAgo,
  toneColor,
  toneId,
  toneLabel
} from "./CueApp.helpers";
import type { ConversationView, StreamMessage, Tab, TranslationView } from "./CueApp.types";
import {
  AuthOverlay,
  MessageBubble,
  NewConversationModal,
  ProviderSelect,
  TranslationBubble
} from "./CueApp.widgets";
import { TaskComposer } from "./tasks/TaskComposer";
import { TaskTrace } from "./tasks/TaskTrace";
import { useTasks } from "./tasks/useTasks";
import styles from "./CueApp.module.css";

export function CueApp() {
  const [theme, setTheme] = useState("light");
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [authError, setAuthError] = useState("");
  const [tab, setTab] = useState<Tab>("chats");
  const [tones, setTones] = useState<ToneInfo[]>(fallbackTones);
  const [providers, setProviders] = useState<ProviderInfo[]>(fallbackProviders);
  const [llmAvailable, setLlmAvailable] = useState(true);
  const [healthChecked, setHealthChecked] = useState(false);
  const [conversations, setConversations] = useState<ConversationView[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [uploadingDoc, setUploadingDoc] = useState(false);
  const [lastSendText, setLastSendText] = useState("");
  const [newModalOpen, setNewModalOpen] = useState(false);

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

  const [translationSessions, setTranslationSessions] = useState<TranslationSession[]>([]);
  const [activeTranslationId, setActiveTranslationId] = useState<string | null>(null);
  const [activeTranslation, setActiveTranslation] = useState<TranslationView | null>(null);
  const [loadingTranslation, setLoadingTranslation] = useState(false);
  const [languages, setLanguages] = useState<Language[]>([]);
  const [translateText, setTranslateText] = useState("");
  const [translateLanguage, setTranslateLanguage] = useState("");
  const [translateProvider, setTranslateProvider] = useState("");
  const [translatePending, setTranslatePending] = useState(false);

  const messagesRef = useRef<HTMLDivElement>(null);
  const translateMessagesRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  const activeConversation = conversations.find((conversation) => conversation.id === activeId) || null;
  const activeNote = notes.find((note) => note.id === activeNoteId) || null;
  const lastMessageContent = activeConversation?.messages.at(-1)?.content;
  const lastTranslationTurn = activeTranslation?.turns.at(-1);

  const noteHtml = useMemo(() => {
    if (!notePreview || !activeNote?.content) return "";
    try {
      const result = marked.parse(activeNote.content);
      return typeof result === "string" ? sanitizeMarkdownHtml(result) : "";
    } catch {
      return escapeHtml(activeNote.content);
    }
  }, [activeNote?.content, notePreview]);

  const handleAuthFailure = useCallback(() => {
    store(AUTH_TOKEN_KEY, null);
    setToken(null);
    setUser(null);
    setConversations([]);
    setActiveId(null);
    setNotes([]);
    setTranslationSessions([]);
    setActiveTranslation(null);
  }, []);

  const api = useCallback(
    async <T,>(path: string, options: ApiOptions = {}) => {
      try {
        return await apiJson<T>(path, { ...options, token });
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) handleAuthFailure();
        throw error;
      }
    },
    [handleAuthFailure, token]
  );

  const {
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
  } = useTasks({ api, token });

  const loadMessages = useCallback(
    async (id: string) => {
      setLoadingMessages(true);
      try {
        const data = await api<PaginatedMessages>(
          `/conversations/${encodeURIComponent(id)}/messages?limit=200&offset=0`
        );
        setConversations((current) =>
          current.map((conversation) =>
            conversation.id === id
              ? { ...conversation, messages: data.items, loaded: true }
              : conversation
          )
        );
      } finally {
        setLoadingMessages(false);
      }
    },
    [api]
  );

  const loadConversations = useCallback(async () => {
    if (!token) return;
    setLoadingConversations(true);
    try {
      const rows = await api<Conversation[]>("/conversations");
      const mapped = rows.map(normalizeConversation);
      setConversations(mapped);
      const firstId = mapped[0]?.id || null;
      setActiveId(firstId);
      if (firstId) await loadMessages(firstId);
    } finally {
      setLoadingConversations(false);
    }
  }, [api, loadMessages, token]);

  useEffect(() => {
    const savedTheme =
      stored(THEME_KEY) ||
      (window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    setTheme(savedTheme);
    document.documentElement.setAttribute("data-theme", savedTheme);
    setToken(stored(AUTH_TOKEN_KEY));
  }, []);

  useEffect(() => {
    if (!token) return;
    Promise.all([
      apiJson<ToneInfo[]>("/tones").then(setTones).catch(() => undefined),
      apiJson<ProviderInfo[]>("/providers").then((rows) => {
        setProviders(rows);
        setTranslateProvider((current) => current || providerForTranslate(rows));
      }).catch(() => undefined),
      apiJson<NoteStyle[]>("/note-styles").then(setNoteStyles).catch(() => undefined),
      apiJson<Language[]>("/languages").then((rows) => {
        setLanguages(rows);
        setTranslateLanguage((current) => current || rows[0]?.label || rows[0]?.id || "");
      }).catch(() => undefined),
      apiJson<{ available: boolean }>("/health/llm").then((data) => {
        setLlmAvailable(data.available !== false);
        setHealthChecked(true);
      }).catch(() => {
        setLlmAvailable(false);
        setHealthChecked(true);
      })
    ]).catch(() => undefined);
  }, [token]);

  useEffect(() => {
    if (!token) return;
    apiJson<User>("/auth/me", { token })
      .then((currentUser) => {
        setUser(currentUser);
        void loadConversations();
      })
      .catch((error) => {
        if (error instanceof ApiError && error.status === 401) handleAuthFailure();
      });
  }, [handleAuthFailure, loadConversations, token]);

  useEffect(() => {
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight });
  }, [activeConversation?.messages.length, lastMessageContent]);

  async function submitAuth(mode: "login" | "register", event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setAuthError("");
    try {
      const data = await apiJson<TokenResponse>(`/auth/${mode}`, {
        method: "POST",
        json: {
          email: String(form.get("email") || ""),
          password: String(form.get("password") || "")
        }
      });
      store(AUTH_TOKEN_KEY, data.access_token);
      setToken(data.access_token);
      setUser(data.user);
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Authentication failed");
    }
  }

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    store(THEME_KEY, next);
    document.documentElement.setAttribute("data-theme", next);
  }

  function logout() {
    handleAuthFailure();
  }

  function updateConversation(id: string, updater: (conversation: ConversationView) => ConversationView) {
    setConversations((current) =>
      current.map((conversation) => (conversation.id === id ? updater(conversation) : conversation))
    );
  }

  function pushMessage(id: string, message: Message | StreamMessage) {
    updateConversation(id, (conversation) => ({
      ...conversation,
      messages: [...conversation.messages, message as Message]
    }));
  }

  async function selectConversation(id: string) {
    if (streamAbortRef.current) streamAbortRef.current.abort();
    setActiveId(id);
    setDraft("");
    await loadMessages(id);
  }

  async function createConversation(payload: {
    title?: string;
    tone_name: string;
    custom_persona?: string;
    participants?: string[];
    provider?: string;
  }) {
    const conversation = await api<Conversation>("/conversations", {
      method: "POST",
      json: {
        title: payload.title || (payload.participants ? payload.participants.join(" & ") : "New conversation"),
        tone: {
          tone_name: payload.tone_name,
          custom_persona: payload.custom_persona || undefined
        },
        participants: payload.participants
      }
    });
    const mapped = normalizeConversation(conversation);
    store(PROVIDER_CONV_PREFIX + mapped.id, payload.provider || defaultProvider(providers)?.id || null);
    setConversations((current) => [mapped, ...current]);
    setActiveId(mapped.id);
    setNewModalOpen(false);
  }

  async function changeTone(label: string) {
    if (!activeConversation) return;
    const previous = activeConversation.tone_name;
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      tone_name: toneId(tones, label)
    }));
    try {
      const data = await api<Conversation>(`/conversations/${activeConversation.id}/tone`, {
        method: "PATCH",
        json: { tone: toneId(tones, label) }
      });
      updateConversation(activeConversation.id, (conversation) => ({
        ...conversation,
        tone_name: data.tone_name
      }));
    } catch {
      updateConversation(activeConversation.id, (conversation) => ({ ...conversation, tone_name: previous }));
    }
  }

  async function deleteConversation(id: string) {
    if (!window.confirm("Delete this conversation? This cannot be undone.")) return;
    await api<void>(`/conversations/${id}`, { method: "DELETE" });
    setConversations((current) => current.filter((conversation) => conversation.id !== id));
    setActiveId((current) => (current === id ? conversations.find((item) => item.id !== id)?.id || null : current));
  }

  async function deleteMessage(messageId: string) {
    if (!activeConversation) return;
    await api<void>(`/conversations/${activeConversation.id}/messages/${messageId}`, { method: "DELETE" });
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      messages: conversation.messages.filter((message) => message.id !== messageId)
    }));
  }

  async function sendMessage(textOverride?: string) {
    const text = (textOverride || draft).trim();
    if (!activeConversation || !text || streaming || suggesting) return;
    if (activeConversation.kind === "duo") return sendDuoMessage(text);
    if (!llmAvailable) return;

    setLastSendText(text);
    setDraft("");
    setStreaming(true);
    const userLocal: StreamMessage = {
      localId: crypto.randomUUID(),
      role: "user",
      content: text,
      user_id: user?.id || ""
    };
    const assistantLocal: StreamMessage = {
      localId: crypto.randomUUID(),
      role: "assistant",
      content: "",
      streaming: true,
      user_id: user?.id || ""
    };
    pushMessage(activeConversation.id, userLocal);
    pushMessage(activeConversation.id, assistantLocal);

    const abort = new AbortController();
    streamAbortRef.current = abort;
    try {
      const headers = new Headers({ "Content-Type": "application/json" });
      if (token) headers.set("Authorization", `Bearer ${token}`);
      headers.set("X-LLM-Provider", providerForConversation(activeConversation.id, providers));
      const response = await fetch(`/api/backend/conversations/${activeConversation.id}/messages?stream=true`, {
        method: "POST",
        headers,
        body: JSON.stringify({ text, tone_override: { tone: activeConversation.tone_name } }),
        signal: abort.signal
      });
      if (response.status === 401) handleAuthFailure();
      if (!response.ok) throw new Error(await responseErrorMessage(response));
      if (!response.body) throw new Error(`${response.status} ${response.statusText}`);
      await readSse(response.body, (event) => {
        if (event.data === "[DONE]") return;
        const payload = JSON.parse(event.data) as Record<string, unknown>;
        if (event.event === "error") throw new Error(String(payload.error || "Stream failed"));
        updateConversation(activeConversation.id, (conversation) => ({
          ...conversation,
          messages: conversation.messages.map((message) => {
            if ((message as StreamMessage).localId === userLocal.localId && payload.user_message_id) {
              return { ...message, id: String(payload.user_message_id) };
            }
            if ((message as StreamMessage).localId !== assistantLocal.localId) return message;
            if (event.event === "delta") {
              return { ...message, content: `${message.content}${String(payload.delta || payload.content || "")}` };
            }
            if (event.event === "citations") {
              return { ...message, citations: payload.citations as Message["citations"] };
            }
            if (event.event === "done") {
              return {
                ...message,
                id: String(payload.assistant_message_id || message.id || ""),
                streaming: false,
                token_count: Number(payload.output_tokens || 0)
              };
            }
            return message;
          })
        }));
      });
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        setDraft(text);
        updateConversation(activeConversation.id, (conversation) => ({
          ...conversation,
          messages: conversation.messages
            .filter((message) => (message as StreamMessage).localId !== userLocal.localId)
            .map((message) =>
              (message as StreamMessage).localId === assistantLocal.localId
                ? { ...message, content: `Send failed: ${(error as Error).message}`, streaming: false, error: true }
                : message
            )
        }));
      }
    } finally {
      setStreaming(false);
      streamAbortRef.current = null;
    }
  }

  async function sendDuoMessage(text: string) {
    if (!activeConversation || !activeConversation.sendAs) return;
    setDraft("");
    setStreaming(true);
    try {
      const response = await api<{ user_message: Message }>(
        `/conversations/${activeConversation.id}/messages`,
        {
          method: "POST",
          json: { user_id: activeConversation.sendAs, content: text }
        }
      );
      pushMessage(activeConversation.id, response.user_message);
    } finally {
      setStreaming(false);
    }
  }

  async function suggestReply() {
    if (!activeConversation?.sendAs || suggesting) return;
    setSuggesting(true);
    try {
      const response = await api<{ message: Message | null; content: string }>(
        `/conversations/${activeConversation.id}/suggest`,
        {
          method: "POST",
          provider: providerForConversation(activeConversation.id, providers),
          json: { for_user: activeConversation.sendAs, persist: true }
        }
      );
      if (response.message) pushMessage(activeConversation.id, response.message);
      else setDraft(response.content);
    } finally {
      setSuggesting(false);
    }
  }

  async function toggleDocuments(next: boolean) {
    if (!activeConversation) return;
    updateConversation(activeConversation.id, (conversation) => ({ ...conversation, use_documents: next }));
    await api<Conversation>(`/conversations/${activeConversation.id}/rag`, {
      method: "PATCH",
      json: { use_documents: next }
    });
    if (next) await loadDocuments(activeConversation.id);
  }

  async function loadDocuments(conversationId: string) {
    const documents = await api<DocumentInfo[]>(`/conversations/${conversationId}/documents`);
    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      documents,
      docsLoaded: true,
      docsLoading: false
    }));
  }

  async function uploadDocument(file: File) {
    if (!activeConversation) return;
    setUploadingDoc(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await apiFetch(`/conversations/${activeConversation.id}/documents`, {
        method: "POST",
        token,
        body: form
      });
      const document = (await response.json()) as DocumentInfo;
      updateConversation(activeConversation.id, (conversation) => ({
        ...conversation,
        documents: [...conversation.documents, document],
        docsLoaded: true
      }));
    } finally {
      setUploadingDoc(false);
    }
  }

  async function deleteDocument(documentId: string) {
    if (!activeConversation) return;
    await api<void>(`/conversations/${activeConversation.id}/documents/${documentId}`, {
      method: "DELETE"
    });
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      documents: conversation.documents.filter((document) => document.id !== documentId)
    }));
  }

  async function loadNotes() {
    const rows = await api<Note[]>("/notes");
    setNotes(rows);
    setActiveNoteId((current) => (current && rows.some((note) => note.id === current) ? current : rows[0]?.id || null));
  }

  async function createNote() {
    const note = await api<Note>("/notes", {
      method: "POST",
      json: { title: "Untitled note", content: "", style: { style_name: "default" } }
    });
    setNotes((current) => [note, ...current]);
    setActiveNoteId(note.id);
  }

  async function saveNote(note: Note, patch: Partial<Pick<Note, "title" | "content">>) {
    setNotes((current) => current.map((item) => (item.id === note.id ? { ...item, ...patch } : item)));
    const updated = await api<Note>(`/notes/${note.id}`, { method: "PATCH", json: patch });
    setNotes((current) => current.map((item) => (item.id === note.id ? updated : item)));
  }

  async function deleteNote(id: string) {
    await api<void>(`/notes/${id}`, { method: "DELETE" });
    setNotes((current) => current.filter((note) => note.id !== id));
    setActiveNoteId((current) => (current === id ? notes.find((note) => note.id !== id)?.id || null : current));
  }

  async function regenerateNote() {
    if (!activeNote || !regenPrompt.trim()) return;
    setRegenPending(true);
    try {
      const data = await api<{ content: string }>(`/notes/${activeNote.id}/regenerate`, {
        method: "POST",
        provider: providerForNote(activeNote.id, providers),
        json: { prompt: regenPrompt }
      });
      setNotes((current) =>
        current.map((note) => (note.id === activeNote.id ? { ...note, content: data.content } : note))
      );
      setRegenPrompt("");
    } finally {
      setRegenPending(false);
    }
  }

  async function loadTranslationSessions() {
    const rows = await api<TranslationSession[]>("/translations");
    setTranslationSessions(rows);
    setActiveTranslationId((current) =>
      current && rows.some((session) => session.id === current) ? current : rows[0]?.id || null
    );
  }

  async function selectTranslationSession(id: string) {
    setActiveTranslationId(id);
    setLoadingTranslation(true);
    try {
      const detail = await api<TranslationSessionDetail>(`/translations/${id}`);
      setActiveTranslation({ ...detail, loaded: true });
      setTranslateLanguage(detail.target_language);
    } finally {
      setLoadingTranslation(false);
    }
  }

  async function runTranslation() {
    if (!translateText.trim() || !translateLanguage) return;
    setTranslatePending(true);
    try {
      const data = await api<{
        session_id: string;
        turn_id: string;
        translated_text: string;
        romanized_text: string;
        model: string;
        target_language: string;
      }>("/translations", {
        method: "POST",
        provider: translateProvider || providerForTranslate(providers) || null,
        json: {
          source_text: translateText,
          target_language: translateLanguage,
          session_id: activeTranslation?.id || null
        }
      });
      const turn: TranslationTurn = {
        id: data.turn_id,
        turn_index: activeTranslation?.turns.length || 0,
        source_text: translateText,
        target_language: data.target_language,
        translated_text: data.translated_text,
        romanized_text: data.romanized_text,
        model: data.model,
        created_at: new Date().toISOString()
      };
      const title = activeTranslation?.title || translateText.slice(0, 50) || data.target_language;
      const sessionSummary: TranslationSession = {
        id: data.session_id,
        user_id: user?.id || "",
        title,
        target_language: data.target_language,
        preview: data.translated_text,
        turn_count: (activeTranslation?.turns.length || 0) + 1,
        is_archived: false,
        created_at: activeTranslation?.created_at || turn.created_at,
        updated_at: turn.created_at
      };
      setTranslationSessions((current) => {
        const without = current.filter((session) => session.id !== data.session_id);
        return [sessionSummary, ...without];
      });
      setActiveTranslationId(data.session_id);
      setActiveTranslation((current) =>
        current?.id === data.session_id || !current
          ? {
              id: data.session_id,
              user_id: user?.id || "",
              title,
              target_language: data.target_language,
              is_archived: false,
              created_at: current?.created_at || turn.created_at,
              updated_at: turn.created_at,
              turns: [...(current?.turns || []), turn],
              loaded: true
            }
          : current
      );
      setTranslateText("");
    } finally {
      setTranslatePending(false);
    }
  }

  async function deleteTranslationSession(id: string) {
    await api<void>(`/translations/${id}`, { method: "DELETE" });
    setTranslationSessions((current) => current.filter((session) => session.id !== id));
    if (activeTranslationId === id) {
      setActiveTranslationId(null);
      setActiveTranslation(null);
    }
  }

  function startNewTranslationSession() {
    setActiveTranslationId(null);
    setActiveTranslation(null);
    setTranslateText("");
  }

  useEffect(() => {
    if (!user) return;
    if (tab === "notes" && !notes.length) void loadNotes();
    if (tab === "translate" && !translationSessions.length) void loadTranslationSessions();
    if (tab === "tasks" && !tasks.length) void loadTasks();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, user]);

  useEffect(() => {
    if (tab !== "translate" || !activeTranslationId) return;
    if (activeTranslation?.id === activeTranslationId && activeTranslation.loaded) return;
    void selectTranslationSession(activeTranslationId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTranslationId, tab]);

  useEffect(() => {
    if (!translateMessagesRef.current) return;
    translateMessagesRef.current.scrollTop = translateMessagesRef.current.scrollHeight;
  }, [lastTranslationTurn?.id, translatePending]);

  useEffect(() => {
    if (activeConversation?.use_documents && !activeConversation.docsLoaded && !activeConversation.docsLoading) {
      void loadDocuments(activeConversation.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConversation?.id, activeConversation?.use_documents]);

  const newButtonLabel = tab === "notes" ? "New note" : tab === "translate" ? "New translation" : tab === "tasks" ? "New task" : "New conversation";

  return (
    <div className={styles.app}>
      <aside className={styles.sidebar}>
        <div className={styles.sidebarTop}>
          <div className={styles.brandRow}>
            <div className={styles.brandMark}>›</div>
            <div className={styles.brandText}>
              <div className={styles.brandName}>Cue</div>
              <div className={styles.brandSub}>your AI prompt</div>
            </div>
            <button className={styles.iconButton} onClick={toggleTheme} title="Toggle dark theme">
              {theme === "dark" ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" />
                  <line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" />
                  <line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
              )}
            </button>
          </div>
          <div className={styles.userRow}>
            <span className={styles.userLabel}>{user ? user.email : "Signed out"}</span>
            {user ? (
              <button className={styles.smallButton} onClick={logout}>
                Logout
              </button>
            ) : null}
          </div>
          <div className={styles.tabBar}>
            {(["chats", "notes", "translate", "tasks"] as Tab[]).map((item) => (
              <button
                key={item}
                className={`${styles.tabButton} ${tab === item ? styles.tabButtonActive : ""}`}
                onClick={() => setTab(item)}
              >
                {item === "chats" ? "Chats" : item === "translate" ? "Translate" : item[0].toUpperCase() + item.slice(1)}
              </button>
            ))}
          </div>
          <button
            className={styles.newButton}
            onClick={() => {
              if (tab === "notes") void createNote();
              else if (tab === "translate") startNewTranslationSession();
              else if (tab === "tasks") startNewTask();
              else setNewModalOpen(true);
            }}
          >
            <span>+</span>
            <span>{newButtonLabel}</span>
          </button>
        </div>
        {renderSidebar()}
      </aside>
      <main className={styles.main}>{renderMain()}</main>
      {!user ? <AuthOverlay onSubmit={submitAuth} error={authError} /> : null}
      {newModalOpen ? (
        <NewConversationModal
          tones={tones}
          providers={providers}
          onClose={() => setNewModalOpen(false)}
          onCreate={createConversation}
        />
      ) : null}
    </div>
  );

  function renderSidebar() {
    if (!user) return <div className={styles.sidebarList} />;
    if (tab === "notes") {
      return (
        <div className={styles.sidebarList}>
          {notes.map((note) => (
            <button
              key={note.id}
              className={`${styles.listItem} ${activeNoteId === note.id ? styles.listItemActive : ""}`}
              onClick={() => setActiveNoteId(note.id)}
            >
              <span className={styles.listTitle}>{note.title || "Untitled note"}</span>
              <span className={styles.preview}>{note.content || "Empty note"}</span>
            </button>
          ))}
        </div>
      );
    }
    if (tab === "translate") {
      return (
        <div className={styles.sidebarList}>
          {translationSessions.map((session) => (
            <button
              key={session.id}
              className={`${styles.listItem} ${activeTranslationId === session.id ? styles.listItemActive : ""}`}
              onClick={() => setActiveTranslationId(session.id)}
            >
              <span className={styles.listTitle}>{session.title || session.target_language}</span>
              <span className={styles.preview}>{session.preview}</span>
            </button>
          ))}
        </div>
      );
    }
    if (tab === "tasks") {
      return (
        <div className={styles.sidebarList}>
          {tasks.map((task) => (
            <button
              key={task.id}
              className={`${styles.listItem} ${activeTaskId === task.id ? styles.listItemActive : ""}`}
              onClick={() => void selectTask(task.id)}
            >
              <span className={styles.listTitle}>{task.goal}</span>
              <span className={styles.preview}>{task.status}</span>
            </button>
          ))}
        </div>
      );
    }
    return (
      <div className={styles.sidebarList}>
        {loadingConversations ? <div className={styles.empty}>Loading conversations...</div> : null}
        {conversations.map((conversation, index) => (
          <button
            key={conversation.id}
            className={`${styles.listItem} ${conversation.id === activeId ? styles.listItemActive : ""}`}
            onClick={() => void selectConversation(conversation.id)}
          >
            <div className={styles.listTitleRow}>
              <div className={styles.listTitle}>
                {conversation.kind === "duo" ? "👥 " : ""}
                {conversation.title}
              </div>
              <span
                className={styles.toneDot}
                style={{ background: toneColor(tones, toneLabel(tones, conversation.tone_name)) }}
              />
            </div>
            <div className={styles.preview}>{newestPreview(conversation)}</div>
            <div className={styles.meta}>{timeAgo(index)}</div>
          </button>
        ))}
      </div>
    );
  }

  function renderMain() {
    if (!user) return <div />;
    if (tab === "notes") return renderNotes();
    if (tab === "translate") return renderTranslations();
    if (tab === "tasks") return renderTasks();
    return renderChat();
  }

  function renderChat() {
    if (!activeConversation) return <div className={styles.empty}>Create a conversation to begin.</div>;
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
            {isDuo ? (
              <>
                <span className={styles.label}>Acting as</span>
                <select
                  className={styles.select}
                  value={activeConversation.sendAs || ""}
                  onChange={(event) =>
                    updateConversation(activeConversation.id, (conversation) => ({
                      ...conversation,
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
                  onChange={(event) => void toggleDocuments(event.target.checked)}
                />{" "}
                Docs
              </label>
            )}
            <span className={styles.label}>Tone</span>
            <select
              className={styles.select}
              value={toneLabel(tones, activeConversation.tone_name)}
              onChange={(event) => void changeTone(event.target.value)}
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
            <button
              className={styles.dangerButton}
              title="Delete conversation"
              onClick={() => void deleteConversation(activeConversation.id)}
            >
              🗑
            </button>
          </div>
        </div>
        {activeConversation.use_documents ? (
          <div className={styles.documentsBar}>
            <label className={styles.ghostButton}>
              {uploadingDoc ? "Uploading..." : "📎 Upload document"}
              <input
                hidden
                type="file"
                accept=".pdf,.txt,.md"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void uploadDocument(file);
                }}
              />
            </label>
            {activeConversation.documents.map((document) => (
              <span className={styles.chip} key={document.id}>
                📄 {document.filename}
                <button className={styles.deleteInline} onClick={() => void deleteDocument(document.id)}>
                  x
                </button>
              </span>
            ))}
            {activeConversation.docsLoaded && !activeConversation.documents.length ? (
              <span className={styles.preview}>No documents yet - upload a .pdf, .txt, or .md to ground replies</span>
            ) : null}
          </div>
        ) : null}
        {healthChecked && !llmAvailable && !isDuo ? <div className={styles.statusBar}>connecting...</div> : null}
        <div className={styles.messages} ref={messagesRef}>
          {loadingMessages ? <div className={styles.empty}>Loading conversation...</div> : null}
          {activeConversation.messages.map((message) => (
            <MessageBubble
              key={message.id || (message as StreamMessage).localId}
              message={message}
              mine={isDuo ? message.user_id === activeConversation.sendAs : message.role === "user"}
              isDuo={isDuo}
              onDelete={message.id ? () => void deleteMessage(message.id) : undefined}
            />
          ))}
        </div>
        <div className={styles.composerWrap}>
          <div className={styles.composer}>
            <textarea
              className={styles.draft}
              rows={3}
              placeholder={isDuo ? `Message as ${activeConversation.sendAs || ""}...` : "Message the assistant..."}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
            />
            {isDuo ? (
              <button className={styles.suggestButton} disabled={suggesting || streaming} onClick={() => void suggestReply()}>
                ✨
              </button>
            ) : null}
            <button className={styles.sendButton} disabled={sendDisabled} onClick={() => void sendMessage()}>
              ↑
            </button>
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

  function renderNotes() {
    if (!activeNote) return <div className={styles.empty}>Create a note to begin.</div>;
    return (
      <>
        <div className={styles.header}>
          <div className={styles.headerTitle}>
            <div className={styles.title}>{activeNote.title || "Untitled note"}</div>
            <div className={styles.subtitle}>{activeNote.style_name}</div>
          </div>
          <div className={styles.controls}>
            <select
              className={styles.select}
              value={providerForNote(activeNote.id, providers)}
              onChange={(event) => {
                store(PROVIDER_NOTE_PREFIX + activeNote.id, event.target.value);
                store(PROVIDER_DEFAULT_KEY, event.target.value);
              }}
            >
              {providers.map((provider) => (
                <option key={provider.id} value={provider.id} disabled={provider.available === false}>
                  {providerLabel(provider)}
                </option>
              ))}
            </select>
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
                <div className={`${styles.textarea} ${styles.notePreview}`} dangerouslySetInnerHTML={{ __html: noteHtml }} />
              ) : (
                <textarea
                  className={`${styles.textarea} ${styles.noteEditor}`}
                  value={activeNote.content}
                  disabled={regenPending}
                  onChange={(event) => void saveNote(activeNote, { content: event.target.value })}
                />
              )}
            </div>
            <div className={`${styles.row} ${regenPending ? styles.noteBodyWrapPending : ""}`}>
              <select className={styles.select} defaultValue={activeNote.style_name} disabled={regenPending}>
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
                onClick={() => void regenerateNote()}
              >
                {regenPending ? "Regenerating..." : "Regenerate"}
              </button>
            </div>
          </div>
        </div>
      </>
    );
  }

  function renderTranslations() {
    const composerLocked = translatePending || !llmAvailable;
    const sendDisabled = !translateText.trim() || composerLocked || !translateLanguage;
    const sessionTitle =
      activeTranslation?.title || activeTranslation?.target_language || translateLanguage || "New translation";

    return (
      <>
        <div className={styles.header}>
          <div className={styles.headerTitle}>
            <div className={styles.title}>{sessionTitle}</div>
            <div className={styles.subtitle}>
              {activeTranslation?.turns.length
                ? `${activeTranslation.turns.length} translation${activeTranslation.turns.length === 1 ? "" : "s"}`
                : "Start a translation chat"}
            </div>
          </div>
          <div className={styles.controls}>
            <span className={styles.label}>Language</span>
            <select
              className={styles.select}
              value={translateLanguage}
              onChange={(event) => setTranslateLanguage(event.target.value)}
            >
              {languages.map((language) => (
                <option key={language.id} value={language.label}>
                  {language.label}
                </option>
              ))}
            </select>
            <span className={styles.label}>Model</span>
            <ProviderSelect
              providers={providers}
              value={translateProvider || providerForTranslate(providers)}
              onChange={(value) => {
                store(PROVIDER_TRANSLATE_KEY, value);
                store(PROVIDER_DEFAULT_KEY, value);
                setTranslateProvider(value);
              }}
            />
            {activeTranslation ? (
              <button
                className={styles.dangerButton}
                onClick={() => void deleteTranslationSession(activeTranslation.id)}
              >
                🗑
              </button>
            ) : null}
          </div>
        </div>
        {healthChecked && !llmAvailable ? (
          <div className={styles.statusBar}>LLM unavailable — translation is disabled.</div>
        ) : null}
        <div className={styles.messages} ref={translateMessagesRef}>
          {loadingTranslation ? <div className={styles.empty}>Loading translation...</div> : null}
          {activeTranslation?.turns.map((turn) => (
            <div key={turn.id}>
              <TranslationBubble mine label="You" text={turn.source_text} />
              <TranslationBubble
                label={`Translation · ${turn.target_language}`}
                text={turn.translated_text}
              />
              {turn.romanized_text.trim() ? (
                <TranslationBubble label="Romanized" text={turn.romanized_text} />
              ) : null}
            </div>
          ))}
        </div>
        <div className={styles.composerWrap}>
          <div className={styles.composer}>
            <textarea
              className={styles.draft}
              rows={3}
              placeholder="Paste or type text to translate..."
              value={translateText}
              onChange={(event) => setTranslateText(event.target.value)}
              onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void runTranslation();
                }
              }}
            />
            <button className={styles.sendButton} disabled={sendDisabled} onClick={() => void runTranslation()}>
              ↑
            </button>
          </div>
          <div className={styles.hint}>
            {translatePending
              ? "Translating..."
              : "Enter to translate · Shift+Enter for new line · continue in the same session for context"}
          </div>
        </div>
      </>
    );
  }

  function renderTasks() {
    return (
      <>
        <div className={styles.header}>
          <div className={styles.headerTitle}>
            <div className={styles.title}>Cue Task Console</div>
            <div className={styles.subtitle}>agentic tool-use surface</div>
          </div>
        </div>
        <div className={styles.panel}>
          <div className={styles.formStack}>
            <TaskComposer
              tools={tools}
              selectedTools={selectedTools}
              taskGoal={taskGoal}
              taskFiles={taskFiles}
              maxSteps={maxSteps}
              taskRunning={taskRunning}
              onGoalChange={setTaskGoal}
              onMaxStepsChange={setMaxSteps}
              onRunTask={() => void runTask()}
              onSelectAllTools={selectAllTools}
              onClearTools={clearTools}
              onChooseTools={chooseTools}
              onToggleTool={toggleTool}
              onAddFiles={addTaskFiles}
              onRemoveFile={removeTaskFile}
            />
            <TaskTrace activeTask={activeTask} />
          </div>
        </div>
      </>
    );
  }
}
