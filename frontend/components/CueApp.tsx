"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { ApiError, apiJson } from "@/lib/api";
import type { ApiOptions } from "@/lib/api";
import type {
  NoteStyle,
  ProviderInfo,
  ToneInfo,
  TokenResponse,
  User
} from "@/lib/types";
import {
  AUTH_TOKEN_KEY,
  THEME_KEY,
  fallbackProviders,
  fallbackTones,
  newestPreview,
  providerForTranslate,
  store,
  stored,
  timeAgo,
  toneColor,
  toneLabel
} from "./CueApp.helpers";
import type { Tab } from "./CueApp.types";
import { AuthOverlay, NewConversationModal } from "./CueApp.widgets";
import { useChat } from "./chat/useChat";
import { ChatPanel } from "./chat/ChatPanel";
import { useNotes } from "./notes/useNotes";
import { NotesPanel } from "./notes/NotesPanel";
import { useTranslate } from "./translate/useTranslate";
import { TranslatePanel } from "./translate/TranslatePanel";
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

  const handleAuthFailure = useCallback(() => {
    store(AUTH_TOKEN_KEY, null);
    setToken(null);
    setUser(null);
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

  const chat = useChat({ api, token, providers, tones, user, onAuthFailure: handleAuthFailure });
  const notes = useNotes({ api, token, providers });
  const translate = useTranslate({ api, token, providers, user });
  const tasks = useTasks({ api, token });

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
      apiJson<ProviderInfo[]>("/providers")
        .then((rows) => setProviders(rows))
        .catch(() => undefined),
      apiJson<NoteStyle[]>("/note-styles").then(notes.loadNoteStyles).catch(() => undefined),
      translate.loadLanguages(),
      apiJson<{ available: boolean }>("/health/llm")
        .then((data) => {
          setLlmAvailable(data.available !== false);
          setHealthChecked(true);
        })
        .catch(() => {
          setLlmAvailable(false);
          setHealthChecked(true);
        })
    ]).catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    if (!token) return;
    apiJson<User>("/auth/me", { token })
      .then((currentUser) => {
        setUser(currentUser);
        void chat.loadConversations();
      })
      .catch((error) => {
        if (error instanceof ApiError && error.status === 401) handleAuthFailure();
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handleAuthFailure, token]);

  useEffect(() => {
    if (!user) return;
    if (tab === "notes" && !notes.notes.length) void notes.loadNotes();
    if (tab === "translate" && !translate.translationSessions.length) void translate.loadTranslationSessions();
    if (tab === "tasks" && !tasks.threads.length) void tasks.loadThreads();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, user]);

  useEffect(() => {
    if (tab !== "translate" || !translate.activeTranslationId) return;
    if (translate.activeTranslation?.id === translate.activeTranslationId && translate.activeTranslation.loaded) return;
    void translate.selectTranslationSession(translate.activeTranslationId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [translate.activeTranslationId, tab]);

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    store(THEME_KEY, next);
    document.documentElement.setAttribute("data-theme", next);
  }

  function logout() {
    handleAuthFailure();
  }

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

  const newButtonLabel =
    tab === "notes"
      ? "New note"
      : tab === "translate"
        ? "New translation"
        : tab === "tasks"
          ? "New thread"
          : "New conversation";

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
                {item === "chats"
                  ? "Chats"
                  : item === "translate"
                    ? "Translate"
                    : item === "tasks"
                      ? "Agent"
                      : item[0].toUpperCase() + item.slice(1)}
              </button>
            ))}
          </div>
          <button
            className={styles.newButton}
            onClick={() => {
              if (tab === "notes") void notes.createNote();
              else if (tab === "translate") translate.startNewTranslationSession();
              else if (tab === "tasks") tasks.startNewThread();
              else chat.setNewModalOpen(true);
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
      {chat.newModalOpen ? (
        <NewConversationModal
          tones={tones}
          providers={providers}
          onClose={() => chat.setNewModalOpen(false)}
          onCreate={chat.createConversation}
        />
      ) : null}
    </div>
  );

  function renderSidebar() {
    if (!user) return <div className={styles.sidebarList} />;
    if (tab === "notes") {
      return (
        <div className={styles.sidebarList}>
          {notes.notes.map((note) => (
            <button
              key={note.id}
              className={`${styles.listItem} ${notes.activeNoteId === note.id ? styles.listItemActive : ""}`}
              onClick={() => notes.setActiveNoteId(note.id)}
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
          {translate.translationSessions.map((session) => (
            <button
              key={session.id}
              className={`${styles.listItem} ${translate.activeTranslationId === session.id ? styles.listItemActive : ""}`}
              onClick={() => translate.setActiveTranslationId(session.id)}
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
          {tasks.threads.map((thread) => {
            const latestTask = thread.tasks.at(-1);
            return (
            <div key={thread.id} className={styles.listItemRow}>
              <button
                className={`${styles.listItem} ${tasks.activeThreadId === thread.id ? styles.listItemActive : ""}`}
                onClick={() => void tasks.selectThread(thread.id)}
              >
                <span className={styles.listTitle}>{thread.title || "Untitled thread"}</span>
                <span className={styles.preview}>
                  {latestTask?.result_summary || latestTask?.status || "No tasks yet"}
                </span>
              </button>
              <button
                className={styles.listItemDelete}
                title="Delete thread"
                aria-label="Delete thread"
                onClick={() => void tasks.deleteThread(thread.id)}
              >
                🗑
              </button>
            </div>
            );
          })}
        </div>
      );
    }
    return (
      <div className={styles.sidebarList}>
        {chat.loadingConversations ? (
          <div className={styles.empty}>Loading conversations...</div>
        ) : null}
        {chat.conversations.map((conversation, index) => (
          <button
            key={conversation.id}
            className={`${styles.listItem} ${conversation.id === chat.activeId ? styles.listItemActive : ""}`}
            onClick={() => void chat.selectConversation(conversation.id)}
          >
            <div className={styles.listTitleRow}>
              <div className={styles.listTitle}>
                {conversation.kind === "duo" ? "👥 " : ""}
                {conversation.title}
              </div>
              <span
                className={styles.toneDot}
                style={{
                  background: toneColor(tones, toneLabel(tones, conversation.tone_name))
                }}
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
    if (tab === "notes") {
      return (
        <NotesPanel
          activeNote={notes.activeNote}
          noteStyles={notes.noteStyles}
          notePreview={notes.notePreview}
          regenPrompt={notes.regenPrompt}
          regenPending={notes.regenPending}
          noteHtml={notes.noteHtml}
          noteEditorRef={notes.noteEditorRef}
          providers={providers}
          setNotePreview={notes.setNotePreview}
          setRegenPrompt={notes.setRegenPrompt}
          saveNote={notes.saveNote}
          deleteNote={notes.deleteNote}
          regenerateNote={notes.regenerateNote}
          changeNoteProvider={notes.changeNoteProvider}
        />
      );
    }
    if (tab === "translate") {
      return (
        <TranslatePanel
          activeTranslation={translate.activeTranslation}
          languages={translate.languages}
          translateText={translate.translateText}
          translateLanguage={translate.translateLanguage}
          translateProvider={translate.translateProvider}
          translatePending={translate.translatePending}
          loadingTranslation={translate.loadingTranslation}
          translateMessagesRef={translate.translateMessagesRef}
          providers={providers}
          llmAvailable={llmAvailable}
          healthChecked={healthChecked}
          setTranslateText={translate.setTranslateText}
          setTranslateLanguage={translate.setTranslateLanguage}
          runTranslation={translate.runTranslation}
          deleteTranslationSession={translate.deleteTranslationSession}
          changeTranslateProvider={translate.changeTranslateProvider}
        />
      );
    }
    if (tab === "tasks") {
      return (
        <>
          <div className={styles.header}>
            <div className={styles.headerTitle}>
              <div className={styles.title}>Cue Agent Console</div>
              <div className={styles.subtitle}>agentic tool-use surface</div>
            </div>
          </div>
          <div className={styles.taskPanel}>
            <div className={styles.taskTraceScroll}>
              <TaskTrace activeThread={tasks.activeThread} />
            </div>
            <div className={styles.taskComposerBar}>
              {tasks.taskError ? (
                <div className={styles.statusBar}>{tasks.taskError}</div>
              ) : null}
              <TaskComposer
                tools={tasks.tools}
                selectedTools={tasks.selectedTools}
                taskGoal={tasks.taskGoal}
                taskFiles={tasks.taskFiles}
                maxSteps={tasks.maxSteps}
                taskRunning={tasks.taskRunning}
                onGoalChange={tasks.setTaskGoal}
                onMaxStepsChange={tasks.setMaxSteps}
                onRunTask={() => void tasks.runTask()}
                onSelectAllTools={tasks.selectAllTools}
                onClearTools={tasks.clearTools}
                onChooseTools={tasks.chooseTools}
                onToggleTool={tasks.toggleTool}
                onAddFiles={tasks.addTaskFiles}
                onRemoveFile={tasks.removeTaskFile}
              />
            </div>
          </div>
        </>
      );
    }
    return (
      <ChatPanel
        activeConversation={chat.activeConversation}
        draft={chat.draft}
        streaming={chat.streaming}
        suggesting={chat.suggesting}
        uploadingDoc={chat.uploadingDoc}
        loadingMessages={chat.loadingMessages}
        messagesRef={chat.messagesRef}
        tones={tones}
        providers={providers}
        llmAvailable={llmAvailable}
        healthChecked={healthChecked}
        user={user}
        setDraft={chat.setDraft}
        sendMessage={chat.sendMessage}
        suggestReply={chat.suggestReply}
        deleteMessage={chat.deleteMessage}
        deleteConversation={chat.deleteConversation}
        toggleDocuments={chat.toggleDocuments}
        uploadDocument={chat.uploadDocument}
        deleteDocument={chat.deleteDocument}
        changeTone={chat.changeTone}
        updateConversation={chat.updateConversation}
      />
    );
  }
}
