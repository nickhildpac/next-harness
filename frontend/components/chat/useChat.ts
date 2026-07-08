"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, readSse, responseErrorMessage } from "@/lib/api";
import type { ApiOptions } from "@/lib/api";
import type { Conversation, DocumentInfo, Message, PaginatedMessages, ProviderInfo, ToneId, ToneInfo, User } from "@/lib/types";
import type { ConversationView, StreamMessage } from "../CueApp.types";
import {
  PROVIDER_CONV_PREFIX,
  PROVIDER_DEFAULT_KEY,
  normalizeConversation,
  providerForConversation,
  store,
  toneId
} from "../CueApp.helpers";

type UseChatParams = {
  api: <T>(path: string, options?: ApiOptions) => Promise<T>;
  token: string | null;
  providers: ProviderInfo[];
  tones: ToneInfo[];
  user: User | null;
  onAuthFailure: () => void;
};

export function useChat({ api, token, providers, tones, user, onAuthFailure }: UseChatParams) {
  const [conversations, setConversations] = useState<ConversationView[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [uploadingDoc, setUploadingDoc] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [newModalOpen, setNewModalOpen] = useState(false);
  const [lastSendText, setLastSendText] = useState("");

  const messagesRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  const activeConversation = conversations.find((c) => c.id === activeId) || null;
  const lastMessageContent = activeConversation?.messages.at(-1)?.content;

  useEffect(() => {
    if (!token) {
      if (streamAbortRef.current) streamAbortRef.current.abort();
      setConversations([]);
      setActiveId(null);
      setDraft("");
      setStreaming(false);
      setSuggesting(false);
      setUploadingDoc(false);
      setLoadingConversations(false);
      setLoadingMessages(false);
      setNewModalOpen(false);
      setLastSendText("");
    }
  }, [token]);

  const updateConversation = useCallback(
    (id: string, updater: (c: ConversationView) => ConversationView) => {
      setConversations((current) => current.map((c) => (c.id === id ? updater(c) : c)));
    },
    []
  );

  const pushMessage = useCallback((id: string, message: Message | StreamMessage) => {
    setConversations((current) =>
      current.map((c) => (c.id === id ? { ...c, messages: [...c.messages, message as Message] } : c))
    );
  }, []);

  const loadMessages = useCallback(
    async (id: string) => {
      setLoadingMessages(true);
      try {
        const data = await api<PaginatedMessages>(
          `/conversations/${encodeURIComponent(id)}/messages?limit=200&offset=0`
        );
        setConversations((current) =>
          current.map((c) => (c.id === id ? { ...c, messages: data.items, loaded: true } : c))
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

  const selectConversation = useCallback(
    async (id: string) => {
      if (streamAbortRef.current) streamAbortRef.current.abort();
      setActiveId(id);
      setDraft("");
      await loadMessages(id);
    },
    [loadMessages]
  );

  const createConversation = useCallback(
    async (payload: {
      title?: string;
      tone_name: string;
      custom_persona?: string;
      participants?: string[];
      provider?: string;
    }) => {
      const conversation = await api<Conversation>("/conversations", {
        method: "POST",
        json: {
          title: payload.title || (payload.participants ? payload.participants.join(" & ") : "New conversation"),
          tone: { tone_name: payload.tone_name, custom_persona: payload.custom_persona || undefined },
          participants: payload.participants
        }
      });
      const mapped = normalizeConversation(conversation);
      store(PROVIDER_CONV_PREFIX + mapped.id, payload.provider || providers[0]?.id || null);
      store(PROVIDER_DEFAULT_KEY, payload.provider || providers[0]?.id || null);
      setConversations((current) => [mapped, ...current]);
      setActiveId(mapped.id);
      setNewModalOpen(false);
    },
    [api, providers]
  );

  const changeTone = useCallback(
    async (label: string, conversationId: string, currentToneName: ToneId | undefined) => {
      updateConversation(conversationId, (c) => ({ ...c, tone_name: toneId(tones, label) }));
      try {
        const data = await api<Conversation>(`/conversations/${conversationId}/tone`, {
          method: "PATCH",
          json: { tone: toneId(tones, label) }
        });
        updateConversation(conversationId, (c) => ({ ...c, tone_name: data.tone_name }));
      } catch {
        if (currentToneName !== undefined) {
          updateConversation(conversationId, (c) => ({ ...c, tone_name: currentToneName }));
        }
      }
    },
    [api, tones, updateConversation]
  );

  const deleteConversation = useCallback(
    async (id: string) => {
      if (!window.confirm("Delete this conversation? This cannot be undone.")) return;
      await api<void>(`/conversations/${id}`, { method: "DELETE" });
      setConversations((current) => {
        const remaining = current.filter((c) => c.id !== id);
        setActiveId((cur) => (cur === id ? remaining[0]?.id || null : cur));
        return remaining;
      });
    },
    [api]
  );

  const deleteMessage = useCallback(
    async (conversationId: string, messageId: string) => {
      await api<void>(`/conversations/${conversationId}/messages/${messageId}`, { method: "DELETE" });
      updateConversation(conversationId, (c) => ({
        ...c,
        messages: c.messages.filter((m) => m.id !== messageId)
      }));
    },
    [api, updateConversation]
  );

  const sendDuoMessage = useCallback(
    async (conversationId: string, sendAs: string, text: string) => {
      setDraft("");
      setStreaming(true);
      try {
        const response = await api<{ user_message: Message }>(
          `/conversations/${conversationId}/messages`,
          { method: "POST", json: { user_id: sendAs, content: text } }
        );
        pushMessage(conversationId, response.user_message);
      } finally {
        setStreaming(false);
      }
    },
    [api, pushMessage]
  );

  const sendMessage = useCallback(
    async (textOverride?: string) => {
      const text = (textOverride || draft).trim();
      if (!activeConversation || !text || streaming || suggesting) return;

      if (activeConversation.kind === "duo") {
        if (activeConversation.sendAs) {
          return sendDuoMessage(activeConversation.id, activeConversation.sendAs, text);
        }
        return;
      }

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
        const response = await fetch(
          `/api/backend/conversations/${activeConversation.id}/messages?stream=true`,
          {
            method: "POST",
            headers,
            body: JSON.stringify({ text, tone_override: { tone: activeConversation.tone_name } }),
            signal: abort.signal
          }
        );
        if (response.status === 401) onAuthFailure();
        if (!response.ok) throw new Error(await responseErrorMessage(response));
        if (!response.body) throw new Error(`${response.status} ${response.statusText}`);
        await readSse(response.body, (event) => {
          if (event.data === "[DONE]") return;
          const payload = JSON.parse(event.data) as Record<string, unknown>;
          if (event.event === "error") throw new Error(String(payload.error || "Stream failed"));
          updateConversation(activeConversation.id, (c) => ({
            ...c,
            messages: c.messages.map((m) => {
              if ((m as StreamMessage).localId === userLocal.localId && payload.user_message_id) {
                return { ...m, id: String(payload.user_message_id) };
              }
              if ((m as StreamMessage).localId !== assistantLocal.localId) return m;
              if (event.event === "delta") {
                return { ...m, content: `${m.content}${String(payload.delta || payload.content || "")}` };
              }
              if (event.event === "citations") {
                return { ...m, citations: payload.citations as Message["citations"] };
              }
              if (event.event === "done") {
                return {
                  ...m,
                  id: String(payload.assistant_message_id || m.id || ""),
                  streaming: false,
                  token_count: Number(payload.output_tokens || 0)
                };
              }
              return m;
            })
          }));
        });
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          setDraft(text);
          updateConversation(activeConversation.id, (c) => ({
            ...c,
            messages: c.messages
              .filter((m) => (m as StreamMessage).localId !== userLocal.localId)
              .map((m) =>
                (m as StreamMessage).localId === assistantLocal.localId
                  ? { ...m, content: `Send failed: ${(error as Error).message}`, streaming: false, error: true }
                  : m
              )
          }));
        }
      } finally {
        setStreaming(false);
        streamAbortRef.current = null;
      }
    },
    [activeConversation, draft, onAuthFailure, providers, pushMessage, sendDuoMessage, streaming, suggesting, token, updateConversation, user]
  );

  const suggestReply = useCallback(
    async (conversationId: string, sendAs: string) => {
      if (suggesting) return;
      setSuggesting(true);
      try {
        const response = await api<{ message: Message | null; content: string }>(
          `/conversations/${conversationId}/suggest`,
          {
            method: "POST",
            provider: providerForConversation(conversationId, providers),
            json: { for_user: sendAs, persist: true }
          }
        );
        if (response.message) pushMessage(conversationId, response.message);
        else setDraft(response.content);
      } finally {
        setSuggesting(false);
      }
    },
    [api, providers, pushMessage, suggesting]
  );

  const loadDocuments = useCallback(
    async (conversationId: string) => {
      const documents = await api<DocumentInfo[]>(`/conversations/${conversationId}/documents`);
      updateConversation(conversationId, (c) => ({ ...c, documents, docsLoaded: true, docsLoading: false }));
    },
    [api, updateConversation]
  );

  const toggleDocuments = useCallback(
    async (conversationId: string, next: boolean) => {
      updateConversation(conversationId, (c) => ({ ...c, use_documents: next }));
      await api<Conversation>(`/conversations/${conversationId}/rag`, {
        method: "PATCH",
        json: { use_documents: next }
      });
      if (next) await loadDocuments(conversationId);
    },
    [api, loadDocuments, updateConversation]
  );

  const uploadDocument = useCallback(
    async (conversationId: string, file: File) => {
      setUploadingDoc(true);
      try {
        const form = new FormData();
        form.append("file", file);
        const response = await apiFetch(`/conversations/${conversationId}/documents`, {
          method: "POST",
          token,
          body: form
        });
        const document = (await response.json()) as DocumentInfo;
        updateConversation(conversationId, (c) => ({
          ...c,
          documents: [...c.documents, document],
          docsLoaded: true
        }));
      } finally {
        setUploadingDoc(false);
      }
    },
    [token, updateConversation]
  );

  const deleteDocument = useCallback(
    async (conversationId: string, documentId: string) => {
      await api<void>(`/conversations/${conversationId}/documents/${documentId}`, { method: "DELETE" });
      updateConversation(conversationId, (c) => ({
        ...c,
        documents: c.documents.filter((d) => d.id !== documentId)
      }));
    },
    [api, updateConversation]
  );

  useEffect(() => {
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight });
  }, [activeConversation?.messages.length, lastMessageContent]);

  useEffect(() => {
    if (
      activeConversation?.use_documents &&
      !activeConversation.docsLoaded &&
      !activeConversation.docsLoading
    ) {
      void loadDocuments(activeConversation.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConversation?.id, activeConversation?.use_documents]);

  return {
    conversations,
    activeId,
    activeConversation,
    draft,
    streaming,
    suggesting,
    uploadingDoc,
    loadingConversations,
    loadingMessages,
    newModalOpen,
    lastSendText,
    messagesRef,
    setDraft,
    setNewModalOpen,
    loadConversations,
    selectConversation,
    createConversation,
    changeTone,
    deleteConversation,
    deleteMessage,
    sendMessage,
    sendDuoMessage,
    suggestReply,
    toggleDocuments,
    loadDocuments,
    uploadDocument,
    deleteDocument,
    updateConversation
  };
}
