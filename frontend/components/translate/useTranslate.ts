"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiOptions } from "@/lib/api";
import type { Language, ProviderInfo, TranslationSession, TranslationSessionDetail, TranslationTurn, User } from "@/lib/types";
import type { TranslationView } from "../CueApp.types";
import {
  PROVIDER_DEFAULT_KEY,
  PROVIDER_TRANSLATE_KEY,
  providerForTranslate,
  store
} from "../CueApp.helpers";

type UseTranslateParams = {
  api: <T>(path: string, options?: ApiOptions) => Promise<T>;
  token: string | null;
  providers: ProviderInfo[];
  user: User | null;
};

export function useTranslate({ api, token, providers, user }: UseTranslateParams) {
  const [translationSessions, setTranslationSessions] = useState<TranslationSession[]>([]);
  const [activeTranslationId, setActiveTranslationId] = useState<string | null>(null);
  const [activeTranslation, setActiveTranslation] = useState<TranslationView | null>(null);
  const [loadingTranslation, setLoadingTranslation] = useState(false);
  const [languages, setLanguages] = useState<Language[]>([]);
  const [translateText, setTranslateText] = useState("");
  const [translateLanguage, setTranslateLanguage] = useState("");
  const [translateProvider, setTranslateProvider] = useState("");
  const [translatePending, setTranslatePending] = useState(false);

  const translateMessagesRef = useRef<HTMLDivElement>(null);

  const lastTranslationTurn = activeTranslation?.turns.at(-1);

  useEffect(() => {
    if (!token) {
      setTranslationSessions([]);
      setActiveTranslationId(null);
      setActiveTranslation(null);
      setLoadingTranslation(false);
      setLanguages([]);
      setTranslateText("");
      setTranslateLanguage("");
      setTranslateProvider("");
      setTranslatePending(false);
    }
  }, [token]);

  useEffect(() => {
    if (providers.length && !translateProvider) {
      setTranslateProvider(providerForTranslate(providers));
    }
  }, [providers, translateProvider]);

  useEffect(() => {
    if (!translateMessagesRef.current) return;
    translateMessagesRef.current.scrollTop = translateMessagesRef.current.scrollHeight;
  }, [lastTranslationTurn?.id, translatePending]);

  const loadLanguages = useCallback(async () => {
    try {
      const rows = await api<Language[]>("/languages");
      setLanguages(rows);
      setTranslateLanguage((current) => current || rows[0]?.label || rows[0]?.id || "");
    } catch {
      // keep empty
    }
  }, [api]);

  const loadTranslationSessions = useCallback(async () => {
    const rows = await api<TranslationSession[]>("/translations");
    setTranslationSessions(rows);
    setActiveTranslationId((current) =>
      current && rows.some((s) => s.id === current) ? current : rows[0]?.id || null
    );
  }, [api]);

  const selectTranslationSession = useCallback(
    async (id: string) => {
      setActiveTranslationId(id);
      setLoadingTranslation(true);
      try {
        const detail = await api<TranslationSessionDetail>(`/translations/${id}`);
        setActiveTranslation({ ...detail, loaded: true });
        setTranslateLanguage(detail.target_language);
      } finally {
        setLoadingTranslation(false);
      }
    },
    [api]
  );

  const runTranslation = useCallback(async () => {
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
        const without = current.filter((s) => s.id !== data.session_id);
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
  }, [activeTranslation, api, providers, translateLanguage, translateProvider, translateText, user]);

  const deleteTranslationSession = useCallback(
    async (id: string) => {
      await api<void>(`/translations/${id}`, { method: "DELETE" });
      setTranslationSessions((current) => current.filter((s) => s.id !== id));
      if (activeTranslationId === id) {
        setActiveTranslationId(null);
        setActiveTranslation(null);
      }
    },
    [activeTranslationId, api]
  );

  const startNewTranslationSession = useCallback(() => {
    setActiveTranslationId(null);
    setActiveTranslation(null);
    setTranslateText("");
  }, []);

  const changeTranslateProvider = useCallback((providerId: string) => {
    store(PROVIDER_TRANSLATE_KEY, providerId);
    store(PROVIDER_DEFAULT_KEY, providerId);
    setTranslateProvider(providerId);
  }, []);

  return {
    translationSessions,
    activeTranslationId,
    activeTranslation,
    loadingTranslation,
    languages,
    translateText,
    translateLanguage,
    translateProvider,
    translatePending,
    translateMessagesRef,
    setActiveTranslationId,
    setTranslateText,
    setTranslateLanguage,
    loadLanguages,
    loadTranslationSessions,
    selectTranslationSession,
    runTranslation,
    deleteTranslationSession,
    startNewTranslationSession,
    changeTranslateProvider
  };
}
