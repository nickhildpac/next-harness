"use client";

import { KeyboardEvent, RefObject } from "react";
import type { Language, ProviderInfo, TranslationSession } from "@/lib/types";
import type { TranslationView } from "../CueApp.types";
import { ProviderSelect, TranslationBubble } from "../CueApp.widgets";
import { providerForTranslate } from "../CueApp.helpers";
import styles from "../CueApp.module.css";

type TranslatePanelProps = {
  activeTranslation: TranslationView | null;
  languages: Language[];
  translateText: string;
  translateLanguage: string;
  translateProvider: string;
  translatePending: boolean;
  loadingTranslation: boolean;
  translateMessagesRef: RefObject<HTMLDivElement>;
  providers: ProviderInfo[];
  llmAvailable: boolean;
  healthChecked: boolean;
  setTranslateText: (text: string) => void;
  setTranslateLanguage: (lang: string) => void;
  runTranslation: () => void;
  deleteTranslationSession: (id: string) => void;
  changeTranslateProvider: (providerId: string) => void;
};

export function TranslatePanel({
  activeTranslation,
  languages,
  translateText,
  translateLanguage,
  translateProvider,
  translatePending,
  loadingTranslation,
  translateMessagesRef,
  providers,
  llmAvailable,
  healthChecked,
  setTranslateText,
  setTranslateLanguage,
  runTranslation,
  deleteTranslationSession,
  changeTranslateProvider
}: TranslatePanelProps) {
  const composerLocked = translatePending || !llmAvailable;
  const sendDisabled = !translateText.trim() || composerLocked || !translateLanguage;
  const sessionTitle =
    activeTranslation?.title ||
    activeTranslation?.target_language ||
    translateLanguage ||
    "New translation";

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
        {activeTranslation ? (
          <div className={styles.controls}>
            <button
              className={styles.dangerButton}
              onClick={() => void deleteTranslationSession(activeTranslation.id)}
            >
              🗑
            </button>
          </div>
        ) : null}
      </div>
      {healthChecked && !llmAvailable ? (
        <div className={styles.statusBar}>LLM unavailable — translation is disabled.</div>
      ) : null}
      <div className={styles.messages} ref={translateMessagesRef}>
        {loadingTranslation ? (
          <div className={styles.empty}>Loading translation...</div>
        ) : null}
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
          <div className={styles.composerActions}>
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
              onChange={changeTranslateProvider}
            />
            <div className={styles.composerActionsSpacer} />
            <button
              className={styles.sendButton}
              disabled={sendDisabled}
              onClick={() => void runTranslation()}
            >
              ↑
            </button>
          </div>
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
