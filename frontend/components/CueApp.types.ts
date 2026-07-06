import type {
  Conversation,
  DocumentInfo,
  Message,
  TranslationSessionDetail
} from "@/lib/types";

export type TranslationView = TranslationSessionDetail & { loaded: boolean };

export type Tab = "chats" | "notes" | "translate" | "tasks";

export type ConversationView = Conversation & {
  messages: Message[];
  loaded: boolean;
  documents: DocumentInfo[];
  docsLoaded: boolean;
  docsLoading: boolean;
  sendAs: string | null;
};

export type StreamMessage = Partial<Message> & {
  localId: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  error?: boolean;
};
