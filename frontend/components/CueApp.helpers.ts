import type { Conversation, ProviderInfo, TaskStep, ToneId, ToneInfo } from "@/lib/types";
import type { ConversationView } from "./CueApp.types";

export const AUTH_TOKEN_KEY = "cue-auth-token";
export const THEME_KEY = "cue-theme";
export const PROVIDER_DEFAULT_KEY = "cue-provider-default";
export const PROVIDER_CONV_PREFIX = "cue-provider-";
export const PROVIDER_NOTE_PREFIX = "cue-provider-note-";
export const PROVIDER_TRANSLATE_KEY = "cue-provider-translate";

export const fallbackTones: ToneInfo[] = [
  { id: "friendly", label: "Friendly", color_key: "Friendly" },
  { id: "professional", label: "Formal", color_key: "Formal" },
  { id: "humorous", label: "Playful", color_key: "Playful" },
  { id: "empathetic", label: "Empathetic", color_key: "Empathetic" },
  { id: "concise", label: "Direct", color_key: "Direct" }
];

const toneColors: Record<string, string> = {
  Friendly: "oklch(0.72 0.15 45)",
  Formal: "oklch(0.65 0.1 250)",
  Playful: "oklch(0.75 0.15 340)",
  Empathetic: "oklch(0.7 0.12 150)",
  Direct: "oklch(0.6 0.14 30)",
  Technical: "oklch(0.58 0.11 250)",
  Humorous: "oklch(0.75 0.15 340)"
};

export const fallbackProviders: ProviderInfo[] = [
  { id: "openrouter", label: "OpenRouter", available: true, model: "" },
  { id: "openai", label: "OpenAI", available: false, model: "" },
  { id: "anthropic", label: "Anthropic", available: false, model: "" },
  { id: "gemini", label: "Gemini", available: false, model: "" },
  { id: "ollama", label: "Ollama (local)", available: true, model: "" }
];

export function stored(key: string) {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function store(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // Ignore private browsing storage failures.
  }
}

export function toneLabel(tones: ToneInfo[], id?: string | null) {
  return tones.find((tone) => tone.id === id)?.label || "Friendly";
}

export function toneId(tones: ToneInfo[], label?: string | null): ToneId {
  return (tones.find((tone) => tone.label === label)?.id || "friendly") as ToneId;
}

export function toneColor(tones: ToneInfo[], labelOrId: string) {
  const tone =
    tones.find((item) => item.label === labelOrId) || tones.find((item) => item.id === labelOrId);
  return toneColors[tone?.color_key || tone?.label || labelOrId] || toneColors.Friendly;
}

export function participants(conversation: Conversation) {
  if (conversation.kind !== "duo") return [];
  return [
    conversation.participant_user_id || conversation.user_id,
    conversation.participant_second_user_id || conversation.second_user_id
  ].filter(Boolean) as string[];
}

export function normalizeConversation(conversation: Conversation): ConversationView {
  const people = participants(conversation);
  return {
    ...conversation,
    title: conversation.title || (people.length ? people.join(" & ") : "New conversation"),
    messages: [],
    loaded: false,
    documents: [],
    docsLoaded: false,
    docsLoading: false,
    sendAs: people[0] || null
  };
}

export function newestPreview(conversation: ConversationView) {
  return conversation.messages.at(-1)?.content || "No messages yet";
}

export function timeAgo(index: number) {
  return ["2m ago", "1h ago", "Yesterday", "3d ago"][index] || "Earlier";
}

export function defaultProvider(providers: ProviderInfo[]) {
  const saved = stored(PROVIDER_DEFAULT_KEY);
  return (
    providers.find((provider) => provider.id === saved && provider.available !== false) ||
    providers.find((provider) => provider.available !== false) ||
    providers[0]
  );
}

export function providerForConversation(id: string, providers: ProviderInfo[]) {
  return stored(PROVIDER_CONV_PREFIX + id) || defaultProvider(providers)?.id || "";
}

export function providerForNote(id: string | null, providers: ProviderInfo[]) {
  return (id && stored(PROVIDER_NOTE_PREFIX + id)) || defaultProvider(providers)?.id || "";
}

export function providerForTranslate(providers: ProviderInfo[]) {
  return stored(PROVIDER_TRANSLATE_KEY) || defaultProvider(providers)?.id || "";
}

export function providerLabel(provider: ProviderInfo) {
  return `${provider.model ? `${provider.label} (${provider.model})` : provider.label}${
    provider.available === false ? " - no key" : ""
  }`;
}

export const taskToolPresets = [
  {
    id: "notes-summary",
    label: "Notes summary",
    description: "List notes, read note bodies, then create a summary note.",
    tools: ["list_notes", "get_note", "create_note"]
  },
  {
    id: "rag",
    label: "Task RAG",
    description: "Ingest, list, and search documents attached to this task run.",
    tools: ["ingest_task_document", "list_task_documents", "search_task_documents"]
  },
  {
    id: "translate",
    label: "Translation",
    description: "Translate text and inspect saved translation sessions.",
    tools: ["translate_text", "list_translations"]
  }
] as const;

export function requiredToolsForTaskGoal(goal: string) {
  const text = goal.toLowerCase();
  const mentionsDocuments = /\b(documents?|files?|pdfs?|uploads?|uploaded)\b/.test(text);
  if (mentionsDocuments) return ["list_task_documents", "search_task_documents"];

  const mentionsNotes = /\bnotes?\b/.test(text);
  if (!mentionsNotes) return [];

  const asksToSummarize = /\b(summarize|summarise|summary|recap)\b/.test(text);
  const asksToCreate = /\b(create|save|write|persist|new)\b/.test(text);
  if (asksToSummarize && asksToCreate) return ["list_notes", "get_note", "create_note"];

  const asksForList = /\b(list|show|display|recent|latest|last)\b/.test(text);
  if (asksForList) return ["list_notes"];

  const asksToRead = /\b(read|get|open|inspect|content|body)\b/.test(text);
  if (asksToRead) return ["list_notes", "get_note"];

  return [];
}

export function taskStepLabel(step: TaskStep) {
  if (step.kind === "thought") return "Reason";
  if (step.kind === "tool_call") return "Tool call";
  if (step.kind === "tool_result") return "Tool result";
  if (step.kind === "final") return "Final answer";
  if (step.kind === "error") return "Error";
  return step.kind;
}

export function taskStepDetail(step: TaskStep) {
  if (step.kind === "tool_call" && step.tool_name) return `Calling ${step.tool_name}`;
  if (step.kind === "tool_result" && step.tool_name) {
    return step.ok === false ? `${step.tool_name} failed` : `${step.tool_name} returned`;
  }
  return step.tool_name || "";
}

export function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const markdownAllowedTags = new Set([
  "A",
  "BLOCKQUOTE",
  "BR",
  "CODE",
  "DEL",
  "EM",
  "H1",
  "H2",
  "H3",
  "H4",
  "H5",
  "H6",
  "HR",
  "LI",
  "OL",
  "P",
  "PRE",
  "STRONG",
  "TABLE",
  "TBODY",
  "TD",
  "TH",
  "THEAD",
  "TR",
  "UL"
]);

const safeLinkProtocols = new Set(["http:", "https:", "mailto:", "tel:"]);

export function sanitizeMarkdownHtml(html: string) {
  if (typeof document === "undefined") return "";
  const template = document.createElement("template");
  template.innerHTML = html;

  for (const element of Array.from(template.content.querySelectorAll("*"))) {
    if (!markdownAllowedTags.has(element.tagName)) {
      element.replaceWith(document.createTextNode(element.textContent || ""));
      continue;
    }

    const href = element.tagName === "A" ? element.getAttribute("href") : null;

    for (const attribute of Array.from(element.attributes)) {
      element.removeAttribute(attribute.name);
    }

    if (element.tagName === "A" && href && isSafeMarkdownHref(href)) {
      element.setAttribute("href", href);
      element.setAttribute("rel", "noreferrer noopener");
    }
  }

  return template.innerHTML;
}

function isSafeMarkdownHref(href: string) {
  try {
    const url = new URL(href, window.location.href);
    return safeLinkProtocols.has(url.protocol);
  } catch {
    return false;
  }
}

export function escapeHtml(value: string) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
