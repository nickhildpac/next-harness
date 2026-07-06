export type ToneId =
  | "friendly"
  | "professional"
  | "humorous"
  | "empathetic"
  | "concise"
  | "technical"
  | "custom";

export type ToneInfo = {
  id: ToneId;
  label: string;
  color_key?: string;
};

export type ProviderInfo = {
  id: string;
  label: string;
  available: boolean;
  model?: string;
};

export type User = {
  id: string;
  email: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
};

export type TokenResponse = {
  access_token: string;
  token_type: string;
  expires_at: string;
  user: User;
};

export type Citation = {
  marker: number;
  document_id: string;
  filename: string;
  page: number | null;
  chunk_index: number;
  score: number;
  snippet: string;
};

export type Conversation = {
  id: string;
  user_id: string;
  second_user_id: string | null;
  participant_user_id: string | null;
  participant_second_user_id: string | null;
  kind: "assistant" | "duo";
  title: string | null;
  tone_name: ToneId;
  custom_persona: string | null;
  use_documents: boolean;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
};

export type Message = {
  id: string;
  conversation_id: string;
  user_id: string;
  role: "user" | "assistant" | "system" | "summary";
  content: string;
  token_count: number;
  model: string | null;
  created_at: string;
  citations: Citation[] | null;
};

export type ConversationDetail = Conversation & {
  messages: Message[];
  summary: string | null;
};

export type PaginatedMessages = {
  items: Message[];
  limit: number;
  offset: number;
  total: number;
};

export type DocumentInfo = {
  id: string;
  conversation_id: string | null;
  task_id: string | null;
  filename: string;
  content_type: string | null;
  size_bytes: number;
  chunk_count: number;
  created_at: string;
  updated_at: string;
};

export type Note = {
  id: string;
  user_id: string;
  title: string | null;
  content: string;
  style_name: string;
  custom_instructions: string | null;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
};

export type NoteStyle = {
  id: string;
  label: string;
};

export type TranslationTurn = {
  id: string;
  turn_index: number;
  source_text: string;
  target_language: string;
  translated_text: string;
  romanized_text: string;
  model: string | null;
  created_at: string;
};

export type TranslationSession = {
  id: string;
  user_id: string;
  title: string | null;
  target_language: string;
  preview: string;
  turn_count: number;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
};

export type TranslationSessionDetail = Omit<TranslationSession, "preview" | "turn_count"> & {
  turns: TranslationTurn[];
};

export type Language = {
  id: string;
  label: string;
};

export type Task = {
  id: string;
  user_id: string;
  goal: string;
  status: string;
  max_steps: number;
  steps_taken: number;
  model: string | null;
  result_summary: string | null;
  error: string | null;
  allowed_tools: string[] | null;
  created_at: string;
  updated_at: string;
};

export type TaskStep = {
  id: string;
  step_index: number;
  kind: string;
  tool_name: string | null;
  content: string | null;
  payload: unknown;
  ok: boolean | null;
  created_at: string;
};

export type TaskDetail = Task & {
  steps: TaskStep[];
};

export type ToolInfo = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
};
