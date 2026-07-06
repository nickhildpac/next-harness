const API_PREFIX = "/api/backend";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
  }
}

export type ApiOptions = RequestInit & {
  token?: string | null;
  provider?: string | null;
  json?: unknown;
};

export function authHeaders(token?: string | null) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function apiFetch(path: string, options: ApiOptions = {}) {
  const headers = new Headers(options.headers);
  if (options.token) headers.set("Authorization", `Bearer ${options.token}`);
  if (options.provider) headers.set("X-LLM-Provider", options.provider);

  let body = options.body;
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  }

  const response = await fetch(`${API_PREFIX}${path}`, {
    ...options,
    headers,
    body,
    cache: "no-store"
  });

  if (!response.ok) {
    throw new ApiError(await responseErrorMessage(response), response.status);
  }

  return response;
}

export async function apiJson<T>(path: string, options: ApiOptions = {}) {
  const response = await apiFetch(path, options);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export async function responseErrorMessage(response: Response) {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") message = body.detail;
    else if (body.detail) message = JSON.stringify(body.detail);
  } catch {
    try {
      const text = await response.text();
      if (text) message = text;
    } catch {
      // Keep the status text.
    }
  }
  return message;
}

export async function readSse(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: { event: string; data: string }) => void
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() || "";
    for (const part of parts) dispatchSse(part, onEvent);
    if (done) break;
  }

  if (buffer.trim()) dispatchSse(buffer, onEvent);
}

function dispatchSse(part: string, onEvent: (event: { event: string; data: string }) => void) {
  const event = { event: "message", data: "" };
  for (const line of part.split(/\r?\n/)) {
    if (line.startsWith("event:")) event.event = line.slice(6).trim();
    if (line.startsWith("data:")) event.data += line.slice(5).trimStart();
  }
  if (event.data) onEvent(event);
}
