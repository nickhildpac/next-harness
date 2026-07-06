import { NextRequest } from "next/server";

const hopByHopHeaders = new Set([
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade"
]);

function backendUrl(request: NextRequest, path: string[]) {
  const base = process.env.FASTAPI_BASE_URL || "http://localhost:8000";
  const url = new URL(request.url);
  const target = new URL(`/${path.join("/")}`, base);
  target.search = url.search;
  return target;
}

async function proxy(request: NextRequest, context: { params: { path: string[] } }) {
  const headers = new Headers(request.headers);
  for (const header of hopByHopHeaders) headers.delete(header);

  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers,
    redirect: "manual",
    cache: "no-store"
  };

  if (!["GET", "HEAD"].includes(request.method)) {
    init.body = request.body;
    // Required by fetch when streaming multipart or SSE request bodies in Node.
    init.duplex = "half";
  }

  const upstream = await fetch(backendUrl(request, context.params.path), init);
  const responseHeaders = new Headers(upstream.headers);
  for (const header of hopByHopHeaders) responseHeaders.delete(header);

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders
  });
}

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
