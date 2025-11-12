const base = import.meta.env.VITE_API_BASE || "http://localhost:8787";

export type Settings = {
  ollamaUrl: string;
  systemPrompt: string;
  searxUrl: string;
  searxAllowInsecure: boolean;
};

export type SessionSummary = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
};

export type SessionMessage = {
  id: string;
  sessionId: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
  metadata?: Record<string, unknown>;
};

export type MemoryEntry = {
  id: string;
  label: string;
  value: string;
  createdAt: string;
};

export type WebSearchResult = {
  title: string;
  url: string;
  snippet: string;
  engine?: string;
  publishedAt?: string;
};

export type TimingLogStep = {
  name: string;
  durationMs: number;
};

export type TimingLogEntry = {
  id: string;
  event: string;
  status: "ok" | "error";
  requestId?: string;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  steps: TimingLogStep[];
  metadata?: Record<string, string | number | boolean | null>;
};

export type WeatherSummary = {
  location: {
    name: string;
    region?: string;
    country?: string;
    timezone?: string;
  };
  temperatureC: number;
  temperatureF: number;
  description: string;
  windKph: number;
  observedAt: string;
  source: string;
};

export type ThinkingPlan = {
  refinedQuery: string;
  summary?: string;
};

export type ThinkingPolish = {
  improvedReply: string;
  summary?: string;
};

export async function* chatStream(payload: {
  model: string;
  messages: { role: "system"|"user"|"assistant"; content: string }[];
  temperature?: number;
  ollama_url: string; // dynamic per request
}) {
  const resp = await fetch(`${base}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!resp.ok || !resp.body) throw new Error(`chat failed: ${resp.status}`);
  const reader = (resp.body as ReadableStream).getReader();
  const dec = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += dec.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const chunk of parts) {
      let evt: string | null = null;
      let data: string | null = null;
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event:")) evt = line.slice(6).trim();
        else if (line.startsWith("data:")) data = line.startsWith("data: ") ? line.slice(6) : line.slice(5);
      }
      if (!evt || data === null) continue;
      if (evt === "token") yield { token: data };
      if (evt === "error") throw new Error(data);
      if (evt === "done") return;
    }
  }
}

export async function transcribeAudio(blob: Blob) {
  const fd = new FormData();
  fd.append("audio", blob, "audio.webm");
  console.debug("[parlance] POST /stt", blob.size);
  const r = await fetch(`${base}/stt`, { method: "POST", body: fd });
  if (!r.ok) {
    let details = "";
    try {
      const data = await r.json();
      details = data?.detail || data?.error || "";
    } catch {}
    console.error("[parlance] stt failed", r.status, details);
    throw new Error(details || "stt failed");
  }
  return r.json() as Promise<{ text: string; language: string }>;
}

export async function tts(text: string): Promise<HTMLAudioElement> {
  const r = await fetch(`${base}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  if (!r.ok) throw new Error("tts failed");
  const buf = await r.arrayBuffer();
  const blob = new Blob([buf], { type: r.headers.get("Content-Type") ?? "audio/wav" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  return audio;
}

export async function listOllamaModels(ollamaUrl: string): Promise<string[]> {
  const r = await fetch(`${base}/ollama/models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ollama_url: ollamaUrl })
  });
  if (!r.ok) throw new Error(`list models failed: ${r.status}`);
  const data = await r.json();
  if (!Array.isArray(data?.models)) return [];
  return data.models.filter((m: unknown): m is string => typeof m === "string");
}

export async function fetchSettings(): Promise<Settings> {
  const r = await fetch(`${base}/settings`);
  if (!r.ok) throw new Error("Failed to load settings");
  return r.json();
}

export async function updateSettings(payload: Settings): Promise<Settings> {
  const r = await fetch(`${base}/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!r.ok) throw new Error("Failed to update settings");
  return r.json();
}

export async function searchWeb(
  query: string,
  options?: { limit?: number; searxUrl?: string; allowInsecure?: boolean }
): Promise<WebSearchResult[]> {
  const body: Record<string, unknown> = {
    query,
    limit: options?.limit ?? 5,
  };
  if (options?.searxUrl) {
    body.searxUrl = options.searxUrl;
  }
  if (typeof options?.allowInsecure === "boolean") {
    body.allowInsecure = options.allowInsecure;
  }
  const r = await fetch(`${base}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!r.ok) {
    const message =
      (data && typeof data.error === "string"
        ? data.error
        : text || `Web search failed (${r.status})`);
    throw new Error(message);
  }
  return Array.isArray(data?.results) ? data.results : [];
}

export async function generateSearchQuery(payload: {
  conversation: { role: "user" | "assistant"; content: string }[];
  latestUserMessage: string;
  model: string;
  ollamaUrl: string;
}): Promise<string> {
  const r = await fetch(`${base}/search/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(text || "Search query generation failed");
  }
  const data = await r.json();
  return typeof data?.query === "string" ? data.query : "";
}

export async function suggestSessionTitle(
  sessionId: string,
  payload: {
    conversation: { role: "user" | "assistant"; content: string }[];
    model: string;
    ollamaUrl: string;
  }
): Promise<string> {
  const r = await fetch(`${base}/sessions/${sessionId}/title/suggest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(text || "Session title generation failed");
  }
  const data = await r.json();
  return typeof data?.title === "string" ? data.title : "";
}

export async function fetchWeather(location: string): Promise<WeatherSummary> {
  const r = await fetch(`${base}/weather`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ location }),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(text || "Weather lookup failed");
  }
  return r.json();
}

export async function browseUrl(url: string, maxChars?: number) {
  const r = await fetch(`${base}/browse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, maxChars }),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(text || "Browse request failed");
  }
  return r.json() as Promise<{ url: string; title: string; summary: string; source: string }>;
}

export async function thinkingPreprocess(payload: {
  userQuery: string;
  model: string;
  ollamaUrl: string;
}): Promise<ThinkingPlan> {
  const r = await fetch(`${base}/thinking/preprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const text = await r.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!r.ok) {
    const message =
      (data && typeof data.error === "string"
        ? data.error
        : text || "Thinking preprocess failed");
    throw new Error(message);
  }
  if (!data?.plan) {
    throw new Error("Thinking preprocess did not return a plan.");
  }
  return data.plan as ThinkingPlan;
}

export async function thinkingPostprocess(payload: {
  userQuery: string;
  refinedQuery: string;
  assistantDraft: string;
  model: string;
  ollamaUrl: string;
}): Promise<ThinkingPolish> {
  const r = await fetch(`${base}/thinking/postprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const text = await r.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!r.ok) {
    const message =
      (data && typeof data.error === "string"
        ? data.error
        : text || "Thinking postprocess failed");
    throw new Error(message);
  }
  if (!data?.result) {
    throw new Error("Thinking postprocess did not return a result.");
  }
  return data.result as ThinkingPolish;
}

export async function fetchTimingLogs(limit = 100): Promise<TimingLogEntry[]> {
  const params = new URLSearchParams();
  if (typeof limit === "number" && Number.isFinite(limit) && limit > 0) {
    params.set("limit", String(Math.min(Math.floor(limit), 500)));
  }
  const path = params.size ? `${base}/timings?${params.toString()}` : `${base}/timings`;
  const r = await fetch(path);
  if (!r.ok) throw new Error("Failed to load timing logs");
  const data = await r.json();
  return Array.isArray(data?.entries) ? data.entries : [];
}

export async function listSessions(): Promise<SessionSummary[]> {
  const r = await fetch(`${base}/sessions`);
  if (!r.ok) throw new Error("Failed to list sessions");
  const data = await r.json();
  return Array.isArray(data?.sessions) ? data.sessions : [];
}

export async function createSession(title?: string): Promise<SessionSummary> {
  const r = await fetch(`${base}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title })
  });
  if (!r.ok) throw new Error("Failed to create session");
  return r.json();
}

export async function renameSession(id: string, title: string): Promise<SessionSummary> {
  const r = await fetch(`${base}/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title })
  });
  if (!r.ok) throw new Error("Failed to rename session");
  return r.json();
}

export async function deleteSession(id: string): Promise<void> {
  const r = await fetch(`${base}/sessions/${id}`, { method: "DELETE" });
  if (!r.ok && r.status !== 204) throw new Error("Failed to delete session");
}

export async function fetchSessionMessages(sessionId: string): Promise<SessionMessage[]> {
  const r = await fetch(`${base}/sessions/${sessionId}/messages`);
  if (!r.ok) throw new Error("Failed to load messages");
  const data = await r.json();
  return Array.isArray(data?.messages) ? data.messages : [];
}

export async function appendSessionMessage(
  sessionId: string,
  role: SessionMessage["role"],
  content: string,
  metadata?: Record<string, unknown>
): Promise<SessionMessage> {
  const r = await fetch(`${base}/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      metadata && Object.keys(metadata).length
        ? { role, content, metadata }
        : { role, content }
    ),
  });
  if (!r.ok) throw new Error("Failed to append message");
  return r.json();
}

export async function listMemoryEntries(): Promise<MemoryEntry[]> {
  const r = await fetch(`${base}/memory`);
  if (!r.ok) throw new Error("Failed to load memory");
  const data = await r.json();
  return Array.isArray(data?.entries) ? data.entries : [];
}

export async function addMemoryEntry(label: string, value: string): Promise<MemoryEntry> {
  const r = await fetch(`${base}/memory`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, value })
  });
  if (!r.ok) throw new Error("Failed to add memory entry");
  return r.json();
}

export async function deleteMemoryEntry(id: string): Promise<void> {
  const r = await fetch(`${base}/memory/${id}`, { method: "DELETE" });
  if (!r.ok && r.status !== 204) throw new Error("Failed to delete memory entry");
}
