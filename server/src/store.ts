import { mkdir, readFile, writeFile } from "fs/promises";
import { existsSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";

const DEFAULT_SYSTEM_PROMPT =
  "You are Parlance, a friendly real-time voice assistant. Answer in short, conversational sentences that sound natural when spoken.";
const DEFAULT_SEARX_URL =
  process.env.DEFAULT_SEARX_URL || "https://searxng.terra.lan";
const DEFAULT_SEARX_ALLOW_INSECURE =
  (process.env.DEFAULT_SEARX_ALLOW_INSECURE || "false") === "true";

type GuardRule = {
  id: string;
  phrase: string;
  response: string;
};

type Settings = {
  ollamaUrl: string;
  systemPrompt: string;
  searxUrl: string;
  searxAllowInsecure: boolean;
};

type Session = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
};

type Message = {
  id: string;
  sessionId: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
  metadata?: Record<string, unknown>;
};

type MemoryEntry = {
  id: string;
  label: string;
  value: string;
  createdAt: string;
};

type TimingStep = {
  name: string;
  durationMs: number;
};

type TimingLogEntry = {
  id: string;
  event: string;
  status: "ok" | "error";
  requestId?: string;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  steps: TimingStep[];
  metadata?: Record<string, string | number | boolean | null>;
};

type StoreData = {
  settings: Settings;
  sessions: Session[];
  messages: Message[];
  memory: MemoryEntry[];
  timings: TimingLogEntry[];
};

const DATA_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "data");
const DATA_FILE = join(DATA_DIR, "store.json");

let cache: StoreData | null = null;
let persistenceEnabled = true;

function normalizeSettings(partial?: Partial<Settings>): Settings {
  return {
    ollamaUrl:
      partial?.ollamaUrl ||
      process.env.DEFAULT_OLLAMA_URL ||
      "http://localhost:11434",
    systemPrompt:
      partial?.systemPrompt ||
      process.env.DEFAULT_SYSTEM_PROMPT ||
      DEFAULT_SYSTEM_PROMPT,
    searxUrl: partial?.searxUrl || DEFAULT_SEARX_URL,
    searxAllowInsecure:
      typeof partial?.searxAllowInsecure === "boolean"
        ? partial.searxAllowInsecure
        : DEFAULT_SEARX_ALLOW_INSECURE,
  };
}

async function ensureStoreLoaded(): Promise<StoreData> {
  if (cache) return cache;
  await ensureDataFile();
  if (persistenceEnabled && existsSync(DATA_FILE)) {
    try {
      const raw = await readFile(DATA_FILE, "utf8");
      cache = JSON.parse(raw) as StoreData;
      cache.settings = normalizeSettings(cache.settings);
      if (!Array.isArray(cache.sessions)) cache.sessions = [];
      if (!Array.isArray(cache.messages)) cache.messages = [];
      else {
        cache.messages = cache.messages.map((message: any) => ({
          ...message,
          metadata:
            message && typeof message.metadata === "object"
              ? message.metadata
              : undefined,
        }));
      }
      if (!Array.isArray(cache.memory)) cache.memory = [];
      if (!Array.isArray(cache.timings)) cache.timings = [];
      return cache;
    } catch (err) {
      console.warn("[store] failed to read data file, falling back to defaults", err);
    }
  }
  cache = createEmptyStore();
  await persistStore(cache);
  return cache!;
}

async function ensureDataFile() {
  if (!persistenceEnabled) return;
  try {
    if (!existsSync(DATA_DIR)) {
      await mkdir(DATA_DIR, { recursive: true });
    }
    if (!existsSync(DATA_FILE)) {
      await writeFile(DATA_FILE, JSON.stringify(createEmptyStore(), null, 2), "utf8");
    }
  } catch (err) {
    persistenceEnabled = false;
    console.warn("[store] persistence disabled (unable to write data directory)", err);
  }
}

function createEmptyStore(): StoreData {
  return {
    settings: normalizeSettings(),
    sessions: [],
    messages: [],
    memory: [],
    timings: [],
  };
}

async function persistStore(data: StoreData) {
  cache = data;
  if (!persistenceEnabled) return;
  try {
    await writeFile(DATA_FILE, JSON.stringify(data, null, 2), "utf8");
  } catch (err) {
    persistenceEnabled = false;
    console.warn("[store] persistence disabled (write failed)", err);
  }
}

export async function getSettings(): Promise<Settings> {
  const store = await ensureStoreLoaded();
  return store.settings;
}

export async function updateSettings(
  changes: Partial<Settings>
): Promise<Settings> {
  const store = await ensureStoreLoaded();
  store.settings = {
    ...store.settings,
    ...changes,
  };
  await persistStore(store);
  return store.settings;
}

export async function listSessions(): Promise<Session[]> {
  const store = await ensureStoreLoaded();
  return store.sessions.sort(
    (a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)
  );
}

export async function createSession(title?: string): Promise<Session> {
  const store = await ensureStoreLoaded();
  const now = new Date().toISOString();
  const session: Session = {
    id: randomUUID(),
    title: title?.trim() || "New Conversation",
    createdAt: now,
    updatedAt: now,
  };
  store.sessions.push(session);
  await persistStore(store);
  return session;
}

export async function renameSession(
  id: string,
  title: string
): Promise<Session> {
  const store = await ensureStoreLoaded();
  const session = store.sessions.find((s) => s.id === id);
  if (!session) {
    throw new Error("Session not found");
  }
  session.title = title.trim() || session.title;
  session.updatedAt = new Date().toISOString();
  await persistStore(store);
  return session;
}

export async function deleteSession(id: string) {
  const store = await ensureStoreLoaded();
  store.sessions = store.sessions.filter((s) => s.id !== id);
  store.messages = store.messages.filter((m) => m.sessionId !== id);
  await persistStore(store);
}

export async function listMessages(sessionId: string): Promise<Message[]> {
  const store = await ensureStoreLoaded();
  return store.messages
    .filter((m) => m.sessionId === sessionId)
    .sort((a, b) => Date.parse(a.createdAt) - Date.parse(b.createdAt));
}

export async function appendMessage(
  sessionId: string,
  role: Message["role"],
  content: string,
  metadata?: Record<string, unknown>
): Promise<Message> {
  const store = await ensureStoreLoaded();
  const session = store.sessions.find((s) => s.id === sessionId);
  if (!session) {
    throw new Error("Session not found");
  }
  const message: Message = {
    id: randomUUID(),
    sessionId,
    role,
    content,
    createdAt: new Date().toISOString(),
    metadata:
      metadata && typeof metadata === "object" ? { ...metadata } : undefined,
  };
  store.messages.push(message);
  session.updatedAt = message.createdAt;
  await persistStore(store);
  return message;
}

export async function listMemory(): Promise<MemoryEntry[]> {
  const store = await ensureStoreLoaded();
  return store.memory.sort(
    (a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt)
  );
}

export async function addMemoryEntry(
  label: string,
  value: string
): Promise<MemoryEntry> {
  const store = await ensureStoreLoaded();
  const entry: MemoryEntry = {
    id: randomUUID(),
    label: label.trim(),
    value: value.trim(),
    createdAt: new Date().toISOString(),
  };
  store.memory.push(entry);
  await persistStore(store);
  return entry;
}

export async function removeMemoryEntry(id: string) {
  const store = await ensureStoreLoaded();
  store.memory = store.memory.filter((entry) => entry.id !== id);
  await persistStore(store);
}

export async function recordTimingLog(
  entry: Omit<TimingLogEntry, "id">
): Promise<TimingLogEntry> {
  const store = await ensureStoreLoaded();
  const log: TimingLogEntry = {
    ...entry,
    id: randomUUID(),
    steps: Array.isArray(entry.steps) ? entry.steps : [],
  };
  const MAX_LOGS = 500;
  store.timings.push(log);
  if (store.timings.length > MAX_LOGS) {
    store.timings = store.timings.slice(-MAX_LOGS);
  }
  await persistStore(store);
  return log;
}

export async function listTimingLogs(limit = 100): Promise<TimingLogEntry[]> {
  const store = await ensureStoreLoaded();
  const safeLimit =
    typeof limit === "number" && Number.isFinite(limit) && limit > 0
      ? Math.min(Math.floor(limit), 500)
      : 100;
  return store.timings
    .slice(-safeLimit)
    .sort((a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt));
}

export type { Settings, Session, Message, MemoryEntry, TimingLogEntry, TimingStep };
export { DEFAULT_SYSTEM_PROMPT };
