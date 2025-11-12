import "dotenv/config";
import express, { type Request, type Response } from "express";
import cors from "cors";
import multer from "multer";
import https from "node:https";
import { randomUUID } from "crypto";
import { openSSE, sendSSE, closeSSE } from "./sse.js";
import { streamOllama } from "./ollama.js";
import {
  addMemoryEntry,
  appendMessage,
  createSession,
  deleteSession,
  getSettings,
  listMemory,
  listMessages,
  listSessions,
  listTimingLogs,
  removeMemoryEntry,
  renameSession,
  updateSettings,
} from "./store.js";
import { createTimingRecorder } from "./telemetry.js";

const app = express();
const CORS_ORIGIN = process.env.CORS_ORIGIN ?? "*";
app.use(cors({ origin: CORS_ORIGIN, credentials: false }));
app.use(express.json({ limit: "6mb" }));

const STT_URL = process.env.STT_URL ?? "http://stt:8000/transcribe";
const TTS_URL = process.env.TTS_URL ?? "http://tts:8600/speak";
const TTS_ENABLED = (process.env.TTS_ENABLED ?? "true") === "true";
const WEATHER_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search";
const WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast";
const BROWSE_MAX_CHARS = 2000;

// Chat SSE — Body: { model, messages, temperature?, ollama_url }
app.post("/chat/stream", async (req: Request, res: Response) => {
  const {
    model = "qwen2.5:7b-instruct",
    messages = [],
    temperature = 0.7,
    ollama_url = "",
  } = req.body || {};
  const hasLiveContext = Array.isArray(messages)
    ? messages.some(
        (msg: any) =>
          msg?.role === "system" &&
          typeof msg?.content === "string" &&
          (msg.content.toLowerCase().includes("web search") ||
            msg.content.includes("[live-data]"))
      )
    : false;
  const timer = createTimingRecorder("chat_stream", {
    model,
    temperature,
    messageCount: Array.isArray(messages) ? messages.length : 0,
    liveContext: hasLiveContext,
  });
  openSSE(res);
  let tokenCount = 0;
  try {
    if (!ollama_url) throw new Error("Missing ollama_url");
    const base = String(ollama_url);
    const lastUserMessage = Array.isArray(messages)
      ? [...messages].reverse().find((msg: any) => msg?.role === "user")
      : null;
    const guardContent =
      typeof lastUserMessage?.content === "string"
        ? lastUserMessage.content
        : "";
    for await (const { token, done } of streamOllama(base, model, messages, {
      temperature,
    })) {
      if (token) {
        tokenCount += token.length;
        sendSSE(res, "token", token);
      }
      if (done) break;
    }
    await timer.finish("ok", { tokens: tokenCount });
    sendSSE(res, "done", "ok");
  } catch (err: any) {
    await timer.finish("error", { error: err.message ?? "chat stream failed" });
    console.error("[chat/stream] failed:", err);
    sendSSE(res, "error", err.message ?? "error");
  } finally {
    closeSSE(res);
  }
});

// STT passthrough — multipart "audio"
const upload = multer({ storage: multer.memoryStorage() });
app.post(
  "/stt",
  upload.single("audio"),
  async (req: Request, res: Response) => {
    const timer = createTimingRecorder("stt", {
      size: req.file?.size ?? null,
    });
    try {
      if (!req.file) {
        await timer.finish("error", { reason: "missing_audio" });
        return res.status(400).json({ error: "missing audio" });
      }
      const form = new FormData();
      const audioBlob = bufferToBlob(
        req.file.buffer,
        req.file.mimetype ?? "audio/webm"
      );
      form.append("audio", audioBlob, "audio.webm");
      const r = await timer.time("transcribe_request", () =>
        fetch(STT_URL, { method: "POST", body: form })
      );
      if (!r.ok) throw new Error(`STT ${r.status} ${r.statusText}`);
      const data = await r.json();
      await timer.finish("ok", { format: req.file.mimetype });
      res.json(data);
    } catch (e: any) {
      await timer.finish("error", { error: e?.message || "stt failed" });
      console.error("[stt] passthrough failed:", e);
      res.status(500).json({ error: e.message ?? "stt failed" });
    }
  }
);

// TTS passthrough — Body: { text }
app.post("/tts", async (req: Request, res: Response) => {
  if (!TTS_ENABLED) return res.status(501).json({ error: "TTS disabled" });
  const timer = createTimingRecorder("tts");
  try {
    const { text } = req.body ?? {};
    if (!text || typeof text !== "string") {
      await timer.finish("error", { reason: "missing_text" });
      return res.status(400).json({ error: "missing text" });
    }
    const r = await timer.time("tts_request", () =>
      fetch(TTS_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
    );
    if (!r.ok) {
      await timer.finish("error", { reason: `status_${r.status}` });
      return res.status(r.status).send(await r.text());
    }
    res.setHeader("Content-Type", r.headers.get("Content-Type") ?? "audio/wav");
    res.send(Buffer.from(await r.arrayBuffer()));
    await timer.finish("ok", { chars: text.length });
  } catch (e: any) {
    await timer.finish("error", { error: e?.message || "tts failed" });
    console.error("[tts] passthrough failed:", e);
    res.status(500).json({ error: e.message ?? "tts failed" });
  }
});

app.post("/ollama/models", async (req: Request, res: Response) => {
  try {
    const { ollama_url = "" } = req.body || {};
    if (!ollama_url) return res.status(400).json({ error: "missing ollama_url" });
    const base = String(ollama_url).replace(/\/+$/, "");
    const r = await fetch(`${base}/api/tags`);
    if (!r.ok) throw new Error(`Ollama ${r.status} ${r.statusText}`);
    const data = await r.json();
    const models = Array.isArray(data?.models)
      ? data.models
          .map((entry: any) => entry?.name || entry?.model)
          .filter((name: unknown): name is string => typeof name === "string" && name.length > 0)
      : [];
    res.json({ models });
  } catch (err: any) {
    console.error("[models] fetch failed:", err);
    res.status(500).json({ error: err.message ?? "model listing failed" });
  }
});

app.get("/health", (_req: Request, res: Response) => res.json({ ok: true }));
app.get("/settings", async (_req: Request, res: Response) => {
  const settings = await getSettings();
  res.json(settings);
});

app.put("/settings", async (req: Request, res: Response) => {
  try {
    const { ollamaUrl, systemPrompt, searxUrl, searxAllowInsecure } =
      req.body ?? {};
    if (!ollamaUrl || typeof ollamaUrl !== "string") {
      return res.status(400).json({ error: "ollamaUrl is required" });
    }
    if (!systemPrompt || typeof systemPrompt !== "string") {
      return res.status(400).json({ error: "systemPrompt is required" });
    }
    if (!searxUrl || typeof searxUrl !== "string") {
      return res.status(400).json({ error: "searxUrl is required" });
    }
    const allowInsecure =
      typeof searxAllowInsecure === "boolean" ? searxAllowInsecure : false;
    const settings = await updateSettings({
      ollamaUrl: ollamaUrl.trim(),
      systemPrompt: systemPrompt.trim(),
      searxUrl: searxUrl.trim(),
      searxAllowInsecure: allowInsecure,
    });
    res.json(settings);
  } catch (err: any) {
    console.error("[settings] update failed", err);
    res.status(500).json({ error: err.message ?? "settings update failed" });
  }
});

app.get("/timings", async (req: Request, res: Response) => {
  try {
    const limitRaw = Array.isArray(req.query.limit)
      ? req.query.limit[0]
      : req.query.limit;
    const limit = limitRaw ? Number(limitRaw) : undefined;
    const entries = await listTimingLogs(limit);
    res.json({ entries });
  } catch (err: any) {
    console.error("[timings] list failed", err);
    res.status(500).json({ error: err.message ?? "timings list failed" });
  }
});

app.post("/search", async (req: Request, res: Response) => {
  const { query, limit = 5, searxUrl, allowInsecure } = req.body ?? {};
  const timer = createTimingRecorder("search", {
    query:
      typeof query === "string" && query.length <= 160 ? query : undefined,
    limit,
  });
  try {
    if (!query || typeof query !== "string") {
      await timer.finish("error", { reason: "missing_query" });
      return res.status(400).json({ error: "query is required" });
    }
    const parsedLimit =
      typeof limit === "number" && Number.isFinite(limit) && limit > 0
        ? Math.min(Math.floor(limit), 10)
        : 5;
    const settings = await getSettings();
    const base =
      (typeof searxUrl === "string" && searxUrl.trim()) ||
      settings.searxUrl ||
      process.env.DEFAULT_SEARX_URL ||
      "https://searxng.terra.lan";
    const allowInsecureTls =
      typeof allowInsecure === "boolean"
        ? allowInsecure
        : settings.searxAllowInsecure;
    const url = new URL("/search", base);
    url.searchParams.set("q", query);
    url.searchParams.set("format", "json");
    url.searchParams.set("language", "en");
    url.searchParams.set("safesearch", "2");
    url.searchParams.set("categories", "general");
    url.searchParams.set("limit", String(parsedLimit));
    const fetchOptions: RequestInit & { agent?: any } = {
      headers: { "User-Agent": "ParlanceSearchProxy/1.0" },
    };
    if (url.protocol === "https:" && allowInsecureTls) {
      fetchOptions.agent = new https.Agent({ rejectUnauthorized: false });
    }
    const r = await timer.time("searx_request", () =>
      fetch(url.toString(), fetchOptions)
    );
    if (!r.ok) {
      let detail: any = null;
      try {
        detail = await r.json();
      } catch {
        detail = await r.text();
      }
      const reason =
        (detail && typeof detail === "object" && "error" in detail
          ? (detail as any).error
          : typeof detail === "string"
          ? detail
          : null) || `${r.status} ${r.statusText}`;
      throw new Error(`SearxNG responded with ${reason}`);
    }
    const data = await r.json();
    const results = Array.isArray(data?.results)
      ? data.results
          .filter(
            (entry: any) =>
              entry &&
              typeof entry.url === "string" &&
              entry.url.length > 0 &&
              typeof entry.title === "string"
          )
          .slice(0, parsedLimit)
          .map((entry: any) => {
            const publishedRaw =
              entry.published ??
              entry.publishedDate ??
              entry.date ??
              entry.published_at ??
              null;
            let publishedAt: string | undefined;
            if (publishedRaw) {
              const parsed = Date.parse(publishedRaw);
              if (!Number.isNaN(parsed)) {
                publishedAt = new Date(parsed).toISOString();
              }
            } else if (
              typeof entry.timestamp === "number" &&
              Number.isFinite(entry.timestamp)
            ) {
              publishedAt = new Date(entry.timestamp * 1000).toISOString();
            }
            return {
              title: entry.title || entry.url,
              url: entry.url,
              snippet:
                entry.content ||
                entry.snippet ||
                entry.description ||
                "",
              engine: entry.engine || entry.source || "searxng",
              publishedAt,
            };
          })
      : [];
    await timer.finish("ok", { results: results.length });
    res.json({ results });
  } catch (err: any) {
    await timer.finish("error", { error: err?.message || "search failed" });
    console.error("[search] failed", err);
    res.status(500).json({ error: err.message ?? "search failed" });
  }
});

app.post("/thinking/preprocess", async (req: Request, res: Response) => {
  const {
    userQuery,
    model = "qwen2.5:7b-instruct",
    ollamaUrl,
  } = req.body ?? {};
  const trimmedQuery =
    typeof userQuery === "string" ? userQuery.trim() : "";
  const timer = createTimingRecorder("thinking_preprocess", {
    model,
    queryLength: trimmedQuery.length || undefined,
  });
  try {
    if (!ollamaUrl || typeof ollamaUrl !== "string") {
      await timer.finish("error", { reason: "missing_ollama" });
      return res.status(400).json({ error: "ollamaUrl is required" });
    }
    if (!trimmedQuery) {
      await timer.finish("error", { reason: "missing_query" });
      return res.status(400).json({ error: "userQuery is required" });
    }
    const messages = [
      {
        role: "system",
        content:
          "You distill messy user input into a focused prompt the assistant can act on while preserving every fact, constraint, and intent exactly. Never invent new requirements or drop details. Return strict JSON: {\"refinedQuery\": string, \"summary\": string}. The refinedQuery must be executable instructions (max 80 words). The summary is a short status update (<15 words) describing what you decided.",
      },
      {
        role: "user",
        content: `User request:\n"""${trimmedQuery}"""\n\nGenerate the JSON now.`,
      },
    ] as { role: "system" | "user" | "assistant"; content: string }[];
    let raw = "";
    await timer.time("ollama_stream", async () => {
      for await (const chunk of streamOllama(ollamaUrl, model, messages, {
        temperature: 0.2,
      })) {
        if (chunk.token) raw += chunk.token;
      }
    });
    const parsed = parseThinkingPlan(raw);
    if (!parsed) {
      await timer.finish("error", { reason: "invalid_json" });
      return res
        .status(422)
        .json({ error: "preprocess result was not valid JSON", raw });
    }
    await timer.finish("ok", {
      refinedLength: parsed.refinedQuery.length,
    });
    res.json({ plan: parsed });
  } catch (err: any) {
    await timer.finish("error", {
      error: err?.message || "thinking_preprocess failed",
    });
    console.error("[thinking/preprocess] failed", err);
    res
      .status(500)
      .json({ error: err.message ?? "thinking preprocess failed" });
  }
});

app.post("/thinking/postprocess", async (req: Request, res: Response) => {
  const {
    userQuery,
    refinedQuery,
    assistantDraft,
    model = "qwen2.5:7b-instruct",
    ollamaUrl,
  } = req.body ?? {};
  const timer = createTimingRecorder("thinking_postprocess", {
    model,
    userLength:
      typeof userQuery === "string" ? userQuery.trim().length : undefined,
    draftLength:
      typeof assistantDraft === "string"
        ? assistantDraft.trim().length
        : undefined,
  });
  const original = typeof userQuery === "string" ? userQuery.trim() : "";
  const refined = typeof refinedQuery === "string" ? refinedQuery.trim() : "";
  const draft =
    typeof assistantDraft === "string" ? assistantDraft.trim() : "";
  try {
    if (!ollamaUrl || typeof ollamaUrl !== "string") {
      await timer.finish("error", { reason: "missing_ollama" });
      return res.status(400).json({ error: "ollamaUrl is required" });
    }
    if (!draft) {
      await timer.finish("error", { reason: "missing_reply" });
      return res.status(400).json({ error: "assistantDraft is required" });
    }
    const instructions = [
      "You upgrade assistant drafts so they better satisfy the user intent without changing meaning or introducing new facts.",
      "Consider both the original user wording and the refined intent, then return strict JSON: {\"improvedReply\": string, \"summary\": string}.",
      "Make the improvedReply concise, direct, and under 220 words, preserving all factual content.",
      "The summary is a brief status (<15 words) describing what changed.",
    ].join(" ");
    const pieces = [
      `User original:\n"""${original || "(none provided)"}"""`,
      `Refined intent:\n"""${refined || "(none provided)"}"""`,
      `Assistant draft:\n"""${draft}"""`,
      "Return the JSON now.",
    ].join("\n\n");
    const messages = [
      { role: "system", content: instructions },
      { role: "user", content: pieces },
    ] as { role: "system" | "user" | "assistant"; content: string }[];
    let raw = "";
    await timer.time("ollama_stream", async () => {
      for await (const chunk of streamOllama(ollamaUrl, model, messages, {
        temperature: 0.2,
      })) {
        if (chunk.token) raw += chunk.token;
      }
    });
    const parsed = parseThinkingPolish(raw);
    if (!parsed) {
      await timer.finish("error", { reason: "invalid_json" });
      return res
        .status(422)
        .json({ error: "postprocess result was not valid JSON", raw });
    }
    await timer.finish("ok", {
      improvedLength: parsed.improvedReply.length,
    });
    res.json({ result: parsed });
  } catch (err: any) {
    await timer.finish("error", {
      error: err?.message || "thinking_postprocess failed",
    });
    console.error("[thinking/postprocess] failed", err);
    res
      .status(500)
      .json({ error: err.message ?? "thinking postprocess failed" });
  }
});

app.post("/search/query", async (req: Request, res: Response) => {
  const {
    conversation = [],
    latestUserMessage,
    model = "qwen2.5:7b-instruct",
    ollamaUrl,
  } = req.body ?? {};
  const timer = createTimingRecorder("search_query", {
    model,
    conversation: Array.isArray(conversation) ? conversation.length : undefined,
  });
  try {
    if (
      !Array.isArray(conversation) ||
      !conversation.every(
        (entry) =>
          entry &&
          typeof entry.role === "string" &&
          typeof entry.content === "string"
      )
    ) {
      await timer.finish("error", { reason: "invalid_conversation" });
      return res.status(400).json({ error: "conversation is invalid" });
    }
    if (!latestUserMessage || typeof latestUserMessage !== "string") {
      await timer.finish("error", { reason: "missing_latest" });
      return res.status(400).json({ error: "latestUserMessage is required" });
    }
    if (!ollamaUrl || typeof ollamaUrl !== "string") {
      await timer.finish("error", { reason: "missing_ollama" });
      return res.status(400).json({ error: "ollamaUrl is required" });
    }
    const trimmedConversation = conversation.slice(-8);
    const lines = trimmedConversation
      .map(
        (entry: any, idx: number) =>
          `${idx + 1}. (${entry.role}) ${entry.content}`
      )
      .join("\n");
    const messages: { role: "system" | "user" | "assistant"; content: string }[] = [
      {
        role: "system",
        content:
          "You turn chat transcripts into focused web search queries. Use concise natural-language keywords, include named entities, and prefer the most recent conversation context. Output only the final query without commentary.",
      },
      {
        role: "user",
        content: `Conversation timeline (most recent first):\n${lines || "(no previous context)"}\n\nLatest user request: ${latestUserMessage}\n\nProvide the single best search query to find up-to-date information relevant to the latest request.`,
      },
    ];
    let query = "";
    await timer.time("ollama_stream", async () => {
      for await (const chunk of streamOllama(
        ollamaUrl,
        model,
        messages,
        {
          temperature: 0.2,
        }
      )) {
        if (chunk.token) query += chunk.token;
      }
    });
    const finalQuery = query.trim().replace(/^["']|["']$/g, "");
    await timer.finish("ok", {
      chars: finalQuery.length,
    });
    res.json({ query: finalQuery || latestUserMessage });
  } catch (err: any) {
    await timer.finish("error", { error: err?.message || "search_query failed" });
    console.error("[search/query] failed", err);
    res.status(500).json({ error: err.message ?? "query generation failed" });
  }
});

app.post("/weather", async (req: Request, res: Response) => {
  const { location } = req.body ?? {};
  const normalized = typeof location === "string" ? normalizeLocationQuery(location) : "";
  const timer = createTimingRecorder("weather", {
    location:
      typeof location === "string" && location.length <= 120 ? location : undefined,
    normalizedLocation: normalized || undefined,
  });
  try {
    if (!location || typeof location !== "string") {
      await timer.finish("error", { reason: "missing_location" });
      return res.status(400).json({ error: "location is required" });
    }
    const geo = await timer.time("geocode", () => fetchGeocode(location));
    if (!geo) {
      await timer.finish("error", { reason: "not_found" });
      return res.status(404).json({
        error: "Unable to find that location. Try specifying the country or region.",
      });
    }
    const weather = await timer.time("forecast", () =>
      fetchWeather(geo.latitude, geo.longitude, geo.timezone)
    );
    await timer.finish("ok", {
      resolved: geo.name,
      country: geo.country,
    });
    res.json({
      location: {
        name: geo.name,
        region: geo.admin1,
        country: geo.country,
        timezone: geo.timezone,
      },
      temperatureC: weather.temperatureC,
      temperatureF: weather.temperatureF,
      description: weather.description,
      windKph: weather.windKph,
      observedAt: weather.observedAt,
      source: "open-meteo.com",
    });
  } catch (err: any) {
    await timer.finish("error", { error: err?.message || "weather failed" });
    console.error("[weather] lookup failed", err);
    res.status(500).json({ error: err.message ?? "weather lookup failed" });
  }
});

app.post("/browse", async (req: Request, res: Response) => {
  const { url, maxChars = BROWSE_MAX_CHARS } = req.body ?? {};
  const timer = createTimingRecorder("browse", {
    url: typeof url === "string" && url.length <= 200 ? url : undefined,
  });
  try {
    if (!url || typeof url !== "string") {
      await timer.finish("error", { reason: "missing_url" });
      return res.status(400).json({ error: "url is required" });
    }
    let parsed: URL;
    try {
      parsed = new URL(url);
    } catch {
      await timer.finish("error", { reason: "invalid_url" });
      return res.status(400).json({ error: "invalid url" });
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      await timer.finish("error", { reason: "unsupported_protocol" });
      return res.status(400).json({ error: "only http/https urls supported" });
    }
    const response = await timer.time("fetch", () =>
      fetch(parsed.toString(), {
        headers: {
          "User-Agent":
            "ParlanceBrowser/1.0 (+https://github.com/)",
          Accept: "text/html,text/plain",
        },
      })
    );
    if (!response.ok) {
      await timer.finish("error", {
        reason: `status_${response.status}`,
      });
      throw new Error(`browse failed ${response.status} ${response.statusText}`);
    }
    const contentType = response.headers.get("Content-Type") || "";
    if (!/text\/(?:html|plain)/i.test(contentType)) {
      await timer.finish("error", { reason: "unsupported_content" });
      return res
        .status(400)
        .json({ error: "URL did not return readable text content." });
    }
    const body = await response.text();
    const { title, summary } = summarizeHtml(body, maxChars);
    await timer.finish("ok", { hostname: parsed.hostname });
    res.json({
      url: parsed.toString(),
      title,
      summary,
      source: parsed.hostname,
    });
  } catch (err: any) {
    await timer.finish("error", { error: err?.message || "browse failed" });
    console.error("[browse] failed", err);
    res.status(500).json({ error: err.message ?? "browse failed" });
  }
});

app.post("/sessions/:id/title/suggest", async (req: Request, res: Response) => {
  const {
    conversation = [],
    model = "qwen2.5:7b-instruct",
    ollamaUrl,
  } = req.body ?? {};
  const timer = createTimingRecorder("title_suggest", {
    model,
    messages: Array.isArray(conversation) ? conversation.length : undefined,
  });
  try {
    if (!ollamaUrl || typeof ollamaUrl !== "string") {
      await timer.finish("error", { reason: "missing_ollama" });
      return res.status(400).json({ error: "ollamaUrl is required" });
    }
    if (
      !Array.isArray(conversation) ||
      !conversation.every(
        (entry) =>
          entry &&
          (entry.role === "user" || entry.role === "assistant") &&
          typeof entry.content === "string"
      )
    ) {
      await timer.finish("error", { reason: "invalid_conversation" });
      return res.status(400).json({ error: "conversation is invalid" });
    }
    const trimmed = conversation
      .slice(-8)
      .map(
        (entry: any, idx: number) =>
          `${idx + 1}. ${entry.role === "assistant" ? "Assistant" : "User"}: ${
            entry.content
          }`
      )
      .join("\n");
    const messages: { role: "system" | "user" | "assistant"; content: string }[] =
      [
        {
          role: "system",
          content:
            "You generate short conversation titles (max 5 words). Titles must be descriptive, avoid meta words like 'discussion' or 'chat', and exclude platform names unless critical. Return only the title text.",
        },
        {
          role: "user",
          content: `Conversation summary:\n${trimmed || "(empty)"}\n\nProvide a concise 5-word title:`,
        },
      ];
    let title = "";
    await timer.time("ollama_stream", async () => {
      for await (const chunk of streamOllama(ollamaUrl, model, messages, {
        temperature: 0.2,
      })) {
        if (chunk.token) title += chunk.token;
      }
    });
    const cleaned = title.split("\n")[0]?.trim() ?? "";
    await timer.finish("ok", { length: cleaned.length });
    res.json({ title: cleaned });
  } catch (err: any) {
    await timer.finish("error", { error: err?.message || "session title suggestion failed" });
    console.error("[sessions/title] suggest failed", err);
    res
      .status(500)
      .json({ error: err.message ?? "session title suggestion failed" });
  }
});

app.get("/sessions", async (_req: Request, res: Response) => {
  const sessions = await listSessions();
  res.json({ sessions });
});

app.post("/sessions", async (req: Request, res: Response) => {
  try {
    const { title } = req.body ?? {};
    const session = await createSession(title);
    res.status(201).json(session);
  } catch (err: any) {
    console.error("[sessions] create failed", err);
    res.status(500).json({ error: err.message ?? "session create failed" });
  }
});

app.patch("/sessions/:id", async (req: Request, res: Response) => {
  try {
    const { title } = req.body ?? {};
    if (!title || typeof title !== "string") {
      return res.status(400).json({ error: "title is required" });
    }
    const session = await renameSession(req.params.id, title);
    res.json(session);
  } catch (err: any) {
    if (err.message === "Session not found") {
      return res.status(404).json({ error: "not found" });
    }
    console.error("[sessions] rename failed", err);
    res.status(500).json({ error: err.message ?? "rename failed" });
  }
});

app.delete("/sessions/:id", async (req: Request, res: Response) => {
  try {
    await deleteSession(req.params.id);
    res.status(204).end();
  } catch (err: any) {
    console.error("[sessions] delete failed", err);
    res.status(500).json({ error: err.message ?? "delete failed" });
  }
});

app.get("/sessions/:id/messages", async (req: Request, res: Response) => {
  try {
    const messages = await listMessages(req.params.id);
    res.json({ messages });
  } catch (err: any) {
    console.error("[messages] list failed", err);
    res.status(500).json({ error: err.message ?? "list messages failed" });
  }
});

app.post("/sessions/:id/messages", async (req: Request, res: Response) => {
  const { role, content, metadata } = req.body ?? {};
  if (!role || typeof role !== "string") {
    return res.status(400).json({ error: "role is required" });
  }
  if (!content || typeof content !== "string") {
    return res.status(400).json({ error: "content is required" });
  }
  if (
    typeof metadata !== "undefined" &&
    (metadata === null ||
      typeof metadata !== "object" ||
      Array.isArray(metadata))
  ) {
    return res.status(400).json({ error: "metadata must be an object" });
  }
  try {
    const message = await appendMessage(
      req.params.id,
      role as "user" | "assistant" | "system",
      content,
      metadata
    );
    res.status(201).json(message);
  } catch (err: any) {
    if (err.message === "Session not found") {
      return res.status(404).json({ error: "not found" });
    }
    console.error("[messages] append failed", err);
    res.status(500).json({ error: err.message ?? "append message failed" });
  }
});

async function fetchGeocode(query: string) {
  const normalizedQuery = normalizeLocationQuery(query);
  if (!normalizedQuery) {
    return null;
  }
  const url = new URL(WEATHER_GEOCODE_URL);
  url.searchParams.set("name", normalizedQuery);
  url.searchParams.set("count", "1");
  url.searchParams.set("language", "en");
  const response = await fetch(url.toString());
  if (!response.ok) {
    throw new Error(`geocode failed ${response.status}`);
  }
  const data = (await response.json()) as any;
  if (!Array.isArray(data?.results) || data.results.length === 0) {
    return null;
  }
  return data.results[0];
}

async function fetchWeather(lat: number, lon: number, timezone?: string) {
  const url = new URL(WEATHER_FORECAST_URL);
  url.searchParams.set("latitude", String(lat));
  url.searchParams.set("longitude", String(lon));
  url.searchParams.set("current", "temperature_2m,weather_code,wind_speed_10m");
  url.searchParams.set("timezone", timezone || "auto");
  const response = await fetch(url.toString());
  if (!response.ok) {
    throw new Error(`weather failed ${response.status}`);
  }
  const data = (await response.json()) as any;
  const current = data?.current;
  if (!current) throw new Error("missing weather data");
  const tempC = Number(current.temperature_2m);
  const weatherCode = Number(current.weather_code);
  return {
    temperatureC: tempC,
    temperatureF: Number.isFinite(tempC) ? tempC * 1.8 + 32 : null,
    description: weatherCodeToDescription(weatherCode),
    windKph: Number(current.wind_speed_10m),
    observedAt: current.time,
  };
}

type ThinkingPlan = {
  refinedQuery: string;
  summary: string;
};

type ThinkingPolish = {
  improvedReply: string;
  summary: string;
};

function parseThinkingPlan(raw: string): ThinkingPlan | null {
  const target = extractJsonObject(raw);
  if (!target || typeof target.refinedQuery !== "string") return null;
  const refined = target.refinedQuery.toString().trim();
  if (!refined) return null;
  return {
    refinedQuery: refined,
    summary: sanitizeSentence(target.summary),
  };
}

function parseThinkingPolish(raw: string): ThinkingPolish | null {
  const target = extractJsonObject(raw);
  if (!target || typeof target.improvedReply !== "string") return null;
  const reply = target.improvedReply.toString().trim();
  if (!reply) return null;
  return {
    improvedReply: reply,
    summary: sanitizeSentence(target.summary),
  };
}

function extractJsonObject(raw: string): any | null {
  if (!raw || typeof raw !== "string") return null;
  const trimmed = raw.trim();
  const attempts = [trimmed];
  const firstBrace = raw.indexOf("{");
  const lastBrace = raw.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    attempts.push(raw.slice(firstBrace, lastBrace + 1));
  }
  for (const chunk of attempts) {
    try {
      return JSON.parse(chunk);
    } catch {
      continue;
    }
  }
  return null;
}

function sanitizeSentence(value: unknown) {
  if (!value || typeof value !== "string") return "";
  return value.replace(/\s+/g, " ").trim();
}

function weatherCodeToDescription(code: number) {
  const mapping: Record<number, string> = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
  };
  return mapping[code] || "Unknown conditions";
}

function normalizeLocationQuery(raw: string) {
  if (!raw) return "";
  const normalized = raw.replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  const patterns = [
    /\b(?:weather|temperature|temp|forecast|rain|raining|snow|snowing|wind|windy|humid|humidity|sunny|storm|stormy|conditions?|condition|hot|cold|warm|cool|degrees?|degree|umbrella|precipitation|climate|heatwave|freezing|chilly|heat|hail|blizzard)\b[^?!]*?\b(?:in|for|at|near|around|by)\s+([^?!]+)/i,
    /\b(?:in|for|at|near|around|by)\s+([^?!]+?)\s+(?:weather|temperature|temp|forecast|conditions?|condition|rain|snow|wind|storm|hot|cold|warm|cool|humid|humidity|degrees?|degree)\b/i,
    /^(?:\b(?:check|show|tell|give|share|provide|fetch|get|find|see|need|want|please|kindly|can|could|would|will|should|let's|lets|what(?:'s| is)?|how)\b\s+)?([^?!]+?)\s+(?:weather|temperature|temp|forecast|conditions?|condition|rain|snow|wind|storm|hot|cold|warm|cool|humid|humidity|degrees?|degree)\b/i,
    /\b(?:weather|forecast)\s+for\s+([^?!]+)/i,
  ];
  for (const pattern of patterns) {
    const match = pattern.exec(normalized);
    if (match?.[1]) {
      const cleaned = cleanLocationCandidate(match[1]);
      if (cleaned) return cleaned;
    }
  }
  const stripped = stripWeatherNoise(normalized);
  const cleanedFallback = cleanLocationCandidate(stripped);
  if (cleanedFallback) return cleanedFallback;
  return cleanLocationCandidate(normalized);
}

function stripWeatherNoise(text: string) {
  return text
    .replace(
      /\b(?:what(?:'s| is)?|tell|show|give|check|checked|checking|checks|me|current|outside|the|weather|temperature|temp|temps|forecast|right|now|today|tonight|tomorrow|please|kindly|do|you|know|how|hot|cold|warm|cool|rain(?:ing)?|snow(?:ing)?|wind(?:y)?|sunny|stormy|conditions?|condition|report|update|info|information|should|i|bring|pack|need|an|a|umbrella|coat|jacket|like|is|it|going|to|be|looking|will|degrees?|degree|humid|humidity|precipitation|climate|freezing|chilly|heatwave|heat|storm|storms?)\b/gi,
      " "
    )
    .replace(/\s+/g, " ")
    .trim();
}

const LEADING_LOCATION_WORDS = new Set(["in", "for", "at", "near", "around", "about", "on", "by"]);

const TRAILING_LOCATION_WORDS = new Set([
  "today",
  "tonight",
  "tomorrow",
  "now",
  "currently",
  "please",
  "pls",
  "thanks",
  "thank",
  "you",
  "outside",
  "there",
  "later",
  "soon",
  "conditions",
  "condition",
  "weather",
  "forecast",
  "temperature",
  "temps",
  "temp",
  "report",
  "update",
  "info",
  "information",
  "like",
  "right",
  "moment",
  "be",
  "degree",
  "degrees",
  "storm",
  "storms",
  "precipitation",
]);

const TRAILING_LOCATION_PHRASES: string[][] = [
  ["right", "now"],
  ["at", "the", "moment"],
  ["this", "morning"],
  ["this", "afternoon"],
  ["this", "evening"],
  ["this", "week"],
  ["this", "weekend"],
  ["later", "today"],
  ["later", "tonight"],
];

function cleanLocationCandidate(fragment: string) {
  if (!fragment) return "";
  let text = fragment.replace(/[?]/g, " ").replace(/\s+/g, " ").trim();
  if (!text) return "";
  text = text.replace(/^[,.\-–—\s]+|[,.\-–—\s]+$/g, "");
  if (!text) return "";
  let tokens = text.split(" ").filter(Boolean);
  while (
    tokens.length &&
    LEADING_LOCATION_WORDS.has(stripPunctuation(tokens[0]).toLowerCase())
  ) {
    tokens.shift();
  }
  let changed = true;
  while (changed && tokens.length) {
    changed = false;
    for (const phrase of TRAILING_LOCATION_PHRASES) {
      if (
        tokens.length >= phrase.length &&
        phrase.every(
          (word, idx) =>
            stripPunctuation(tokens[tokens.length - phrase.length + idx]).toLowerCase() === word
        )
      ) {
        tokens = tokens.slice(0, -phrase.length);
        changed = true;
        break;
      }
    }
    if (!changed && tokens.length) {
      const last = stripPunctuation(tokens[tokens.length - 1]).toLowerCase();
      if (TRAILING_LOCATION_WORDS.has(last)) {
        tokens = tokens.slice(0, -1);
        changed = true;
      }
    }
  }
  const cleaned = tokens.join(" ").trim();
  if (!cleaned) return "";
  const trimmed = cleaned.replace(/[?.,!]+$/g, "").trim();
  if (!trimmed) return "";
  const orIndex = trimmed.toLowerCase().indexOf(" or ");
  const candidate = orIndex > 0 ? trimmed.slice(0, orIndex).trim() : trimmed;
  return looksLikeLocationCandidate(candidate) ? candidate : "";
}

function stripPunctuation(text: string) {
  return text.replace(/^[,.;:!?]+|[,.;:!?]+$/g, "");
}

const NON_LOCATION_TOKENS = new Set([
  "what",
  "what's",
  "whats",
  "is",
  "it",
  "the",
  "a",
  "an",
  "please",
  "thanks",
  "thank",
  "you",
  "current",
  "currently",
  "today",
  "tonight",
  "tomorrow",
  "right",
  "now",
  "outside",
  "there",
  "how",
  "hot",
  "cold",
  "warm",
  "cool",
  "rain",
  "raining",
  "snow",
  "snowing",
  "wind",
  "windy",
  "humid",
  "humidity",
  "storm",
  "stormy",
  "forecast",
  "weather",
  "temperature",
  "temp",
  "temps",
  "degree",
  "degrees",
  "conditions",
  "condition",
  "report",
  "update",
  "info",
  "information",
  "should",
  "i",
  "need",
  "bring",
  "pack",
  "umbrella",
  "coat",
  "jacket",
  "going",
  "to",
  "be",
  "will",
  "do",
  "you",
  "me",
  "for",
  "in",
  "at",
  "near",
  "around",
  "by",
  "like",
  "about",
  "give",
  "check",
  "checked",
  "checking",
  "checks",
  "tell",
  "show",
  "looking",
  "city",
  "town",
  "area",
  "region",
  "place",
  "status",
]);

function looksLikeLocationCandidate(text: string) {
  if (!text) return false;
  const tokens = text
    .split(/\s+/)
    .map((token) => stripPunctuation(token).toLowerCase())
    .filter(Boolean);
  if (!tokens.length) return false;
  return tokens.some((token) => !NON_LOCATION_TOKENS.has(token));
}

function summarizeHtml(html: string, maxChars: number) {
  const withoutScripts = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ");
  const titleMatch = withoutScripts.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const title = titleMatch
    ? decodeEntities(titleMatch[1].trim()).slice(0, 120)
    : "Untitled page";
  const text = decodeEntities(
    withoutScripts
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim()
  );
  const summary = text.slice(0, Math.min(maxChars, text.length));
  return { title, summary };
}

function decodeEntities(text: string) {
  return text
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">");
}

app.get("/memory", async (_req: Request, res: Response) => {
  const entries = await listMemory();
  res.json({ entries });
});

app.post("/memory", async (req: Request, res: Response) => {
  const { label, value } = req.body ?? {};
  if (!label || typeof label !== "string") {
    return res.status(400).json({ error: "label is required" });
  }
  if (!value || typeof value !== "string") {
    return res.status(400).json({ error: "value is required" });
  }
  try {
    const entry = await addMemoryEntry(label, value);
    res.status(201).json(entry);
  } catch (err: any) {
    console.error("[memory] add failed", err);
    res.status(500).json({ error: err.message ?? "add memory failed" });
  }
});

app.delete("/memory/:id", async (req: Request, res: Response) => {
  try {
    await removeMemoryEntry(req.params.id);
    res.status(204).end();
  } catch (err: any) {
    console.error("[memory] delete failed", err);
    res.status(500).json({ error: err.message ?? "delete memory failed" });
  }
});

app.listen(8787, () => console.log("[server] :8787"));

function bufferToBlob(buffer: Buffer, mime: string) {
  // Copy into a fresh ArrayBuffer (ensures type is ArrayBuffer, not SharedArrayBuffer).
  const ab = new ArrayBuffer(buffer.length);
  const view = new Uint8Array(ab);
  view.set(buffer);
  return new Blob([ab], { type: mime });
}
