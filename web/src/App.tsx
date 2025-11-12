import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  KeyboardEvent as ReactKeyboardEvent,
} from "react";
import {
  chatStream,
  listOllamaModels,
  tts,
  transcribeAudio,
  fetchSettings,
  updateSettings as updateSettingsApi,
  listSessions as listSessionsApi,
  createSession as createSessionApi,
  renameSession as renameSessionApi,
  deleteSession as deleteSessionApi,
  fetchSessionMessages,
  appendSessionMessage,
  listMemoryEntries,
  addMemoryEntry,
  deleteMemoryEntry,
  searchWeb,
  generateSearchQuery,
  suggestSessionTitle,
  fetchWeather,
  browseUrl,
  fetchTimingLogs,
  thinkingPreprocess,
  thinkingPostprocess,
} from "./api";
import type {
  MemoryEntry as MemoryEntryType,
  SessionSummary,
  WebSearchResult,
  WeatherSummary,
  TimingLogEntry,
} from "./api";
import { useAudioQueue } from "./useAudioQueue";

type Msg = {
  role: "user" | "assistant" | "system";
  content: string;
  sources?: WebSearchResult[];
  latencyMs?: number;
  completedAt?: number;
  workflowStages?: WorkflowStage[];
};
type SearchInsight = {
  query: string;
  status: "searching" | "success" | "error";
  results: WebSearchResult[];
  error?: string;
  timestamp: number;
};
const DEFAULT_SYSTEM_PROMPT =
  "You are Parlance, a friendly real-time voice assistant. Answer in short, conversational sentences that sound natural when spoken.";
const SYSTEM_PROMPT_SUFFIX =
  "State facts directly without mentioning that the information came from system messages or metadata. Avoid Markdown, emoji, or formatting symbols—return plain text suitable for speech.";
const SETTINGS_TABS = [
  "General",
  "Web Search",
  "Memory",
  "Ollama Config",
] as const;
const DIAGNOSTICS_POLL_INTERVAL = 5000;
const DEFAULT_OLLAMA_URL = "http://localhost:11434";
const DEFAULT_SEARX_URL = "http://localhost:8080";
const TITLE_UPDATE_INTERVAL = 3;

type WorkflowStageId = "stt" | "model" | "tts";
type WorkflowStageStatus = "pending" | "running" | "success" | "error";

type WorkflowStage = {
  id: WorkflowStageId;
  label: string;
  status: WorkflowStageStatus;
  durationMs?: number;
  startedAt?: number;
};

const WORKFLOW_STAGE_LABELS: Record<WorkflowStageId, string> = {
  stt: "Speech-to-text",
  model: "Model thinking",
  tts: "Voice playback",
};

type ThinkingStage =
  | "idle"
  | "stt"
  | "pre"
  | "model"
  | "post"
  | "ttsPrep"
  | "ttsPlay";

function normalizeAssistantText(text: string) {
  if (!text) return "";
  let result = text.replace(/\r/g, "");
  result = result.replace(/(\w)\s*'\s*(\w)/g, "$1'$2");
  result = result.replace(/\s+([,!?;:.])/g, "$1");
  result = result.replace(/[ \t]{2,}/g, " ");
  result = result.replace(/ \n/g, "\n").replace(/\n /g, "\n");
  result = result.replace(/\n{3,}/g, "\n\n");
  return result.trimStart();
}

export default function App() {
  const [model, setModel] = useState("qwen2.5:7b-instruct");
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [modelStatus, setModelStatus] = useState<"idle" | "loading" | "error">(
    "loading"
  );
  const [modelHint, setModelHint] = useState("Fetching models…");
  const [modelRefreshKey, setModelRefreshKey] = useState(0);

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [listening, setListening] = useState(false);
  const [pendingResume, setPendingResume] = useState(false);
  const [statusNote, setStatusNote] = useState<string>("");
  const [queuedUtterance, setQueuedUtterance] = useState<string | null>(null);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [ollamaUrl, setOllamaUrl] = useState(
    () => localStorage.getItem("OLLAMA_URL") || DEFAULT_OLLAMA_URL
  );
  const [ollamaDraft, setOllamaDraft] = useState(
    () => localStorage.getItem("OLLAMA_URL") || DEFAULT_OLLAMA_URL
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState(
    () => localStorage.getItem("SYSTEM_PROMPT") || DEFAULT_SYSTEM_PROMPT
  );
  const [systemPromptDraft, setSystemPromptDraft] = useState(
    () => localStorage.getItem("SYSTEM_PROMPT") || DEFAULT_SYSTEM_PROMPT
  );
  const [searxUrl, setSearxUrl] = useState(
    () => localStorage.getItem("SEARX_URL") || DEFAULT_SEARX_URL
  );
  const [searxDraft, setSearxDraft] = useState(
    () => localStorage.getItem("SEARX_URL") || DEFAULT_SEARX_URL
  );
  const [searxAllowInsecure, setSearxAllowInsecure] = useState(
    () => localStorage.getItem("SEARX_INSECURE") === "1"
  );
  const [searxAllowInsecureDraft, setSearxAllowInsecureDraft] = useState(
    () => localStorage.getItem("SEARX_INSECURE") === "1"
  );
  const [settingsTab, setSettingsTab] =
    useState<(typeof SETTINGS_TABS)[number]>("General");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [memoryEntries, setMemoryEntries] = useState<MemoryEntryType[]>([]);
  const [memoryLabel, setMemoryLabel] = useState("");
  const [memoryValue, setMemoryValue] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [timingLogs, setTimingLogs] = useState<TimingLogEntry[]>([]);
  const [timingLoading, setTimingLoading] = useState(false);
  const [timingError, setTimingError] = useState<string | null>(null);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(
    () => localStorage.getItem("WEB_SEARCH_ENABLED") === "1"
  );
  const [searxReachable, setSearxReachable] = useState(() =>
    Boolean(localStorage.getItem("SEARX_URL") || DEFAULT_SEARX_URL)
  );
  const [searchingWeb, setSearchingWeb] = useState(false);
  const [lastSearch, setLastSearch] = useState<SearchInsight | null>(null);
  const [micMuted, setMicMuted] = useState(false);
  const [audioMuted, setAudioMuted] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [audioVolume, setAudioVolume] = useState(() => {
    if (typeof window === "undefined") return 1;
    const stored = window.localStorage.getItem("APP_VOLUME");
    const parsed = Number(stored);
    if (!Number.isFinite(parsed)) return 1;
    return Math.min(1, Math.max(0, parsed));
  });
  const [thinkingEnabled, setThinkingEnabled] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem("THINKING_ENABLED") === "1";
  });
  const [thinkingStatusLine, setThinkingStatusLine] = useState("");
  const [thinkingStage, setThinkingStage] = useState<ThinkingStage>("idle");
  const [browsePanelOpen, setBrowsePanelOpen] = useState(false);
  const [browseUrlDraft, setBrowseUrlDraft] = useState("");
  const [browseLoading, setBrowseLoading] = useState(false);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [browseAttachment, setBrowseAttachment] = useState<{
    url: string;
    title: string;
    summary: string;
    source: string;
  } | null>(null);
  const { enqueue, playing, currentAudio, stop } = useAudioQueue();

  const loadTimingLogs = useCallback(async (withSpinner = false) => {
    if (withSpinner) setTimingLoading(true);
    try {
      const entries = await fetchTimingLogs(100);
      setTimingLogs(entries);
      setTimingError(null);
    } catch (err: any) {
      setTimingError(err?.message || "Failed to load timing logs.");
    } finally {
      if (withSpinner) setTimingLoading(false);
    }
  }, []);

  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const monitorIdRef = useRef<number | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<BlobPart[]>([]);
  const recorderMimeRef = useRef<string>("audio/webm");
  const speechTimeoutRef = useRef<number | null>(null);
  const lastRmsLogRef = useRef<number>(0);
  const speakingRef = useRef(false);
  const silenceStartRef = useRef<number | null>(null);
  const thinkingStatusTimeoutRef = useRef<number | null>(null);
  const thinkingStageTimeoutRef = useRef<number | null>(null);
  const greetingsDoneRef = useRef(false);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const scrollAnchorRef = useRef<HTMLDivElement>(null);
  const listeningRef = useRef(listening);
  const streamingRef = useRef(streaming);
  const playingRef = useRef(playing);
  const sessionActiveRef = useRef(sessionActive);
  const messagesRef = useRef<Msg[]>([]);
  const sessionTitleCountsRef = useRef<Record<string, number>>({});
  const micMutedRef = useRef(micMuted);
  const audioMutedRef = useRef(audioMuted);
  const audioVolumeRef = useRef(audioVolume);
  const voiceEnabledRef = useRef(voiceEnabled);
  const voiceInitRef = useRef(false);
  const pendingWorkflowStagesRef = useRef<WorkflowStage[] | null>(null);

  useEffect(() => {
    if (!scrollAnchorRef.current) return;
    scrollAnchorRef.current.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  useEffect(() => {
    if (modelStatus === "idle" && queuedUtterance) {
      const utterance = queuedUtterance;
      setQueuedUtterance(null);
      setTimeout(() => {
        sendUserText(utterance, { autopilot: true });
      }, 0);
    }
  }, [modelStatus, queuedUtterance]);

  useEffect(() => {
    async function bootstrap() {
      setInitializing(true);
      try {
        const [settingsData, sessionList, memoryList] = await Promise.all([
          fetchSettings().catch(() => ({
            ollamaUrl,
            systemPrompt,
            searxUrl,
            searxAllowInsecure,
          })),
          listSessionsApi().catch(() => []),
          listMemoryEntries().catch(() => []),
        ]);
        if (settingsData?.ollamaUrl) {
          setOllamaUrl(settingsData.ollamaUrl);
          setOllamaDraft(settingsData.ollamaUrl);
        }
        if (settingsData?.systemPrompt) {
          setSystemPrompt(settingsData.systemPrompt);
          setSystemPromptDraft(settingsData.systemPrompt);
        }
        if (settingsData?.searxUrl) {
          setSearxUrl(settingsData.searxUrl);
          setSearxDraft(settingsData.searxUrl);
        }
        if (typeof settingsData?.searxAllowInsecure === "boolean") {
          setSearxAllowInsecure(settingsData.searxAllowInsecure);
          setSearxAllowInsecureDraft(settingsData.searxAllowInsecure);
        }
        setMemoryEntries(memoryList);
        let sessionsToUse = sessionList;
        let sessionId = sessionsToUse[0]?.id ?? null;
        if (!sessionId) {
          try {
            const newSession = await createSessionApi();
            sessionsToUse = [newSession, ...sessionsToUse];
            sessionId = newSession.id;
          } catch (err) {
            console.error("[parlance] failed to create default session", err);
          }
        }
        sessionsToUse.forEach((session) => {
          if (!(session.id in sessionTitleCountsRef.current)) {
            sessionTitleCountsRef.current[session.id] = 0;
          }
        });
        setSessions(sessionsToUse);
        if (sessionId) {
          setActiveSessionId(sessionId);
          await loadSessionMessages(sessionId);
        }
      } catch (err) {
        console.error("[parlance] bootstrap failed", err);
      } finally {
        setInitializing(false);
      }
    }
    bootstrap();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setOllamaDraft(ollamaUrl);
  }, [ollamaUrl]);

  useEffect(() => {
    setSystemPromptDraft(systemPrompt);
  }, [systemPrompt]);

  useEffect(() => {
    setSearxDraft(searxUrl);
  }, [searxUrl]);

  useEffect(() => {
    setSearxReachable(Boolean(searxUrl));
  }, [searxUrl]);

  useEffect(() => {
    setSearxAllowInsecureDraft(searxAllowInsecure);
  }, [searxAllowInsecure]);

  useEffect(() => {
    localStorage.setItem("WEB_SEARCH_ENABLED", webSearchEnabled ? "1" : "0");
    if (!webSearchEnabled) {
      setLastSearch(null);
    }
  }, [webSearchEnabled]);

  useEffect(() => {
    if (!ollamaUrl?.trim()) {
      setModelOptions([]);
      setModelStatus("error");
      setModelHint("Add an Ollama URL to continue.");
      return;
    }
    let cancelled = false;
    async function loadModels() {
      setModelStatus("loading");
      setModelHint("Fetching models…");
      try {
        const names = await listOllamaModels(ollamaUrl);
        if (cancelled) return;
        if (!names.length) {
          setModelOptions([]);
          setModelStatus("error");
          setModelHint("No models available on this Ollama host.");
          return;
        }
        setModelOptions(names);
        setModel((current) => (names.includes(current) ? current : names[0]));
        setModelStatus("idle");
        setModelHint(
          `Loaded ${names.length} model${names.length > 1 ? "s" : ""}.`
        );
      } catch (err) {
        if (cancelled) return;
        console.error(err);
        setModelStatus("error");
        setModelHint("Unable to reach Ollama. Check URL and CORS.");
      }
    }
    loadModels();
    return () => {
      cancelled = true;
    };
  }, [ollamaUrl, modelRefreshKey]);

  const modelStatusDot =
    modelStatus === "loading"
      ? "bg-amber-300 animate-ping"
      : modelStatus === "error"
      ? "bg-red-400"
      : "bg-emerald-400";
  const composerDisabled =
    streaming || !input.trim() || modelStatus !== "idle" || searchingWeb;
  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;
  const sidebarContent = (
    <div className="flex h-full flex-col">
      <div className="sticky top-0 z-10 border-b border-white/5 bg-slate-950/95 px-4 pb-3 pt-4 shadow-[0_6px_14px_rgba(0,0,0,0.45)] backdrop-blur">
        <p className="text-xs uppercase tracking-[0.35em] text-white/50">
          Parlance
        </p>
        <div className="mt-1 flex items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-white">Sessions</h2>
          <button
            type="button"
            onClick={handleNewSession}
            aria-label="Start new conversation"
            className="rounded-2xl border border-white/15 bg-white/5 p-2 text-white/80 transition hover:border-sky-400/60 hover:text-white"
          >
            <PlusIcon />
          </button>
        </div>
      </div>
      <div className="flex-1 space-y-1 overflow-y-auto px-4 pb-20 pt-4">
        {sessions.length === 0 && (
          <div className="rounded-xl border border-dashed border-white/10 bg-white/5 px-3 py-6 text-center text-xs text-white/50">
            No conversations yet.
          </div>
        )}
        {sessions.map((session) => {
          const isActive = session.id === activeSessionId;
          const timeLabel = new Date(session.updatedAt).toLocaleString(
            undefined,
            {
              month: "short",
              day: "numeric",
              hour: "numeric",
              minute: "2-digit",
            }
          );
          return (
            <div
              key={session.id}
              className={`group flex items-center gap-2 rounded-xl px-3 py-2 text-left transition ${
                isActive
                  ? "bg-sky-500/15 text-white"
                  : "text-white/70 hover:bg-white/10"
              }`}
            >
              <button
                type="button"
                className="flex min-w-0 flex-1 flex-col text-left"
                onClick={() => handleSelectSession(session.id)}
              >
                <span className="block max-w-full truncate text-sm font-medium">
                  {session.title}
                </span>
                <span className="text-[11px] text-white/50">{timeLabel}</span>
              </button>
              <button
                type="button"
                aria-label="Delete conversation"
                className="invisible rounded-full p-1 text-white/40 transition group-hover:visible group-hover:text-red-300"
                onClick={(event) => {
                  event.stopPropagation();
                  handleDeleteSession(session.id);
                }}
              >
                <TrashIcon />
              </button>
            </div>
          );
        })}
      </div>
      <div className="sticky bottom-0 border-t border-white/5 bg-slate-950/95 px-4 pb-4 pt-3 space-y-2">
        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          className="flex w-full items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-white/80 hover:border-sky-400/60 hover:bg-sky-500/10 hover:text-white"
        >
          <SettingsIcon />
          Settings
        </button>
        <button
          type="button"
          onClick={() => setDiagnosticsOpen(true)}
          className="flex w-full items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-white/80 hover:border-emerald-400/60 hover:bg-emerald-500/10 hover:text-white"
        >
          <DiagnosticsIcon />
          Diagnostics
        </button>
      </div>
    </div>
  );

  useEffect(() => {
    listeningRef.current = listening;
  }, [listening]);

  useEffect(() => {
    micMutedRef.current = micMuted;
  }, [micMuted]);

  useEffect(() => {
    audioMutedRef.current = audioMuted;
  }, [audioMuted]);

  useEffect(() => {
    voiceEnabledRef.current = voiceEnabled;
  }, [voiceEnabled]);

  useEffect(() => {
    streamingRef.current = streaming;
  }, [streaming]);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    if (playing) {
      announceThinkingStage("ttsPlay", "Reading your response aloud…");
    } else if (thinkingStage === "ttsPlay" && !streaming) {
      clearThinkingStatus();
    }
  }, [playing, thinkingStage, streaming]);

  useEffect(() => {
    if (thinkingStage === "ttsPrep" && !streaming && !playing) {
      setPipelineStage("idle");
    }
  }, [thinkingStage, streaming, playing]);

  useEffect(() => {
    sessionActiveRef.current = sessionActive;
  }, [sessionActive]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if (voiceEnabled) {
      if (!sessionActive && modelStatus === "idle" && !voiceInitRef.current) {
        voiceInitRef.current = true;
        beginSession({ greet: false })
          .catch((err) => {
            console.error("[parlance] voice start failed", err);
            setVoiceEnabled(false);
          })
          .finally(() => {
            voiceInitRef.current = false;
          });
      }
    } else {
      voiceInitRef.current = false;
      stopCurrentAudioPlayback();
      if (sessionActive) {
        stopSession();
      }
    }
  }, [voiceEnabled, sessionActive, modelStatus]);

  useEffect(() => {
    audioVolumeRef.current = audioVolume;
    if (typeof window !== "undefined") {
      window.localStorage.setItem("APP_VOLUME", String(audioVolume));
    }
  }, [audioVolume]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(
        "THINKING_ENABLED",
        thinkingEnabled ? "1" : "0"
      );
    }
    if (!thinkingEnabled) {
      clearThinkingStatus();
    }
  }, [thinkingEnabled]);

  useEffect(() => {
    if (currentAudio.current) {
      currentAudio.current.muted = audioMuted;
      currentAudio.current.volume = audioMuted ? 0 : audioVolume;
    }
  }, [audioMuted, audioVolume, currentAudio]);

  useEffect(() => {
    if (!settingsOpen) return;
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setSettingsOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    setSettingsTab("General");
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [settingsOpen]);

  useEffect(() => {
    if (!diagnosticsOpen) return;
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setDiagnosticsOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [diagnosticsOpen]);

  useEffect(() => {
    if (!diagnosticsOpen) return;
    void loadTimingLogs(true);
    const id = window.setInterval(() => {
      void loadTimingLogs(false);
    }, DIAGNOSTICS_POLL_INTERVAL);
    return () => {
      clearInterval(id);
    };
  }, [diagnosticsOpen, loadTimingLogs]);

  useEffect(() => {
    if (!sessionActive) return;
    if (micMuted) {
      if (listeningRef.current) setListening(false);
      return;
    }
    if (!pendingResume && !listeningRef.current && !playing && !streaming) {
      setListening(true);
    }
  }, [playing, streaming, sessionActive, pendingResume, listening, micMuted]);

  async function loadSessionMessages(sessionId: string) {
    setMessagesLoading(true);
    try {
      const serverMessages = await fetchSessionMessages(sessionId);
      const formatted = serverMessages.map((msg) => {
        const metadata =
          msg.metadata && typeof msg.metadata === "object"
            ? (msg.metadata as Record<string, unknown>)
            : undefined;
        const workflowStages = normalizeWorkflowStagesFromMetadata(
          metadata?.workflowStages
        );
        const latencyMs =
          typeof metadata?.latencyMs === "number"
            ? metadata.latencyMs
            : undefined;
        const completedAt =
          typeof metadata?.completedAt === "number"
            ? metadata.completedAt
            : undefined;
        return {
          role: msg.role === "system" ? "assistant" : msg.role,
          content: msg.content,
          workflowStages,
          latencyMs,
          completedAt,
        } as Msg;
      });
      setMessages(formatted);
    } catch (err) {
      console.error("[parlance] loadSessionMessages failed", err);
      setMessages([]);
    } finally {
      setMessagesLoading(false);
    }
  }

  async function ensureActiveSession(): Promise<string> {
    if (activeSessionId) return activeSessionId;
    const session = await createSessionApi();
    sessionTitleCountsRef.current[session.id] = 0;
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    setMessages([]);
    return session.id;
  }

  async function maybeAutoTitle(sessionId: string) {
    const counts = sessionTitleCountsRef.current;
    counts[sessionId] = (counts[sessionId] || 0) + 1;
    if (counts[sessionId] % TITLE_UPDATE_INTERVAL !== 0) return;
    await refreshSessionTitle(sessionId);
  }

  async function refreshSessionTitle(sessionId: string) {
    const session = sessions.find((s) => s.id === sessionId);
    if (!session || !shouldAutoRenameSession(session)) return;
    if (sessionId !== activeSessionId) return;
    const summaryConversation = buildTitleConversation(messagesRef.current);
    if (summaryConversation.length < 2) return;
    let suggestion = "";
    try {
      suggestion = await suggestSessionTitle(sessionId, {
        conversation: summaryConversation,
        model,
        ollamaUrl,
      });
    } catch (err) {
      console.error("[parlance] suggest session title failed", err);
    }
    if (!suggestion) {
      suggestion = deriveTitleFromText(
        messagesRef.current[messagesRef.current.length - 1]?.content || ""
      );
    }
    suggestion = cleanTitleSuggestion(suggestion);
    if (!suggestion || session.title === suggestion) return;
    try {
      const updated = await renameSessionApi(sessionId, suggestion);
      setSessions((prev) => {
        const others = prev.filter((s) => s.id !== updated.id);
        return [updated, ...others].sort(
          (a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)
        );
      });
    } catch (err) {
      console.error("[parlance] auto title failed", err);
    }
  }

  function shouldAutoRenameSession(session: SessionSummary) {
    const defaultTitles = [
      "New Conversation",
      "Conversation console",
      "Conversation",
    ];
    if (!session.title) return true;
    const normalized = session.title.toLowerCase();
    return (
      defaultTitles.includes(session.title) ||
      normalized.startsWith("new conversation")
    );
  }

  function touchSession(sessionId: string, title?: string) {
    setSessions((prev) => {
      const next = prev.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              title: title ?? session.title,
              updatedAt: new Date().toISOString(),
            }
          : session
      );
      return next.sort(
        (a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)
      );
    });
  }

  async function handleNewSession() {
    try {
      const session = await createSessionApi();
      sessionTitleCountsRef.current[session.id] = 0;
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setMessages([]);
      setSidebarOpen(false);
    } catch (err) {
      console.error("[parlance] create session failed", err);
    }
  }

  async function handleSelectSession(sessionId: string) {
    if (sessionId === activeSessionId) {
      setSidebarOpen(false);
      return;
    }
    stopCurrentAudioPlayback();
    setActiveSessionId(sessionId);
    setSidebarOpen(false);
    await loadSessionMessages(sessionId);
  }

  async function handleRenameSession(session: SessionSummary) {
    const title = prompt("Rename conversation", session.title);
    if (!title || title.trim() === session.title) return;
    try {
      const updated = await renameSessionApi(session.id, title.trim());
      setSessions((prev) => {
        const others = prev.filter((s) => s.id !== session.id);
        return [updated, ...others].sort(
          (a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)
        );
      });
    } catch (err) {
      console.error("[parlance] rename session failed", err);
    }
  }

  async function handleDeleteSession(sessionId: string) {
    if (!confirm("Delete this conversation?")) return;
    try {
      await deleteSessionApi(sessionId);
      const remaining = sessions.filter((s) => s.id !== sessionId);
      setSessions(remaining);
      delete sessionTitleCountsRef.current[sessionId];
      if (activeSessionId === sessionId) {
        const nextSession = remaining[0];
        if (nextSession) {
          setActiveSessionId(nextSession.id);
          await loadSessionMessages(nextSession.id);
        } else {
          await handleNewSession();
        }
      }
    } catch (err) {
      console.error("[parlance] delete session failed", err);
    }
  }

  async function handleAddMemoryEntry(event?: React.FormEvent) {
    event?.preventDefault();
    const label = memoryLabel.trim();
    const value = memoryValue.trim();
    if (!label || !value) return;
    try {
      const entry = await addMemoryEntry(label, value);
      setMemoryEntries((prev) => [entry, ...prev]);
      setMemoryLabel("");
      setMemoryValue("");
    } catch (err) {
      console.error("[parlance] add memory failed", err);
    }
  }

  async function handleDeleteMemoryEntry(id: string) {
    try {
      await deleteMemoryEntry(id);
      setMemoryEntries((prev) => prev.filter((entry) => entry.id !== id));
    } catch (err) {
      console.error("[parlance] delete memory failed", err);
    }
  }

  function toggleMicMute() {
    setMicMuted((prev) => {
      const next = !prev;
      if (next) {
        setListening(false);
        setStatusNote("Microphone muted");
      } else if (sessionActive && !streaming && !playing) {
        setStatusNote("Listening…");
        setListening(true);
      }
      return next;
    });
  }

  function toggleAudioMute() {
    setAudioMuted((prev) => !prev);
  }

  function handleVolumeChange(value: number) {
    const clamped = Math.min(1, Math.max(0, value));
    setAudioVolume(clamped);
    if (currentAudio.current) {
      currentAudio.current.volume = audioMuted ? 0 : clamped;
    }
  }

  function toggleVoiceEnabled() {
    setVoiceEnabled((prev) => !prev);
  }

  function stopCurrentAudioPlayback() {
    stop();
  }

  async function handleBrowseFetch(event?: React.FormEvent) {
    event?.preventDefault();
    const targetUrl = browseUrlDraft.trim();
    if (!targetUrl) {
      setBrowseError("Enter a URL to load.");
      return;
    }
    setBrowseLoading(true);
    setBrowseError(null);
    try {
      const result = await browseUrl(targetUrl, 2000);
      setBrowseAttachment(result);
      setBrowsePanelOpen(false);
      setStatusNote(`Attached ${result.title} for reference.`);
    } catch (err: any) {
      setBrowseError(err?.message || "Failed to load page.");
    } finally {
      setBrowseLoading(false);
    }
  }

  function clearBrowseAttachment() {
    setBrowseAttachment(null);
    setBrowseError(null);
  }

  function refreshTimingLogs() {
    void loadTimingLogs(true);
  }

  function setThinkingStatus(text: string, autoClearMs?: number) {
    setThinkingStatusLine(text);
    if (thinkingStatusTimeoutRef.current) {
      clearTimeout(thinkingStatusTimeoutRef.current);
      thinkingStatusTimeoutRef.current = null;
    }
    if (text && autoClearMs && autoClearMs > 0) {
      thinkingStatusTimeoutRef.current = window.setTimeout(() => {
        setThinkingStatusLine("");
        thinkingStatusTimeoutRef.current = null;
      }, autoClearMs);
    }
  }

  function clearThinkingStatus() {
    setPipelineStage("idle");
    setThinkingStatus("");
  }

  function setPipelineStage(stage: ThinkingStage, autoClearMs?: number) {
    setThinkingStage(stage);
    if (thinkingStageTimeoutRef.current) {
      clearTimeout(thinkingStageTimeoutRef.current);
      thinkingStageTimeoutRef.current = null;
    }
    if (autoClearMs && autoClearMs > 0) {
      thinkingStageTimeoutRef.current = window.setTimeout(() => {
        setThinkingStage("idle");
        thinkingStageTimeoutRef.current = null;
      }, autoClearMs);
    }
  }

  function announceThinkingStage(
    stage: ThinkingStage,
    text: string,
    autoClearMs?: number
  ) {
    setPipelineStage(stage, autoClearMs);
    if (!thinkingEnabled) return;
    setThinkingStatus(text, autoClearMs);
  }

  function updateMessageAt(index: number, updater: (message: Msg) => Msg) {
    setMessages((prev) => {
      if (!prev[index]) return prev;
      const copy = [...prev];
      const next = updater(copy[index]);
      copy[index] = next;
      return copy;
    });
  }

  function handleComposerKey(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!composerDisabled) sendUserText(input);
    }
  }

  async function sendUserText(text: string, opts?: { autopilot?: boolean }) {
    const trimmed = text.trim();
    if (!trimmed) return;
    if (modelStatus !== "idle") {
      if (opts?.autopilot) {
        setQueuedUtterance(trimmed);
        setStatusNote("Preparing model…");
        console.debug("[parlance] queued utterance while model loading");
      } else {
        alert("Model list is still loading. Please wait a moment.");
      }
      return;
    }
    console.debug("[parlance] sending text", trimmed);
    const sessionId = await ensureActiveSession();
    const userMsg = { role: "user", content: trimmed } as Msg;
    const baseMessages = messagesRef.current;
    const newMessages: Msg[] = [...baseMessages, userMsg];
    setMessages(newMessages);
    setInput("");
    const workflowStages = cloneWorkflowStages(
      pendingWorkflowStagesRef.current ??
        createWorkflowStages(false, voiceEnabledRef.current)
    );
    pendingWorkflowStagesRef.current = null;

    let searchContext: string | undefined;
    let searchSources: WebSearchResult[] | undefined;
    let weatherContext: string | undefined;
    const browseContext = browseAttachment
      ? formatBrowseContext(browseAttachment)
      : undefined;
    let searchQueryUsed = trimmed;
    const needsWeather = shouldUseWeatherService(trimmed);
    const searchActive = !needsWeather && webSearchEnabled;

    if (needsWeather) {
      try {
        setStatusNote("Fetching live weather data…");
        const weather = await fetchWeather(trimmed);
        weatherContext = formatWeatherContext(weather);
        setStatusNote(
          `Weather updated for ${weather.location.name}, ${
            weather.location.country || ""
          }`.trim()
        );
      } catch (err) {
        console.error("[parlance] weather lookup failed", err);
        setStatusNote(
          "Weather lookup failed. I'll answer with general knowledge."
        );
      }
    }

    let conversationForSearch: {
      role: "user" | "assistant";
      content: string;
    }[] = [];
    if (searchActive) {
      conversationForSearch = buildSearchConversation(messagesRef.current);
    }
    if (searchActive) {
      setSearchingWeb(true);
      try {
        const generatedQuery = await generateSearchQuery({
          conversation: conversationForSearch,
          latestUserMessage: trimmed,
          model,
          ollamaUrl,
        });
        if (generatedQuery && generatedQuery.trim()) {
          searchQueryUsed = generatedQuery.trim();
        }
      } catch (err) {
        console.error("[parlance] search query generation failed", err);
      }
      const timestamp = Date.now();
      setLastSearch({
        query: searchQueryUsed,
        status: "searching",
        results: [],
        timestamp,
      });
      try {
        const results = await searchWeb(searchQueryUsed, {
          limit: 8,
          searxUrl,
          allowInsecure: searxAllowInsecure,
        });
        const curated = curateSearchResults(searchQueryUsed, results);
        if (curated.length) {
          searchContext = formatSearchContext(trimmed, curated);
          searchSources = curated;
        } else {
          searchContext = describeEmptySearchContext(trimmed);
        }
        setLastSearch({
          query: searchQueryUsed,
          status: "success",
          results: curated,
          timestamp: Date.now(),
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Web search failed";
        console.error("[parlance] web search failed", err);
        setSearxReachable(false);
        setWebSearchEnabled(false);
        setLastSearch({
          query: searchQueryUsed,
          status: "error",
          results: [],
          error: message,
          timestamp: Date.now(),
        });
        searchContext = describeSearchFailureContext(trimmed, message);
      } finally {
        setSearchingWeb(false);
      }
    }
    appendSessionMessage(sessionId, "user", trimmed).catch((err) =>
      console.error("[parlance] append user message failed", err)
    );
    void maybeAutoTitle(sessionId);
    touchSession(sessionId);
    let thinkingPromptRewrite: string | undefined;
    let thinkingTurnActive = false;
    if (thinkingEnabled) {
      announceThinkingStage("pre", "Thinking: understanding your request…");
      try {
        const plan = await thinkingPreprocess({
          userQuery: trimmed,
          model,
          ollamaUrl,
        });
        thinkingPromptRewrite = plan.refinedQuery?.trim();
        thinkingTurnActive = Boolean(thinkingPromptRewrite);
        if (plan.summary) {
          announceThinkingStage("pre", plan.summary);
        } else if (thinkingTurnActive) {
          announceThinkingStage("pre", "Thinking: clarified intent.");
        }
      } catch (err: any) {
        console.error("[parlance] thinking preprocess failed", err);
        announceThinkingStage(
          "model",
          err?.message
            ? `Thinking paused: ${err.message}`
            : "Thinking mode unavailable—continuing normally.",
          4000
        );
        thinkingPromptRewrite = undefined;
        thinkingTurnActive = false;
      }
    }
    const requestStartedAt = performance.now();
    const assistantResult = await streamAssistant(newMessages, sessionId, {
      searchContext,
      sources: searchSources,
      weatherContext,
      browseContext,
      workflowStages,
      requestStartedAt,
      thinkingPromptRewrite: thinkingTurnActive
        ? thinkingPromptRewrite
        : undefined,
      skipPersistence: thinkingTurnActive,
    });
    if (!thinkingTurnActive) {
      setPipelineStage("idle");
      return;
    }
    const assistantText = assistantResult?.text ?? "";
    const assistantIndex =
      typeof assistantResult?.messageIndex === "number"
        ? assistantResult.messageIndex
        : -1;
    const assistantMetadata = assistantResult?.metadata;
    if (!assistantText || assistantIndex < 0) {
      clearThinkingStatus();
      return;
    }
    announceThinkingStage("post", "Thinking: polishing response…");
    let finalReply = assistantText;
    try {
      const polish = await thinkingPostprocess({
        userQuery: trimmed,
        refinedQuery: thinkingPromptRewrite || trimmed,
        assistantDraft: assistantText,
        model,
        ollamaUrl,
      });
      if (polish.improvedReply && polish.improvedReply.trim()) {
        finalReply = polish.improvedReply.trim();
      }
      if (polish.summary) {
        announceThinkingStage("post", polish.summary, 4000);
      } else {
        announceThinkingStage("post", "Thinking complete.", 4000);
      }
    } catch (err: any) {
      console.error("[parlance] thinking postprocess failed", err);
      announceThinkingStage(
        "post",
        err?.message
          ? `Thinking polish failed: ${err.message}`
          : "Thinking polish failed. Showing first draft.",
        4000
      );
    }
    updateMessageAt(assistantIndex, (msg) => ({ ...msg, content: finalReply }));
    appendSessionMessage(
      sessionId,
      "assistant",
      finalReply,
      assistantMetadata
    ).catch((err) => console.error("[parlance] append assistant failed", err));
    touchSession(sessionId);
  }

  async function streamAssistant(
    history: Msg[],
    sessionId: string,
    options?: {
      prompt?: string;
      autoResume?: boolean;
      searchContext?: string;
      sources?: WebSearchResult[];
      weatherContext?: string;
      browseContext?: string;
      workflowStages?: WorkflowStage[];
      requestStartedAt?: number;
      thinkingPromptRewrite?: string;
      skipPersistence?: boolean;
    }
  ): Promise<{
    text: string;
    messageIndex: number;
    metadata?: Record<string, unknown>;
  } | null> {
    const responseStartedAt = options?.requestStartedAt ?? performance.now();
    setStreaming(true);
    setListening(false);
    const initialWorkflowStages = cloneWorkflowStages(
      options?.workflowStages ??
        createWorkflowStages(false, voiceEnabledRef.current)
    );
    let metadataForPersistence: Record<string, unknown> | undefined;
    const assistantMessageIndex = history.length;
    const acc: Msg[] = [
      ...history,
      { role: "assistant", content: "", workflowStages: initialWorkflowStages },
    ];
    setMessages(acc);
    let currentWorkflowStages = initialWorkflowStages;
    const syncWorkflowStages = (next: WorkflowStage[]) => {
      currentWorkflowStages = next;
      setMessages((prev) => {
        if (!prev.length) return prev;
        const copy = [...prev];
        const lastIndex = copy.length - 1;
        const last = copy[lastIndex];
        if (!last || last.role !== "assistant") return prev;
        copy[lastIndex] = { ...last, workflowStages: next };
        return copy;
      });
    };
    const mutateWorkflowStages = (
      updater: (stages: WorkflowStage[]) => WorkflowStage[]
    ) => {
      const next = updater(cloneWorkflowStages(currentWorkflowStages));
      syncWorkflowStages(next);
    };
    const getStageStatus = (id: WorkflowStageId) =>
      currentWorkflowStages.find((stage) => stage.id === id)?.status;
    mutateWorkflowStages((stages) => startWorkflowStage(stages, "model"));
    announceThinkingStage("model", "Model is thinking…");
    let assistantRaw = "";
    const conversation: Msg[] = [...history];
    if (options?.prompt) {
      conversation.push({ role: "user", content: options.prompt });
    }
    if (options?.thinkingPromptRewrite) {
      for (let i = conversation.length - 1; i >= 0; i--) {
        if (conversation[i].role === "user") {
          conversation[i] = {
            ...conversation[i],
            content: options.thinkingPromptRewrite,
          };
          break;
        }
      }
    }
    const lastUserMessage = [...conversation]
      .reverse()
      .find((message) => message.role === "user");
    const includeDateTime = lastUserMessage
      ? userRequestedDateTime(lastUserMessage.content)
      : false;
    const payloadMessages: { role: Msg["role"]; content: string }[] = [
      {
        role: "system",
        content: systemPrompt || DEFAULT_SYSTEM_PROMPT,
      },
      { role: "system", content: SYSTEM_PROMPT_SUFFIX },
    ];
    if (memoryEntries.length) {
      payloadMessages.push({
        role: "system",
        content:
          "Long-term memory about the user:\n" +
          memoryEntries
            .map((entry) => `- ${entry.label}: ${entry.value}`)
            .join("\n"),
      });
    }
    if (includeDateTime) {
      payloadMessages.push({
        role: "system",
        content: describeCurrentDateTime(),
      });
    }
    if (options?.weatherContext) {
      payloadMessages.push({
        role: "system",
        content: "[live-data] weather\n" + options.weatherContext,
      });
    }
    if (options?.browseContext) {
      payloadMessages.push({
        role: "system",
        content: "[live-data] web\n" + options.browseContext,
      });
    }
    if (options?.searchContext) {
      payloadMessages.push({
        role: "system",
        content: options.searchContext,
      });
    }
    payloadMessages.push(
      ...conversation.map((entry) => ({
        role: entry.role,
        content: entry.content,
      }))
    );
    const payload = {
      model,
      ollama_url: ollamaUrl,
      messages: payloadMessages,
    };
    let finalText = "";
    let latencyMs = 0;
    let completedAt = 0;
    try {
      for await (const chunk of chatStream(payload)) {
        if (chunk.token) {
          assistantRaw += chunk.token;
          const formatted = normalizeAssistantText(assistantRaw);
          setMessages((prev) => {
            const copy = [...prev];
            const current = copy[copy.length - 1] || {
              role: "assistant",
              content: "",
            };
            copy[copy.length - 1] = { ...current, content: formatted };
            return copy;
          });
        }
      }
      finalText = normalizeAssistantText(assistantRaw);
      latencyMs = Math.max(0, performance.now() - responseStartedAt);
      completedAt = Date.now();
      mutateWorkflowStages((stages) =>
        completeWorkflowStage(stages, "model", "success")
      );
      setMessages((prev) => {
        const copy = [...prev];
        const current = copy[copy.length - 1] || {
          role: "assistant",
          content: "",
        };
        copy[copy.length - 1] = {
          ...current,
          content: finalText,
          sources:
            options?.sources && options.sources.length
              ? options.sources
              : undefined,
          latencyMs,
          completedAt,
        };
        return copy;
      });
      const voiceActive = voiceEnabledRef.current;
      if (finalText) {
        let ttsError: unknown = null;
        if (voiceActive) {
          announceThinkingStage("ttsPrep", "Preparing your voice reply…");
          mutateWorkflowStages((stages) => startWorkflowStage(stages, "tts"));
          try {
            const speechText = sanitizeForSpeech(finalText);
            const audio = await tts(speechText);
            mutateWorkflowStages((stages) =>
              completeWorkflowStage(stages, "tts", "success")
            );
            audio.muted = audioMutedRef.current;
            audio.volume = audioMutedRef.current ? 0 : audioVolumeRef.current;
            enqueue(audio);
            setPendingResume(true);
            announceThinkingStage("ttsPlay", "Reading your response aloud…");
          } catch (ttsErr) {
            mutateWorkflowStages((stages) =>
              completeWorkflowStage(stages, "tts", "error")
            );
            ttsError = ttsErr;
          }
        } else {
          mutateWorkflowStages((stages) => removeWorkflowStage(stages, "tts"));
          setPipelineStage("idle");
        }
        const workflowMetadata = sanitizeWorkflowStagesForStorage(
          currentWorkflowStages
        );
        const hasLatency = typeof latencyMs === "number";
        const metadata =
          workflowMetadata || hasLatency
            ? {
                ...(hasLatency ? { latencyMs } : {}),
                ...(typeof completedAt === "number" ? { completedAt } : {}),
                ...(workflowMetadata
                  ? { workflowStages: workflowMetadata }
                  : {}),
              }
            : undefined;
        metadataForPersistence = metadata;
        if (!options?.skipPersistence) {
          appendSessionMessage(
            sessionId,
            "assistant",
            finalText,
            metadata
          ).catch((err) =>
            console.error("[parlance] append assistant failed", err)
          );
          touchSession(sessionId);
        }
        if (lastUserMessage?.content && finalText) {
          // thinking analysis handled elsewhere now
        }
        if (ttsError) throw ttsError;
      } else {
        if (voiceActive) {
          mutateWorkflowStages((stages) =>
            completeWorkflowStage(stages, "tts", "success")
          );
          setListening(true);
        } else {
          mutateWorkflowStages((stages) => removeWorkflowStage(stages, "tts"));
          setPipelineStage("idle");
        }
      }
    } catch (e) {
      if (getStageStatus("model") !== "success") {
        mutateWorkflowStages((stages) =>
          completeWorkflowStage(stages, "model", "error")
        );
      }
      console.error(e);
      alert("Chat failed. Check server and Ollama URL.");
      setPipelineStage("idle");
    } finally {
      setStreaming(false);
    }
    if (finalText) {
      return {
        text: finalText,
        messageIndex: assistantMessageIndex,
        metadata: metadataForPersistence,
      };
    }
    return null;
  }

  async function persistSettings(
    nextUrl: string,
    nextPrompt: string,
    nextSearchUrl: string,
    nextSearchAllowInsecure: boolean
  ) {
    try {
      const saved = await updateSettingsApi({
        ollamaUrl: nextUrl,
        systemPrompt: nextPrompt,
        searxUrl: nextSearchUrl,
        searxAllowInsecure: nextSearchAllowInsecure,
      });
      setOllamaUrl(saved.ollamaUrl);
      setOllamaDraft(saved.ollamaUrl);
      setSystemPrompt(saved.systemPrompt);
      setSystemPromptDraft(saved.systemPrompt);
      setSearxUrl(saved.searxUrl);
      setSearxDraft(saved.searxUrl);
      setSearxAllowInsecure(saved.searxAllowInsecure);
      setSearxAllowInsecureDraft(saved.searxAllowInsecure);
      localStorage.setItem("OLLAMA_URL", saved.ollamaUrl);
      localStorage.setItem("SYSTEM_PROMPT", saved.systemPrompt);
      localStorage.setItem("SEARX_URL", saved.searxUrl);
      localStorage.setItem(
        "SEARX_INSECURE",
        saved.searxAllowInsecure ? "1" : "0"
      );
      setModelRefreshKey((key) => key + 1);
    } catch (err) {
      console.error("[parlance] persist settings failed", err);
      alert("Failed to save settings.");
    }
  }

  async function saveOllamaSettings() {
    const normalized = ollamaDraft.trim();
    if (!normalized) return;
    await persistSettings(
      normalized,
      systemPrompt,
      searxUrl,
      searxAllowInsecure
    );
  }

  async function saveSystemPrompt() {
    const normalized = systemPromptDraft.trim() || DEFAULT_SYSTEM_PROMPT;
    await persistSettings(ollamaUrl, normalized, searxUrl, searxAllowInsecure);
  }

  async function resetSystemPrompt() {
    await persistSettings(
      ollamaUrl,
      DEFAULT_SYSTEM_PROMPT,
      searxUrl,
      searxAllowInsecure
    );
  }

  async function saveSearxSettings() {
    const normalized = searxDraft.trim() || DEFAULT_SEARX_URL;
    await persistSettings(
      ollamaUrl,
      systemPrompt,
      normalized,
      searxAllowInsecureDraft
    );
  }

  async function resetSearxSettings() {
    await persistSettings(ollamaUrl, systemPrompt, DEFAULT_SEARX_URL, false);
  }

  async function beginSession(options?: { greet?: boolean }) {
    if (sessionActive) return;
    const shouldGreet = Boolean(options?.greet);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      console.info(
        "[parlance] mic stream granted",
        stream.getAudioTracks().map((track) => track.label || "mic")
      );
      const ctx = new AudioContext();
      await ctx.resume();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      const source = ctx.createMediaStreamSource(stream);
      source.connect(analyser);
      audioCtxRef.current = ctx;
      analyserRef.current = analyser;
      if (!voiceEnabledRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        analyserRef.current = null;
        audioCtxRef.current?.close();
        audioCtxRef.current = null;
        return;
      }
      setSessionActive(true);
      setStatusNote(shouldGreet ? "Initializing conversation…" : "Listening…");
      greetingsDoneRef.current = shouldGreet ? false : true;
      startMonitor();
      if (shouldGreet) {
        await startGreeting();
      } else {
        setListening(true);
      }
    } catch (err: any) {
      alert("Microphone access is required to start.");
      console.error("[parlance] beginSession failed", err);
    }
  }

  function stopSession() {
    stopCurrentAudioPlayback();
    setSessionActive(false);
    setListening(false);
    setStatusNote("");
    setMicMuted(false);
    speakingRef.current = false;
    silenceStartRef.current = null;
    recorderRef.current?.stop();
    recorderRef.current = null;
    if (monitorIdRef.current) cancelAnimationFrame(monitorIdRef.current);
    monitorIdRef.current = null;
    analyserRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  async function startGreeting() {
    if (greetingsDoneRef.current) return;
    greetingsDoneRef.current = true;
    const sessionId = await ensureActiveSession();
    const history = [...messagesRef.current];
    await streamAssistant(history, sessionId, {
      prompt:
        "Greet the user warmly, explain that you'll handle the conversation hands-free, and invite them to start speaking whenever they're ready.",
      autoResume: true,
      requestStartedAt: performance.now(),
    });
    setStatusNote("Listening…");
    setListening(true);
  }

  function startMonitor() {
    if (!analyserRef.current) return;
    const analyser = analyserRef.current;
    console.info("[parlance] monitor started");
    const SPEECH_TRIGGER = 0.028;
    const SILENCE_TRIGGER = 0.012;
    const SILENCE_DURATION = 600;
    const dataArray = new Uint8Array(analyser.fftSize);
    const tick = () => {
      if (!analyserRef.current) return;
      analyser.getByteTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const value = (dataArray[i] - 128) / 128;
        sum += value * value;
      }
      const rms = Math.sqrt(sum / dataArray.length);
      const now = performance.now();
      if (rms > SPEECH_TRIGGER && now - lastRmsLogRef.current > 500) {
        lastRmsLogRef.current = now;
        console.debug("[parlance] rms trigger", rms.toFixed(4), {
          listening: listeningRef.current,
          speaking: speakingRef.current,
          streaming: streamingRef.current,
          playing: playingRef.current,
        });
      }
      const assistantBusy = streamingRef.current || playingRef.current;
      const sessionLive = sessionActiveRef.current;
      const micSuppressed = micMutedRef.current;

      if (sessionLive && micSuppressed && listeningRef.current) {
        setListening(false);
      } else if (
        sessionLive &&
        !micSuppressed &&
        listeningRef.current === assistantBusy
      ) {
        setListening(!assistantBusy);
        if (!assistantBusy) setStatusNote("Listening…");
      }

      if (
        sessionLive &&
        !assistantBusy &&
        !micSuppressed &&
        rms > SPEECH_TRIGGER &&
        !speakingRef.current &&
        !recorderRef.current
      ) {
        handleSpeechStart();
        silenceStartRef.current = null;
      } else if (speakingRef.current) {
        const shouldStop = assistantBusy || rms < SILENCE_TRIGGER;
        if (shouldStop) {
          if (!silenceStartRef.current) {
            silenceStartRef.current = performance.now();
          } else if (
            performance.now() - silenceStartRef.current >
            SILENCE_DURATION
          ) {
            handleSpeechEnd();
          }
        } else {
          silenceStartRef.current = null;
        }
      } else {
        silenceStartRef.current = null;
      }

      if (!sessionLive && speakingRef.current) {
        speakingRef.current = false;
        setUserSpeaking(false);
        if (recorderRef.current) {
          recorderRef.current.stop();
          recorderRef.current = null;
        }
      }

      if (!sessionLive && speakingRef.current) {
        speakingRef.current = false;
        setUserSpeaking(false);
        if (recorderRef.current) {
          recorderRef.current.stop();
          recorderRef.current = null;
        }
      }
      monitorIdRef.current = requestAnimationFrame(tick);
    };
    monitorIdRef.current = requestAnimationFrame(tick);
  }

  function clearSpeechTimeout() {
    if (speechTimeoutRef.current) {
      clearTimeout(speechTimeoutRef.current);
      speechTimeoutRef.current = null;
    }
  }

  function handleSpeechStart() {
    if (micMutedRef.current) {
      console.debug("[parlance] speech ignored because mic is muted");
      return;
    }
    if (speakingRef.current || !streamRef.current || recorderRef.current) {
      if (!speakingRef.current) {
        console.debug("[parlance] speech blocked", {
          existingRecorder: !!recorderRef.current,
          hasStream: !!streamRef.current,
        });
      }
      return;
    }
    if (!listeningRef.current) {
      console.debug("[parlance] speech ignored because not listening");
      return;
    }
    speakingRef.current = true;
    setUserSpeaking(true);
    console.debug("[parlance] speech detected → recording");
    recordingChunksRef.current = [];
    const options = selectRecorderOptions();
    const mimeType = options?.mimeType || "audio/webm";
    recorderMimeRef.current = mimeType;
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(streamRef.current, { mimeType });
    } catch (err) {
      console.error("[parlance] MediaRecorder init failed", err);
      speakingRef.current = false;
      setUserSpeaking(false);
      setStatusNote("Microphone error — check browser permissions.");
      return;
    }
    console.info("[parlance] recorder started", recorder.mimeType || mimeType);
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordingChunksRef.current.push(event.data);
        console.debug("[parlance] chunk", event.data.size, "bytes");
      } else {
        console.debug("[parlance] empty chunk");
      }
    };
    recorder.onstop = () => {
      console.debug(
        "[parlance] recorder stopped",
        recordingChunksRef.current.length,
        "chunks"
      );
      recorderRef.current = null;
      processUtterance(recordingChunksRef.current);
    };
    recorderRef.current = recorder;
    recorder.start();
    speechTimeoutRef.current = window.setTimeout(() => {
      if (speakingRef.current) handleSpeechEnd();
    }, 6000);
    setStatusNote("Hearing you…");
  }

  function handleSpeechEnd() {
    if (!speakingRef.current) return;
    speakingRef.current = false;
    setUserSpeaking(false);
    silenceStartRef.current = null;
    recorderRef.current?.stop();
    console.debug("[parlance] speech ended → stopping recorder");
    setStatusNote("Processing your speech…");
    clearSpeechTimeout();
  }

  async function processUtterance(chunks: BlobPart[]) {
    if (!chunks.length) {
      setListening(true);
      setStatusNote("");
      setUserSpeaking(false);
      setPipelineStage("idle");
      return;
    }
    const blob = new Blob(chunks, { type: recorderMimeRef.current });
    console.info("[parlance] captured utterance", {
      size: blob.size,
      type: blob.type,
      chunks: chunks.length,
    });
    recordingChunksRef.current = [];
    if (blob.size < 200) {
      console.warn("[parlance] blob too small, skipping");
      setStatusNote("Say a bit more so I can capture it.");
      setListening(true);
      setUserSpeaking(false);
      setPipelineStage("idle");
      return;
    }
    let workflowStages: WorkflowStage[] | null = null;
    try {
      setListening(false);
      setStatusNote("Transcribing…");
      setTranscribing(true);
      announceThinkingStage("stt", "STT: transcribing speech…");
      workflowStages = createWorkflowStages(true, voiceEnabledRef.current);
      workflowStages = startWorkflowStage(workflowStages, "stt");
      pendingWorkflowStagesRef.current = workflowStages;
      const { text } = await transcribeAudio(blob);
      workflowStages = completeWorkflowStage(workflowStages, "stt", "success");
      pendingWorkflowStagesRef.current = workflowStages;
      console.info("[parlance] transcription result", text);
      setTranscribing(false);
      const trimmed = text?.trim() ?? "";
      if (trimmed) {
        announceThinkingStage("pre", "Thinking: preparing response…");
        const cleaned = cleanTranscriptionText(trimmed);
        if (cleaned !== trimmed) {
          console.debug("[parlance] cleaned transcription", {
            raw: trimmed,
            cleaned,
          });
        }
        setStatusNote("Thinking…");
        await sendUserText(cleaned, { autopilot: true });
      } else {
        setStatusNote("Didn't catch that—try again.");
        setListening(true);
        setPipelineStage("idle");
      }
    } catch (err: any) {
      workflowStages = workflowStages
        ? completeWorkflowStage(workflowStages, "stt", "error")
        : null;
      pendingWorkflowStagesRef.current = workflowStages;
      setTranscribing(false);
      setStatusNote(err?.message || "Transcription failed.");
      setListening(true);
      announceThinkingStage(
        "stt",
        err?.message || "STT failed. Try again.",
        4000
      );
      setPipelineStage("idle");
    }
  }

  useEffect(() => {
    if (sessionActive && pendingResume && !playing && !streaming) {
      setPendingResume(false);
      setStatusNote("Listening…");
      setListening(true);
    }
  }, [pendingResume, playing, streaming, sessionActive]);

  useEffect(() => {
    return () => {
      stopSession();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    return () => {
      if (thinkingStatusTimeoutRef.current) {
        clearTimeout(thinkingStatusTimeoutRef.current);
      }
      if (thinkingStageTimeoutRef.current) {
        clearTimeout(thinkingStageTimeoutRef.current);
      }
    };
  }, []);

  const conversationPaneClass = `flex flex-1 min-h-0 h-full flex-col px-2 xl:px-20 ${
    voiceEnabled ? "" : "lg:pr-10"
  }`;
  const pipelineStageText = (() => {
    switch (thinkingStage) {
      case "stt":
        return "Transcribing speech…";
      case "pre":
        return "Thinking through your request…";
      case "model":
        return "Model is thinking…";
      case "post":
        return "Polishing response…";
      case "ttsPrep":
        return "Preparing your voice reply…";
      case "ttsPlay":
        return "Reading your response aloud…";
      default:
        return "";
    }
  })();
  const conversationIndicatorText =
    thinkingStatusLine ||
    pipelineStageText ||
    (transcribing
      ? "Transcribing speech…"
      : streaming
      ? "Model is thinking…"
      : playing
      ? "Reading your response aloud…"
      : "");
  const conversationIndicatorActive = Boolean(conversationIndicatorText);

  const requiresOllamaSetup =
    modelStatus !== "loading" &&
    (!ollamaUrl?.trim() || modelOptions.length === 0);

  if (requiresOllamaSetup) {
    return (
      <OllamaSetupScreen
        value={ollamaDraft}
        onChange={setOllamaDraft}
        onSave={() => {
          void saveOllamaSettings();
        }}
        hint={modelHint}
      />
    );
  }

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">
      <aside className="hidden w-72 border-r border-white/10 bg-slate-950/70 lg:flex lg:flex-col lg:sticky lg:top-0 lg:h-screen">
        {sidebarContent}
      </aside>
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden">
          <button
            type="button"
            className="flex-1 bg-black/60"
            aria-label="Close sidebar"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="flex w-72 border-l border-white/10 bg-slate-950/95">
            {sidebarContent}
          </div>
        </div>
      )}
      <div className="flex flex-1 min-h-0 flex-col pl-2 lg:pl-4">
        <div className="flex items-center gap-3 border-b border-white/5 bg-slate-950/80 px-4 py-3 lg:hidden">
          <button
            type="button"
            className="rounded-2xl border border-white/15 bg-white/5 p-2 text-white"
            aria-label="Open sessions sidebar"
            onClick={() => setSidebarOpen(true)}
          >
            ☰
          </button>
          <div className="flex flex-1 items-center gap-3">
            <p className="text-xs uppercase tracking-[0.35em] text-white/50">
              Model
            </p>
            <select
              className="flex-1 rounded-2xl border border-white/15 bg-white/5 px-3 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
              disabled={modelOptions.length === 0 || modelStatus === "loading"}
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              {modelOptions.map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => setModelRefreshKey((key) => key + 1)}
              className="rounded-2xl border border-white/15 bg-white/5 p-2 text-white/70 hover:text-white"
              aria-label="Refresh models"
            >
              <RefreshIcon />
            </button>
          </div>
        </div>
        <div className="flex w-full flex-1 min-h-0 flex-col gap-4 pl-3 lg:pl-6">
          {initializing ? (
            <div className="flex flex-1 items-center justify-center text-white/60">
              Loading workspace…
            </div>
          ) : (
            <>
              <div className="flex flex-1 min-h-0 flex-col gap-8 lg:flex-row lg:items-start">
                <aside className={conversationPaneClass}>
                  <div className="flex items-center justify-between pb-3">
                    {messagesLoading && (
                      <span className="text-xs text-white/50">Loading…</span>
                    )}
                  </div>
                  <div className="flex-1 min-h-0 overflow-y-auto pr-2">
                    <div
                      ref={scrollerRef}
                      className="space-y-4 text-sm mt-4"
                      style={{ minHeight: "320px" }}
                    >
                      {messages.length === 0 && !messagesLoading && (
                        <div className="rounded-2xl border border-dashed border-white/10 bg-white/5 px-4 py-6 text-center text-white/50">
                          Send a message to start the conversation.
                        </div>
                      )}
                      {messages.map((message, idx) => (
                        <MessageBubble key={idx} message={message} />
                      ))}
                      {conversationIndicatorActive && (
                        <div className="flex items-center gap-2 text-xs text-sky-200/80 pl-2">
                          <span className="ml-[-2px] h-2 w-2 rounded-full bg-sky-400 animate-ping" />
                          {conversationIndicatorText}
                        </div>
                      )}
                      <div ref={scrollAnchorRef} />
                    </div>
                  </div>
                  <form
                    className="sticky bottom-0 mt-4 space-y-3 rounded-2xl border border-white/5 bg-black/30 p-3 backdrop-blur"
                    onSubmit={(e) => {
                      e.preventDefault();
                      sendUserText(input);
                    }}
                  >
                    <textarea
                      className="h-20 w-full resize-none rounded-2xl border border-white/10 bg-transparent px-3 py-2 text-sm text-white placeholder:text-white/40 focus:border-sky-500 focus:outline-none"
                      placeholder="Type a manual message (Shift+Enter for newline)."
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleComposerKey}
                    />
                    <div className="flex flex-col gap-2">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => setWebSearchEnabled((prev) => !prev)}
                            aria-pressed={webSearchEnabled}
                            aria-label="Toggle web search"
                            className={`flex items-center gap-2 rounded-2xl border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.3em] transition ${
                              webSearchEnabled
                                ? "border-sky-400/70 bg-sky-500/20 text-sky-100"
                                : "border-white/15 bg-transparent text-white/60 hover:border-white/40 hover:text-white"
                            }`}
                            disabled={!searxReachable}
                          >
                            <SearchIcon />
                            {webSearchEnabled && <span>Web Search</span>}
                          </button>
                          {!searxReachable && (
                            <span className="text-[11px] uppercase tracking-[0.3em] text-amber-300/80">
                              SearxNG unreachable—check Settings → Web Search
                            </span>
                          )}

                          <button
                            type="button"
                            onClick={() => {
                              setBrowsePanelOpen((prev) => !prev);
                              setBrowseError(null);
                            }}
                            aria-label="Attach web page"
                            className={`flex items-center gap-2 rounded-2xl border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.3em] transition ${
                              browsePanelOpen || browseAttachment
                                ? "border-purple-400/70 bg-purple-500/20 text-purple-100"
                                : "border-white/15 bg-transparent text-white/60 hover:border-white/40 hover:text-white"
                            }`}
                          >
                            <GlobeIcon />
                            {browseAttachment && (
                              <span className="h-2 w-2 rounded-full bg-purple-300"></span>
                            )}
                          </button>
                          <button
                            type="button"
                            onClick={toggleVoiceEnabled}
                            aria-pressed={voiceEnabled}
                            className={`flex items-center gap-2 rounded-2xl border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.3em] transition ${
                              voiceEnabled
                                ? "border-pink-400/70 bg-pink-500/20 text-pink-100"
                                : "border-white/15 bg-transparent text-white/60 hover:border-white/40 hover:text-white"
                            }`}
                          >
                            <MicrophoneIcon />
                            {voiceEnabled && <span>Voice</span>}
                          </button>
                          <button
                            type="button"
                            onClick={() => setThinkingEnabled((prev) => !prev)}
                            aria-pressed={thinkingEnabled}
                            className={`flex items-center gap-2 rounded-2xl border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.3em] transition ${
                              thinkingEnabled
                                ? "border-amber-400/70 bg-amber-500/20 text-amber-100"
                                : "border-white/15 bg-transparent text-white/60 hover:border-white/40 hover:text-white"
                            }`}
                          >
                            <BrainIcon />
                            {thinkingEnabled && <span>Thinking</span>}
                          </button>
                          <button
                            type="button"
                            title="Live web context will be added to your next prompt. Powered by SearxNG."
                            aria-label="Web search info"
                            className="rounded-full border border-white/10 bg-transparent p-1.5 text-white/50 hover:border-sky-400/50 hover:text-sky-100"
                          >
                            <InfoIcon />
                          </button>
                        </div>
                        <button
                          type="submit"
                          disabled={composerDisabled}
                          className="flex min-w-[120px] items-center justify-center gap-2 rounded-xl border border-sky-500/40 bg-sky-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-sky-100 transition hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/5 disabled:text-white/30"
                        >
                          <SendIcon />
                          {searchingWeb ? "Searching…" : "Send"}
                        </button>
                      </div>
                      <div className="flex flex-wrap items-center justify-between text-[11px]">
                        <span className="text-white/40">
                          Shift+Enter for newline
                        </span>
                        {webSearchEnabled && renderSearchStatus(lastSearch)}
                        {browseAttachment && (
                          <span className="text-purple-200">
                            Attached: {browseAttachment.title}
                          </span>
                        )}
                      </div>
                      {thinkingEnabled && thinkingStatusLine && (
                        <div className="text-xs text-amber-100/80">
                          {thinkingStatusLine}
                        </div>
                      )}
                      {browsePanelOpen && (
                        <div className="rounded-2xl border border-white/10 bg-black/30 p-3 space-y-2">
                          <div className="text-xs uppercase tracking-[0.3em] text-white/40">
                            Load web page
                          </div>
                          <div className="flex gap-2">
                            <input
                              className="flex-1 rounded-2xl border border-white/15 bg-white/5 px-3 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
                              placeholder="https://example.com/article"
                              value={browseUrlDraft}
                              onChange={(e) =>
                                setBrowseUrlDraft(e.target.value)
                              }
                            />
                            <button
                              type="button"
                              disabled={browseLoading}
                              onClick={() => {
                                void handleBrowseFetch();
                              }}
                              className="rounded-2xl border border-white/15 bg-white/5 px-3 py-2 text-xs font-semibold uppercase tracking-[0.3em] text-white/70 transition hover:border-sky-400/60 hover:text-white disabled:opacity-50"
                            >
                              {browseLoading ? "Loading…" : "Attach"}
                            </button>
                          </div>
                          {browseError && (
                            <p className="text-xs text-red-300">
                              {browseError}
                            </p>
                          )}
                        </div>
                      )}
                      {browseAttachment && (
                        <div className="rounded-2xl border border-purple-400/40 bg-purple-500/10 p-3 text-sm text-white/80 space-y-1">
                          <div className="flex items-center justify-between gap-3">
                            <p className="font-semibold">
                              {browseAttachment.title}
                            </p>
                            <button
                              type="button"
                              onClick={clearBrowseAttachment}
                              className="text-xs uppercase tracking-[0.3em] text-purple-200 hover:text-purple-50"
                            >
                              Remove
                            </button>
                          </div>
                          <p className="text-xs text-white/60 break-words">
                            {browseAttachment.url}
                          </p>
                          <p className="text-xs text-white/60 line-clamp-3">
                            {browseAttachment.summary}
                          </p>
                        </div>
                      )}
                    </div>
                  </form>
                </aside>

                {voiceEnabled && (
                  <section className="w-full border-l border-white/10 bg-slate-950/80 p-6 sm:p-8 backdrop-blur lg:w-[380px] lg:sticky lg:top-4 lg:min-h-[calc(100vh-2rem)]">
                    <div className="flex flex-col items-center gap-8 text-center">
                      <VoiceOrb
                        listening={sessionActive && listening && !micMuted}
                        playing={playing}
                        streaming={streaming}
                        audioMuted={audioMuted}
                      />
                      <div className="min-h-[1.5rem] text-sm text-white/60">
                        {statusNote ||
                          (sessionActive
                            ? "Say something and I'll take it from there."
                            : "Voice is enabled—hang tight while the mic connects.")}
                      </div>
                      {sessionActive && (
                        <div className="flex flex-wrap justify-center gap-3">
                          <button
                            type="button"
                            onClick={toggleMicMute}
                            aria-pressed={micMuted}
                            className={`flex h-11 w-11 items-center justify-center rounded-2xl border transition ${
                              micMuted
                                ? "border-amber-400/60 text-amber-200"
                                : "border-white/15 text-white/70 hover:text-white"
                            }`}
                          >
                            {micMuted ? (
                              <MicrophoneOffIcon />
                            ) : (
                              <MicrophoneIcon />
                            )}
                            <span className="sr-only">
                              {micMuted
                                ? "Unmute microphone"
                                : "Mute microphone"}
                            </span>
                          </button>
                          <button
                            type="button"
                            onClick={toggleAudioMute}
                            aria-pressed={audioMuted}
                            className={`flex h-11 w-11 items-center justify-center rounded-2xl border transition ${
                              audioMuted
                                ? "border-emerald-400/60 text-emerald-200"
                                : "border-white/15 text-white/70 hover:text-white"
                            }`}
                          >
                            {audioMuted ? (
                              <SpeakerOffIcon />
                            ) : (
                              <SpeakerSimpleIcon />
                            )}
                            <span className="sr-only">
                              {audioMuted
                                ? "Unmute app audio"
                                : "Mute app audio"}
                            </span>
                          </button>
                        </div>
                      )}
                      {sessionActive && (
                        <div className="w-full max-w-[220px] space-y-1 text-left">
                          <label className="flex items-center justify-between text-[11px] uppercase tracking-[0.3em] text-white/40">
                            <span>AI volume</span>
                            <span>{Math.round(audioVolume * 100)}%</span>
                          </label>
                          <input
                            type="range"
                            min={0}
                            max={100}
                            value={Math.round(audioVolume * 100)}
                            onChange={(e) =>
                              handleVolumeChange(Number(e.target.value) / 100)
                            }
                            className="w-full accent-sky-400"
                            aria-label="Adjust AI playback volume"
                          />
                        </div>
                      )}
                      {sessionActive && (
                        <>
                          <div className="flex flex-wrap justify-center gap-6 text-left">
                            <SpeakerVisualizer
                              title="You"
                              playing={userSpeaking}
                            />
                          </div>
                        </>
                      )}
                      <div className="w-full flex flex-wrap items-center justify-center gap-3">
                        <button
                          onClick={() => setVoiceEnabled(false)}
                          className="rounded-2xl border border-red-500/40 bg-red-500/10 px-6 py-3 text-sm font-semibold text-red-100 hover:bg-red-500/20"
                        >
                          Stop Voice
                        </button>
                      </div>
                    </div>
                  </section>
                )}
              </div>
            </>
          )}
        </div>
      </div>
      {settingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center px-4 py-8">
          <button
            type="button"
            className="absolute inset-0 bg-black/70"
            aria-label="Close settings"
            onClick={() => setSettingsOpen(false)}
          />
          <div className="relative z-10 w-full max-w-2xl rounded-3xl border border-white/10 bg-slate-950/95 p-6 shadow-2xl backdrop-blur flex flex-col max-h-[85vh] min-h-[60vh]">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-white/40">
                  Workspace settings
                </p>
                <h2 className="text-xl font-semibold text-white">
                  Models & memory
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setSettingsOpen(false)}
                className="flex h-10 w-10 items-center justify-center rounded-2xl border border-white/15 bg-white/5 text-white/70 transition hover:border-red-400/60 hover:text-white"
                aria-label="Close settings"
              >
                <CloseIcon />
              </button>
            </div>
            <div className="mt-6 flex flex-col gap-6 flex-1 min-h-0">
              <div className="flex gap-2 border-b border-white/10 pb-2 text-sm font-semibold text-white/60">
                {SETTINGS_TABS.map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => setSettingsTab(tab)}
                    className={`rounded-full px-3 py-1 transition ${
                      settingsTab === tab
                        ? "bg-sky-500/20 text-white border border-sky-500/40"
                        : "text-white/50 hover:text-white"
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              <div className="flex-1 overflow-y-auto space-y-6 pr-1 min-h-0">
                {settingsTab === "General" && (
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <label className="text-xs uppercase tracking-[0.3em] text-white/40">
                        Active model
                        <span className="ml-1 text-[10px] text-white/50">
                          ({modelOptions.length || 0})
                        </span>
                      </label>
                      <div className="flex gap-2">
                        <select
                          className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
                          disabled={
                            modelOptions.length === 0 ||
                            modelStatus === "loading"
                          }
                          value={model}
                          onChange={(e) => setModel(e.target.value)}
                        >
                          {modelOptions.map((name) => (
                            <option key={name}>{name}</option>
                          ))}
                        </select>
                        <button
                          type="button"
                          aria-label="Refresh models"
                          title="Refresh models"
                          onClick={() => setModelRefreshKey((key) => key + 1)}
                          className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/15 bg-white/5 text-white/70 transition hover:border-emerald-400/60 hover:text-white"
                        >
                          <RefreshIcon />
                        </button>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-white/60">
                        <span
                          className={`h-2 w-2 rounded-full ${modelStatusDot}`}
                        />
                        <span>{modelHint}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <label className="text-xs uppercase tracking-[0.3em] text-white/40">
                          System prompt prefix
                        </label>
                        <button
                          type="button"
                          className="rounded-xl border border-white/10 px-3 py-1 text-[11px] uppercase tracking-[0.3em] text-white/60 transition hover:border-red-400/50 hover:text-white"
                          onClick={() => {
                            void resetSystemPrompt();
                          }}
                        >
                          Reset
                        </button>
                      </div>
                      <textarea
                        className="min-h-[120px] w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white focus:border-sky-500 focus:outline-none"
                        value={systemPromptDraft}
                        onChange={(e) => setSystemPromptDraft(e.target.value)}
                        placeholder={DEFAULT_SYSTEM_PROMPT}
                      />
                      <button
                        type="button"
                        onClick={() => {
                          void saveSystemPrompt();
                        }}
                        className="w-full rounded-2xl border border-white/10 bg-white/10 py-2 text-sm font-medium text-white transition hover:border-sky-500/40 hover:bg-sky-500/10"
                      >
                        Save prompt
                      </button>
                      <p className="text-[11px] text-white/40">
                        Core safety instructions remain appended automatically.
                      </p>
                    </div>
                  </div>
                )}

                {settingsTab === "Web Search" && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="text-base font-semibold text-white">
                        Web search endpoint
                      </h3>
                      <button
                        type="button"
                        className="rounded-xl border border-white/10 px-3 py-1 text-[11px] uppercase tracking-[0.3em] text-white/60 transition hover:border-red-400/50 hover:text-white"
                        onClick={() => {
                          setSearxDraft(DEFAULT_SEARX_URL);
                          setSearxAllowInsecureDraft(false);
                          void resetSearxSettings();
                        }}
                      >
                        Reset
                      </button>
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs uppercase tracking-[0.3em] text-white/40">
                        SearxNG base URL
                      </label>
                      <div className="flex gap-2">
                        <input
                          className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
                          value={searxDraft}
                          onChange={(e) => setSearxDraft(e.target.value)}
                          placeholder={DEFAULT_SEARX_URL}
                        />
                        <button
                          type="button"
                          onClick={() => {
                            void saveSearxSettings();
                          }}
                          className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold uppercase tracking-[0.25em] text-white/70 transition hover:border-emerald-400/60 hover:text-white"
                        >
                          <SaveIcon />
                          Save
                        </button>
                      </div>
                    </div>
                    <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/80">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-white/40 bg-transparent"
                        checked={searxAllowInsecureDraft}
                        onChange={(e) =>
                          setSearxAllowInsecureDraft(e.target.checked)
                        }
                      />
                      <div>
                        <div className="text-[11px] uppercase tracking-[0.3em] text-white/50">
                          Allow self-signed TLS
                        </div>
                        <p className="text-[11px] text-white/50">
                          Disable certificate verification for trusted, local
                          SearxNG instances.
                        </p>
                      </div>
                    </label>
                    <p className="text-[11px] text-white/40">
                      Requests are proxied through this SearxNG instance
                      whenever web search is enabled.
                    </p>
                  </div>
                )}

                {settingsTab === "Memory" && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="text-base font-semibold text-white">
                        Memory
                      </h3>
                      <span className="text-xs text-white/50">
                        {memoryEntries.length} entries
                      </span>
                    </div>
                    <div className="space-y-2 rounded-2xl border border-white/10 bg-white/5 p-3 text-sm">
                      {memoryEntries.length === 0 && (
                        <p className="text-center text-xs text-white/50">
                          No memories stored yet.
                        </p>
                      )}
                      {memoryEntries.map((entry) => (
                        <div
                          key={entry.id}
                          className="rounded-xl border border-white/10 bg-black/30 p-3"
                        >
                          <div className="flex items-center justify-between text-xs text-white/60">
                            <span className="font-semibold text-white">
                              {entry.label}
                            </span>
                            <button
                              type="button"
                              onClick={() => handleDeleteMemoryEntry(entry.id)}
                              className="text-red-300 hover:text-red-200"
                            >
                              Remove
                            </button>
                          </div>
                          <p className="mt-1 text-sm text-white/80">
                            {entry.value}
                          </p>
                        </div>
                      ))}
                    </div>
                    <form className="space-y-2" onSubmit={handleAddMemoryEntry}>
                      <input
                        className="w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
                        placeholder="Label (e.g., Favorites)"
                        value={memoryLabel}
                        onChange={(e) => setMemoryLabel(e.target.value)}
                      />
                      <textarea
                        className="min-h-[80px] w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
                        placeholder="Memory details"
                        value={memoryValue}
                        onChange={(e) => setMemoryValue(e.target.value)}
                      />
                      <button
                        type="submit"
                        className="w-full rounded-2xl border border-white/10 bg-white/10 py-2 text-sm font-medium text-white transition hover:border-emerald-400/40 hover:bg-emerald-400/10"
                      >
                        Add memory
                      </button>
                    </form>
                  </div>
                )}

                {settingsTab === "Ollama Config" && (
                  <div className="space-y-2">
                    <label className="text-xs uppercase tracking-[0.3em] text-white/40">
                      Ollama endpoint
                    </label>
                    <div className="flex gap-2">
                      <input
                        className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white focus:border-sky-500 focus:outline-none"
                        value={ollamaDraft}
                        onChange={(e) => setOllamaDraft(e.target.value)}
                        placeholder="https://ollama.local:11434"
                      />
                      <button
                        aria-label="Save Ollama endpoint"
                        title="Save Ollama endpoint"
                        className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/15 bg-white/5 text-white/70 transition hover:border-emerald-400/60 hover:text-white"
                        onClick={() => {
                          void saveOllamaSettings();
                        }}
                        type="button"
                      >
                        <SaveIcon />
                      </button>
                    </div>
                    <p className="text-xs text-white/50">
                      This host is used for both the LLM and the transcription
                      pipeline.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      {diagnosticsOpen && (
        <DiagnosticsPanel
          logs={timingLogs}
          loading={timingLoading}
          error={timingError}
          onClose={() => setDiagnosticsOpen(false)}
          onRefresh={refreshTimingLogs}
        />
      )}
    </div>
  );
}

function MessageBubble({ message }: { message: Msg }) {
  const isUser = message.role === "user";
  const wrapper = isUser ? "items-end text-right" : "items-start text-left";
  const bubble = isUser
    ? "bg-sky-500/15 border border-sky-500/30"
    : "bg-slate-800/80 border border-white/10";
  const showWorkflow =
    !isUser && message.workflowStages && message.workflowStages.length > 0;
  return (
    <div className={`flex flex-col gap-2 ${wrapper} mb-3`}>
      <span className="text-[11px] uppercase tracking-[0.3em] text-white/40">
        {isUser ? "You" : "Parlance"}
      </span>
      <div
        className={`max-w-full rounded-2xl px-4 py-3 text-base leading-relaxed text-white whitespace-pre-wrap ${bubble}`}
      >
        {message.content || <span className="text-white/50">…</span>}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3 space-y-1 border-t border-white/10 pt-2 text-left">
            <p className="text-[11px] uppercase tracking-[0.3em] text-white/50">
              Sources
            </p>
            <ul className="space-y-1 text-sm">
              {message.sources.slice(0, 5).map((source, idx) => (
                <li key={`${source.url}-${idx}`}>
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky-300 hover:text-sky-200 underline decoration-dotted"
                  >
                    {source.title || source.url}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      {showWorkflow && (
        <WorkflowTimeline stages={message.workflowStages ?? []} />
      )}
    </div>
  );
}

function VoiceOrb({
  listening,
  playing,
  streaming,
  audioMuted,
}: {
  listening: boolean;
  playing: boolean;
  streaming: boolean;
  audioMuted: boolean;
}) {
  const activity = playing ? 1 : streaming ? 0.4 : listening ? 0.1 : 0;
  const outerScale = 1 + activity * 0.4;
  const glowOpacity = playing ? 0.9 : streaming ? 0.6 : 0.3;
  const state = playing
    ? audioMuted
      ? "Speaking (muted)"
      : "Speaking"
    : streaming
    ? "Thinking"
    : listening
    ? "Standby"
    : "Idle";
  const borderStyles = playing
    ? "border-sky-400 bg-sky-500/15"
    : streaming
    ? "border-amber-400 bg-amber-500/10"
    : "border-white/10 bg-white/5";
  return (
    <div className="relative h-64 w-64">
      <div
        className={`absolute inset-0 rounded-full bg-gradient-to-br from-sky-500/30 via-indigo-500/20 to-transparent blur-3xl transition-all duration-300 ${
          audioMuted ? "opacity-30" : ""
        }`}
        style={{ transform: `scale(${outerScale})` }}
      />
      <div
        className={`absolute inset-8 rounded-full border ${borderStyles} shadow-inner transition-colors duration-300 ${
          audioMuted ? "opacity-60" : ""
        }`}
        style={{ boxShadow: `0 0 25px rgba(14,165,233,${glowOpacity})` }}
      />
      <div
        className={`absolute inset-0 m-auto flex h-48 w-48 items-center justify-center rounded-full border ${
          playing
            ? "border-sky-400/70 bg-sky-500/20"
            : streaming
            ? "border-amber-400/60 bg-amber-500/10"
            : "border-white/10 bg-white/5"
        } transition-colors duration-300`}
      >
        <div className="text-center">
          <div className="text-xs uppercase tracking-[0.4em] text-white/50">
            {state}
          </div>
          <div className="mt-2 text-2xl font-semibold text-white">
            {playing
              ? audioMuted
                ? "Muted"
                : "Talking"
              : streaming
              ? "Processing"
              : listening
              ? "Ready"
              : "Idle"}
          </div>
        </div>
      </div>
    </div>
  );
}

function SpeakerVisualizer({
  title,
  playing,
}: {
  title: string;
  playing: boolean;
}) {
  const bars = Array.from({ length: 5 });
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-black/30 px-4 py-3">
      <div className="flex gap-1">
        {bars.map((_, idx) => {
          const baseHeight = 10 + idx * 4;
          return (
            <span
              key={idx}
              className={`block w-2 rounded-full bg-sky-400 ${
                playing ? "animate-[pulse_0.9s_ease-in-out_infinite]" : ""
              }`}
              style={{
                height: playing
                  ? `${baseHeight + 10}px`
                  : `${baseHeight + 2}px`,
                animationDelay: `${idx * 0.12}s`,
                transition: "height 0.2s ease, opacity 0.2s ease",
                opacity: playing ? 1 : 0.6,
              }}
            />
          );
        })}
      </div>
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-white/40">
          {title}
        </div>
        <div className="text-sm text-white">
          {playing ? "Speaking now" : "Quiet"}
        </div>
      </div>
    </div>
  );
}

function selectRecorderOptions(): MediaRecorderOptions | undefined {
  if (
    typeof MediaRecorder === "undefined" ||
    typeof MediaRecorder.isTypeSupported !== "function"
  ) {
    return {};
  }
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
    "audio/mpeg",
  ];
  for (const mime of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(mime)) {
        return { mimeType: mime };
      }
    } catch (err) {
      console.warn("[parlance] mimeType check failed", mime, err);
    }
  }
  console.warn(
    "[parlance] falling back to browser default MediaRecorder mimeType"
  );
  return {};
}

function SendIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M5 12h14" />
      <path d="M12 5l7 7-7 7" />
    </svg>
  );
}

function GlobeIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M9 3.5a2.5 2.5 0 0 0-2.5 2.5v1.5A2.5 2.5 0 0 0 4 10v2a3 3 0 0 0 3 3v1.5A2.5 2.5 0 0 0 9.5 19" />
      <path d="M15 3.5a2.5 2.5 0 0 1 2.5 2.5v1.5A2.5 2.5 0 0 1 20 10v2a3 3 0 0 1-3 3v1.5A2.5 2.5 0 0 1 14.5 19" />
      <path d="M9 8h6" />
      <path d="M9 12h6" />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="23 4 23 10 17 10" />
      <polyline points="1 20 1 14 7 14" />
      <path d="M3.51 9a9 9 0 0 1 14.13-3.36L23 10M1 14l5.36 4.36A9 9 0 0 0 20.49 15" />
    </svg>
  );
}

function SaveIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.09A1.65 1.65 0 0 0 9 4.6V4a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.09a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}
function TrashIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}
function describeCurrentDateTime() {
  const now = new Date();
  const timezone =
    Intl.DateTimeFormat().resolvedOptions().timeZone || "local time";
  const formatted = now.toLocaleString(undefined, {
    dateStyle: "full",
    timeStyle: "long",
  });
  return `Current local date and time: ${formatted} (timezone: ${timezone}, ISO: ${now.toISOString()}). Use this information whenever a user asks about the current date or time.`;
}

function userRequestedDateTime(content: string | undefined) {
  if (!content) return false;
  const text = content.toLowerCase();
  return (
    /(?:what(?:'s| is)?\s+the\s+)?(?:time|date|day|year)\b/.test(text) ||
    /\bcurrent\s+(?:time|date|year)\b/.test(text) ||
    /\btime\s+is\s+it\b/.test(text) ||
    /\byear\s+is\s+it\b/.test(text)
  );
}

function sanitizeForSpeech(text: string) {
  if (!text) return "";
  let clean = text;
  clean = clean.replace(/\*\*(.*?)\*\*/g, "$1");
  clean = clean.replace(/\*(.*?)\*/g, "$1");
  clean = clean.replace(/__(.*?)__/g, "$1");
  clean = clean.replace(/`([^`]+)`/g, "$1");
  clean = clean.replace(/\[(.*?)\]\((.*?)\)/g, "$1");
  clean = clean.replace(/^[-*+]\s+/gm, "");
  clean = clean.replace(/_{1,2}/g, "");
  clean = convertTemperaturesForSpeech(clean);
  clean = convertFractionsForSpeech(clean);
  clean = clean.replace(/\s{2,}/g, " ");
  return clean.trim();
}

function formatSearchContext(
  query: string,
  results: WebSearchResult[]
): string {
  const normalized = results.map((result, idx) => {
    const snippet = (result.snippet || "")
      .replace(/\s+/g, " ")
      .replace(/\[[^\]]*\]/g, "")
      .trim();
    const clipped =
      snippet.length > 280 ? `${snippet.slice(0, 277)}…` : snippet;
    const publishedLabel = result.publishedAt
      ? ` (Published ${formatDateForContext(result.publishedAt)})`
      : "";
    return `${idx + 1}. ${
      result.title
    }${publishedLabel} — ${clipped} (Source: ${result.url})`;
  });
  return [
    `Fresh web search results for: "${query}"`,
    "Ground your next response in these findings and mention relevant sources naturally when you rely on them.",
    ...normalized,
  ].join("\n");
}

function describeEmptySearchContext(query: string) {
  return `Web search was enabled but returned no useful results for "${query}". Let the user know no live data was available before sharing any relevant background knowledge.`;
}

function describeSearchFailureContext(query: string, error: string) {
  return `Web search failed for "${query}" due to: ${error}. Be transparent about this limitation before answering with existing knowledge only.`;
}

function renderSearchStatus(insight: SearchInsight | null) {
  if (!insight) return null;
  let text = "";
  let tone = "text-white/40";
  if (insight.status === "searching") {
    text = "Fetching web context…";
    tone = "text-sky-200";
  } else if (insight.status === "error") {
    text = `Web search failed: ${insight.error || "Unknown error"}.`;
    tone = "text-red-200";
  } else if (insight.results.length > 0) {
    const count = insight.results.length;
    text = `${count} result${
      count === 1 ? "" : "s"
    } were shared with the model.`;
    tone = "text-emerald-200";
  } else {
    text = "No live results were available; relying on existing knowledge.";
  }
  return <span className={tone}>{text}</span>;
}

function buildSearchConversation(messages: Msg[]) {
  return [...messages]
    .filter((msg) => msg.role === "user" || msg.role === "assistant")
    .slice(-8)
    .reverse()
    .map((msg) => ({
      role: msg.role as "user" | "assistant",
      content: truncateForSearch(msg.content, 280),
    }));
}

function buildTitleConversation(messages: Msg[]) {
  return messages
    .filter((msg) => msg.role === "user" || msg.role === "assistant")
    .slice(-8)
    .map((msg) => ({
      role: msg.role === "assistant" ? "assistant" : "user",
      content: truncateForSearch(msg.content, 160),
    }));
}

function truncateForSearch(text: string, limit: number) {
  if (!text) return "";
  if (text.length <= limit) return text;
  return text.slice(0, limit - 1) + "…";
}

function cleanTitleSuggestion(text: string) {
  if (!text) return "";
  let result = text.replace(/\s+/g, " ").replace(/[“”"]/g, "").trim();
  result = result.replace(/\b(chat|conversation|discussion)\b/gi, "").trim();
  const words = result.split(/\s+/).filter(Boolean).slice(0, 5);
  let normalized = words.join(" ");
  normalized = normalized.replace(/[^a-z0-9\s'-]/gi, "").trim();
  if (!normalized) return "";
  normalized = normalized.replace(/\s{2,}/g, " ").trim();
  const titleWords = normalized
    .split(" ")
    .map((word, idx) => (idx === 0 ? capitalizeWord(word) : word));
  return titleWords.join(" ").slice(0, 60).trim();
}

function capitalizeWord(word: string) {
  if (!word) return "";
  return word.charAt(0).toUpperCase() + word.slice(1);
}

const WEATHER_KEYWORD_PATTERNS = [
  /\bweather\b/i,
  /\bforecast\b/i,
  /\btemperature\b/i,
  /\btemp(?:s|erature)?\b/i,
  /\bdegrees?\b/i,
  /\brain(?:ing)?\b/i,
  /\bsnow(?:ing)?\b/i,
  /\bstorm(?:y)?\b/i,
  /\bwind(?:y)?\b/i,
  /\bhumid(?:ity)?\b/i,
  /\bsunny\b/i,
  /\bcloudy\b/i,
  /\bumbrella\b/i,
  /\bheat(?:wave)?\b/i,
  /\bcold\b/i,
  /\bhot\b/i,
  /\bchill(?:y)?\b/i,
  /\bfreez(?:e|ing)\b/i,
  /\bprecipitation\b/i,
  /\bclimate\b/i,
  /\bconditions?\b/i,
];

function shouldUseWeatherService(content: string) {
  if (!content) return false;
  return WEATHER_KEYWORD_PATTERNS.some((pattern) => pattern.test(content));
}

function formatWeatherContext(summary: WeatherSummary) {
  const tempC =
    typeof summary.temperatureC === "number"
      ? `${summary.temperatureC.toFixed(1)}°C`
      : "N/A";
  const tempF =
    typeof summary.temperatureF === "number"
      ? `${summary.temperatureF.toFixed(1)}°F`
      : null;
  const parts = [
    `Current weather for ${summary.location.name}${
      summary.location.country ? `, ${summary.location.country}` : ""
    }:`,
    summary.description,
    tempF ? `${tempC} (${tempF})` : tempC,
    `Wind ${
      typeof summary.windKph === "number" ? summary.windKph.toFixed(1) : "N/A"
    } km/h`,
    `Observed at ${summary.observedAt}.`,
    "Use this live data when answering related questions.",
  ];
  return parts.join(" ");
}

function createWorkflowStages(
  includeStt: boolean,
  includeTts = true
): WorkflowStage[] {
  const stages: WorkflowStage[] = [];
  if (includeStt) {
    stages.push({
      id: "stt",
      label: WORKFLOW_STAGE_LABELS.stt,
      status: "pending",
    });
  }
  stages.push(
    {
      id: "model",
      label: WORKFLOW_STAGE_LABELS.model,
      status: "pending",
    },
    ...(includeTts
      ? [
          {
            id: "tts",
            label: WORKFLOW_STAGE_LABELS.tts,
            status: "pending",
          } as WorkflowStage,
        ]
      : [])
  );
  return stages;
}

function cloneWorkflowStages(stages: WorkflowStage[]): WorkflowStage[] {
  return stages.map((stage) => ({ ...stage }));
}

function startWorkflowStage(
  stages: WorkflowStage[],
  id: WorkflowStageId,
  startedAt?: number
): WorkflowStage[] {
  const startTime = startedAt ?? performance.now();
  return stages.map((stage) =>
    stage.id === id
      ? {
          ...stage,
          status: "running",
          startedAt: startTime,
          durationMs: undefined,
        }
      : stage
  );
}

function completeWorkflowStage(
  stages: WorkflowStage[],
  id: WorkflowStageId,
  status: "success" | "error",
  completedAt?: number
): WorkflowStage[] {
  const endTime = completedAt ?? performance.now();
  return stages.map((stage) => {
    if (stage.id !== id) return stage;
    const startTime = stage.startedAt ?? endTime;
    return {
      ...stage,
      status,
      durationMs: Math.max(0, endTime - startTime),
      startedAt: startTime,
    };
  });
}

function removeWorkflowStage(stages: WorkflowStage[], id: WorkflowStageId) {
  return stages.filter((stage) => stage.id !== id);
}

function formatBrowseContext(attachment: {
  url: string;
  title: string;
  summary: string;
  source: string;
}) {
  const snippet = attachment.summary.slice(0, 2000);
  return `Web page: ${attachment.title} (Source: ${attachment.url})\nContent snippet:\n${snippet}`;
}

function WorkflowTimeline({ stages }: { stages: WorkflowStage[] }) {
  if (!stages.length) return null;
  return (
    <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-white/70">
      {stages.map((stage) => {
        const status = stage.status;
        let statusLabel = "Pending";
        if (status === "success") {
          statusLabel =
            typeof stage.durationMs === "number"
              ? formatLatencyDuration(stage.durationMs)
              : "Done";
        } else if (status === "running") {
          statusLabel = "In progress…";
        } else if (status === "error") {
          statusLabel = "Failed";
        }
        const toneClass = workflowStageTone(status);
        return (
          <div
            key={stage.id}
            className={`flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-2.5 py-1 ${toneClass}`}
            title={stage.label}
          >
            <WorkflowStageIcon id={stage.id} status={status} />
            <span className="text-white/80">{statusLabel}</span>
          </div>
        );
      })}
    </div>
  );
}

function workflowStageTone(status: WorkflowStageStatus) {
  if (status === "success") {
    return "text-emerald-100";
  }
  if (status === "running") {
    return "text-sky-100";
  }
  if (status === "error") {
    return "text-rose-100";
  }
  return "text-white/70";
}

function WorkflowStageIcon({
  id,
  status,
}: {
  id: WorkflowStageId;
  status: WorkflowStageStatus;
}) {
  const baseClass =
    "flex h-5 w-5 items-center justify-center rounded-full bg-white/10";
  const statusRing =
    status === "success"
      ? "ring-emerald-400/40"
      : status === "running"
      ? "ring-sky-400/40"
      : status === "error"
      ? "ring-rose-400/60"
      : "ring-white/10";
  return (
    <span className={`${baseClass} ring-1 ${statusRing}`}>
      {id === "stt" && <MicIcon />}
      {id === "model" && <CloudIcon />}
      {id === "tts" && <SpeakerIcon />}
    </span>
  );
}

function timingStatusBadgeClass(status: TimingLogEntry["status"]) {
  return status === "ok"
    ? "border border-emerald-400/40 bg-emerald-500/10 text-emerald-100"
    : "border border-rose-400/40 bg-rose-500/10 text-rose-100";
}

function formatTimingDuration(ms: number) {
  if (!Number.isFinite(ms)) return "—";
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)} s`;
  }
  return `${Math.round(ms)} ms`;
}

function formatTimingSteps(steps: TimingLogEntry["steps"]) {
  if (!steps || !steps.length) return "—";
  return steps
    .map((step) => `${step.name}: ${formatTimingDuration(step.durationMs)}`)
    .join(", ");
}

function formatTimingMetadata(
  metadata: TimingLogEntry["metadata"] | undefined
) {
  if (!metadata) return "—";
  const entries = Object.entries(metadata);
  if (!entries.length) return "—";
  return entries.map(([key, value]) => `${key}: ${value ?? "null"}`).join(", ");
}

function formatTimingTimestamp(iso: string) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

type DiagnosticsPanelProps = {
  logs: TimingLogEntry[];
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onRefresh: () => void;
};

function DiagnosticsPanel({
  logs,
  loading,
  error,
  onClose,
  onRefresh,
}: DiagnosticsPanelProps) {
  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-slate-950 text-white">
      <header className="flex items-center justify-between border-b border-white/10 px-6 py-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-white/40">
            Diagnostics
          </p>
          <h2 className="text-2xl font-semibold text-white">
            Live service timings
          </h2>
          <p className="text-sm text-white/60">
            Automatically refreshes every {DIAGNOSTICS_POLL_INTERVAL / 1000}{" "}
            seconds.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/80 transition hover:border-emerald-400/60 hover:text-white disabled:opacity-50"
          >
            <RefreshIcon />
            Refresh now
          </button>
          <button
            type="button"
            onClick={onClose}
            className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/15 bg-white/5 text-white/70 transition hover:border-red-400/60 hover:text-white"
            aria-label="Close diagnostics"
          >
            <CloseIcon />
          </button>
        </div>
      </header>
      <main className="flex-1 overflow-hidden p-6">
        {error && (
          <div className="mb-4 rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            {error}
          </div>
        )}
        <div className="h-full overflow-hidden rounded-3xl border border-white/10 bg-black/10">
          <div className="h-full overflow-auto">
            <table className="min-w-full text-sm text-white/80">
              <thead className="sticky top-0 bg-slate-950/80 backdrop-blur">
                <tr className="text-xs uppercase tracking-[0.2em] text-white/40">
                  <th className="px-4 py-3 text-left">Event</th>
                  <th className="px-4 py-3 text-left">Status</th>
                  <th className="px-4 py-3 text-left">Duration</th>
                  <th className="px-4 py-3 text-left">Steps</th>
                  <th className="px-4 py-3 text-left">Metadata</th>
                  <th className="px-4 py-3 text-left">Started</th>
                </tr>
              </thead>
              <tbody>
                {loading && logs.length === 0 && (
                  <tr>
                    <td
                      colSpan={6}
                      className="px-4 py-8 text-center text-white/50"
                    >
                      Loading timing data…
                    </td>
                  </tr>
                )}
                {!loading && logs.length === 0 && (
                  <tr>
                    <td
                      colSpan={6}
                      className="px-4 py-8 text-center text-white/50"
                    >
                      No timings recorded yet. Trigger a weather, search, STT,
                      or TTS call to populate this table.
                    </td>
                  </tr>
                )}
                {logs.map((entry) => (
                  <tr key={entry.id} className="border-t border-white/5">
                    <td className="px-4 py-3 font-semibold text-white">
                      {entry.event}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${timingStatusBadgeClass(
                          entry.status
                        )}`}
                      >
                        {entry.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      {formatTimingDuration(entry.durationMs)}
                    </td>
                    <td className="px-4 py-3 text-white/70">
                      {formatTimingSteps(entry.steps)}
                    </td>
                    <td className="px-4 py-3 text-white/70">
                      {formatTimingMetadata(entry.metadata)}
                    </td>
                    <td className="px-4 py-3 text-white/60">
                      {formatTimingTimestamp(entry.startedAt)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

type OllamaSetupProps = {
  value: string;
  onChange: (value: string) => void;
  onSave: () => void;
  hint: string;
};

function OllamaSetupScreen({
  value,
  onChange,
  onSave,
  hint,
}: OllamaSetupProps) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-slate-950 px-6 text-white">
      <div className="w-full max-w-xl space-y-6 rounded-3xl border border-white/10 bg-black/30 p-8 text-center shadow-2xl backdrop-blur">
        <p className="text-xs uppercase tracking-[0.4em] text-white/50">
          First-time setup
        </p>
        <h1 className="text-3xl font-semibold text-white">Connect Ollama</h1>
        <p className="text-sm text-white/70">
          Parlance needs a reachable Ollama host before it can load any models.
          Run <code className="rounded bg-white/10 px-1">ollama serve</code> on
          your machine (or point to any compatible endpoint), then provide the
          base URL below.
        </p>
        <div className="space-y-3 text-left">
          <label className="text-xs uppercase tracking-[0.3em] text-white/40">
            Ollama base URL
          </label>
          <input
            className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white focus:border-sky-500 focus:outline-none"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="http://localhost:11434"
            autoFocus
          />
          <button
            type="button"
            onClick={onSave}
            className="w-full rounded-2xl border border-emerald-400/40 bg-emerald-500/10 py-2 text-sm font-semibold uppercase tracking-[0.3em] text-emerald-100 transition hover:bg-emerald-500/20"
          >
            Save &amp; retry
          </button>
          <p className="text-sm text-amber-200/80">{hint}</p>
        </div>
        <p className="text-xs text-white/50">
          Need help? See the README for instructions on installing and running
          Ollama locally.
        </p>
      </div>
    </div>
  );
}

function MicIcon() {
  return (
    <svg
      className="h-3 w-3"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  );
}

function CloudIcon() {
  return (
    <svg
      className="h-3 w-3"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M17.5 19H7a5 5 0 0 1 0-10 6 6 0 0 1 11.4 1" />
    </svg>
  );
}

function SpeakerIcon() {
  return (
    <svg
      className="h-3 w-3"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M11 5 6 9H3v6h3l5 4z" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      <path d="M19.07 4.93a9 9 0 0 1 0 14.14" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function DiagnosticsIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="3 12 6 12 9 3 15 21 18 12 21 12" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="11" cy="11" r="7" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function MicrophoneIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  );
}

function MicrophoneOffIcon() {
  return (
    <svg
      className="h-4 w-4"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M9 9v3a3 3 0 0 0 5.12 2.12" />
      <path d="M5 10a7 7 0 0 0 11 5" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
      <path d="M15 9.34V5a3 3 0 0 0-5.12-2.12" />
      <line x1="2" y1="2" x2="22" y2="22" />
    </svg>
  );
}

function SpeakerSimpleIcon() {
  return (
    <svg
      className="h-5 w-5"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M11 5 6 9H3v6h3l5 4z" />
      <path d="M19 5a8 8 0 0 1 0 14" />
      <path d="M16 8a5 5 0 0 1 0 8" />
    </svg>
  );
}

function SpeakerOffIcon() {
  return (
    <svg
      className="h-5 w-5"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m6 9-3 3h4l5 4V5z" />
      <line x1="22" y1="9" x2="16" y2="15" />
      <line x1="16" y1="9" x2="22" y2="15" />
    </svg>
  );
}

function curateSearchResults(
  query: string,
  results: WebSearchResult[]
): WebSearchResult[] {
  if (!results.length) return results;
  const wantsRecent =
    /\b(latest|recent|today|tonight|now|current|breaking|newest|news|update)\b/i.test(
      query
    );
  const now = Date.now();
  const RECENT_WINDOW = 1000 * 60 * 60 * 24 * 120; // ~4 months
  const enriched = results.map((result, index) => {
    const publishedMs = result.publishedAt
      ? Date.parse(result.publishedAt)
      : NaN;
    const isFresh = Number.isFinite(publishedMs)
      ? now - publishedMs <= RECENT_WINDOW
      : false;
    const recencyScore = Number.isFinite(publishedMs)
      ? 1 - Math.min(1, (now - publishedMs) / (RECENT_WINDOW * 2))
      : 0.3;
    let score = 1 + recencyScore;
    if (wantsRecent) {
      score += isFresh ? 1 : -0.5;
    }
    return { result, score, index, publishedMs, isFresh };
  });
  enriched.sort((a, b) => {
    if (b.score === a.score) return a.index - b.index;
    return b.score - a.score;
  });
  const prioritized = enriched.map((entry) => entry.result).slice(0, 8);
  if (!wantsRecent) return prioritized.slice(0, 5);
  const fresh = enriched.filter((entry) => entry.isFresh).map((e) => e.result);
  const fallback = enriched
    .filter((entry) => !entry.isFresh)
    .map((entry) => entry.result);
  const combined = [...fresh, ...fallback].filter((item, idx, arr) => {
    const firstIndex = arr.findIndex(
      (candidate) =>
        candidate.url === item.url && candidate.title === item.title
    );
    return firstIndex === idx;
  });
  const final = combined.length ? combined : prioritized;
  return final.slice(0, 5);
}

function formatDateForContext(iso: string) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const formatter = new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  const relative = formatRelativeDate(date);
  return `${formatter.format(date)}${relative ? ` (${relative})` : ""}`;
}

function formatRelativeDate(date: Date) {
  const diffMs = Date.now() - date.getTime();
  if (!Number.isFinite(diffMs)) return "";
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (diffDays < 1) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  const diffWeeks = Math.floor(diffDays / 7);
  if (diffWeeks < 4)
    return `${diffWeeks} week${diffWeeks === 1 ? "" : "s"} ago`;
  const diffMonths = Math.floor(diffDays / 30);
  if (diffMonths < 12)
    return `${diffMonths} month${diffMonths === 1 ? "" : "s"} ago`;
  const diffYears = Math.floor(diffMonths / 12);
  return `${diffYears} year${diffYears === 1 ? "" : "s"} ago`;
}

function cleanTranscriptionText(text: string) {
  if (!text) return "";
  let result = text
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([,.!?])/g, "$1")
    .trim();
  const articleCandidates = [
    "nice",
    "great",
    "good",
    "delicious",
    "tasty",
    "simple",
    "easy",
    "quick",
    "hearty",
    "recipe",
  ];
  result = result.replace(/\bour\s+([a-z]+)/gi, (match, word) => {
    const lower = String(word).toLowerCase();
    if (!articleCandidates.includes(lower)) return match;
    const article = /^[aeiou]/i.test(word) ? "an" : "a";
    return `${article} ${word}`;
  });
  result = result.replace(/\b(gim+e)\b/gi, "give me");
  result = result.replace(/\b(im|iam)\b/gi, "I'm");
  if (/^[a-z]/.test(result)) {
    result = result[0].toUpperCase() + result.slice(1);
  }
  if (!/[.!?]$/.test(result)) result += ".";
  return result;
}

function convertTemperaturesForSpeech(text: string) {
  return text.replace(/(\d+)(?:\s?°)?\s?(F|C)\b/gi, (_match, value, unit) => {
    const number = String(value);
    const scale = unit?.toUpperCase() === "C" ? "celsius" : "fahrenheit";
    return `${number} degrees ${scale}`;
  });
}

function convertFractionsForSpeech(text: string) {
  const fractionMap: Record<string, string> = {
    "1/2": "one half",
    "1/3": "one third",
    "2/3": "two thirds",
    "1/4": "one quarter",
    "3/4": "three quarters",
    "1/8": "one eighth",
    "3/8": "three eighths",
    "5/8": "five eighths",
    "7/8": "seven eighths",
  };
  return text.replace(/\b(\d+\/\d+)\b/g, (match) => {
    const lower = match.trim();
    if (fractionMap[lower]) return fractionMap[lower];
    const [num, den] = lower.split("/");
    if (!num || !den) return match;
    return `${num} over ${den}`;
  });
}

function deriveTitleFromText(text: string) {
  if (!text) return "";
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return "";
  const cutoff = 40;
  const snippet =
    cleaned.length > cutoff ? `${cleaned.slice(0, cutoff).trim()}…` : cleaned;
  return snippet.charAt(0).toUpperCase() + snippet.slice(1);
}

function formatLatencyDuration(ms: number) {
  if (!Number.isFinite(ms) || ms < 0) return "";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms < 60000) return `${Math.round(ms / 1000)} s`;
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.round((ms % 60000) / 1000);
  const parts = [`${minutes}m`];
  if (seconds) parts.push(`${seconds}s`);
  return parts.join(" ");
}

function normalizeWorkflowStagesFromMetadata(
  value: unknown
): WorkflowStage[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const normalized: WorkflowStage[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object") continue;
    const obj = entry as Record<string, unknown>;
    const id = obj.id;
    const status = obj.status;
    if (
      (id !== "stt" && id !== "model" && id !== "tts") ||
      (status !== "pending" &&
        status !== "running" &&
        status !== "success" &&
        status !== "error")
    ) {
      continue;
    }
    normalized.push({
      id,
      label:
        typeof obj.label === "string"
          ? obj.label
          : WORKFLOW_STAGE_LABELS[id as WorkflowStageId],
      status,
      durationMs:
        typeof obj.durationMs === "number" ? obj.durationMs : undefined,
      startedAt: typeof obj.startedAt === "number" ? obj.startedAt : undefined,
    });
  }
  return normalized.length ? normalized : undefined;
}

function sanitizeWorkflowStagesForStorage(
  stages?: WorkflowStage[]
): WorkflowStage[] | undefined {
  if (!stages || !stages.length) return undefined;
  const filtered = stages
    .filter((stage) => stage.status !== "pending")
    .map((stage) => ({
      ...stage,
      label: stage.label || WORKFLOW_STAGE_LABELS[stage.id],
    }));
  return filtered.length ? filtered : undefined;
}
