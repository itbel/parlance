import express, { type Request, type Response } from "express";
import { KokoroTTS } from "kokoro-js";
import { env as hfEnv } from "@huggingface/transformers";

type KokoroDevice = "cpu" | "webgpu" | "wasm" | null;
type KokoroDType = "fp32" | "fp16" | "q8" | "q4" | "q4f16";

const app = express();
app.use(express.json({ limit: "2mb" }));

const MODEL_ID = process.env.KOKORO_MODEL_ID || process.env.KOKORO_MODEL_PATH || "onnx-community/Kokoro-82M-v1.0-ONNX";
const DEFAULT_VOICE = process.env.KOKORO_VOICE || "af_heart";
const DEVICE = (process.env.KOKORO_DEVICE || "cpu") as KokoroDevice;
const DTYPE = (process.env.KOKORO_DTYPE || "q8") as KokoroDType;

if (MODEL_ID.startsWith("/") || MODEL_ID.startsWith("./")) {
  hfEnv.allowLocalModels = true;
}

let kokoroPromise: Promise<KokoroTTS> | null = null;
function loadKokoro() {
  if (!kokoroPromise) {
    console.log(`[tts] loading Kokoro model (${MODEL_ID}) on ${DEVICE}/${DTYPE}`);
    kokoroPromise = KokoroTTS.from_pretrained(MODEL_ID, { device: DEVICE ?? undefined, dtype: DTYPE }).then(
      (instance) => {
        console.log("[tts] Kokoro ready");
        return instance;
      },
      (err) => {
        kokoroPromise = null;
        console.error("[tts] Kokoro failed to load:", err);
        throw err;
      }
    );
  }
  return kokoroPromise;
}

app.post("/speak", async (req: Request, res: Response) => {
  const text = typeof req.body?.text === "string" ? req.body.text.trim() : "";
  if (!text) return res.status(400).json({ error: "missing text" });

  try {
    const tts = await loadKokoro();
    const voice = chooseVoice(tts, req.body?.voice);
    const speed = clampSpeed(req.body?.speed);
    const audio = await tts.generate(text, { voice, speed });
    const wav = audio.toWav();
    res.setHeader("Content-Type", "audio/wav");
    res.send(Buffer.from(wav));
  } catch (err) {
    console.error("[tts] synthesis failed:", err);
    res.status(500).json({ error: "tts failed" });
  }
});

app.get("/voices", async (_req: Request, res: Response) => {
  try {
    const tts = await loadKokoro();
    res.json(tts.voices);
  } catch {
    res.status(503).json({ error: "kokoro not ready" });
  }
});

app.get("/health", async (_req: Request, res: Response) => {
  const ready = kokoroPromise !== null;
  res.json({ ok: ready, model: MODEL_ID, device: DEVICE, dtype: DTYPE });
});

app.listen(8600, () => {
  console.log("[tts] :8600");
  loadKokoro().catch(() => {});
});

type VoiceId = keyof KokoroTTS["voices"];

function chooseVoice(tts: KokoroTTS, requested?: unknown): VoiceId {
  const desired = typeof requested === "string" ? requested : DEFAULT_VOICE;
  if (desired in tts.voices) return desired as VoiceId;
  if (DEFAULT_VOICE in tts.voices) return DEFAULT_VOICE as VoiceId;
  return Object.keys(tts.voices)[0] as VoiceId;
}

function clampSpeed(speed: unknown) {
  const value = typeof speed === "number" ? speed : Number(speed ?? 1);
  if (!Number.isFinite(value)) return 1;
  return Math.min(2, Math.max(0.5, value));
}
