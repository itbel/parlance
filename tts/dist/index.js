import express from "express";
import { KokoroTTS } from "kokoro-js";
import { env as hfEnv } from "@huggingface/transformers";
const app = express();
app.use(express.json({ limit: "2mb" }));
const MODEL_ID = process.env.KOKORO_MODEL_ID || process.env.KOKORO_MODEL_PATH || "onnx-community/Kokoro-82M-v1.0-ONNX";
const DEFAULT_VOICE = process.env.KOKORO_VOICE || "af_heart";
const DEVICE = (process.env.KOKORO_DEVICE || "cpu");
const DTYPE = (process.env.KOKORO_DTYPE || "q8");
if (MODEL_ID.startsWith("/") || MODEL_ID.startsWith("./")) {
    hfEnv.allowLocalModels = true;
}
let kokoroPromise = null;
function loadKokoro() {
    if (!kokoroPromise) {
        console.log(`[tts] loading Kokoro model (${MODEL_ID}) on ${DEVICE}/${DTYPE}`);
        kokoroPromise = KokoroTTS.from_pretrained(MODEL_ID, { device: DEVICE ?? undefined, dtype: DTYPE }).then((instance) => {
            console.log("[tts] Kokoro ready");
            return instance;
        }, (err) => {
            kokoroPromise = null;
            console.error("[tts] Kokoro failed to load:", err);
            throw err;
        });
    }
    return kokoroPromise;
}
app.post("/speak", async (req, res) => {
    const text = typeof req.body?.text === "string" ? req.body.text.trim() : "";
    if (!text)
        return res.status(400).json({ error: "missing text" });
    try {
        const tts = await loadKokoro();
        const voice = chooseVoice(tts, req.body?.voice);
        const speed = clampSpeed(req.body?.speed);
        const audio = await tts.generate(text, { voice, speed });
        const wav = audio.toWav();
        res.setHeader("Content-Type", "audio/wav");
        res.send(Buffer.from(wav));
    }
    catch (err) {
        console.error("[tts] synthesis failed:", err);
        res.status(500).json({ error: "tts failed" });
    }
});
app.get("/voices", async (_req, res) => {
    try {
        const tts = await loadKokoro();
        res.json(tts.voices);
    }
    catch {
        res.status(503).json({ error: "kokoro not ready" });
    }
});
app.get("/health", async (_req, res) => {
    const ready = kokoroPromise !== null;
    res.json({ ok: ready, model: MODEL_ID, device: DEVICE, dtype: DTYPE });
});
app.listen(8600, () => {
    console.log("[tts] :8600");
    loadKokoro().catch(() => { });
});
function chooseVoice(tts, requested) {
    const desired = typeof requested === "string" ? requested : DEFAULT_VOICE;
    if (desired in tts.voices)
        return desired;
    if (DEFAULT_VOICE in tts.voices)
        return DEFAULT_VOICE;
    return Object.keys(tts.voices)[0];
}
function clampSpeed(speed) {
    const value = typeof speed === "number" ? speed : Number(speed ?? 1);
    if (!Number.isFinite(value))
        return 1;
    return Math.min(2, Math.max(0.5, value));
}
