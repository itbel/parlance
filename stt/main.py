import os, io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
COMPUTE = os.getenv("WHISPER_COMPUTE", "auto")  # auto|cpu|cuda
device = "cuda" if COMPUTE == "cuda" else "auto"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)

app = FastAPI()

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    if not data or len(data) < 1500:
        raise HTTPException(status_code=400, detail="audio too short")
    try:
        segments, info = model.transcribe(
            io.BytesIO(data),
            beam_size=1,
            vad_filter=True,
            language="en"
        )
        text = "".join([seg.text for seg in segments]).strip()
        return JSONResponse({"text": text, "language": info.language})
    except Exception as exc:
        raise HTTPException(status_code=422, detail="transcription failed") from exc

@app.get("/health")
def health():
    return {"ok": True, "model": WHISPER_MODEL, "device": device}
