# Parlance

Parlance is a realtime assistant that combines local speech‑to‑text, an Ollama‑hosted LLM, and Kokoro text‑to‑speech behind a single web UI. This repository contains the pieces that Parlance owns (web UI, API server, Whisper STT, and Kokoro TTS). You are expected to run Ollama and a SearxNG instance separately and tell Parlance where to find them.

## Architecture

| Service | Purpose | Notes |
| --- | --- | --- |
| `web` | Vite/React front-end served by Nginx | Talks to the API server via `VITE_API_BASE`. |
| `server` | Express API that orchestrates sessions, forwards prompts to Ollama, and proxies SearxNG | Reads defaults from env (`DEFAULT_OLLAMA_URL`, `DEFAULT_SEARX_URL`). |
| `stt` | Whisper CPP container for microphone input | Exposes `POST /transcribe`. |
| `tts` | Kokoro-JS container for voice output | Exposes `POST /speak`. |

External dependencies you must provide:

* **Ollama** (or a compatible `/api/chat` endpoint) to run the LLM models.
* **SearxNG** for optional web search context.

## Quick start

1. **Install prerequisites**
   * Docker / Docker Compose v2
   * An Ollama instance reachable by the server (e.g. `ollama serve` on your host).
   * A SearxNG instance (docker or bare metal).

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` so `DEFAULT_OLLAMA_URL` points at your Ollama host (e.g. `http://host.docker.internal:11434`) and `DEFAULT_SEARX_URL` points at your SearxNG instance.

3. **Launch Parlance**
   ```bash
   docker compose up --build
   ```

   Containers:
   * API server → http://localhost:8787
   * Web UI → http://localhost:5173
   * Whisper STT → http://localhost:8000
   * Kokoro TTS → http://localhost:8600

4. **Open the UI** at http://localhost:5173. The first screen prompts for an Ollama URL; enter the base URL of the Ollama instance you prepared (e.g. `http://host.docker.internal:11434`) and click “Save & retry”. Once models appear in the dropdown you can start chatting. Configure SearxNG under **Settings → Web Search** before enabling the Web Search toggle in the composer.

## Environment reference

The main variables in `.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `DEFAULT_OLLAMA_URL` | `http://localhost:11434` | Base URL of your Ollama (or compatible) instance. |
| `DEFAULT_SEARX_URL` | `http://localhost:8080` | Base URL of your SearxNG instance. |
| `DEFAULT_SEARX_ALLOW_INSECURE` | `false` | Whether to allow self-signed TLS certs when calling SearxNG. |
| `STT_URL` | `http://stt:8000/transcribe` | Endpoint that the API server will call for transcription. |
| `TTS_URL` | `http://tts:8600/speak` | Endpoint that the API server will call for speech synthesis. |
| `TTS_ENABLED` | `true` | Toggle voice output (set `false` to disable TTS entirely). |
| `CORS_ORIGIN` | `http://localhost:5173` | Front-end origin allowed by the API server. |
| `VITE_API_BASE` | `http://localhost:8787` | API base embedded in the front-end during the Docker build. |
| `WHISPER_MODEL` | `base` | Whisper model pulled inside the STT container. |
| `WHISPER_COMPUTE` | `auto` | `auto`, `cpu`, or `cuda`. Set `cuda` if you want GPU acceleration. |
| `KOKORO_MODEL_ID` | `onnx-community/Kokoro-82M-v1.0-ONNX` | Model loaded by the TTS container. |
| `KOKORO_DEVICE` | `cpu` | Set to `cpu`, `webgpu`, or `cuda`. Use `cuda` when a GPU is available. |
| `KOKORO_DTYPE` | `q8` | Precision to load (e.g., `fp32`, `fp16`, `q8`). |
| `KOKORO_VOICE` | `af_heart` | Default voice ID. |

## Development tips

* To run the API locally without Docker: `cd server && npm install && npm run dev`.
* To run the web app with hot reload: `cd web && npm install && npm run dev` (ensure `VITE_API_BASE` points at your API host).
* The STT/TTS containers cache models under `stt/model_cache` and `tts/kokoro`. These folders are mounted as bind volumes so you can persist downloads between runs.

## Packaging / distribution

The repository is ready to share:

1. Commit `.env.example` (but not your `.env`).
2. Document Ollama/Searx prerequisites in your own release notes or include the “Quick start” section above.
3. Users clone the repo, provide their own Ollama + SearxNG endpoints, and run `docker compose up --build`.

Because Ollama and SearxNG stay outside of this compose stack, deployments remain flexible: you can point Parlance at remote instances, hosted providers, or local daemons simply by changing the environment variables or the Settings tab inside the app.

## GPU acceleration

Whisper and Kokoro can use NVIDIA GPUs:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.
2. Set `WHISPER_COMPUTE=cuda` and `KOKORO_DEVICE=cuda` in `.env` (or leave `auto` to let each service decide).
3. The compose file already reserves one GPU for both `stt` and `tts` via `deploy.resources`. If you're running on CPU-only hardware, comment out those sections.

### Windows + WSL2 quick start

On Windows, the easiest way to expose the GPU to Docker is via WSL2:

1. Install the latest NVIDIA driver for Windows that advertises WSL support.
2. Install WSL2 with an Ubuntu distribution (`wsl --install -d Ubuntu`) and reboot.
3. Install Docker Desktop for Windows and enable “Use the WSL 2 based engine”.
4. In Docker Desktop → Settings → Resources → WSL integration, enable your Ubuntu distro.
5. Open the Ubuntu shell (`wsl.exe`) and run `nvidia-smi` to confirm the GPU is visible.
6. Clone this repo inside WSL, create `.env`, and run `docker compose up --build`.
7. Set `WHISPER_COMPUTE=cuda` and `KOKORO_DEVICE=cuda` (or leave them at `cpu` if you’d rather run without the GPU).

Once Docker Desktop is using WSL2, both the STT and TTS containers share the same GPU automatically. If you hit `libcublas`/`cudnn` errors, double-check that the NVIDIA driver, WSL, and Docker Desktop versions meet NVIDIA’s requirements.
