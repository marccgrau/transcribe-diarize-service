# 🎙️ Transcribe + Diarize Service

A containerized **FastAPI** service for:
- **Speech-to-text transcription** (Whisper + Faster-Whisper backend)
- **Speaker diarization** using NVIDIA’s **Sortformer** (`nvidia/diar_sortformer_4spk-v1`)
- **End-to-end pipeline** combining transcription + diarization with optional punctuation restoration

Designed for research and production environments, with GPU acceleration (CUDA/cuDNN) and containerized deployment.  

---

## ✨ Features

- **Transcription (`/transcribe`)**
  - Upload `.wav` (or converted `.m4a`, `.mp3`, etc.)
  - Uses [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for fast speech-to-text
  - Returns full transcript, language, and word-level timestamps

- **Diarization (`/diarize`)**
  - Runs NVIDIA’s Sortformer diarization model
  - Assigns speaker labels (`speaker_0`, `speaker_1`, …) to segments

- **Combined pipeline (`/transcribe-diarize`)**
  - Aligns transcribed words with diarized speaker segments
  - Restores punctuation with [`deepmultilingualpunctuation`](https://github.com/oliverguhr/deepmultilingual-punctuation-restoration)
  - Exports results as **JSON**, **TXT**, or **SRT**

- **Health & warmup endpoints**
  - `/health` – lightweight liveness/readiness probe
  - `/warmup` – triggers model load so first real request is fast

---

## 🗂 Project structure

```
src/
  core/
    utils.py                  # alignment, timestamp helpers
  services/
    transcription_service.py  # whisper transcription
    diarization_sortformer_service.py  # Sortformer wrapper
    pipeline_service.py       # combine transcription + diarization
  main.py                     # FastAPI app with endpoints
Dockerfile                    # production image
Dockerfile.dev                # dev image with hot-reload
docker-compose.yaml           # manage dev/prod containers
requirements.txt
```

---

## 🚀 Running locally

### 1. Clone the repo
```bash
git clone https://github.com/marccgrau/transcribe-diarize-service.git
cd transcribe-diarize-service
```

### 2. Install dependencies (optional bare-metal dev)
```bash
uv venv
source .venv/bin/activate
uv sync
```

Run FastAPI:
```bash
uvicorn src.main:app --reload --app-dir src --port 8000
```

Open API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Running with Docker

### 1. Build and run (production)
```bash
docker build -t transcribe-server:latest .
docker run --rm --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  transcribe-server:latest
```

### 2. Development with hot-reload
We provide a `docker-compose.yaml` with profiles:

```bash
# start dev container with hot-reload, source mounted
docker compose --profile dev up --build
```

In this mode:
- Code changes under `src/` are immediately reflected
- No need to rebuild the container each time

---

## 🧪 Usage examples

### Transcription
```bash
curl -X POST "http://localhost:8000/transcribe?model=base" \
  -F "file=@/path/to/audio.wav"
```

### Diarization
```bash
curl -X POST "http://localhost:8000/diarize" \
  -F "file=@/path/to/audio.wav"
```

### Transcribe + Diarize
```bash
curl -X POST "http://localhost:8000/transcribe-diarize?format=json&model=base" \
  -F "file=@/path/to/audio.wav"
```

Output formats:
- `json` (default) → structured segments
- `txt` → readable speaker transcript
- `srt` → subtitle file

---

## 🖥 Health checks

- `GET /health` → returns runtime info (uptime, python, torch, cuda, device)
- `GET /warmup` → loads Whisper, Sortformer, punctuation models into memory

---

## ⚙️ Notes

- Audio should be **16 kHz mono WAV**. Convert with ffmpeg:
  ```bash
  ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
  ```
- Whisper models (`tiny`, `base`, `small`, `medium`, `large-v2`) can be selected via `?model=...`
- Hugging Face token (`HF_TOKEN`) is required for downloading Sortformer model on first run
- GPU strongly recommended for production workloads

---

## 📜 License

MIT
