# src/main.py
import io
import os
from typing import List, Optional

from src.core.utils import format_timestamp
from src.services.diarization_sortformer_service import diarize_audio
from src.services.pipeline_service import transcribe_and_diarize
from src.services.transcription_service import transcribe_audio
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

app = FastAPI(title="ASR + Sortformer Diarization API", version="2.0.0")


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    transcript: str
    language: Optional[str]
    segments: List[TranscriptionSegment]


class DiarizationSegment(BaseModel):
    speaker: str
    start: float
    end: float


class DiarizationResponse(BaseModel):
    segments: List[DiarizationSegment]


class DiarizedTranscriptSegment(BaseModel):
    speaker: str
    start_time: float
    end_time: float
    text: str


class DiarizedTranscriptResponse(BaseModel):
    segments: List[DiarizedTranscriptSegment]


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    model: str = Query("large-v3"),
):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    res = transcribe_audio(tmp, model_name=model, language=language)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return {
        "transcript": res["transcript"],
        "language": res.get("language"),
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in res["segments"]
        ],
    }


@app.post("/diarize", response_model=DiarizationResponse)
async def diarize_endpoint(file: UploadFile = File(...)):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    segs = diarize_audio(tmp)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return {"segments": segs}


@app.post("/transcribe-diarize")
async def transcribe_diarize_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    model: str = Query("large-v3"),
    format: str = Query("json", description="json|txt|srt"),
):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    sents = transcribe_and_diarize(tmp, language=language, model_name=model)
    try:
        os.remove(tmp)
    except Exception:
        pass

    if format == "json":
        return {
            "segments": [
                {
                    "speaker": s["speaker"],
                    "start_time": round(s["start_time"] / 1000.0, 3),
                    "end_time": round(s["end_time"] / 1000.0, 3),
                    "text": s["text"],
                }
                for s in sents
            ]
        }
    if format == "srt":
        lines = []
        for i, s in enumerate(sents, 1):
            lines.append(str(i))
            lines.append(
                f"{format_timestamp(s['start_time'], True)} --> {format_timestamp(s['end_time'], True)}"
            )
            lines.append(f"{s['speaker']}: {s['text'].strip().replace('-->', '->')}")
            lines.append("")
        return PlainTextResponse("\n".join(lines), media_type="text/plain")
    if format == "txt":
        buf = io.StringIO()
        prev = None
        for s in sents:
            if s["speaker"] != prev and prev is not None:
                buf.write("\n")
            if s["speaker"] != prev:
                buf.write(f"{s['speaker']}: {s['text']}")
            else:
                buf.write(" " + s["text"])
            prev = s["speaker"]
        return PlainTextResponse(buf.getvalue(), media_type="text/plain")
    return {"error": "format must be json|txt|srt"}


@app.get("/warmup")
async def warmup():
    # whisper
    try:
        dummy = "/data/test-recording.wav"
        import numpy as np
        from scipy.io import wavfile

        wavfile.write(dummy, 16000, (np.zeros(8000)).astype("float32"))
        transcribe_audio(dummy, model_name="base", language="en")
    except Exception:
        pass
    # punctuation
    try:
        from deepmultilingualpunctuation import PunctuationModel

        _ = PunctuationModel(model="kredor/punctuate-all")
    except Exception:
        pass
    # diarization
    try:
        from src.services.diarization_sortformer_service import diarize_audio, _load_sortformer
        _load_sortformer()
        # quick 0.5s silent wav to initialize kernels/caches
        import numpy as np
        from scipy.io import wavfile
        dummy = "/data/test-recording.wav"
        sr = 16000
        wavfile.write(dummy, sr, np.zeros(sr//2, dtype=np.int16))
        diarize_audio(dummy)
    except Exception:
        pass
    try:
        os.remove(dummy)
    except Exception:
        pass
    return {"status": "warmed"}

@app.get("/health", summary="Lightweight health probe")
def health():
    """Simple liveness/readiness check for orchestrators and manual smoke tests."""
    uptime = round(time.time() - START_TIME, 2)
    torch_version = getattr(torch, "__version__", None) if torch else None
    cuda_version = getattr(torch.version, "cuda", None) if (torch and torch.cuda.is_available()) else None
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    return JSONResponse({
        "status": "ok",
        "uptime_seconds": uptime,
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda": cuda_version,
        "device": device,
        "pid": os.getpid(),
    })