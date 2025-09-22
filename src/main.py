# src/main.py
import io
import logging
import os
import platform
import time
from typing import List, Optional

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from pydantic import BaseModel

from src.core.utils import format_timestamp
from src.services.diarization_sortformer_service import diarize_audio
from src.services.pipeline_service import transcribe_and_diarize
from src.services.transcription_service import transcribe_audio

try:
    import torch
except Exception:
    torch = None

app = FastAPI(title="ASR + Sortformer Diarization API", version="2.0.0")
START_TIME = time.time()


# ---------- logging setup ----------
class InterceptHandler(logging.Handler):
    """Redirect stdlib 'logging' records to loguru."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.remove()
    # console sink
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # redirect std logging (incl. uvicorn) to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False


@app.on_event("startup")
def _on_startup():
    setup_logging()
    logger.info("API starting up… version={}", app.version)
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    logger.info(
        "Runtime: python={} torch={} cuda={} device={}",
        platform.python_version(),
        getattr(torch, "__version__", None) if torch else None,
        getattr(torch.version, "cuda", None)
        if (torch and torch.cuda.is_available())
        else None,
        device,
    )


# simple request timing middleware (logs method, path, status, ms)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    except Exception:
        # log the original exception but re-raise so the client sees 500
        ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "Unhandled error during request: {} {} [{:.1f} ms]",
            request.method,
            request.url.path,
            ms,
        )
        raise
    finally:
        ms = (time.perf_counter() - start) * 1000
        status = getattr(response, "status_code", 500)
        logger.info(
            "{method} {path} -> {status} [{ms:.1f} ms]",
            method=request.method,
            path=request.url.path,
            status=status,
            ms=ms,
        )


# ---------- models ----------
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


# ---------- endpoints ----------
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    model: str = Query("base"),
):
    tmp = f"/tmp/{file.filename}"
    logger.info(
        "POST /transcribe file={} lang={} model={}", file.filename, language, model
    )
    with open(tmp, "wb") as f:
        f.write(await file.read())
    logger.debug("Saved upload -> {}", tmp)
    try:
        res = transcribe_audio(tmp, model_name=model, language=language)
        logger.info(
            "Transcription done: segments={} language={}",
            len(res.get("segments", [])),
            res.get("language"),
        )
        return {
            "transcript": res["transcript"],
            "language": res.get("language"),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in res["segments"]
            ],
        }
    finally:
        try:
            os.remove(tmp)
            logger.debug("Deleted temp file {}", tmp)
        except Exception:
            logger.warning("Failed to delete temp file {}", tmp)


@app.post("/diarize", response_model=DiarizationResponse)
async def diarize_endpoint(file: UploadFile = File(...)):
    tmp = f"/tmp/{file.filename}"
    logger.info("POST /diarize file={}", file.filename)
    with open(tmp, "wb") as f:
        f.write(await file.read())
    logger.debug("Saved upload -> {}", tmp)
    try:
        segs = diarize_audio(tmp)
        logger.info("Diarization done: segments={}", len(segs))
        return {"segments": segs}
    finally:
        try:
            os.remove(tmp)
            logger.debug("Deleted temp file {}", tmp)
        except Exception:
            logger.warning("Failed to delete temp file {}", tmp)


@app.post("/transcribe-diarize")
async def transcribe_diarize_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    model: str = Query("base"),
    format: str = Query("json", description="json|txt|srt"),
):
    tmp = f"/tmp/{file.filename}"
    logger.info(
        "POST /transcribe-diarize file={} lang={} model={} format={}",
        file.filename,
        language,
        model,
        format,
    )
    with open(tmp, "wb") as f:
        f.write(await file.read())
    logger.debug("Saved upload -> {}", tmp)
    try:
        sents = transcribe_and_diarize(tmp, language=language, model_name=model)
        logger.info("Pipeline done: sentence_segments={}", len(sents))

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
                lines.append(
                    f"{s['speaker']}: {s['text'].strip().replace('-->', '->')}"
                )
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
        logger.warning("Invalid format requested: {}", format)
        return {"error": "format must be json|txt|srt"}
    finally:
        try:
            os.remove(tmp)
            logger.debug("Deleted temp file {}", tmp)
        except Exception:
            logger.warning("Failed to delete temp file {}", tmp)


@app.get("/warmup")
async def warmup():
    logger.info("GET /warmup starting…")
    dummy = "data/test-recording.wav"
    # whisper
    try:
        import numpy as np
        from scipy.io import wavfile

        wavfile.write(dummy, 16000, (np.zeros(8000)).astype("float32"))
        transcribe_audio(dummy, model_name="base", language="en")
        logger.info("Warmup: whisper OK")
    except Exception:
        logger.exception("Warmup: whisper failed")

    # punctuation
    try:
        from deepmultilingualpunctuation import PunctuationModel

        _ = PunctuationModel(model="kredor/punctuate-all")
        logger.info("Warmup: punctuation OK")
    except Exception:
        logger.exception("Warmup: punctuation failed")

    # diarization
    try:
        from src.services.diarization_sortformer_service import (
            _load_sortformer,
            diarize_audio,
        )

        _load_sortformer()
        import numpy as np
        from scipy.io import wavfile

        sr = 16000
        wavfile.write(dummy, sr, np.zeros(sr // 2, dtype=np.int16))
        diarize_audio(dummy)
        logger.info("Warmup: sortformer OK")
    except Exception:
        logger.exception("Warmup: sortformer failed")
    finally:
        try:
            os.remove(dummy)
            logger.debug("Warmup: deleted {}", dummy)
        except Exception:
            logger.warning("Warmup: failed to delete {}", dummy)
    return {"status": "warmed"}


@app.get("/health", summary="Lightweight health probe")
def health():
    """Simple liveness/readiness check for orchestrators and manual smoke tests."""
    uptime = round(time.time() - START_TIME, 2)
    torch_version = getattr(torch, "__version__", None) if torch else None
    cuda_version = (
        getattr(torch.version, "cuda", None)
        if (torch and torch.cuda.is_available())
        else None
    )
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    payload = {
        "status": "ok",
        "uptime_seconds": uptime,
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda": cuda_version,
        "device": device,
        "pid": os.getpid(),
    }
    logger.debug("Health probe -> {}", payload)
    return JSONResponse(payload)
