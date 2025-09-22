# main.py

import io
import os
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from diarization_service import diarize_audio
from pipeline_service import transcribe_and_diarize
from transcription_service import transcribe_audio
from utils import format_timestamp

app = FastAPI(title="Speech-to-Text Diarization API", version="1.0.0")


# Pydantic models for responses
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
    model: str = Query("large-v2", description="Whisper model name"),
):
    """
    Transcribe an audio file to text. Optionally specify the language (or leave None for auto-detection)
    and the Whisper model size.
    """
    # Save uploaded file to a temporary location
    contents = await file.read()
    tmpfile = f"/tmp/input_{file.filename}"
    with open(tmpfile, "wb") as f:
        f.write(contents)
    # Perform transcription
    result = transcribe_audio(tmpfile, model_name=model, language=language)
    # Remove temporary file
    try:
        os.remove(tmpfile)
    except Exception:
        pass
    # Prepare response
    segments_out = []
    for seg in result["segments"]:
        # Convert times to seconds (they are already float seconds in transcription output)
        segments_out.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"],
            }
        )
    return {
        "transcript": result["transcript"],
        "language": result.get("language"),
        "segments": segments_out,
    }


@app.post("/diarize", response_model=DiarizationResponse)
async def diarize_endpoint(
    file: UploadFile = File(...),
    domain: str = Query(
        "general", description="Domain type: 'general', 'meeting', or 'telephonic'"
    ),
):
    """
    Perform speaker diarization on an audio file. Returns a list of time segments with speaker labels.
    """
    contents = await file.read()
    tmpfile = f"/tmp/diarize_{file.filename}"
    with open(tmpfile, "wb") as f:
        f.write(contents)
    segments = diarize_audio(tmpfile, domain=domain)
    try:
        os.remove(tmpfile)
    except Exception:
        pass
    # The diarization service returns times in seconds already
    return {"segments": segments}


@app.post("/transcribe-diarize")
async def transcribe_diarize_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    model: str = Query("large-v2"),
    format: str = Query("json", description="Output format: 'json', 'txt', or 'srt'"),
    separate_music: bool = Query(
        False, description="Separate music (vocals only) before processing"
    ),
):
    """
    Perform transcription with speaker diarization on an audio file.
    Returns a speaker-labeled transcription in the requested format (JSON, plain text, or SRT).
    """
    contents = await file.read()
    tmpfile = f"/tmp/pipeline_{file.filename}"
    with open(tmpfile, "wb") as f:
        f.write(contents)
    sentences = transcribe_and_diarize(
        tmpfile, language=language, model_name=model, separate_music=separate_music
    )
    try:
        os.remove(tmpfile)
    except Exception:
        pass

    if format.lower() == "json":
        # Convert ms times to seconds float with 3 decimal precision
        output_segments = []
        for seg in sentences:
            start_sec = round(seg["start_time"] / 1000.0, 3)
            end_sec = round(seg["end_time"] / 1000.0, 3)
            output_segments.append(
                {
                    "speaker": seg["speaker"],
                    "start_time": start_sec,
                    "end_time": end_sec,
                    "text": seg["text"],
                }
            )
        return {"segments": output_segments}
    elif format.lower() == "srt":
        # Generate SRT string
        srt_lines = []
        for idx, seg in enumerate(sentences, start=1):
            start_ms = seg["start_time"]
            end_ms = seg["end_time"]
            srt_lines.append(f"{idx}")
            srt_lines.append(
                f"{format_timestamp(start_ms, always_include_hours=True, decimal_marker=',')} --> {format_timestamp(end_ms, always_include_hours=True, decimal_marker=',')}"
            )
            # Ensure no stray SRT arrow in text
            text_line = seg["speaker"] + ": " + seg["text"].strip().replace("-->", "->")
            srt_lines.append(text_line)
            srt_lines.append("")  # blank line between entries
        srt_content = "\n".join(srt_lines)
        return PlainTextResponse(content=srt_content, media_type="text/plain")
    elif format.lower() == "txt":
        # Combine segments into plain text with speaker labels
        text_output = io.StringIO()
        last_speaker = None
        for seg in sentences:
            speaker = seg["speaker"]
            sentence_text = seg["text"].strip()
            if speaker != last_speaker:
                # New paragraph for new speaker
                if last_speaker is not None:
                    text_output.write("\n")
                text_output.write(f"{speaker}: {sentence_text}")
            else:
                # Same speaker, continue in same paragraph
                text_output.write(" " + sentence_text)
            last_speaker = speaker
        content = text_output.getvalue()
        return PlainTextResponse(content=content, media_type="text/plain")
    else:
        return {"error": "Unsupported format. Choose 'json', 'txt', or 'srt'."}


@app.get("/warmup")
async def warmup_endpoint():
    """
    Load all models into memory to warm up the service (for example, to avoid cold-start delays).
    """
    # Warm up transcription (load Whisper model)
    try:
        transcribe_audio(
            "", model_name="base", language="en"
        )  # using a smaller model to quickly load something
    except Exception:
        pass
    # Warm up alignment (load an English alignment model if available)
    from whisperx import load_align_model

    try:
        load_align_model(
            language_code="en", device=("cuda" if torch.cuda.is_available() else "cpu")
        )
    except Exception:
        pass
    # Warm up punctuation model (English as example)
    from deepmultilingualpunctuation import PunctuationModel

    try:
        _ = PunctuationModel(model="kredor/punctuate-all")
    except Exception:
        pass
    # Warm up diarization (load NeMo MSDD models by running a tiny job)
    try:
        # Create a short silent audio for diarization warmup
        import numpy as np
        from scipy.io import wavfile

        sr = 16000
        wavfile.write(
            "/tmp/silence.wav", sr, np.zeros(sr // 2, dtype=np.int16)
        )  # 0.5 sec silence
        diarize_audio("/tmp/silence.wav", domain="general")
    except Exception:
        pass
    return {"status": "warmed"}
