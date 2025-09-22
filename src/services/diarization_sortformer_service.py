# app/services/diarization_sortformer_service.py
"""
Sortformer-first diarization service with robust fallbacks.

Priority:
1) NVIDIA NeMo 2.x Sortformer diarizer (if available)
2) Hugging Face-sortformer wrapper (if available in env)
3) pyannote.audio pipeline fallback

All return: List[{"speaker": "Speaker N", "start": float_sec, "end": float_sec}]
"""

import logging
import os
import tempfile
from typing import Dict, List

from pydub import AudioSegment

# gates
_HAS_NEMO_SORTFORMER = False
_HAS_HF_SORTFORMER = False
_HAS_PYANNOTE = False

# Try NeMo 2.x Sortformer
try:
    # NeMo 2.x often reorganized; we try an indicative import path
    # If your env exposes a different path, adapt here.
    from nemo.collections.asr.models.diarization import (
        NeuralDiarizationModel as NemoDiarModel,  # type: ignore
    )

    _HAS_NEMO_SORTFORMER = True
except Exception:
    pass

# Try a HF-native sortformer wrapper (placeholder pattern)
try:
    # hypothetical: from sortformer_diarization import SortformerPipeline
    from transformers import (
        pipeline as hf_pipeline,  # we will try a diarization pipeline name
    )

    _HAS_HF_SORTFORMER = True
except Exception:
    pass

# Fallback: pyannote
try:
    from pyannote.audio import Pipeline as PyannotePipeline

    _HAS_PYANNOTE = True
except Exception:
    pass


# cache singletons
_pyannote_pipe = None
_hf_diar_pipe = None
_nemo_model = None


def _ensure_wav_mono16k(in_path: str) -> str:
    wav = os.path.join(
        tempfile.gettempdir(), f"mono16k_{os.path.basename(in_path)}.wav"
    )
    AudioSegment.from_file(in_path).set_frame_rate(16000).set_channels(1).export(
        wav, format="wav"
    )
    return wav


def _from_nemo_sortformer(wav_path: str) -> List[Dict]:
    global _nemo_model
    if _nemo_model is None:
        # Load your specific Sortformer checkpoint or pretrained name here.
        # Example (adjust to your env): NemoDiarModel.from_pretrained("diarization_sortformer")
        _nemo_model = NemoDiarModel.from_pretrained(
            "diarization_sortformer"
        )  # <-- replace with your exact model
    diar = _nemo_model.diarize(
        [wav_path]
    )  # expected to return RTTM-like or segment dicts
    segments = []
    # Normalize output
    # Example expected: list of {"start":s, "end":e, "speaker":"SPEAKER_00"} or RTTM lines
    for seg in diar:
        s = float(seg["start"])
        e = float(seg["end"])
        spk = seg.get("speaker", "0")
        if isinstance(spk, str) and any(ch.isdigit() for ch in spk):
            try:
                spk_idx = int("".join([c for c in spk if c.isdigit()]))
            except ValueError:
                spk_idx = 0
        else:
            spk_idx = int(spk) if isinstance(spk, int) else 0
        segments.append({"speaker": f"Speaker {spk_idx}", "start": s, "end": e})
    return segments


def _from_hf_sortformer(wav_path: str) -> List[Dict]:
    global _hf_diar_pipe
    if _hf_diar_pipe is None:
        # Try to construct a diarization pipeline. Some deployments register a "diarization" task.
        # If your env provides a specific repo id, set via env DIAR_HF_MODEL.
        repo_id = os.getenv("DIAR_HF_MODEL", "nvidia/sortformer-diarization")
        try:
            _hf_diar_pipe = hf_pipeline("diarization", model=repo_id)
        except Exception as e:
            raise RuntimeError(f"HF sortformer pipeline unavailable: {e}")
    out = _hf_diar_pipe(wav_path)
    segments = []
    for seg in out:
        s = float(seg["start"])
        e = float(seg["end"])
        spk = seg.get("speaker", "0")
        spk_idx = (
            int("".join([c for c in str(spk) if c.isdigit()]))
            if any(ch.isdigit() for ch in str(spk))
            else int(spk)
        )
        segments.append({"speaker": f"Speaker {spk_idx}", "start": s, "end": e})
    return segments


def _from_pyannote(wav_path: str) -> List[Dict]:
    global _pyannote_pipe
    if _pyannote_pipe is None:
        repo_id = os.getenv("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1")
        _pyannote_pipe = PyannotePipeline.from_pretrained(
            repo_id, use_auth_token=os.getenv("HF_TOKEN", None)
        )
    diar = _pyannote_pipe(wav_path)
    segments = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": f"Speaker {speaker}",
                "start": float(turn.start),
                "end": float(turn.end),
            }
        )
    # Normalize speakers to 0..N-1 ordering
    uniq = {spk: i for i, spk in enumerate(sorted({s["speaker"] for s in segments}))}
    for s in segments:
        s["speaker"] = f"Speaker {uniq[s['speaker']]}"
    return segments


def diarize_audio(audio_path: str) -> List[Dict]:
    """
    Returns segments [{speaker, start, end}] in seconds.
    """
    wav = _ensure_wav_mono16k(audio_path)
    # priority chain
    if _HAS_NEMO_SORTFORMER:
        try:
            return _from_nemo_sortformer(wav)
        except Exception as e:
            logging.warning(f"NeMo Sortformer failed: {e}. Falling back.")
    if _HAS_HF_SORTFORMER:
        try:
            return _from_hf_sortformer(wav)
        except Exception as e:
            logging.warning(f"HF Sortformer failed: {e}. Falling back.")
    if _HAS_PYANNOTE:
        return _from_pyannote(wav)
    raise RuntimeError(
        "No diarization backend available. Please install NeMo Sortformer or pyannote.audio."
    )
