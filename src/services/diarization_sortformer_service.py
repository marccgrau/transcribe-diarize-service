# src/services/diarization_sortformer_service.py
"""
Sortformer diarization via NeMo only.

- Loads nvidia/diar_sortformer_4spk-v1 (requires HF token if repo is gated).
- Accepts any audio; converts to 16 kHz mono WAV.
- Returns normalized segments: [{"speaker":"Speaker N","start":float_sec,"end":float_sec}, ...]
"""

import os
import tempfile
from typing import List, Dict
from pydub import AudioSegment

import torch
from nemo.collections.asr.models import SortformerEncLabelModel

# Cache the model in-process
_sortformer_model: SortformerEncLabelModel | None = None

# Default HF repo id (Sortformer 4 speakers). If you use another checkpoint, change here or via env.
SORTFORMER_REPO = os.getenv("SORTFORMER_REPO", "nvidia/diar_sortformer_4spk-v1")


def _ensure_wav_mono16k(in_path: str) -> str:
    """Convert any input audio to 16kHz mono wav in /tmp, return the new path."""
    out_path = os.path.join(tempfile.gettempdir(), f"sf_{os.path.basename(in_path)}.wav")
    AudioSegment.from_file(in_path).set_frame_rate(16000).set_channels(1).export(out_path, format="wav")
    return out_path


def _load_sortformer() -> SortformerEncLabelModel:
    global _sortformer_model
    if _sortformer_model is not None:
        return _sortformer_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # If model is private, ensure HF token is provided via env HF_TOKEN
    _sortformer_model = SortformerEncLabelModel.from_pretrained(SORTFORMER_REPO, map_location=device)
    _sortformer_model.eval()
    return _sortformer_model


def diarize_audio(audio_path: str, batch_size: int = 1) -> List[Dict]:
    """
    Run Sortformer diarization on a single audio file.
    Returns segments with seconds and "Speaker N" labels.
    """
    model = _load_sortformer()
    wav = _ensure_wav_mono16k(audio_path)

    # The NeMo API supports: str path, list of paths, or manifest path.
    # For a single clip, pass the path directly.
    # Output: list of tuples or dicts (begin_sec, end_sec, speaker_idx)
    predicted_segments = model.diarize(audio=wav, batch_size=batch_size)

    # Normalize to our canonical JSON schema
    norm: List[Dict] = []
    for seg in predicted_segments:
        # NVIDIA docs: format 'begin_seconds, end_seconds, speaker_index'
        # Handle tuple or dict gracefully
        if isinstance(seg, (list, tuple)) and len(seg) >= 3:
            b, e, spk = float(seg[0]), float(seg[1]), int(seg[2])
        elif isinstance(seg, dict):
            b, e = float(seg.get("begin_seconds") or seg.get("start") or 0.0), float(seg.get("end_seconds") or seg.get("end") or 0.0)
            spk = int(seg.get("speaker_index") or seg.get("speaker") or 0)
        else:
            # Unknown format; skip
            continue
        norm.append({"speaker": f"Speaker {spk}", "start": b, "end": e})
    return norm