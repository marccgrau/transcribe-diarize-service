# src/services/diarization_sortformer_service.py
"""
Sortformer diarization via NeMo only.

- Loads nvidia/diar_sortformer_4spk-v1 (requires HF token if repo is gated).
- Accepts any audio; converts to 16 kHz mono WAV.
- Returns normalized segments: [{"speaker":"Speaker N","start":float_sec,"end":float_sec}, ...]
"""

import os
import re
import tempfile
from typing import Any, Dict, List

import torch
from loguru import logger
from nemo.collections.asr.models import SortformerEncLabelModel
from pydub import AudioSegment

# Cache the model in-process
_sortformer_model: SortformerEncLabelModel | None = None

# Default HF repo id (Sortformer 4 speakers). If you use another checkpoint, change here or via env.
SORTFORMER_REPO = os.getenv("SORTFORMER_REPO", "nvidia/diar_sortformer_4spk-v1")

# Regex for string-shaped segments, e.g. "0.720 5.120 speaker_0"
_STR_SEG_RE = re.compile(
    r"^\s*(?P<begin>\d+(?:\.\d+)?)\s+(?P<end>\d+(?:\.\d+)?)\s+(?:speaker[_\s-]?)(?P<idx>\d+)\s*$",
    re.IGNORECASE,
)


def _ensure_wav_mono16k(in_path: str) -> str:
    """Convert any input audio to 16kHz mono wav in /tmp, return the new path."""
    out_path = os.path.join(
        tempfile.gettempdir(), f"sf_{os.path.basename(in_path)}.wav"
    )
    AudioSegment.from_file(in_path).set_frame_rate(16000).set_channels(1).export(
        out_path, format="wav"
    )
    return out_path


def _load_sortformer() -> SortformerEncLabelModel:
    """Lazy-load the Sortformer model once per process."""
    global _sortformer_model
    if _sortformer_model is not None:
        return _sortformer_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("[sortformer] loading model {} on {}", SORTFORMER_REPO, device)
    _sortformer_model = SortformerEncLabelModel.from_pretrained(
        SORTFORMER_REPO, map_location=device
    )
    _sortformer_model.eval()
    logger.info("[sortformer] model ready")
    return _sortformer_model


def _parse_segment(seg: Any) -> Dict[str, float] | None:
    """
    Normalize a single diarization segment to {"speaker": "Speaker N", "start": float, "end": float}.
    Returns None if unrecognized.
    """

    # 1) tuple/list of (begin, end, speaker_idx)
    if isinstance(seg, (list, tuple)):
        # Some outputs are ["0.720 5.120 speaker_0"] (a list with a *single* string)
        if len(seg) == 1 and isinstance(seg[0], str):
            s = seg[0]
            m = _STR_SEG_RE.match(s)
            if m:
                b = float(m.group("begin"))
                e = float(m.group("end"))
                idx = int(m.group("idx"))
                return {"speaker": f"Speaker {idx}", "start": b, "end": e}
            return None

        if len(seg) >= 3:
            try:
                b = float(seg[0])
                e = float(seg[1])
                # speaker index may be int or a string like "speaker_0"
                spk = seg[2]
                if isinstance(spk, str):
                    # try to extract digits
                    m = re.search(r"(\d+)", spk)
                    idx = int(m.group(1)) if m else int(float(spk))
                else:
                    idx = int(spk)
                return {"speaker": f"Speaker {idx}", "start": b, "end": e}
            except Exception:
                return None

        # unrecognized tuple/list
        return None

    # 2) dict-like outputs
    if isinstance(seg, dict):
        # accept multiple key aliases
        b = seg.get("begin_seconds")
        e = seg.get("end_seconds")
        idx = seg.get("speaker_index")
        if b is None:
            b = seg.get("start")
        if e is None:
            e = seg.get("end")
        if idx is None:
            idx = seg.get("speaker")
        try:
            if b is not None and e is not None and idx is not None:
                # idx might be "speaker_0"
                if isinstance(idx, str):
                    m = re.search(r"(\d+)", idx)
                    idx = int(m.group(1)) if m else int(float(idx))
                else:
                    idx = int(idx)
                return {"speaker": f"Speaker {idx}", "start": float(b), "end": float(e)}
        except Exception:
            return None
        return None

    # 3) plain string: "0.720 5.120 speaker_0"
    if isinstance(seg, str):
        m = _STR_SEG_RE.match(seg)
        if m:
            b = float(m.group("begin"))
            e = float(m.group("end"))
            idx = int(m.group("idx"))
            return {"speaker": f"Speaker {idx}", "start": b, "end": e}
        return None

    # Unknown type
    return None


def diarize_audio(audio_path: str, batch_size: int = 1) -> List[Dict]:
    """
    Run Sortformer diarization on a single audio file.
    Returns segments with seconds and "Speaker N" labels.
    """
    model = _load_sortformer()
    wav = _ensure_wav_mono16k(audio_path)
    logger.info(
        "[sortformer] diarizing file={} batch_size={}",
        os.path.basename(audio_path),
        batch_size,
    )

    predicted_segments = model.diarize(audio=wav, batch_size=batch_size)

    norm: List[Dict] = []
    for raw in predicted_segments:
        parsed = _parse_segment(raw)
        if parsed is None:
            logger.warning("[sortformer] unknown segment format: {}", raw)
            continue
        norm.append(parsed)

    # Optional: sort & merge tiny gaps (keeps output tidy if needed)
    norm.sort(key=lambda x: (x["speaker"], x["start"], x["end"]))
    logger.info("[sortformer] segments={}", len(norm))
    return norm
