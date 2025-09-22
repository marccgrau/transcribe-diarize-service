# app/services/transcription_service.py
import logging

import torch
import whisperx
from app.core.utils import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
    WAV2VEC2_LANGS,
    filter_missing_timestamps,
    find_numeral_symbol_tokens,
)
from faster_whisper import WhisperModel

_whisper = None
_whisper_name = None


def _normalize_lang(language: str | None, model_name: str) -> str | None:
    if language:
        lang = language.strip().lower()
        if lang not in LANGUAGES:
            if lang in TO_LANGUAGE_CODE:
                lang = TO_LANGUAGE_CODE[lang]
            else:
                raise ValueError(f"Unsupported language: {language}")
    else:
        lang = None
    if model_name.endswith(".en"):
        return "en"
    return lang


def transcribe_audio(
    audio_path: str,
    model_name: str = "large-v3",
    language: str | None = None,
    suppress_numerals: bool = True,
) -> dict:
    global _whisper, _whisper_name
    language = _normalize_lang(language, model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _whisper is None or _whisper_name != model_name:
        compute = "int8_float16" if device == "cuda" else "int8"
        _whisper = WhisperModel(model_name, device=device, compute_type=compute)
        _whisper_name = model_name

    suppress_tokens = None
    if suppress_numerals and getattr(_whisper, "hf_tokenizer", None) is not None:
        suppress_tokens = find_numeral_symbol_tokens(_whisper.hf_tokenizer)

    # transcribe (non-batched, vad_filter to trim silences)
    seg_iter, info = _whisper.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=False,
        suppress_tokens=suppress_tokens,
        vad_filter=True,
    )
    segments_data = list(seg_iter)
    detected = info.language if info else language
    # build phrase segments
    segments = []
    full_text = ""
    for s in segments_data:
        start = float(getattr(s, "start", s["start"]))
        end = float(getattr(s, "end", s["end"]))
        text = getattr(s, "text", s["text"]).strip()
        segments.append({"start": start, "end": end, "text": text})
        full_text += text + " "

    # alignment if supported for language
    word_segments = []
    use_align = (detected in WAV2VEC2_LANGS) if detected else False
    if use_align and segments:
        try:
            align_model, meta = whisperx.load_align_model(
                language_code=detected, device=device
            )
            result = whisperx.align(
                segments_data, align_model, meta, audio_path, device=device
            )
            word_segments = filter_missing_timestamps(result["word_segments"])
        except Exception as e:
            logging.warning(
                f"Alignment failed ({detected}): {e}. Falling back to whisper word timestamps."
            )
            seg_iter2, _ = _whisper.transcribe(
                audio_path,
                language=detected,
                beam_size=5,
                word_timestamps=True,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )
            for seg in seg_iter2:
                for w in getattr(seg, "words", []) or []:
                    word_segments.append(
                        {"word": w.word, "start": float(w.start), "end": float(w.end)}
                    )
        finally:
            try:
                del align_model
            except Exception:
                pass
            torch.cuda.empty_cache()
    else:
        # try to get word timestamps directly
        seg_iter2, _ = _whisper.transcribe(
            audio_path,
            language=detected,
            beam_size=5,
            word_timestamps=True,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )
        for seg in seg_iter2:
            for w in getattr(seg, "words", []) or []:
                word_segments.append(
                    {"word": w.word, "start": float(w.start), "end": float(w.end)}
                )

    return {
        "transcript": full_text.strip(),
        "language": detected,
        "segments": segments,
        "word_segments": word_segments,
    }
