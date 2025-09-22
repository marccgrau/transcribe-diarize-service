# transcription_service.py

import logging

import torch
import whisperx
from faster_whisper import WhisperModel

from utils import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
    WAV2VEC2_LANGS,
    filter_missing_timestamps,
    find_numeral_symbol_tokens,
)

# Global model instances for caching
_whisper_model = None
_whisper_model_name = None


def process_language_arg(language: str, model_name: str) -> str:
    """
    Normalize the language argument to a valid language code for Whisper.
    Converts full language names to codes and handles English-only model cases.
    """
    if language:
        language = language.strip().lower()
    # If language is a name, convert to code; if it's already code, ensure it is in LANGUAGES
    if language and language not in LANGUAGES:
        if language in TO_LANGUAGE_CODE:
            language = TO_LANGUAGE_CODE[language]  # convert name to code
        else:
            raise ValueError(f"Unsupported language: {language}")
    # If using an English-only model, override language to English
    if model_name.endswith(".en"):
        if language and language != "en":
            logging.warning(
                f"Model {model_name} is English-only, ignoring requested language '{language}'. Using 'en'."
            )
        language = "en"
    return language or None  # Whisper can auto-detect if None


def transcribe_audio(
    audio_path: str,
    model_name: str = "large-v2",
    language: str = None,
    suppress_numerals: bool = True,
) -> dict:
    """
    Transcribe the given audio file using Whisper. Optionally perform word-level alignment.
    Returns a dictionary with transcribed text, detected language, and segments.
    """
    global _whisper_model, _whisper_model_name

    # Normalize language input
    language = process_language_arg(language, model_name)
    use_alignment = False
    if language:
        use_alignment = language in WAV2VEC2_LANGS
    else:
        # If language is None, Whisper will detect it; we will decide on alignment after detection
        use_alignment = True  # assume we can align if detected language is supported

    # Lazy-load Whisper model if not already loaded or if a different model is requested
    if _whisper_model is None or _whisper_model_name != model_name:
        # Determine compute precision
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = (
            "int8_float16" if device == "cuda" else "int8"
        )  # int8 precision on CPU, mixed on GPU
        _whisper_model = WhisperModel(
            model_name, device=device, compute_type=compute_type
        )
        _whisper_model_name = model_name

    # Prepare numeral token suppression if needed
    suppress_tokens = None
    if suppress_numerals:
        try:
            tokenizer = (
                _whisper_model.hf_tokenizer
            )  # faster_whisper exposes the HF tokenizer
        except AttributeError:
            tokenizer = None
        if tokenizer:
            suppress_tokens = find_numeral_symbol_tokens(tokenizer)

    # Perform transcription using faster-whisper
    segments = []
    full_text = ""
    language_detected = None

    # We will transcribe in chunks if audio is long to avoid out-of-memory.
    # Use whisperx's load_audio to get waveform and length, then decide chunking.
    audio = whisperx.load_audio(audio_path)
    duration = (
        len(audio) / 16000.0
    )  # assuming whisperx returns 16k sampled audio by default
    # If duration is very long, consider chunking:
    if duration > 120 and hasattr(_whisper_model, "transcribe"):
        # Use batched transcription from whisperx for long files
        result = whisperx.transcribe_with_faster_whisper(
            _whisper_model, audio, language=language, suppress_tokens=suppress_tokens
        )
        # The whisperx helper returns a dict with 'segments' and 'language'
        segments_data = result["segments"]
        language_detected = result.get("language", None)
    else:
        # Use direct transcription
        seg_iterator, info = _whisper_model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=False,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )
        segments_data = list(seg_iterator)
        language_detected = info.language if info else None

    # Use detected language if not provided
    if language_detected and language is None:
        language = language_detected
        if language not in WAV2VEC2_LANGS:
            use_alignment = False

    # Build segments output (without word-level timestamps yet)
    for seg in segments_data:
        # Each segment could be a tuple or Struct with .start, .end, .text
        seg_start = float(seg.start) if hasattr(seg, "start") else float(seg["start"])
        seg_end = float(seg.end) if hasattr(seg, "end") else float(seg["end"])
        seg_text = seg.text if hasattr(seg, "text") else seg["text"]
        segments.append({"start": seg_start, "end": seg_end, "text": seg_text.strip()})
        full_text += seg_text + " "

    # If alignment is available for this language, refine word timestamps using WhisperX
    word_segments = []
    if use_alignment and segments:
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=language or "en",
                device=("cuda" if torch.cuda.is_available() else "cpu"),
            )
            result_aligned = whisperx.align(
                segments_data,
                align_model,
                metadata,
                audio_path,
                device=("cuda" if torch.cuda.is_available() else "cpu"),
            )
            word_segments = filter_missing_timestamps(result_aligned["word_segments"])
        except Exception as e:
            logging.warning(
                f"Alignment model failed or not available for language {language}: {e}"
            )
            # If alignment fails, try to get word timestamps from Whisper (if model supported it)
            if hasattr(_whisper_model, "transcribe"):
                # Re-run transcription with word_timestamps=True
                seg_iter, info = _whisper_model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,
                    word_timestamps=True,
                    suppress_tokens=suppress_tokens,
                    vad_filter=True,
                )
                segments_data = list(seg_iter)
                word_segments = []
                for seg in segments_data:
                    if hasattr(seg, "words") and seg.words:
                        for w in seg.words:
                            # w may be a tuple (start, end, word) or an object with .start, .end, .word
                            w_start = (
                                float(w.start) if hasattr(w, "start") else float(w[0])
                            )
                            w_end = float(w.end) if hasattr(w, "end") else float(w[1])
                            w_text = w.word if hasattr(w, "word") else w[2]
                            word_segments.append(
                                {"word": w_text, "start": w_start, "end": w_end}
                            )
        finally:
            # Free alignment model to save memory (if loaded)
            try:
                del align_model
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()
    else:
        # If not using alignment but Whisper model did not provide word timestamps, we can derive them if possible
        # (FasterWhisper provides word timestamps only if word_timestamps=True)
        word_segments = []
        if segments_data and hasattr(segments_data[0], "words"):
            # If we have word-level info from an initial transcription (not aligned)
            for seg in segments_data:
                for w in seg.words:
                    w_start = float(w.start) if hasattr(w, "start") else float(w[0])
                    w_end = float(w.end) if hasattr(w, "end") else float(w[1])
                    w_text = w.word if hasattr(w, "word") else w[2]
                    word_segments.append(
                        {"word": w_text, "start": w_start, "end": w_end}
                    )

    # Final result structure
    return {
        "transcript": full_text.strip(),
        "language": language or language_detected or None,
        "segments": segments,
        "word_segments": word_segments,
    }
