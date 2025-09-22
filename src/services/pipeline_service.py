# src/services/pipeline_service.py
import re
from typing import Any, Iterable, Tuple

from deepmultilingualpunctuation import PunctuationModel
from loguru import logger

from src.core.utils import (
    PUNCT_MODEL_LANGS,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
)
from src.services.diarization_sortformer_service import diarize_audio
from src.services.transcription_service import transcribe_audio

_punct_cache: dict[str, PunctuationModel] = {}

_ENDINGS = {".", "?", "!"}


def _extract_tok_punct(item: Any) -> Tuple[str, str]:
    """
    Normalize outputs from deepmultilingualpunctuation.PunctuationModel.predict.

    Accepts:
      - (token, punct)
      - (token, punct, confidence)
      - {"token":..., "punct":...} or common aliases
      - fallback to ("", "O") when unknown
    """
    if isinstance(item, (list, tuple)):
        if len(item) >= 2:
            tok = str(item[0])
            punct = str(item[1])
            return tok, punct
        elif len(item) == 1:
            return str(item[0]), "O"
        return "", "O"

    if isinstance(item, dict):
        tok = str(item.get("token") or item.get("word") or item.get("text") or "")
        punct = str(
            item.get("punct")
            or item.get("label")
            or item.get("prediction")
            or item.get("punc")
            or "O"
        )
        return tok, punct

    # string / other types
    try:
        s = str(item)
    except Exception:
        return "", "O"
    return s, "O"


def _iter_tok_punct(preds: Iterable[Any]) -> Iterable[Tuple[str, str]]:
    for p in preds:
        yield _extract_tok_punct(p)


def transcribe_and_diarize(
    audio_path: str,
    language: str | None = None,
    model_name: str = "base",
    separate_music: bool = False,
) -> list:
    logger.info(
        "[pipeline] transcribe+diarize start lang={} model={}", language, model_name
    )

    tr = transcribe_audio(
        audio_path, model_name=model_name, language=language, suppress_numerals=True
    )
    words = tr.get("word_segments", [])
    if not tr.get("transcript") or not words:
        logger.warning("[pipeline] empty transcript or no word timestamps")
        return []

    # Diarize
    spk_segments = diarize_audio(audio_path)
    spk_ts = []
    for s in spk_segments:
        spk_idx = int(s["speaker"].split()[-1])
        spk_ts.append([int(s["start"] * 1000), int(s["end"] * 1000), spk_idx])

    # Map words -> speakers
    mapping = get_words_speaker_mapping(words, spk_ts, anchor="start")

    # Optional punctuation restoration
    lang = (tr.get("language") or "en").split("-")[0]
    if lang in PUNCT_MODEL_LANGS:
        if lang not in _punct_cache:
            _punct_cache[lang] = PunctuationModel(model="kredor/punctuate-all")
        pm = _punct_cache[lang]

        tokens = [m["word"] for m in mapping]
        preds = list(_iter_tok_punct(pm.predict(tokens)))

        if len(preds) != len(mapping):
            logger.warning(
                "[pipeline] punctuation length mismatch: tokens={} preds={}",
                len(mapping),
                len(preds),
            )

        updated = []
        for m, (tok, punct) in zip(mapping, preds):
            w = m["word"]
            # The model may emit 'O' (no punct) or ',', '.', '?', '!' etc.
            if punct in _ENDINGS and w and w[-1] not in _ENDINGS:
                # avoid acronyms like U.S.A.
                if re.fullmatch(r"(?:[A-Za-z]\.){2,}", w):
                    new_w = w
                else:
                    new_w = w + punct
            else:
                new_w = w
            nm = m.copy()
            nm["word"] = new_w
            updated.append(nm)

        # If preds shorter than mapping, append the remainder unchanged
        if len(updated) < len(mapping):
            updated.extend(mapping[len(updated) :])

        mapping = updated
    else:
        logger.info("[pipeline] no punctuation model for {}", lang)

    # Realign minor boundary slips, then sentence-ize
    mapping = get_realigned_ws_mapping_with_punctuation(mapping)
    sents = get_sentences_speaker_mapping(mapping, spk_ts)
    logger.info("[pipeline] produced {} sentences", len(sents))
    return sents
