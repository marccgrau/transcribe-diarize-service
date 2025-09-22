# src/services/pipeline_service.py
import logging
import re

from src.core.utils import (
    PUNCT_MODEL_LANGS,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
)
from src.services.diarization_sortformer_service import diarize_audio
from src.services.transcription_service import transcribe_audio
from deepmultilingualpunctuation import PunctuationModel

_punct_cache = {}


def transcribe_and_diarize(
    audio_path: str,
    language: str | None = None,
    model_name: str = "large-v3",
    separate_music: bool = False,
) -> list:
    # optional music separation dropped for simplicity/latency; Sortformer is robust on speech
    tr = transcribe_audio(
        audio_path, model_name=model_name, language=language, suppress_numerals=True
    )
    words = tr.get("word_segments", [])
    if not tr["transcript"] or not words:
        return []
    spk_segments = diarize_audio(audio_path)
    # build ms speaker timeline
    spk_ts = []
    for s in spk_segments:
        spk_idx = int(s["speaker"].split()[-1])
        spk_ts.append([int(s["start"] * 1000), int(s["end"] * 1000), spk_idx])

    mapping = get_words_speaker_mapping(words, spk_ts, anchor="start")

    # punctuation restoration
    lang = (tr.get("language") or "en").split("-")[0]
    if lang in PUNCT_MODEL_LANGS:
        if lang not in _punct_cache:
            _punct_cache[lang] = PunctuationModel(model="kredor/punctuate-all")
        pm = _punct_cache[lang]
        tokens = [m["word"] for m in mapping]
        preds = pm.predict(tokens)
        updated = []
        for m, (tok, punct) in zip(mapping, preds):
            w = m["word"]
            if punct in ".?!" and w and w[-1] not in ".?!":
                # avoid U.S.A.
                if re.fullmatch(r"(?:[A-Za-z]\.){2,}", w):
                    new_w = w
                else:
                    new_w = w + punct
            else:
                new_w = w
            nm = m.copy()
            nm["word"] = new_w
            updated.append(nm)
        mapping = updated
    else:
        logging.info("No punctuation restoration for language: %s", lang)

    mapping = get_realigned_ws_mapping_with_punctuation(mapping)
    sents = get_sentences_speaker_mapping(mapping, spk_ts)
    return sents
