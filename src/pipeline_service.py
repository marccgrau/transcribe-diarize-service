# pipeline_service.py

import logging
import os
import re
from pathlib import Path

from deepmultilingualpunctuation import PunctuationModel

from diarization_service import diarize_audio
from transcription_service import transcribe_audio
from utils import (
    PUNCT_MODEL_LANGS,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
)

# Cache punctuation model per language to avoid reloading
_punct_models = {}


def transcribe_and_diarize(
    audio_path: str,
    language: str = None,
    model_name: str = "large-v2",
    separate_music: bool = False,
) -> list:
    """
    Run the full pipeline: transcribe the audio and then perform diarization,
    returning a list of speaker-labeled transcript segments.
    """
    # If requested, optionally perform music source separation (vocals extraction) before pipeline.
    # This can improve diarization when there's background music.
    processed_audio_path = audio_path
    if separate_music:
        try:
            import subprocess

            tmp_output_dir = audio_path + "_demucs"
            cmd = [
                "python3",
                "-m",
                "demucs.separate",
                "-n",
                "htdemucs",
                "--two-stems",
                "vocals",
                "-o",
                tmp_output_dir,
                "--filename",
                "vocals.wav",
                audio_path,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                # Replace processed_audio_path with the separated vocals track
                base_name = Path(audio_path).stem
                vocals_path = os.path.join(
                    tmp_output_dir, "htdemucs", base_name, "vocals.wav"
                )
                if os.path.exists(vocals_path):
                    processed_audio_path = vocals_path
            else:
                logging.warning(
                    "Demucs separation failed, proceeding with original audio."
                )
        except Exception as e:
            logging.warning(
                f"Music separation error: {e}. Proceeding with original audio."
            )
    # Step 1: Transcription
    transcription_result = transcribe_audio(
        processed_audio_path,
        model_name=model_name,
        language=language,
        suppress_numerals=True,
    )
    transcript_text = transcription_result["transcript"]
    word_segments = transcription_result.get("word_segments", [])
    detected_lang = transcription_result.get("language", language)
    if not transcript_text or not word_segments:
        # No transcription obtained (audio might be silent or unintelligible)
        return []

    # Step 2: Diarization
    speaker_segments = diarize_audio(processed_audio_path)

    if not speaker_segments:
        # If diarization returned nothing (shouldn't happen if there was transcript), just return transcript as single speaker
        return [
            {
                "speaker": "Speaker 0",
                "start_time": 0.0,
                "end_time": word_segments[-1]["end"] if word_segments else 0.0,
                "text": transcript_text,
            }
        ]

    # Convert speaker_segments (which uses 'start' and 'end' in seconds) to ms int for mapping
    spk_ts = []
    for seg in speaker_segments:
        s_ms = int(seg["start"] * 1000)
        e_ms = int(seg["end"] * 1000)
        # Extract speaker index number from label "Speaker X"
        spk_label = seg["speaker"]
        try:
            spk_idx = int(spk_label.split()[-1])
        except ValueError:
            spk_idx = seg["speaker"]
        spk_ts.append([s_ms, e_ms, spk_idx])

    # Map each word to a speaker
    word_spk_mapping = get_words_speaker_mapping(word_segments, spk_ts, anchor="start")

    # Step 3: Punctuation restoration for better segmentation
    if detected_lang and detected_lang.split("-")[0] in PUNCT_MODEL_LANGS:
        lang_key = detected_lang.split("-")[
            0
        ]  # use base language code (e.g., 'en' from 'en-US')
        if lang_key not in _punct_models:
            _punct_models[lang_key] = PunctuationModel(model="kredor/punctuate-all")
        punct_model = _punct_models[lang_key]
        # The punctuation model expects a list of words or a string. We'll provide the list of words to get per-word punctuation.
        words_list = [entry["word"] for entry in word_spk_mapping]
        punct_predictions = punct_model.predict(words_list)
        # Append punctuation to words based on model output
        # The predict output is a list of (word, punct) tuples
        new_mapping = []
        for word_entry, punct_tuple in zip(word_spk_mapping, punct_predictions):
            word_text = word_entry["word"]
            punct = punct_tuple[1]  # punctuation symbol predicted after this word
            if punct in ".?!":
                # Only add sentence-ending punctuation if it's not already present
                if word_text and word_text[-1] not in ".?!":
                    # Avoid adding extra period for acronyms like U.S.A.
                    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", word_text):
                        # If the word is an acronym with periods (like U.S.A.), skip adding punctuation
                        new_word = word_text
                    else:
                        new_word = word_text + punct
                else:
                    new_word = word_text
            else:
                new_word = word_text
            updated_entry = word_entry.copy()
            updated_entry["word"] = new_word
            new_mapping.append(updated_entry)
        word_spk_mapping = new_mapping
    else:
        logging.info(
            f"Punctuation model not available for language {detected_lang}, skipping punctuation restoration."
        )

    # Step 4: Realign speaker labels using punctuation cues
    word_spk_mapping = get_realigned_ws_mapping_with_punctuation(word_spk_mapping)

    # Step 5: Group words into sentences with speakers
    sentences = get_sentences_speaker_mapping(word_spk_mapping, spk_ts)
    return sentences
