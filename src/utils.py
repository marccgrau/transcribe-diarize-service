# utils.py

import logging
import os
import shutil
from math import floor

# Language codes and model mapping from WhisperX for alignment support
try:
    # WhisperX provides language mappings and available alignment models
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
    from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
except ImportError:
    # If whisperx is not installed or during tests, define minimal fallbacks
    DEFAULT_ALIGN_MODELS_HF = {}
    DEFAULT_ALIGN_MODELS_TORCH = {}
    LANGUAGES = {}
    TO_LANGUAGE_CODE = {}

# List of languages supported by the punctuation model (DeepMultilingualPunctuation)
PUNCT_MODEL_LANGS = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "bg",
    "pl",
    "cs",
    "sk",
    "sl",
]

# Determine which languages have an alignment model available in WhisperX
WAV2VEC2_LANGS = list(DEFAULT_ALIGN_MODELS_TORCH.keys()) + list(
    DEFAULT_ALIGN_MODELS_HF.keys()
)


def format_timestamp(
    milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."
) -> str:
    """
    Format a time in milliseconds to SRT timestamp format (HH:MM:SS,mmm).
    """
    if milliseconds < 0:
        milliseconds = 0
    hours = int(milliseconds // 3_600_000)
    milliseconds -= hours * 3_600_000
    minutes = int(milliseconds // 60_000)
    milliseconds -= minutes * 60_000
    seconds = int(milliseconds // 1_000)
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{int(milliseconds):03d}"


def find_numeral_symbol_tokens(tokenizer) -> list:
    """
    Identify token IDs in a tokenizer that represent numerals or symbols (digits, %, $, £).
    Returns a list of token IDs to suppress.
    """
    numeral_tokens = []
    for token, token_id in tokenizer.get_vocab().items():
        if any(char.isdigit() or char in "%$£" for char in token):
            numeral_tokens.append(token_id)
    return numeral_tokens


def filter_missing_timestamps(word_timestamps: list) -> list:
    """
    Fill in missing timestamps for words (if any have None start times) by
    merging them with adjacent words. Operates in-place and returns a cleaned list.
    """
    if not word_timestamps:
        return []
    # If first word is missing a start, assume it starts at 0
    if word_timestamps[0].get("start") is None:
        word_timestamps[0]["start"] = 0.0
        # set its end to the start of the next word (if available)
        next_start = _get_next_start_timestamp(word_timestamps, 0)
        word_timestamps[0]["end"] = next_start if next_start is not None else 0.0

    result = [word_timestamps[0]]
    # Iterate from second word onwards
    for i in range(1, len(word_timestamps)):
        ws = word_timestamps[i]
        if ws.get("start") is None and ws.get("word") is not None:
            # If current word has no timestamp, use previous end as start
            ws["start"] = result[-1]["end"]
            # Use next valid start as end
            next_start = _get_next_start_timestamp(word_timestamps, i)
            ws["end"] = next_start if next_start is not None else ws["start"]
        if ws.get("word") is not None:
            result.append(ws)
    return result


def _get_next_start_timestamp(word_timestamps: list, current_index: int):
    """
    Helper for filter_missing_timestamps: find the next word with a valid start time.
    Returns the start time of the next word, or None if none.
    """
    for j in range(current_index + 1, len(word_timestamps)):
        if word_timestamps[j].get("start") is not None:
            return word_timestamps[j]["start"]
    return None


def get_words_speaker_mapping(
    word_ts: list,
    speaker_ts: list,
    anchor: str = "start",
) -> list:
    """
    Map each word (with timestamp) to a speaker, given speaker turn time segments.
    `word_ts` is a list of dicts: {"word": str, "start": float_sec, "end": float_sec}
    `speaker_ts` is a list of [start_ms, end_ms, speaker_id] segments.
    `anchor` determines which point of the word to anchor for speaker decision:
       "start", "mid", or "end" of the word.
    Returns a list of dicts: {"word": str, "start_time": int_ms, "end_time": int_ms, "speaker": speaker_id}
    """
    mapping = []
    turn_index = 0
    if not word_ts or not speaker_ts:
        return mapping

    # Start with the first speaker segment
    seg_start, seg_end, seg_spk = speaker_ts[0]
    for wd in word_ts:
        # Convert word timestamps to milliseconds
        w_start = int(floor(wd["start"] * 1000))
        w_end = int(floor(wd["end"] * 1000))
        # Choose the anchor point in the word interval
        if anchor == "end":
            w_anchor = w_end
        elif anchor == "mid":
            w_anchor = (w_start + w_end) // 2
        else:  # "start"
            w_anchor = w_start

        # Advance speaker turn index until the word anchor falls within the current speaker segment
        while w_anchor > seg_end and turn_index < len(speaker_ts) - 1:
            turn_index += 1
            seg_start, seg_end, seg_spk = speaker_ts[turn_index]
        # Assign the current word to the current speaker segment
        mapping.append(
            {
                "word": wd["word"],
                "start_time": w_start,
                "end_time": w_end,
                "speaker": seg_spk,
            }
        )
    return mapping


# Sentence-ending punctuation characters
_SENT_END_CHARS = ".?!"


def get_realigned_ws_mapping_with_punctuation(
    word_spk_mapping: list, max_words_in_sentence: int = 50
) -> list:
    """
    Adjust the word-speaker mapping by reassigning entire sentences to a single speaker when possible.
    This addresses situations where a short interjection by another speaker might have been incorrectly labeled.
    It uses punctuation as a cue for sentence boundaries.
    """
    n = len(word_spk_mapping)
    if n == 0:
        return word_spk_mapping

    # Extract word list and initial speaker list
    words = [entry["word"] for entry in word_spk_mapping]
    speakers = [entry["speaker"] for entry in word_spk_mapping]

    def is_sentence_end(idx: int) -> bool:
        return idx >= 0 and words[idx] and words[idx][-1] in _SENT_END_CHARS

    # Realign speakers based on majority within sentence boundaries
    i = 0
    while i < n:
        # If current word is end of a sentence or last word, just move on
        if is_sentence_end(i):
            i += 1
            continue
        # Find sentence start (go backwards)
        j = i
        while (
            j > 0
            and j - i < max_words_in_sentence
            and speakers[j - 1] == speakers[j]
            and not is_sentence_end(j - 1)
        ):
            j -= 1
        # j is now the index of the first word of a potential sentence
        # Find sentence end (go forwards)
        k = i
        while k < n and k - j < max_words_in_sentence and not is_sentence_end(k):
            k += 1
        # k is now the index of the last word in the sentence (inclusive)
        if k >= n:
            k = n - 1
        # Determine the majority speaker in this sentence span [j, k]
        span_speakers = speakers[j : k + 1]
        majority_spk = max(set(span_speakers), key=span_speakers.count)
        # Only realign if the majority speaker constitutes more than half the sentence
        if span_speakers.count(majority_spk) >= len(span_speakers) // 2:
            for x in range(j, k + 1):
                speakers[x] = majority_spk
        i = k + 1

    # Construct realigned mapping list
    realigned = []
    for idx, entry in enumerate(word_spk_mapping):
        new_entry = entry.copy()
        new_entry["speaker"] = speakers[idx]
        realigned.append(new_entry)
    return realigned


def get_sentences_speaker_mapping(word_spk_mapping: list, speaker_ts: list) -> list:
    """
    Group word-level speaker mappings into sentence-level segments.
    Returns a list of sentences with structure:
    {"speaker": "Speaker X", "start_time": ms_int, "end_time": ms_int, "text": str}
    """
    sentences = []
    if not word_spk_mapping:
        return sentences

    # Initialize the first sentence segment
    current_spk = word_spk_mapping[0]["speaker"]
    sent_start = word_spk_mapping[0]["start_time"]
    sent_text = ""
    for i, wd in enumerate(word_spk_mapping):
        spk = wd["speaker"]
        w = wd["word"]
        # Append the word to sentence text
        sent_text += w + " "
        # Determine if this word ends a sentence (check punctuation) or speaker changes next
        last_word_of_sentence = w and w[-1] in _SENT_END_CHARS
        next_spk = (
            word_spk_mapping[i + 1]["speaker"]
            if i < len(word_spk_mapping) - 1
            else None
        )

        if next_spk != spk or last_word_of_sentence:
            # Sentence boundary reached (speaker changed or punctuation ended sentence)
            sent_end = wd["end_time"]
            sentences.append(
                {
                    "speaker": f"Speaker {spk}",
                    "start_time": sent_start,
                    "end_time": sent_end,
                    "text": sent_text.strip(),
                }
            )
            # Start a new sentence segment if more words remain
            if i < len(word_spk_mapping) - 1:
                sent_start = word_spk_mapping[i + 1]["start_time"]
                sent_text = ""
                current_spk = word_spk_mapping[i + 1]["speaker"]
    return sentences


def create_diarization_config(output_dir: str, domain_type: str = "general") -> str:
    """
    Create a NeMo diarization configuration file for the given domain type (meeting/telephonic/general).
    Downloads the config from NVIDIA's repository if not already present.
    Returns the path to the config file.
    """
    # Map domain to pre-defined config file names in NeMo examples
    domain_type = domain_type.lower()
    valid_domains = {"meeting", "telephonic", "general"}
    if domain_type not in valid_domains:
        domain_type = "general"
    config_name = f"diar_infer_{domain_type}.yaml"
    config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{config_name}"
    config_path = os.path.join(output_dir, config_name)
    if not os.path.exists(config_path):
        try:
            import wget

            wget.download(config_url, out=config_path)
        except Exception as e:
            logging.error(f"Failed to download NeMo config: {e}")
            raise
    return config_path


def cleanup_dir(path: str):
    """Delete a file or directory (recursively) if it exists."""
    if not path or path in (".", "/"):
        # Safety check: don't remove current or root directory
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
