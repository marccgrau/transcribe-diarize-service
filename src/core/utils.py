# app/core/utils.py
import os
import shutil
from math import floor

try:
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
    from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
except Exception:
    DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH = {}, {}
    LANGUAGES, TO_LANGUAGE_CODE = {}, {}

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
WAV2VEC2_LANGS = list(DEFAULT_ALIGN_MODELS_TORCH.keys()) + list(
    DEFAULT_ALIGN_MODELS_HF.keys()
)
_SENT_END = ".?!"


def format_timestamp(ms: float, always_include_hours=False, decimal_marker=",") -> str:
    ms = max(0, int(ms))
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1_000
    ms -= s * 1_000
    hh = f"{h:02d}:" if always_include_hours or h > 0 else ""
    return f"{hh}{m:02d}:{s:02d}{decimal_marker}{ms:03d}"


def find_numeral_symbol_tokens(tokenizer) -> list:
    ids = []
    for tok, tid in tokenizer.get_vocab().items():
        if any(c.isdigit() or c in "%$£" for c in tok):
            ids.append(tid)
    return ids


def _next_start(word_ts, i):
    for j in range(i + 1, len(word_ts)):
        if word_ts[j].get("start") is not None:
            return word_ts[j]["start"]
    return None


def filter_missing_timestamps(word_ts: list) -> list:
    if not word_ts:
        return []
    if word_ts[0].get("start") is None:
        word_ts[0]["start"] = 0.0
        nxt = _next_start(word_ts, 0)
        word_ts[0]["end"] = nxt if nxt is not None else 0.0
    out = [word_ts[0]]
    for i in range(1, len(word_ts)):
        w = word_ts[i]
        if w.get("start") is None and w.get("word") is not None:
            w["start"] = out[-1]["end"]
            nxt = _next_start(word_ts, i)
            w["end"] = nxt if nxt is not None else w["start"]
        if w.get("word") is not None:
            out.append(w)
    return out


def get_words_speaker_mapping(
    word_ts: list, spk_ts: list, anchor: str = "start"
) -> list:
    if not word_ts or not spk_ts:
        return []
    out = []
    idx = 0
    s, e, sp = spk_ts[0]
    for wd in word_ts:
        ws = int(floor(wd["start"] * 1000))
        we = int(floor(wd["end"] * 1000))
        a = ws if anchor == "start" else (we if anchor == "end" else (ws + we) // 2)
        while a > e and idx < len(spk_ts) - 1:
            idx += 1
            s, e, sp = spk_ts[idx]
        out.append(
            {"word": wd["word"], "start_time": ws, "end_time": we, "speaker": sp}
        )
    return out


def get_realigned_ws_mapping_with_punctuation(
    mapping: list, max_words_in_sentence: int = 60
) -> list:
    if not mapping:
        return mapping
    words = [m["word"] for m in mapping]
    spks = [m["speaker"] for m in mapping]

    def is_end(i):
        return i >= 0 and words[i] and words[i][-1] in _SENT_END

    i = 0
    n = len(mapping)
    while i < n:
        if is_end(i):
            i += 1
            continue
        # find sentence start
        j = i
        while (
            j > 0
            and j - i < max_words_in_sentence
            and spks[j - 1] == spks[j]
            and not is_end(j - 1)
        ):
            j -= 1
        # find sentence end
        k = i
        while k < n and k - j < max_words_in_sentence and not is_end(k):
            k += 1
        if k >= n:
            k = n - 1
        span = spks[j : k + 1]
        maj = max(set(span), key=span.count)
        if span.count(maj) >= len(span) // 2:
            for t in range(j, k + 1):
                spks[t] = maj
        i = k + 1
    out = []
    for t, m in enumerate(mapping):
        nm = m.copy()
        nm["speaker"] = spks[t]
        out.append(nm)
    return out


def get_sentences_speaker_mapping(mapping: list, _spk_ts_unused=None) -> list:
    if not mapping:
        return []
    sentences = []
    curr_spk = mapping[0]["speaker"]
    start = mapping[0]["start_time"]
    text = ""
    for i, wd in enumerate(mapping):
        text += wd["word"] + " "
        end = wd["end_time"]
        next_spk = mapping[i + 1]["speaker"] if i < len(mapping) - 1 else None
        is_end = wd["word"] and wd["word"][-1] in _SENT_END
        if next_spk != wd["speaker"] or is_end:
            sentences.append(
                {
                    "speaker": f"Speaker {wd['speaker']}",
                    "start_time": start,
                    "end_time": end,
                    "text": text.strip(),
                }
            )
            if i < len(mapping) - 1:
                start = mapping[i + 1]["start_time"]
                text = ""
    return sentences


def cleanup_dir(path: str):
    if not path or path in (".", "/"):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
