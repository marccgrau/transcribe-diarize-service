# app/tests/test_diarization.py
from app.core.utils import get_words_speaker_mapping


def test_mapping_simple():
    words = [
        {"word": "Hi", "start": 0.0, "end": 0.2},
        {"word": "there.", "start": 0.2, "end": 0.6},
        {"word": "How", "start": 1.0, "end": 1.2},
        {"word": "are", "start": 1.2, "end": 1.3},
        {"word": "you?", "start": 1.3, "end": 1.6},
    ]
    spk_ts = [
        [0, 700, 0],
        [900, 1700, 1],
    ]
    m = get_words_speaker_mapping(words, spk_ts, "start")
    assert [w["speaker"] for w in m] == [0, 0, 1, 1, 1]
