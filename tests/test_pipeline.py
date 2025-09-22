# app/tests/test_pipeline.py
from app.core.utils import (
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
)


def test_realign_and_group():
    mapping = [
        {"word": "Good", "start_time": 0, "end_time": 200, "speaker": 0},
        {"word": "morning.", "start_time": 200, "end_time": 600, "speaker": 0},
        {"word": "How", "start_time": 1000, "end_time": 1200, "speaker": 1},
        {"word": "are", "start_time": 1200, "end_time": 1300, "speaker": 0},  # mislabel
        {"word": "you?", "start_time": 1300, "end_time": 1500, "speaker": 1},
    ]
    realigned = get_realigned_ws_mapping_with_punctuation(mapping)
    # second sentence should be speaker 1 throughout
    assert all(w["speaker"] == 1 for w in realigned[2:])
    sents = get_sentences_speaker_mapping(realigned, [])
    assert len(sents) == 2
    assert sents[0]["speaker"] == "Speaker 0"
    assert sents[1]["speaker"] == "Speaker 1"
