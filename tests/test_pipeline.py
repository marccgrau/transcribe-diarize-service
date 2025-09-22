# tests/test_pipeline.py
from utils import (
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
)


def test_word_speaker_mapping_and_realignment():
    # Simulate word timestamps (in seconds) and speaker segments (in ms)
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world.", "start": 0.5, "end": 1.0},  # sentence ends here
        {"word": "How", "start": 1.2, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 1.7},
        {"word": "you", "start": 1.7, "end": 1.9},
        {"word": "doing?", "start": 1.9, "end": 2.2},  # sentence ends here
    ]
    # Speaker segments in milliseconds (two speakers alternating)
    speaker_ts = [
        [0, 1000, 0],  # Speaker 0 from 0s to 1s
        [1200, 2000, 1],  # Speaker 1 from 1.2s to 2.0s (1000 ms = 1s, etc.)
    ]
    mapping = get_words_speaker_mapping(words, speaker_ts, anchor="start")
    # Initially, "Hello" and "world." should map to speaker 0, "How are you doing?" to speaker 1
    assert mapping[0]["speaker"] == 0
    assert mapping[1]["speaker"] == 0
    assert mapping[2]["speaker"] == 1
    assert mapping[-1]["speaker"] == 1
    # Now imagine a scenario where a word in the second sentence was mis-assigned speaker 0 (simulate error)
    mapping[3]["speaker"] = 0  # mis-assigned
    mapping[4]["speaker"] = 1
    mapping[5]["speaker"] = 1
    realigned = get_realigned_ws_mapping_with_punctuation(mapping)
    # The second sentence "How are you doing?" should be majority speaker 1, so all words set to 1
    for w in realigned[2:]:
        assert w["speaker"] == 1


def test_sentence_grouping():
    word_spk_map = [
        {"word": "Good", "start_time": 0, "end_time": 500, "speaker": 0},
        {"word": "morning.", "start_time": 500, "end_time": 1000, "speaker": 0},
        {"word": "Thank", "start_time": 1500, "end_time": 1700, "speaker": 1},
        {"word": "you.", "start_time": 1700, "end_time": 1900, "speaker": 1},
        {"word": "Goodbye.", "start_time": 2000, "end_time": 2500, "speaker": 0},
    ]
    sentences = get_sentences_speaker_mapping(word_spk_map, speaker_ts=[])
    # We expect three sentence segments:
    # Speaker 0: "Good morning."
    # Speaker 1: "Thank you."
    # Speaker 0: "Goodbye."
    assert len(sentences) == 3
    assert sentences[0]["speaker"] == "Speaker 0"
    assert sentences[0]["text"].strip() == "Good morning."
    assert sentences[1]["speaker"] == "Speaker 1"
    assert sentences[1]["text"].strip() == "Thank you."
    assert sentences[2]["speaker"] == "Speaker 0"
    assert sentences[2]["text"].strip() == "Goodbye."
