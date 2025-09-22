# tests/test_transcription.py
import pytest

from transcription_service import process_language_arg


def test_process_language_arg_basic():
    # Test that known language names are converted to codes
    assert process_language_arg("English", model_name="base") == "en"
    assert process_language_arg("GERMAN", model_name="large-v2") == "de"
    # Test that unsupported language raises
    with pytest.raises(ValueError):
        process_language_arg("Klingon", model_name="base")
    # Test English-only model behavior
    # If model is 'base.en', any non-en language input should default to 'en'
    assert process_language_arg("fr", model_name="base.en") == "en"
    assert process_language_arg(None, model_name="base.en") == "en"
