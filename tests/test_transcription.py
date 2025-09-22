# app/tests/test_transcription.py
import pytest
from app.services.transcription_service import _normalize_lang


def test_normalize_lang():
    assert _normalize_lang("English", "base") == "en"
    assert _normalize_lang("GERMAN", "large-v3") == "de"
    with pytest.raises(ValueError):
        _normalize_lang("Klingon", "base")
    assert _normalize_lang("fr", "base.en") == "en"
    assert _normalize_lang(None, "base.en") == "en"
