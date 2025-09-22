# tests/test_diarization.py
from diarization_service import _parse_rttm


def test_parse_rttm_line():
    # Prepare a fake RTTM content for testing
    fake_rttm_lines = [
        "SPEAKER fakefile 1 10.000 5.000 <NA> <NA> speaker_0 <NA> <NA>",
        "SPEAKER fakefile 1 15.000 3.000 <NA> <NA> speaker_1 <NA> <NA>",
    ]
    # Write to a temp file
    import tempfile

    tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    for line in fake_rttm_lines:
        tmp.write(line + "\n")
    tmp.flush()
    tmp.close()
    # Parse it
    segments = _parse_rttm(tmp.name)
    # Clean up temp file
    import os

    os.remove(tmp.name)
    # We expect two segments parsed
    assert len(segments) == 2
    seg1, seg2 = segments[0], segments[1]
    # First segment
    assert seg1["speaker"] == "Speaker 0"
    assert abs(seg1["start"] - 10.0) < 1e-6
    assert abs(seg1["end"] - 15.0) < 1e-6  # 10 + 5 duration
    # Second segment
    assert seg2["speaker"] == "Speaker 1"
    assert abs(seg2["start"] - 15.0) < 1e-6
    assert abs(seg2["end"] - 18.0) < 1e-6  # 15 + 3
