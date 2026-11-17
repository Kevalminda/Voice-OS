import json

import pytest

from audio.transcript_normalizer import TranscriptNormalizer


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("पाइथन और जेमिनी", "Python और Gemini"),
        ("मैं वॉइस ओएस प्रोजेक्ट पर काम कर रहा हूँ।", "मैं VoiceOS project पर काम कर रहा हूँ।"),
        ("मेरा नाम केवल मिन्दा है।", "मेरा नाम Keval Minda है।"),
        ("Voice OS application", "VoiceOS application"),
        ("केवल एक नोट हटाओ", "केवल एक नोट हटाओ"),
        ("मिन्दा", "मिन्दा"),
        ("केवल मिन्दा को हटाओ", "केवल मिन्दा को हटाओ"),
        ("केवल Minda को हटाओ", "केवल Minda को हटाओ"),
        ("My name is केवल मिन्दा", "My name is Keval Minda"),
        ("केवल आज मिन्दा आया", "केवल आज मिन्दा आया"),
        ("केवल। मिन्दा", "केवल। मिन्दा"),
        ("पाइथनवाला और mypythonpackage", "पाइथनवाला और mypythonpackage"),
        ("Voice. OS", "Voice. OS"),
        ("Save note 42: don't delete Python", "Save note 42: don't delete Python"),
        ("voice-over project", "voice-over project"),
        ("", ""),
        ("Dbms microphone Faster Whisper", "DBMS microphone Faster-Whisper"),
        ("डी बी एम एस और फास्टर व्हिस्पर", "DBMS और Faster-Whisper"),
        ("faster-whisper में माइक्रोफ़ोन", "Faster-Whisper में microphone"),
        ("Faster. Whisper", "Faster. Whisper"),
        ("कल परीक्षा है और केवल एक काम बाकी है।", "कल परीक्षा है और केवल एक काम बाकी है।"),
        ("Cable Minda uses Python", "Cable Minda uses Python"),
    ],
)
def test_conservative_normalization(raw, expected):
    result = TranscriptNormalizer().normalize(raw)
    assert result.raw_text == raw
    assert result.normalized_text == expected


def test_custom_vocabulary(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(
        json.dumps(
            {
                "vocabulary": {"MyApp": ["माई ऐप"]},
                "ambiguous_terms": [],
                "name_groups": [],
            }
        ),
        encoding="utf-8",
    )
    result = TranscriptNormalizer.from_file(str(path)).normalize("माई ऐप खोलो")
    assert result.normalized_text == "MyApp खोलो"
    assert result.corrections == (("माई ऐप", "MyApp"),)


@pytest.mark.parametrize(
    "vocab",
    [
        {"A": ["same"], "B": ["same"]},
        {"A": "wrong"},
        {"A": [""]},
        {"A": ["one.two"]},
    ],
)
def test_invalid_vocabulary(vocab):
    with pytest.raises(ValueError):
        TranscriptNormalizer(vocab)


def test_longest_phrase_and_no_cascading():
    normalizer = TranscriptNormalizer({"A": ["x y"], "B": ["x"]})
    assert normalizer.normalize("x y x").normalized_text == "A B"


@pytest.mark.parametrize(
    "raw",
    [
        "My name is Mira Soren and the number is twenty seven.",
        "मेरा नाम अभिनव है और संख्या २७ है।",
        "Version 3.11 and ID 42",
    ],
)
def test_review_flags_preserve_uncertain_names_and_numeric_values(raw):
    result = TranscriptNormalizer().normalize(raw)
    assert result.raw_text == result.normalized_text == raw
    assert any(flag["kind"] == "number" for flag in result.review_flags)
    if "नाम" in raw or "name" in raw:
        assert any(flag["kind"] == "name" for flag in result.review_flags)


def test_numeric_review_uses_whole_spans_and_no_fuzzy_corrections():
    raw = "Check twenty-seven and १२३; someone uses Python."
    result = TranscriptNormalizer().normalize(raw)
    assert [flag["text"] for flag in result.review_flags if flag["kind"] == "number"] == [
        "twenty-seven",
        "१२३",
    ]
    assert result.normalized_text == raw
    assert not TranscriptNormalizer().normalize("I am testing Python").review_flags


def test_configured_name_correction_keeps_raw_and_requests_review():
    raw = "मेरा नाम केवल मिन्दा है और संख्या तीन है।"
    result = TranscriptNormalizer().normalize(raw)
    assert result.raw_text == raw
    assert result.normalized_text == "मेरा नाम Keval Minda है और संख्या तीन है।"
    assert {flag["kind"] for flag in result.review_flags} == {"name", "number"}
