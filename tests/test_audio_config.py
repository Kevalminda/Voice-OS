import pytest

from audio.audio_config import (
    USE_CONFIG_LANGUAGE,
    AudioConfig,
    language_code,
    microphone_device,
    select_language,
)


@pytest.mark.parametrize(
    "options",
    [
        {"duration": 0},
        {"duration": float("inf")},
        {"sample_rate": 48000},
        {"device": "gpu"},
        {"language": "hinglish"},
        {"silence_rms": 0},
        {"beam_size": 0},
        {"microphone_id": True},
        {"model_size": ""},
        {"vad_threshold": 0},
        {"vad_threshold": float("nan")},
        {"multilingual": "true"},
        {"temperatures": ()},
        {"temperatures": (1.1,)},
        {"recording_mode": "always"},
        {"silence_duration": 0},
        {"max_recording_duration": 121},
        {"initial_silence_timeout": 61},
        {"capture_vad_threshold": 1},
        {"speech_start_ms": 0},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        AudioConfig(**options)


def test_env_configuration(monkeypatch):
    monkeypatch.setenv("VOICEOS_MICROPHONE", "9")
    monkeypatch.setenv("VOICEOS_STT_LANGUAGE", "hi")
    monkeypatch.setenv("VOICEOS_STT_VAD", "false")
    monkeypatch.setenv("VOICEOS_RECORD_SECONDS", "15")
    config = AudioConfig.from_env()
    assert config.microphone_id == 9
    assert config.language == "hi"
    assert config.vad_filter is False
    assert config.duration == 15
    assert language_code("auto") is None
    assert microphone_device("") is None


def test_bad_env_boolean(monkeypatch):
    monkeypatch.setenv("VOICEOS_STT_VAD", "maybe")
    with pytest.raises(ValueError):
        AudioConfig.from_env()


def test_automatic_capture_environment(monkeypatch):
    monkeypatch.setenv("VOICEOS_SILENCE_SECONDS", "2.5")
    monkeypatch.setenv("VOICEOS_MAX_RECORD_SECONDS", "45")
    monkeypatch.setenv("VOICEOS_INITIAL_SILENCE_SECONDS", "6")
    config = AudioConfig.from_env()
    assert config.recording_mode == "automatic"
    assert config.silence_duration == 2.5
    assert config.max_recording_duration == 45
    assert config.initial_silence_timeout == 6


@pytest.mark.parametrize(
    "configured,requested,mode,source",
    [
        (None, USE_CONFIG_LANGUAGE, "auto", "configuration"),
        ("en", USE_CONFIG_LANGUAGE, "en", "configuration"),
        ("hi", USE_CONFIG_LANGUAGE, "hi", "configuration"),
        ("hi", "en", "en", "call"),
        ("en", "hi", "hi", "call"),
        ("hi", "auto", "auto", "call"),
        ("en", None, "auto", "call"),
        (" EN ", USE_CONFIG_LANGUAGE, "en", "configuration"),
    ],
)
def test_selection_precedence(configured, requested, mode, source):
    selection = select_language(configured, requested)
    assert (selection.mode, selection.source) == (mode, source)
    assert selection.whisper_language == (None if mode == "auto" else mode)


@pytest.mark.parametrize("value", ["hinglish", "fr", [], {}, True, 1])
def test_invalid_language_selection(value):
    with pytest.raises(ValueError):
        select_language(None, value)
