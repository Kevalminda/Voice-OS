from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio import AudioConfig, SpeechToText
from audio.diagnostics import (
    compare_modes,
    compare_stress_cases,
    load_wav,
    save_wav,
    stress_audio_cases,
)
from audio.speech_to_text import AudioCaptureError, ModelInitializationError, SpeechToTextError


@pytest.fixture
def samples():
    # Synthetic signal for plumbing tests, never an accuracy evaluation.
    return np.concatenate((np.full(8000, 0.05, dtype=np.float32), np.zeros(8000, dtype=np.float32)))


@pytest.fixture
def stt(monkeypatch):
    engine = SpeechToText(config=AudioConfig(device="cpu"))
    model = MagicMock()
    model.transcribe.side_effect = lambda audio, **kwargs: (
        iter([SimpleNamespace(text=" Python प्रोजेक्ट ")]),
        SimpleNamespace(language=kwargs["language"] or "hi", language_probability=0.8),
    )
    engine.model = model
    engine.execution_device, engine.compute_type = "cpu", "int8"
    return engine


@pytest.mark.parametrize(
    "audio",
    [
        np.array([], dtype=np.float32),
        np.zeros(16000, dtype=np.float32),
        np.full(16000, 1e-5, dtype=np.float32),
    ],
)
def test_empty_and_near_silent(stt, audio):
    result = stt.transcribe(audio)
    assert result["status"] == "no_speech"
    assert result["empty_transcription"]
    assert result["raw_text"] == result["normalized_text"] == ""
    assert result["language_probability"] is None
    stt.model.transcribe.assert_not_called()


def test_review_metadata_is_serializable_and_keeps_raw_text(stt, samples):
    import json

    raw = "My name is Mira and the ID is २७."
    stt.model.transcribe.side_effect = lambda audio, **kwargs: (
        iter([SimpleNamespace(text=raw)]),
        SimpleNamespace(language="hi", language_probability=0.8),
    )
    result = stt.transcribe(samples)
    assert result["raw_text"] == result["normalized_text"] == raw
    assert {flag["kind"] for flag in result["review_flags"]} == {"name", "number"}
    assert json.loads(json.dumps(result, ensure_ascii=False))["raw_text"] == raw


@pytest.mark.parametrize("language", ["en", "hi", None])
def test_language_modes_and_raw_preservation(stt, samples, language):
    result = stt.transcribe(samples, language)
    assert result["raw_text"] == "Python प्रोजेक्ट"
    assert result["normalized_text"] == result["text"] == "Python project"
    assert result["language"] == (language or "hi")
    assert result["language_probability"] == 0.8
    assert result["recording_duration"] == 1
    assert result["transcription_duration"] > 0
    options = stt.model.transcribe.call_args.kwargs
    assert options["language"] == language
    assert options["task"] == "transcribe"
    assert options["vad_parameters"]["speech_pad_ms"] == 300


def test_env_language_explicit_auto_override(monkeypatch, stt, samples):
    from dataclasses import replace

    stt.config = replace(stt.config, language="hi")
    stt.transcribe(samples)
    assert stt.model.transcribe.call_args.kwargs["language"] == "hi"
    stt.transcribe(samples, None)
    assert stt.model.transcribe.call_args.kwargs["language"] is None


def test_empty_whisper_output(stt, samples):
    stt.model.transcribe.side_effect = None
    stt.model.transcribe.return_value = (
        iter([SimpleNamespace(text=" ")]),
        SimpleNamespace(language="en", language_probability=1),
    )
    assert stt.transcribe(samples)["status"] == "empty_output"


@pytest.mark.parametrize(
    "audio",
    [
        np.array([1], dtype=np.int16),
        np.zeros((2, 2)),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1.1]),
    ],
)
def test_invalid_audio(stt, audio):
    with pytest.raises(ValueError):
        stt.transcribe(audio)


@pytest.fixture
def microphone(monkeypatch):
    device = MagicMock()
    device.query_devices.return_value = {"max_input_channels": 2}
    device.rec.side_effect = lambda frames, **kwargs: np.zeros((frames, 1), dtype=np.float32)
    device.wait.return_value = None
    monkeypatch.setattr("audio.speech_to_text._sounddevice", lambda: device)
    return device


def test_capture_configurable_device_duration(microphone):
    stt = SpeechToText(
        config=AudioConfig(microphone_id="My Microphone", duration=8, recording_mode="fixed")
    )
    audio = stt.record_audio()
    assert audio.shape == (128000,)
    assert audio.dtype == np.float32
    assert microphone.rec.call_args.kwargs == {
        "samplerate": 16000,
        "channels": 1,
        "dtype": "float32",
        "device": "My Microphone",
    }


@pytest.mark.parametrize("failure", ["output", "invalid", "unsupported", "overflow", "short"])
def test_microphone_errors(microphone, failure):
    if failure == "output":
        microphone.query_devices.return_value = {"max_input_channels": 0}
    elif failure == "invalid":
        microphone.query_devices.side_effect = RuntimeError("Invalid device")
    elif failure == "unsupported":
        microphone.check_input_settings.side_effect = RuntimeError("unsupported rate")
    elif failure == "overflow":
        microphone.wait.return_value = True
    else:
        microphone.rec.return_value = np.zeros((10, 1), dtype=np.float32)
        microphone.rec.side_effect = None
    with pytest.raises(AudioCaptureError):
        SpeechToText(config=AudioConfig(microphone_id=999, recording_mode="fixed")).record_audio()


@pytest.mark.parametrize("duration", [0, -1, float("nan"), 121])
def test_invalid_capture_duration(stt, duration):
    with pytest.raises(ValueError):
        stt.record_audio(duration)


def test_model_loaded_once(monkeypatch, samples, stt):
    model = stt.model
    engine = SpeechToText(config=AudioConfig(device="cpu"))
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(engine, "_create_model", factory)
    engine.transcribe(samples)
    second = engine.transcribe(samples)
    factory.assert_called_once_with("cpu", "float32")
    assert second["model_load_duration"] == 0


def test_model_error_is_bounded(monkeypatch, samples):
    engine = SpeechToText(config=AudioConfig(device="cpu"))
    factory = MagicMock(side_effect=OSError("Missing model"))
    monkeypatch.setattr(engine, "_create_model", factory)
    for _ in range(2):
        with pytest.raises(ModelInitializationError, match="initialize"):
            engine.transcribe(samples)
    assert factory.call_count == 1


def test_cuda_initialization_fallback(monkeypatch, samples, stt):
    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda device: {"int8_float16"})
    engine = SpeechToText(config=AudioConfig())
    factory = MagicMock(side_effect=[RuntimeError("CUDA unavailable"), stt.model])
    monkeypatch.setattr(engine, "_create_model", factory)
    result = engine.transcribe(samples)
    assert result["execution_device"] == "cpu"
    assert factory.call_args_list[0].args == ("cuda", "int8_float16")
    assert factory.call_args_list[1].args == ("cpu", "float32")


def test_lazy_cuda_failure_fallback_once(monkeypatch, samples, stt):
    engine = SpeechToText(config=AudioConfig())
    broken = MagicMock()

    def segments():
        yield SimpleNamespace(text="partial text must be discarded")
        raise RuntimeError("cudnn library unavailable")

    broken.transcribe.return_value = (segments(), SimpleNamespace(language="en"))
    engine.model, engine.execution_device, engine.compute_type = broken, "cuda", "float16"
    factory = MagicMock(return_value=stt.model)
    monkeypatch.setattr(engine, "_create_model", factory)
    result = engine.transcribe(samples)
    assert result["raw_text"] == "Python प्रोजेक्ट"
    factory.assert_called_once_with("cpu", "float32")
    stt.model.transcribe.side_effect = RuntimeError("CPU failure")
    with pytest.raises(SpeechToTextError):
        engine.transcribe(samples)
    assert factory.call_count == 1


def test_no_gpu_uses_cpu(monkeypatch, stt, samples):
    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    engine = SpeechToText(config=AudioConfig())
    factory = MagicMock(return_value=stt.model)
    monkeypatch.setattr(engine, "_create_model", factory)
    assert engine.transcribe(samples)["execution_device"] == "cpu"
    factory.assert_called_once_with("cpu", "float32")


def test_diagnostic_identical_audio(stt, samples):
    results = compare_modes(stt, samples)
    assert [r["language_mode"] for r in results] == ["en", "hi", "auto"]
    assert all(call.args[0] is samples for call in stt.model.transcribe.call_args_list)


def test_wav_roundtrip_and_no_overwrite(tmp_path, samples):
    path = tmp_path / "audio.wav"
    save_wav(str(path), samples)
    np.testing.assert_allclose(load_wav(str(path)), samples, atol=1e-4)
    with pytest.raises(FileExistsError):
        save_wav(str(path), samples)


def test_english_only_rejects_hindi(stt, samples):
    from dataclasses import replace

    stt.config = replace(stt.config, model_size="distil-small.en")
    with pytest.raises(ValueError, match="English-only"):
        stt.transcribe(samples, "hi")


def test_listen_timings(stt, samples, monkeypatch):
    monkeypatch.setattr(stt, "record_audio", lambda duration: samples)
    result = stt.listen_and_transcribe()
    assert result["recording_time"] > 0
    assert result["total_duration"] >= result["recording_time"] + result["transcription_duration"]


def test_record_once_workflow(monkeypatch, tmp_path, stt, samples):
    import json

    from audio import diagnostics

    recorder = MagicMock(return_value=samples)
    monkeypatch.setattr(stt, "record_audio", recorder)
    monkeypatch.setattr(diagnostics, "SpeechToText", lambda config: stt)
    monkeypatch.setattr(diagnostics, "load_dotenv", lambda: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    nlu = MagicMock()
    monkeypatch.setattr("core.command_processor.process_text_command", nlu)
    report = tmp_path / "report.json"
    assert diagnostics.main(["--repeat", "2", "--output", str(report)]) == 0
    recorder.assert_called_once()
    nlu.assert_not_called()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert len(data["results"]) == 6
    assert all(call.args[0] is samples for call in stt.model.transcribe.call_args_list)


def test_missing_model_on_cuda_does_not_retry_cpu(monkeypatch, samples):
    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda device: {"int8_float16"})
    engine = SpeechToText(config=AudioConfig(device="cuda"))
    factory = MagicMock(side_effect=OSError("Model not in local cache"))
    monkeypatch.setattr(engine, "_create_model", factory)
    with pytest.raises(ModelInitializationError):
        engine.transcribe(samples)
    assert factory.call_count == 1


def test_default_input_device(microphone):
    SpeechToText(config=AudioConfig()).record_audio(duration=0.1)
    assert microphone.rec.call_args.kwargs["device"] is None


def test_local_english_only_model_metadata(stt, samples):
    stt.model.model.is_multilingual = False
    with pytest.raises(ValueError, match="English-only"):
        stt.transcribe(samples, "hi")
    stt.model.transcribe.assert_not_called()


def test_auto_prefers_measured_cuda_float16(monkeypatch, samples, stt):
    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(
        ctranslate2, "get_supported_compute_types", lambda device: {"float16", "int8_float16"}
    )
    engine = SpeechToText(config=AudioConfig())
    factory = MagicMock(return_value=stt.model)
    monkeypatch.setattr(engine, "_create_model", factory)
    engine.transcribe(samples)
    factory.assert_called_once_with("cuda", "float16")


def test_boundary_and_clipping_warnings(stt):
    result = stt.transcribe(np.ones(16000, dtype=np.float32))
    assert result["audio_diagnostics"]["possible_truncation"]
    assert result["audio_diagnostics"]["clipped_fraction"] == 1
    assert len(result["warnings"]) == 2


@pytest.mark.parametrize("language", ["en", "hi", "auto", None])
def test_detection_provenance(stt, samples, language):
    result = stt.transcribe(samples, language)
    automatic = language in (None, "auto")
    assert result["language_selection_source"] == "call"
    assert result["detected_language"] == ("hi" if automatic else None)
    assert result["language_probability_kind"] == ("detection" if automatic else "forced")


@pytest.mark.parametrize(
    "language,raw",
    [
        ("en", "Save a note about Python and Gemini."),
        ("hi", "मेरी परीक्षा कल है।"),
        (None, "मैं Python और Gemini के साथ VoiceOS project पर काम कर रहा हूँ।"),
    ],
)
def test_multilingual_output_contract_with_mocked_model(stt, samples, language, raw):
    """These canned outputs test preservation, not recognition accuracy on synthetic tones."""
    stt.model.transcribe.side_effect = lambda *args, **kwargs: (
        iter([SimpleNamespace(text=raw)]),
        SimpleNamespace(language=language or "hi", language_probability=0.8),
    )
    result = stt.transcribe(samples, language)
    assert result["raw_text"] == raw
    assert result["normalized_text"] == raw
    stt.model.transcribe.assert_called_once()
    assert stt.model.transcribe.call_args.kwargs["task"] == "transcribe"


@pytest.mark.parametrize("probability", [0.51, 0.939])
def test_ambiguous_detection_never_reranks_or_forces_english(stt, samples, probability):
    raw = "कल परीक्षा है। Python project भी है।"
    candidates = [("hi", probability), ("en", 1 - probability)]
    stt.model.transcribe.side_effect = None
    stt.model.transcribe.return_value = (
        iter([SimpleNamespace(text=raw)]),
        SimpleNamespace(
            language="hi", language_probability=probability, all_language_probs=candidates
        ),
    )
    result = stt.transcribe(samples, "auto")
    assert result["raw_text"] == raw
    assert result["language"] == result["detected_language"] == "hi"
    assert result["language_candidates"] == candidates
    stt.model.transcribe.assert_called_once()
    assert stt.model.transcribe.call_args.kwargs["language"] is None


def test_stress_signals_are_repeatable_and_labeled(samples):
    first, second = stress_audio_cases(samples), stress_audio_cases(samples)
    assert [case[0] for case in first] == ["silence", "noise_only", "noisy", "short", "long"]
    for (name, description, audio), repeated in zip(first, second):
        assert description
        np.testing.assert_array_equal(audio, repeated[2])
        assert audio.dtype == np.float32
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) <= 1
    assert first[-1][2].size == 45 * 16000
    assert np.count_nonzero(first[0][2]) == 0


def test_stress_model_calls_are_mocked(stt, samples):
    results = compare_stress_cases(stt, samples, language="en")
    assert len(results) == 5
    assert results[0]["results"][0]["status"] == "no_speech"
    assert results[0]["results"][0]["language_probability_kind"] == "unavailable"
    # Noise may produce speech or empty output in reality: never assert a fabricated transcript.
    assert results[1]["results"][0]["audio_diagnostics"]["rms"] > 0.001
    assert results[-1]["results"][0]["recording_duration"] == 45


@pytest.mark.parametrize("filename", ["test_audio.wav", "stt-mixed.wav"])
def test_available_saved_wav_contract_with_mocked_model(stt, filename):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / filename
    if not path.exists():
        pytest.skip("Optional private saved recording is not present.")
    audio = load_wav(str(path))
    result = stt.transcribe(audio, "auto")
    assert result["recording_duration"] == audio.size / 16000
    assert result["audio_diagnostics"]["samples"] == audio.size
    # No ground-truth label or accuracy claim is inferred from this mocked decode.


def test_diagnostic_selected_mode_overrides_environment(monkeypatch, stt, samples, tmp_path):
    import json

    from audio import diagnostics

    monkeypatch.setenv("VOICEOS_STT_LANGUAGE", "hi")
    monkeypatch.setattr(diagnostics, "load_dotenv", lambda: None)
    monkeypatch.setattr(diagnostics, "SpeechToText", lambda config: stt)
    monkeypatch.setattr(diagnostics, "load_wav", lambda path: samples)
    output = tmp_path / "report.json"
    assert (
        diagnostics.main(
            [
                "--input",
                "ignored.wav",
                "--preferred-language",
                "auto",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluation_mode"] == "auto"
    assert len(report["results"]) == 1
    assert report["results"][0]["language_mode"] == "auto"
    assert stt.model.transcribe.call_args.kwargs["language"] is None


def test_stress_requires_saved_input():
    from audio import diagnostics

    with pytest.raises(SystemExit) as error:
        diagnostics.main(["--stress"])
    assert error.value.code == 2


def test_explicit_startup_loading_is_reused(monkeypatch, samples, stt):
    engine = SpeechToText(config=AudioConfig(device="cpu"))
    factory = MagicMock(return_value=stt.model)
    monkeypatch.setattr(engine, "_create_model", factory)
    assert engine.load_model() > 0
    assert engine.load_model() == 0
    result = engine.transcribe(samples)
    assert result["model_load_duration"] == 0
    factory.assert_called_once()


@pytest.mark.parametrize("language,enabled", [(None, True), ("en", False), ("hi", False)])
def test_multilingual_decoding_respects_explicit_language(stt, samples, language, enabled):
    from dataclasses import replace

    stt.config = replace(stt.config, multilingual=True, temperatures=(0.0,), vad_threshold=0.35)
    result = stt.transcribe(samples, language)
    options = stt.model.transcribe.call_args.kwargs
    assert options["multilingual"] is enabled
    assert options["temperature"] == (0.0,)
    assert options["vad_parameters"]["min_speech_duration_ms"] == 0
    assert result["multilingual_decoding"] is enabled


def test_segment_and_vad_diagnostics_preserve_raw(stt, samples):
    stt.model.transcribe.side_effect = None
    stt.model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    text=" Python ",
                    start=0.1,
                    end=0.5,
                    avg_logprob=-0.4,
                    no_speech_prob=0.01,
                    temperature=0.0,
                )
            ]
        ),
        SimpleNamespace(language="en", language_probability=1, duration_after_vad=0.8),
    )
    result = stt.transcribe(samples, "en")
    assert result["raw_text"] == "Python"
    assert result["segments"][0]["text"] == " Python "
    assert result["segments"][0]["start"] == 0.1
    assert result["speech_duration_after_vad"] == 0.8


@pytest.mark.parametrize("raw", ["...", "ौ ौ ौ", "?!?"])
def test_nonlexical_model_output_preserved_but_not_usable(stt, samples, raw):
    stt.model.transcribe.side_effect = None
    stt.model.transcribe.return_value = (
        iter([SimpleNamespace(text=raw)]),
        SimpleNamespace(language="hi", language_probability=1),
    )
    result = stt.transcribe(samples, "hi")
    assert result["raw_text"] == raw
    assert result["status"] == "unusable_output"
    assert result["empty_transcription"] is False
    assert result["has_lexical_content"] is False


def test_cpu_int8_remains_explicitly_configurable(monkeypatch, stt, samples):
    engine = SpeechToText(config=AudioConfig(device="cpu", compute_type="int8"))
    factory = MagicMock(return_value=stt.model)
    monkeypatch.setattr(engine, "_create_model", factory)
    engine.transcribe(samples)
    factory.assert_called_once_with("cpu", "int8")


@pytest.mark.parametrize(
    "raw",
    [
        "Mera naam Keval hai aur main VoiceOS project par kaam kar raha hoon.",
        "मेरा नाम Keval है और मैं VoiceOS project पर काम कर रहा हूँ।",
        "I am testing Python and Gemini.",
    ],
)
def test_native_client_preserves_mocked_language_without_llm(stt, samples, raw, monkeypatch):
    from dataclasses import replace

    stt.config = replace(stt.config, language=None, multilingual=True, initial_prompt="")
    llm = MagicMock()
    monkeypatch.setattr("nlu.llm_client.generate_response", llm)
    stt.model.transcribe.side_effect = None
    stt.model.transcribe.return_value = (
        iter([SimpleNamespace(text=raw)]),
        SimpleNamespace(language="hi", language_probability=0.8),
    )
    result = stt.transcribe(samples)
    assert result["raw_text"] == result["normalized_text"] == raw
    options = stt.model.transcribe.call_args.kwargs
    assert options["task"] == "transcribe" and options["language"] is None
    assert options["initial_prompt"] is None and options["multilingual"] is True
    llm.assert_not_called()


def test_quiet_short_speech_not_rejected_by_diluted_global_rms(stt):
    audio = np.zeros(160000, np.float32)
    audio[5000:8200] = 0.005
    result = stt.transcribe(
        audio,
        recording_info={
            "mode": "automatic",
            "speech_started": True,
            "stop_reason": "silence",
        },
    )
    assert result["audio_diagnostics"]["near_silent"]
    assert result["status"] == "ok"
    stt.model.transcribe.assert_called_once()


def test_flat_zero_audio_not_decoded_despite_inconsistent_capture_flag(stt):
    stt.transcribe(
        np.zeros(16000, np.float32),
        recording_info={
            "mode": "automatic",
            "speech_started": True,
            "stop_reason": "silence",
        },
    )
    stt.model.transcribe.assert_not_called()
