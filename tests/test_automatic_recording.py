"""Deterministic stream/VAD tests; no physical microphone or recognition accuracy claims."""

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio import AudioConfig, SpeechToText
from audio.recording import (
    FRAME_SAMPLES,
    CaptureFailure,
    SilenceEndpoint,
    StreamingSileroVad,
    capture_automatic,
)
from audio.speech_to_text import AudioCaptureError


def endpoint(probabilities, **settings):
    config = AudioConfig(**settings)
    state = SilenceEndpoint(config)
    for probability in probabilities:
        state.accept(probability)
        if state.stop_reason:
            break
    return state


@pytest.mark.parametrize("language", ["English", "Hindi", "Hinglish"])
def test_endpoint_is_independent_of_language(language):
    state = endpoint([0.9] * 20 + [0.01] * 63)
    assert state.speech_started
    assert state.stop_reason == "silence"
    assert 2 <= state.information()["trailing_silence_duration"] < 2.0321


@pytest.mark.parametrize("pause_frames", [1, 16, 31, 55, 62])
def test_pauses_shorter_than_two_seconds_do_not_end_sentence(pause_frames):
    state = endpoint([0.9] * 10 + [0.01] * pause_frames + [0.9] * 10 + [0.01] * 63)
    assert state.stop_reason == "silence"
    assert state.samples == (10 + pause_frames + 10 + 63) * FRAME_SAMPLES


def test_speech_resumes_at_release_threshold():
    state = endpoint([0.9] * 3 + [0.01] * 60 + [0.4] * 4 + [0.01] * 63)
    assert state.samples == 130 * FRAME_SAMPLES


def test_continuous_non_speech_noise_uses_initial_timeout():
    state = endpoint([0.01] * 300)
    assert not state.speech_started
    assert state.stop_reason == "initial_silence_timeout"
    assert state.samples == 8 * 16000


def test_speech_onset_near_initial_timeout_is_kept():
    state = endpoint([0.01] * 249 + [0.9] * 10 + [0.01] * 63)
    assert state.speech_started and state.stop_reason == "silence"


def test_continuous_speech_or_speech_like_noise_hits_maximum():
    state = endpoint([0.9] * 200, max_recording_duration=3, initial_silence_timeout=1)
    assert state.stop_reason == "max_duration"
    assert state.samples == 3 * 16000
    assert state.speech_started


def simulated_microphone(monkeypatch, frames, *, statuses=None):
    from audio import recording

    source = iter(frames)
    status_source = iter(statuses) if statuses is not None else None
    holder = {}
    real_queue = queue.Queue

    class PullQueue(real_queue):
        def get(self, timeout=None):
            try:
                frame = next(source)
            except StopIteration:
                raise queue.Empty from None
            status = next(status_source) if status_source is not None else False
            holder["callback"](frame, frame.shape[0], None, status)
            return real_queue.get(self, block=False)

    class Stream:
        def __init__(self, **kwargs):
            holder["callback"] = kwargs["callback"]
            holder["options"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            holder["closed"] = True

    monkeypatch.setattr(recording.queue, "Queue", PullQueue)
    return SimpleNamespace(InputStream=Stream), holder


def test_stream_preserves_initial_audio_final_words_and_trailing_silence(monkeypatch):
    probabilities = [0.01] * 10 + [0.9] * 20 + [0.01] * 30 + [0.9] * 5 + [0.01] * 63
    frames = [np.full((512, 1), (i + 1) / 1000, np.float32) for i in range(len(probabilities))]
    sd, holder = simulated_microphone(monkeypatch, frames)
    vad = SimpleNamespace(reset=MagicMock(), probability=MagicMock(side_effect=probabilities))
    audio, info = capture_automatic(sd, AudioConfig(), vad)
    np.testing.assert_array_equal(audio, np.concatenate(frames).flatten())
    assert info["stop_reason"] == "silence"
    assert holder["closed"]
    assert holder["options"]["samplerate"] == 16000
    assert holder["options"]["channels"] == 1
    assert holder["options"]["dtype"] == "float32"
    vad.reset.assert_called_once()


@pytest.mark.parametrize(
    "bad_frame",
    [
        np.zeros((0, 1), np.float32),
        np.zeros((512, 1), np.int16),
        np.full((512, 1), np.nan, np.float32),
        np.full((512, 1), 2, np.float32),
    ],
)
def test_invalid_microphone_blocks_close_stream(monkeypatch, bad_frame):
    sd, holder = simulated_microphone(monkeypatch, [bad_frame])
    vad = SimpleNamespace(reset=MagicMock(), probability=MagicMock())
    with pytest.raises(CaptureFailure, match="invalid|incomplete"):
        capture_automatic(sd, AudioConfig(), vad)
    assert holder["closed"]
    vad.probability.assert_not_called()


def test_microphone_status_errors_close_stream(monkeypatch):
    sd, holder = simulated_microphone(
        monkeypatch, [np.zeros((512, 1), np.float32)], statuses=[True]
    )
    with pytest.raises(CaptureFailure, match="overflow"):
        capture_automatic(sd, AudioConfig(), SimpleNamespace(reset=lambda: None))
    assert holder["closed"]


def test_stalled_microphone_is_bounded(monkeypatch):
    sd, holder = simulated_microphone(monkeypatch, [])
    with pytest.raises(CaptureFailure, match="stopped delivering"):
        capture_automatic(sd, AudioConfig(), SimpleNamespace(reset=lambda: None))
    assert holder["closed"]


def test_wall_clock_safety_limit_closes_stream(monkeypatch):
    from audio import recording

    sd, holder = simulated_microphone(monkeypatch, [])
    times = iter([0, 100, 100])
    monkeypatch.setattr(recording.time, "monotonic", lambda: next(times))
    audio, info = capture_automatic(sd, AudioConfig(), SimpleNamespace(reset=lambda: None))
    assert info["stop_reason"] == "max_duration" and not audio.size
    assert holder["closed"]


@pytest.mark.parametrize("inputs", [("input", "h", "c"), ("input", "state", "sr")])
def test_silero_reuses_session_and_resets_recurrent_state(monkeypatch, inputs):
    session = MagicMock()
    session.get_inputs.return_value = [SimpleNamespace(name=name) for name in inputs]
    if "h" in inputs:
        session.run.return_value = (
            np.array([0.8]),
            np.ones((1, 1, 128), np.float32),
            np.ones((1, 1, 128), np.float32),
        )
    else:
        session.run.return_value = (np.array([0.8]), np.ones((2, 1, 128), np.float32))
    factory = MagicMock(return_value=SimpleNamespace(session=session))
    monkeypatch.setattr("faster_whisper.vad.get_vad_model", factory)
    vad = StreamingSileroVad()
    frame = np.ones(512, np.float32) * 0.1
    assert vad.probability(frame) == 0.8
    vad.probability(frame)
    assert vad.session is session
    factory.assert_called_once()
    vad.reset()
    assert not vad.h.any() and not vad.c.any() and not vad.state.any() and not vad.context.any()


def test_initial_timeout_skips_whisper_despite_background_rms():
    engine = SpeechToText(config=AudioConfig())
    engine.model = MagicMock()
    audio = np.full(16000, 0.03, np.float32)
    result = engine.transcribe(
        audio,
        recording_info={
            "mode": "automatic",
            "speech_started": False,
            "stop_reason": "initial_silence_timeout",
        },
    )
    assert result["status"] == "no_speech"
    assert result["recording_duration"] == 1
    assert not result["audio_diagnostics"]["near_silent"]
    engine.model.transcribe.assert_not_called()


def test_streaming_vad_load_failure_does_not_retry_forever(monkeypatch):
    factory = MagicMock(side_effect=RuntimeError("missing ONNX"))
    monkeypatch.setattr("audio.speech_to_text.StreamingSileroVad", factory)
    engine = SpeechToText(config=AudioConfig())
    for _ in range(2):
        with pytest.raises(AudioCaptureError, match="initialize streaming"):
            engine.load_vad()
    factory.assert_called_once()


def test_streaming_vad_loaded_once(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr("audio.speech_to_text.StreamingSileroVad", factory)
    engine = SpeechToText(config=AudioConfig())
    engine.load_vad()
    assert engine.load_vad() == 0
    factory.assert_called_once()


def test_callback_queue_overflow_is_rejected_and_closed():
    holder = {}

    class OverflowingStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            for _ in range(65):
                self.callback(np.zeros((512, 1), np.float32), 512, None, False)
            return self

        def __exit__(self, *args):
            holder["closed"] = True

    with pytest.raises(CaptureFailure, match="processing overflowed"):
        capture_automatic(
            SimpleNamespace(InputStream=OverflowingStream),
            AudioConfig(),
            SimpleNamespace(reset=lambda: None),
        )
    assert holder["closed"]


def test_ready_notification_happens_after_stream_opens(monkeypatch):
    frames = [np.zeros((512, 1), np.float32)] * 8
    sd, holder = simulated_microphone(monkeypatch, frames)
    ready = MagicMock(side_effect=lambda: holder["options"])
    config = AudioConfig(initial_silence_timeout=0.25)
    capture_automatic(
        sd,
        config,
        SimpleNamespace(reset=lambda: None, probability=lambda frame: 0),
        on_started=ready,
    )
    ready.assert_called_once()


def test_automatic_capture_does_not_use_fixed_recorder(monkeypatch):
    sd = MagicMock()
    sd.query_devices.return_value = {"max_input_channels": 1}
    monkeypatch.setattr("audio.speech_to_text._sounddevice", lambda: sd)
    recorder = MagicMock(
        return_value=(
            np.zeros(512, np.float32),
            {
                "mode": "automatic",
                "speech_started": False,
                "stop_reason": "initial_silence_timeout",
            },
        )
    )
    monkeypatch.setattr("audio.speech_to_text.capture_automatic", recorder)
    engine = SpeechToText(config=AudioConfig())
    engine._capture_vad = MagicMock()
    engine.record_audio()
    recorder.assert_called_once()
    sd.rec.assert_not_called()


def test_safety_sample_limit_preserves_exact_prefix(monkeypatch):
    frames = [np.full((512, 1), 0.2, np.float32)] * 100
    sd, holder = simulated_microphone(monkeypatch, frames)
    config = AudioConfig(max_recording_duration=3, initial_silence_timeout=1)
    audio, info = capture_automatic(
        sd, config, SimpleNamespace(reset=lambda: None, probability=lambda frame: 0.9)
    )
    assert audio.size == 3 * 16000
    assert info["stop_reason"] == "max_duration"
    assert holder["closed"]
