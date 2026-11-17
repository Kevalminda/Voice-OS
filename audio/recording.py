"""Streaming microphone endpoint detection using the bundled Silero ONNX VAD."""

import math
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from audio.audio_config import AudioConfig

FRAME_SAMPLES = 512  # Silero's 32 ms frame at 16 kHz.


class CaptureFailure(RuntimeError):
    pass


class StreamingSileroVad:
    """Reuse the cached ONNX session; keep recurrent state local to each capture."""

    def __init__(self):
        from faster_whisper.vad import get_vad_model

        self.session = get_vad_model().session
        self.inputs = {item.name for item in self.session.get_inputs()}
        if self.inputs not in ({"input", "h", "c"}, {"input", "state", "sr"}):
            raise CaptureFailure(
                "Unsupported bundled Silero model interface. Check Faster-Whisper."
            )
        self.reset()

    def reset(self):
        self.context = np.zeros((1, 64), np.float32)
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.state = np.zeros((2, 1, 128), np.float32)

    def probability(self, frame: np.ndarray) -> float:
        if frame.shape != (FRAME_SAMPLES,) or frame.dtype != np.float32:
            raise CaptureFailure("Streaming VAD requires 512 mono float32 samples.")
        data = np.concatenate((self.context, frame[None, :]), axis=1)
        if "h" in self.inputs:
            output, self.h, self.c = self.session.run(
                None,
                {"input": data, "h": self.h, "c": self.c},
            )
        else:
            output, self.state = self.session.run(
                None,
                {"input": data, "state": self.state, "sr": np.array(16000, np.int64)},
            )
        self.context = frame[None, -64:].copy()
        value = float(np.asarray(output).reshape(-1)[0])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise CaptureFailure("Streaming VAD returned an invalid speech probability.")
        return value


@dataclass
class SilenceEndpoint:
    """Sample-based state machine; preserve samples rather than trimming guessed boundaries."""

    config: AudioConfig
    samples: int = 0
    speech_started: bool = False
    onset_samples: int = 0
    silent_samples: int = 0
    stop_reason: str | None = None

    def accept(self, probability: float, samples=FRAME_SAMPLES) -> int:
        if self.stop_reason:
            return 0
        limit = int(self.config.max_recording_duration * self.config.sample_rate)
        retained = min(samples, limit - self.samples)
        self.samples += retained
        if not self.speech_started:
            self.onset_samples = (
                self.onset_samples + retained
                if probability >= self.config.capture_vad_threshold
                else 0
            )
            if self.onset_samples >= self.config.speech_start_ms * self.config.sample_rate / 1000:
                self.speech_started = True
        else:
            release_threshold = max(0.01, self.config.capture_vad_threshold - 0.15)
            self.silent_samples = (
                0 if probability >= release_threshold else self.silent_samples + retained
            )
            if self.silent_samples >= self.config.silence_duration * self.config.sample_rate:
                self.stop_reason = "silence"
        if (
            not self.speech_started
            and self.onset_samples == 0
            and self.samples >= (self.config.initial_silence_timeout * self.config.sample_rate)
        ):
            self.stop_reason = "initial_silence_timeout"
        if self.samples >= limit and self.stop_reason is None:
            self.stop_reason = "max_duration"
        return retained

    def information(self) -> dict:
        return {
            "mode": "automatic",
            "stop_reason": self.stop_reason,
            "speech_started": self.speech_started,
            "captured_duration": self.samples / self.config.sample_rate,
            "trailing_silence_duration": self.silent_samples / self.config.sample_rate,
        }


def capture_automatic(sd, config: AudioConfig, vad: StreamingSileroVad, on_started=None):
    """Only copy frames in the real-time callback; run inference in the consumer thread."""
    pending = queue.Queue(maxsize=64)
    overflow = threading.Event()

    def callback(indata, frames, time_info, status):
        try:
            pending.put_nowait((indata.copy(), bool(status)))
        except queue.Full:
            overflow.set()

    vad.reset()
    endpoint = SilenceEndpoint(config)
    blocks = []
    started = time.monotonic()
    with sd.InputStream(
        samplerate=config.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        device=config.microphone_id,
        callback=callback,
    ):
        if on_started is not None:
            on_started()
        while endpoint.stop_reason is None:
            if overflow.is_set():
                raise CaptureFailure(
                    "Microphone processing overflowed. Retry or select another device."
                )
            remaining = config.max_recording_duration - (time.monotonic() - started)
            if remaining <= 0:
                endpoint.stop_reason = "max_duration"
                break
            try:
                data, bad_status = pending.get(timeout=min(1.0, remaining))
            except queue.Empty as error:
                raise CaptureFailure(
                    "Microphone stopped delivering audio. Check the input device."
                ) from error
            if bad_status:
                raise CaptureFailure("Microphone overflow/underflow detected. Please retry.")
            if data.shape != (FRAME_SAMPLES, 1) or data.dtype != np.float32:
                raise CaptureFailure("Microphone returned an invalid or incomplete audio block.")
            frame = data[:, 0]
            if not np.isfinite(frame).all() or np.max(np.abs(frame)) > 1:
                raise CaptureFailure("Microphone returned invalid audio samples.")
            retained = endpoint.accept(vad.probability(frame))
            blocks.append(frame[:retained].copy())
    audio = np.concatenate(blocks) if blocks else np.array([], dtype=np.float32)
    info = endpoint.information()
    info["capture_wall_time"] = time.monotonic() - started
    return audio, info
