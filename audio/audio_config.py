"""Validated settings; importing this module never opens a microphone or model."""

import math
import os
from dataclasses import dataclass

USE_CONFIG_LANGUAGE = object()


def language_code(value: str | None) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("STT language must be auto, en or hi.")
    value = value.strip().lower() if value is not None else None
    if value is None or value == "auto":
        return None
    if value not in {"en", "hi"}:
        raise ValueError("STT language must be auto, en or hi; Hinglish has no separate code.")
    return value


@dataclass(frozen=True)
class LanguageSelection:
    """A decoder choice, never a transcript-quality estimate or translation request."""

    mode: str
    source: str

    @property
    def whisper_language(self) -> str | None:
        return language_code(self.mode)


def select_language(configured: str | None, requested=USE_CONFIG_LANGUAGE) -> LanguageSelection:
    """Explicit calls (including auto/None) override the configured preferred mode."""
    source = "configuration" if requested is USE_CONFIG_LANGUAGE else "call"
    value = configured if requested is USE_CONFIG_LANGUAGE else requested
    return LanguageSelection(mode=language_code(value) or "auto", source=source)


def microphone_device(value: str | int | None) -> str | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return int(value) if value.lstrip("-").isdigit() else value
    if type(value) is not int:
        raise ValueError("Microphone must be an input device ID or name.")
    return value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true or false.")
    return value in {"true", "1"}


@dataclass(frozen=True)
class AudioConfig:
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    sample_rate: int = 16000
    microphone_id: int | str | None = None
    duration: float = 12.0
    recording_mode: str = "automatic"
    silence_duration: float = 2.0
    max_recording_duration: float = 60.0
    initial_silence_timeout: float = 8.0
    capture_vad_threshold: float = 0.5
    speech_start_ms: int = 96
    language: str | None = None
    silence_rms: float = 0.001
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 300
    vad_threshold: float = 0.5
    multilingual: bool = True
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    initial_prompt: str = ""
    vocabulary_path: str | None = None
    local_files_only: bool = False
    recording_directory: str | None = None

    def __post_init__(self):
        if not self.model_size.strip():
            raise ValueError("STT model must not be empty.")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("STT device must be auto, cuda or cpu.")
        if self.compute_type not in {
            "auto",
            "float16",
            "float32",
            "int8",
            "int8_float16",
            "int8_float32",
        }:
            raise ValueError("Unsupported STT compute type.")
        # Faster-Whisper interprets ndarray input at 16 kHz; reject mislabeled audio.
        if self.sample_rate != 16000:
            raise ValueError("STT ndarray capture requires a 16000 Hz sample rate.")
        if not math.isfinite(self.duration) or not 0 < self.duration <= 120:
            raise ValueError("Recording duration must be greater than 0 and at most 120 seconds.")
        if self.recording_mode not in {"automatic", "fixed"}:
            raise ValueError("Recording mode must be automatic or fixed.")
        for name, value, low, high in (
            ("silence_duration", self.silence_duration, 0.25, 10),
            ("max_recording_duration", self.max_recording_duration, 1, 120),
            ("initial_silence_timeout", self.initial_silence_timeout, 0.25, 120),
        ):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid {name}: expected {low}–{high} seconds.")
        if not 0 < self.capture_vad_threshold < 1 or not 32 <= self.speech_start_ms <= 500:
            raise ValueError("Invalid capture VAD threshold or speech onset duration.")
        if self.initial_silence_timeout > self.max_recording_duration:
            raise ValueError("Initial silence timeout must not exceed maximum recording duration.")
        if not math.isfinite(self.silence_rms) or not 0 < self.silence_rms < 1:
            raise ValueError("Silence RMS threshold must be between 0 and 1.")
        if type(self.beam_size) is not int or not 1 <= self.beam_size <= 10:
            raise ValueError("Beam size must be an integer from 1 to 10.")
        if not 0 <= self.vad_min_silence_ms <= 5000 or not 0 <= self.vad_speech_pad_ms <= 2000:
            raise ValueError("Invalid VAD silence/padding settings.")
        if not math.isfinite(self.vad_threshold) or not 0 < self.vad_threshold < 1:
            raise ValueError("VAD threshold must be between 0 and 1.")
        if type(self.multilingual) is not bool:
            raise ValueError("Multilingual decoding must be true or false.")
        if not self.temperatures or any(
            not math.isfinite(t) or not 0 <= t <= 1 for t in self.temperatures
        ):
            raise ValueError("Decode temperatures must be nonempty values between 0 and 1.")
        language_code(self.language)
        microphone_device(self.microphone_id)

    @classmethod
    def from_env(cls):
        return cls(
            model_size=os.getenv("VOICEOS_STT_MODEL", "small"),
            device=os.getenv("VOICEOS_STT_DEVICE", "auto"),
            compute_type=os.getenv("VOICEOS_STT_COMPUTE_TYPE", "auto"),
            microphone_id=microphone_device(os.getenv("VOICEOS_MICROPHONE")),
            duration=float(os.getenv("VOICEOS_RECORD_SECONDS", "12")),
            recording_mode=os.getenv("VOICEOS_RECORDING_MODE", "automatic"),
            silence_duration=float(os.getenv("VOICEOS_SILENCE_SECONDS", "2")),
            max_recording_duration=float(os.getenv("VOICEOS_MAX_RECORD_SECONDS", "60")),
            initial_silence_timeout=float(os.getenv("VOICEOS_INITIAL_SILENCE_SECONDS", "8")),
            capture_vad_threshold=float(os.getenv("VOICEOS_CAPTURE_VAD_THRESHOLD", "0.5")),
            speech_start_ms=int(os.getenv("VOICEOS_SPEECH_START_MS", "96")),
            language=language_code(os.getenv("VOICEOS_STT_LANGUAGE", "auto")),
            silence_rms=float(os.getenv("VOICEOS_SILENCE_RMS", "0.001")),
            beam_size=int(os.getenv("VOICEOS_STT_BEAM_SIZE", "5")),
            vad_filter=_boolean("VOICEOS_STT_VAD", True),
            vad_min_silence_ms=int(os.getenv("VOICEOS_VAD_MIN_SILENCE_MS", "500")),
            vad_speech_pad_ms=int(os.getenv("VOICEOS_VAD_SPEECH_PAD_MS", "300")),
            vad_threshold=float(os.getenv("VOICEOS_VAD_THRESHOLD", "0.5")),
            multilingual=_boolean("VOICEOS_STT_MULTILINGUAL", True),
            temperatures=tuple(
                float(t)
                for t in os.getenv("VOICEOS_STT_TEMPERATURES", "0,0.2,0.4,0.6,0.8,1").split(",")
            ),
            initial_prompt=os.getenv("VOICEOS_STT_PROMPT", ""),
            vocabulary_path=os.getenv("VOICEOS_VOCABULARY_PATH") or None,
            local_files_only=_boolean("VOICEOS_STT_LOCAL_FILES_ONLY", False),
            recording_directory=os.getenv("VOICEOS_RECORDING_DIRECTORY") or None,
        )
