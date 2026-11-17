"""Microphone capture and reusable inference; no actions or network LLM calls."""

import gc
import logging
import math
import time
import unicodedata
from dataclasses import asdict, replace
from typing import NotRequired, TypedDict

import numpy as np

from audio.audio_config import (
    USE_CONFIG_LANGUAGE,
    AudioConfig,
    microphone_device,
    select_language,
)
from audio.capture_archive import finish_capture, preserve_capture
from audio.recording import CaptureFailure, StreamingSileroVad, capture_automatic
from audio.transcript_normalizer import TranscriptNormalizer

logger = logging.getLogger(__name__)


class SpeechToTextError(RuntimeError):
    """An actionable audio/model error suitable for the CLI."""


class AudioCaptureError(SpeechToTextError):
    pass


class ModelInitializationError(SpeechToTextError):
    pass


class TranscriptResult(TypedDict):
    text: str  # Compatibility alias for normalized_text.
    raw_text: str
    normalized_text: str
    language: str | None
    language_probability: float | None
    language_mode: str
    language_selection_source: str
    detected_language: str | None
    language_probability_kind: str
    language_candidates: list
    status: str
    empty_transcription: bool
    recording_duration: float
    recording_time: float
    model_load_duration: float
    transcription_duration: float
    normalization_duration: float
    total_duration: float
    audio_diagnostics: dict
    warnings: list[str]
    corrections: list
    execution_device: str
    compute_type: str
    speech_duration_after_vad: float | None
    multilingual_decoding: bool
    segments: list[dict]
    has_lexical_content: bool
    recording_info: dict
    recording_files: NotRequired[dict]
    review_flags: NotRequired[list[dict[str, str]]]


def _sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as error:
        raise AudioCaptureError(
            "Install sounddevice and check the Windows audio driver."
        ) from error
    return sounddevice


def list_input_devices() -> list[dict]:
    try:
        sd = _sounddevice()
        return [
            {
                "id": index,
                "name": device["name"],
                "max_input_channels": device["max_input_channels"],
                "default_samplerate": device["default_samplerate"],
            }
            for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0
        ]
    except SpeechToTextError:
        raise
    except Exception as error:
        raise AudioCaptureError(
            "Cannot enumerate microphones. Check Windows audio settings."
        ) from error


def _audio_array(audio) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.dtype.kind != "f":
        raise ValueError("Audio must be mono floating-point samples at 16000 Hz.")
    if not np.isfinite(array).all() or (array.size and np.max(np.abs(array)) > 1):
        raise ValueError("Audio must contain finite samples between -1 and 1.")
    return np.ascontiguousarray(array, dtype=np.float32)


def audio_diagnostics(audio: np.ndarray, config: AudioConfig) -> dict:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    tail = audio[-int(config.sample_rate * 0.2) :]
    head = audio[: int(config.sample_rate * 0.2)]
    tail_rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float64)))) if tail.size else 0.0
    return {
        "sample_rate": config.sample_rate,
        "dtype": str(audio.dtype),
        "channels": 1,
        "samples": int(audio.size),
        "rms": rms,
        "peak": peak,
        "near_silent": rms < config.silence_rms,
        "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0,
        "tail_rms": tail_rms,
        "head_rms": float(np.sqrt(np.mean(np.square(head, dtype=np.float64))))
        if head.size
        else 0.0,
        "possible_truncation": tail_rms >= max(config.silence_rms * 3, rms * 0.5),
    }


def _gpu_error(error: Exception) -> bool:
    # Never retry a missing model/download as a CUDA failure.
    return any(
        term in str(error).lower()
        for term in (
            "cuda",
            "cudnn",
            "cublas",
            "out of memory",
            "float16",
            "compute type",
        )
    )


class SpeechToText:
    def __init__(
        self,
        model_size=None,
        device=None,
        compute_type=None,
        sample_rate=None,
        microphone_id=None,
        *,
        config: AudioConfig | None = None,
        normalizer: TranscriptNormalizer | None = None,
    ):
        settings = config or AudioConfig.from_env()
        overrides = {
            key: value
            for key, value in {
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
                "sample_rate": sample_rate,
                "microphone_id": microphone_id,
            }.items()
            if value is not None
        }
        self.config = replace(settings, **overrides)
        self.sample_rate = self.config.sample_rate
        self.microphone_id = microphone_device(self.config.microphone_id)
        self.normalizer = normalizer or (
            TranscriptNormalizer.from_file(self.config.vocabulary_path)
            if self.config.vocabulary_path
            else TranscriptNormalizer()
        )
        self.model = None
        self.model_load_duration = 0.0
        self.execution_device = "unloaded"
        self.compute_type = "unloaded"
        self._initialization_error = None
        self._cpu_fallback_used = False
        self._fallback_reason = None
        self._capture_vad = None
        self.vad_load_duration = 0.0
        self.last_recording_info = None
        self._vad_initialization_error = None
        self.last_capture_files = None

    def configuration(self) -> dict:
        return asdict(self.config)

    def load_model(self) -> float:
        """Initialize once before capture; return only the loading time incurred by this call."""
        previous = self.model_load_duration
        self._load_model()
        return self.model_load_duration - previous

    def load_vad(self) -> float:
        """Prepare the cached streaming VAD before activation, once per instance."""
        if self._capture_vad is not None or self.config.recording_mode == "fixed":
            return 0.0
        if self._vad_initialization_error:
            raise AudioCaptureError(self._vad_initialization_error)
        started = time.perf_counter()
        try:
            self._capture_vad = StreamingSileroVad()
        except Exception as error:
            self._vad_initialization_error = (
                "Cannot initialize streaming VAD. Check speech dependencies."
            )
            raise AudioCaptureError(self._vad_initialization_error) from error
        elapsed = time.perf_counter() - started
        self.vad_load_duration += elapsed
        return elapsed

    def _create_model(self, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=compute_type,
            local_files_only=self.config.local_files_only,
        )

    def _load_model(self):
        if self.model is not None:
            return
        if self._initialization_error is not None:
            raise ModelInitializationError(self._initialization_error)
        start = time.perf_counter()
        try:
            import ctranslate2

            device = self.config.device
            if device == "auto":
                try:
                    device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
                except RuntimeError:
                    device = "cpu"
            try:
                supported = ctranslate2.get_supported_compute_types(device)
                compute = self.config.compute_type
                if compute == "auto":
                    # Measured small/float16 fits this 4 GB GPU and was faster than int8_float16.
                    preferences = (
                        ("float16", "int8_float16", "float32")
                        if device == "cuda"
                        else ("float32", "int8")
                    )
                    compute = next((kind for kind in preferences if kind in supported), None)
                if compute not in supported:
                    raise RuntimeError("Requested compute type is unsupported on this device.")
                self.model = self._create_model(device, compute)
                self.execution_device, self.compute_type = device, compute
            except Exception as error:
                if device != "cuda" or not _gpu_error(error):
                    raise
                self._fallback_to_cpu()
        except Exception as error:
            self._initialization_error = (
                "Cannot initialize STT. Check model path/cache, speech dependencies and CUDA "
                "libraries; try VOICEOS_STT_DEVICE=cpu."
            )
            raise ModelInitializationError(self._initialization_error) from error
        finally:
            self.model_load_duration += time.perf_counter() - start

    def _fallback_to_cpu(self):
        self._cpu_fallback_used = True
        self._fallback_reason = "CUDA unavailable/incompatible; using CPU float32."
        logger.warning(self._fallback_reason)
        self.model = None
        gc.collect()
        self.model = self._create_model("cpu", "float32")
        self.execution_device, self.compute_type = "cpu", "float32"

    def record_audio(self, duration=None, *, on_started=None) -> np.ndarray:
        """Default automatic endpointing; explicit duration retains fixed benchmark capture."""
        self.last_recording_info = None
        seconds = self.config.duration if duration is None else float(duration)
        if not math.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError("Recording duration must be greater than 0 and at most 120 seconds.")
        frames = int(seconds * self.sample_rate)
        if frames < 1:
            raise ValueError("Recording must contain at least one sample.")
        sd = _sounddevice()
        try:
            device = sd.query_devices(self.microphone_id, kind="input")
            if device["max_input_channels"] < 1:
                raise AudioCaptureError("Selected microphone is not an input device.")
            sd.check_input_settings(
                device=self.microphone_id,
                channels=1,
                dtype="float32",
                samplerate=self.sample_rate,
            )
            if duration is None and self.config.recording_mode == "automatic":
                self.load_vad()
                audio, info = capture_automatic(
                    sd,
                    replace(self.config, microphone_id=self.microphone_id),
                    self._capture_vad,
                    on_started=on_started,
                )
                self.last_recording_info = info
                return audio
            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.microphone_id,
            )
            if on_started is not None:
                on_started()
            status = sd.wait()
            if status:
                raise AudioCaptureError(
                    "Recording overflow/underflow: retry or select another device."
                )
            audio = _audio_array(audio)
            if audio.size != frames:
                raise AudioCaptureError(
                    "Microphone returned an incomplete recording. Please retry."
                )
            self.last_recording_info = {
                "mode": "fixed",
                "stop_reason": "fixed_duration",
                "captured_duration": audio.size / self.sample_rate,
            }
            return audio
        except (KeyboardInterrupt, EOFError):
            sd.stop()
            raise
        except AudioCaptureError:
            raise
        except CaptureFailure as error:
            raise AudioCaptureError(str(error)) from error
        except Exception as error:
            raise AudioCaptureError(
                "Cannot record from the selected microphone at 16000 Hz. Check device ID/name, "
                "Windows microphone permission and exclusive access; use --list-devices."
            ) from error

    def _decode(self, audio, language):
        # Check model metadata too: local English-only directories need not end in .en.
        underlying_model = getattr(self.model, "model", None)
        if getattr(underlying_model, "is_multilingual", None) is False and language != "en":
            raise ValueError("English-only models require language=en; use small/medium for Hindi.")
        segments, info = self.model.transcribe(
            audio,
            task="transcribe",
            language=language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": self.config.vad_min_silence_ms,
                "speech_pad_ms": self.config.vad_speech_pad_ms,
                "threshold": self.config.vad_threshold,
                "min_speech_duration_ms": 0,
            },
            initial_prompt=self.config.initial_prompt or None,
            condition_on_previous_text=False,
            # Explicit en/hi stays forced even when per-window detection is enabled for auto.
            multilingual=self.config.multilingual and language is None,
            temperature=self.config.temperatures,
        )
        # Inference is lazy: exhaust segments inside the timer and error boundary.
        details = []
        texts = []
        for segment in segments:
            if segment.text.strip():
                texts.append(segment.text.strip())
            details.append(
                {
                    "text": segment.text,
                    "start": getattr(segment, "start", None),
                    "end": getattr(segment, "end", None),
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "no_speech_prob": getattr(segment, "no_speech_prob", None),
                    "temperature": getattr(segment, "temperature", None),
                }
            )
        return " ".join(texts), info, details

    def transcribe(
        self, audio, language=USE_CONFIG_LANGUAGE, *, recording_info=None
    ) -> TranscriptResult:
        start = time.perf_counter()
        selection = select_language(self.config.language, language)
        language = selection.whisper_language
        if self.config.model_size.endswith(".en") and language != "en":
            raise ValueError("English-only models require language=en; use small/medium for Hindi.")
        audio = _audio_array(audio)
        diagnostics = audio_diagnostics(audio, self.config)
        capture_info = recording_info or {}
        no_speech_capture = capture_info.get("mode") == "automatic" and not capture_info.get(
            "speech_started"
        )
        capture_confirms_speech = (
            capture_info.get("mode") == "automatic"
            and capture_info.get("speech_started")
            and diagnostics["peak"] >= self.config.silence_rms
        )
        # Leading/trailing silence can dilute whole-recording RMS around a quiet short word.
        skip_for_level = diagnostics["near_silent"] and not capture_confirms_speech
        warnings = []
        if diagnostics["possible_truncation"] and capture_info.get("stop_reason") != "silence":
            warnings.append(
                "Audio is active at the end; increase duration and leave trailing silence."
            )
        if diagnostics["clipped_fraction"] > 0.01:
            warnings.append("Audio may be clipped; lower microphone gain.")
        if capture_info.get("stop_reason") == "max_duration":
            warnings.append("Recording reached its safety limit; the sentence may be incomplete.")
        raw, info = "", None
        segments = []
        load_before = self.model_load_duration
        transcription_duration = 0.0
        if audio.size and not skip_for_level and not no_speech_capture:
            self._load_model()
            inference_start = time.perf_counter()
            fallback_load = 0.0
            try:
                raw, info, segments = self._decode(audio, language)
            except Exception as error:
                if (
                    self.execution_device != "cuda"
                    or self._cpu_fallback_used
                    or not _gpu_error(error)
                ):
                    if isinstance(error, ValueError):
                        raise
                    raise SpeechToTextError(
                        "Transcription failed; check model/audio settings."
                    ) from error
                fallback_start = time.perf_counter()
                try:
                    self._fallback_to_cpu()
                except Exception as fallback_error:
                    self._initialization_error = (
                        "CPU fallback failed. Check the model installation."
                    )
                    raise ModelInitializationError(self._initialization_error) from fallback_error
                finally:
                    fallback_load = time.perf_counter() - fallback_start
                    self.model_load_duration += fallback_load
                try:
                    raw, info, segments = self._decode(audio, language)
                except Exception as cpu_error:
                    raise SpeechToTextError(
                        "Transcription failed after CPU fallback."
                    ) from cpu_error
            transcription_duration = time.perf_counter() - inference_start - fallback_load
        normalization_start = time.perf_counter()
        normalized = self.normalizer.normalize(raw)
        normalization_duration = time.perf_counter() - normalization_start
        has_lexical_content = any(unicodedata.category(char)[0] in {"L", "N"} for char in raw)
        if raw and not has_lexical_content:
            warnings.append("Model output contains no letters or numbers. Please re-record.")
        if self._fallback_reason:
            warnings.append(self._fallback_reason)
        return {
            "text": normalized.normalized_text,
            "raw_text": raw,
            "normalized_text": normalized.normalized_text,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "language_mode": selection.mode,
            "language_selection_source": selection.source,
            # Forced language probabilities are library placeholders (usually 1), not detection.
            "detected_language": getattr(info, "language", None) if language is None else None,
            "language_probability_kind": (
                "unavailable" if info is None else "detection" if language is None else "forced"
            ),
            "language_candidates": list(getattr(info, "all_language_probs", None) or []),
            "status": "ok"
            if has_lexical_content
            else "unusable_output"
            if raw
            else "no_speech"
            if skip_for_level or no_speech_capture
            else "empty_output",
            "empty_transcription": not bool(raw),
            "recording_duration": audio.size / self.sample_rate,
            "recording_time": 0.0,
            "model_load_duration": self.model_load_duration - load_before,
            "transcription_duration": max(0.0, transcription_duration),
            "normalization_duration": normalization_duration,
            "total_duration": time.perf_counter() - start,
            "audio_diagnostics": diagnostics,
            "warnings": warnings,
            "corrections": list(normalized.corrections),
            "review_flags": list(normalized.review_flags),
            "execution_device": self.execution_device,
            "compute_type": self.compute_type,
            "speech_duration_after_vad": getattr(info, "duration_after_vad", None),
            "multilingual_decoding": self.config.multilingual and language is None,
            "segments": segments,
            "has_lexical_content": has_lexical_content,
            "recording_info": capture_info,
        }

    def listen_and_transcribe(
        self, duration=None, language=USE_CONFIG_LANGUAGE, *, on_recording_started=None
    ) -> TranscriptResult:
        start = time.perf_counter()
        self.last_capture_files = None
        audio = (
            self.record_audio(duration, on_started=on_recording_started)
            if on_recording_started is not None
            else self.record_audio(duration)
        )
        recording_time = time.perf_counter() - start
        files = None
        if self.config.recording_directory:
            try:
                files = preserve_capture(
                    audio,
                    self.config.recording_directory,
                    self.last_recording_info or {},
                    self.configuration(),
                )
                self.last_capture_files = files
            except OSError as error:
                raise AudioCaptureError(
                    "Cannot preserve capture. Check the recording directory."
                ) from error
        try:
            result = self.transcribe(audio, language, recording_info=self.last_recording_info)
        except Exception:
            if files:
                try:
                    finish_capture(files, None)
                except OSError:
                    logger.error("Audio was preserved, but failure metadata could not be updated.")
            raise
        result["recording_time"] = recording_time
        result["total_duration"] = time.perf_counter() - start
        if files:
            result["recording_files"] = files
            try:
                finish_capture(files, result)
            except OSError as error:
                raise AudioCaptureError(
                    "Audio saved, but transcript metadata could not be saved."
                ) from error
        return result
