"""Explicit opt-in to the measured candidate; application defaults remain independent."""

from dataclasses import replace
from pathlib import Path

from audio.audio_config import AudioConfig

STT_PROFILES = ("medium-candidate",)


def apply_stt_profile(config: AudioConfig, profile: str | None) -> AudioConfig:
    if profile is None:
        return config
    if profile != "medium-candidate":
        raise ValueError("Unknown STT profile.")
    cached = Path(__file__).resolve().parents[1] / ".models" / "faster-whisper-medium"
    return replace(
        config,
        model_size=str(cached) if (cached / "model.bin").is_file() else "medium",
        device="cuda",
        compute_type="int8_float16",
        language=None,
        multilingual=True,
        beam_size=5,
        initial_prompt="",
        temperatures=(0.0,),
        vad_filter=True,
        vad_threshold=0.5,
        vad_min_silence_ms=500,
        vad_speech_pad_ms=300,
    )
