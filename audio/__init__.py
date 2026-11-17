"""VoiceOS speech support. No microphone/model side effects on import."""

from audio.audio_config import AudioConfig
from audio.speech_to_text import SpeechToText, SpeechToTextError, list_input_devices

__all__ = ["AudioConfig", "SpeechToText", "SpeechToTextError", "list_input_devices"]
