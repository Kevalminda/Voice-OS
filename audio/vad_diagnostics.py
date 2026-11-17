"""Inspect discarded waveform intervals; VAD decisions are not human speech-loss labels."""

import time
from pathlib import Path

import numpy as np

from audio.audio_config import AudioConfig


def inspect_vad(audio: np.ndarray, config: AudioConfig) -> dict:
    started = time.perf_counter()
    if config.vad_filter:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        ranges = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=config.vad_threshold,
                min_speech_duration_ms=0,
                min_silence_duration_ms=config.vad_min_silence_ms,
                speech_pad_ms=config.vad_speech_pad_ms,
            ),
        )
    else:
        ranges = [{"start": 0, "end": audio.size}] if audio.size else []
    retained = []
    discarded = []
    cursor = 0
    for region in ranges:
        start, end = max(cursor, int(region["start"])), min(audio.size, int(region["end"]))
        if start > cursor:
            discarded.append((cursor, start))
        if end > start:
            retained.append({"start_sample": start, "end_sample": end})
        cursor = max(cursor, end)
    if cursor < audio.size:
        discarded.append((cursor, audio.size))
    details = []
    for start, end in discarded:
        part = audio[start:end]
        details.append(
            {
                "start_sample": start,
                "end_sample": end,
                "start_seconds": start / config.sample_rate,
                "end_seconds": end / config.sample_rate,
                "rms": float(np.sqrt(np.mean(np.square(part, dtype=np.float64)))),
                "peak": float(np.max(np.abs(part))),
            }
        )
    kept = sum(region["end_sample"] - region["start_sample"] for region in retained)
    return {
        "retained_ranges": retained,
        "discarded_ranges": details,
        "input_duration": audio.size / config.sample_rate,
        "retained_duration": kept / config.sample_rate,
        "discarded_duration": (audio.size - kept) / config.sample_rate,
        "audit_duration": time.perf_counter() - started,
        "lost_final_words": None,
        "lost_quiet_syllables": None,
        "interpretation": "Discarded signal can include speech; human review is required.",
    }


def export_discarded(audio: np.ndarray, audit: dict, directory: Path, identifier: str):
    """Lossless excerpts plus playback WAVs, without overwriting previous exports."""
    from audio.diagnostics import save_wav

    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for index, region in enumerate(audit["discarded_ranges"]):
        part = audio[region["start_sample"] : region["end_sample"]]
        stem = directory / f"{identifier}-discarded-{index}"
        raw = stem.with_suffix(".npy")
        wav = stem.with_suffix(".wav")
        with raw.open("xb") as output:
            np.save(output, part, allow_pickle=False)
        save_wav(str(wav), part)
        files.append({"raw": str(raw), "playback": str(wav)})
    return files
