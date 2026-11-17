"""Record once, compare language modes on identical audio; never invoke application actions."""

import argparse
import hashlib
import json
import math
import subprocess
import threading
import time
import wave
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from audio.audio_config import USE_CONFIG_LANGUAGE, AudioConfig, language_code, microphone_device
from audio.speech_to_text import SpeechToText, SpeechToTextError, list_input_devices


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Diagnostic WAV must be 16000 Hz mono PCM16.")
        return (
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        )


def save_wav(path: str, audio: np.ndarray):
    # Never overwrite a previous recording, including the supplied test_audio.wav.
    with Path(path).open("xb") as output, wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((audio * 32767).astype("<i2").tobytes())


class GpuMemorySampler:
    """Optional sampled total GPU memory, including other processes; not exact process peak."""

    def __init__(self):
        self.samples = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stopped.is_set():
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=2,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    stderr=subprocess.DEVNULL,
                )
                self.samples.append([int(line) for line in value.strip().splitlines()])
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self.stopped.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join(timeout=3)

    def report(self, start_index=0):
        samples = self.samples[start_index:]
        return {
            "samples": len(samples),
            "sampled_peak_total_mib": max((max(s) for s in samples if s), default=None),
            "scope": "Total device memory including other processes; sampled, not exact peak.",
        }


def compare_modes(
    stt: SpeechToText,
    audio: np.ndarray,
    repeats: int = 1,
    language=USE_CONFIG_LANGUAGE,
) -> list[dict]:
    if not 1 <= repeats <= 10:
        raise ValueError("Repeats must be from 1 to 10.")
    # Explicit en/hi/None override any environment language setting.
    modes = ("en",) if stt.config.model_size.endswith(".en") else ("en", "hi", None)
    if language is not USE_CONFIG_LANGUAGE:
        modes = (language_code(language),)
    results = []
    for repeat in range(repeats):
        for language in modes:
            result = stt.transcribe(audio, language=language)
            results.append({"repeat": repeat + 1, **result})
    return results


def stress_audio_cases(audio: np.ndarray, sample_rate: int = 16000):
    """Deterministic derived signals; these are not new human speech recordings."""
    if not audio.size:
        raise ValueError("Stress cases require a nonempty source recording.")
    rng = np.random.default_rng(42)
    noisy = np.clip(audio + rng.normal(0, 0.02, audio.shape), -1, 1).astype(np.float32)
    noise_only = rng.normal(0, 0.02, sample_rate * 2).astype(np.float32)
    long_frames = sample_rate * 45
    long_audio = np.tile(audio, math.ceil(long_frames / audio.size))[:long_frames]
    return [
        ("silence", "Synthetic digital silence, 2 seconds", np.zeros(sample_rate * 2, np.float32)),
        ("noise_only", "Synthetic seeded Gaussian noise, RMS approximately 0.02", noise_only),
        ("noisy", "Saved speech plus seeded Gaussian noise, target noise RMS 0.02", noisy),
        (
            "short",
            "First 2 seconds of saved speech; may cut words",
            audio[: sample_rate * 2].copy(),
        ),
        ("long", "Saved speech tiled/trimmed to 45 seconds; not natural long speech", long_audio),
    ]


def compare_stress_cases(stt, audio, repeats=1, language=USE_CONFIG_LANGUAGE):
    cases = []
    for case_id, description, samples in stress_audio_cases(audio):
        samples.setflags(write=False)
        cases.append(
            {
                "case_id": case_id,
                "provenance": description,
                "audio_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                "results": compare_modes(stt, samples, repeats, language),
            }
        )
    return cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input", help="Reuse a 16 kHz mono PCM16 WAV instead of recording.")
    parser.add_argument("--save-audio", help="Save the single recording to a new WAV file.")
    parser.add_argument("--output", help="Save JSON report (contains transcripts).")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--microphone")
    parser.add_argument(
        "--language",
        "--preferred-language",
        choices=("auto", "en", "hi"),
        help="Test one selected mode; omit to compare en/hi/auto on the same audio.",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="With --input, also evaluate deterministic silence/noise/short/long derivatives.",
    )
    parser.add_argument("--model")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--compute-type")
    parser.add_argument("--prompt", help="Vocabulary hint; use an empty string to disable.")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--multilingual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="Include warm passes; maximum 10.")
    parser.add_argument("--gpu-memory", action="store_true")
    args = parser.parse_args(argv)
    if args.stress and not args.input:
        parser.error("--stress requires --input to make derived cases repeatable.")
    load_dotenv()
    try:
        if args.list_devices:
            print(json.dumps(list_input_devices(), ensure_ascii=True, indent=2))
            return 0
        overrides = {
            key: value
            for key, value in {
                "duration": args.duration,
                "microphone_id": microphone_device(args.microphone),
                "model_size": args.model,
                "device": args.device,
                "compute_type": args.compute_type,
                "initial_prompt": args.prompt,
                "multilingual": args.multilingual,
            }.items()
            if value is not None
        }
        if args.no_vad:
            overrides["vad_filter"] = False
        if args.local_files_only:
            overrides["local_files_only"] = True
        config = replace(AudioConfig.from_env(), recording_mode="fixed", **overrides)
        stt = SpeechToText(config=config)
        if not 1 <= args.repeat <= 10:
            raise ValueError("Repeats must be from 1 to 10.")
        started = time.perf_counter()
        if args.input:
            audio = load_wav(args.input)
            recording_time = 0.0
        else:
            input(
                f"Press Enter, then speak for up to {config.duration:g}s; leave trailing silence: "
            )
            started = time.perf_counter()
            record_start = time.perf_counter()
            audio = stt.record_audio()
            recording_time = time.perf_counter() - record_start
        if args.save_audio:
            save_wav(args.save_audio, audio)
        audio.setflags(write=False)
        requested = args.language if args.language is not None else USE_CONFIG_LANGUAGE
        with GpuMemorySampler() if args.gpu_memory else nullcontext() as sampler:
            results = compare_modes(stt, audio, args.repeat, requested)
            stress_results = (
                compare_stress_cases(stt, audio, args.repeat, requested) if args.stress else []
            )
        memory = sampler.report() if sampler is not None else None
        report = {
            "configuration": stt.configuration(),
            "input_source": str(Path(args.input)) if args.input else "microphone",
            "evaluation_mode": args.language or "compare_all",
            "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
            "recording_time": recording_time,
            "workflow_duration": time.perf_counter() - started,
            "model_load_duration": stt.model_load_duration,
            "gpu_memory": memory,
            "probability_note": "Language probability is not transcription accuracy; forced modes "
            "may report 1.0. Cold first pass includes lazy runtime/VAD setup.",
            "results": results,
            "stress_cases": stress_results,
        }
        serialized = json.dumps(report, ensure_ascii=True, indent=2)
        if args.output:
            Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0
    except (SpeechToTextError, ValueError, OSError, wave.Error) as error:
        print(f"Diagnostic failed: {error}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("Diagnostic cancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
