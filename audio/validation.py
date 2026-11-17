"""Manual hardware validation of the real voice loop using saved audio, never live capture.

Only the microphone source, terminal answers and application action/storage boundaries
are substituted. Whisper, Silero, normalization, archiving and transcript review run.
"""

import argparse
import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from audio import AudioConfig, SpeechToText
from audio.benchmark import load_manifest, load_recording, score_transcript, write_json
from audio.diagnostics import GpuMemorySampler
from audio.recording import FRAME_SAMPLES, SilenceEndpoint, StreamingSileroVad


def replay_endpoint(audio: np.ndarray, config: AudioConfig) -> dict:
    """Exercise real streaming Silero/state transitions without opening a device."""
    vad = StreamingSileroVad()
    endpoint = SilenceEndpoint(config)
    for offset in range(0, audio.size, FRAME_SAMPLES):
        frame = audio[offset : offset + FRAME_SAMPLES]
        if frame.size != FRAME_SAMPLES:
            break
        endpoint.accept(vad.probability(frame))
        if endpoint.stop_reason:
            break
    return {**endpoint.information(), "source": "saved waveform; no physical microphone"}


def validate_application(manifest: Path, case_id: str, archive: Path) -> dict:
    import audio as audio_package
    import main as application

    case = next(case for case in load_manifest(manifest)["cases"] if case["id"] == case_id)
    waveform = load_recording(manifest.parent / case["path"])
    digest = hashlib.sha256(waveform.tobytes()).hexdigest()
    if digest != case.get("audio_sha256"):
        raise ValueError("Application replay requires a hash-bound preserved recording.")
    captures = []
    instances = []
    model_creations = []

    class ReplaySpeechToText(SpeechToText):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

        def _create_model(self, device, compute_type):
            model_creations.append({"device": device, "compute_type": compute_type})
            return super()._create_model(device, compute_type)

        def record_audio(self, duration=None, *, on_started=None):
            self.last_recording_info = dict(case.get("recording_info") or {})
            self.last_recording_info.update(
                {"source": "saved_waveform_replay", "source_audio_sha256": digest}
            )
            if on_started:
                on_started()
            return waveform.copy()

        def listen_and_transcribe(self, *args, **kwargs):
            result = super().listen_and_transcribe(*args, **kwargs)
            captures.append(result)
            return result

    answers = iter(["", "no", "", "no", "exit"])
    commands = MagicMock(side_effect=AssertionError("Replay must never submit an action."))
    storage = MagicMock()
    print("SAVED-AUDIO APPLICATION REPLAY: microphone and terminal answers are substituted.")
    with (
        GpuMemorySampler() as sampler,
        patch.object(audio_package, "SpeechToText", ReplaySpeechToText),
        patch("builtins.input", side_effect=lambda prompt: next(answers)),
        patch.object(application, "process_text_command", commands),
        patch.object(application, "initialize_database", storage),
        patch.dict(os.environ, {"VOICEOS_STT_LOCAL_FILES_ONLY": "true"}),
    ):
        application.main(
            [
                "--voice",
                "--stt-profile",
                "medium-candidate",
                "--save-recordings",
                str(archive),
            ]
        )
    if len(instances) != 1 or len(captures) != 2 or len(model_creations) != 1:
        raise RuntimeError("Application replay did not complete two captures with one model.")
    if commands.call_count or any(
        row["execution_device"] != "cuda" or row["compute_type"] != "int8_float16"
        for row in captures
    ):
        raise RuntimeError("Application candidate validation failed; inspect execution/review.")
    engine = instances[0]
    return {
        "validation_kind": "real application, saved-waveform replay; no live microphone",
        "substitutions": [
            "microphone source",
            "terminal no/no/exit answers",
            "database initialization",
            "guarded action boundary",
        ],
        "audio_sha256": digest,
        "reference_text": case.get("reference_text")
        if case.get("reference_verified") and case.get("reference_source") == "human"
        else None,
        "configuration": engine.configuration(),
        "model_creations": model_creations,
        "model_load_duration": engine.model_load_duration,
        "vad_load_duration": engine.vad_load_duration,
        "command_submissions": commands.call_count,
        "transcripts": captures,
        "raw_metrics": [score_transcript(case, row["raw_text"]) for row in captures],
        "normalized_metrics": [score_transcript(case, row["normalized_text"]) for row in captures],
        "gpu_memory": sampler.report(),
        "endpoint_replay": replay_endpoint(waveform, engine.config),
        "digital_silence_endpoint": {
            **replay_endpoint(np.zeros(16000 * 9, np.float32), engine.config),
            "source": "synthetic digital silence; no physical microphone",
        },
        "timing_note": "Replay recording_time is memory copy time, not physical capture latency.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_application(args.manifest, args.case, args.archive)
    write_json(args.output, report)
    print(f"Saved application validation: {args.output}")


if __name__ == "__main__":
    main()
