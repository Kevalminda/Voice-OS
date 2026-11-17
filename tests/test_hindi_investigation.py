"""Offline contracts, using mocked recognition/VAD decisions rather than accuracy fixtures."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio import AudioConfig, SpeechToText, benchmark
from audio.capture_archive import finish_capture, preserve_capture
from audio.comparison_report import render_comparison
from audio.diagnostics import GpuMemorySampler
from audio.vad_diagnostics import export_discarded, inspect_vad


def test_archive_is_lossless_unique_and_reference_unavailable(tmp_path):
    audio = np.array([0.012345678, -0.000987654], np.float32)
    first = preserve_capture(audio, str(tmp_path), {"stop_reason": "silence"}, {})
    second = preserve_capture(audio, str(tmp_path), {}, {})
    assert first != second
    np.testing.assert_array_equal(np.load(first["audio"]), audio)
    result = {"raw_text": "MOCK raw", "normalized_text": "MOCK normalized"}
    finish_capture(first, result)
    metadata = json.loads(Path(first["metadata"]).read_text(encoding="utf-8"))
    assert metadata["transcript"] == result
    assert metadata["reference_text"] is None and metadata["reference_verified"] is False


def test_failed_inference_preserves_audio(tmp_path, monkeypatch):
    engine = SpeechToText(config=AudioConfig(recording_directory=str(tmp_path)))
    audio = np.full(16000, 0.02, np.float32)
    monkeypatch.setattr(engine, "record_audio", lambda duration: audio)
    monkeypatch.setattr(engine, "transcribe", MagicMock(side_effect=RuntimeError("model failure")))
    with pytest.raises(RuntimeError, match="model failure"):
        engine.listen_and_transcribe()
    np.testing.assert_array_equal(np.load(engine.last_capture_files["audio"]), audio)
    metadata = json.loads(Path(engine.last_capture_files["metadata"]).read_text(encoding="utf-8"))
    assert metadata["status"] == "transcription_failed"


def test_failed_capture_does_not_report_previous_recording(tmp_path, monkeypatch):
    engine = SpeechToText(config=AudioConfig(recording_directory=str(tmp_path)))
    engine.last_capture_files = {"metadata": "previous.json"}
    monkeypatch.setattr(
        engine, "record_audio", MagicMock(side_effect=RuntimeError("device failure"))
    )
    with pytest.raises(RuntimeError, match="device failure"):
        engine.listen_and_transcribe()
    assert engine.last_capture_files is None


def test_import_never_uses_model_text_as_reference(tmp_path):
    manifest = tmp_path / "session.json"
    benchmark.write_json(manifest, {"cases": [{"id": "hindi", "path": None}]})
    files = preserve_capture(np.full(16000, 0.02, np.float32), str(tmp_path), {}, {})
    finish_capture(files, {"raw_text": "MODEL OUTPUT"})
    benchmark.import_capture(manifest, Path(files["metadata"]), "hindi")
    case = benchmark.load_manifest(manifest)["cases"][0]
    assert case["reference_text"] is None and not case["reference_verified"]
    assert benchmark.score_transcript(case, "MODEL OUTPUT")["errors"] is None


def test_reference_withdrawal_preserves_waveform_and_excludes_scores(tmp_path):
    audio_path = tmp_path / "captured.npy"
    np.save(audio_path, np.full(16000, 0.02, np.float32))
    original = audio_path.read_bytes()
    manifest = tmp_path / "session.json"
    benchmark.write_json(
        manifest,
        {
            "cases": [
                {
                    "id": "hindi",
                    "path": "captured.npy",
                    "reference_text": "MOCK reference",
                    "reference_verified": True,
                    "reference_source": "human",
                }
            ]
        },
    )
    benchmark.invalidate_reference(manifest, "hindi", "Wrong speaker confirmed on playback")
    benchmark.invalidate_reference(manifest, "hindi", "Playback confirmed background speech")
    case = benchmark.load_manifest(manifest)["cases"][0]
    assert case["withdrawn_reference_text"] == "MOCK reference"
    assert case["reference_text"] is None and case["reference_verified"] is False
    assert case["reference_source"] is None
    assert audio_path.read_bytes() == original
    metrics = benchmark.score_transcript(case, "MOCK reference")
    assert metrics["errors"] is None and metrics["reference_status"] == "unavailable"
    row = {
        "profile": "invest_base",
        "transcript": {"transcription_duration": 0.1},
        "raw_metrics": metrics,
        "normalized_metrics": metrics,
    }
    summary = benchmark.aggregate_results([row])["invest_base"]
    assert summary["scored_calls"] == 0
    assert summary["raw_wer"] is None and summary["normalized_cer"] is None


def test_vad_audit_reports_complement_without_claiming_speech_loss(monkeypatch, tmp_path):
    mock = MagicMock(return_value=[{"start": 100, "end": 300}, {"start": 500, "end": 900}])
    monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps", mock)
    audio = np.full(1000, 0.02, np.float32)
    audit = inspect_vad(audio, AudioConfig())
    assert audit["retained_duration"] == 600 / 16000
    assert audit["discarded_duration"] == 400 / 16000
    assert [(r["start_sample"], r["end_sample"]) for r in audit["discarded_ranges"]] == [
        (0, 100),
        (300, 500),
        (900, 1000),
    ]
    assert audit["lost_final_words"] is None and audit["lost_quiet_syllables"] is None
    files = export_discarded(audio, audit, tmp_path, "hindi")
    np.testing.assert_array_equal(np.load(files[-1]["raw"]), audio[900:])
    with pytest.raises(FileExistsError):
        export_discarded(audio, audit, tmp_path, "hindi")


def test_disabled_vad_keeps_every_sample():
    audit = inspect_vad(np.zeros(16000, np.float32), AudioConfig(vad_filter=False))
    assert audit["discarded_duration"] == 0 and not audit["discarded_ranges"]


def test_call_memory_samples_do_not_inherit_previous_peaks():
    sampler = GpuMemorySampler()
    sampler.samples = [[1900], [800], [900]]
    assert sampler.report(1)["sampled_peak_total_mib"] == 900
    assert sampler.report(3)["sampled_peak_total_mib"] is None


def test_investigation_base_controls_and_prompts_are_explicit():
    assert benchmark.PROFILE_OVERRIDES["invest_base"]["language"] is None
    assert benchmark.PROFILE_OVERRIDES["invest_base"]["initial_prompt"] == ""
    for name in benchmark.INVESTIGATION_PROFILES:
        AudioConfig(**benchmark.PROFILE_OVERRIDES[name])
    assert benchmark.PROFILE_OVERRIDES["invest_forced_hi"]["multilingual"] is False
    assert benchmark.PROFILE_OVERRIDES["invest_prompt_hindi"]["language"] is None


def test_report_does_not_select_accuracy_winner_from_latency():
    config = {
        "model_size": "small",
        "compute_type": "float16",
        "language": None,
        "beam_size": 5,
        "initial_prompt": "",
        "vad_filter": False,
    }
    metrics = {"reference_status": "unavailable", "errors": None}
    report = {
        "source_report": "mock-run.json",
        "reference_validation_note": "MOCK withdrawn association",
        "configuration": config,
        "model_load_duration": 1,
        "results": [
            {
                "case_id": "saved",
                "audio_sha256": "a" * 64,
                "profile": "invest_base",
                "repeat": 1,
                "configuration": config,
                "raw_metrics": metrics,
                "normalized_metrics": metrics,
                "transcript": {
                    "raw_text": "MOCK | <script>",
                    "normalized_text": "MOCK",
                    "transcription_duration": 0.1,
                },
            }
        ],
    }
    rendered = render_comparison([report])
    assert "No accuracy winner" in rendered
    assert "Unavailable" in rendered
    assert "&lt;script&gt;" in rendered and "MOCK \\|" in rendered
    assert "mock-run.json" in rendered and "MOCK withdrawn association" in rendered
    assert "Source report" in rendered


def test_cuda_requirement_rejects_fallback_before_decoding(tmp_path):
    manifest = tmp_path / "session.json"
    benchmark.write_json(manifest, {"cases": []})
    engine = SimpleNamespace(load_model=lambda: 0.1, execution_device="cpu")
    with pytest.raises(RuntimeError, match="CUDA was required"):
        benchmark.run_benchmark(
            manifest,
            AudioConfig(),
            ["invest_base"],
            engine_factory=lambda config: engine,
            require_cuda=True,
        )
