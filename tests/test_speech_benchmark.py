"""Offline metric/benchmark contracts. All decoded texts below are mock fixtures."""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio import AudioConfig
from audio import benchmark as bench


@pytest.mark.parametrize(
    "reference,hypothesis,distance",
    [
        ("kitten", "sitting", 3),
        ("", "abc", 3),
        ("abc", "", 3),
        (["a", "b", "c"], ["a", "c"], 1),
    ],
)
def test_edit_distance(reference, hypothesis, distance):
    assert bench.edit_distance(reference, hypothesis) == distance


def test_error_rates_and_declared_text_policy():
    assert bench.error_rates("Hello, Python!", "hello python")["wer"] == 0
    assert bench.error_rates("कल परीक्षा है।", "कल परीक्षा है")["cer"] == 0
    assert bench.error_rates("a", "a b c")["wer"] == 2
    empty = bench.error_rates("", "unexpected speech")
    assert empty["wer"] is None and empty["cer"] is None
    assert empty["word_errors"] == 2
    assert bench.error_rates("one two", "")["wer"] == 1


@pytest.mark.parametrize(
    "case",
    [
        {},
        {"reference_text": "example", "reference_verified": False},
        {"reference_text": "example", "reference_verified": True},
        {"reference_text": "example", "reference_verified": "true", "reference_source": "human"},
        {"reference_text": "example", "reference_verified": True, "reference_source": "whisper"},
    ],
)
def test_unverified_or_generated_reference_is_never_scored(case):
    assert bench.score_transcript(case, "example")["errors"] is None


def test_verified_reference_and_exact_term_recall():
    case = {
        "reference_text": "Keval Minda uses Python and VoiceOS",
        "reference_verified": True,
        "reference_source": "human",
        "terms": ["Python", "VoiceOS", "Gemini"],
        "names": ["Keval Minda"],
    }
    result = bench.score_transcript(case, "Keval Minda uses Python and voice os")
    assert result["errors"]["wer"] > 0
    assert result["terms"]["reference_terms"] == ["Python", "VoiceOS"]
    assert result["terms"]["recall"] == 0.5
    assert result["names"]["recall"] == 1
    assert bench.term_recall("Python", "pythonista", ["Python"])["recall"] == 0


@pytest.fixture
def manifest(tmp_path):
    audio = np.concatenate((np.full(8000, 0.02, np.float32), np.zeros(8000, np.float32)))
    np.save(tmp_path / "speech.npy", audio)
    path = tmp_path / "manifest.json"
    bench.write_json(
        path,
        {
            "cases": [
                {
                    "id": "available",
                    "category": "english",
                    "path": "speech.npy",
                    "reference_text": None,
                    "reference_verified": False,
                },
                {"id": "pending", "category": "hindi", "path": None, "reference_verified": False},
            ]
        },
    )
    return path


def test_model_reuse_shared_audio_and_missing_categories(manifest):
    engine = SimpleNamespace(config=AudioConfig(), load_model=MagicMock(return_value=0.5))
    engine.transcribe = MagicMock(
        return_value={
            "raw_text": "MOCK",
            "normalized_text": "MOCK",
            "transcription_duration": 0.01,
        }
    )
    engine.configuration = lambda: dict(language=engine.config.language)
    factory = MagicMock(return_value=engine)
    report = bench.run_benchmark(
        manifest, AudioConfig(), ["baseline", "english"], repeats=2, engine_factory=factory
    )
    factory.assert_called_once()
    engine.load_model.assert_called_once()
    assert report["model_load_duration"] == 0.5
    assert len(engine.transcribe.call_args_list) == 4
    original = engine.transcribe.call_args_list[0].args[0]
    assert all(call.args[0] is original for call in engine.transcribe.call_args_list)
    assert not original.flags.writeable
    assert report["summary"]["baseline"]["raw_wer"] is None
    assert report["summary"]["english"]["scored_calls"] == 0
    assert "hindi" in report["missing_categories"]
    assert report["results"][-1]["status"] == "missing_recording"


def test_reference_is_bound_to_audio_hash(manifest):
    reference = manifest.parent / "reference.txt"
    reference.write_text("Human-checked words", encoding="utf-8")
    bench.verify_reference(manifest, "available", reference)
    case = bench.load_manifest(manifest)["cases"][0]
    assert case["reference_source"] == "human" and case["reference_verified"]
    expected = hashlib.sha256(bench.load_recording(manifest.parent / case["path"]).tobytes())
    assert case["audio_sha256"] == expected.hexdigest()
    np.save(manifest.parent / case["path"], np.ones(16000, np.float32))
    with pytest.raises(ValueError, match="changed"):
        bench.verify_reference(manifest, "available", reference)


def test_duplicate_case_ids_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    bench.write_json(path, {"cases": [{"id": "same"}, {"id": "same"}]})
    with pytest.raises(ValueError):
        bench.load_manifest(path)


def test_capture_is_lossless_unverified_and_does_not_load_model(manifest, monkeypatch):
    audio = np.array([0.012345678, -0.00123456, 0], dtype=np.float32)
    engine = SimpleNamespace(record_audio=MagicMock(return_value=audio))
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(bench, "SpeechToText", factory)
    monkeypatch.setattr(bench.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(bench, "_beep", lambda frequency=880: None)
    bench.record_case(manifest, "pending", 12, wait=False, delay=0)
    case = bench.load_manifest(manifest)["cases"][1]
    np.testing.assert_array_equal(bench.load_recording(manifest.parent / case["path"]), audio)
    assert case["reference_text"] is None and case["reference_verified"] is False
    assert case["wav_path"] == "pending.wav"
    assert not hasattr(engine, "load_model")
    with pytest.raises(ValueError, match="preserved"):
        bench.record_case(manifest, "pending", 12, wait=False, delay=0)


def test_script_counts_do_not_infer_language():
    counts = bench.script_counts("Python और Gemini")
    assert counts["latin_letters"] == 12
    assert counts["devanagari_letters"] == 2
    assert "translation_error" not in counts


def test_automatic_reference_capture_uses_same_engine_and_stays_unverified(manifest, monkeypatch):
    audio = np.full(512, 0.02, np.float32)
    engine = SimpleNamespace(
        load_vad=MagicMock(),
        last_recording_info={"mode": "automatic", "stop_reason": "silence", "speech_started": True},
    )
    engine.record_audio = MagicMock(side_effect=lambda on_started: (on_started(), audio)[1])
    factory = MagicMock(return_value=engine)
    beep = MagicMock()
    monkeypatch.setattr(bench, "SpeechToText", factory)
    monkeypatch.setattr(bench, "_beep", beep)
    monkeypatch.setattr(bench.time, "sleep", lambda seconds: None)
    bench.record_case(manifest, "pending", 12, wait=False, delay=0, automatic=True, max_duration=30)
    config = factory.call_args.kwargs["config"]
    assert config.recording_mode == "automatic" and config.max_recording_duration == 30
    engine.load_vad.assert_called_once()
    beep.assert_not_called()
    case = bench.load_manifest(manifest)["cases"][1]
    assert case["reference_verified"] is False
    assert case["recording_info"]["stop_reason"] == "silence"
