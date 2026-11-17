"""Offline candidate/measurement contracts, with mock transcripts and no hardware claims."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import main
from audio import AudioConfig
from audio.benchmark import grouped_summaries, language_coverage, language_group
from audio.comparison_report import render_comparison
from audio.stt_profiles import apply_stt_profile
from audio.validation import replay_endpoint


def test_candidate_freezes_measured_settings_without_promoting_default():
    config = AudioConfig(language="en", beam_size=1, initial_prompt="unvalidated hint")
    assert apply_stt_profile(config, None) is config
    candidate = apply_stt_profile(config, "medium-candidate")
    assert "medium" in candidate.model_size
    assert candidate.device == "cuda" and candidate.compute_type == "int8_float16"
    assert candidate.language is None and candidate.multilingual
    assert candidate.initial_prompt == "" and candidate.beam_size == 5
    assert candidate.temperatures == (0.0,)
    assert (candidate.vad_filter, candidate.vad_threshold) == (True, 0.5)
    assert (candidate.vad_min_silence_ms, candidate.vad_speech_pad_ms) == (500, 300)
    assert candidate.recording_mode == "automatic" and candidate.silence_duration == 2
    assert AudioConfig().model_size == "small"
    with pytest.raises(ValueError, match="Unknown"):
        apply_stt_profile(config, "not-a-profile")


def test_actual_cli_uses_shared_candidate_and_preserves_explicit_overrides(monkeypatch, capsys):
    monkeypatch.setenv("VOICEOS_STT_MODEL", "small")
    monkeypatch.setenv("VOICEOS_STT_LANGUAGE", "en")
    monkeypatch.setenv("VOICEOS_STT_BEAM_SIZE", "1")
    monkeypatch.setenv("VOICEOS_STT_PROMPT", "unvalidated hint")
    factory = MagicMock()
    factory.return_value.load_model.return_value = 0.1
    factory.return_value.execution_device = "cuda"
    factory.return_value.compute_type = "int8_float16"
    monkeypatch.setattr("audio.SpeechToText", factory)
    monkeypatch.setattr("builtins.input", lambda prompt: "exit")
    main.main(["--voice", "--stt-profile", "medium-candidate"])
    assert factory.call_args.kwargs["config"] == apply_stt_profile(
        AudioConfig.from_env(), "medium-candidate"
    )
    assert "cuda/int8_float16" in capsys.readouterr().out
    factory.return_value.load_model.assert_called_once()
    factory.return_value.listen_and_transcribe.assert_not_called()
    main.main(["--voice", "--stt-profile", "medium-candidate", "--language", "en"])
    assert factory.call_args.kwargs["config"].language == "en"


def mock_row(sample, category, repeat, seconds, errors):
    metrics = {"reference_status": "verified", "errors": errors}
    return {
        "case_id": sample,
        "language_group": language_group({"category": category}),
        "profile": "baseline",
        "repeat": repeat,
        "transcript": {"transcription_duration": seconds},
        "raw_metrics": metrics,
        "normalized_metrics": metrics,
    }


def test_scores_stay_separate_and_warm_latency_excludes_first_pass():
    en_errors = {
        "word_errors": 1,
        "reference_words": 2,
        "character_errors": 2,
        "reference_characters": 10,
    }
    hi_errors = {**en_errors, "word_errors": 0, "character_errors": 0}
    rows = [
        mock_row("en", "english", 1, 20, en_errors),
        mock_row("en", "english", 2, 2, en_errors),
        mock_row("hi", "hindi", 1, 10, hi_errors),
        mock_row("hi", "hindi", 2, 1, hi_errors),
    ]
    languages = grouped_summaries(rows, "language_group")
    assert languages["english"]["baseline"]["raw_wer"] == 0.5
    assert languages["hindi"]["baseline"]["raw_wer"] == 0
    assert languages["english"]["baseline"]["median_warm_decode_seconds"] == 2
    assert languages["english"]["baseline"]["warm_calls"] == 1
    samples = grouped_summaries(rows, "case_id")
    assert samples["hi"]["baseline"]["raw_cer"] == 0
    coverage = language_coverage(
        [
            {"id": "en", "category": "english"},
            {"id": "hi", "category": "hindi"},
            {"id": "mix", "category": "hinglish_technical"},
        ],
        rows,
    )
    assert coverage["hinglish"]["missing_sample_ids"] == ["mix"]
    assert coverage["hinglish"]["verified_sample_ids"] == []
    assert language_group({"category": "unverified_saved"}) == "unclassified"


def test_coverage_never_counts_repeats_or_unverified_text_as_new_references():
    row = mock_row("mix", "hinglish_technical", 1, 1, None)
    row["raw_metrics"]["reference_status"] = "unavailable"
    coverage = language_coverage([{"id": "mix", "category": "hinglish_technical"}], [row, row])
    assert coverage["hinglish"]["available_sample_ids"] == ["mix"]
    assert coverage["hinglish"]["unverified_sample_ids"] == ["mix"]
    assert coverage["hinglish"]["verified_sample_ids"] == []
    assert coverage["english"]["verified_sample_ids"] == []


def test_endpoint_replay_uses_streaming_vad_without_microphone(monkeypatch):
    vad = MagicMock()
    vad.probability.side_effect = [0.9] * 5 + [0.01] * 63
    factory = MagicMock(return_value=vad)
    monkeypatch.setattr("audio.validation.StreamingSileroVad", factory)
    info = replay_endpoint(np.full(68 * 512, 0.02, np.float32), AudioConfig())
    assert info["stop_reason"] == "silence" and info["speech_started"]
    assert info["trailing_silence_duration"] == 2.016
    assert info["captured_duration"] == 68 * 512 / 16000
    factory.assert_called_once()
    assert vad.probability.call_count == 68
    assert "no physical microphone" in info["source"]


def test_comparison_exposes_missing_languages_and_unavailable_accuracy():
    report = {
        "configuration": {"model_size": "MOCK", "compute_type": "MOCK"},
        "model_load_duration": 0,
        "results": [],
        "coverage_by_language": language_coverage(
            [{"id": "english", "category": "english"}, {"id": "hindi", "category": "hindi"}], []
        ),
    }
    rendered = render_comparison([report])
    assert "english | 0 / 0" in rendered and "hindi | 0 / 0" in rendered
    assert "Unavailable" in rendered and "No accuracy winner" in rendered
