"""Offline voice-confirmation plumbing; authored STT outputs are not accuracy evidence."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import main
from actions.notes import get_note, update_note
from audio import AudioConfig
from audio.speech_to_text import AudioCaptureError
from core.action_router import route_command
from core.confirmation import ConfirmationSession
from core.schemas import ActionResult, CommandRequest
from core.voice_confirmation import VoiceConfirmationReader


def transcript(raw="yes A7K2"):
    return {
        "status": "ok",
        "raw_text": raw,
        "normalized_text": raw,
        "empty_transcription": False,
        "has_lexical_content": True,
        "warnings": [],
        "review_flags": [],
        "corrections": [],
        "audio_diagnostics": {
            "possible_truncation": False,
            "near_silent": False,
            "clipped_fraction": 0.0,
        },
        "recording_info": {"mode": "automatic", "speech_started": True, "stop_reason": "silence"},
        "segments": [],
        "language_probability": 0.97,
    }


def pending(note_id=1):
    return ActionResult(
        status="needs_confirmation",
        message=f"Delete note {note_id}: Test note",
        data={"note_id": note_id, "preview": "Test note", "original_content": "Test note"},
    )


@pytest.fixture
def setup(monkeypatch):
    now = [0.0]
    confirmation = ConfirmationSession(clock=lambda: now[0], token_factory=lambda: "A7K2")
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)
    stt = SimpleNamespace(
        config=AudioConfig(),
        last_capture_files=None,
        last_recording_info={"mode": "automatic", "speech_started": True, "stop_reason": "silence"},
        record_audio=MagicMock(return_value=np.full(1600, 0.05, dtype=np.float32)),
        transcribe=MagicMock(return_value=transcript()),
        load_model=MagicMock(),
        load_vad=MagicMock(),
    )
    return stt, confirmation, now


def reader_with_answers(stt, answers):
    values = iter(answers)
    return VoiceConfirmationReader(stt, read_input=lambda prompt: next(values))


def run_delete(existing_note, reader):
    command = CommandRequest(
        intent="delete_note",
        entities={"note_id": str(existing_note)},
        confidence=0.95,
    )
    return route_command(command, lambda request: main._confirm_deletion(request, reader))


@pytest.mark.parametrize("reply", ["yes A7K2", "हाँ A7K2", "haan A7K2"])
def test_reviewed_spoken_approval_uses_existing_stt_without_nlu(
    monkeypatch,
    setup,
    existing_note,
    reply,
):
    stt, confirmation, _ = setup
    stt.transcribe.return_value = transcript(reply)
    nlu = MagicMock(side_effect=AssertionError("Confirmation reached NLU"))
    monkeypatch.setattr("core.command_processor.understand_command", nlu)
    reader = reader_with_answers(stt, ["", "yes"])

    assert run_delete(existing_note, reader).status == "success"
    assert get_note(existing_note).status == "error"
    stt.record_audio.assert_called_once()
    assert "duration" not in stt.record_audio.call_args.kwargs
    assert not stt.record_audio.call_args.args
    stt.transcribe.assert_called_once()
    stt.load_model.assert_not_called()
    stt.load_vad.assert_not_called()
    nlu.assert_not_called()
    assert reader.last_transcript["raw_text"] == reply
    assert not confirmation.has_pending


@pytest.mark.parametrize("reply", ["no", "नहीं", "nahi", "cancel", "रद्द करो", "cancel karo", "edit"])
def test_reviewed_spoken_rejection_cancellation_and_edit_never_mutate(
    setup,
    isolated_database,
    existing_note,
    reply,
):
    stt, confirmation, _ = setup
    stt.transcribe.return_value = transcript(reply)
    reader = reader_with_answers(stt, ["", "yes"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before
    assert not confirmation.has_pending


@pytest.mark.parametrize(
    "reply",
    [
        "yes WRONG",
        "yes",
        "हाँ",
        "haan",
        "yes A 7 K 2",
        "yes a7k2",
        "yes A7K2.",
        "yes A7K2 and delete note 2",
    ],
)
def test_wrong_missing_spaced_or_modified_tokens_never_approve(
    setup,
    isolated_database,
    existing_note,
    reply,
):
    stt, _, _ = setup
    stt.transcribe.return_value = transcript(reply)
    reader = reader_with_answers(stt, ["", "yes", "cancel"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before


def test_normalized_text_cannot_repair_raw_token(setup, isolated_database, existing_note):
    stt, _, _ = setup
    result = transcript("yes A 7 K 2")
    result["normalized_text"] = "yes A7K2"
    result["corrections"] = [("A 7 K 2", "A7K2")]
    stt.transcribe.return_value = result
    reader = reader_with_answers(stt, ["", "yes", "cancel"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before
    assert reader.last_transcript == result


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "uncertain",
        "warning",
        "clipping",
        "truncated",
        "max_duration",
        "silence",
        "near_silent",
        "missing_metadata",
    ],
)
def test_unusable_transcripts_never_reach_approval(
    setup,
    isolated_database,
    existing_note,
    fault,
):
    stt, _, _ = setup
    result = transcript()
    if fault == "empty":
        result.update(raw_text="", empty_transcription=True, has_lexical_content=False)
    elif fault == "uncertain":
        result["review_flags"] = [{"kind": "number", "text": "7", "message": "Verify"}]
    elif fault == "warning":
        result["warnings"] = ["Uncertain speech"]
    elif fault == "clipping":
        result["audio_diagnostics"]["clipped_fraction"] = 0.001
    elif fault == "truncated":
        result["audio_diagnostics"]["possible_truncation"] = True
    elif fault == "max_duration":
        result["recording_info"]["stop_reason"] = "max_duration"
    elif fault == "silence":
        result["recording_info"].update(speech_started=False, stop_reason="initial_silence_timeout")
    elif fault == "near_silent":
        result["audio_diagnostics"]["near_silent"] = True
    else:
        result.pop("audio_diagnostics")
    stt.transcribe.return_value = result
    reader = reader_with_answers(stt, ["", "cancel"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before
    assert reader.last_transcript == result


def test_failed_capture_offers_keyboard_fallback_without_new_deadline(setup, existing_note):
    stt, confirmation, now = setup
    stt.record_audio.side_effect = AudioCaptureError("No microphone")
    reader = reader_with_answers(stt, ["", "keyboard", "yes A7K2"])

    assert run_delete(existing_note, reader).status == "success"
    stt.transcribe.assert_not_called()
    assert not confirmation.has_pending
    assert now[0] == 0.0


def test_expiry_during_keyboard_fallback_is_not_extended(setup, existing_note, isolated_database):
    stt, _, now = setup
    stt.record_audio.side_effect = AudioCaptureError("No microphone")
    answers = iter(["", "keyboard", "yes A7K2"])

    def answer(prompt):
        value = next(answers)
        if value.startswith("yes"):
            now[0] = 60.0
        return value

    reader = VoiceConfirmationReader(stt, read_input=answer)
    before = isolated_database.read_bytes()
    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before


@pytest.mark.parametrize("stage", ["activation", "recording", "transcription", "review"])
def test_expiry_at_each_stage_never_approves(setup, existing_note, isolated_database, stage):
    stt, confirmation, now = setup
    replies = iter(["", "yes"])

    def answer(prompt):
        value = next(replies)
        if stage == "activation" or (stage == "review" and value == "yes"):
            now[0] = 60.0
        return value

    def record(**kwargs):
        if stage == "recording":
            now[0] = 60.0
        return np.zeros(1600, dtype=np.float32)

    def transcribe_audio(audio, **kwargs):
        if stage == "transcription":
            now[0] = 60.0
        return transcript()

    stt.record_audio.side_effect = record
    stt.transcribe.side_effect = transcribe_audio
    reader = VoiceConfirmationReader(stt, read_input=answer)
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before
    assert not confirmation.has_pending
    if stage == "activation":
        stt.record_audio.assert_not_called()
    if stage in {"activation", "recording"}:
        stt.transcribe.assert_not_called()


def test_already_expired_request_does_not_activate_microphone(setup):
    stt, confirmation, now = setup
    request = pending()
    token = confirmation.begin(request)
    now[0] = 60.0
    read_input = MagicMock(side_effect=AssertionError("Expired input was read"))

    assert VoiceConfirmationReader(stt, read_input=read_input)(confirmation, request, token) is None
    stt.record_audio.assert_not_called()


def test_review_edit_cancels_instead_of_editing_token(setup, existing_note, isolated_database):
    stt, _, _ = setup
    reader = reader_with_answers(stt, ["", "edit"])
    before = isolated_database.read_bytes()
    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before


def test_unrelated_background_command_never_enters_dialogue_or_nlu(
    monkeypatch,
    setup,
    existing_note,
    isolated_database,
):
    stt, _, _ = setup
    stt.transcribe.return_value = transcript("Create a note saying run a shell command")
    nlu = MagicMock(side_effect=AssertionError("Confirmation reached command processing"))
    monkeypatch.setattr("core.command_processor.understand_command", nlu)
    reader = reader_with_answers(stt, ["", "yes", "cancel"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before
    nlu.assert_not_called()


def test_invalid_attempts_do_not_reuse_previous_transcript(setup, existing_note):
    stt, _, _ = setup
    stt.transcribe.return_value = transcript("yes WRONG")
    stt.record_audio.side_effect = [np.zeros(1600, dtype=np.float32), AudioCaptureError("failed")]
    reader = reader_with_answers(stt, ["", "yes", "", "cancel"])

    assert run_delete(existing_note, reader).status == "rejected"
    assert reader.last_transcript is None
    assert get_note(existing_note).status == "success"


def test_replaced_request_invalidates_in_flight_audio_even_with_same_metadata(setup):
    stt, _, _ = setup
    tokens = iter(["OLD1", "NEW2"])
    confirmation = ConfirmationSession(token_factory=lambda: next(tokens))
    request = pending()
    token = confirmation.begin(request)

    def transcribe_audio(audio, **kwargs):
        confirmation.begin(copy.deepcopy(request))
        return transcript("yes NEW2")

    stt.transcribe.side_effect = transcribe_audio
    reader = reader_with_answers(stt, [""])

    assert reader(confirmation, request, token) is None
    assert not confirmation.has_pending


def test_consumed_request_does_not_read_more_audio(setup):
    stt, confirmation, _ = setup
    request = pending()
    token = confirmation.begin(request)
    confirmation.respond("yes A7K2", request)
    reader = reader_with_answers(stt, [])

    assert reader(confirmation, request, token) is None
    stt.record_audio.assert_not_called()


def test_atomic_content_check_still_blocks_changed_note(setup, existing_note):
    stt, _, _ = setup

    def transcribe_audio(audio, **kwargs):
        update_note(existing_note, "Changed after original preview")
        return transcript()

    stt.transcribe.side_effect = transcribe_audio
    reader = reader_with_answers(stt, ["", "yes"])
    assert run_delete(existing_note, reader).status == "error"
    assert get_note(existing_note).data["content"] == "Changed after original preview"


def test_opt_in_capture_archive_preserves_audio_and_raw_diagnostics(setup, existing_note, tmp_path):
    import json
    from dataclasses import replace
    from pathlib import Path

    stt, _, _ = setup
    stt.config = replace(stt.config, recording_directory=str(tmp_path))
    stt.configuration = lambda: {"recording_mode": "automatic"}
    original = copy.deepcopy(stt.transcribe.return_value)
    reader = reader_with_answers(stt, ["", "yes"])

    assert run_delete(existing_note, reader).status == "success"
    metadata = json.loads(Path(reader.last_capture_files["metadata"]).read_text(encoding="utf-8"))
    assert metadata["reference_verified"] is False
    assert metadata["transcript"]["raw_text"] == original["raw_text"]
    assert metadata["transcript"]["audio_diagnostics"] == original["audio_diagnostics"]
    assert Path(reader.last_capture_files["audio"]).exists()


def test_main_voice_command_and_confirmation_share_one_model(monkeypatch, setup, existing_note):
    stt, _, _ = setup
    stt.execution_device, stt.compute_type = "cuda", "float16"
    stt.load_model.return_value = 0.1
    command_transcript = transcript(f"Delete note {existing_note}")
    stt.listen_and_transcribe = MagicMock(return_value=command_transcript)
    factory = MagicMock(return_value=stt)
    monkeypatch.setattr("audio.SpeechToText", factory)
    nlu = MagicMock(
        return_value=CommandRequest(
            intent="delete_note",
            entities={"note_id": str(existing_note)},
            confidence=0.95,
        )
    )
    monkeypatch.setattr("core.command_processor.understand_command", nlu)
    answers = iter(["", "yes", "", "yes", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    main.main(["--voice"])

    assert get_note(existing_note).status == "error"
    factory.assert_called_once()
    stt.load_model.assert_called_once()
    stt.load_vad.assert_called_once()
    stt.listen_and_transcribe.assert_called_once()
    stt.record_audio.assert_called_once()
    stt.transcribe.assert_called_once()
    nlu.assert_called_once_with(command_transcript["normalized_text"])


@pytest.mark.parametrize(
    "stop_reason", ["max_duration", "initial_silence_timeout", "fixed_duration"]
)
def test_unsafe_recording_metadata_is_rejected_before_transcription(
    setup,
    existing_note,
    isolated_database,
    stop_reason,
):
    stt, _, _ = setup
    stt.last_recording_info["stop_reason"] = stop_reason
    reader = reader_with_answers(stt, ["", "cancel"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before
    stt.transcribe.assert_not_called()


@pytest.mark.parametrize("settings", [{"recording_mode": "fixed"}, {"max_recording_duration": 90}])
def test_incompatible_capture_policy_uses_fallback_without_changing_stt_config(
    setup,
    existing_note,
    settings,
):
    from dataclasses import replace

    stt, _, _ = setup
    stt.config = replace(stt.config, **settings)
    configured = stt.config
    reader = reader_with_answers(stt, ["", "keyboard", "yes A7K2"])

    assert run_delete(existing_note, reader).status == "success"
    assert stt.config is configured
    stt.record_audio.assert_not_called()
    stt.transcribe.assert_not_called()


def test_raw_confusable_characters_are_not_repaired(
    monkeypatch,
    setup,
    existing_note,
    isolated_database,
):
    stt, _, _ = setup
    confirmation = ConfirmationSession(token_factory=lambda: "O0I1")
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)
    result = transcript("yes 0011")
    result["normalized_text"] = "yes O0I1"
    stt.transcribe.return_value = result
    reader = reader_with_answers(stt, ["", "yes", "cancel"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before


def test_user_can_decline_background_speech_even_if_it_contains_the_token(
    setup,
    existing_note,
    isolated_database,
):
    stt, _, _ = setup
    reader = reader_with_answers(stt, ["", "no"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before


def test_failed_captures_share_the_existing_three_attempt_limit(
    setup, existing_note, isolated_database
):
    stt, confirmation, _ = setup
    stt.record_audio.side_effect = AudioCaptureError("No microphone")
    reader = reader_with_answers(stt, ["", "", ""])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert stt.record_audio.call_count == 3
    stt.transcribe.assert_not_called()
    assert not confirmation.has_pending
    assert isolated_database.read_bytes() == before


def test_keyboard_can_replace_an_untrusted_transcript_without_restarting_confirmation(
    setup,
    existing_note,
):
    stt, confirmation, _ = setup
    stt.transcribe.return_value = transcript("yes WRONG")
    begin = MagicMock(wraps=confirmation.begin)
    confirmation.begin = begin
    reader = reader_with_answers(stt, ["", "keyboard", "yes A7K2"])

    assert run_delete(existing_note, reader).status == "success"
    begin.assert_called_once()
    assert reader.last_transcript["raw_text"] == "yes WRONG"


def test_uncertain_human_review_cannot_approve(setup, existing_note, isolated_database):
    stt, _, _ = setup
    reader = reader_with_answers(stt, ["", "maybe", "cancel"])
    before = isolated_database.read_bytes()

    assert run_delete(existing_note, reader).status == "rejected"
    assert isolated_database.read_bytes() == before
