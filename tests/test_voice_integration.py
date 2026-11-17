import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import main
from actions.notes import get_note
from audio import AudioConfig
from core.schemas import CommandRequest


@pytest.fixture
def voice(monkeypatch):
    transcript = {
        "empty_transcription": False,
        "raw_text": "Delete note one",
        "normalized_text": "Delete note one",
        "warnings": [],
        "audio_diagnostics": {"possible_truncation": False},
    }
    engine = SimpleNamespace(
        config=AudioConfig(),
        execution_device="cuda",
        compute_type="float16",
        listen_and_transcribe=MagicMock(return_value=transcript),
        load_model=MagicMock(return_value=0.1),
        load_vad=MagicMock(return_value=0.1),
    )
    monkeypatch.setattr("audio.SpeechToText", lambda config: engine)
    return transcript


def inputs(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))


@pytest.mark.parametrize("approval,deleted", [("yes", True), ("no", False)])
def test_voice_deletion_uses_existing_confirmation(
    monkeypatch, voice, existing_note, approval, deleted
):
    nlu = MagicMock(
        return_value=CommandRequest(
            intent="delete_note",
            entities={"note_id": str(existing_note)},
            confidence=0.99,
        )
    )
    monkeypatch.setattr("core.command_processor.understand_command", nlu)
    inputs(monkeypatch, ["", "yes", approval, "exit"])
    main.main(["--voice"])
    nlu.assert_called_once_with(voice["normalized_text"])
    assert (get_note(existing_note).status == "error") is deleted


@pytest.mark.parametrize(
    "command,status",
    [
        (CommandRequest(intent="unknown", confidence=0.9), "rejected"),
        (CommandRequest(intent="get_notes", confidence=0.2), "needs_clarification"),
        (CommandRequest(intent="create_note", confidence=0.9), "needs_clarification"),
    ],
)
def test_voice_preserves_rejection_and_clarification(monkeypatch, voice, command, status, capsys):
    monkeypatch.setattr("core.command_processor.understand_command", lambda text: command)
    inputs(monkeypatch, ["", "yes", "exit"])
    main.main(["--voice"])
    assert f"[{status}]" in capsys.readouterr().out


@pytest.mark.parametrize("reason", ["empty", "truncated", "declined", "nonlexical", "max_duration"])
def test_unusable_or_unapproved_speech_never_reaches_nlu(monkeypatch, voice, reason):
    if reason == "empty":
        voice["empty_transcription"] = True
    if reason == "truncated":
        voice["audio_diagnostics"]["possible_truncation"] = True
    if reason == "nonlexical":
        voice["has_lexical_content"] = False
    if reason == "max_duration":
        voice["recording_info"] = {"stop_reason": "max_duration"}
    nlu = MagicMock()
    monkeypatch.setattr("core.command_processor.understand_command", nlu)
    inputs(monkeypatch, ["", "no", "exit"] if reason == "declined" else ["", "exit"])
    main.main(["--voice"])
    nlu.assert_not_called()


def test_voice_edit_and_normalized_preview(monkeypatch, voice, capsys):
    voice["normalized_text"] = "Delete note 1"
    nlu = MagicMock(return_value=CommandRequest(intent="unknown", confidence=0.9))
    monkeypatch.setattr("core.command_processor.understand_command", nlu)
    inputs(monkeypatch, ["", "edit", "Show my notes", "exit"])
    main.main(["--voice"])
    nlu.assert_called_once_with("Show my notes")
    assert "Normalized: Delete note 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "option,value,expected",
    [
        ("--language", "auto", None),
        ("--language", "en", "en"),
        ("--preferred-language", "hi", "hi"),
    ],
)
def test_cli_language_overrides_env(monkeypatch, voice, option, value, expected):
    monkeypatch.setenv("VOICEOS_STT_LANGUAGE", "hi" if expected != "hi" else "en")
    factory = MagicMock()
    factory.return_value.load_model.return_value = 0.1
    monkeypatch.setattr("audio.SpeechToText", factory)
    inputs(monkeypatch, ["exit"])
    main.main(["--voice", option, value])
    assert factory.call_args.kwargs["config"].language == expected


@pytest.mark.parametrize(
    "raw",
    [
        "powershell Remove-Item everything",
        "DROP TABLE notes;",
        "execute arbitrary Python code",
    ],
)
def test_unrestricted_actions_are_rejected_via_existing_nlu(monkeypatch, voice, raw, capsys):
    voice["raw_text"] = voice["normalized_text"] = raw
    nlu = MagicMock(return_value=CommandRequest(intent="unknown", confidence=0.99))
    monkeypatch.setattr("core.command_processor.understand_command", nlu)
    inputs(monkeypatch, ["", "yes", "exit"])
    main.main(["--voice", "--language", "en"])
    nlu.assert_called_once_with(raw)
    assert "[rejected]" in capsys.readouterr().out


def test_voice_startup_loads_once_before_multiple_captures(monkeypatch, voice):
    events = []
    engine = SimpleNamespace(
        config=AudioConfig(),
        execution_device="cuda",
        compute_type="float16",
        load_model=MagicMock(side_effect=lambda: events.append("load") or 0.1),
        load_vad=MagicMock(return_value=0.1),
        listen_and_transcribe=MagicMock(
            side_effect=lambda **kwargs: events.append("capture") or voice
        ),
    )
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr("audio.SpeechToText", factory)
    monkeypatch.setattr(
        "core.command_processor.understand_command",
        lambda text: CommandRequest(intent="unknown", confidence=0.99),
    )
    inputs(monkeypatch, ["", "yes", "", "yes", "exit"])
    main.main(["--voice"])
    factory.assert_called_once()
    engine.load_model.assert_called_once()
    engine.load_vad.assert_called_once()
    assert events == ["load", "capture", "capture"]


def test_voice_cli_optional_decoding_overrides(monkeypatch, voice):
    factory = MagicMock()
    factory.return_value.load_model.return_value = 0.1
    monkeypatch.setattr("audio.SpeechToText", factory)
    inputs(monkeypatch, ["exit"])
    main.main(["--voice", "--multilingual", "--prompt="])
    config = factory.call_args.kwargs["config"]
    assert config.multilingual is True
    assert config.initial_prompt == ""


def test_vad_silence_endpoint_can_complete_with_background_rms(monkeypatch, voice):
    voice["audio_diagnostics"]["possible_truncation"] = True
    voice["recording_info"] = {"stop_reason": "silence"}
    nlu = MagicMock(return_value=CommandRequest(intent="unknown", confidence=0.99))
    monkeypatch.setattr("core.command_processor.understand_command", nlu)
    inputs(monkeypatch, ["", "yes", "exit"])
    main.main(["--voice"])
    nlu.assert_called_once()


def test_automatic_endpoint_cli_overrides(monkeypatch, voice):
    factory = MagicMock()
    factory.return_value.load_model.return_value = 0.1
    monkeypatch.setattr("audio.SpeechToText", factory)
    inputs(monkeypatch, ["exit"])
    main.main(["--voice", "--duration", "25", "--silence-seconds", "2.5", "--initial-timeout", "5"])
    config = factory.call_args.kwargs["config"]
    assert config.recording_mode == "automatic"
    assert config.max_recording_duration == 25
    assert config.silence_duration == 2.5
    assert config.initial_silence_timeout == 5


def test_recording_archive_cli_overrides_environment(monkeypatch, voice, tmp_path):
    monkeypatch.setenv("VOICEOS_RECORDING_DIRECTORY", "previous-folder")
    factory = MagicMock()
    factory.return_value.load_model.return_value = 0.1
    monkeypatch.setattr("audio.SpeechToText", factory)
    inputs(monkeypatch, ["exit"])
    main.main(["--voice", "--save-recordings", str(tmp_path)])
    assert factory.call_args.kwargs["config"].recording_directory == str(tmp_path)
    factory.return_value.listen_and_transcribe.assert_not_called()


def test_speech_only_mode_never_calls_nlu_or_database(monkeypatch, voice, capsys):
    voice["raw_text"] = voice["normalized_text"] = "Restart the computer"
    voice["review_flags"] = [{"kind": "number", "text": "27", "message": "Verify number."}]
    processor = MagicMock(side_effect=AssertionError("No command submission in speech-only mode"))
    database = MagicMock()
    monkeypatch.setattr(main, "process_text_command", processor)
    monkeypatch.setattr(main, "initialize_database", database)
    inputs(monkeypatch, ["", "yes", "exit"])
    main.main(["--voice", "--speech-only"])
    processor.assert_not_called()
    database.assert_not_called()
    output = capsys.readouterr().out
    assert "Review number: 27" in output
    assert "Application actions are disabled in speech-only mode. [rejected]" in output


def test_speech_only_edit_uses_existing_review_without_action_submission(
    monkeypatch, voice, capsys
):
    processor = MagicMock()
    monkeypatch.setattr(main, "process_text_command", processor)
    inputs(monkeypatch, ["", "edit", "Corrected speech", "exit"])
    main.main(["--voice", "--speech-only"])
    processor.assert_not_called()
    assert "[rejected]" in capsys.readouterr().out


def test_speech_only_requires_voice():
    with pytest.raises(SystemExit) as error:
        main.main(["--speech-only"])
    assert error.value.code == 2


def test_redirected_voice_output_uses_utf8(monkeypatch):
    import io
    import sys

    raw = io.BytesIO()
    output = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", output)
    main._configure_voice_output()
    print("हिंदी", flush=True)
    assert raw.getvalue().decode("utf-8").strip() == "हिंदी"


@pytest.mark.parametrize("signal", [KeyboardInterrupt, EOFError])
def test_speech_only_shutdown_without_actions(monkeypatch, voice, signal, capsys):
    processor = MagicMock()
    monkeypatch.setattr(main, "process_text_command", processor)
    monkeypatch.setattr("builtins.input", MagicMock(side_effect=signal))
    main.main(["--voice", "--speech-only"])
    processor.assert_not_called()
    assert "VoiceOS shutting down." in capsys.readouterr().out


def test_microphone_error_allows_retry_without_model_reload(monkeypatch, voice, capsys):
    from audio.speech_to_text import AudioCaptureError

    engine = SimpleNamespace(
        config=AudioConfig(),
        execution_device="cpu",
        compute_type="float32",
        load_model=MagicMock(return_value=0.1),
        load_vad=MagicMock(return_value=0.1),
        listen_and_transcribe=MagicMock(
            side_effect=[AudioCaptureError("Microphone unavailable"), voice]
        ),
    )
    monkeypatch.setattr("audio.SpeechToText", lambda config: engine)
    processor = MagicMock()
    monkeypatch.setattr(main, "process_text_command", processor)
    inputs(monkeypatch, ["", "", "no", "exit"])
    main.main(["--voice", "--speech-only"])
    engine.load_model.assert_called_once()
    engine.load_vad.assert_called_once()
    assert engine.listen_and_transcribe.call_count == 2
    processor.assert_not_called()
    assert "Microphone unavailable" in capsys.readouterr().out


@pytest.mark.parametrize(
    "text,content,language",
    [
        ("Create a note saying buy milk", "buy milk", "english"),
        ("मेरे लिए एक नोट बनाओ कि कल DBMS पढ़ना है", "कल DBMS पढ़ना है", "hindi"),
        ("Mere liye ek note banao ki kal DBMS padhna hai", "kal DBMS padhna hai", "hinglish"),
    ],
    ids=["english", "hindi", "hinglish"],
)
def test_reviewed_multilingual_voice_uses_real_command_pipeline(
    monkeypatch, voice, text, content, language, capsys,
):
    """STT and Gemini are mocked; parsing, validation, routing and storage are real."""
    voice["raw_text"] = voice["normalized_text"] = text
    generate = MagicMock(return_value=json.dumps({
        "intent": "create_note", "entities": {"content": content}, "language": language,
        "confidence": 0.95, "missing_fields": [],
    }, ensure_ascii=False))
    monkeypatch.setattr("nlu.service.generate_response", generate)
    inputs(monkeypatch, ["", "yes", "exit"])

    main.main(["--voice"])

    generate.assert_called_once()
    assert generate.call_args.args[0].split("User command:\n", 1)[1].strip() == text
    assert get_note(1).data["content"] == content
    assert voice["raw_text"] == text
    assert voice["normalized_text"] == text
    output = capsys.readouterr().out
    assert f"Heard: {text}" in output
    assert "Your note has been saved. [success]" in output


def test_voice_clarification_uses_reviewed_followup_without_model_reload(
    monkeypatch, voice, capsys,
):
    command_text = "Mere liye ek note banao"
    content = "kal VoiceOS mein Python aur DBMS पढ़ना है"
    turns = [{**voice, "raw_text": text, "normalized_text": text}
             for text in (command_text, content)]
    engine = SimpleNamespace(
        config=AudioConfig(), execution_device="cuda", compute_type="float16",
        load_model=MagicMock(return_value=0.1), load_vad=MagicMock(return_value=0.1),
        listen_and_transcribe=MagicMock(side_effect=turns),
    )
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr("audio.SpeechToText", factory)
    generate = MagicMock(return_value=json.dumps({
        "intent": "create_note", "entities": {}, "language": "hinglish",
        "confidence": 0.95, "missing_fields": ["content"],
    }))
    monkeypatch.setattr("nlu.service.generate_response", generate)
    inputs(monkeypatch, ["", "yes", "", "yes", "exit"])

    main.main(["--voice"])

    assert get_note(1).data["content"] == content
    generate.assert_called_once()
    assert generate.call_args.args[0].split("User command:\n", 1)[1].strip() == command_text
    factory.assert_called_once()
    engine.load_model.assert_called_once()
    engine.load_vad.assert_called_once()
    assert engine.listen_and_transcribe.call_count == 2
    assert "[needs_clarification]" in capsys.readouterr().out
