"""Deterministic timing checks with offline STT/NLU and real validation/storage."""

import copy
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors

import main
from actions.notes import get_note
from audio import AudioConfig
from audio.speech_to_text import AudioCaptureError
from core import command_processor
from core.confirmation import ConfirmationSession
from core.dialogue import DialogueSession
from core.schemas import CommandRequest
from core.timing import InteractionTiming, current_timing
from core.voice_confirmation import VoiceConfirmationReader
from nlu import llm_client


class Clock:
    def __init__(self):
        self.value = 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value

    def advance(self, seconds):
        self.value += seconds


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
        "recording_duration": 2.0,
        "recording_time": 0.0,
        "transcription_duration": 2.5,
        "normalization_duration": 0.5,
        "total_duration": 3.0,
        "segments": [{"text": raw, "avg_logprob": -0.2}],
    }


@pytest.fixture
def confirmation_setup(monkeypatch):
    measurement, authorization = Clock(), Clock()
    session = ConfirmationSession(clock=authorization, token_factory=lambda: "A7K2")
    monkeypatch.setattr(main, "ConfirmationSession", lambda: session)
    original = transcript()

    def record(**kwargs):
        measurement.advance(2)
        return [0.1]

    def decode(audio, **kwargs):
        measurement.advance(3)
        return original

    stt = SimpleNamespace(
        config=AudioConfig(),
        last_recording_info=original["recording_info"],
        last_capture_files=None,
        record_audio=MagicMock(side_effect=record),
        transcribe=MagicMock(side_effect=decode),
    )
    return measurement, authorization, session, stt, original


def pending(note_id):
    return main.ActionResult(
        status="needs_confirmation",
        message="Delete test note",
        data={"note_id": note_id, "preview": "Test note", "original_content": "Test note"},
    )


def test_voice_command_capture_decode_and_edit_review(monkeypatch):
    clock = Clock()
    timing = InteractionTiming(clock=clock)

    def listen(**kwargs):
        clock.advance(6.5)
        result = transcript("कल DBMS पढ़ना है")
        result.update(recording_time=4.0, transcription_duration=2.0, normalization_duration=0.5)
        return result

    answers = iter([(10, ""), (7, "edit"), (3, "कल Python और DBMS पढ़ना है")])

    def answer(prompt):
        seconds, text = next(answers)
        clock.advance(seconds)
        return text

    monkeypatch.setattr("builtins.input", answer)
    stt = SimpleNamespace(config=AudioConfig(), listen_and_transcribe=listen)
    assert main._voice_text(stt, timing=timing) == "कल Python और DBMS पढ़ना है"
    timing.finish()
    result = timing.snapshot()
    assert result["command_capture_duration"] == 4
    assert result["transcription_duration"] == 2
    assert result["normalization_duration"] == 0.5
    assert result["transcript_review_duration"] == 10
    assert result["total_interaction_duration"] == 16.5
    assert result["nlu_duration"] == result["action_execution_duration"] == 0


def test_nlu_timing_includes_retry_backoff_and_parsing(monkeypatch):
    clock = Clock()
    timing = InteractionTiming(clock=clock)
    client = MagicMock()
    client.__enter__.return_value = client
    responses = iter(
        [
            errors.APIError(429, {"message": "Synthetic retry"}),
            SimpleNamespace(
                text=json.dumps(
                    {
                        "intent": "create_note",
                        "entities": {"content": "DBMS"},
                        "confidence": 0.99,
                    }
                )
            ),
        ]
    )

    def generate(**kwargs):
        clock.advance(0.25)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(llm_client, "get_gemini_client", lambda: client)
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: None)
    monkeypatch.setattr(llm_client.time, "sleep", clock.advance)
    monkeypatch.setenv("GEMINI_MODEL", "offline-model")
    monkeypatch.setenv("GEMINI_MAX_ATTEMPTS", "3")
    client.models.generate_content.side_effect = generate
    import nlu.service

    parse = nlu.service.parse_nlu_response

    def parse_response(text):
        clock.advance(0.5)
        return parse(text)

    monkeypatch.setattr(nlu.service, "parse_nlu_response", parse_response)
    result = command_processor.process_text_command("Create a note", timing=timing)
    assert result.status == "success"
    assert timing.snapshot()["nlu_duration"] == 2
    assert client.models.generate_content.call_count == 2


def test_clarification_waits_share_id_and_do_not_repeat_nlu(monkeypatch, existing_note):
    clock = Clock()

    def understand(text):
        clock.advance(2)
        return CommandRequest(intent="update_note", confidence=0.99)

    nlu = MagicMock(side_effect=understand)
    monkeypatch.setattr(command_processor, "understand_command", nlu)
    import core.action_router

    update = core.action_router.update_note

    def save(note_id, content):
        clock.advance(1)
        return update(note_id, content)

    monkeypatch.setattr(core.action_router, "update_note", save)
    dialogue = DialogueSession(measurement_clock=clock)
    assert dialogue.process_text_command("Update a note").status == "needs_clarification"
    interaction_id = dialogue.timing.snapshot()["interaction_id"]
    clock.advance(10)
    assert dialogue.process_text_command(str(existing_note)).status == "needs_clarification"
    clock.advance(5)
    assert dialogue.process_text_command("कल DBMS पढ़ना hai").status == "success"
    record = dialogue.last_timing
    assert record["interaction_id"] == interaction_id
    assert record["clarification_wait_duration"] == 15
    assert record["nlu_duration"] == 2
    assert record["action_execution_duration"] == 1
    assert record["total_interaction_duration"] == 18
    nlu.assert_called_once()
    dialogue.process_text_command("Update another note")
    assert dialogue.timing.snapshot()["interaction_id"] != interaction_id


@pytest.mark.parametrize("reply", ["/cancel", "/new Show notes", "yes"])
def test_pending_timing_cancellation_replacement_and_invalid_reply(monkeypatch, reply):
    clock = Clock()
    nlu = MagicMock(
        side_effect=[
            CommandRequest(intent="create_note", confidence=0.99),
            CommandRequest(intent="get_notes", confidence=0.99),
        ]
    )
    monkeypatch.setattr(command_processor, "understand_command", nlu)
    dialogue = DialogueSession(measurement_clock=clock)
    dialogue.process_text_command("Create a note")
    old = dialogue.timing
    old_id = old.snapshot()["interaction_id"]
    clock.advance(4)
    result = dialogue.process_text_command(reply)
    if reply.startswith("/new"):
        assert result.status == "success"
        assert dialogue.last_timing["interaction_id"] != old_id
        assert old.finished
    elif reply == "/cancel":
        assert result.status == "rejected"
        assert dialogue.last_timing["clarification_wait_duration"] == 4
    else:
        assert result.status == "needs_clarification"
        assert dialogue.timing.snapshot()["interaction_id"] == old_id
    assert nlu.call_count == (2 if reply.startswith("/new") else 1)


def test_confirmation_capture_transcribe_review_and_raw_diagnostics(
    confirmation_setup, existing_note
):
    measurement, authorization, _, stt, original = confirmation_setup
    before = copy.deepcopy(original)
    answers = iter([(4, ""), (5, "yes")])

    def answer(prompt):
        seconds, value = next(answers)
        measurement.advance(seconds)
        return value

    reader = VoiceConfirmationReader(stt, read_input=answer, measurement_clock=measurement)
    timing = InteractionTiming(clock=measurement)
    assert main._confirm_deletion(pending(existing_note), reader, timing=timing) is True
    record = timing.snapshot()
    assert record["confirmation_capture_duration"] == 2
    assert record["confirmation_transcription_duration"] == 3
    assert record["confirmation_review_duration"] == 5
    assert record["confirmation_total_duration"] == 14
    assert record["confirmation_attempt_count"] == 1
    assert record["confirmation_outcome"] == "approved"
    assert record["keyboard_fallback_duration"] == 0
    assert reader.last_transcript["recording_time"] == 2
    assert reader.last_transcript["total_duration"] == 5
    assert reader.last_transcript["segments"] == before["segments"]
    assert reader.last_transcript["transcription_duration"] == before["transcription_duration"]
    assert original == before
    assert authorization.calls == 12


def test_confirmation_failures_and_keyboard_fallback_share_attempts(
    confirmation_setup, existing_note
):
    clock, _, _, stt, _ = confirmation_setup

    def fail(**kwargs):
        clock.advance(2)
        raise AudioCaptureError("Synthetic failure")

    stt.record_audio.side_effect = fail
    answers = iter(["", "", "keyboard", "yes A7K2"])

    def answer(prompt):
        value = next(answers)
        if value.startswith("yes"):
            clock.advance(3)
        return value

    reader = VoiceConfirmationReader(stt, read_input=answer, measurement_clock=clock)
    timing = InteractionTiming(clock=clock)
    assert main._confirm_deletion(pending(existing_note), reader, timing=timing) is True
    record = timing.snapshot()
    assert record["confirmation_capture_duration"] == 4
    assert record["confirmation_transcription_duration"] == 0
    assert record["keyboard_fallback_duration"] == 3
    assert record["confirmation_total_duration"] == 7
    assert record["confirmation_attempt_count"] == 3
    assert reader.last_transcript is None
    stt.transcribe.assert_not_called()


@pytest.mark.parametrize(
    "reply,outcome", [("cancel", "cancelled"), ("no", "rejected"), ("edit", "cancelled")]
)
def test_confirmation_skipped_phases(confirmation_setup, existing_note, reply, outcome):
    clock, _, _, stt, _ = confirmation_setup
    reader = VoiceConfirmationReader(stt, read_input=lambda prompt: reply, measurement_clock=clock)
    timing = InteractionTiming(clock=clock)
    assert main._confirm_deletion(pending(existing_note), reader, timing=timing) is False
    record = timing.snapshot()
    assert record["confirmation_outcome"] == outcome
    for key in (
        "confirmation_capture_duration",
        "confirmation_transcription_duration",
        "confirmation_review_duration",
        "keyboard_fallback_duration",
    ):
        assert record[key] == 0
    stt.record_audio.assert_not_called()


@pytest.mark.parametrize("stage", ["capture", "transcription", "review"])
def test_expiry_with_timing_never_deletes(
    confirmation_setup, existing_note, isolated_database, stage
):
    measurement, authorization, _, stt, original = confirmation_setup
    record_audio, decode = stt.record_audio.side_effect, stt.transcribe.side_effect

    def record(**kwargs):
        result = record_audio(**kwargs)
        if stage == "capture":
            authorization.value = 60
        return result

    def transcribe_audio(audio, **kwargs):
        result = decode(audio, **kwargs)
        if stage == "transcription":
            authorization.value = 60
        return result

    answers = iter(["", "yes"])

    def answer(prompt):
        reply = next(answers)
        if stage == "review" and reply == "yes":
            authorization.value = 60
        return reply

    stt.record_audio.side_effect, stt.transcribe.side_effect = record, transcribe_audio
    reader = VoiceConfirmationReader(stt, read_input=answer, measurement_clock=measurement)
    timing = InteractionTiming(clock=measurement)
    before = isolated_database.read_bytes()
    command = CommandRequest(
        intent="delete_note", entities={"note_id": str(existing_note)}, confidence=0.99
    )
    # Route the supplied request directly; no model call is needed for this safety check.
    from core.action_router import route_command

    result = route_command(
        command,
        lambda request: main._confirm_deletion(
            request,
            reader,
            timing=timing,
        ),
    )
    assert result.status == "rejected"
    assert timing.snapshot()["confirmation_outcome"] == "expired"
    assert isolated_database.read_bytes() == before
    assert original["recording_time"] == 0


def test_measurement_clock_cannot_extend_authorization(confirmation_setup, existing_note):
    measurement, authorization, _, stt, _ = confirmation_setup

    def answer(prompt):
        measurement.advance(1000000)
        authorization.value = 59.999
        return "" if "Press Enter" in prompt else "yes"

    reader = VoiceConfirmationReader(stt, read_input=answer, measurement_clock=measurement)
    assert main._confirm_deletion(pending(existing_note), reader) is True
    assert authorization.value == 59.999
    assert reader.timing.snapshot()["confirmation_total_duration"] > 1000000


def test_action_timing_excludes_confirmation_wait(monkeypatch, confirmation_setup, existing_note):
    clock, _, _, stt, _ = confirmation_setup
    monkeypatch.setattr(
        command_processor,
        "understand_command",
        lambda text: CommandRequest(
            intent="delete_note",
            entities={"note_id": str(existing_note)},
            confidence=0.99,
        ),
    )
    reader = VoiceConfirmationReader(stt, read_input=lambda prompt: "", measurement_clock=clock)
    replies = iter(["", "yes"])
    reader._input = lambda prompt: next(replies)
    timing = InteractionTiming(clock=clock)
    result = command_processor.process_text_command(
        "Delete note",
        lambda request: main._confirm_deletion(request, reader, timing=timing),
        timing=timing,
    )
    assert result.status == "success"
    assert timing.snapshot()["confirmation_total_duration"] == 5
    assert timing.snapshot()["action_execution_duration"] == 0
    assert get_note(existing_note).status == "error"


def test_main_publishes_complete_record_after_result(monkeypatch):
    clock, records = Clock(), []

    def understand(text):
        clock.advance(2)
        return CommandRequest(intent="get_notes", confidence=0.99)

    nlu = MagicMock(side_effect=understand)
    monkeypatch.setattr(command_processor, "understand_command", nlu)
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    answers = iter(["Show notes", "exit"])

    def answer(prompt):
        clock.advance(10)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    main.main(measurement_clock=clock, timing_records=records)
    assert len(records) == 1
    assert records[0]["nlu_duration"] == 2
    assert records[0]["total_interaction_duration"] == 2
    assert records[0]["command_capture_duration"] == 0
    assert records[0]["confirmation_outcome"] is None
    nlu.assert_called_once()


def test_unavailable_measurement_clock_does_not_change_results(monkeypatch):
    def broken_clock():
        raise RuntimeError("Synthetic measurement failure")

    timing = InteractionTiming(clock=broken_clock)
    monkeypatch.setattr(
        command_processor,
        "understand_command",
        lambda text: CommandRequest(
            intent="get_notes",
            confidence=0.99,
        ),
    )
    assert command_processor.process_text_command("Show notes", timing=timing).status == "success"
    assert timing.snapshot()["nlu_duration"] is None


def test_main_voice_pipeline_records_command_and_confirmation(
    monkeypatch,
    confirmation_setup,
    existing_note,
    caplog,
):
    clock, _, _, stt, _ = confirmation_setup
    records = []

    def listen(**kwargs):
        clock.advance(3.25)
        result = transcript("Delete this note")
        result.update(recording_time=2, transcription_duration=1, normalization_duration=0.25)
        return result

    stt.listen_and_transcribe = MagicMock(side_effect=listen)
    stt.load_model = MagicMock(return_value=0)
    stt.load_vad = MagicMock(return_value=0)
    stt.execution_device, stt.compute_type = "cpu", "float32"
    monkeypatch.setattr("audio.SpeechToText", lambda config: stt)
    monkeypatch.setattr(main, "load_dotenv", lambda: None)

    def understand(text):
        clock.advance(2)
        return CommandRequest(
            intent="delete_note",
            entities={"note_id": str(existing_note)},
            confidence=0.99,
        )

    nlu = MagicMock(side_effect=understand)
    monkeypatch.setattr(command_processor, "understand_command", nlu)
    import core.action_router

    delete = core.action_router.delete_note

    def remove(note_id, **kwargs):
        clock.advance(0.5)
        return delete(note_id, **kwargs)

    monkeypatch.setattr(core.action_router, "delete_note", remove)
    answers = iter([(100, ""), (4, "yes"), (5, ""), (6, "yes"), (100, "exit")])

    def answer(prompt):
        seconds, value = next(answers)
        clock.advance(seconds)
        return value

    monkeypatch.setattr("builtins.input", answer)
    with caplog.at_level(logging.DEBUG, logger="main"):
        main.main(["--voice"], measurement_clock=clock, timing_records=records)
    assert len(records) == 1
    record = records[0]
    expected = {
        "command_capture_duration": 2,
        "transcription_duration": 1,
        "normalization_duration": 0.25,
        "transcript_review_duration": 4,
        "nlu_duration": 2,
        "clarification_wait_duration": 0,
        "action_execution_duration": 0.5,
        "total_interaction_duration": 25.75,
        "confirmation_capture_duration": 2,
        "confirmation_transcription_duration": 3,
        "confirmation_review_duration": 6,
        "keyboard_fallback_duration": 0,
        "confirmation_total_duration": 16,
        "confirmation_attempt_count": 1,
        "confirmation_outcome": "approved",
    }
    assert {key: record[key] for key in expected} == expected
    assert len(record["interaction_id"]) == 32
    assert get_note(existing_note).status == "error"
    nlu.assert_called_once_with("Delete this note")
    assert current_timing.get() is None
    logged = [entry for entry in caplog.records if entry.name == "main"]
    assert len(logged) == 1
    assert json.loads(logged[0].getMessage().removeprefix("Interaction timing: ")) == record
    assert "A7K2" not in logged[0].getMessage()


@pytest.mark.parametrize("mode", ["silence", "declined", "capture-failure"])
def test_command_unusable_and_failed_phases(monkeypatch, mode):
    clock = Clock()
    timing = InteractionTiming(clock=clock)

    def listen(**kwargs):
        clock.advance(2)
        if mode == "capture-failure":
            raise AudioCaptureError("Synthetic capture failure")
        result = transcript()
        result.update(recording_time=2, transcription_duration=0, normalization_duration=0)
        if mode == "silence":
            result.update(empty_transcription=True, has_lexical_content=False)
        return result

    answers = iter(["", "no"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    stt = SimpleNamespace(config=AudioConfig(), listen_and_transcribe=listen)
    assert main._voice_text(stt, timing=timing) == ""
    record = timing.snapshot()
    if mode == "capture-failure":
        assert record["command_capture_duration"] is None
        assert record["transcription_duration"] is None
    else:
        assert record["command_capture_duration"] == 2
        assert record["transcription_duration"] == 0
    assert record["nlu_duration"] == record["action_execution_duration"] == 0


def test_nlu_failure_keeps_elapsed_time_and_failure_status(monkeypatch):
    clock = Clock()
    timing = InteractionTiming(clock=clock)

    def fail(text):
        clock.advance(3)
        raise RuntimeError("Synthetic failure")

    monkeypatch.setattr(command_processor, "understand_command", fail)
    result = command_processor.process_text_command("Show notes", timing=timing)
    assert result.status == "error"
    assert timing.snapshot()["nlu_duration"] == 3
    assert timing.snapshot()["action_execution_duration"] == 0
    assert current_timing.get() is None


def test_repeated_invalid_confirmation_attempts(confirmation_setup, existing_note, monkeypatch):
    clock, _, _, stt, original = confirmation_setup
    original["raw_text"] = original["normalized_text"] = "yes WRONG"
    answers = iter(["", "yes"] * 3)
    nlu = MagicMock(side_effect=AssertionError("Confirmation must not invoke NLU"))
    monkeypatch.setattr(command_processor, "understand_command", nlu)
    reader = VoiceConfirmationReader(
        stt,
        read_input=lambda prompt: next(answers),
        measurement_clock=clock,
    )
    assert main._confirm_deletion(pending(existing_note), reader) is False
    record = reader.timing.snapshot()
    assert record["confirmation_attempt_count"] == 3
    assert record["confirmation_outcome"] == "cancelled"
    assert record["confirmation_capture_duration"] == 6
    assert record["confirmation_transcription_duration"] == 9
    assert record["confirmation_total_duration"] == 15
    assert reader.last_transcript["recording_time"] == 2
    nlu.assert_not_called()


@pytest.mark.parametrize("approval", [1, "yes", False, None])
def test_timed_router_preserves_literal_true_requirement(monkeypatch, existing_note, approval):
    clock = Clock()
    timing = InteractionTiming(clock=clock)
    monkeypatch.setattr(
        command_processor,
        "understand_command",
        lambda text: CommandRequest(
            intent="delete_note",
            entities={"note_id": str(existing_note)},
            confidence=0.99,
        ),
    )

    def confirm(request):
        clock.advance(4)
        return approval

    result = command_processor.process_text_command("Delete note", confirm, timing=timing)
    assert result.status == "rejected"
    assert timing.snapshot()["action_execution_duration"] == 0
    assert timing.snapshot()["confirmation_total_duration"] == 4
    assert get_note(existing_note).status == "success"


def test_keyboard_confirmation_reuses_processor_timing(
    monkeypatch, confirmation_setup, existing_note
):
    clock, _, _, _, _ = confirmation_setup
    timing = InteractionTiming(clock=clock)
    monkeypatch.setattr(
        command_processor,
        "understand_command",
        lambda text: CommandRequest(
            intent="delete_note",
            entities={"note_id": str(existing_note)},
            confidence=0.99,
        ),
    )

    def answer(prompt):
        clock.advance(4)
        return "yes A7K2"

    monkeypatch.setattr("builtins.input", answer)
    result = command_processor.process_text_command(
        "Delete note",
        main.confirm_deletion,
        timing=timing,
    )
    assert result.status == "success"
    record = timing.snapshot()
    assert record["confirmation_outcome"] == "approved"
    assert record["confirmation_attempt_count"] == 1
    assert record["keyboard_fallback_duration"] == record["confirmation_total_duration"] == 4
    assert record["action_execution_duration"] == 0


def test_timing_context_is_nested_and_isolated():
    first, second = InteractionTiming(clock=Clock()), InteractionTiming(clock=Clock())
    with first.measure("nlu_duration"):
        assert current_timing.get() is first
        with second.measure("nlu_duration"):
            assert current_timing.get() is second
        assert current_timing.get() is first
    assert current_timing.get() is None
    assert first.snapshot()["interaction_id"] != second.snapshot()["interaction_id"]


def test_interrupted_confirmation_records_elapsed_review_and_cancels(
    confirmation_setup,
    existing_note,
):
    clock, _, session, stt, _ = confirmation_setup

    def answer(prompt):
        if "exactly" in prompt:
            clock.advance(4)
            raise EOFError
        return ""

    reader = VoiceConfirmationReader(stt, read_input=answer, measurement_clock=clock)
    with pytest.raises(EOFError):
        main._confirm_deletion(pending(existing_note), reader)
    assert reader.timing.snapshot()["confirmation_review_duration"] == 4
    assert reader.timing.snapshot()["confirmation_outcome"] == "interrupted"
    assert not session.has_pending


def test_main_clarification_wait_excludes_capture_review_and_question_display(monkeypatch):
    clock, records = Clock(), []
    recordings = iter(["Create a note", "कल DBMS पढ़ना hai"])

    def listen(**kwargs):
        clock.advance(3.5)
        result = transcript(next(recordings))
        result.update(recording_time=1, transcription_duration=2, normalization_duration=0.5)
        return result

    stt = SimpleNamespace(
        config=AudioConfig(),
        execution_device="cpu",
        compute_type="float32",
        listen_and_transcribe=MagicMock(side_effect=listen),
        load_model=MagicMock(return_value=0),
        load_vad=MagicMock(),
    )
    monkeypatch.setattr("audio.SpeechToText", lambda config: stt)
    monkeypatch.setattr(main, "load_dotenv", lambda: None)

    def understand(text):
        clock.advance(2)
        return CommandRequest(intent="create_note", confidence=0.99)

    nlu = MagicMock(side_effect=understand)
    monkeypatch.setattr(command_processor, "understand_command", nlu)
    answers = iter([(100, ""), (1, "yes"), (5, ""), (1, "yes"), (100, "exit")])

    def answer(prompt):
        seconds, value = next(answers)
        clock.advance(seconds)
        return value

    monkeypatch.setattr("builtins.input", answer)
    import builtins

    original_print = builtins.print

    def display(*args, **kwargs):
        if args and "What note content" in str(args[0]):
            clock.advance(2)
        original_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", display)
    main.main(["--voice"], measurement_clock=clock, timing_records=records)
    assert len(records) == 1
    record = records[0]
    assert record["clarification_wait_duration"] == 5
    assert record["command_capture_duration"] == 2
    assert record["transcription_duration"] == 4
    assert record["normalization_duration"] == 1
    assert record["transcript_review_duration"] == 2
    assert record["total_interaction_duration"] == 18
    nlu.assert_called_once()


@pytest.mark.parametrize("failure", ["log", "collector"])
def test_telemetry_failure_does_not_interrupt_completed_command(monkeypatch, failure):
    clock, records = Clock(), []
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        command_processor,
        "understand_command",
        lambda text: CommandRequest(
            intent="get_notes",
            confidence=0.99,
        ),
    )
    answers = iter(["Show notes", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    logger = main.logging.getLogger(main.__name__)
    if failure == "log":
        monkeypatch.setattr(
            logger, "debug", MagicMock(side_effect=OSError("Synthetic log failure"))
        )
        collector = records
    else:
        collector = SimpleNamespace(append=MagicMock(side_effect=OSError("Synthetic sink failure")))
    main.main(measurement_clock=clock, timing_records=collector)
    assert len(records) == (1 if failure == "log" else 0)


def test_interrupted_replacement_publishes_separate_interactions(monkeypatch):
    clock, records = Clock(), []
    monkeypatch.setattr(main, "load_dotenv", lambda: None)

    def understand(text):
        if text == "Create a note":
            clock.advance(1)
            return CommandRequest(intent="create_note", confidence=0.99)
        clock.advance(2)
        raise KeyboardInterrupt

    monkeypatch.setattr(command_processor, "understand_command", understand)
    answers = iter(["Create a note", "/new Show notes"])

    def answer(prompt):
        text = next(answers)
        if text.startswith("/new"):
            clock.advance(4)
        return text

    monkeypatch.setattr("builtins.input", answer)
    main.main(measurement_clock=clock, timing_records=records)
    assert len(records) == 2
    assert records[0]["interaction_id"] != records[1]["interaction_id"]
    assert records[0]["nlu_duration"] == 1
    assert records[0]["clarification_wait_duration"] == 4
    assert records[0]["total_interaction_duration"] == 5
    assert records[1]["nlu_duration"] == records[1]["total_interaction_duration"] == 2
    assert current_timing.get() is None


def test_new_prefix_without_pending_command_creates_one_record(monkeypatch):
    records = []
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        command_processor,
        "understand_command",
        lambda text: CommandRequest(
            intent="get_notes",
            confidence=0.99,
        ),
    )
    answers = iter(["/new Show notes", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    main.main(measurement_clock=Clock(), timing_records=records)
    assert len(records) == 1
