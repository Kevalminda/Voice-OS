"""Offline dialogue contracts: Gemini is mocked; validation and storage remain real."""

import json
from unittest.mock import MagicMock

import pytest

import main
from actions.notes import get_note, get_notes, update_note
from core import command_processor
from core.dialogue import DialogueSession
from core.schemas import CommandRequest
from core.validation import validate_command


@pytest.fixture
def generate(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("nlu.service.generate_response", mock)
    return mock


def response(intent="create_note", entities=None, language="english", **kwargs):
    return json.dumps(
        {
            "intent": intent,
            "entities": entities or {},
            "language": language,
            "confidence": 0.95,
            "missing_fields": [],
            **kwargs,
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    "text,answer,language",
    [
        ("Create a note", "Buy milk and review DBMS with Python.", "english"),
        ("मेरे लिए एक नोट बनाओ", "कल DBMS पढ़ना है और Gemini इस्तेमाल करना है।", "hindi"),
        ("Mere liye ek note banao", "kal VoiceOS mein Python aur DBMS पढ़ना है", "hinglish"),
    ],
    ids=["english", "hindi", "hinglish"],
)
def test_multilingual_completion_preserves_original_request(
    monkeypatch,
    generate,
    isolated_database,
    text,
    answer,
    language,
):
    generate.return_value = response(language=language, missing_fields=["content"])
    session = DialogueSession()
    before = isolated_database.read_bytes()
    route = MagicMock(wraps=command_processor.route_command)
    monkeypatch.setattr(command_processor, "route_command", route)

    pending = session.process_text_command(text)

    assert pending.status == "needs_clarification"
    assert pending.data["requested_field"] == "content"
    assert session.requested_field == "content"
    assert session.original_command.entities == {}
    assert session.original_command.missing_fields == ["content"]
    assert isolated_database.read_bytes() == before
    route.assert_not_called()

    completed = session.process_text_command(answer)

    assert completed.status == "success"
    assert get_note(completed.data["note_id"]).data["content"] == answer
    command = route.call_args.args[0]
    assert command.intent == "create_note"
    assert command.language == language
    assert command.confidence == 0.95
    assert command.entities == {"content": answer}
    assert command.missing_fields == []
    assert not session.has_pending
    assert session.original_command is None
    generate.assert_called_once()


def test_multiple_missing_fields_are_collected_one_at_a_time(generate, existing_note):
    generate.return_value = response("update_note", missing_fields=["note_id", "content"])
    session = DialogueSession()
    assert session.process_text_command("Update a note").data["requested_field"] == "note_id"

    pending = session.process_text_command(str(existing_note))

    assert pending.status == "needs_clarification"
    assert pending.data["requested_field"] == "content"
    assert session.pending_command.entities == {"note_id": str(existing_note)}
    assert session.original_command.entities == {}
    assert get_note(existing_note).data["content"] == "Test note"
    result = session.process_text_command("आज DBMS पढ़ना है।")
    assert result.status == "success"
    assert get_note(existing_note).data["content"] == "आज DBMS पढ़ना है।"
    assert not session.has_pending
    generate.assert_called_once()


def test_supplied_entities_and_returned_snapshots_cannot_be_overwritten(generate, existing_note):
    generate.return_value = response("update_note", {"note_id": existing_note}, language="hindi")
    session = DialogueSession()
    session.process_text_command("इस नोट को बदलो")
    session.pending_command.entities["note_id"] = "999"
    session.original_command.entities["note_id"] = "999"

    assert session.process_text_command("नोट नंबर 999: VoiceOS DBMS").status == "success"
    assert get_note(existing_note).data["content"] == "नोट नंबर 999: VoiceOS DBMS"
    assert get_note(999).status == "error"


@pytest.mark.parametrize("answer", ["", " ", "yes", "हाँ", "haan", "maybe", "yes or no"])
def test_ambiguous_content_replies_retry_without_execution(generate, isolated_database, answer):
    generate.return_value = response()
    session = DialogueSession(max_retries=2)
    session.process_text_command("Create a note")
    before = isolated_database.read_bytes()

    result = session.process_text_command(answer)

    assert result.status == "needs_clarification"
    assert result.data["requested_field"] == "content"
    assert session.has_pending
    assert isolated_database.read_bytes() == before
    assert session.process_text_command("buy milk").status == "success"
    generate.assert_called_once()


@pytest.mark.parametrize(
    "answer",
    ["twelve", "12 or 13", "0", "-1", "१", "9223372036854775808", "12; DROP TABLE notes", "yes"],
)
def test_invalid_ids_retry_without_inventing_or_changing_entities(
    generate,
    isolated_database,
    existing_note,
    answer,
):
    generate.return_value = response("delete_note")
    session = DialogueSession()
    session.process_text_command("Delete a note")
    before = isolated_database.read_bytes()
    confirm = MagicMock(return_value=True)

    result = session.process_text_command(answer, confirm)

    assert result.status == "needs_clarification"
    assert session.pending_command.entities == {}
    assert session.requested_field == "note_id"
    assert isolated_database.read_bytes() == before
    confirm.assert_not_called()
    assert get_note(existing_note).status == "success"


def test_retry_exhaustion_clears_pending_without_mutation(generate, isolated_database):
    generate.return_value = response()
    session = DialogueSession(max_retries=2)
    session.process_text_command("Create a note")
    before = isolated_database.read_bytes()

    assert session.process_text_command("maybe").status == "needs_clarification"
    result = session.process_text_command("yes")

    assert result.status == "rejected"
    assert "restate" in result.message.lower()
    assert not session.has_pending
    assert isolated_database.read_bytes() == before
    generate.assert_called_once()


@pytest.mark.parametrize("text", ["/cancel", "cancel", "रद्द करो", "cancel karo"])
def test_cancellation_clears_pending_without_mutation(generate, isolated_database, text):
    generate.return_value = response()
    session = DialogueSession()
    session.process_text_command("Create a note")
    before = isolated_database.read_bytes()

    assert session.process_text_command(text).status == "rejected"
    assert not session.has_pending
    assert isolated_database.read_bytes() == before
    generate.assert_called_once()


def test_expiry_rejects_late_answer_without_reinterpreting_it(generate, isolated_database):
    now = [100.0]
    generate.return_value = response()
    session = DialogueSession(timeout_seconds=10, clock=lambda: now[0])
    session.process_text_command("Create a note")
    before = isolated_database.read_bytes()
    now[0] = 110.0

    result = session.process_text_command("buy milk")

    assert result.status == "rejected"
    assert "expired" in result.message.lower()
    assert not session.has_pending
    assert isolated_database.read_bytes() == before
    generate.assert_called_once()


def test_progress_does_not_extend_expiry(generate, existing_note):
    now = [0.0]
    generate.return_value = response("update_note")
    session = DialogueSession(timeout_seconds=10, clock=lambda: now[0])
    session.process_text_command("Update a note")
    now[0] = 9.0
    assert session.process_text_command(str(existing_note)).status == "needs_clarification"
    now[0] = 10.0
    assert session.process_text_command("new content").status == "rejected"
    assert not session.has_pending
    assert get_note(existing_note).data["content"] == "Test note"


def test_explicit_replacement_discards_the_old_request(generate):
    generate.side_effect = [response(), response("get_notes")]
    session = DialogueSession()
    session.process_text_command("Create a note")

    result = session.process_text_command("/new Show my notes")

    assert result.status == "success"
    assert not session.has_pending
    assert "saved notes" in result.message
    assert generate.call_count == 2


def test_replacement_failure_does_not_restore_old_request(generate):
    generate.side_effect = [response(), ValueError("Invalid NLU response")]
    session = DialogueSession()
    session.process_text_command("Create a note")

    assert session.process_text_command("/new something unclear").status == "rejected"
    assert not session.has_pending


def test_cross_session_isolation(generate):
    generate.side_effect = [response(language="hindi"), response(language="hinglish")]
    first, second = DialogueSession(), DialogueSession()
    first.process_text_command("नोट बनाओ")
    second.process_text_command("note banao")

    first.cancel()

    assert not first.has_pending
    assert second.requested_field == "content"
    result = second.process_text_command("kal DBMS padhna hai")
    assert get_note(result.data["note_id"]).data["content"] == "kal DBMS padhna hai"
    assert not second.has_pending


@pytest.mark.parametrize(
    "payload",
    [
        response("unknown"),
        response(confidence=0.649999),
        response("create_reminder", {"task": "read"}),
        response("web_search"),
        response(entities={"content": "already present"}, missing_fields=["content"]),
        response(missing_fields=["title"]),
    ],
    ids=["unknown", "low-confidence", "reminder", "search", "uncertain", "unknown-field"],
)
def test_only_identifiable_supported_incomplete_commands_are_retained(
    generate,
    isolated_database,
    payload,
):
    generate.return_value = payload
    session = DialogueSession()
    before = isolated_database.read_bytes()

    result = session.process_text_command("an incomplete command")

    assert result.status in {"rejected", "needs_clarification"}
    assert not session.has_pending
    assert isolated_database.read_bytes() == before


def test_unidentified_validation_response_requires_restatement(monkeypatch, generate):
    from core.schemas import ActionResult

    generate.return_value = response()
    monkeypatch.setattr(
        "core.dialogue.validate_command",
        lambda command: ActionResult(
            status="needs_clarification",
            message="Some detail is unclear.",
        ),
    )
    session = DialogueSession()

    result = session.process_text_command("Create a note")

    assert result.status == "needs_clarification"
    assert "restate" in result.message.lower()
    assert not session.has_pending


def test_exact_confidence_threshold_can_be_completed(generate):
    generate.return_value = response(confidence=0.65)
    session = DialogueSession()
    assert session.process_text_command("Create a note").status == "needs_clarification"
    assert session.process_text_command("buy milk").status == "success"


def test_completed_request_is_revalidated_and_consumed_before_routing(monkeypatch, generate):
    generate.return_value = response()
    session = DialogueSession()

    def route(command, confirm_action=None):
        assert not session.has_pending
        assert validate_command(command).status == "success"
        raise ValueError("Rejected at final boundary")

    monkeypatch.setattr(command_processor, "route_command", route)
    session.process_text_command("Create a note")
    assert session.process_text_command("buy milk").status == "rejected"
    assert not session.has_pending


@pytest.mark.parametrize(
    "approval,status",
    [(None, "needs_confirmation"), (False, "rejected"), ("yes", "rejected"), (True, "success")],
)
def test_completed_deletion_keeps_synchronous_confirmation(
    generate,
    existing_note,
    approval,
    status,
):
    generate.return_value = response("delete_note")
    session = DialogueSession()
    session.process_text_command("Delete a note")
    confirm = None if approval is None else MagicMock(return_value=approval)

    result = session.process_text_command(str(existing_note), confirm)

    assert result.status == status
    assert not session.has_pending
    assert (get_note(existing_note).status == "error") is (approval is True)
    generate.assert_called_once()


def test_completed_deletion_preserves_atomic_content_check(generate, existing_note):
    generate.return_value = response("delete_note")
    session = DialogueSession()
    session.process_text_command("Delete a note")

    def confirm(pending):
        assert pending.data["preview"] == "Test note"
        update_note(existing_note, "Changed during confirmation")
        return True

    assert session.process_text_command(str(existing_note), confirm).status == "error"
    assert get_note(existing_note).data["content"] == "Changed during confirmation"
    assert not session.has_pending


def test_unsafe_non_text_reply_cancels_instead_of_guessing(generate, isolated_database):
    generate.return_value = response()
    session = DialogueSession()
    session.process_text_command("Create a note")
    before = isolated_database.read_bytes()

    assert session.process_text_command({"content": "injected"}).status == "rejected"
    assert not session.has_pending
    assert isolated_database.read_bytes() == before


def test_original_nlu_model_is_not_mutated(monkeypatch):
    original = CommandRequest(intent="create_note", confidence=0.95, missing_fields=["content"])
    before = original.model_dump()
    monkeypatch.setattr(command_processor, "understand_command", lambda text: original)
    session = DialogueSession()
    session.process_text_command("Create a note")
    assert session.process_text_command("VoiceOS DBMS").status == "success"
    assert original.model_dump() == before


def test_cli_collects_followup_instead_of_sending_it_to_nlu(monkeypatch, generate, capsys):
    generate.return_value = response(language="hindi")
    answers = iter(["मेरे लिए नोट बनाओ", "कल DBMS पढ़ना है", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    main.main()

    assert "कल DBMS पढ़ना है" in get_notes().message
    assert "[needs_clarification]" in capsys.readouterr().out
    generate.assert_called_once()


@pytest.mark.parametrize("metadata", [None, "", "title", "content, content", "content, note_id"])
def test_invalid_missing_field_metadata_never_creates_pending_state(
    monkeypatch,
    generate,
    isolated_database,
    metadata,
):
    from core.schemas import ActionResult

    generate.return_value = response()
    monkeypatch.setattr(
        "core.dialogue.validate_command",
        lambda command: ActionResult(
            status="needs_clarification",
            message="A detail is missing.",
            data={"missing_fields": metadata},
        ),
    )
    before = isolated_database.read_bytes()
    session = DialogueSession()

    result = session.process_text_command("Create a note")

    assert result.status == "needs_clarification"
    assert "restate" in result.message.lower()
    assert not session.has_pending
    assert isolated_database.read_bytes() == before


def test_invalid_supplied_id_is_not_corrected_or_executed(generate, isolated_database):
    generate.return_value = response("update_note", {"note_id": "12; DROP TABLE notes"})
    session = DialogueSession()
    before = isolated_database.read_bytes()
    session.process_text_command("Update a note")

    assert session.process_text_command("new content").status == "rejected"
    assert not session.has_pending
    assert isolated_database.read_bytes() == before


def test_sql_like_content_reply_is_stored_literally(generate, existing_note):
    generate.return_value = response()
    session = DialogueSession()
    session.process_text_command("Create a note")
    content = "'); DROP TABLE notes; -- DBMS पढ़ना है"

    result = session.process_text_command(content)

    assert result.status == "success"
    assert get_note(result.data["note_id"]).data["content"] == content
    assert get_note(existing_note).data["content"] == "Test note"


def test_later_yes_does_not_resume_an_unconfirmed_deletion(generate, existing_note):
    generate.side_effect = [response("delete_note"), response("unknown")]
    session = DialogueSession()
    session.process_text_command("Delete a note")
    assert session.process_text_command(str(existing_note)).status == "needs_confirmation"

    assert session.process_text_command("yes").status == "rejected"
    assert not session.has_pending
    assert get_note(existing_note).data["content"] == "Test note"


@pytest.mark.parametrize("signal", [KeyboardInterrupt, EOFError])
def test_interrupted_confirmation_does_not_leave_pending_state(generate, existing_note, signal):
    generate.return_value = response("delete_note")
    session = DialogueSession()
    session.process_text_command("Delete a note")

    def confirm(pending):
        raise signal()

    if signal is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            session.process_text_command(str(existing_note), confirm)
    else:
        assert session.process_text_command(str(existing_note), confirm).status == "error"
    assert not session.has_pending
    assert get_note(existing_note).data["content"] == "Test note"


def test_cli_blank_followup_reprompts_then_completes(monkeypatch, generate, capsys):
    generate.return_value = response()
    answers = iter(["Create a note", "", "buy milk", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    main.main()

    assert get_note(1).data["content"] == "buy milk"
    assert "ambiguous or invalid" in capsys.readouterr().out
    generate.assert_called_once()


def test_cli_shutdown_clears_the_session(monkeypatch, generate, isolated_database):
    generate.return_value = response()
    session = DialogueSession()
    monkeypatch.setattr(main, "DialogueSession", lambda: session)
    answers = iter(["Create a note", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    before = isolated_database.read_bytes()

    main.main()

    assert not session.has_pending
    assert isolated_database.read_bytes() == before


@pytest.mark.parametrize(
    "settings",
    [
        {"max_retries": 0},
        {"max_retries": True},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_session_limits_are_rejected(settings):
    with pytest.raises(ValueError):
        DialogueSession(**settings)
