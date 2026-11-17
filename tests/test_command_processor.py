from core.command_processor import process_text_command
from core.schemas import CommandRequest, Intent, Language


def test_process_create_note(monkeypatch):

    fake_command = CommandRequest(
        intent=Intent.CREATE_NOTE,
        entities={"content": "My project submission is on Friday"},
        language=Language.ENGLISH,
        confidence=0.98,
    )

    def fake_understand_command(user_text):
        return fake_command

    monkeypatch.setattr("core.command_processor.understand_command", fake_understand_command)

    result = process_text_command("Save a note about my project submission.")

    assert result.status == "success"
    assert result.message == "Your note has been saved."


def test_delete_requires_confirmation(monkeypatch, existing_note):
    command = CommandRequest(
        intent=Intent.DELETE_NOTE,
        entities={"note_id": str(existing_note)},
        language=Language.ENGLISH,
        confidence=0.98,
    )

    monkeypatch.setattr("core.command_processor.understand_command", lambda text: command)

    result = process_text_command("Delete note 1.")

    assert result.status == "needs_confirmation"


def test_delete_cancelled(monkeypatch, existing_note):
    command = CommandRequest(
        intent=Intent.DELETE_NOTE,
        entities={"note_id": str(existing_note)},
        language=Language.ENGLISH,
        confidence=0.98,
    )

    monkeypatch.setattr("core.command_processor.understand_command", lambda text: command)

    result = process_text_command("Delete note 1.", confirm_action=lambda pending: False)

    assert result.status == "rejected"
    assert result.message == "Note deletion cancelled."
