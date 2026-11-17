from actions.notes import create_note, get_notes, update_note
from core.schemas import CommandRequest, Intent


def test_update_existing_note():

    command = CommandRequest(
        intent=Intent.CREATE_NOTE, entities={"content": "Original note"}, confidence=0.95
    )

    created = create_note(command)
    note_id = created.data["note_id"]

    result = update_note(note_id, "Updated note")

    assert result.status == "success"

    notes = get_notes()
    assert "Updated note" in notes.message
    assert "Original note" not in notes.message


def test_update_nonexistent_note():

    result = update_note(999999, "Updated content")

    assert result.status == "error"


def test_update_empty_content():

    result = update_note(1, "   ")

    assert result.status == "needs_clarification"
