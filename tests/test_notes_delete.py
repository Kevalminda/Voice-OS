from actions.notes import create_note, delete_note, get_notes
from core.schemas import CommandRequest, Intent


def test_delete_existing_note():

    command = CommandRequest(
        intent=Intent.CREATE_NOTE, entities={"content": "Temporary deletion test"}, confidence=0.95
    )

    created = create_note(command)
    assert created.status == "success"

    note_id = created.data["note_id"]

    result = delete_note(note_id)

    assert result.status == "success"

    notes = get_notes()
    assert "Temporary deletion test" not in notes.message


def test_delete_nonexistent_note():

    result = delete_note(999999)

    assert result.status == "error"


def test_delete_invalid_note_id():

    result = delete_note(0)

    assert result.status == "rejected"
