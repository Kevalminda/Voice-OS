from core.action_router import route_command
from core.schemas import CommandRequest, Intent


def test_note_routing():
    command = CommandRequest(
        intent=Intent.CREATE_NOTE, entities={"content": "Buy groceries"}, confidence=0.95
    )

    result = route_command(command)

    assert result.status == "success"
    assert result.message == "Your note has been saved."


def test_reminder_routing():
    command = CommandRequest(
        intent=Intent.CREATE_REMINDER,
        entities={"task": "Call Mom", "time": "7 PM"},
        confidence=0.95,
    )

    result = route_command(command)

    assert result.status == "rejected"
    assert "not implemented" in result.message


def test_invalid_command_not_routed():
    command = CommandRequest(
        intent=Intent.CREATE_REMINDER, entities={"task": "Call Mom"}, confidence=0.95
    )

    result = route_command(command)

    assert result.status == "needs_clarification"


def test_low_confidence_not_routed():
    command = CommandRequest(
        intent=Intent.CREATE_NOTE, entities={"content": "Buy groceries"}, confidence=0.30
    )

    result = route_command(command)

    assert result.status == "needs_clarification"


def test_get_notes_routing():
    command = CommandRequest(intent=Intent.GET_NOTES, confidence=0.90)

    result = route_command(command)

    assert result.status == "success"
    assert (
        "saved notes" in result.message.lower() or "my first voiceos note" in result.message.lower()
    )


def test_delete_note_routing():

    command = CommandRequest(
        intent=Intent.DELETE_NOTE, entities={"note_id": "999999"}, confidence=0.95
    )

    result = route_command(command)

    assert result.status == "error"
    assert "No note found" in result.message
