from unittest.mock import MagicMock

import pytest

from core import action_router
from core.action_router import route_command
from core.schemas import ActionResult, CommandRequest, Intent


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


@pytest.mark.parametrize(
    "intent,entities",
    [("unknown", {}), ("create_reminder", {"task": "read", "time": "tomorrow"}),
     ("web_search", {"query": "VoiceOS"})],
)
def test_unsupported_intents_never_call_note_handlers(monkeypatch, intent, entities):
    handlers = []
    for name in ("create_note", "get_notes", "get_note", "update_note", "delete_note"):
        handler = MagicMock()
        monkeypatch.setattr(action_router, name, handler)
        handlers.append(handler)
    confirm = MagicMock(return_value=True)

    result = route_command(
        CommandRequest(intent=intent, entities=entities, confidence=0.99), confirm,
    )

    assert result.status == "rejected"
    confirm.assert_not_called()
    for handler in handlers:
        handler.assert_not_called()


def test_router_revalidates_entities_changed_after_construction(monkeypatch):
    handler = MagicMock()
    monkeypatch.setattr(action_router, "create_note", handler)
    command = CommandRequest(
        intent="create_note", entities={"content": "buy milk"}, confidence=0.99,
    )
    command.entities["execute_sql"] = "DROP TABLE notes"

    with pytest.raises(ValueError):
        route_command(command)

    handler.assert_not_called()


def test_valid_command_at_confidence_threshold_reaches_handler(monkeypatch):
    command = CommandRequest(
        intent="create_note", entities={"content": "buy milk"}, confidence=0.65,
    )
    handler = MagicMock(return_value=ActionResult(status="success", message="Saved"))
    monkeypatch.setattr(action_router, "create_note", handler)

    assert route_command(command).status == "success"
    handler.assert_called_once_with(command)
