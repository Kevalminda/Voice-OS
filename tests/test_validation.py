import pytest

from core.schemas import CommandRequest, Intent, Language
from core.validation import validate_command


def test_valid_reminder():
    command = CommandRequest(
        intent=Intent.CREATE_REMINDER,
        entities={"task": "Call Mom", "time": "7 PM"},
        language=Language.ENGLISH,
        confidence=0.95,
    )

    result = validate_command(command)

    assert result.status == "success"


def test_missing_reminder_details():
    command = CommandRequest(
        intent=Intent.CREATE_REMINDER, entities={"task": "Call Mom"}, confidence=0.95
    )

    result = validate_command(command)

    assert result.status == "needs_clarification"
    assert "time" in result.data["missing_fields"]


def test_low_confidence():
    command = CommandRequest(
        intent=Intent.CREATE_NOTE, entities={"content": "Buy groceries"}, confidence=0.40
    )

    result = validate_command(command)

    assert result.status == "needs_clarification"


def test_unknown_command():
    command = CommandRequest(intent=Intent.UNKNOWN, confidence=0.95)

    result = validate_command(command)

    assert result.status == "rejected"


@pytest.mark.parametrize(
    "confidence,status",
    [(0.0, "needs_clarification"), (0.649999, "needs_clarification"),
     (0.65, "success"), (0.650001, "success"), (1.0, "success")],
)
def test_confidence_boundary_preserves_existing_policy(confidence, status):
    command = CommandRequest(
        intent="create_note", entities={"content": "buy milk"}, confidence=confidence,
    )

    assert validate_command(command).status == status


@pytest.mark.parametrize(
    "intent,entities,missing",
    [
        ("create_note", {}, "content"),
        ("create_note", {"content": None}, "content"),
        ("create_note", {"content": "  "}, "content"),
        ("delete_note", {}, "note_id"),
        ("delete_note", {"note_id": None}, "note_id"),
        ("update_note", {"note_id": "12"}, "content"),
        ("update_note", {"content": "कल DBMS पढ़ना है"}, "note_id"),
        ("update_note", {"note_id": "12", "content": "  "}, "content"),
    ],
)
def test_missing_note_entities_request_clarification(intent, entities, missing):
    command = CommandRequest(intent=intent, entities=entities, confidence=0.95)

    result = validate_command(command)

    assert result.status == "needs_clarification"
    assert missing in result.data["missing_fields"].split(", ")


def test_model_reported_missing_fields_prevent_execution():
    command = CommandRequest(
        intent="create_note", entities={"content": "buy milk"}, confidence=0.95,
        missing_fields=["content"],
    )

    assert validate_command(command).status == "needs_clarification"
