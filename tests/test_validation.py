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
