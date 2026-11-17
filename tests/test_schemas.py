import pytest
from pydantic import ValidationError

from core.schemas import CommandRequest, Intent, Language


def test_valid_command():
    command = CommandRequest(
        intent=Intent.CREATE_REMINDER,
        entities={"task": "Call Mom", "time": "7 PM"},
        language=Language.ENGLISH,
        confidence=0.95,
    )

    assert command.intent == Intent.CREATE_REMINDER
    assert command.confidence == 0.95


def test_invalid_confidence():
    with pytest.raises(ValidationError):
        CommandRequest(intent=Intent.CREATE_NOTE, confidence=1.5)


def test_unknown_field():
    with pytest.raises(ValidationError):
        CommandRequest(intent=Intent.CREATE_NOTE, confidence=0.8, random_field="not allowed")
