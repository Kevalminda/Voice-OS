from core.schemas import Intent, Language
from nlu.service import understand_command


def test_understand_create_note(monkeypatch):
    def fake_generate_response(prompt):
        return """
        {
            "intent": "create_note",
            "entities": {
                "content": "My project submission is on Friday"
            },
            "language": "en",
            "confidence": 0.98,
            "missing_fields": []
        }
        """

    monkeypatch.setattr("nlu.service.generate_response", fake_generate_response)

    command = understand_command("Save a note that my project submission is on Friday.")

    assert command.intent == Intent.CREATE_NOTE
    assert command.language == Language.ENGLISH
    assert command.entities["content"] == ("My project submission is on Friday")


def test_empty_command():
    import pytest

    with pytest.raises(ValueError):
        understand_command("")
