import json

import pytest

from core.schemas import Intent, Language
from nlu.parser import parse_nlu_response


def test_parse_create_note_response():
    response = """
    {
        "intent": "create_note",
        "entities": {
            "content": "I have a DBMS exam tomorrow"
        },
        "language": "en",
        "confidence": 0.98,
        "missing_fields": []
    }
    """

    command = parse_nlu_response(response)

    assert command.intent == Intent.CREATE_NOTE
    assert command.language == Language.ENGLISH
    assert command.entities["content"] == ("I have a DBMS exam tomorrow")
    assert command.confidence == 0.98


def test_normalize_text_to_content_for_create_note():
    response = """
    {
        "intent": "create_note",
        "entities": {
            "text": "My DBMS exam is tomorrow"
        },
        "language": "en",
        "confidence": 0.98,
        "missing_fields": []
    }
    """

    command = parse_nlu_response(response)

    assert command.intent == Intent.CREATE_NOTE
    assert command.entities["content"] == ("My DBMS exam is tomorrow")
    assert "text" not in command.entities


def test_normalize_integer_entity_values():
    response = """
    {
        "intent": "delete_note",
        "entities": {
            "note_id": 1
        },
        "language": "en",
        "confidence": 0.98,
        "missing_fields": []
    }
    """

    command = parse_nlu_response(response)

    assert command.entities["note_id"] == "1"
    assert isinstance(command.entities["note_id"], str)


@pytest.mark.parametrize("ensure_ascii", [False, True], ids=["unicode", "json-escapes"])
@pytest.mark.parametrize(
    "language,content",
    [
        (Language.ENGLISH, "Keval Minda uses VoiceOS, Python, Gemini and DBMS-12."),
        (Language.HINDI, "केवल मिंडा आज VoiceOS में Python, Gemini और DBMS-12 पढ़ेगा।"),
        (Language.HINGLISH, "kal VoiceOS mein Python aur Gemini ke saath DBMS-12 पढ़ना है"),
    ],
    ids=["english", "hindi", "hinglish"],
)
def test_content_preserves_language_script_and_identifiers(language, content, ensure_ascii):
    response = json.dumps({
        "intent": "create_note", "entities": {"content": content},
        "language": language.value, "confidence": 0.95, "missing_fields": [],
    }, ensure_ascii=ensure_ascii)

    command = parse_nlu_response(response)

    assert command.language == language
    assert command.entities == {"content": content}
