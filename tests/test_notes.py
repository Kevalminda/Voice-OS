from actions.notes import get_notes


def test_get_notes():
    result = get_notes()

    assert result.status == "success"
    assert (
        "My first VoiceOS note" in result.message
        or "Buy groceries" in result.message
        or "saved notes" in result.message
    )
