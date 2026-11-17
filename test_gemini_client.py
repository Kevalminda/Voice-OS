"""Optional live smoke test. Explicit opt-in only; may consume API quota."""

import os

import pytest


@pytest.mark.live
def test_live_gemini():
    if os.getenv("VOICEOS_LIVE_TEST") != "1":
        pytest.skip("Set VOICEOS_LIVE_TEST=1 to allow a live Gemini request.")
    from core.schemas import Intent
    from nlu.service import understand_command

    command = understand_command("Save a note: VoiceOS smoke test")
    assert command.intent == Intent.CREATE_NOTE
    assert command.entities.get("content")
