"""Multilingual output checks using actual encoders and simulated stream capabilities."""

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import main
from actions.notes import update_note


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_redirected_cp1252_stream_preserves_hindi_as_utf8(monkeypatch, stream_name):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, stream_name, stream)

    main._configure_voice_output()
    print("VoiceOS हिंदी", file=getattr(sys, stream_name), flush=True)

    assert raw.getvalue().decode("utf-8") == "VoiceOS हिंदी\n"


@pytest.mark.parametrize("content", ["Buy milk", "कल DBMS पढ़ना है।", "kal DBMS पढ़ना hai"],
                         ids=["english", "hindi", "hinglish"])
def test_text_mode_displays_multilingual_action_result_on_redirected_output(
    monkeypatch, existing_note, content, caplog,
):
    assert update_note(existing_note, content).status == "success"
    generate = MagicMock(return_value=json.dumps({
        "intent": "get_notes", "entities": {}, "language": "unknown",
        "confidence": 0.95, "missing_fields": [],
    }))
    monkeypatch.setattr("nlu.service.generate_response", generate)
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    output_bytes, error_bytes = io.BytesIO(), io.BytesIO()
    output = io.TextIOWrapper(output_bytes, encoding="cp1252", newline="\n", write_through=True)
    errors = io.TextIOWrapper(error_bytes, encoding="cp1252", newline="\n", write_through=True)
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    answers = iter(["Show my notes", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    main.main()
    print("हिंदी diagnostics", file=sys.stderr, flush=True)
    sys.stdout.flush()

    rendered = output_bytes.getvalue().decode("utf-8")
    assert content in rendered
    assert "[success]" in rendered
    assert "VoiceOS shutting down." in rendered
    assert error_bytes.getvalue().decode("utf-8") == "हिंदी diagnostics\n"
    assert "Application stopped" not in caplog.text
    generate.assert_called_once()


def test_unicode_memory_streams_without_reconfigure_work(monkeypatch):
    output, errors = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)

    main._configure_voice_output()
    print("English हिंदी", flush=True)
    print("हिंदी error", file=sys.stderr, flush=True)

    assert output.getvalue() == "English हिंदी\n"
    assert errors.getvalue() == "हिंदी error\n"


class LegacyEncodedStream:
    """A text writer with an actual cp1252 encoder and no reconfigure method."""

    encoding = "cp1252"

    def __init__(self):
        self.raw = bytearray()

    def write(self, text):
        self.raw.extend(text.encode(self.encoding))
        return len(text)

    def flush(self):
        pass


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_unconfigurable_encoded_stream_uses_safe_unicode_escapes(monkeypatch, stream_name):
    original = LegacyEncodedStream()
    monkeypatch.setattr(sys, stream_name, original)

    main._configure_voice_output()
    configured = getattr(sys, stream_name)
    main._configure_voice_output()
    assert getattr(sys, stream_name) is configured
    print("English हिंदी", file=configured, flush=True)

    expected = "English हिंदी\n".encode("cp1252", errors="backslashreplace")
    assert bytes(original.raw) == expected


def test_failed_stdout_reconfiguration_does_not_block_stderr(monkeypatch):
    original = LegacyEncodedStream()
    original.reconfigure = MagicMock(side_effect=io.UnsupportedOperation("Unavailable"))
    error_bytes = io.BytesIO()
    errors = io.TextIOWrapper(error_bytes, encoding="cp1252", newline="\n")
    monkeypatch.setattr(sys, "stdout", original)
    monkeypatch.setattr(sys, "stderr", errors)

    main._configure_voice_output()
    print("हिंदी", flush=True)
    print("हिंदी", file=sys.stderr, flush=True)

    assert bytes(original.raw) == "हिंदी\n".encode("cp1252", errors="backslashreplace")
    assert error_bytes.getvalue().decode("utf-8") == "हिंदी\n"


def test_interactive_stream_retains_its_configuration_and_english_output(monkeypatch):
    output = io.StringIO()
    reconfigure = MagicMock()
    terminal = SimpleNamespace(
        isatty=lambda: True, reconfigure=reconfigure, write=output.write, flush=output.flush,
    )
    monkeypatch.setattr(sys, "stdout", terminal)

    main._configure_voice_output()
    print("VoiceOS ready", flush=True)

    assert sys.stdout is terminal
    reconfigure.assert_not_called()
    assert output.getvalue() == "VoiceOS ready\n"
