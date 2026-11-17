import socket

import pytest

from storage import database


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch, request):
    path = tmp_path / "notes.db"
    monkeypatch.setenv("VOICEOS_DATABASE_PATH", str(path))
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    database.initialize_database()
    if not request.node.get_closest_marker("live"):

        def no_network(*args, **kwargs):
            raise AssertionError("Network access is forbidden in offline tests.")

        monkeypatch.setattr(socket.socket, "connect", no_network)
        monkeypatch.setattr(socket, "create_connection", no_network)
    return path


@pytest.fixture
def existing_note():
    from actions.notes import create_note
    from core.schemas import CommandRequest

    return create_note(
        CommandRequest(intent="create_note", entities={"content": "Test note"}, confidence=0.99)
    ).data["note_id"]
