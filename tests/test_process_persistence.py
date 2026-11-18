"""Exercise real SQLite persistence in fresh interpreters using temporary databases."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHILD_BOOTSTRAP = """
import json
import socket
import sys

sys.path[:0] = json.loads(sys.argv[1])

def no_network(*args, **kwargs):
    raise AssertionError("Network access is forbidden in persistence tests.")

socket.socket.connect = no_network
socket.create_connection = no_network
payload = json.load(sys.stdin)
exec(compile(sys.argv[2], "<persistence-child>", "exec"))
"""


@pytest.fixture
def run_process(tmp_path):
    """Pass JSON as payload and require every child database to stay under tmp_path."""
    import_paths = [str(Path(path).resolve()) for path in sys.path if path]

    def run(code, *, database_path, payload=None, cwd=None):
        directory = Path(cwd or tmp_path).resolve()
        database_path = Path(database_path)
        resolved_database = (directory / database_path).resolve()
        assert directory.is_relative_to(tmp_path.resolve())
        assert resolved_database.is_relative_to(tmp_path.resolve())
        directory.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["VOICEOS_DATABASE_PATH"] = str(database_path)
        environment["VOICEOS_LOG_LEVEL"] = "INFO"
        environment.pop("GEMINI_API_KEY", None)
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-X",
                "utf8",
                "-c",
                CHILD_BOOTSTRAP,
                json.dumps(import_paths),
                dedent(code),
            ],
            input=json.dumps(payload or {}, ensure_ascii=False),
            encoding="utf-8",
            capture_output=True,
            cwd=directory,
            env=environment,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert not completed.stderr, completed.stderr
        return json.loads(completed.stdout)

    return run


@pytest.mark.parametrize(
    "content",
    [
        "English persistence check",
        "कल DBMS पढ़ना है और मेरी किताब सुरक्षित रखनी है।",
        "Aaj VoiceOS project में Python aur Gemini इस्तेमाल karna hai.",
        "'); DROP TABLE notes; -- SELECT * FROM notes;",
    ],
    ids=["english", "hindi", "hinglish", "sql-like-content"],
)
def test_create_update_retrieve_delete_across_three_processes(run_process, tmp_path, content):
    database_path = tmp_path / "shared" / "notes.db"
    identifier = str(uuid4())
    original = f"{identifier} original {content}"
    updated = f"{identifier} updated {content}"
    created = run_process(
        """
        import os
        from actions.notes import create_note, update_note
        from core.schemas import CommandRequest
        from storage.database import initialize_database

        initialize_database()
        result = create_note(CommandRequest(
            intent="create_note", entities={"content": payload["original"]}, confidence=0.99,
        ))
        assert result.status == "success"
        note_id = result.data["note_id"]
        assert update_note(note_id, payload["updated"]).status == "success"
        print(json.dumps({"note_id": note_id, "pid": os.getpid(), "python": sys.executable}))
        """,
        database_path=database_path,
        payload={"original": original, "updated": updated},
        cwd=tmp_path / "process_a",
    )
    assert created["pid"] != os.getpid()
    assert created["python"] == sys.executable
    assert database_path.is_file()

    retrieved = run_process(
        """
        from contextlib import closing
        from actions.notes import delete_note, get_note, get_notes
        from storage.database import get_connection, initialize_database

        initialize_database()
        note = get_note(payload["note_id"])
        assert note.status == "success"
        listing = get_notes()
        assert listing.status == "success"
        with closing(get_connection()) as connection:
            actual_path = connection.execute("PRAGMA database_list").fetchone()["file"]
            assert connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 1
        deleted = delete_note(payload["note_id"], expected_content=payload["updated"])
        print(json.dumps({
            "content": note.data["content"], "listing": listing.message,
            "delete_status": deleted.status, "database_path": actual_path,
        }, ensure_ascii=False))
        """,
        database_path=database_path,
        payload={"note_id": created["note_id"], "updated": updated},
        cwd=tmp_path / "process_b",
    )
    assert retrieved["content"] == updated
    assert updated in retrieved["listing"]
    assert original not in retrieved["listing"]
    assert retrieved["delete_status"] == "success"
    assert Path(retrieved["database_path"]).resolve() == database_path.resolve()

    absent = run_process(
        """
        from contextlib import closing
        from actions.notes import get_note, get_notes
        from storage.database import get_connection, initialize_database

        initialize_database()
        note = get_note(payload["note_id"])
        with closing(get_connection()) as connection:
            count = connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        print(json.dumps({
            "lookup_status": note.status, "lookup_message": note.message,
            "listing": get_notes().message, "count": count,
        }))
        """,
        database_path=database_path,
        payload={"note_id": created["note_id"]},
        cwd=tmp_path / "process_c",
    )
    assert absent["lookup_status"] == "error"
    assert absent["lookup_message"] == f"No note found with ID {created['note_id']}."
    assert absent["count"] == 0
    assert absent["listing"] == "You don't have any saved notes yet."


APPLICATION_SCRIPT = """
import builtins
import io
from contextlib import redirect_stderr, redirect_stdout

import main
import nlu.service

main.load_dotenv = lambda: None
commands = iter(payload["commands"])
responses = iter(payload["responses"])
nlu.service.generate_response = lambda prompt: json.dumps(next(responses))

def read_input(prompt):
    if prompt.startswith("Approve with yes "):
        assert payload.get("approve_deletion") is True
        token = prompt.split(";", 1)[0].rsplit(" ", 1)[1]
        return "yes " + token
    return next(commands)

builtins.input = read_input
output, errors = io.StringIO(), io.StringIO()
with redirect_stdout(output), redirect_stderr(errors):
    main.main()
assert list(commands) == []
assert list(responses) == []
print(json.dumps({"output": output.getvalue(), "errors": errors.getvalue()}, ensure_ascii=False))
"""


def test_application_restarts_preserve_updates_and_confirmed_deletion(run_process, tmp_path):
    database_path = tmp_path / "application" / "notes.db"
    content = f"{uuid4()} कल VoiceOS project में DBMS पढ़ना hai।"
    first = run_process(
        APPLICATION_SCRIPT,
        database_path=database_path,
        payload={
            "commands": ["Create a note", "Update note 1", "exit"],
            "responses": [
                {
                    "intent": "create_note",
                    "entities": {"content": "Original content"},
                    "confidence": 0.99,
                },
                {
                    "intent": "update_note",
                    "entities": {"note_id": "1", "content": content},
                    "confidence": 0.99,
                },
            ],
        },
        cwd=tmp_path / "application_a",
    )
    assert "Your note has been saved. [success]" in first["output"]
    assert "Note 1 updated successfully. [success]" in first["output"]

    second = run_process(
        APPLICATION_SCRIPT,
        database_path=database_path,
        payload={
            "commands": ["Show my notes", "Delete note 1", "exit"],
            "responses": [
                {"intent": "get_notes", "entities": {}, "confidence": 0.99},
                {"intent": "delete_note", "entities": {"note_id": "1"}, "confidence": 0.99},
            ],
            "approve_deletion": True,
        },
        cwd=tmp_path / "application_b",
    )
    assert content in second["output"]
    assert "Note 1 deleted successfully. [success]" in second["output"]

    third = run_process(
        APPLICATION_SCRIPT,
        database_path=database_path,
        payload={
            "commands": ["Show my notes", "exit"],
            "responses": [{"intent": "get_notes", "entities": {}, "confidence": 0.99}],
        },
        cwd=tmp_path / "application_c",
    )
    assert "You don't have any saved notes yet. [success]" in third["output"]
    assert content not in third["output"]
    for result in (first, second, third):
        assert "VoiceOS shutting down." in result["output"]
        assert result["errors"] == ""


def test_repeated_initialization_preserves_notes_and_schema(run_process, tmp_path):
    database_path = tmp_path / "repeated.db"
    result = run_process(
        """
        from contextlib import closing
        from actions.notes import create_note, get_note
        from core.schemas import CommandRequest
        from storage.database import get_connection, initialize_database

        initialize_database()
        note = create_note(CommandRequest(
            intent="create_note", entities={"content": "Keep this note"}, confidence=0.99,
        ))
        assert note.status == "success"
        with closing(get_connection()) as connection:
            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'notes'"
            ).fetchone()[0]
        for _ in range(5):
            initialize_database()
        with closing(get_connection()) as connection:
            assert connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'notes'"
            ).fetchone()[0] == schema
        print(json.dumps(get_note(note.data["note_id"]).model_dump()))
        """,
        database_path=database_path,
    )
    assert result["status"] == "success"
    assert result["data"]["content"] == "Keep this note"


def test_independent_connections_commit_and_rollback_across_processes(run_process, tmp_path):
    database_path = tmp_path / "connections.db"
    result = run_process(
        """
        from contextlib import closing
        from storage.database import get_connection, initialize_database

        initialize_database()
        with closing(get_connection()) as first, closing(get_connection()) as second:
            assert first is not second
            cursor = first.execute("INSERT INTO notes (content) VALUES (?)", ("Committed हिंदी",))
            note_id = cursor.lastrowid
            assert second.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
            first.commit()
            assert second.execute("SELECT content FROM notes").fetchone()[0] == "Committed हिंदी"
            second.execute(
                "UPDATE notes SET content = ? WHERE id = ?", ("Updated DBMS हिंदी", note_id),
            )
            second.commit()
            assert first.execute("SELECT content FROM notes").fetchone()[0] == "Updated DBMS हिंदी"
        with closing(get_connection()) as uncommitted:
            uncommitted.execute("INSERT INTO notes (content) VALUES (?)", ("Must roll back",))
        print(json.dumps({"note_id": note_id}))
        """,
        database_path=database_path,
    )
    reopened = run_process(
        """
        from contextlib import closing
        from actions.notes import get_note
        from storage.database import get_connection

        with closing(get_connection()) as connection:
            count = connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        print(json.dumps({"note": get_note(payload["note_id"]).model_dump(), "count": count}))
        """,
        database_path=database_path,
        payload=result,
    )
    assert reopened["count"] == 1
    assert reopened["note"]["status"] == "success"
    assert reopened["note"]["data"]["content"] == "Updated DBMS हिंदी"


def test_default_path_is_anchored_to_source_across_working_directories(run_process, tmp_path):
    code = """
    from storage.database import DATABASE_PATH
    print(json.dumps({"path": str(DATABASE_PATH), "absolute": DATABASE_PATH.is_absolute()}))
    """
    for name in ("first_directory", "second_directory"):
        result = run_process(code, database_path=tmp_path / "unused.db", cwd=tmp_path / name)
        assert result["absolute"] is True
        assert Path(result["path"]) == PROJECT_ROOT / "storage" / "voiceos.db"
    assert not (tmp_path / "unused.db").exists()


def test_relative_override_depends_on_working_directory(run_process, tmp_path):
    relative_path = Path("relative") / "notes.db"
    first_directory = tmp_path / "first"
    code = """
    from contextlib import closing
    from storage.database import get_connection, initialize_database

    initialize_database()
    with closing(get_connection()) as connection:
        before = connection.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        connection.execute("INSERT INTO notes (content) VALUES (?)", ("Relative path check",))
        connection.commit()
        path = connection.execute("PRAGMA database_list").fetchone()["file"]
    print(json.dumps({"before": before, "path": path}))
    """
    first = run_process(code, database_path=relative_path, cwd=first_directory)
    second = run_process(code, database_path=relative_path, cwd=first_directory)
    third = run_process(code, database_path=relative_path, cwd=tmp_path / "second")
    assert (first["before"], second["before"], third["before"]) == (0, 1, 0)
    assert first["path"] == second["path"]
    assert Path(first["path"]) == first_directory / relative_path
    assert Path(third["path"]) == tmp_path / "second" / relative_path


def test_missing_parent_directories_are_created(run_process, tmp_path):
    database_path = tmp_path / "new" / "nested" / "notes.db"
    assert not database_path.parent.exists()
    result = run_process(
        """
        from actions.notes import get_notes
        from storage.database import initialize_database

        initialize_database()
        print(json.dumps(get_notes().model_dump()))
        """,
        database_path=database_path,
    )
    assert database_path.is_file()
    assert result["status"] == "success"


@pytest.mark.parametrize("invalid_path", ["directory", "parent-is-file"])
def test_invalid_database_paths_shutdown_safely(run_process, tmp_path, invalid_path):
    if invalid_path == "directory":
        database_path = tmp_path / "directory.db"
        database_path.mkdir()
    else:
        parent = tmp_path / "parent-file"
        parent.write_text("Temporary test sentinel", encoding="utf-8")
        database_path = parent / "notes.db"
    result = run_process(
        APPLICATION_SCRIPT,
        database_path=database_path,
        payload={"commands": [], "responses": []},
    )
    assert "Application stopped. Check local configuration and storage." in result["errors"]
    assert "VoiceOS shutting down." in result["output"]
    assert "Welcome to VoiceOS" not in result["output"]
    assert "Traceback" not in result["errors"]
    assert str(database_path) not in result["errors"]
    if invalid_path == "directory":
        assert list(database_path.iterdir()) == []
    else:
        assert parent.read_text(encoding="utf-8") == "Temporary test sentinel"
