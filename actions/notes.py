import logging
import sqlite3
from contextlib import closing

from core.schemas import ActionResult, CommandRequest
from storage.database import get_connection


def create_note(command: CommandRequest) -> ActionResult:

    content = command.entities.get("content", "")

    # Extra safety check
    if not content or not content.strip():
        return ActionResult(status="needs_clarification", message="Note content cannot be empty.")

    try:
        with closing(get_connection()) as connection:
            cursor = connection.execute(
                "INSERT INTO notes (content) VALUES (?)", (content.strip(),)
            )

            connection.commit()

            note_id = cursor.lastrowid

        return ActionResult(
            status="success", message="Your note has been saved.", data={"note_id": note_id}
        )

    except sqlite3.Error:
        logging.getLogger(__name__).error("Note storage operation failed.")
        return ActionResult(status="error", message="Unable to save the note.")


def get_notes() -> ActionResult:

    try:
        with closing(get_connection()) as connection:
            cursor = connection.execute(
                """
                SELECT id, content, created_at
                FROM notes
                ORDER BY id DESC
                """
            )

            notes = cursor.fetchall()

        if not notes:
            return ActionResult(status="success", message="You don't have any saved notes yet.")

        # Convert database records into readable text
        notes_text = "\n".join(
            f"{note['id']}. {note['content']} (Created: {note['created_at']})" for note in notes
        )

        return ActionResult(status="success", message=f"Here are your saved notes:\n{notes_text}")

    except sqlite3.Error:
        logging.getLogger(__name__).error("Note storage operation failed.")
        return ActionResult(status="error", message="Unable to retrieve your notes.")


def delete_note(note_id: int, *, expected_content: str | None = None) -> ActionResult:

    if type(note_id) is not int or not 0 < note_id <= 9223372036854775807:
        return ActionResult(status="rejected", message="Invalid note ID.")

    try:
        with closing(get_connection()) as connection:
            if expected_content is None:
                cursor = connection.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            else:
                cursor = connection.execute(
                    "DELETE FROM notes WHERE id = ? AND content = ?",
                    (note_id, expected_content),
                )

            connection.commit()

            if cursor.rowcount == 0:
                return ActionResult(
                    status="error",
                    message=f"No note found with ID {note_id}, or it changed. Please retry.",
                )

        return ActionResult(status="success", message=f"Note {note_id} deleted successfully.")

    except sqlite3.Error:
        logging.getLogger(__name__).error("Note storage operation failed.")
        return ActionResult(status="error", message="Unable to delete the note.")


def update_note(note_id: int, content: str) -> ActionResult:

    if type(note_id) is not int or not 0 < note_id <= 9223372036854775807:
        return ActionResult(status="rejected", message="Invalid note ID.")

    if not content or not content.strip():
        return ActionResult(
            status="needs_clarification", message="Updated note content cannot be empty."
        )

    try:
        with closing(get_connection()) as connection:
            cursor = connection.execute(
                """
                UPDATE notes
                SET content = ?
                WHERE id = ?
                """,
                (content.strip(), note_id),
            )

            connection.commit()

            if cursor.rowcount == 0:
                return ActionResult(
                    status="error",
                    message=f"No note found with ID {note_id}, or it changed. Please retry.",
                )

        return ActionResult(status="success", message=f"Note {note_id} updated successfully.")

    except sqlite3.Error:
        logging.getLogger(__name__).error("Note storage operation failed.")
        return ActionResult(status="error", message="Unable to update the note.")


def get_note(note_id: int) -> ActionResult:
    if type(note_id) is not int or not 0 < note_id <= 9223372036854775807:
        return ActionResult(status="rejected", message="Invalid note ID.")
    try:
        with closing(get_connection()) as connection:
            note = connection.execute(
                "SELECT id, content FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
        if note is None:
            return ActionResult(status="error", message=f"No note found with ID {note_id}.")
        return ActionResult(
            status="success",
            message="Note found.",
            data={"note_id": note["id"], "content": note["content"]},
        )
    except sqlite3.Error:
        logging.getLogger(__name__).error("Note lookup failed.")
        return ActionResult(status="error", message="Unable to retrieve the note.")
