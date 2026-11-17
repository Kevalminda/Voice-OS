from collections.abc import Callable

from actions.notes import create_note, delete_note, get_note, get_notes, update_note
from core.schemas import ActionResult, CommandRequest, Intent
from core.validation import validate_command


def route_command(
    command: CommandRequest,
    confirm_action: Callable[[ActionResult], bool] | None = None,
) -> ActionResult:
    # Revalidate even an existing model: callers may have mutated its fields.
    command = CommandRequest.model_validate(command.model_dump())
    validation = validate_command(command)
    if validation.status != "success":
        return validation
    if command.intent == Intent.CREATE_NOTE:
        return create_note(command)
    if command.intent == Intent.GET_NOTES:
        return get_notes()
    if command.intent == Intent.UPDATE_NOTE:
        return update_note(int(command.entities["note_id"]), command.entities["content"])
    if command.intent == Intent.DELETE_NOTE:
        note_id = int(command.entities["note_id"])
        note = get_note(note_id)
        if note.status != "success":
            return note
        content = note.data["content"]
        preview = " ".join(content.split())[:120]
        pending = ActionResult(
            status="needs_confirmation",
            message=f"Delete note {note_id}: {preview}",
            data={"note_id": note_id, "preview": preview},
        )
        if confirm_action is None:
            return pending
        if confirm_action(pending) is not True:
            return ActionResult(status="rejected", message="Note deletion cancelled.")
        # Compare the original content atomically to avoid deleting an edited note.
        return delete_note(note_id, expected_content=content)
    return ActionResult(status="rejected", message="This action is not implemented yet.")
