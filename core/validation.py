from core.schemas import ActionResult, CommandRequest, Intent

CONFIDENCE_THRESHOLD = 0.65


REQUIRED_FIELDS = {
    Intent.CREATE_NOTE: ["content"],
    Intent.GET_NOTES: [],
    Intent.CREATE_REMINDER: ["task", "time"],
    Intent.WEB_SEARCH: ["query"],
    Intent.DELETE_NOTE: ["note_id"],
    Intent.UPDATE_NOTE: ["note_id", "content"],
}


def validate_command(command: CommandRequest) -> ActionResult:

    # 1. Reject unknown commands
    if command.intent == Intent.UNKNOWN:
        return ActionResult(status="rejected", message="Unknown command. Cannot execute.")

    # 2. Check confidence
    if command.confidence < CONFIDENCE_THRESHOLD:
        return ActionResult(
            status="needs_clarification",
            message="I am not confident enough to execute this command.",
        )

    # 3. Check required information
    required = REQUIRED_FIELDS.get(command.intent)

    if required is None:
        return ActionResult(status="rejected", message="This intent is not supported.")

    # 4. Identify missing or empty fields
    missing = [
        field
        for field in required
        if not command.entities.get(field) or not command.entities[field].strip()
    ]

    if missing:
        return ActionResult(
            status="needs_clarification",
            message="Some required information is missing.",
            data={"missing_fields": ", ".join(missing)},
        )

    if command.missing_fields:
        return ActionResult(
            status="needs_clarification", message="Please supply the missing command details."
        )

    if command.intent in (Intent.DELETE_NOTE, Intent.UPDATE_NOTE):
        value = command.entities["note_id"]
        if (
            not value.isascii()
            or not value.isdecimal()
            or len(value) > 19
            or not 0 < int(value) <= 9223372036854775807
        ):
            return ActionResult(status="rejected", message="Invalid note ID.")

    # 5. Validation successful
    return ActionResult(status="success", message="Command passed validation.")
