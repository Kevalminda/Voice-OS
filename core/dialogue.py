"""In-memory missing-entity collection; deletion approval stays in the existing router."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from core import command_processor
from core.schemas import ActionResult, CommandRequest, Intent
from core.timing import InteractionTiming
from core.validation import CONFIDENCE_THRESHOLD, REQUIRED_FIELDS, validate_command

_SUPPORTED_INTENTS = {Intent.CREATE_NOTE, Intent.GET_NOTES, Intent.UPDATE_NOTE, Intent.DELETE_NOTE}
_CANCEL = {"/cancel", "cancel", "रद्द करो", "cancel karo"}
_AMBIGUOUS = {
    "yes",
    "y",
    "no",
    "n",
    "ok",
    "okay",
    "edit",
    "maybe",
    "not sure",
    "i don't know",
    "yes or no",
    "yes/no",
    "हाँ",
    "हां",
    "haan",
    "han",
    "नहीं",
    "nahi",
    "nahin",
    "पता नहीं",
    "pata nahi",
    "हाँ या नहीं",
    "haan ya nahi",
}


@dataclass
class _PendingClarification:
    original: CommandRequest
    command: CommandRequest
    requested_field: str
    expires_at: float
    invalid_replies: int = 0


class DialogueSession:
    """Collect one field per turn. /cancel cancels; /new <command> explicitly replaces.

    Content replies are literal text, never new commands or deletion approval.
    Expiry is checked on the next processed turn, without extending the deadline.
    This object belongs to one synchronous application session, not shared threads.
    """

    def __init__(
        self,
        *,
        max_retries: int = 3,
        timeout_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        measurement_clock: Callable[[], float] = time.perf_counter,
    ):
        if type(max_retries) is not int or max_retries < 1:
            raise ValueError("max_retries must be a positive integer.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or not callable(clock)
        ):
            raise ValueError("A positive finite timeout and callable clock are required.")
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._clock = clock
        self._pending: _PendingClarification | None = None
        self._measurement_clock = measurement_clock
        self.timing = InteractionTiming(clock=measurement_clock)
        self._last_timing = None

    @property
    def last_timing(self):
        return self._last_timing.snapshot() if self._last_timing is not None else None

    def prepare_turn(self):
        if self.timing.finished:
            self.timing = InteractionTiming(clock=self._measurement_clock)
        return self.timing

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def requested_field(self) -> str | None:
        return self._pending.requested_field if self._pending else None

    @property
    def pending_command(self) -> CommandRequest | None:
        return self._pending.command.model_copy(deep=True) if self._pending else None

    @property
    def original_command(self) -> CommandRequest | None:
        return self._pending.original.model_copy(deep=True) if self._pending else None

    def cancel(self) -> ActionResult:
        self._pending = None
        if self.timing.started and not self.timing.finished:
            self.timing.finish()
            self._last_timing = self.timing
        return ActionResult(status="rejected", message="Clarification cancelled. No action taken.")

    def process_text_command(
        self,
        user_text: str,
        confirm_action: Callable[[ActionResult], bool] | None = None,
    ) -> ActionResult:
        self.prepare_turn().resume()
        try:
            result = command_processor._process_safely(
                lambda: self._process_turn(user_text, confirm_action)
            )
        except BaseException:
            self._pending = None
            self.timing.finish()
            self._last_timing = self.timing
            raise
        if result.status != "needs_clarification":
            self._pending = None
        if self._pending is not None:
            self.timing.wait_for_reply()
        else:
            self.timing.finish()
            self._last_timing = self.timing
        return result

    def _process_turn(self, user_text, confirm_action):
        if not isinstance(user_text, str):
            return self._restate("The answer must be text.")
        if self._pending and self._clock() >= self._pending.expires_at:
            return self._restate("The pending clarification expired.")
        text = user_text.strip()
        if " ".join(text.casefold().split()) in _CANCEL:
            return self.cancel()
        parts = text.split(maxsplit=1)
        if parts and parts[0].casefold() == "/new":
            if self._pending is not None:
                self.timing.finish()
                self._last_timing = self.timing
                self.timing = InteractionTiming(clock=self._measurement_clock)
                self.timing.resume()
            self._pending = None
            if len(parts) != 2:
                return self._restate("Provide a complete command after /new.")
            text = parts[1]
        elif self._pending:
            return self._collect(text, confirm_action)
        if not text:
            return self._restate("The command cannot be empty.")
        command = command_processor._understand_command(text, self.timing)
        command = CommandRequest.model_validate(command.model_dump())
        validation = validate_command(command)
        if command.intent not in _SUPPORTED_INTENTS:
            return self._restate("This command is not supported.")
        if validation.status == "success":
            return command_processor._route_command(command, confirm_action, self.timing)
        fields = self._identified_fields(command, validation)
        if not fields:
            return self._unidentified(validation)
        self._pending = _PendingClarification(
            original=command.model_copy(deep=True),
            command=command,
            requested_field=fields[0],
            expires_at=self._clock() + self._timeout,
        )
        return self._question(fields)

    @staticmethod
    def _identified_fields(command, validation):
        # Do not infer a field from English messages or generic model uncertainty.
        metadata = validation.data.get("missing_fields")
        if (
            validation.status != "needs_clarification"
            or command.confidence < CONFIDENCE_THRESHOLD
            or not isinstance(metadata, str)
            or not metadata
        ):
            return None
        fields = [field.strip() for field in metadata.split(",")]
        actual = [
            field for field in REQUIRED_FIELDS[command.intent] if not command.entities.get(field)
        ]
        if (
            len(fields) != len(set(fields))
            or set(fields) != set(actual)
            or not set(command.missing_fields).issubset(fields)
        ):
            return None
        return fields

    def _collect(self, text, confirm_action):
        pending = self._pending
        if pending.requested_field == "note_id":
            valid = (
                text.isascii()
                and text.isdecimal()
                and len(text) <= 19
                and 0 < int(text) <= 9223372036854775807
            )
        else:
            valid = (
                any(character.isalnum() for character in text)
                and " ".join(text.casefold().split()) not in _AMBIGUOUS
            )
        if not valid:
            pending.invalid_replies += 1
            if pending.invalid_replies >= self._max_retries:
                return self._restate("The clarification retry limit was reached.")
            return self._question(prefix="That answer is ambiguous or invalid. ")
        data = pending.command.model_dump()
        data["entities"][pending.requested_field] = text
        data["missing_fields"] = [
            field for field in data["missing_fields"] if field != pending.requested_field
        ]
        command = CommandRequest.model_validate(data)
        validation = validate_command(command)
        if validation.status == "success":
            # Consume before entering the router or its synchronous deletion callback.
            self._pending = None
            return command_processor._route_command(command, confirm_action, self.timing)
        fields = self._identified_fields(command, validation)
        if not fields:
            return self._unidentified(validation)
        pending.command = command
        pending.requested_field = fields[0]
        return self._question(fields)

    def _question(self, fields=None, prefix=""):
        pending = self._pending
        if fields is None:
            fields = [
                field
                for field in REQUIRED_FIELDS[pending.command.intent]
                if not pending.command.entities.get(field)
            ]
        question = (
            "Which note ID? Reply with one positive ASCII decimal ID."
            if pending.requested_field == "note_id"
            else "What note content should I use? Reply with the content only."
        )
        return ActionResult(
            status="needs_clarification",
            message=f"{prefix}{question} Use /cancel to cancel or /new <command> to replace.",
            data={
                "requested_field": pending.requested_field,
                "missing_fields": ", ".join(fields),
                "retries_remaining": self._max_retries - pending.invalid_replies,
            },
        )

    def _unidentified(self, validation):
        self._pending = None
        if validation.status == "needs_clarification":
            return ActionResult(
                status="needs_clarification",
                message=f"{validation.message} Please restate the complete command.",
            )
        return validation

    def _restate(self, reason):
        self._pending = None
        return ActionResult(
            status="rejected", message=f"{reason} Please restate the complete command."
        )
