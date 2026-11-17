"""Single-use synchronous deletion confirmation, independent of NLU and audio."""

import math
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from core.schemas import ActionResult

_AFFIRMATIVE = {"yes", "y", "हाँ", "हां", "haan", "han"}
_NEGATIVE = {"no", "n", "nahi", "nahin", "नहीं", "ना"}
_CANCEL = {"cancel", "/cancel", "रद्द करो", "cancel karo"}


class ConfirmationOutcome(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass(frozen=True)
class ConfirmationResult:
    outcome: ConfirmationOutcome
    edit_requested: bool = False


@dataclass(frozen=True)
class _DeletionBinding:
    note_id: int
    original_content: str = field(repr=False)
    preview: str = field(repr=False)


@dataclass
class _PendingConfirmation:
    binding: _DeletionBinding = field(repr=False)
    token: str = field(repr=False)
    expires_at: float
    invalid_replies: int = 0


class ConfirmationSession:
    """One active deletion per synchronous session; no timers or database access.

    The caller must check a reply after input returns. A token is exact and
    case-sensitive; only response words are case-folded. Inject token_factory
    and clock for offline tests. Production tokens use secrets.token_hex.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[], str] | None = None,
    ):
        if type(max_retries) is not int or max_retries < 1:
            raise ValueError("max_retries must be a positive integer.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or not callable(clock)
            or (token_factory is not None and not callable(token_factory))
        ):
            raise ValueError("A finite positive timeout and callable sources are required.")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._clock = clock
        self._token_factory = token_factory or (lambda: secrets.token_hex(8).upper())
        self._issued_tokens: set[str] = set()
        self._pending: _PendingConfirmation | None = None

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def token(self) -> str | None:
        return self._pending.token if self._pending else None

    @property
    def remaining_retries(self) -> int:
        return self._max_retries - self._pending.invalid_replies if self._pending else 0

    def begin(self, pending: ActionResult) -> str:
        # Replacement always invalidates the earlier request, even on failure.
        self.cancel()
        binding = self._binding(pending)
        token = self._token_factory()
        if (
            not isinstance(token, str)
            or not token
            or not token.isascii()
            or not token.isalnum()
            or token in self._issued_tokens
        ):
            raise ValueError("The token source must produce a fresh ASCII alphanumeric token.")
        expires_at = self._now() + self._timeout
        if not math.isfinite(expires_at):
            raise ValueError("Invalid confirmation deadline.")
        self._issued_tokens.add(token)
        self._pending = _PendingConfirmation(binding, token, expires_at)
        return token

    def check_pending(self, pending: ActionResult) -> ConfirmationResult | None:
        """Check binding/expiry without spending an attempt or extending the deadline."""
        current = self._pending
        if current is None:
            return ConfirmationResult(ConfirmationOutcome.INVALID)
        try:
            if self._now() >= current.expires_at:
                return self._finish(ConfirmationResult(ConfirmationOutcome.EXPIRED))
            if self._binding(pending) != current.binding:
                return self._finish(ConfirmationResult(ConfirmationOutcome.INVALID))
        except (ValueError, TypeError):
            return self._finish(ConfirmationResult(ConfirmationOutcome.INVALID))
        return None

    def respond(self, response: str, pending: ActionResult) -> ConfirmationResult:
        invalid = self.check_pending(pending)
        if invalid is not None:
            return invalid
        current = self._pending
        result = self._parse_reply(response, current.token)
        if result.outcome is ConfirmationOutcome.INVALID:
            current.invalid_replies += 1
            if current.invalid_replies < self._max_retries:
                return result
            return self.cancel()
        if result.outcome is ConfirmationOutcome.APPROVED:
            try:
                if self._now() >= current.expires_at:
                    return self._finish(ConfirmationResult(ConfirmationOutcome.EXPIRED))
            except (ValueError, TypeError):
                return self._finish(ConfirmationResult(ConfirmationOutcome.INVALID))
        # Approval is consumed before returning to the router's deletion call.
        return self._finish(result)

    def cancel(self) -> ConfirmationResult:
        return self._finish(ConfirmationResult(ConfirmationOutcome.CANCELLED))

    def _finish(self, result):
        self._pending = None
        return result

    def _now(self):
        value = self._clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("The confirmation clock must return a finite number.")
        return value

    @staticmethod
    def _binding(pending):
        if (
            not isinstance(pending, ActionResult)
            or pending.status != "needs_confirmation"
            or not isinstance(pending.data, dict)
        ):
            raise ValueError("A pending deletion snapshot is required.")
        note_id = pending.data.get("note_id")
        content = pending.data.get("original_content")
        preview = pending.data.get("preview")
        if (
            type(note_id) is not int
            or not 0 < note_id <= 9223372036854775807
            or not isinstance(content, str)
            or not isinstance(preview, str)
            or preview != " ".join(content.split())[:120]
        ):
            raise ValueError("Invalid deletion snapshot.")
        return _DeletionBinding(note_id, content, preview)

    @staticmethod
    def _parse_reply(response, token):
        if not isinstance(response, str):
            return ConfirmationResult(ConfirmationOutcome.INVALID)
        parts = response.split()
        phrase = " ".join(parts).casefold()
        if phrase in _NEGATIVE:
            return ConfirmationResult(ConfirmationOutcome.REJECTED)
        if phrase in _CANCEL:
            return ConfirmationResult(ConfirmationOutcome.CANCELLED)
        if phrase == "edit":
            return ConfirmationResult(ConfirmationOutcome.CANCELLED, edit_requested=True)
        if (
            len(parts) == 2
            and parts[0].casefold() in _AFFIRMATIVE
            and parts[1].isascii()
            and secrets.compare_digest(parts[1], token)
        ):
            return ConfirmationResult(ConfirmationOutcome.APPROVED)
        return ConfirmationResult(ConfirmationOutcome.INVALID)
