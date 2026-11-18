import logging
from collections.abc import Callable
from contextlib import nullcontext

from core.action_router import route_command
from core.schemas import ActionResult
from core.timing import InteractionTiming
from nlu.service import understand_command

logger = logging.getLogger(__name__)


def _process_safely(operation: Callable[[], ActionResult]) -> ActionResult:
    """Share the existing error boundary with contextual command processing."""
    try:
        result = operation()
        logger.debug("Command completed: %s", result.status)
        return result
    except ValueError:
        return ActionResult(
            status="rejected", message="Invalid command or configuration. Please check and retry."
        )
    except Exception as error:
        logger.error("Command processing failed. Error type: %s", type(error).__name__)
        return ActionResult(
            status="error", message="Unable to process your command. Please retry."
        )


def process_text_command(
    user_text: str,
    confirm_action: Callable[[ActionResult], bool] | None = None,
    *,
    timing: InteractionTiming | None = None,
) -> ActionResult:
    return _process_safely(
        lambda: _route_command(_understand_command(user_text, timing), confirm_action, timing)
    )


def _understand_command(user_text, timing=None):
    with timing.measure("nlu_duration") if timing is not None else nullcontext():
        return understand_command(user_text)


def _route_command(command, confirm_action=None, timing=None):
    callback = confirm_action
    if timing is not None and confirm_action is not None:
        def callback(pending):
            with timing.measure("confirmation_total_duration"):
                return confirm_action(pending)

    measurement = (
        timing.measure("action_execution_duration", exclude="confirmation_total_duration")
        if timing is not None else nullcontext()
    )
    with measurement:
        return route_command(command, confirm_action=callback)
