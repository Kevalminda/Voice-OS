import logging
from collections.abc import Callable

from core.action_router import route_command
from core.schemas import ActionResult
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
    except Exception:
        logger.exception("Command processing failed.")
        return ActionResult(
            status="error", message="Unable to process your command. Please retry."
        )


def process_text_command(
    user_text: str,
    confirm_action: Callable[[ActionResult], bool] | None = None,
) -> ActionResult:
    return _process_safely(
        lambda: route_command(understand_command(user_text), confirm_action=confirm_action)
    )
