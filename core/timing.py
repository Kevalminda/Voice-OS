"""In-memory interaction measurements, separate from requests and authorization.

Durations are seconds. Skipped phases are zero, unavailable measurements are None,
and executed phases retain elapsed time even when they fail. Command STT fields
come from its existing perf_counter measurements. Confirmation transcription
covers the complete transcribe call. Action execution covers router/storage work
with synchronous confirmation excluded. Totals include human waits and overlap
their component phases, so components must not be summed to reconstruct totals.
One ID covers a pending command across turns. An explicit replacement starts a
new measurement when recognized. Totals start at activation/text submission and
exclude model startup and initial activation waiting. Completed application
records are available through DEBUG logging and the optional in-memory collector.
"""

import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

DURATION_FIELDS = (
    "command_capture_duration",
    "transcription_duration",
    "normalization_duration",
    "transcript_review_duration",
    "nlu_duration",
    "clarification_wait_duration",
    "action_execution_duration",
    "total_interaction_duration",
    "confirmation_capture_duration",
    "confirmation_transcription_duration",
    "confirmation_review_duration",
    "keyboard_fallback_duration",
    "confirmation_total_duration",
)

current_timing = ContextVar("voiceos_interaction_timing", default=None)


class InteractionTiming:
    def __init__(self, *, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self.record = dict.fromkeys(DURATION_FIELDS, 0.0)
        self.record.update(
            interaction_id=uuid4().hex,
            confirmation_attempt_count=0,
            confirmation_outcome=None,
        )
        self.started = False
        self.finished = False
        self.published = False
        self._started_at = None
        self._waiting = False
        self._wait_started = None
        self._active = set()

    def now(self):
        try:
            value = self._clock()
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                return float(value)
        except Exception:
            pass
        return None

    def elapsed(self, started):
        ended = self.now()
        if started is None or ended is None or ended < started:
            return None
        return ended - started

    def add(self, field, duration):
        previous = self.record[field]
        if previous is None or duration is None:
            self.record[field] = None
        elif (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and (math.isfinite(duration) and duration >= 0)
        ):
            self.record[field] = previous + duration
        else:
            self.record[field] = None

    def resume(self):
        if not self.started:
            self.started = True
            self._started_at = self.now()
        if self._waiting:
            self.add("clarification_wait_duration", self.elapsed(self._wait_started))
            self._waiting = False

    def wait_for_reply(self, *, displayed=False):
        if not self._waiting or displayed:
            self._wait_started = self.now()
            self._waiting = True

    def finish(self):
        if self._waiting:
            self.resume()
        if self.started:
            self.record["total_interaction_duration"] = self.elapsed(self._started_at)
        self.finished = True

    def snapshot(self):
        result = self.record.copy()
        if self.started and not self.finished:
            result["total_interaction_duration"] = self.elapsed(self._started_at)
        return result

    def is_measuring(self, field):
        return field in self._active

    @contextmanager
    def measure(self, field, *, exclude=None):
        if field in self._active:
            yield
            return
        self.resume()
        started = self.now()
        excluded_before = self.record[exclude] if exclude else 0.0
        self._active.add(field)
        token = current_timing.set(self)
        try:
            yield
        finally:
            duration = self.elapsed(started)
            if exclude:
                excluded_after = self.record[exclude]
                if duration is None or excluded_before is None or excluded_after is None:
                    duration = None
                else:
                    duration = max(0.0, duration - (excluded_after - excluded_before))
            self.add(field, duration)
            self._active.remove(field)
            current_timing.reset(token)
