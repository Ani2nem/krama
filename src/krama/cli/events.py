"""Building typed domain events from command-line arguments.

A thin, deliberately boring layer. Its whole job is to turn a dispatcher's shorthand
into a validated event, and to refuse anything it cannot construct - the event log is
the source of truth, so a malformed entry is worse than a rejected command.

This is also the seam the triage agent will sit behind later: it will produce exactly
these typed events from a sentence someone typed, and everything downstream stays the
same. Building the deterministic path first means the agent has a target to hit
rather than a shape to invent.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime

from krama.domain.events import (
    Event,
    JobCancelled,
    JobCompleted,
    JobConfirmed,
    JobDispatched,
    JobOverran,
    ShiftExtensionWithdrawn,
    TrafficDelay,
    VanRestored,
    VanUnavailable,
    WorkerRestored,
    WorkerUnavailable,
)
from krama.domain.models import TimeWindow


class EventArgumentError(ValueError):
    """The command line did not carry enough to build a valid event."""


def _envelope(at: datetime, dispatch_id: str) -> dict[str, object]:
    return {
        "event_id": f"e-{uuid.uuid4().hex[:10]}",
        # occurred_at and recorded_at both default to the given moment here; the
        # distinction earns its keep when a real channel reports something late.
        "occurred_at": at,
        "recorded_at": at,
        "dispatch_id": dispatch_id,
        "actor": "dispatcher:cli",
    }


def _require[T](value: T | None, name: str) -> T:
    if value is None:
        raise EventArgumentError(f"{name} is required for this event")
    return value


def build_event(
    kind: str,
    target: str | None,
    *,
    at: datetime,
    dispatch_id: str,
    until: datetime | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    minutes: int | None = None,
    multiplier: float | None = None,
    commitment_cost: float = 0.0,
    reason: str = "",
) -> Event:
    """Construct one event, or raise with a message a human can act on."""
    base = _envelope(at, dispatch_id)

    builders: dict[str, Callable[[], Event]] = {
        "van-unavailable": lambda: VanUnavailable(
            **base,
            van_id=_require(target, "a van id"),
            from_time=max(window_start, at) if window_start else at,
            until_time=until or window_end,
            reason=reason,
        ),
        "van-restored": lambda: VanRestored(
            **base,
            van_id=_require(target, "a van id"),
            window_start=window_start,
            window_end=window_end,
        ),
        "worker-unavailable": lambda: WorkerUnavailable(
            **base,
            worker_id=_require(target, "a worker id"),
            # A window may place the absence on a future day, but never before this
            # moment: backdating an outage would retroactively invalidate work the
            # person already did today.
            from_time=max(window_start, at) if window_start else at,
            until_time=until or window_end,
            reason=reason,
        ),
        "worker-restored": lambda: WorkerRestored(
            **base,
            worker_id=_require(target, "a worker id"),
            window_start=window_start,
            window_end=window_end,
        ),
        "extension-withdrawn": lambda: ShiftExtensionWithdrawn(
            **base,
            worker_id=_require(target, "a worker id"),
            # The agreement being taken back is a fact about a DAY; the window
            # start names it, or today when none is given.
            on_date=(window_start or at).date(),
            reason=reason,
        ),
        "job-dispatched": lambda: JobDispatched(**base, job_id=_require(target, "a job id")),
        "job-cancelled": lambda: JobCancelled(
            **base, job_id=_require(target, "a job id"), reason=reason
        ),
        "job-overran": lambda: JobOverran(
            **base,
            job_id=_require(target, "a job id"),
            extra_minutes=_require(minutes, "--minutes"),
        ),
        "job-completed": lambda: JobCompleted(
            **base,
            job_id=_require(target, "a job id"),
            actual_duration_min=_require(minutes, "--minutes"),
        ),
        "job-confirmed": lambda: JobConfirmed(
            **base,
            job_id=_require(target, "a job id"),
            # A promise is a window, so both ends are required rather than guessed.
            # How wide the customer's window was is exactly what the autonomy policy
            # later depends on to decide whether a change needs a phone call.
            window=TimeWindow(
                start=_require(window_start, "--window-start"),
                end=_require(window_end, "--window-end"),
            ),
            commitment_cost=commitment_cost,
        ),
        "traffic-delay": lambda: TrafficDelay(
            **base,
            multiplier=_require(multiplier, "--multiplier"),
            from_time=at,
            until_time=until,
            note=reason,
        ),
    }

    if kind not in builders:
        known = ", ".join(sorted(builders))
        raise EventArgumentError(f"unknown event {kind!r}; known events: {known}")
    return builders[kind]()


KINDS: tuple[str, ...] = (
    "van-unavailable",
    "van-restored",
    "worker-unavailable",
    "worker-restored",
    "job-dispatched",
    "job-confirmed",
    "job-cancelled",
    "job-overran",
    "job-completed",
    "traffic-delay",
)
