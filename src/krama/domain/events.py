"""The append-only event log.

World state is ``fold(events)``. Nothing else is authoritative. This is what lets
any scenario - a van breaking down at 10:40 on a Tuesday - replay byte-identically
in a test, which is the precondition for the entire eval layer.

Every event carries a ``dispatch_id``: the correlation ID minted the moment raw
text enters the system, propagated through agents, MCP calls, and the solver, so a
whole disruption is one trace end to end.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field

from krama.domain.models import (
    Frozen,
    Job,
    JobId,
    Location,
    TimeWindow,
    Van,
    VanId,
    Worker,
    WorkerId,
)


class EventBase(Frozen):
    """Common envelope.

    ``occurred_at`` is when the thing happened in the world; ``recorded_at`` is when
    we learned about it. They differ constantly in field service - a van breaks at
    10:40 and the dispatcher hears about it at 10:52 - and conflating them corrupts
    any replay that depends on what was knowable at a given moment.
    """

    event_id: str
    occurred_at: datetime
    recorded_at: datetime
    dispatch_id: str
    actor: str = "system"


# --------------------------------------------------------------------------- roster


class WorkerRegistered(EventBase):
    type: Literal["worker_registered"] = "worker_registered"
    worker: Worker


class VanRegistered(EventBase):
    type: Literal["van_registered"] = "van_registered"
    van: Van


# ------------------------------------------------------------------------- job life


class JobRequested(EventBase):
    type: Literal["job_requested"] = "job_requested"
    job: Job


class JobSlotOffered(EventBase):
    """Slots quoted to a customer on a call. Recorded even if they decline, because
    a declined cheap slot followed by an accepted expensive one is a real signal."""

    type: Literal["job_slot_offered"] = "job_slot_offered"
    job_id: JobId
    offered: tuple[TimeWindow, ...]
    marginal_costs: tuple[float, ...] = ()


class JobConfirmed(EventBase):
    """The customer was given a window. From here the window costs money to move."""

    type: Literal["job_confirmed"] = "job_confirmed"
    job_id: JobId
    window: TimeWindow
    commitment_cost: float = 0.0


class JobDispatched(EventBase):
    type: Literal["job_dispatched"] = "job_dispatched"
    job_id: JobId


class JobStarted(EventBase):
    type: Literal["job_started"] = "job_started"
    job_id: JobId


class JobCompleted(EventBase):
    type: Literal["job_completed"] = "job_completed"
    job_id: JobId
    actual_duration_min: int = Field(gt=0)


class JobOverran(EventBase):
    """In-progress job is taking longer than estimated. Distinct from completion -
    it arrives while the crew is still on site and the rest of the day is at risk."""

    type: Literal["job_overran"] = "job_overran"
    job_id: JobId
    extra_minutes: int = Field(gt=0)


class JobCancelled(EventBase):
    type: Literal["job_cancelled"] = "job_cancelled"
    job_id: JobId
    reason: str = ""


class CustomerRescheduled(EventBase):
    type: Literal["customer_rescheduled"] = "customer_rescheduled"
    job_id: JobId
    new_windows: tuple[TimeWindow, ...]


class JobDeferred(EventBase):
    """Job pushed out of the horizon. Increments ``deferral_count``, which raises its
    unserved penalty so the optimizer cannot quietly defer it forever."""

    type: Literal["job_deferred"] = "job_deferred"
    job_id: JobId


# ----------------------------------------------------------------------- resources


class WorkerUnavailable(EventBase):
    type: Literal["worker_unavailable"] = "worker_unavailable"
    worker_id: WorkerId
    from_time: datetime
    until_time: datetime | None = None
    reason: str = ""


class WorkerRestored(EventBase):
    """Back at work. With no window: from this moment on - whatever was recorded,
    however long it said it would last, the person standing in the doorway wins.
    With a window: back for just that stretch, carving it out of any outage that
    covered it - "he can do Wednesday after all" without cancelling the sick week."""

    type: Literal["worker_restored"] = "worker_restored"
    worker_id: WorkerId
    window_start: datetime | None = None
    window_end: datetime | None = None


class VanUnavailable(EventBase):
    type: Literal["van_unavailable"] = "van_unavailable"
    van_id: VanId
    from_time: datetime
    until_time: datetime | None = None
    reason: str = ""


class VanRestored(EventBase):
    type: Literal["van_restored"] = "van_restored"
    van_id: VanId
    window_start: datetime | None = None
    window_end: datetime | None = None


class TrafficDelay(EventBase):
    """A multiplier layered on top of the travel matrix.

    ``None`` on either endpoint means "any", so a region-wide slowdown is one event
    rather than a fan-out over every corridor pair.
    """

    type: Literal["traffic_delay"] = "traffic_delay"
    origin_geohash5: str | None = None
    dest_geohash5: str | None = None
    multiplier: float = Field(gt=0)
    from_time: datetime
    until_time: datetime | None = None
    note: str = ""


# ---------------------------------------------------------------------------- plan


class PlanCommitted(EventBase):
    type: Literal["plan_committed"] = "plan_committed"
    plan_id: str
    parent_id: str | None = None
    content_hash: str = ""


class PlanProposed(EventBase):
    type: Literal["plan_proposed"] = "plan_proposed"
    proposal_id: str
    candidate_plan_ids: tuple[str, ...]
    recommended_plan_id: str
    rationale: str = ""


class ProposalApproved(EventBase):
    """Dispatcher accepted a proposal. Doubles as a ground-truth eval label."""

    type: Literal["proposal_approved"] = "proposal_approved"
    proposal_id: str
    chosen_plan_id: str
    note: str = ""


class ProposalRejected(EventBase):
    type: Literal["proposal_rejected"] = "proposal_rejected"
    proposal_id: str
    note: str = ""


class OvertimeOffered(EventBase):
    """A booked job runs past shift, and the hours went up for grabs.

    The owner's policy: overtime is volunteered, not assigned. The solver proves the
    slot feasible and pencils somebody in, the crew gets first refusal on the extra
    money, and the schedule never waits on the group chat - if nobody claims by the
    deadline, the pencilled-in fitter stands.
    """

    type: Literal["overtime_offered"] = "overtime_offered"
    job_id: JobId
    on_date: date
    #: Everyone qualified, rostered and overtime-eligible that day - the people the
    #: text went to.
    offered_to: tuple[WorkerId, ...]
    #: Who the solver pencilled in; the fallback if nobody claims.
    fallback: WorkerId
    overtime_minutes: int
    claim_deadline: datetime


class RateCardChanged(EventBase):
    """The owner set real prices - the one edit that retires "estimated costs".

    Values land in the event log like every other fact, so the board's prices are
    auditable ("who changed the labour rate, and when") and survive any redeploy.
    Keys are PricingParams field names; unknown keys are refused upstream.
    """

    type: Literal["rate_card_changed"] = "rate_card_changed"
    entries: dict[str, float]


class DepotMoved(EventBase):
    """The shop's address changed - deliberately, loudly, and in one place.

    The depot was once wrong by four road miles and every route ever planned
    carried the error, so this is the most consequential coordinate in the system.
    It is guarded accordingly: the API that emits this requires the new address
    typed twice and geocoded to a building, and the fold rewrites every van's
    home in the same breath so no record can disagree with another.
    """

    type: Literal["depot_moved"] = "depot_moved"
    location: Location


class WorkerRemoved(EventBase):
    """Off the roster - left the company, not merely out sick. The log keeps that
    they existed; the world stops offering them work."""

    type: Literal["worker_removed"] = "worker_removed"
    worker_id: WorkerId


class VanRemoved(EventBase):
    type: Literal["van_removed"] = "van_removed"
    van_id: VanId


class ShiftExtended(EventBase):
    """A fitter agreed to work past their rostered day, once, for a named date.

    This is how "nobody certified can work those hours" becomes bookable: the
    dispatcher asked, the person said yes, and the agreement is recorded as a fact
    with a date on it. Scheduling honours it for that day only; pay does not change
    shape - every minute past the rostered end is overtime, exactly as if the
    standard allowance had covered it.
    """

    type: Literal["shift_extended"] = "shift_extended"
    worker_id: WorkerId
    on_date: date
    until_time: datetime
    reason: str = ""
    #: The crew ask that gathered this yes, when one did. A yes given to "can you
    #: stay for Jimmy" is not a standing offer to work evenings: scoping it here is
    #: what lets the fold release it when Jimmy cancels, and what stops a different
    #: customer's quote from silently spending it.
    ask_id: str = ""
    #: Who the fitter agreed to stay for, in words the rota can show.
    for_customer: str = ""


class ShiftExtensionWithdrawn(EventBase):
    """The yes is taken back: the fitter cannot stay late that day after all.

    Life happens between the agreement and the evening - a sick kid, a ride that
    fell through. Without this event an extension could only lapse through its
    customer's fate, which left "Marcus can't stay anymore" unrecordable and the
    rota chip lying. Scheduling reacts exactly as it would to any availability
    change: the validator flags what the withdrawal breaks, visibly.
    """

    type: Literal["shift_extension_withdrawn"] = "shift_extension_withdrawn"
    worker_id: WorkerId
    on_date: date
    reason: str = ""


class CrewAskOpened(EventBase):
    """A customer wants hours nobody can currently work; the crew is being asked.

    The call does not end with "no" - it ends with "let me check and call you
    back". This records the check so it survives a page reload and a shift change:
    who wants what, which fitters could cover it if they agree, and the words the
    customer used, so booking them later starts from their own call.
    """

    type: Literal["crew_ask_opened"] = "crew_ask_opened"
    ask_id: str
    customer_name: str
    phone: str
    transcript: str
    #: What saying yes means: work this date until this hour.
    on_date: date
    until_time: datetime
    candidate_ids: tuple[WorkerId, ...]
    detail: str = ""


class CrewAskClosed(EventBase):
    """The check-back happened: somebody said yes (and was extended), or everyone
    said no and the customer was told."""

    type: Literal["crew_ask_closed"] = "crew_ask_closed"
    ask_id: str
    outcome: str = ""
    #: The booking that resolved the ask, when the outcome is "booked". The link a
    #: later cancellation follows to let the crew off the hook they said yes to.
    job_id: str = ""


class OvertimeClaimed(EventBase):
    """A fitter said yes. First reply wins; the fold ignores the rest."""

    type: Literal["overtime_claimed"] = "overtime_claimed"
    job_id: JobId
    worker_id: WorkerId


Event = Annotated[
    WorkerRegistered
    | VanRegistered
    | JobRequested
    | JobSlotOffered
    | JobConfirmed
    | JobDispatched
    | JobStarted
    | JobCompleted
    | JobOverran
    | JobCancelled
    | CustomerRescheduled
    | JobDeferred
    | WorkerUnavailable
    | WorkerRestored
    | VanUnavailable
    | VanRestored
    | TrafficDelay
    | PlanCommitted
    | PlanProposed
    | ProposalApproved
    | ProposalRejected
    | OvertimeOffered
    | OvertimeClaimed
    | ShiftExtended
    | ShiftExtensionWithdrawn
    | CrewAskOpened
    | CrewAskClosed
    | WorkerRemoved
    | VanRemoved
    | DepotMoved
    | RateCardChanged,
    Field(discriminator="type"),
]
