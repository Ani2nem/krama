"""Deriving world state from the event log.

``WorldState`` is a *view*, never a stored fact: it is always ``fold(events, as_of)``.
Folding to an arbitrary ``as_of`` is what makes "what did we know at 10:52?" an
answerable question, which scenario replay and disruption evals both depend on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from krama.domain.enums import CommitmentState
from krama.domain.events import (
    CrewAskClosed,
    CrewAskOpened,
    CustomerRescheduled,
    DepotMoved,
    Event,
    JobCancelled,
    JobCompleted,
    JobConfirmed,
    JobDeferred,
    JobDispatched,
    JobOverran,
    JobRequested,
    JobStarted,
    OvertimeClaimed,
    OvertimeOffered,
    PlanCommitted,
    RateCardChanged,
    ShiftExtended,
    TrafficDelay,
    VanRegistered,
    VanRemoved,
    VanRestored,
    VanUnavailable,
    WorkerRegistered,
    WorkerRemoved,
    WorkerRestored,
    WorkerUnavailable,
)
from krama.domain.models import (
    Frozen,
    Job,
    JobId,
    Van,
    VanId,
    Worker,
    WorkerId,
)


class Unavailability(Frozen):
    """A resource outage. ``until_time`` of ``None`` means open-ended."""

    from_time: datetime
    until_time: datetime | None = None
    reason: str = ""

    def covers(self, start: datetime, end: datetime) -> bool:
        """True if the outage overlaps the half-open interval ``[start, end)``."""
        if end <= self.from_time:
            return False
        return self.until_time is None or start < self.until_time


class OvertimeOffer(Frozen):
    """Hours past shift, up for grabs.

    ``claimed_by`` empty until somebody says yes; first reply wins and the fold
    ignores the rest. The deadline is when the fallback - the fitter the solver
    pencilled in - is told the hours are theirs; a claim after it but before the
    job is still honoured, because a dispatcher would honour it too.
    """

    job_id: JobId
    on_date: date
    offered_to: tuple[WorkerId, ...]
    fallback: WorkerId
    overtime_minutes: int
    claim_deadline: datetime
    claimed_by: WorkerId | None = None

    def status(self, now: datetime) -> str:
        if self.claimed_by:
            return "claimed"
        return "expired" if now >= self.claim_deadline else "open"


class CrewAsk(Frozen):
    """An unresolved "can anyone make these hours?" - red on the board until closed."""

    ask_id: str
    customer_name: str
    phone: str
    transcript: str
    on_date: date
    until_time: datetime
    candidate_ids: tuple[WorkerId, ...]
    detail: str = ""


class TrafficOverride(Frozen):
    """A multiplier layered on the travel matrix for a corridor and period."""

    origin_geohash5: str | None
    dest_geohash5: str | None
    multiplier: float
    from_time: datetime
    until_time: datetime | None = None

    def applies_to(self, origin_gh5: str, dest_gh5: str, at: datetime) -> bool:
        if at < self.from_time or (self.until_time is not None and at >= self.until_time):
            return False
        if self.origin_geohash5 is not None and self.origin_geohash5 != origin_gh5:
            return False
        return not (self.dest_geohash5 is not None and self.dest_geohash5 != dest_gh5)


@dataclass
class WorldState:
    """Everything the solver and agents need to know, at one moment in time."""

    as_of: datetime
    workers: dict[WorkerId, Worker] = field(default_factory=dict)
    vans: dict[VanId, Van] = field(default_factory=dict)
    jobs: dict[JobId, Job] = field(default_factory=dict)
    worker_outages: dict[WorkerId, list[Unavailability]] = field(default_factory=dict)
    van_outages: dict[VanId, list[Unavailability]] = field(default_factory=dict)
    overtime_offers: dict[JobId, OvertimeOffer] = field(default_factory=dict)
    #: One-day agreements to work late, keyed by (worker, date). The value is how
    #: late; scheduling honours it for that date, pay treats it as plain overtime.
    shift_extensions: dict[tuple[WorkerId, date], datetime] = field(default_factory=dict)
    #: Which crew ask gathered each extension, and who the fitter agreed to stay
    #: for. A yes is for a customer, not a blank cheque: these let the fold release
    #: an agreement when its customer cancels or is told no, and let a quote refuse
    #: to spend someone's late stay on a different caller without asking again.
    extension_sources: dict[tuple[WorkerId, date], str] = field(default_factory=dict)
    extension_notes: dict[tuple[WorkerId, date], str] = field(default_factory=dict)
    #: Crew ask -> the booking that resolved it (CrewAskClosed outcome "booked").
    ask_bookings: dict[str, str] = field(default_factory=dict)
    #: Unresolved crew asks - promises to call a customer back.
    crew_asks: dict[str, CrewAsk] = field(default_factory=dict)
    #: Owner-set prices, overriding the estimated rate card field by field. The
    #: config file stays the shipped default; what the business actually charges
    #: lives here, in the log, where changes have authors and timestamps.
    rate_card: dict[str, float] = field(default_factory=dict)
    traffic_overrides: list[TrafficOverride] = field(default_factory=list)
    committed_plan_id: str | None = None
    applied_event_count: int = 0

    # ------------------------------------------------------------------ queries

    def active_jobs(self) -> list[Job]:
        """Jobs still needing service, oldest request first for stable ordering."""
        return sorted(
            (j for j in self.jobs.values() if j.is_active),
            key=lambda j: (j.requested_at, j.id),
        )

    def schedulable_jobs(self) -> list[Job]:
        """Active jobs that the solver may place - excludes in-flight work."""
        return [j for j in self.active_jobs() if j.commitment_state != CommitmentState.DISPATCHED]

    def extension_for(self, worker_id: WorkerId, on_date: date) -> datetime | None:
        return self.shift_extensions.get((worker_id, on_date))

    def extension_covers(self, worker_id: WorkerId, on_date: date, phone: str) -> bool:
        """May THIS caller's booking ride this fitter's late stay?

        Yes when the extension is unscoped (recorded by hand, no ask behind it), or
        when the ask that gathered it is still open and belongs to this caller's
        phone number. A yes consumed by a booked job, or given for someone else's
        hours, is not this caller's to spend - ask the crew again.
        """
        src = self.extension_sources.get((worker_id, on_date))
        if src is None:
            return True
        ask = self.crew_asks.get(src)
        if ask is None:
            return False
        mine = _digits(phone)
        return bool(mine) and _digits(ask.phone) == mine

    def extension_customer(self, worker_id: WorkerId, on_date: date) -> str:
        return self.extension_notes.get((worker_id, on_date), "")

    def is_worker_available(self, worker_id: WorkerId, start: datetime, end: datetime) -> bool:
        if worker_id not in self.workers:
            return False
        return not any(o.covers(start, end) for o in self.worker_outages.get(worker_id, []))

    def is_van_available(self, van_id: VanId, start: datetime, end: datetime) -> bool:
        if van_id not in self.vans:
            return False
        return not any(o.covers(start, end) for o in self.van_outages.get(van_id, []))

    def available_workers(self, start: datetime, end: datetime) -> list[Worker]:
        return [w for w in self.workers.values() if self.is_worker_available(w.id, start, end)]

    def available_vans(self, start: datetime, end: datetime) -> list[Van]:
        return [v for v in self.vans.values() if self.is_van_available(v.id, start, end)]

    def traffic_multiplier(self, origin_gh5: str, dest_gh5: str, at: datetime) -> float:
        """Product of every override in force. Overlapping delays compound."""
        multiplier = 1.0
        for override in self.traffic_overrides:
            if override.applies_to(origin_gh5, dest_gh5, at):
                multiplier *= override.multiplier
        return multiplier


def _carve_outages(outages: list[Unavailability], start: datetime, end: datetime | None) -> None:
    """Remove the interval ``[start, end)`` from every outage, splitting as needed.

    This replaced "close the most recent open-ended outage", which made a restore
    silently do nothing against an outage recorded with an explicit until - the board
    grew a ↺ button that could never bring anyone back. A restore means the person is
    available for that stretch, full stop: an outage inside the window disappears, one
    overlapping an edge is trimmed, and one spanning the whole window becomes two.
    ``end`` of ``None`` means "from start onwards", the plain "he's back" case.
    """
    kept: list[Unavailability] = []
    for outage in outages:
        before_window = outage.until_time is not None and outage.until_time <= start
        after_window = end is not None and outage.from_time >= end
        if before_window or after_window:
            kept.append(outage)
            continue
        if outage.from_time < start:
            kept.append(outage.model_copy(update={"until_time": start}))
        if end is not None and (outage.until_time is None or outage.until_time > end):
            kept.append(outage.model_copy(update={"from_time": end}))
    outages[:] = kept


def _apply(state: WorldState, event: Event) -> None:
    """Apply one event. Structural matching keeps each branch narrowed to its own type."""
    match event:
        case WorkerRegistered():
            state.workers[event.worker.id] = event.worker

        case VanRegistered():
            state.vans[event.van.id] = event.van

        case JobRequested():
            job = event.job
            if job.commitment_state is CommitmentState.DRAFT:
                job = job.model_copy(update={"commitment_state": CommitmentState.PROVISIONAL})
            state.jobs[job.id] = job

        case JobConfirmed():
            if existing := state.jobs.get(event.job_id):
                state.jobs[existing.id] = existing.model_copy(
                    update={
                        "commitment_state": CommitmentState.CONFIRMED,
                        "commitment_cost": event.commitment_cost,
                        "windows": (event.window,),
                    }
                )

        case JobDispatched() | JobStarted():
            if existing := state.jobs.get(event.job_id):
                state.jobs[existing.id] = existing.model_copy(
                    update={"commitment_state": CommitmentState.DISPATCHED}
                )

        case JobCompleted():
            if existing := state.jobs.get(event.job_id):
                state.jobs[existing.id] = existing.model_copy(
                    update={
                        "commitment_state": CommitmentState.COMPLETED,
                        "estimated_duration_min": event.actual_duration_min,
                    }
                )

        case JobOverran():
            # The crew is still on site; the estimate was wrong and the rest of the
            # day is now at risk. Grow the duration so the re-solve sees reality.
            if existing := state.jobs.get(event.job_id):
                state.jobs[existing.id] = existing.model_copy(
                    update={
                        "estimated_duration_min": existing.estimated_duration_min
                        + event.extra_minutes
                    }
                )

        case JobCancelled():
            if existing := state.jobs.get(event.job_id):
                state.jobs[existing.id] = existing.model_copy(
                    update={"commitment_state": CommitmentState.CANCELLED}
                )
            state.overtime_offers.pop(event.job_id, None)
            # Late stays agreed for this booking lapse with it: the fitter said yes
            # to this customer's evening, not to evenings in general.
            for ask_id, booked in list(state.ask_bookings.items()):
                if booked == event.job_id:
                    _release_ask_extensions(state, ask_id)
                    state.ask_bookings.pop(ask_id, None)

        case CustomerRescheduled():
            # A confirmed window the customer themselves moved is no longer a promise
            # we are keeping, so the commitment cost that protected it is released.
            if existing := state.jobs.get(event.job_id):
                state.jobs[existing.id] = existing.model_copy(
                    update={
                        "windows": event.new_windows,
                        "commitment_state": CommitmentState.PROVISIONAL,
                        "commitment_cost": 0.0,
                    }
                )

        case JobDeferred():
            if existing := state.jobs.get(event.job_id):
                state.jobs[existing.id] = existing.model_copy(
                    update={"deferral_count": existing.deferral_count + 1}
                )

        case WorkerUnavailable():
            state.worker_outages.setdefault(event.worker_id, []).append(
                Unavailability(
                    from_time=event.from_time,
                    until_time=event.until_time,
                    reason=event.reason,
                )
            )

        case WorkerRestored():
            _carve_outages(
                state.worker_outages.get(event.worker_id, []),
                event.window_start or event.occurred_at,
                event.window_end,
            )

        case VanUnavailable():
            state.van_outages.setdefault(event.van_id, []).append(
                Unavailability(
                    from_time=event.from_time,
                    until_time=event.until_time,
                    reason=event.reason,
                )
            )

        case VanRestored():
            _carve_outages(
                state.van_outages.get(event.van_id, []),
                event.window_start or event.occurred_at,
                event.window_end,
            )

        case OvertimeOffered():
            state.overtime_offers[event.job_id] = OvertimeOffer(
                job_id=event.job_id,
                on_date=event.on_date,
                offered_to=event.offered_to,
                fallback=event.fallback,
                overtime_minutes=event.overtime_minutes,
                claim_deadline=event.claim_deadline,
            )

        case RateCardChanged():
            state.rate_card.update(event.entries)

        case DepotMoved():
            for van_id, van in list(state.vans.items()):
                state.vans[van_id] = van.model_copy(update={"home_depot": event.location})

        case WorkerRemoved():
            state.workers.pop(event.worker_id, None)
            state.worker_outages.pop(event.worker_id, None)

        case VanRemoved():
            state.vans.pop(event.van_id, None)
            state.van_outages.pop(event.van_id, None)

        case ShiftExtended():
            key = (event.worker_id, event.on_date)
            standing = state.shift_extensions.get(key)
            # Two agreements for the same day keep the later hour; "I can stay to
            # seven" does not cancel "I can stay to eight".
            if standing is None or event.until_time > standing:
                state.shift_extensions[key] = event.until_time
                if event.ask_id:
                    state.extension_sources[key] = event.ask_id
                else:
                    state.extension_sources.pop(key, None)
                if event.for_customer:
                    state.extension_notes[key] = event.for_customer
                else:
                    state.extension_notes.pop(key, None)

        case CrewAskOpened():
            state.crew_asks[event.ask_id] = CrewAsk(
                ask_id=event.ask_id,
                customer_name=event.customer_name,
                phone=event.phone,
                transcript=event.transcript,
                on_date=event.on_date,
                until_time=event.until_time,
                candidate_ids=event.candidate_ids,
                detail=event.detail,
            )

        case CrewAskClosed():
            state.crew_asks.pop(event.ask_id, None)
            if event.outcome == "booked" and event.job_id:
                # The yeses stand: they back this booking now, and lapse with it.
                state.ask_bookings[event.ask_id] = event.job_id
            else:
                # The customer was told no (or the ask abandoned). Nobody is
                # staying late for a job that is not happening.
                _release_ask_extensions(state, event.ask_id)

        case OvertimeClaimed():
            offer = state.overtime_offers.get(event.job_id)
            # First reply wins; a claim from someone never offered is noise.
            if offer and offer.claimed_by is None and event.worker_id in offer.offered_to:
                state.overtime_offers[event.job_id] = offer.model_copy(
                    update={"claimed_by": event.worker_id}
                )

        case TrafficDelay():
            state.traffic_overrides.append(
                TrafficOverride(
                    origin_geohash5=event.origin_geohash5,
                    dest_geohash5=event.dest_geohash5,
                    multiplier=event.multiplier,
                    from_time=event.from_time,
                    until_time=event.until_time,
                )
            )

        case PlanCommitted():
            state.committed_plan_id = event.plan_id

        case _:
            # JobSlotOffered, PlanProposed, ProposalApproved and ProposalRejected are
            # recorded for audit and eval labelling but do not mutate world state.
            pass


def _digits(phone: str) -> str:
    return "".join(ch for ch in phone if ch.isdigit())


def _release_ask_extensions(state: WorldState, ask_id: str) -> None:
    """Forget every late-stay agreement a crew ask gathered."""
    for key, src in list(state.extension_sources.items()):
        if src == ask_id:
            state.shift_extensions.pop(key, None)
            state.extension_sources.pop(key, None)
            state.extension_notes.pop(key, None)


def fold(events: Iterable[Event], as_of: datetime | None = None) -> WorldState:
    """Replay the log into a world state.

    Events are applied in ``recorded_at`` order - the order we *learned* things -
    and any event recorded after ``as_of`` is excluded, so the result is exactly
    what was knowable at that moment.
    """
    ordered: Sequence[Event] = sorted(events, key=lambda e: (e.recorded_at, e.event_id))
    if as_of is not None:
        ordered = [e for e in ordered if e.recorded_at <= as_of]

    resolved_as_of = as_of or (ordered[-1].recorded_at if ordered else datetime.min)
    state = WorldState(as_of=resolved_as_of)
    for event in ordered:
        _apply(state, event)
    state.applied_event_count = len(ordered)
    return state
