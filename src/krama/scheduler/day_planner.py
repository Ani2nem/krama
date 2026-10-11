"""Single-day crew formation, van assignment, and routing, as one CP-SAT model.

Why these are solved together rather than in stages: assignment quality depends on
travel cost, and travel cost depends on assignment. Clustering jobs geographically
and then routing each cluster locks in bad assignments before knowing what they cost.

Why CP-SAT rather than the OR-Tools routing library: full depth here means a crew is
*one or two workers plus a van*, chosen per day. Pre-enumerating crews for six workers
and four vans gives ~84 candidate "vehicles" with no native way to say each worker and
each van is used at most once. Expressed directly in CP-SAT, that is two lines.

Crews are indexed by van - "crew k" just means "the team using van k" - which fixes
the van-assignment decision by construction and removes a large symmetry group.

Two deliberate approximations, both made safe downstream:

1. **One travel matrix per solve**, not a time-dependent one. CP-SAT cannot cheaply
   express "this leg costs more because you drive it at 16:45". The matrix is built
   *pessimistically* (worst traffic bucket the shift can span), so real travel is at
   worst as long as planned; a route that materializes faster only ever arrives early
   and waits. Optimistic matrices would do the opposite and silently break hard windows.
2. **A blended labour rate** in the objective, rather than each crew's exact mix.
   The objective only has to *rank* routes. Once a route is materialized, its true
   dollar cost is computed exactly by :mod:`krama.scheduler.costing`.

The output is a sequence, not a schedule. Concrete times come from
:func:`~krama.scheduler.routing.materialize_route`, and are then checked
independently by :mod:`krama.domain.invariants`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo

from ortools.sat.python import cp_model

from krama.config import BusinessParams
from krama.domain.enums import (
    CommitmentState,
    UnservedReason,
    WindowHardness,
)
from krama.domain.models import (
    CrewRoute,
    Job,
    JobId,
    Location,
    UnservedJob,
    VanId,
    Worker,
    WorkerId,
)
from krama.domain.state import Unavailability, WorldState
from krama.domain.travel import TravelOracle
from krama.formatting import clock_range
from krama.scheduler.routing import materialize_route
from krama.scheduler.travel.base import TimeBucket

MINUTES_PER_DAY = 24 * 60

#: Crew sessions per van per day: out from the shop, work, back to the shop. Two
#: is enough for the real pattern (day work, then an agreed evening) without
#: paying for a combinatorial day of micro-trips.
SESSIONS_PER_DAY = 2

#: What an ordinary solve pays to break a confirmed promise. Not a real dollar
#: amount - a wall. High enough that no sum of fuel, overtime and lateness within a
#: day can reach it, low enough that two genuinely conflicting promises degrade to
#: one visible breach instead of an infeasible model.
PROMISE_BREACH_PENALTY = 25_000.0


@dataclass(frozen=True, slots=True)
class SolveParams:
    """Cost weights and search limits.

    Every weight is dollars, which is what makes them arguable instead of arbitrary:
    ``labor_rate_per_minute`` is a real loaded wage, ``unserved_penalty_base`` is what
    the business thinks a dropped job costs in goodwill and lost revenue.
    """

    business_tz: tzinfo

    labor_rate_per_minute: float = 0.92
    #: What a minute past shift costs in cash. Multiplies the loaded rate: overtime
    #: pay is wholly marginal money - the fitter would otherwise be off the clock -
    #: unlike rostered wages, which are owed whether the van rolls or not.
    overtime_rate_multiplier: float = 1.5
    unserved_penalty_base: float = 500.0
    #: Added per prior deferral. Without this, cost minimization drops the same
    #: far-out, low-revenue customer every single day until they leave.
    deferral_escalation: float = 250.0
    lateness_per_minute: float = 2.0
    revenue_weight: float = 0.5

    allow_overtime: bool = True
    overtime_minutes: int = 120
    #: Margin preferred before a hard deadline. Priced at ``lateness_per_minute``
    #: rather than enforced: eating into safety margin is the same economic idea as
    #: running late, just earlier and cheaper. The invariant checker still accepts a
    #: plan that uses the full window, so a repair under pressure may spend it.
    hard_window_buffer_minutes: int = 0

    #: Traffic bucket used to build the (pessimistic) matrix. ``None`` means
    #: "worst bucket the shift can span", which is the safe default.
    travel_bucket: TimeBucket | None = None

    #: Only the k nearest destinations get an arc out of each stop.
    #:
    #: The model has one boolean per crew per ordered pair, so arcs grow as
    #: crews x n squared - and growing the crew to match the volume makes it worse,
    #: not better. Measured: 25 jobs solves in tens of milliseconds, 50 returns UNKNOWN
    #: inside five seconds, 100 likewise. Pruning is what makes the difference between
    #: a model that answers and one that does not.
    #:
    #: A van will never drive from one stop to a job forty miles away when thirty
    #: closer ones are waiting, so the arcs removed are ones no good route would use.
    #: ``None`` keeps every arc, which is right below the threshold where the full
    #: model is fast anyway.
    #:
    #: Six is aggressive, and deliberately so. Pruning only applies above
    #: ``prune_above``, and in that range the honest comparison is not "pruned versus
    #: unpruned" but "a plan versus none": at 35 jobs k=12 returns UNKNOWN and k=6
    #: serves 27. There is no quality being traded away, because there is no
    #: alternative answer to trade it against.
    k_nearest: int | None = 6
    #: Below this many stops, keep every arc.
    #:
    #: Measured at a ten-second budget. Unpruned: 25 jobs serves 23, and 28 already
    #: returns UNKNOWN - the full model falls off a cliff rather than degrading. Pruned
    #: at k=6: 32 serves 23 and 40 serves 18.
    #:
    #: So the threshold sits exactly at the real business size, which is the last point
    #: the full model answers. An earlier value of 28 left a hole: a 28-job day was
    #: above the size the full model could handle and below the size that triggered
    #: pruning, so it returned nothing while a 32-job day planned fine.
    prune_above: int = 25

    max_solve_seconds: float = 10.0
    #: A budget measured in solver work rather than wall clock.
    #:
    #: A wall-clock limit is not reproducible: when the clock stops the search rather
    #: than the search exhausting itself, the answer depends on how fast the machine
    #: happened to be. Measured - ten jobs at a 1.5 second budget gave two different
    #: objectives across three runs, while the same problem at thirty seconds gave one.
    #: That is fine for a dispatcher, who wants a guaranteed response time, and fatal
    #: for an eval baseline or a committed snapshot, which would churn on a loaded CI
    #: machine and look like a regression.
    #:
    #: Set it where the same input must give the same answer; leave it unset where a
    #: bounded wait matters more. Both limits apply when both are set.
    max_deterministic_time: float | None = None
    #: Single-threaded search makes results reproducible, which scenario replay and
    #: eval baselines depend on. Raise only for interactive solves that are not scored.
    search_workers: int = 1
    random_seed: int = 0

    # ------------------------------------------------------------------ repair mode
    #: Where each job sits in the plan being repaired, as ``(van_id, minutes past
    #: local midnight)``. Staying put is free; moving is charged ``change_penalty``.
    incumbent: Mapping[JobId, tuple[str, int]] | None = None
    #: Cost of disturbing a job that already had a place. Even provisional work churns
    #: routes crews have looked at, so a plan that changes three things beats an
    #: equivalent one that changes thirty.
    change_penalty: float = 0.0
    #: Scales ``commitment_cost`` when a promised window has to be released. Strategies
    #: vary this to trade promises against throughput.
    reschedule_penalty_multiplier: float = 1.0
    #: Whether the model may break a promise at all.
    #:
    #: Off for ordinary planning: a promised window is simply a constraint, and a
    #: morning solve that quietly moved an appointment somebody arranged their day
    #: around would be doing the one thing this system exists to prevent. Repair
    #: turns it on, because under pressure breaking a promise is sometimes the least
    #: bad option - but the release is then reported, priced, and put to a human.
    allow_promise_release: bool = False

    @classmethod
    def from_business(
        cls,
        business: BusinessParams,
        business_tz: tzinfo,
        *,
        allow_overtime: bool = True,
        travel_bucket: TimeBucket | None = None,
        max_solve_seconds: float | None = None,
        reproducible: bool = False,
    ) -> SolveParams:
        """Build solve weights from ``config/business_params.yaml``.

        Keeping the numbers out of code is what makes their provenance auditable;
        see :mod:`krama.config`.
        """
        return cls(
            business_tz=business_tz,
            labor_rate_per_minute=business.labor.loaded_rate_per_minute.value,
            overtime_rate_multiplier=business.labor.overtime_multiplier.value,
            unserved_penalty_base=business.penalties.unserved_base.value,
            deferral_escalation=business.penalties.deferral_escalation.value,
            lateness_per_minute=business.penalties.lateness_per_minute.value,
            revenue_weight=business.penalties.revenue_weight.value,
            allow_overtime=allow_overtime,
            overtime_minutes=int(business.labor.overtime_max_minutes.value),
            hard_window_buffer_minutes=int(business.scheduling.hard_window_buffer_minutes.value),
            travel_bucket=travel_bucket,
            # When reproducibility is what matters, the deterministic budget must be
            # the limit that binds; the wall clock becomes a safety net against a
            # pathological problem rather than the thing shaping the answer.
            max_solve_seconds=(
                max_solve_seconds
                if max_solve_seconds is not None
                else (
                    business.solver.deterministic_budget.value * 5
                    if reproducible
                    else business.solver.max_solve_seconds.value
                )
            ),
            search_workers=int(business.solver.search_workers.value),
            random_seed=int(business.solver.random_seed.value),
            max_deterministic_time=(
                business.solver.deterministic_budget.value if reproducible else None
            ),
        )


@dataclass(frozen=True, slots=True)
class DayPlanResult:
    routes: tuple[CrewRoute, ...]
    unserved: tuple[UnservedJob, ...]
    objective_cost: float
    status: str
    #: Promises this plan breaks. Never silent: the checker refuses a plan that
    #: moves a confirmed window unless told the release was authorised.
    released_promises: tuple[JobId, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def _minutes_since_midnight(moment: datetime, tz: tzinfo) -> int:
    local = moment.astimezone(tz)
    return local.hour * 60 + local.minute


def _day_bounds(on_date: date, tz: tzinfo) -> tuple[datetime, datetime]:
    start = datetime.combine(on_date, time(0, 0), tzinfo=tz)
    return start, start + timedelta(days=1)


def _clamp_to_day(moment: datetime, day_start: datetime, tz: tzinfo) -> int | None:
    """Minutes from local midnight, or ``None`` if the moment is on another day."""
    delta = (moment.astimezone(tz) - day_start).total_seconds() / 60.0
    if delta < -MINUTES_PER_DAY or delta > 2 * MINUTES_PER_DAY:
        return None
    return round(delta)


def _cents(dollars: float) -> int:
    return round(dollars * 100)


def _panes(job: Job) -> int:
    return job.glass_spec.pane_count if job.glass_spec else 1


def _unserved_penalty(job: Job, params: SolveParams) -> float:
    """What it costs to leave this job unscheduled today.

    Escalates with prior deferrals so a job cannot be postponed indefinitely, and
    scales with revenue and priority so the optimizer drops the cheapest thing first.
    """
    priority_multiplier = {"emergency": 6.0, "high": 2.5, "normal": 1.0, "low": 0.6}[
        job.priority.value
    ]
    base = params.unserved_penalty_base * priority_multiplier
    escalation = params.deferral_escalation * job.deferral_count
    return base + escalation + params.revenue_weight * job.revenue


#: Local-minute span of each traffic bucket.
_BUCKET_SPANS: dict[TimeBucket, tuple[int, int]] = {
    TimeBucket.EARLY: (0, 7 * 60),
    TimeBucket.AM_PEAK: (7 * 60, 9 * 60 + 30),
    TimeBucket.MIDDAY: (9 * 60 + 30, 15 * 60),
    TimeBucket.PM_PEAK: (15 * 60, 18 * 60 + 30),
    TimeBucket.EVENING: (18 * 60 + 30, MINUTES_PER_DAY),
}


def _worst_bucket_for_shift(start_min: int, end_min: int) -> TimeBucket:
    """The most congested bucket a shift running ``start..end`` can touch."""
    spans = _BUCKET_SPANS
    severity = {
        TimeBucket.PM_PEAK: 5,
        TimeBucket.AM_PEAK: 4,
        TimeBucket.MIDDAY: 3,
        TimeBucket.EARLY: 2,
        TimeBucket.EVENING: 1,
    }
    touched = [b for b, (s, e) in spans.items() if start_min < e and s < end_min]
    return max(touched, key=lambda b: severity[b]) if touched else TimeBucket.MIDDAY


#: Representative departure time for each bucket, used to build the matrix.
_BUCKET_PROBE_HOUR: dict[TimeBucket, int] = {
    TimeBucket.EARLY: 6,
    TimeBucket.AM_PEAK: 8,
    TimeBucket.MIDDAY: 12,
    TimeBucket.PM_PEAK: 16,
    TimeBucket.EVENING: 19,
}


def _windows_touch_day(job: Job, day_start: datetime, tz: tzinfo) -> bool:
    """True if any declared window overlaps this calendar day.

    A job whose only window is on Thursday is not "unschedulable Monday for lack of
    capacity" - it simply is not Monday's problem. Conflating the two sends the
    dispatcher hunting for capacity that was never the constraint.
    """
    if not job.windows:
        return True
    day_end = day_start + timedelta(days=1)
    return any(w.start < day_end and day_start < w.end for w in job.windows)


def _largest_free_interval(
    outages: Iterable[Unavailability],
    day_start: datetime,
    span_start: int,
    span_end: int,
) -> tuple[int, int] | None:
    """Longest stretch of ``[span_start, span_end)`` (minutes from midnight) not under outage.

    A resource that goes down mid-shift is still usable for whichever side of the
    outage is longer. Taking the single largest window rather than modelling every
    fragment keeps the CP-SAT model to one contiguous shift per crew, which is worth
    far more than the sliver of capacity it gives up.
    """
    blocked: list[tuple[int, int]] = []
    for outage in outages:
        start = (outage.from_time - day_start).total_seconds() / 60.0
        end = (
            (outage.until_time - day_start).total_seconds() / 60.0
            if outage.until_time is not None
            else float(MINUTES_PER_DAY)
        )
        lo, hi = max(span_start, int(start)), min(span_end, int(end))
        if lo < hi:
            blocked.append((lo, hi))

    if not blocked:
        return (span_start, span_end) if span_start < span_end else None

    blocked.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in blocked:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    best: tuple[int, int] | None = None
    cursor = span_start
    for lo, hi in [*merged, (span_end, span_end)]:
        if lo > cursor and (best is None or lo - cursor > best[1] - best[0]):
            best = (cursor, lo)
        cursor = max(cursor, hi)
    return best


def _prunable_arcs(travel_min: list[list[int]], params: SolveParams) -> set[tuple[int, int]]:
    """Which ordered pairs the routing model may use.

    Every arc to and from the depot is kept: a crew must be able to start and finish
    anywhere, and removing that is how pruning turns a feasible day infeasible. Between
    stops, each keeps an arc to its ``k`` nearest neighbours by travel time.

    Asymmetry is deliberate. A is among B's nearest without B being among A's, and
    keeping only mutual pairs would strand the outlying stop - so the union is taken,
    which costs a few arcs and removes a whole class of surprise.
    """
    n = len(travel_min)
    stops = n - 1
    if params.k_nearest is None or stops <= params.prune_above:
        return {(i, j) for i in range(n) for j in range(n) if i != j}

    allowed = {(0, j) for j in range(1, n)} | {(j, 0) for j in range(1, n)}
    for i in range(1, n):
        nearest = sorted((j for j in range(1, n) if j != i), key=lambda j: travel_min[i][j])[
            : params.k_nearest
        ]
        for j in nearest:
            allowed.add((i, j))
            allowed.add((j, i))
    return allowed


def _probe_times(
    day_start: datetime,
    shift_start: int,
    shift_end: int,
    forced: TimeBucket | None,
) -> list[datetime]:
    """Departure times at which to sample travel, one per bucket the shift touches.

    The matrix is built by taking the worst leg across these probes. Sampling a single
    "pessimistic" hour is not enough: a ``TrafficDelay`` event scoped to 07:00-12:00 is
    completely invisible to a matrix probed at 16:00, and the solver then plans around
    congestion it does not know exists.

    Cost is bounded - at most five probes, and legs are cached - and the payoff is a
    matrix that never *under*-estimates, which is what keeps materialized routes
    feasible against hard windows.
    """
    if forced is not None:
        return [day_start + timedelta(hours=_BUCKET_PROBE_HOUR[forced])]
    touched = [
        bucket for bucket, (lo, hi) in _BUCKET_SPANS.items() if shift_start < hi and lo < shift_end
    ]
    if not touched:
        touched = [TimeBucket.MIDDAY]
    return [day_start + timedelta(hours=_BUCKET_PROBE_HOUR[b]) for b in touched]


def _shift_window(worker: Worker, on_date: date) -> tuple[int, int] | None:
    """The ROSTERED day in minutes. Deliberately ignorant of one-day extensions:
    overtime pay anchors to this end, so it must not move when someone agrees to
    stay late. Reach questions use :func:`_reach_end` on top of it."""
    hours = worker.hours_for(on_date.weekday())
    if hours is None:
        return None
    return hours.start.hour * 60 + hours.start.minute, hours.end.hour * 60 + hours.end.minute


def _reach_end(
    world: WorldState,
    worker: Worker,
    on_date: date,
    span_end: int,
    day_start: datetime,
    tz: tzinfo,
) -> int:
    """How late this person may actually be kept that day: the rostered end, raised
    by a one-day ShiftExtended agreement when one exists."""
    extension = world.extension_for(worker.id, on_date)
    if extension is None:
        return span_end
    minutes = _clamp_to_day(extension, day_start, tz)
    return span_end if minutes is None else max(span_end, minutes)


# ------------------------------------------------------------------------ diagnosis


def _diagnose(
    job: Job,
    world: WorldState,
    eligible_workers: Sequence[Worker],
    on_date: date,
    day_start: datetime,
    params: SolveParams,
) -> UnservedJob:
    """Explain a non-placement in terms the dispatcher can act on.

    Ordered most-specific first: a job that no certified worker can do is a hiring
    or scheduling problem, which is different advice from "the day was simply full".
    """
    if not _windows_touch_day(job, day_start, params.business_tz):
        when = ", ".join(
            w.start.astimezone(params.business_tz).date().isoformat() for w in job.windows
        )
        return UnservedJob(
            job_id=job.id,
            reason=UnservedReason.WINDOW_ON_ANOTHER_DAY,
            detail=f"customer window falls on {when}, not {on_date.isoformat()}",
        )

    certified = [w for w in eligible_workers if job.required_certifications <= w.certifications]
    if not certified:
        held = {c for w in eligible_workers for c in w.certifications}
        missing = sorted(c.value for c in job.required_certifications - held)
        return UnservedJob(
            job_id=job.id,
            reason=UnservedReason.NO_CERTIFIED_WORKER,
            detail=(
                f"no available worker holds {missing}"
                if missing
                else "certified workers are all unavailable"
            ),
        )

    if job.crew_size > len(certified):
        return UnservedJob(
            job_id=job.id,
            reason=UnservedReason.CREW_SIZE_UNAVAILABLE,
            detail=(
                f"needs a crew of {job.crew_size} but only {len(certified)} "
                "certified worker(s) are available"
            ),
        )

    ready_on = job.earliest_material_date(job.requested_at.date())
    if ready_on > on_date:
        late = [m.part_code for m in job.materials if not m.in_stock]
        return UnservedJob(
            job_id=job.id,
            reason=UnservedReason.MATERIALS_NOT_AVAILABLE,
            detail=f"{late or ['materials']} not available until {ready_on.isoformat()}",
        )

    for window in job.hard_windows:
        window_minutes = (window.end - window.start).total_seconds() / 60.0
        if window_minutes < job.estimated_duration_min:
            return UnservedJob(
                job_id=job.id,
                reason=UnservedReason.HARD_WINDOW_UNREACHABLE,
                detail=(
                    f"hard window is {int(window_minutes)} min but the job needs "
                    f"{job.estimated_duration_min} min"
                ),
            )
        shifts = [s for s in (_shift_window(w, on_date) for w in certified) if s]
        window_start = _clamp_to_day(window.start, day_start, params.business_tz)
        window_end = _clamp_to_day(window.end, day_start, params.business_tz)
        if window_start is None or window_end is None:
            continue
        if not any(
            shift_start <= window_end and window_start <= shift_end
            for shift_start, shift_end in shifts
        ):
            return UnservedJob(
                job_id=job.id,
                reason=UnservedReason.HARD_WINDOW_UNREACHABLE,
                detail="hard window does not overlap any certified worker's shift",
            )

    # Per part, not "any part": a job blocked on one missing item stays silent
    # under an any() over every material and van.
    if job.commitment_state is CommitmentState.CONFIRMED and job.windows:
        # A promised window is binding during ordinary planning, so a job that cannot
        # be fitted inside it is not "the day was full" - it is "we could not keep the
        # promise", which is what a dispatcher has to ring somebody about.
        promised = job.windows[0]
        opens = _clamp_to_day(promised.start, day_start, params.business_tz)
        closes = _clamp_to_day(promised.end, day_start, params.business_tz)
        if opens is not None and closes is not None:
            # Usable time, not declared shift: a worker held up until half ten is
            # not available at eight, and checking the roster rather than the
            # outages made this branch unreachable.
            usable = [
                _largest_free_interval(
                    world.worker_outages.get(worker.id, ()),
                    day_start,
                    span[0],
                    _reach_end(world, worker, on_date, span[1], day_start, params.business_tz),
                )
                for worker in certified
                if (span := _shift_window(worker, on_date)) is not None
            ]
            reachable = any(
                max(window[0], opens) + job.estimated_duration_min <= min(window[1], closes)
                for window in usable
                if window
            )
            if not reachable:
                return UnservedJob(
                    job_id=job.id,
                    reason=UnservedReason.HARD_WINDOW_UNREACHABLE,
                    detail=(
                        "nobody certified is free for long enough inside the promised "
                        "window "
                        + clock_range(
                            promised.start.astimezone(params.business_tz),
                            promised.end.astimezone(params.business_tz),
                        )
                        + "; "
                        "keeping it would need the customer telephoned"
                    ),
                )

    for material in job.materials:
        if not material.in_stock:
            continue
        if not any(
            van.has_stock(material.part_code, material.quantity) for van in world.vans.values()
        ):
            return UnservedJob(
                job_id=job.id,
                reason=UnservedReason.VAN_CAPACITY_EXCEEDED,
                detail=f"no van stocks {material.quantity}x {material.part_code}",
            )

    return UnservedJob(
        job_id=job.id,
        reason=UnservedReason.NO_CAPACITY_IN_HORIZON,
        detail="feasible in principle, but the day had no room at an acceptable cost",
    )


# ---------------------------------------------------------------------------- solver


def plan_day(
    *,
    world: WorldState,
    travel: TravelOracle,
    on_date: date,
    candidate_job_ids: Sequence[JobId],
    params: SolveParams,
    locked_job_ids: Sequence[JobId] = (),
    pinned_workers: Mapping[JobId, tuple[WorkerId, ...]] | None = None,
) -> DayPlanResult:
    """Build and solve one day.

    ``locked_job_ids`` are jobs that must be served today (used by repair mode to pin
    work already dispatched). Everything else is optional at the price of its
    unserved penalty.
    """
    tz = params.business_tz
    day_start, _ = _day_bounds(on_date, tz)

    # ---------------------------------------------------------------- candidates
    jobs: list[Job] = []
    pre_unserved: list[UnservedJob] = []
    for job_id in candidate_job_ids:
        job = world.jobs.get(job_id)
        if job is None or not job.is_active:
            continue
        ready_on = job.earliest_material_date(job.requested_at.date())
        if ready_on > on_date:
            pending_parts = [m.part_code for m in job.materials if not m.in_stock]
            pre_unserved.append(
                UnservedJob(
                    job_id=job.id,
                    reason=UnservedReason.MATERIALS_NOT_AVAILABLE,
                    detail=(
                        f"{pending_parts or ['materials']} not available "
                        f"until {ready_on.isoformat()}"
                    ),
                )
            )
            continue
        if not _windows_touch_day(job, day_start, tz):
            when = ", ".join(w.start.astimezone(tz).date().isoformat() for w in job.windows)
            pre_unserved.append(
                UnservedJob(
                    job_id=job.id,
                    reason=UnservedReason.WINDOW_ON_ANOTHER_DAY,
                    detail=f"customer window falls on {when}, not {on_date.isoformat()}",
                )
            )
            continue
        jobs.append(job)

    # ------------------------------------------------------------------ resources
    # Outages are intersected with the shift rather than used as an on/off filter.
    # A van that dies at 10:40 was perfectly usable at 06:00, and excluding it from
    # the whole day throws away a morning of capacity for no reason.
    shift_span: dict[WorkerId, tuple[int, int]] = {}
    #: Workers whose usable day ends at an OUTAGE edge rather than their shift
    #: close - "out 10 to 11" truncating the morning. Overtime extends shift
    #: closes, never absences: letting the standard allowance reach past a
    #: dentist appointment scheduled Priya's install straight through it.
    cut_short: dict[WorkerId, bool] = {}
    workers: list[Worker] = []
    for worker in sorted(world.workers.values(), key=lambda w: w.id):
        span = _shift_window(worker, on_date)
        if span is None:
            continue
        usable = _largest_free_interval(
            world.worker_outages.get(worker.id, ()), day_start, span[0], span[1]
        )
        if usable is None:
            continue
        workers.append(worker)
        shift_span[worker.id] = usable
        cut_short[worker.id] = usable[1] < span[1]

    van_window: dict[VanId, tuple[int, int]] = {}
    vans: list[VanId] = []
    for van in sorted(world.vans.values(), key=lambda v: v.id):
        usable = _largest_free_interval(
            world.van_outages.get(van.id, ()), day_start, 0, MINUTES_PER_DAY
        )
        if usable is None:
            continue
        vans.append(van.id)
        van_window[van.id] = usable

    if not jobs or not workers or not vans:
        unserved = tuple(
            pre_unserved + [_diagnose(j, world, workers, on_date, day_start, params) for j in jobs]
        )
        return DayPlanResult(
            routes=(),
            unserved=unserved,
            objective_cost=sum(_unserved_penalty(world.jobs[u.job_id], params) for u in unserved),
            status="EMPTY",
            metrics={"candidates": float(len(jobs)), "crews": 0.0},
        )

    overtime = params.overtime_minutes if params.allow_overtime else 0
    earliest_shift = min(s for s, _ in shift_span.values())
    latest_shift = max(e for _, e in shift_span.values()) + overtime

    # ------------------------------------------------------------- travel matrix
    probes = _probe_times(day_start, earliest_shift, latest_shift, params.travel_bucket)

    depot: Location = world.vans[vans[0]].home_depot
    nodes: list[Location] = [depot] + [j.location for j in jobs]
    n = len(nodes)
    travel_min = [[0] * n for _ in range(n)]
    travel_mi = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            legs = [travel.leg(nodes[i], nodes[j], probe) for probe in probes]
            travel_min[i][j] = max(leg.minutes for leg in legs)
            travel_mi[i][j] = max(leg.miles for leg in legs)

    model = cp_model.CpModel()
    # Two crew sessions per van per day. A session is one depot-rooted circuit:
    # leave the shop, work, come back. The second session is how "Dan solo all
    # morning, then Dan + Priya together in one van at five" becomes schedulable -
    # crews re-form AT THE SHOP, never at a kerb, and no van is ever left anywhere
    # but home. Slots 0..V-1 are each van's first session, V..2V-1 the second.
    num_vans = len(vans)
    num_crews = num_vans * SESSIONS_PER_DAY

    def van_of(k: int) -> str:
        return vans[k % num_vans]

    job_index = {job.id: idx + 1 for idx, job in enumerate(jobs)}
    locked = set(locked_job_ids)

    # ---------------------------------------------------------------- variables
    assign = {
        (w.id, k): model.new_bool_var(f"assign_{w.id}_{k}")
        for w in workers
        for k in range(num_crews)
    }
    visit = {
        (job.id, k): model.new_bool_var(f"visit_{job.id}_{k}")
        for job in jobs
        for k in range(num_crews)
    }
    served = {job.id: model.new_bool_var(f"served_{job.id}") for job in jobs}
    start = {
        (job.id, k): model.new_int_var(0, MINUTES_PER_DAY, f"start_{job.id}_{k}")
        for job in jobs
        for k in range(num_crews)
    }
    late = {
        (job.id, k): model.new_int_var(0, MINUTES_PER_DAY, f"late_{job.id}_{k}")
        for job in jobs
        for k in range(num_crews)
    }
    # Releasing a promise: only meaningful for work a customer was actually told
    # about, and priced at that job's own commitment_cost.
    released = {
        job.id: (
            model.new_bool_var(f"released_{job.id}")
            if job.commitment_state is CommitmentState.CONFIRMED
            else model.new_constant(0)
        )
        for job in jobs
    }
    incumbent = params.incumbent or {}
    stay = {job.id: model.new_bool_var(f"stay_{job.id}") for job in jobs if job.id in incumbent}
    crew_active = [model.new_bool_var(f"active_{k}") for k in range(num_crews)]
    crew_start = [
        model.new_int_var(0, MINUTES_PER_DAY, f"crew_start_{k}") for k in range(num_crews)
    ]
    crew_end = [model.new_int_var(0, MINUTES_PER_DAY, f"crew_end_{k}") for k in range(num_crews)]

    # --------------------------------------------------------------- assignment
    for worker in workers:
        # One crew per SESSION, not per day: the morning pair and the evening pair
        # may be different people. A fitter on two sessions works them in order -
        # the second cannot leave the shop before the first brings them back.
        model.add_at_most_one(assign[worker.id, k] for k in range(num_vans))
        model.add_at_most_one(assign[worker.id, k] for k in range(num_vans, num_crews))
        for early in range(num_vans):
            for late_slot in range(num_vans, num_crews):
                model.add(crew_start[late_slot] >= crew_end[early]).only_enforce_if(
                    assign[worker.id, early], assign[worker.id, late_slot]
                )

    # A van's second session leaves after its first returns; and the second slot
    # stays dark unless the first is used, so equal plans have one canonical shape.
    for i in range(num_vans):
        later = num_vans + i
        model.add_implication(crew_active[later], crew_active[i])
        model.add(crew_start[later] >= crew_end[i]).only_enforce_if(
            crew_active[i], crew_active[later]
        )

    for k in range(num_crews):
        headcount = sum(assign[w.id, k] for w in workers)
        model.add(headcount <= 2)
        model.add(headcount >= 1).only_enforce_if(crew_active[k])
        model.add(headcount == 0).only_enforce_if(~crew_active[k])

        # Crew hours are the intersection of its members' shifts, plus any overtime.
        # A one-day ShiftExtended agreement raises one member's ceiling further: the
        # person said yes to a named hour, and that yes outranks the standard
        # allowance for that date only.
        for worker in workers:
            span_start, span_end = shift_span[worker.id]
            model.add(crew_start[k] >= span_start).only_enforce_if(assign[worker.id, k])
            allowance = 0
            if not cut_short[worker.id]:
                allowance = overtime if worker.overtime_eligible else 0
                extended = world.extension_for(worker.id, on_date)
                if extended is not None:
                    extended_min = _clamp_to_day(extended, day_start, params.business_tz)
                    if extended_min is not None:
                        allowance = max(allowance, extended_min - span_end)
            model.add(crew_end[k] <= span_end + allowance).only_enforce_if(assign[worker.id, k])
        # The crew cannot start before its van is available or finish after it goes down.
        usable_start, usable_end = van_window[van_of(k)]
        model.add(crew_start[k] >= usable_start)
        model.add(crew_end[k] <= usable_end)
        model.add(crew_end[k] >= crew_start[k])

    van_index = {van_id: k for k, van_id in enumerate(vans)}
    terms_released_penalty: list[cp_model.LinearExpr] = []
    for job in jobs:
        model.add(sum(visit[job.id, k] for k in range(num_crews)) == served[job.id])
        if job.id in locked:
            model.add(served[job.id] == 1)

        # Dropping a promise is a way of breaking it, not a way of avoiding the
        # cost. This used to apply only to jobs carried in from an incumbent plan,
        # which meant a freshly CONFIRMED booking - promised on the phone minutes
        # ago, never yet in any committed plan - could be dropped at the plain
        # unserved price. The dispatcher read "no room at an acceptable cost" about
        # a job two fitters had just agreed to stay late for.
        if job.commitment_state is CommitmentState.CONFIRMED:
            model.add(released[job.id] >= 1 - served[job.id])

        if not params.allow_promise_release:
            # Not forbidden outright - PRICED beyond any economics. released == 0
            # here turned one unkeepable promise into INFEASIBLE for the whole day,
            # taking five keepable customers down with it. At a five-figure price
            # the solver keeps every promise it possibly can, and when one truly
            # cannot be kept it breaks that one visibly - the checker flags it, the
            # banner goes red, one customer gets a call - while the rest of the day
            # stands.
            terms_released_penalty.append(released[job.id] * _cents(PROMISE_BREACH_PENALTY))

        placement = incumbent.get(job.id)
        if placement is None:
            continue
        van_id, minutes = placement
        k0 = van_index.get(van_id)
        if k0 is None:
            # The van it was on is gone - a breakdown. It cannot stay put.
            model.add(stay[job.id] == 0)
            continue

        # Staying put is free, moving is charged. Soft by construction: the model
        # prices travel pessimistically, so a materialized arrival cannot be asserted
        # as an equality - the solver would correctly call it unreachable. Work that
        # is genuinely immovable is carved out of the problem entirely; see
        # :func:`krama.scheduler.repair.carve_out_locked`.
        #
        # Either of the van's sessions counts as staying put: an evening job that
        # re-forms its crew at the shop is still on the same van at the same hour,
        # and charging it as "moved" would make re-crewing look like churn.
        same_van = [k0, num_vans + k0]
        model.add(sum(visit[job.id, k] for k in same_van) == 1).only_enforce_if(stay[job.id])
        for k in same_van:
            model.add(start[job.id, k] >= minutes).only_enforce_if(stay[job.id], visit[job.id, k])

    # ------------------------------------------------------- crew fitness per job
    # A claimed overtime offer pins its claimant: the fitter who said yes to the
    # extra hours serves the job, wherever cost would otherwise have put it. The pin
    # dissolves if the claimant is not available that day (marked out after
    # claiming) - the offer flow re-opens rather than the job becoming unservable.
    available_ids = {w.id for w in workers}
    claimed_by: dict[str, str] = {
        offer.job_id: offer.claimed_by
        for offer in world.overtime_offers.values()
        if offer.claimed_by is not None
        and offer.on_date == on_date
        and offer.claimed_by in available_ids
    }
    # Transient pins, from the caller: "this job is served by exactly these people
    # TODAY". The quote's slot card names a crew, and the booking's own commit used
    # to run a fresh solve free to pick an equal-cost somebody else - so the card
    # lied the moment the button was pressed. Pins dissolve per-worker if the named
    # person is unavailable that day, same rule as a claimed overtime offer.
    pins: dict[str, tuple[str, ...]] = {
        job_id: tuple(w for w in names if w in available_ids)
        for job_id, names in (pinned_workers or {}).items()
    }

    for job in jobs:
        for k in range(num_crews):
            model.add_implication(visit[job.id, k], crew_active[k])
            claimant = claimed_by.get(job.id)
            if claimant is not None:
                model.add(assign[claimant, k] == 1).only_enforce_if(visit[job.id, k])
            for pinned in pins.get(job.id, ()):
                model.add(assign[pinned, k] == 1).only_enforce_if(visit[job.id, k])
            # A crew must be at least as large as the job demands; a two-person crew
            # may do one-person work, never the reverse.
            model.add(sum(assign[w.id, k] for w in workers) >= job.crew_size).only_enforce_if(
                visit[job.id, k]
            )
            for cert in sorted(job.required_certifications, key=lambda c: c.value):
                holders = [w for w in workers if cert in w.certifications]
                if not holders:
                    model.add(visit[job.id, k] == 0)
                    break
                model.add(sum(assign[w.id, k] for w in holders) >= 1).only_enforce_if(
                    visit[job.id, k]
                )

    # -------------------------------------------------------------- van capacity
    # Racks limit what a van carries AT ONCE, so each session gets the full rack -
    # the crew reloads at the shop between runs. Stock is the business's parts for
    # the day, so a van's two sessions draw from one shelf, not two.
    part_codes = {m.part_code for job in jobs for m in job.materials if m.in_stock}
    for k in range(num_crews):
        van = world.vans[van_of(k)]
        model.add(sum(visit[job.id, k] * _panes(job) for job in jobs) <= van.rack_slots)
    for i in range(num_vans):
        van = world.vans[vans[i]]
        sessions = [i, num_vans + i]
        for part_code in sorted(part_codes):
            demand = [
                visit[job.id, k] * m.quantity
                for k in sessions
                for job in jobs
                for m in job.materials
                if m.in_stock and m.part_code == part_code
            ]
            if demand:
                model.add(sum(demand) <= van.stock.get(part_code, 0))

    # -------------------------------------------------------------------- routing
    # Which arcs the model is allowed to use at all.
    allowed_arcs = _prunable_arcs(travel_min, params)

    arc_vars: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for k in range(num_crews):
        arcs: list[cp_model.ArcT] = []
        # Depot sits out entirely when the crew is idle.
        arcs.append((0, 0, ~crew_active[k]))
        for job in jobs:
            idx = job_index[job.id]
            arcs.append((idx, idx, ~visit[job.id, k]))

        for i in range(n):
            for j in range(n):
                if i == j or (i, j) not in allowed_arcs:
                    continue
                arc = model.new_bool_var(f"arc_{k}_{i}_{j}")
                arc_vars[k, i, j] = arc
                arcs.append((i, j, arc))

                if i == 0:
                    job = jobs[j - 1]
                    model.add(start[job.id, k] >= crew_start[k] + travel_min[0][j]).only_enforce_if(
                        arc
                    )
                elif j == 0:
                    job = jobs[i - 1]
                    model.add(
                        crew_end[k]
                        >= start[job.id, k] + job.estimated_duration_min + travel_min[i][0]
                    ).only_enforce_if(arc)
                else:
                    from_job, to_job = jobs[i - 1], jobs[j - 1]
                    model.add(
                        start[to_job.id, k]
                        >= start[from_job.id, k]
                        + from_job.estimated_duration_min
                        + travel_min[i][j]
                    ).only_enforce_if(arc)
        model.add_circuit(arcs)

    # -------------------------------------------------------------- time windows
    for job in jobs:
        duration = job.estimated_duration_min
        for k in range(num_crews):
            model.add(start[job.id, k] >= crew_start[k]).only_enforce_if(visit[job.id, k])
            model.add(start[job.id, k] + duration <= crew_end[k]).only_enforce_if(visit[job.id, k])

            if not job.windows:
                model.add(late[job.id, k] == 0)
                continue

            selectors = []
            for w_idx, window in enumerate(job.windows):
                win_start = _clamp_to_day(window.start, day_start, tz)
                win_end = _clamp_to_day(window.end, day_start, tz)
                selector = model.new_bool_var(f"win_{job.id}_{k}_{w_idx}")
                selectors.append(selector)

                if (
                    win_start is None
                    or win_end is None
                    or win_end <= 0
                    or win_start >= MINUTES_PER_DAY
                ):
                    model.add(selector == 0)
                    continue

                model.add(start[job.id, k] >= max(win_start, 0)).only_enforce_if(selector)
                # A promise binds like a hard window. Releasing it is possible and
                # priced (see `released`), but never free and never silent.
                binding = (
                    window.hardness is WindowHardness.HARD
                    or job.commitment_state is CommitmentState.CONFIRMED
                )
                if binding:
                    # The deadline itself is hard. The buffer is not: it is priced as
                    # encroachment so the planner prefers margin but never refuses
                    # work for want of it. Clamping it as a constraint looked safe and
                    # was not - a crew cannot arrive before its shift starts, so a
                    # generous buffer silently made feasible jobs unschedulable.
                    model.add(start[job.id, k] + duration <= win_end).only_enforce_if(selector)
                    model.add(
                        late[job.id, k]
                        >= start[job.id, k]
                        + duration
                        - (win_end - params.hard_window_buffer_minutes)
                    ).only_enforce_if(selector)
                else:
                    model.add(
                        late[job.id, k] >= start[job.id, k] + duration - win_end
                    ).only_enforce_if(selector)

            # A released promise no longer needs a window to sit in. Everything
            # else must satisfy one of its declared windows.
            model.add(sum(selectors) >= visit[job.id, k] - released[job.id])
            model.add(sum(selectors) <= visit[job.id, k])
            model.add(late[job.id, k] == 0).only_enforce_if(~visit[job.id, k])

    # ------------------------------------------------------------------ objective
    #
    # Marginal cash only. An earlier version charged every driving minute at the
    # loaded wage and charged overtime nothing at all, and the optimum under those
    # prices was exactly what a dispatcher reported as broken: every job of the day
    # piled onto one fitter, 36 minutes into her overtime, while five rostered
    # colleagues sat idle. Under the real cash flows that plan is the expensive one.
    # A rostered fitter's wage is owed whether their van rolls or not - deploying
    # them costs fuel, not salary - while every minute past shift is money that
    # would otherwise not be spent, at time and a half. So: vehicle miles and
    # overtime minutes are costs; in-shift labour is capacity.
    late_per_min = _cents(params.lateness_per_minute)
    ot_per_min = _cents(params.labor_rate_per_minute * params.overtime_rate_multiplier)

    terms: list[cp_model.LinearExpr] = []
    #: Each fitter's overtime variables across sessions, for the per-day cap below.
    member_ot_vars: dict[str, list[cp_model.IntVar]] = {w.id: [] for w in workers}
    for k in range(num_crews):
        van_id = van_of(k)
        van_cost_per_mile = world.vans[van_id].cost_per_mile

        # Vehicle cost is per van-mile regardless of who is aboard.
        vehicle_terms = [
            arc_vars[k, i, j] * _cents(travel_mi[i][j] * van_cost_per_mile)
            for (i, j) in allowed_arcs
            if _cents(travel_mi[i][j] * van_cost_per_mile)
        ]
        terms.extend(vehicle_terms)

        # Overtime, per assigned member past their own shift close. only_enforce_if
        # leaves the variable floating for the unassigned, and minimization settles
        # it at zero; for the assigned it settles at max(0, crew_end - their close),
        # which covers late driving as well as late work - a van on the road at six
        # is a fitter being paid at six.
        for worker in workers:
            span_close = shift_span[worker.id][1]
            # Ceiling is the whole day, not the standard allowance: an extended
            # shift can run further than 120 minutes, and a bound tighter than the
            # crew-end constraints would turn that agreement into INFEASIBLE.
            member_ot = model.new_int_var(0, MINUTES_PER_DAY, f"ot_{worker.id}_{k}")
            # Overtime is minutes WORKED past the shift close, not minutes elapsed
            # since it. A session that leaves the shop at five did not pay its crew
            # from three - charging from the close made every evening run look two
            # hours more expensive than it is.
            late_departure = model.new_bool_var(f"late_dep_{worker.id}_{k}")
            model.add(crew_start[k] > span_close).only_enforce_if(late_departure)
            model.add(crew_start[k] <= span_close).only_enforce_if(~late_departure)
            model.add(member_ot >= crew_end[k] - span_close).only_enforce_if(
                assign[worker.id, k], ~late_departure
            )
            model.add(member_ot >= crew_end[k] - crew_start[k]).only_enforce_if(
                assign[worker.id, k], late_departure
            )
            member_ot_vars[worker.id].append(member_ot)
            terms.append(member_ot * ot_per_min)

    # The overtime allowance is a DAY's budget, not a per-session one: two sessions
    # may split the 120 minutes (or an agreed extension), never bank it twice.
    for worker in workers:
        span_close = shift_span[worker.id][1]
        allowance = 0
        if not cut_short[worker.id]:
            allowance = overtime if worker.overtime_eligible else 0
            extended = world.extension_for(worker.id, on_date)
            if extended is not None:
                extended_min = _clamp_to_day(extended, day_start, params.business_tz)
                if extended_min is not None:
                    allowance = max(allowance, extended_min - span_close)
        model.add(sum(member_ot_vars[worker.id]) <= allowance)

    # Epsilon tie-break: one cent per van index, so equal-cost plans always pick the
    # same vans instead of reshuffling whenever an unrelated cost term shifts. Real
    # cost differences are dollars, so this can never outvote an actual decision -
    # but it keeps plans stable, which is what a dispatcher notices.
    for k in range(num_crews):
        terms.append(crew_active[k] * k)

    terms.extend(terms_released_penalty)

    change_cost = _cents(params.change_penalty)
    if change_cost:
        terms.extend((1 - stay_var) * change_cost for stay_var in stay.values())

    for job in jobs:
        if job.commitment_state is CommitmentState.CONFIRMED and job.commitment_cost:
            terms.append(
                released[job.id]
                * _cents(job.commitment_cost * params.reschedule_penalty_multiplier)
            )

    for job in jobs:
        penalty = _cents(_unserved_penalty(job, params))
        terms.append((1 - served[job.id]) * penalty)
        for k in range(num_crews):
            terms.append(late[job.id, k] * late_per_min)

    # ------------------------------------------------------- breaking exact ties
    #
    # Two plans can cost exactly the same and put different people in different vans.
    # That is not symmetry - Sofia and Alex hold different certifications and are paid
    # differently - it is degeneracy, and the blended labour rate this model uses by
    # design is part of what creates it: the objective deliberately cannot see the
    # difference between a $57 tech and a $52 one, so on a day where both are qualified
    # it has no reason to prefer either.
    #
    # Left alone, the answer is whichever the search reached first, which is a property
    # of the machine rather than the problem. The same OR-Tools version, the same seed,
    # one search worker and a deterministic budget all failed to fix it: both answers
    # were OPTIMAL and cost $405.61 to the cent. It surfaced as golden board snapshots
    # passing on arm64 and failing on x86_64.
    #
    # So the tie is broken here instead, lexicographically: among plans of equal cost,
    # prefer the one that puts earlier-named workers on earlier vans. Scaling the real
    # objective above the largest possible tie-break value makes this exact rather than
    # approximate - no tie-break can ever outweigh a single cent of real cost.
    primary = sum(terms)

    # Two tie-breaks, ordered. Idle time between jobs is not charged for, so a crew
    # that could start at 06:23 may equally start at 06:24 and the objective cannot
    # tell - which showed up as golden snapshots differing by one minute between
    # architectures, the same way the crew assignment did before it was broken. Earlier
    # is also simply better: slack belongs at the end of a day, where an overrun can
    # use it, rather than at the start where it cannot.
    #
    # First tier: no riders. In-shift labour is capacity rather than cost now, so
    # nothing in the primary objective stops a second fitter riding along on
    # one-person work - free in cash terms, but it burns the flexibility of having
    # them somewhere else when the phone rings. Fewer people deployed wins any cash
    # tie before timing does.
    deployed = sum(assign.values())
    # Tight ceilings matter here: the lexicographic scaling multiplies them all
    # together, and with session slots doubling num_crews the old loose bounds
    # overflowed int64 and the whole model came back MODEL_INVALID. Each worker
    # holds at most one crew per session; each job is visited at most once.
    deployed_ceiling = len(workers) * SESSIONS_PER_DAY

    # `start` is only constrained when its job is actually visited, so an unvisited
    # pair sits at zero and contributes nothing - and a job is visited by at most
    # one crew, so the sum is bounded by jobs, not jobs x crews.
    earliness = sum(start.values())
    earliness_ceiling = len(jobs) * MINUTES_PER_DAY

    # Then, among plans that are equal on cost and on timing, prefer earlier-named
    # workers on earlier vans.
    worker_order = sum(
        assign[worker.id, k] * (rank * num_crews + k)
        for rank, worker in enumerate(workers)
        for k in range(num_crews)
    )
    worker_ceiling = SESSIONS_PER_DAY * len(workers) * (len(workers) * num_crews + 1)

    timing_order = earliness * (worker_ceiling + 1) + worker_order
    timing_ceiling = (earliness_ceiling + 1) * (worker_ceiling + 1)
    tie_break = deployed * (timing_ceiling + 1) + timing_order
    tie_break_ceiling = (deployed_ceiling + 1) * (timing_ceiling + 1)
    model.minimize(primary * (tie_break_ceiling + 1) + tie_break)

    # --------------------------------------------------------------------- solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.max_solve_seconds
    if params.max_deterministic_time is not None:
        solver.parameters.max_deterministic_time = params.max_deterministic_time
    solver.parameters.num_search_workers = params.search_workers
    solver.parameters.random_seed = params.random_seed
    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        unserved = tuple(
            pre_unserved + [_diagnose(j, world, workers, on_date, day_start, params) for j in jobs]
        )
        return DayPlanResult(
            routes=(),
            unserved=unserved,
            objective_cost=sum(_unserved_penalty(world.jobs[u.job_id], params) for u in unserved),
            status=status_name,
            # Report the model's size even when it ran out of time. A solve that gave
            # up is precisely when its shape is worth knowing, and omitting the arc
            # counts here means the only run you cannot inspect is the failing one.
            metrics={
                "candidates": float(len(jobs)),
                "crews": float(num_crews),
                "arcs_per_crew": float(len(allowed_arcs)),
                "arcs_pruned": float(max(0, n * (n - 1) - len(allowed_arcs))),
                "travel_probes": float(len(probes)),
            },
        )

    # ------------------------------------------------------------------ extract
    routes: list[CrewRoute] = []
    scheduled: set[JobId] = set()
    for k in range(num_crews):
        van_id = van_of(k)
        if not solver.boolean_value(crew_active[k]):
            continue
        crew_workers = [w.id for w in workers if solver.boolean_value(assign[w.id, k])]
        sequence = sorted(
            (job for job in jobs if solver.boolean_value(visit[job.id, k])),
            key=lambda job: solver.value(start[job.id, k]),
        )
        if not sequence or not crew_workers:
            continue
        scheduled.update(job.id for job in sequence)
        shift_start_min = solver.value(crew_start[k])
        routes.append(
            materialize_route(
                world=world,
                travel=travel,
                # The later session is its own named crew: possibly different
                # people in the same van, formed at the shop after the first run.
                crew_id=f"crew-{van_id}" + ("-pm" if k >= num_vans else ""),
                worker_ids=crew_workers,
                van_id=van_id,
                on_date=on_date,
                job_sequence=[job.id for job in sequence],
                shift_start=day_start + timedelta(minutes=shift_start_min),
            )
        )

    unserved_out = list(pre_unserved)
    for job in jobs:
        if job.id not in scheduled:
            unserved_out.append(_diagnose(job, world, workers, on_date, day_start, params))

    released_promises = tuple(
        job.id
        for job in jobs
        if job.commitment_state is CommitmentState.CONFIRMED and solver.value(released[job.id])
    )

    return DayPlanResult(
        routes=tuple(routes),
        unserved=tuple(unserved_out),
        released_promises=released_promises,
        # The primary term, not the scaled objective the solver minimised.
        objective_cost=solver.value(primary) / 100.0,
        status=status_name,
        metrics={
            "candidates": float(len(jobs)),
            "crews": float(len(routes)),
            "scheduled": float(len(scheduled)),
            "solve_seconds": solver.wall_time,
            "best_bound": solver.best_objective_bound / 100.0,
            "travel_probes": float(len(probes)),
            "arcs_per_crew": float(len(allowed_arcs)),
            "arcs_pruned": float(max(0, n * (n - 1) - len(allowed_arcs))),
            "moved": float(sum(1 for v in stay.values() if not solver.boolean_value(v))),
            "promises_released": float(
                sum(
                    1
                    for job in jobs
                    if job.commitment_state is CommitmentState.CONFIRMED
                    and solver.value(released[job.id])
                )
            ),
        },
    )


def unserved_are_expected(result: DayPlanResult) -> bool:
    """True when nothing was dropped for a reason the business would call surprising."""
    return all(
        u.reason
        in {
            UnservedReason.MATERIALS_NOT_AVAILABLE,
            UnservedReason.NO_CAPACITY_IN_HORIZON,
            UnservedReason.WINDOW_ON_ANOTHER_DAY,
        }
        for u in result.unserved
    )


__all__ = [
    "DayPlanResult",
    "SolveParams",
    "plan_day",
    "unserved_are_expected",
]
