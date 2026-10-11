"""Independent verification that a plan is actually feasible.

This module is deliberately written *without* reference to how the solver works. It
re-derives every checkable fact from the plan, the world state, and a travel oracle.
If the solver and this checker ever disagree, the checker wins and the plan is
rejected - that is the entire point. A small language model in the loop makes this
more important, not less: agents propose, deterministic code disposes.

``validate_plan`` is a runtime guard, not a test. Nothing reaches the database
without passing it.

Window semantics, which are easy to get wrong:

* A **hard** window means the service must *start and finish* inside it. "Must be
  done before we open at 09:00" is not satisfied by starting at 08:55.
* A **soft** window constrains the start only. Finishing late is priced by the
  objective as a lateness penalty, not treated as infeasible.
* Under either kind, service may not *begin* before the window opens - nobody is
  there to let the crew in.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo

from krama.domain.enums import (
    CommitmentState,
    ViolationCode,
    WindowHardness,
)
from krama.domain.models import (
    CrewRoute,
    Frozen,
    Job,
    JobId,
    PlanVersion,
    Stop,
    TimeWindow,
    VanId,
    WorkerId,
)
from krama.domain.state import WorldState
from krama.domain.travel import TravelOracle


class Violation(Frozen):
    """One breach of a scheduling invariant, addressed to whoever must fix it."""

    code: ViolationCode
    detail: str
    job_id: JobId | None = None
    crew_id: str | None = None
    worker_id: WorkerId | None = None
    van_id: VanId | None = None
    on_date: date | None = None

    def __str__(self) -> str:
        where = " ".join(
            part
            for part in (
                f"job={self.job_id}" if self.job_id else "",
                f"crew={self.crew_id}" if self.crew_id else "",
                f"worker={self.worker_id}" if self.worker_id else "",
                f"van={self.van_id}" if self.van_id else "",
            )
            if part
        )
        return f"[{self.code.value}] {self.detail}" + (f" ({where})" if where else "")


def _say_window(window: TimeWindow, tz: tzinfo) -> str:
    start = window.start.astimezone(tz)
    end = window.end.astimezone(tz)
    return f"{start:%a %-I:%M %p} - {end:%-I:%M %p}"


def _say_moment(moment: datetime, tz: tzinfo) -> str:
    local = moment.astimezone(tz)
    return f"{local:%a %-I:%M %p}"


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Tolerances for the checker.

    ``travel_slack_minutes`` exists only to absorb minute-rounding between the
    solver's integer model and the oracle's recomputation. It is not a budget for
    being wrong - keep it at 1 or 2.
    """

    business_tz: tzinfo
    travel_slack_minutes: int = 1
    allow_overtime: bool = True
    #: How far past a shift close overtime may legally keep someone. The check
    #: used to wave through ANY end for an eligible worker, which meant
    #: withdrawing the extension that justified an evening changed nothing on
    #: the board - the plan kept a fitter five hours late and called it fine.
    overtime_max_minutes: int = 120
    #: Promises a human has agreed to break. A plan that moves a confirmed window
    #: without its job appearing here is rejected - the release has to be a decision
    #: somebody made, not something a solver did quietly and a checker inferred.
    released_job_ids: frozenset[JobId] = frozenset()


# --------------------------------------------------------------------------- helpers


def _local(moment: datetime, tz: tzinfo) -> datetime:
    return moment.astimezone(tz)


def _route_span(route: CrewRoute) -> tuple[datetime, datetime] | None:
    """Clock-in to clock-out, including the drive out and the drive home."""
    if not route.stops:
        return None
    first, last = route.stops[0], route.stops[-1]
    start = first.arrival - timedelta(minutes=first.travel_minutes_from_prev)
    end = last.departure + timedelta(minutes=route.return_to_depot_minutes)
    return start, end


def _overlaps(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _is_binding(job: Job, window: TimeWindow, released: frozenset[JobId] = frozenset()) -> bool:
    """Whether this window must contain the whole service.

    Two independent reasons it might. ``hardness`` says what the customer *needs* - a
    storefront that can only be worked on before it opens. ``CONFIRMED`` says what we
    *promised*, and a promise binds even when the underlying preference was soft:
    once someone has been told "between nine and eleven" and arranged their morning
    around it, that window stops being a preference the optimiser may spend.
    """
    if job.id in released:
        # A human has agreed to break this promise, so what remains is whatever the
        # customer originally needed. A solver cannot reach this state on its own -
        # the release has to be a decision somebody made and passed in, which is what
        # keeps "we moved your appointment" from being something the system does
        # quietly.
        return window.hardness is WindowHardness.HARD
    return (
        window.hardness is WindowHardness.HARD or job.commitment_state is CommitmentState.CONFIRMED
    )


def _matching_window(
    job: Job, stop: Stop, released: frozenset[JobId] = frozenset()
) -> TimeWindow | None:
    """The first declared window this stop satisfies, if any.

    The asymmetry is the whole point of having two kinds. A **hard** window must
    contain the entire service - "done before we open at nine" is not satisfied by
    starting at 08:55. A **soft** window only says when the crew may *begin*; running
    past its end is lateness, which the objective prices per minute rather than
    refuses.

    Treating a late soft window as infeasible, as an earlier version did, quietly
    turns every soft window into a hard one: a customer who said "Tuesday morning
    would suit" makes their job unschedulable the moment the morning fills up, rather
    than simply expensive to serve in the afternoon.
    """
    for window in job.windows:
        # Nobody is there to let the crew in before the window opens, whatever its kind.
        if stop.arrival < window.start:
            continue
        if _is_binding(job, window, released) and stop.departure > window.end:
            continue
        return window
    return None


# ---------------------------------------------------------------------------- checks


def _check_references(plan: PlanVersion, world: WorldState) -> list[Violation]:
    out: list[Violation] = []
    for route in plan.routes:
        for worker_id in route.worker_ids:
            if worker_id not in world.workers:
                out.append(
                    Violation(
                        code=ViolationCode.UNKNOWN_ENTITY_REFERENCE,
                        detail=f"route references unknown worker {worker_id!r}",
                        crew_id=route.crew_id,
                        worker_id=worker_id,
                        on_date=route.date,
                    )
                )
        if route.van_id not in world.vans:
            out.append(
                Violation(
                    code=ViolationCode.UNKNOWN_ENTITY_REFERENCE,
                    detail=f"route references unknown van {route.van_id!r}",
                    crew_id=route.crew_id,
                    van_id=route.van_id,
                    on_date=route.date,
                )
            )
        for stop in route.stops:
            if stop.job_id not in world.jobs:
                out.append(
                    Violation(
                        code=ViolationCode.UNKNOWN_ENTITY_REFERENCE,
                        detail=f"route references unknown job {stop.job_id!r}",
                        crew_id=route.crew_id,
                        job_id=stop.job_id,
                        on_date=route.date,
                    )
                )
    return out


def _check_duplicate_assignments(plan: PlanVersion) -> list[Violation]:
    seen: dict[JobId, str] = {}
    out: list[Violation] = []
    for route in plan.routes:
        for stop in route.stops:
            if stop.job_id in seen:
                out.append(
                    Violation(
                        code=ViolationCode.DUPLICATE_JOB_ASSIGNMENT,
                        detail=(
                            f"job scheduled on both crew {seen[stop.job_id]!r} "
                            f"and crew {route.crew_id!r}"
                        ),
                        job_id=stop.job_id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
            else:
                seen[stop.job_id] = route.crew_id
    return out


def _check_crew_fitness(plan: PlanVersion, world: WorldState) -> list[Violation]:
    """Certifications and crew size.

    A crew satisfies a job when the *union* of its members' certifications covers
    the requirement - one certified lead may work alongside an uncertified hand.
    """
    out: list[Violation] = []
    for route in plan.routes:
        workers = [world.workers[w] for w in route.worker_ids if w in world.workers]
        held = frozenset().union(*(w.certifications for w in workers)) if workers else frozenset()
        for stop in route.stops:
            job = world.jobs.get(stop.job_id)
            if job is None:
                continue
            missing = job.required_certifications - held
            if missing:
                out.append(
                    Violation(
                        code=ViolationCode.MISSING_CERTIFICATION,
                        detail=(
                            f"crew lacks {sorted(c.value for c in missing)} "
                            f"required by {job.service_type.value}"
                        ),
                        job_id=job.id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
            # "at least", not "exactly": a two-person crew may do one-person work,
            # never the reverse. Oversized crews are a cost question, not a feasibility one.
            if len(route.worker_ids) < job.crew_size:
                out.append(
                    Violation(
                        code=ViolationCode.CREW_SIZE_MISMATCH,
                        detail=(
                            f"job needs a crew of at least {job.crew_size}, "
                            f"route has {len(route.worker_ids)}"
                        ),
                        job_id=job.id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
    return out


def _check_resource_exclusivity(plan: PlanVersion) -> list[Violation]:
    """No worker and no van may be in two places at once."""
    out: list[Violation] = []
    worker_spans: dict[WorkerId, list[tuple[tuple[datetime, datetime], str]]] = {}
    van_spans: dict[VanId, list[tuple[tuple[datetime, datetime], str]]] = {}

    for route in plan.routes:
        span = _route_span(route)
        if span is None:
            continue
        for worker_id in route.worker_ids:
            for other_span, other_crew in worker_spans.get(worker_id, []):
                if _overlaps(span, other_span):
                    out.append(
                        Violation(
                            code=ViolationCode.WORKER_DOUBLE_BOOKED,
                            detail=(
                                f"worker is on crew {other_crew!r} and crew "
                                f"{route.crew_id!r} at overlapping times"
                            ),
                            worker_id=worker_id,
                            crew_id=route.crew_id,
                            on_date=route.date,
                        )
                    )
            worker_spans.setdefault(worker_id, []).append((span, route.crew_id))

        for other_span, other_crew in van_spans.get(route.van_id, []):
            if _overlaps(span, other_span):
                out.append(
                    Violation(
                        code=ViolationCode.VAN_DOUBLE_BOOKED,
                        detail=(
                            f"van is on crew {other_crew!r} and crew "
                            f"{route.crew_id!r} at overlapping times"
                        ),
                        van_id=route.van_id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
        van_spans.setdefault(route.van_id, []).append((span, route.crew_id))
    return out


def _check_resource_availability(plan: PlanVersion, world: WorldState) -> list[Violation]:
    """Nobody is scheduled through a known outage."""
    out: list[Violation] = []
    for route in plan.routes:
        span = _route_span(route)
        if span is None:
            continue
        start, end = span
        for worker_id in route.worker_ids:
            if worker_id in world.workers and not world.is_worker_available(worker_id, start, end):
                out.append(
                    Violation(
                        code=ViolationCode.WORKER_UNAVAILABLE,
                        detail="worker is scheduled during a recorded outage",
                        worker_id=worker_id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
        if route.van_id in world.vans and not world.is_van_available(route.van_id, start, end):
            out.append(
                Violation(
                    code=ViolationCode.VAN_UNAVAILABLE,
                    detail="van is scheduled during a recorded outage",
                    van_id=route.van_id,
                    crew_id=route.crew_id,
                    on_date=route.date,
                )
            )
    return out


def _check_travel_consistency(
    plan: PlanVersion, world: WorldState, travel: TravelOracle, config: ValidationConfig
) -> list[Violation]:
    """Recompute every leg. The solver does not get to assert its own arrival times."""
    out: list[Violation] = []
    slack = timedelta(minutes=config.travel_slack_minutes)

    for route in plan.routes:
        van = world.vans.get(route.van_id)
        previous_stop: Stop | None = None
        for stop in route.stops:
            job = world.jobs.get(stop.job_id)
            if job is None:
                continue

            if previous_stop is None or stop.fresh_run:
                # The first stop of the day - or of a fresh run after the crew went
                # home rather than idling out a long gap - departs from the depot.
                origin = van.home_depot if van else None
                depart_at = stop.arrival - timedelta(minutes=stop.travel_minutes_from_prev)
                if previous_stop is not None and stop.arrival < previous_stop.departure:
                    out.append(
                        Violation(
                            code=ViolationCode.TRAVEL_TIME_INCONSISTENT,
                            detail=(
                                f"arrives {stop.arrival.isoformat()} before leaving the "
                                f"previous stop at {previous_stop.departure.isoformat()}"
                            ),
                            job_id=stop.job_id,
                            crew_id=route.crew_id,
                            on_date=route.date,
                        )
                    )
            else:
                previous_job = world.jobs.get(previous_stop.job_id)
                origin = previous_job.location if previous_job else None
                depart_at = previous_stop.departure

                if stop.arrival < previous_stop.departure:
                    out.append(
                        Violation(
                            code=ViolationCode.TRAVEL_TIME_INCONSISTENT,
                            detail=(
                                f"arrives {stop.arrival.isoformat()} before leaving the "
                                f"previous stop at {previous_stop.departure.isoformat()}"
                            ),
                            job_id=stop.job_id,
                            crew_id=route.crew_id,
                            on_date=route.date,
                        )
                    )

            if origin is not None:
                expected = travel.leg(origin, job.location, depart_at)
                earliest = depart_at + timedelta(minutes=expected.minutes) - slack
                if stop.arrival < earliest:
                    out.append(
                        Violation(
                            code=ViolationCode.TRAVEL_TIME_INCONSISTENT,
                            detail=(
                                f"claims arrival at {stop.arrival.isoformat()} but the drive "
                                f"takes {expected.minutes} min from {depart_at.isoformat()}"
                            ),
                            job_id=stop.job_id,
                            crew_id=route.crew_id,
                            on_date=route.date,
                        )
                    )

            if stop.service_minutes < job.estimated_duration_min:
                out.append(
                    Violation(
                        code=ViolationCode.TRAVEL_TIME_INCONSISTENT,
                        detail=(
                            f"allots {stop.service_minutes} min on site but the job needs "
                            f"{job.estimated_duration_min}"
                        ),
                        job_id=job.id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
            previous_stop = stop
    return out


def _check_windows(
    plan: PlanVersion, world: WorldState, config: ValidationConfig
) -> list[Violation]:
    out: list[Violation] = []
    for route in plan.routes:
        for stop in route.stops:
            job = world.jobs.get(stop.job_id)
            if job is None or not job.windows:
                continue

            if _matching_window(job, stop, config.released_job_ids) is None:
                declared = ", ".join(
                    f"{w.start.isoformat()}..{w.end.isoformat()} ({w.hardness.value})"
                    for w in job.windows
                )
                early = all(stop.arrival < w.start for w in job.windows)
                code = (
                    ViolationCode.CONFIRMED_WINDOW_MOVED
                    if job.commitment_state is CommitmentState.CONFIRMED
                    else ViolationCode.HARD_WINDOW_VIOLATED
                )
                out.append(
                    Violation(
                        code=code,
                        detail=(
                            (
                                f"{job.customer_name} was promised "
                                f"{_say_window(job.windows[0], config.business_tz)} and this "
                                f"plan has them at {_say_moment(stop.arrival, config.business_tz)}"
                                " - they have not been told"
                            )
                            if job.commitment_state is CommitmentState.CONFIRMED
                            else (
                                (
                                    "crew arrives before any window opens"
                                    if early
                                    else "service runs past the end of a hard window"
                                )
                                + f": {stop.arrival.isoformat()}..{stop.departure.isoformat()} "
                                f"against [{declared}]"
                            )
                        ),
                        job_id=job.id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )

            if stop.arrival < job.requested_at:
                out.append(
                    Violation(
                        code=ViolationCode.SCHEDULED_BEFORE_REQUEST,
                        detail=(
                            f"scheduled {stop.arrival.isoformat()} before the job was "
                            f"requested at {job.requested_at.isoformat()}"
                        ),
                        job_id=job.id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
    return out


def _check_working_hours(
    plan: PlanVersion, world: WorldState, config: ValidationConfig
) -> list[Violation]:
    out: list[Violation] = []
    for route in plan.routes:
        span = _route_span(route)
        if span is None:
            continue
        start_local = _local(span[0], config.business_tz)
        end_local = _local(span[1], config.business_tz)

        for worker_id in route.worker_ids:
            worker = world.workers.get(worker_id)
            if worker is None:
                continue
            hours = worker.hours_for(start_local.weekday())
            if hours is None:
                out.append(
                    Violation(
                        code=ViolationCode.OUTSIDE_WORKING_HOURS,
                        detail=f"worker does not work on weekday {start_local.weekday()}",
                        worker_id=worker_id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
                continue

            if start_local.time() < hours.start:
                out.append(
                    Violation(
                        code=ViolationCode.OUTSIDE_WORKING_HOURS,
                        detail=(
                            f"route starts {start_local.time()} before the shift "
                            f"begins at {hours.start}"
                        ),
                        worker_id=worker_id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
            allowance = (
                config.overtime_max_minutes
                if config.allow_overtime and worker.overtime_eligible
                else 0
            )
            allowed_end = datetime.combine(
                end_local.date(), hours.end, tzinfo=config.business_tz
            ) + timedelta(minutes=allowance)
            extension = world.extension_for(worker_id, route.date)
            if extension is not None:
                allowed_end = max(allowed_end, extension.astimezone(config.business_tz))
            if end_local > allowed_end:
                reason = (
                    "past their overtime reach - a late stay needs the fitter's yes"
                    if allowance
                    else (
                        "overtime is disabled for this solve"
                        if not config.allow_overtime
                        else "worker is not overtime eligible"
                    )
                )
                out.append(
                    Violation(
                        code=ViolationCode.OUTSIDE_WORKING_HOURS,
                        detail=(
                            f"route keeps them to {end_local.time()} but their day "
                            f"ends at {allowed_end.time()} - {reason}"
                        ),
                        worker_id=worker_id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
    return out


def _check_van_capacity(plan: PlanVersion, world: WorldState) -> list[Violation]:
    """Rack slots and consumable stock, per van per day."""
    out: list[Violation] = []
    for route in plan.routes:
        van = world.vans.get(route.van_id)
        if van is None:
            continue

        panes = 0
        needed: dict[str, int] = {}
        for stop in route.stops:
            job = world.jobs.get(stop.job_id)
            if job is None:
                continue
            panes += job.glass_spec.pane_count if job.glass_spec else 1
            for material in job.materials:
                if material.in_stock:
                    needed[material.part_code] = (
                        needed.get(material.part_code, 0) + material.quantity
                    )

        if panes > van.rack_slots:
            out.append(
                Violation(
                    code=ViolationCode.VAN_CAPACITY_EXCEEDED,
                    detail=f"route carries {panes} panes but the van has {van.rack_slots} slots",
                    van_id=van.id,
                    crew_id=route.crew_id,
                    on_date=route.date,
                )
            )
        for part_code, quantity in sorted(needed.items()):
            if not van.has_stock(part_code, quantity):
                out.append(
                    Violation(
                        code=ViolationCode.VAN_CAPACITY_EXCEEDED,
                        detail=(
                            f"route needs {quantity}x {part_code} but the van stocks "
                            f"{van.stock.get(part_code, 0)}"
                        ),
                        van_id=van.id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
    return out


def _check_materials(plan: PlanVersion, world: WorldState) -> list[Violation]:
    """A part on a three-day order cannot be installed tomorrow, however good the route."""
    out: list[Violation] = []
    for route in plan.routes:
        for stop in route.stops:
            job = world.jobs.get(stop.job_id)
            if job is None or not job.materials:
                continue
            ready_on = job.earliest_material_date(job.requested_at.date())
            if route.date < ready_on:
                late = [m.part_code for m in job.materials if not m.in_stock]
                out.append(
                    Violation(
                        code=ViolationCode.MATERIALS_UNAVAILABLE,
                        detail=(
                            f"scheduled {route.date.isoformat()} but {late or ['materials']} "
                            f"are not available until {ready_on.isoformat()}"
                        ),
                        job_id=job.id,
                        crew_id=route.crew_id,
                        on_date=route.date,
                    )
                )
    return out


# ------------------------------------------------------------------------- entrypoint


def validate_plan(
    plan: PlanVersion,
    world: WorldState,
    travel: TravelOracle,
    config: ValidationConfig,
) -> tuple[Violation, ...]:
    """Every invariant, in one pass. Empty result means the plan may be committed."""
    violations: list[Violation] = []
    violations += _check_references(plan, world)
    violations += _check_duplicate_assignments(plan)
    violations += _check_crew_fitness(plan, world)
    violations += _check_resource_exclusivity(plan)
    violations += _check_resource_availability(plan, world)
    violations += _check_travel_consistency(plan, world, travel, config)
    violations += _check_windows(plan, world, config)
    violations += _check_working_hours(plan, world, config)
    violations += _check_van_capacity(plan, world)
    violations += _check_materials(plan, world)
    return tuple(violations)


def validate_against_baseline(
    plan: PlanVersion,
    baseline: PlanVersion,
    world: WorldState,
) -> tuple[Violation, ...]:
    """Checks that only make sense relative to the plan being replaced.

    Repair mode may reshuffle the day, but work already dispatched or completed is
    physically fixed - a crew on site cannot be retroactively moved.
    """
    out: list[Violation] = []
    for route in baseline.routes:
        for stop in route.stops:
            job = world.jobs.get(stop.job_id)
            if job is None or job.commitment_state not in {
                CommitmentState.DISPATCHED,
                CommitmentState.COMPLETED,
            }:
                continue
            found = plan.stop_for(stop.job_id)
            if found is None:
                out.append(
                    Violation(
                        code=ViolationCode.LOCKED_JOB_MOVED,
                        detail=(
                            f"job is {job.commitment_state.value} in the committed plan but "
                            "absent from the proposed plan"
                        ),
                        job_id=stop.job_id,
                        crew_id=route.crew_id,
                    )
                )
                continue
            new_route, new_stop = found
            if new_stop.arrival != stop.arrival or set(new_route.worker_ids) != set(
                route.worker_ids
            ):
                out.append(
                    Violation(
                        code=ViolationCode.LOCKED_JOB_MOVED,
                        detail=(
                            f"job is {job.commitment_state.value} but moved from "
                            f"{stop.arrival.isoformat()} on crew {route.crew_id!r} to "
                            f"{new_stop.arrival.isoformat()} on crew {new_route.crew_id!r}"
                        ),
                        job_id=stop.job_id,
                        crew_id=new_route.crew_id,
                    )
                )
    return tuple(out)


def summarize(violations: Sequence[Violation] | Iterable[Violation]) -> str:
    items = list(violations)
    if not items:
        return "plan is feasible: 0 violations"
    lines = [f"plan is INFEASIBLE: {len(items)} violation(s)"]
    lines += [f"  - {v}" for v in items]
    return "\n".join(lines)
