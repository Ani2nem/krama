"""Repairing a committed plan after something goes wrong.

A disruption is not a re-planning problem. Most of the day is still fine, crews are
already out, and customers have been told things. Re-solving from scratch would be
correct and useless: it would produce a better schedule that breaks every promise in
it.

So repair is optimisation with three extra facts the morning solve does not have:

* **Work already in flight is physically fixed.** A crew on site cannot be moved by
  an optimiser.
* **A promise costs money to break.** ``commitment_cost`` - the numeric weight behind
  "I'll take off work that day" - is what the model spends when it must release one.
  That field exists precisely so this decision is priced rather than arbitrary.
* **Change itself has a cost.** Even moving provisional work churns routes that crews
  have already looked at, so a plan that changes three things beats an equivalent one
  that changes thirty.

Repair returns several labelled candidates rather than one answer, because choosing
between "keep every promise and serve less" and "serve more and make two phone calls"
is a judgement about this business on this day. The solver prices the options; a human
or the coordinator agent picks.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from krama.domain.diff import PlanDiff, diff_plans
from krama.domain.enums import CommitmentState
from krama.domain.models import CrewRoute, JobId, PlanVersion
from krama.domain.state import Unavailability, WorldState
from krama.domain.travel import TravelOracle
from krama.scheduler.day_planner import SolveParams
from krama.scheduler.horizon import HorizonParams, HorizonResult, plan_horizon


@dataclass(frozen=True, slots=True)
class RepairStrategy:
    """A named way of trading promises against throughput."""

    name: str
    description: str
    change_penalty: float = 0.0
    release_multiplier: float = 1.0
    allow_overtime: bool = True
    unserved_multiplier: float = 1.0


#: The three trade-offs a dispatcher actually reasons about on a bad morning.
STRATEGIES: tuple[RepairStrategy, ...] = (
    RepairStrategy(
        name="least_disruption",
        description="Keep every promise. Move as little as possible, serve fewer jobs.",
        change_penalty=60.0,
        release_multiplier=4.0,
    ),
    RepairStrategy(
        name="most_jobs",
        description="Serve the most work today, accepting phone calls where needed.",
        change_penalty=5.0,
        release_multiplier=1.0,
        unserved_multiplier=2.5,
    ),
    RepairStrategy(
        name="no_overtime",
        description="Finish within shift hours, whatever that costs in coverage.",
        change_penalty=20.0,
        release_multiplier=2.0,
        allow_overtime=False,
    ),
)


@dataclass(frozen=True, slots=True)
class RepairCandidate:
    strategy: RepairStrategy
    plan: PlanVersion
    result: HorizonResult
    diff: PlanDiff
    #: Promises this candidate breaks. The checker will not accept the plan without
    #: being told, and a dispatcher should not accept it without being asked.
    released_promises: tuple[JobId, ...] = ()

    @property
    def changes(self) -> int:
        return len(self.diff.changes)

    @property
    def customer_calls(self) -> int:
        return len(self.diff.customer_visible_changes)

    @property
    def jobs_served(self) -> int:
        return len(self.result.scheduled_job_ids)

    def summarize(self) -> str:
        return (
            f"{self.strategy.name:<18} {self.jobs_served} served, "
            f"{self.changes} change(s), {self.customer_calls} call(s)"
        )


@dataclass(frozen=True, slots=True)
class RepairOptions:
    candidates: tuple[RepairCandidate, ...]
    baseline: PlanVersion

    @property
    def best_by_fewest_calls(self) -> RepairCandidate | None:
        """Default recommendation: keep promises first, then serve as much as possible.

        Deliberately not "lowest cost". A plan that saves forty dollars by moving an
        appointment someone booked time off for is not the cheaper plan once the
        phone call and the lost goodwill are counted, and those are not in the
        objective.
        """
        if not self.candidates:
            return None
        return min(
            self.candidates,
            key=lambda c: (c.customer_calls, -c.jobs_served, c.changes),
        )


def locked_jobs(world: WorldState) -> list[JobId]:
    """Work that is physically fixed: a crew is on site or has finished."""
    return sorted(
        job.id
        for job in world.jobs.values()
        if job.commitment_state in {CommitmentState.DISPATCHED, CommitmentState.COMPLETED}
    )


def underway_jobs(baseline: PlanVersion | None, world: WorldState, now: datetime) -> list[JobId]:
    """Jobs the CLOCK has dispatched, whether or not anyone typed the paperwork.

    `locked_jobs` trusts the event log: a stop is immovable once somebody recorded
    job-dispatched. Nobody records that minute-by-minute on a real morning - and a
    re-plan at 8:30 was free to hand Priya's 8-to-10 install to Dan while she stood
    in the customer's hallway. So the committed plan's own times are consulted too:
    once a stop's crew would already be rolling toward it (departure time reached),
    the job is as physically fixed as a dispatched one, and it is carved out and
    carried verbatim instead of re-planned.
    """
    if baseline is None:
        return []
    out: set[JobId] = set()
    for route in baseline.routes:
        for stop in route.stops:
            rolling_from = stop.arrival - timedelta(
                minutes=stop.run_break_minutes + stop.travel_minutes_from_prev
            )
            # In motion NOW: rolling toward the door or standing inside it. A stop
            # already finished is history the solver cannot re-place anyway, and a
            # stop later today is still honestly movable - the lock is exactly as
            # wide as the physical fact it models.
            if rolling_from <= now < stop.departure:
                job = world.jobs.get(stop.job_id)
                if job is not None and job.is_active:
                    out.add(stop.job_id)
    return sorted(out)


def repair_plan(
    *,
    world: WorldState,
    travel: TravelOracle,
    baseline: PlanVersion,
    start: date,
    params: SolveParams,
    horizon_params: HorizonParams,
    strategies: Sequence[RepairStrategy] = STRATEGIES,
    also_locked: Sequence[JobId] = (),
) -> RepairOptions:
    """Produce one repaired candidate per strategy, each diffed against the committed plan."""
    candidates: list[RepairCandidate] = []
    locked = sorted({*locked_jobs(world), *also_locked})
    planning_world, carried = carve_out_locked(world, baseline, locked)

    for index, strategy in enumerate(strategies):
        tuned = replace(
            params,
            allow_overtime=strategy.allow_overtime,
            unserved_penalty_base=params.unserved_penalty_base * strategy.unserved_multiplier,
            # Releasing a promise is priced through the same commitment weight the
            # customer's words produced; the strategy only scales how dearly.
            reschedule_penalty_multiplier=strategy.release_multiplier,
            change_penalty=strategy.change_penalty,
            incumbent=_incumbent_of(baseline),
            # Under pressure, breaking a promise is sometimes the least bad option.
            # It is allowed here, priced through commitment_cost, and reported.
            allow_promise_release=True,
        )
        result = plan_horizon(
            world=planning_world,
            travel=travel,
            start=start,
            params=tuned,
            horizon_params=horizon_params,
        )
        plan = PlanVersion(
            id=f"{baseline.id}-repair-{index}",
            parent_id=baseline.id,
            created_at=world.as_of,
            horizon_start=baseline.horizon_start,
            horizon_end=baseline.horizon_end,
            routes=(*carried, *result.routes),
            unserved=result.unserved,
            label=strategy.name,
        )
        candidates.append(
            RepairCandidate(
                strategy=strategy,
                plan=plan,
                result=result,
                diff=diff_plans(baseline, plan, world),
                released_promises=result.released_promises,
            )
        )

    return RepairOptions(candidates=tuple(candidates), baseline=baseline)


def _days_of(plan: PlanVersion) -> dict[JobId, date]:
    """Which day each job currently sits on. Locked work keeps it."""
    return {stop.job_id: route.date for route in plan.routes for stop in route.stops}


def _incumbent_of(plan: PlanVersion) -> dict[JobId, tuple[str, int]]:
    """Where each job sits today, as ``(van_id, minutes past midnight)``.

    The solver uses this to charge for change: staying put is free, moving is not.
    """
    out: dict[JobId, tuple[str, int]] = {}
    for route in plan.routes:
        for stop in route.stops:
            local = stop.arrival
            out[stop.job_id] = (route.van_id, local.hour * 60 + local.minute)
    return out


def carve_out_locked(
    world: WorldState,
    baseline: PlanVersion,
    locked: Sequence[JobId],
) -> tuple[WorldState, tuple[CrewRoute, ...]]:
    """Remove work that has already happened from the planning problem.

    An optimiser does not get to move a crew that is already on site, and it cannot
    reorder a morning that has been driven. So in-flight work is not modelled as a
    job to place - it is modelled as capacity that is gone, and its routes are carried
    into the repaired plan verbatim.

    Doing it this way also sidesteps a real trap. The solver prices travel
    pessimistically; a materialized arrival was computed at the actual departure time
    and is often earlier than the pessimistic matrix believes possible. Pinning that
    arrival as an equality makes the day unsatisfiable - which is exactly what
    happened before this existed.

    Each affected route contributes its prefix up to and including the last locked
    stop. Anything earlier on that route has already been driven past, so it is part
    of the same immovable past.
    """
    locked_set = set(locked)
    carried: list[CrewRoute] = []
    consumed_workers: dict[str, list[Unavailability]] = {}
    consumed_vans: dict[str, list[Unavailability]] = {}
    removed: set[JobId] = set()

    for route in baseline.routes:
        last_locked = max(
            (i for i, stop in enumerate(route.stops) if stop.job_id in locked_set),
            default=None,
        )
        if last_locked is None:
            continue

        prefix = route.stops[: last_locked + 1]
        carried.append(
            route.model_copy(
                update={"stops": prefix, "return_to_depot_minutes": 0, "return_to_depot_miles": 0.0}
            )
        )
        removed.update(stop.job_id for stop in prefix)

        # The crew and van are unavailable from leaving the depot until the last
        # locked stop is finished. Availability already knows how to intersect that
        # with a shift, so the rest of the day remains usable.
        busy_from = prefix[0].arrival - timedelta(minutes=prefix[0].travel_minutes_from_prev)
        busy_until = prefix[-1].departure
        outage = Unavailability(
            from_time=busy_from, until_time=busy_until, reason="already dispatched"
        )
        for worker_id in route.worker_ids:
            consumed_workers.setdefault(worker_id, []).append(outage)
        consumed_vans.setdefault(route.van_id, []).append(outage)

    reduced = copy(world)
    reduced.jobs = {k: v for k, v in world.jobs.items() if k not in removed}
    reduced.worker_outages = {
        k: [*world.worker_outages.get(k, []), *extra] for k, extra in consumed_workers.items()
    } | {k: v for k, v in world.worker_outages.items() if k not in consumed_workers}
    reduced.van_outages = {
        k: [*world.van_outages.get(k, []), *extra] for k, extra in consumed_vans.items()
    } | {k: v for k, v in world.van_outages.items() if k not in consumed_vans}
    return reduced, tuple(carried)
