"""Exact dollar cost of a materialized plan.

Distinct from the solver's objective on purpose. CP-SAT optimizes an approximation -
one pessimistic travel matrix, integer minutes, a blended labour rate - because that
is what keeps the model linear and fast. This module prices the plan that actually
came out, leg by leg, at the times those legs really happen.

The gap between the two is diagnostic rather than embarrassing: a large divergence
means the approximation is drifting from reality and the matrix or the bucket choice
needs attention. The CLI prints both side by side for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo

from krama.config import BusinessParams
from krama.domain.enums import NOT_A_FAILURE, WindowHardness
from krama.domain.models import CostBreakdown, CrewRoute, PlanVersion, UnservedJob
from krama.domain.state import WorldState


@dataclass(frozen=True, slots=True)
class RouteCost:
    """Per-crew detail, so an expensive day can be attributed to a specific route.

    Marginal cash, matching the solver's objective: fuel and overtime, not rostered
    wages. A fitter's shift is owed whether the van rolls or not, so charging their
    driving minutes here made an idle colleague look expensive to deploy - which is
    how every job of a day once piled onto one fitter's overtime while five rostered
    people sat free. Payroll belongs in the quote's cost-to-serve, where per-job
    profitability is judged; it does not belong in the number that decides routes.
    """

    crew_id: str
    vehicle: float
    overtime: float
    lateness: float
    travel_minutes: int
    person_travel_minutes: int
    travel_miles: float
    working_minutes: int
    idle_minutes: int
    overtime_minutes: int

    @property
    def total(self) -> float:
        return self.vehicle + self.overtime + self.lateness

    @property
    def utilization(self) -> float:
        """Share of the crew's day spent on site rather than driving or waiting."""
        span = self.working_minutes
        return 0.0 if span <= 0 else (span - self.travel_minutes - self.idle_minutes) / span


def _local(moment: datetime, tz: tzinfo) -> datetime:
    return moment.astimezone(tz)


def cost_route(
    route: CrewRoute,
    world: WorldState,
    business: BusinessParams,
    tz: tzinfo,
) -> RouteCost:
    rate = business.labor.loaded_rate_per_minute.value
    # The full multiplied rate, not the premium over base: past shift the whole
    # wage is marginal money, since off the clock the fitter costs nothing.
    ot_rate = rate * business.labor.overtime_multiplier.value
    late_rate = business.penalties.lateness_per_minute.value

    headcount = len(route.worker_ids)
    van = world.vans.get(route.van_id)
    per_mile = van.cost_per_mile if van else 0.0

    travel_minutes = route.total_travel_minutes
    person_travel = travel_minutes * headcount

    # Slack: crew-minutes that are neither driving nor on site, because the next
    # window has not opened yet. The crew leaves late rather than idling on a
    # doorstep, but the time is unproductive either way, so it is surfaced rather
    # than buried - a lot of it usually means the sequence is wrong.
    idle = 0
    previous_departure: datetime | None = None
    for stop in route.stops:
        expected = (
            previous_departure
            + timedelta(minutes=stop.run_break_minutes + stop.travel_minutes_from_prev)
            if previous_departure
            else None
        )
        if expected is not None and stop.arrival > expected:
            idle += int((stop.arrival - expected).total_seconds() // 60)
        previous_departure = stop.departure

    if route.stops:
        first, last = route.stops[0], route.stops[-1]
        day_start = first.arrival - timedelta(minutes=first.travel_minutes_from_prev)
        day_end = last.departure + timedelta(minutes=route.return_to_depot_minutes)
        working = int((day_end - day_start).total_seconds() // 60)
    else:
        day_end, working = None, 0

    overtime_minutes = 0
    if day_end is not None:
        end_local = _local(day_end, tz)
        start_local = _local(day_start, tz)
        for worker_id in route.worker_ids:
            worker = world.workers.get(worker_id)
            if worker is None:
                continue
            hours = worker.hours_for(end_local.weekday())
            if hours is None:
                continue
            shift_end = end_local.replace(
                hour=hours.end.hour, minute=hours.end.minute, second=0, microsecond=0
            )
            # Minutes WORKED past the close, not minutes elapsed since it: an
            # evening session that leaves the shop at five pays its crew from
            # five, not from three. Mirrors the solver's objective exactly.
            overtime_from = max(start_local, shift_end)
            if end_local > overtime_from:
                overtime_minutes += int((end_local - overtime_from).total_seconds() // 60)

    lateness_minutes = 0
    for stop in route.stops:
        job = world.jobs.get(stop.job_id)
        if job is None:
            continue
        soft = [w for w in job.windows if w.hardness is WindowHardness.SOFT]
        if not soft:
            continue
        overshoot = min(
            max(0, int((stop.departure - window.end).total_seconds() // 60)) for window in soft
        )
        lateness_minutes += overshoot

    return RouteCost(
        crew_id=route.crew_id,
        vehicle=route.total_travel_miles * per_mile,
        overtime=overtime_minutes * ot_rate,
        lateness=lateness_minutes * late_rate,
        travel_minutes=travel_minutes,
        person_travel_minutes=person_travel,
        travel_miles=route.total_travel_miles,
        working_minutes=working,
        idle_minutes=idle,
        overtime_minutes=overtime_minutes,
    )


def unserved_cost(
    unserved: tuple[UnservedJob, ...],
    world: WorldState,
    business: BusinessParams,
) -> float:
    """Penalty for work this solve could have placed but did not.

    Mirrors the solver's ``_unserved_penalty`` so the two stay comparable; if that
    formula changes, this must change with it. Structural exclusions
    (:data:`~krama.domain.enums.NOT_A_FAILURE`) are skipped for the same reason
    the solver never made them candidates.
    """
    priority_multiplier = {"emergency": 6.0, "high": 2.5, "normal": 1.0, "low": 0.6}
    base = business.penalties.unserved_base.value
    escalation = business.penalties.deferral_escalation.value
    revenue_weight = business.penalties.revenue_weight.value

    total = 0.0
    for item in unserved:
        if item.reason in NOT_A_FAILURE:
            continue
        job = world.jobs.get(item.job_id)
        if job is None:
            continue
        total += (
            base * priority_multiplier[job.priority.value]
            + escalation * job.deferral_count
            + revenue_weight * job.revenue
        )
    return total


def cost_plan(
    plan: PlanVersion,
    world: WorldState,
    business: BusinessParams,
    tz: tzinfo,
    unserved: tuple[UnservedJob, ...] = (),
) -> tuple[CostBreakdown, list[RouteCost]]:
    """Exact cost of the plan as materialized, plus per-crew attribution."""
    per_route = [cost_route(route, world, business, tz) for route in plan.routes]
    breakdown = CostBreakdown(
        vehicle=sum(r.vehicle for r in per_route),
        overtime=sum(r.overtime for r in per_route),
        lateness_penalty=sum(r.lateness for r in per_route),
        unserved_penalty=unserved_cost(unserved, world, business),
    )
    return breakdown, per_route
