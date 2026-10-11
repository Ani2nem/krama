"""Turning a *sequence* of jobs into a concrete route with real clock times.

The solver decides which jobs a crew does and in what order. This module decides
what that costs in wall-clock terms: when the van leaves, how long each drive takes
at the time of day it actually happens, and when the crew gets home.

Keeping it separate matters because the solver optimizes over an approximation
(integer minutes, a snapshot matrix) while this replays the sequence against the
travel oracle at the real departure time of every leg. Anything the approximation
got wrong shows up here as a later arrival, and then the invariant checker catches
it. Solver, materializer, and checker are three independent views of the same plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta

from krama.domain.enums import WindowHardness
from krama.domain.models import (
    CrewRoute,
    Job,
    JobId,
    Location,
    Stop,
    VanId,
    WorkerId,
)
from krama.domain.state import WorldState
from krama.domain.travel import TravelLeg, TravelOracle

#: Idle this long at a customer's kerb and the crew drives back to the shop
#: instead - nobody waits out a four-hour gap in somebody's driveway. The split
#: is a materialization fact, not a solver decision: the sequence is unchanged,
#: but the afternoon stop becomes a fresh departure from the depot, and both
#: extra legs (home, and out again) are priced as the real driving they are.
RUN_BREAK_AFTER_MINUTES = 90


def _earliest_start(job: Job, arrival: datetime) -> datetime:
    """When service can actually begin, accounting for waiting at a closed door.

    Picks the first declared window the crew can still satisfy after arriving; if
    they are early, they wait. Returns the raw arrival when no window fits, so the
    infeasibility surfaces in the invariant checker rather than being silently
    papered over here.
    """
    duration = timedelta(minutes=job.estimated_duration_min)
    best: datetime | None = None
    for window in job.windows:
        start = max(arrival, window.start)
        fits = window.hardness is WindowHardness.SOFT or start + duration <= window.end
        if fits and (best is None or start < best):
            best = start
    return best if best is not None else arrival


def _depart_and_arrive(
    travel: TravelOracle,
    origin: Location,
    job: Job,
    ready_at: datetime,
    window_released: bool = False,
) -> tuple[TravelLeg, datetime]:
    """Choose a departure time and the resulting arrival.

    A crew that would reach a customer before their window opens does not idle on the
    doorstep - it leaves later. That matters for more than realism: travel time depends
    on *departure* time, so pricing a leg at the moment the crew became free while the
    schedule implies they left forty minutes later produces a stored travel time that
    no longer matches the road. The invariant checker recomputes each leg from
    ``arrival - travel_minutes``, and correctly rejects the mismatch.

    So the departure is solved as a fixed point: leave at ``target - travel(departure)``,
    re-evaluating travel at that departure until it settles. Traffic buckets are coarse,
    so this converges in a step or two; the loop is bounded and falls back to the
    honest "leave immediately and wait" answer if it does not.
    """
    leg = travel.leg(origin, job.location, ready_at)
    naive_arrival = ready_at + timedelta(minutes=leg.minutes)
    # A RELEASED promise is served when the crew gets there, not when the window
    # says. The solver only releases when the window genuinely cannot be kept
    # (and prices it); waiting for it anyway re-broke a fitter's availability -
    # Priya's freed 7:30 start was "politely" pushed to 8:00, straight into the
    # dentist appointment the early start existed to avoid.
    target = naive_arrival if window_released else _earliest_start(job, naive_arrival)
    if target <= naive_arrival:
        return leg, target

    depart = target - timedelta(minutes=leg.minutes)
    for _ in range(3):
        if depart <= ready_at:
            return leg, max(target, naive_arrival)
        candidate = travel.leg(origin, job.location, depart)
        next_depart = target - timedelta(minutes=candidate.minutes)
        if next_depart == depart:
            return candidate, target
        depart = next_depart

    settled = travel.leg(origin, job.location, max(depart, ready_at))
    return settled, max(target, max(depart, ready_at) + timedelta(minutes=settled.minutes))


def materialize_route(
    *,
    world: WorldState,
    travel: TravelOracle,
    crew_id: str,
    worker_ids: Sequence[WorkerId],
    van_id: VanId,
    on_date: date,
    job_sequence: Sequence[JobId],
    shift_start: datetime,
    released: frozenset[JobId] = frozenset(),
) -> CrewRoute:
    """Replay a job sequence against the travel oracle to produce concrete stops.

    ``shift_start`` is when the crew leaves the depot. Every leg is priced at its own
    departure time, so a route that crosses into rush hour pays rush-hour minutes.
    """
    van = world.vans[van_id]
    position = van.home_depot
    clock = shift_start
    stops: list[Stop] = []

    for job_id in job_sequence:
        job = world.jobs[job_id]
        leg, arrival = _depart_and_arrive(
            travel, position, job, clock, window_released=job_id in released
        )
        fresh_run = False
        break_minutes, break_miles = 0, 0.0
        idle = arrival - timedelta(minutes=leg.minutes) - clock
        if stops and idle >= timedelta(minutes=RUN_BREAK_AFTER_MINUTES):
            home_leg = travel.leg(position, van.home_depot, clock)
            ready_at_shop = clock + timedelta(minutes=home_leg.minutes)
            out_leg, out_arrival = _depart_and_arrive(
                travel, van.home_depot, job, ready_at_shop, window_released=job_id in released
            )
            # Only split when the round trip genuinely fits the gap - a depot on
            # the far side of town must not make the customer later.
            if out_arrival <= arrival:
                fresh_run = True
                break_minutes, break_miles = home_leg.minutes, home_leg.miles
                leg, arrival = out_leg, out_arrival
        departure = arrival + timedelta(minutes=job.estimated_duration_min)
        stops.append(
            Stop(
                job_id=job_id,
                arrival=arrival,
                departure=departure,
                travel_minutes_from_prev=leg.minutes,
                travel_miles_from_prev=leg.miles,
                fresh_run=fresh_run,
                run_break_minutes=break_minutes,
                run_break_miles=break_miles,
            )
        )
        position = job.location
        clock = departure

    home = travel.leg(position, van.home_depot, clock) if stops else None
    return CrewRoute(
        crew_id=crew_id,
        date=on_date,
        worker_ids=tuple(worker_ids),
        van_id=van_id,
        stops=tuple(stops),
        return_to_depot_minutes=home.minutes if home else 0,
        return_to_depot_miles=home.miles if home else 0.0,
    )
