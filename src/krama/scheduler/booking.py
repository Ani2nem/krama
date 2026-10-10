"""What does it cost us to serve this job, and when is it cheapest?

This is the highest-value thing the deterministic engine does, and the cleanest
illustration of the split the whole project rests on. When a call comes in, the
useful question is not "what is the optimal schedule" - it is "where does this job
fit, and what does each option cost?" The agent runs the conversation; the solver
computes the money.

The method is marginal-cost insertion. Hold the committed plan fixed, try the job in
each feasible day, and price the difference. Concretely, for a 90-minute residential
job out in one direction:

    Tue 14:00   already two stops on that street that afternoon, +18 min detour   $22
    Wed 09:00   moderate detour                                                   $41
    Thu 08:00   a dedicated trip out and back                                     $82

A dispatcher can then say "I can do Tuesday afternoon or Thursday morning" and know
that Tuesday is not a preference, it is four times cheaper to serve. Over a year that
is the difference between 22 and 28 jobs a week on the same headcount.

Two things make the number trustworthy. Existing work is *locked*, so a quote can
never be cheap because it silently displaced a customer who was already promised a
slot. And the delta is computed from materialized routes rather than the solver's
objective, so it is real dollars of driving and labour rather than a score that
includes penalty terms for the job not yet existing.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, tzinfo

from krama.config import BusinessParams
from krama.domain.enums import UnservedReason, WindowHardness
from krama.domain.models import Job, JobId, TimeWindow
from krama.domain.state import WorldState
from krama.domain.travel import TravelOracle
from krama.formatting import clock_range
from krama.scheduler.costing import RouteCost, cost_route
from krama.scheduler.day_planner import SolveParams, plan_day


@dataclass(frozen=True, slots=True)
class SlotSuggestion:
    """One bookable option, priced and explained."""

    on_date: date
    arrival: datetime
    quoted_window: TimeWindow
    marginal_cost: float
    crew_id: str
    worker_names: tuple[str, ...]
    #: The same people as ids - what a booking pins so the commit cannot quietly
    #: swap in an equal-cost somebody else after the card named these names.
    worker_ids: tuple[str, ...]
    added_travel_minutes: int
    added_travel_miles: float
    reason: str
    #: Minutes this placement pushes a crew past its shift. Charged at the after-hours
    #: rate, because it is paid at the overtime rate.
    overtime_minutes: int = 0
    #: Where this job's own drive begins in the trial route - "the shop", or the
    #: previous customer's name - plus that leg's minutes and miles. The detail
    #: card already said it after booking; the owner wanted it BEFORE, on every
    #: card, which is fair: the drive is part of what is being sold.
    from_label: str = "the shop"
    leg_minutes: int = 0
    leg_miles: float = 0.0

    def describe(self, tz: tzinfo) -> str:
        start = self.quoted_window.start.astimezone(tz)
        end = self.quoted_window.end.astimezone(tz)
        return (
            f"{start:%a %d %b} {clock_range(start, end)}  ${self.marginal_cost:,.2f}  {self.reason}"
        )


@dataclass(frozen=True, slots=True)
class UnavailableDay:
    """A day the job cannot be served on, with the reason a dispatcher can act on."""

    on_date: date
    reason: UnservedReason
    detail: str


@dataclass(frozen=True, slots=True)
class BookingOptions:
    slots: tuple[SlotSuggestion, ...]
    unavailable: tuple[UnavailableDay, ...]
    evaluated_days: int

    @property
    def best(self) -> SlotSuggestion | None:
        return self.slots[0] if self.slots else None

    @property
    def savings_vs_worst(self) -> float:
        """What choosing well is worth. The number that makes this feature pay.

        Computed over costs, not positions - the list is ordered by date now, so
        first-minus-last would measure the calendar.
        """
        if len(self.slots) < 2:
            return 0.0
        costs = [s.marginal_cost for s in self.slots]
        return max(costs) - min(costs)


@dataclass(frozen=True, slots=True)
class BaselineDay:
    """What the day costs before the new job is inserted."""

    operating_cost: float
    served: frozenset[JobId]
    travel_minutes: int
    travel_miles: float
    overtime_minutes: int = 0


@dataclass
class BaselineCache:
    """The cost of a day as it already stands, remembered between quotes.

    Half of a quote is re-solving the untouched day, and that answer does not change
    between one caller and the next: the plan only moves when something is committed.
    Caching it halves the wait, which at the real operating point is the difference
    between three seconds and a second and a half - and three seconds is where a
    dispatcher starts apologising for the pause.

    Keyed on the day and the exact set of jobs in it, so a booking, a cancellation or
    a disruption invalidates it by construction rather than by remembering to.
    """

    _entries: dict[tuple[date, frozenset[JobId], float], BaselineDay] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, on_date: date, jobs: Sequence[JobId], budget: float) -> BaselineDay | None:
        found = self._entries.get((on_date, frozenset(jobs), budget))
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, on_date: date, jobs: Sequence[JobId], budget: float, day: BaselineDay) -> None:
        self._entries[(on_date, frozenset(jobs), budget)] = day


def _world_with(world: WorldState, job: Job) -> WorldState:
    """A shallow copy of the world with one extra job. The original is untouched."""
    clone = copy(world)
    clone.jobs = {**world.jobs, job.id: job}
    return clone


def _route_operating_cost(route_cost: RouteCost) -> float:
    """Driving and labour only - the marginal cost of *serving*, not of penalties."""
    return float(route_cost.vehicle + route_cost.overtime)


def _reason_for(
    added_minutes: int,
    added_miles: float,
    neighbours: int,
    dedicated: bool,
) -> str:
    """Explain the price in the terms a dispatcher would use on the phone."""
    if dedicated:
        return f"a dedicated trip out and back, +{added_minutes} min driving"
    if neighbours and added_minutes == 0:
        # Same ~150m cell as work already booked: below the travel cache's resolution,
        # so the detour genuinely is free. Saying "+0 min" reads like a bug.
        return f"on the same block as {neighbours} stop(s) already booked that day"
    if neighbours and added_minutes <= 20:
        near = "stop" if neighbours == 1 else "stops"
        return f"already {neighbours} {near} nearby that day, +{added_minutes} min detour"
    if neighbours:
        return f"{neighbours} other stop(s) that day, +{added_minutes} min detour"
    return f"+{added_minutes} min / {added_miles:.1f} mi added to the day"


def _within_preferred_hours(
    draft: Job,
    on_date: date,
    tz: tzinfo,
    earliest_hour: int | None,
    latest_hour: int | None,
    not_before: datetime | None = None,
) -> Job:
    """The draft with the caller's stated hours attached as a window for this day.

    Soft, not hard: the planner may finish a few minutes past a stated preference and
    price the lateness, which is the right trade when the alternative is refusing a
    booking over four minutes. A genuinely unbreakable constraint - "before we open" -
    arrives as a hard_constraint and is a separate thing.
    """
    # "Now" is a constraint like any other. At 9:14 on a Wednesday evening the board
    # offered "Wed, arrive 4:00 PM" - five hours in the past - because nothing ever
    # told the search what time it was. The floor applies only to the day it is on.
    floor = not_before if (not_before and not_before.astimezone(tz).date() == on_date) else None
    if earliest_hour is None and latest_hour is None and floor is None:
        return draft
    opens = datetime.combine(on_date, time(earliest_hour or 0), tzinfo=tz)
    if floor is not None:
        opens = max(opens, floor)
    closes = datetime.combine(on_date, time(latest_hour or 23, 59), tzinfo=tz)
    return draft.model_copy(
        update={"windows": (TimeWindow(start=opens, end=closes, hardness=WindowHardness.SOFT),)}
    )


def suggest_booking_slots(
    *,
    world: WorldState,
    travel: TravelOracle,
    draft: Job,
    horizon: list[date],
    params: SolveParams,
    business: BusinessParams,
    limit: int = 5,
    cache: BaselineCache | None = None,
    earliest_hour: int | None = None,
    latest_hour: int | None = None,
    not_before: datetime | None = None,
) -> BookingOptions:
    """Rank the days this job could be served on by what serving it actually costs.

    ``earliest_hour`` and ``latest_hour`` are the customer's own words turned into
    numbers - "free after four" is 16. They constrain the search rather than decorating
    it: a caller who says four in the afternoon and is offered six in the morning on
    every day of the week has been ignored, however good the price was.
    """
    tz = params.business_tz
    quoted_minutes = int(business.scheduling.quoted_window_minutes.value)

    # A quote is two solves per day with a caller waiting, so it gets its own ceiling.
    # Inheriting the batch budget made a twenty-five job day take twenty seconds to
    # answer, which is not a feature anybody would use.
    #
    # Unless reproducibility was asked for. Clamping the wall clock below a
    # deterministic budget makes the clock the binding limit again and quietly undoes
    # it - a caller who wants the same answer every time has accepted the longer wait.
    if params.max_deterministic_time is None:
        params = replace(params, max_solve_seconds=business.solver.quote_solve_seconds.value)

    slots: list[SlotSuggestion] = []
    unavailable: list[UnavailableDay] = []

    for on_date in horizon:
        # The customer's window, as a window on this particular day. Applied to the
        # draft rather than to the search, so the solver enforces it the same way it
        # enforces every other promised window - and so a day where it cannot be met
        # reports that rather than quietly offering something else.
        wanted = _within_preferred_hours(draft, on_date, tz, earliest_hour, latest_hour, not_before)

        existing = [
            job.id
            for job in world.schedulable_jobs()
            if any(
                w.start.astimezone(tz).date() == on_date
                or (w.start.date() <= on_date <= w.end.date())
                for w in job.windows
            )
        ]

        remembered = cache.get(on_date, existing, params.max_solve_seconds) if cache else None
        if remembered is None:
            baseline = plan_day(
                world=world,
                travel=travel,
                on_date=on_date,
                candidate_job_ids=existing,
                params=params,
            )
            baseline_costs = [cost_route(route, world, business, tz) for route in baseline.routes]
            remembered = BaselineDay(
                operating_cost=sum(_route_operating_cost(c) for c in baseline_costs),
                served=frozenset(j for route in baseline.routes for j in route.job_ids),
                travel_minutes=sum(r.total_travel_minutes for r in baseline.routes),
                travel_miles=sum(r.total_travel_miles for r in baseline.routes),
                overtime_minutes=sum(c.overtime_minutes for c in baseline_costs),
            )
            if cache is not None:
                cache.put(on_date, existing, params.max_solve_seconds, remembered)

        # Everything already placed stays placed. A quote must never look cheap
        # because it quietly displaced someone who was already promised a slot.
        candidate_world = _world_with(world, wanted)
        # The draft is LOCKED, not merely offered. A quote answers "can we serve
        # this and at what cost", never "is this customer worth serving" - but an
        # unlocked draft with no revenue on it yet was exactly that second question,
        # and the solver answered it: a 600-minute overtime evening cost more than
        # the unserved penalty, so the trial dropped the caller and the board said
        # "no room" about a slot two fitters had just agreed to work.
        trial = plan_day(
            world=candidate_world,
            travel=travel,
            on_date=on_date,
            candidate_job_ids=[*existing, draft.id],
            params=params,
            locked_job_ids=sorted({*remembered.served, draft.id}),
        )

        placement = next(
            (
                (route, stop)
                for route in trial.routes
                for stop in route.stops
                if stop.job_id == draft.id
            ),
            None,
        )
        if placement is None:
            miss = next((u for u in trial.unserved if u.job_id == draft.id), None)
            unavailable.append(
                UnavailableDay(
                    on_date=on_date,
                    reason=miss.reason if miss else UnservedReason.NO_CAPACITY_IN_HORIZON,
                    detail=miss.detail if miss else "no room on this day",
                )
            )
            continue

        route, stop = placement

        # A yes is for a customer, not a blank cheque. The day-solve honours a
        # late-stay agreement for the whole day (it must - the job it backs is in
        # the same solve), so a placement that only works because of someone ELSE'S
        # extension slips through feasibility. Catch it here: if this draft's stop
        # runs past a crew member's ordinary reach (shift plus standard overtime)
        # and that member's extension was not agreed for this caller, the day is
        # not offered - the dispatcher asks the crew again instead of assuming.
        borrowed = None
        stop_end = stop.arrival + timedelta(minutes=wanted.estimated_duration_min)
        for worker_id in route.worker_ids:
            member = world.workers.get(worker_id)
            hours = member.hours_for(on_date.weekday()) if member else None
            if member is None or hours is None:
                continue
            ordinary = datetime.combine(on_date, hours.end, tzinfo=tz)
            if member.overtime_eligible and params.allow_overtime:
                ordinary += timedelta(minutes=params.overtime_minutes)
            if stop_end > ordinary and not world.extension_covers(worker_id, on_date, draft.phone):
                borrowed = (member.name, world.extension_customer(worker_id, on_date))
                break
        if borrowed is not None:
            name, for_whom = borrowed
            whose = f" for {for_whom}" if for_whom else ""
            unavailable.append(
                UnavailableDay(
                    on_date=on_date,
                    reason=UnservedReason.NO_CAPACITY_IN_HORIZON,
                    detail=(
                        f"these hours only work because {name} agreed to stay late{whose} - "
                        "ask the crew again for this job"
                    ),
                )
            )
            continue

        before = route.stops[: list(route.stops).index(stop)]
        previous = candidate_world.jobs.get(before[-1].job_id) if before else None
        from_label = f"{previous.customer_name}'s" if previous else "the shop"
        trial_costs = [cost_route(r, candidate_world, business, tz) for r in trial.routes]
        trial_cost = sum(_route_operating_cost(c) for c in trial_costs)
        trial_overtime = sum(c.overtime_minutes for c in trial_costs)
        # Travel deltas come from the trial alone: the baseline's route objects are
        # not kept, only its cost, which is all the marginal figure needs.
        added_minutes = max(
            0, sum(r.total_travel_minutes for r in trial.routes) - remembered.travel_minutes
        )
        added_miles = max(
            0.0, sum(r.total_travel_miles for r in trial.routes) - remembered.travel_miles
        )

        neighbours = len(route.stops) - 1
        dedicated = neighbours == 0
        # The window opens at the estimate and runs forward, rather than straddling
        # it. Centred, a 6:27 arrival was quoted to the customer as "5:27 to 7:27",
        # which is not a time anybody would read down a phone - and it promised a
        # half-hour earlier than the crew could possibly manage, which is the wrong
        # direction to be wrong in. Rounded back to the quarter hour because "we'll be
        # there between 6:15 and 8:15" is what a person says.
        opens = stop.arrival.replace(minute=stop.arrival.minute // 15 * 15, second=0, microsecond=0)
        window = TimeWindow(start=opens, end=opens + timedelta(minutes=quoted_minutes))

        slots.append(
            SlotSuggestion(
                on_date=on_date,
                arrival=stop.arrival,
                quoted_window=window,
                marginal_cost=max(0.0, trial_cost - remembered.operating_cost),
                crew_id=route.crew_id,
                worker_names=tuple(
                    world.workers[w].name for w in route.worker_ids if w in world.workers
                ),
                worker_ids=tuple(route.worker_ids),
                from_label=from_label,
                leg_minutes=stop.travel_minutes_from_prev,
                leg_miles=round(stop.travel_miles_from_prev, 2),
                added_travel_minutes=added_minutes,
                added_travel_miles=added_miles,
                reason=_reason_for(added_minutes, added_miles, neighbours, dedicated),
                # Only the overtime this job adds. A day already running late is not
                # this customer's bill.
                overtime_minutes=max(0, trial_overtime - remembered.overtime_minutes),
            )
        )

    # Soonest first. The caller on the phone wants a date; cost ranks the *selection*
    # (which five make the cut) but a list that opens with next Monday and buries this
    # Friday at the bottom reads as broken, because for the customer it is. The
    # cheaper-if-flexible tab is where cost ordering lives.
    slots.sort(key=lambda s: (s.marginal_cost, s.on_date))
    kept = slots[:limit]
    kept.sort(key=lambda s: (s.on_date, s.arrival))
    return BookingOptions(
        slots=tuple(kept),
        unavailable=tuple(unavailable),
        evaluated_days=len(horizon),
    )
