"""Marginal-cost booking suggestions.

Drafts reuse existing fixture locations so every leg is in the frozen snapshot -
these tests run on real road distances, offline.

The property that matters most is the displacement one: a quote must never look
cheap because it quietly moved a customer who had already been promised a slot.
"""

from __future__ import annotations

from datetime import date

import pytest

from krama.config import BusinessParams
from krama.domain.enums import Certification, ServiceType
from krama.domain.models import Job, TimeWindow
from krama.domain.state import WorldState
from krama.fixtures.sample_business import WEEK_START, _at
from krama.scheduler.booking import BookingOptions, suggest_booking_slots
from krama.scheduler.day_planner import SolveParams


@pytest.fixture
def business() -> BusinessParams:
    return BusinessParams.load()


@pytest.fixture
def params(business: BusinessParams) -> SolveParams:
    from zoneinfo import ZoneInfo

    return SolveParams.from_business(business, ZoneInfo(business.meta.timezone))


HORIZON = [date.fromordinal(WEEK_START.toordinal() + i) for i in range(5)]


def draft_at(
    world: WorldState,
    like_job: str,
    *,
    certs: set[Certification] | None = None,
    duration: int = 90,
    crew: int = 1,
) -> Job:
    """A prospective job at an existing site, so the frozen snapshot covers it."""
    return Job(
        id="draft",
        customer_id="c-draft",
        customer_name="New caller",
        location=world.jobs[like_job].location,
        service_type=ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT,
        required_certifications=frozenset(
            certs if certs is not None else {Certification.RESIDENTIAL_GLAZING}
        ),
        crew_size=crew,
        estimated_duration_min=duration,
        revenue=700.0,
        windows=tuple(TimeWindow(start=_at(d, 8), end=_at(d, 17)) for d in range(5)),
        requested_at=_at(0, 7),
    )


def quote(world, travel, params, business, draft: Job) -> BookingOptions:
    return suggest_booking_slots(
        world=world,
        travel=travel,
        draft=draft,
        horizon=HORIZON,
        params=params,
        business=business,
    )


# --------------------------------------------------------------------- ranking


def test_slots_arrive_soonest_first(world, travel, params, business):
    """The caller on the phone wants a date. A list that opened with next Monday and
    buried this Friday at the bottom read as broken, because for the customer it was.
    Cost still decides *which* days make the cut; the calendar decides the order."""
    options = quote(world, travel, params, business, draft_at(world, "j-402"))
    dates = [s.on_date for s in options.slots]
    assert dates == sorted(dates)
    assert options.best is options.slots[0]


def test_clustering_is_cheaper_than_an_isolated_trip(world, travel, params, business):
    """The whole point. A job beside existing stops costs a detour; a job on its own
    costs a round trip, and the difference is what a dispatcher should be steering by."""
    near = quote(world, travel, params, business, draft_at(world, "j-402"))
    assert near.best is not None
    costs = [s.marginal_cost for s in near.slots]
    assert min(costs) < max(costs), "a shared day and a dedicated trip price apart"
    assert near.savings_vs_worst == max(costs) - min(costs)


def test_the_cheapest_slot_shares_a_day_with_nearby_work(world, travel, params, business):
    options = quote(world, travel, params, business, draft_at(world, "j-402"))
    cheapest = min(options.slots, key=lambda s: s.marginal_cost)
    dearest = max(options.slots, key=lambda s: s.marginal_cost)
    assert cheapest.added_travel_minutes < dearest.added_travel_minutes


def test_a_dedicated_trip_is_explained_as_one(world, travel, params, business):
    """A day with nothing else on it means the whole round trip is this job's.

    The assertion used to be that a dedicated trip is the dearest option, which was
    true of the old geography and is not a property of anything. Moving the business
    produced a day costing twice the dedicated trip on six minutes of extra driving:
    it was the fullest day in the week, and the insertion tipped a crew into overtime.
    Cheaper to drive half an hour on an empty Friday than to buy an hour of overtime
    on a full Monday, which is the sort of thing the dollar objective exists to notice.

    What a dedicated trip does guarantee is that none of its travel is shared, so its
    *added travel* is the largest. That is the property worth asserting.
    """
    options = quote(world, travel, params, business, draft_at(world, "j-402"))
    lonely = [s for s in options.slots if "dedicated trip" in s.reason]
    assert lonely, "an otherwise-empty day should be described as a dedicated trip"
    assert lonely[-1].added_travel_minutes == max(s.added_travel_minutes for s in options.slots)


# ------------------------------------------------------------------ safety


def test_quoting_never_displaces_already_placed_work(world, travel, params, business):
    """A quote that looked cheap because it bumped a promised customer would be
    worse than useless. Existing work is locked while the trial insertion runs."""
    from krama.scheduler.day_planner import plan_day

    draft = draft_at(world, "j-402")
    baseline = plan_day(
        world=world,
        travel=travel,
        on_date=WEEK_START,
        candidate_job_ids=[j.id for j in world.active_jobs()],
        params=params,
    )
    placed_before = {j for route in baseline.routes for j in route.job_ids}

    options = quote(world, travel, params, business, draft)
    monday = [s for s in options.slots if s.on_date == WEEK_START]
    assert monday, "Monday should be bookable"

    after = plan_day(
        world=world,
        travel=travel,
        on_date=WEEK_START,
        candidate_job_ids=[j.id for j in world.active_jobs()],
        params=params,
    )
    assert {j for route in after.routes for j in route.job_ids} == placed_before


def test_marginal_cost_is_never_negative(world, travel, params, business):
    """Adding work cannot make a day cheaper to run."""
    options = quote(world, travel, params, business, draft_at(world, "j-402"))
    assert all(s.marginal_cost >= 0 for s in options.slots)


def test_quoted_window_is_the_configured_width_and_contains_the_arrival(
    world, travel, params, business
):
    options = quote(world, travel, params, business, draft_at(world, "j-402"))
    expected = business.scheduling.quoted_window_minutes.value
    for slot in options.slots:
        width = (slot.quoted_window.end - slot.quoted_window.start).total_seconds() / 60
        assert width == expected
        assert slot.quoted_window.contains(slot.arrival)


# ----------------------------------------------------------------- constraints


def test_a_scarce_certification_gates_every_offered_crew(world, travel, params, business):
    """Only Marcus and Priya hold the commercial certification. Certifications pool
    across a crew, so the second seat can be anyone - but one of those two must be on
    every crew offered, which is what makes storefront work the bottleneck."""
    draft = draft_at(
        world, "j-401", certs={Certification.COMMERCIAL_STOREFRONT}, duration=120, crew=2
    )
    options = quote(world, travel, params, business, draft)
    assert options.slots
    for slot in options.slots:
        assert len(slot.worker_names) == 2
        assert {"Marcus", "Priya"} & set(slot.worker_names), (
            f"{slot.worker_names} has nobody commercially certified"
        )


def test_unbookable_days_are_explained(world, travel, params, business):
    """A job needing a certification nobody has should report why, not go silent."""
    from krama.domain.state import Unavailability

    for worker_id in ("w-marcus", "w-priya"):
        world.worker_outages[worker_id] = [
            Unavailability(from_time=_at(0, 0), until_time=_at(5, 0), reason="out")
        ]
    draft = draft_at(
        world, "j-401", certs={Certification.COMMERCIAL_STOREFRONT}, duration=120, crew=2
    )
    options = quote(world, travel, params, business, draft)
    assert options.slots == ()
    assert options.unavailable
    assert all(u.detail for u in options.unavailable)


def test_every_day_in_the_horizon_is_considered(world, travel, params, business):
    options = quote(world, travel, params, business, draft_at(world, "j-402"))
    assert options.evaluated_days == len(HORIZON)
    assert len(options.slots) + len(options.unavailable) == len(HORIZON)


def test_a_fitter_held_for_a_day_is_not_offered_that_day(world, travel, params, business):
    """The rota promise: mark Dan out Monday and no Monday slot may contain Dan.

    Reported from the board as "I held Dan and said he was busy but the system booked
    him anyway" - the outage had been recorded against the wrong day, but the planner
    honouring a correctly-scoped day outage is the half worth pinning here.
    """
    from krama.domain.state import Unavailability

    world.worker_outages["w-dan"] = [
        Unavailability(from_time=_at(0, 0), until_time=_at(0, 23), reason="busy")
    ]
    # After four, residential: the slot that historically always landed on Dan.
    options = suggest_booking_slots(
        world=world,
        travel=travel,
        draft=draft_at(world, "j-402", duration=120),
        horizon=HORIZON,
        params=params,
        business=business,
        earliest_hour=16,
    )
    by_day = {s.on_date: s.worker_names for s in options.slots}
    monday = HORIZON[0]
    if monday in by_day:
        assert "Dan" not in by_day[monday], "a held fitter must not be booked"
    other_days = [names for day, names in by_day.items() if day != monday]
    assert other_days, "the rest of the week is unaffected by a one-day hold"


def test_every_slot_names_where_its_drive_begins(world, travel, params, business):
    """The drive is part of what is being sold, so the card says where it starts
    and what it costs - before booking, not only on the detail popup after."""
    options = quote(world, travel, params, business, draft_at(world, "j-402"))
    assert options.slots
    for slot in options.slots:
        assert slot.from_label, "every drive has a door it leaves from"
        assert slot.leg_minutes >= 0 and slot.leg_miles >= 0
    labels = {s.from_label for s in options.slots}
    assert "the shop" in labels or any(label.endswith("'s") for label in labels)


# ------------------------------------------------------------------ run breaks


def test_a_long_gap_sends_the_crew_home_between_runs(world, travel):
    """Nobody waits out a four-hour gap at a customer's kerb. A morning job and an
    evening job on the same crew become two runs: the evening stop departs the
    shop, and both extra legs - home, and out again - are counted as the real
    driving they are."""
    from krama.domain.enums import WindowHardness
    from krama.domain.models import TimeWindow
    from krama.scheduler.routing import materialize_route

    morning = world.jobs["j-401"]
    evening_site = world.jobs["j-402"]
    evening = evening_site.model_copy(
        update={
            "windows": (
                TimeWindow(
                    start=_at(0, 17),
                    end=_at(0, 21),
                    hardness=WindowHardness.SOFT,
                ),
            )
        }
    )
    world.jobs[evening.id] = evening

    route = materialize_route(
        world=world,
        travel=travel,
        crew_id="crew-van-1",
        worker_ids=("w-dan",),
        van_id="van-1",
        on_date=morning.windows[0].start.date(),
        job_sequence=[morning.id, evening.id],
        shift_start=_at(0, 8),
    )
    first, second = route.stops
    assert not first.fresh_run
    assert second.fresh_run, "a 90+ minute kerb wait means the crew went home"
    assert second.run_break_minutes > 0, "the trip home is real driving"
    assert second.arrival >= evening.windows[0].start

    # The totals carry every leg: out, between stops is replaced by home+out, home.
    legs = sum(s.travel_minutes_from_prev + s.run_break_minutes for s in route.stops)
    assert route.total_travel_minutes == legs + route.return_to_depot_minutes

    # And the plan validator accepts the fresh departure rather than flagging the
    # depot leg as an impossible drive from the previous customer.
    from krama.config import BusinessParams
    from krama.domain.invariants import ValidationConfig, validate_plan
    from krama.domain.models import PlanVersion

    business = BusinessParams.load()
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(business.meta.timezone)
    plan = PlanVersion(
        id="p-test",
        parent_id=None,
        created_at=_at(0, 7),
        horizon_start=route.date,
        horizon_end=route.date,
        routes=(route,),
        unserved=(),
    )
    config = ValidationConfig(business_tz=tz)
    codes = [v.code.value for v in validate_plan(plan, world, travel, config)]
    assert "travel_time_inconsistent" not in codes
