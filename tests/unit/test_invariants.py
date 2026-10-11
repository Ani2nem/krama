"""Every invariant, proven by breaking a known-good plan one way at a time.

The structure is deliberate: each test starts from a plan the checker accepts, makes
exactly one thing wrong, and asserts that specific code appears. A checker that
cannot distinguish "van is double-booked" from "van lacks stock" is not much of a
guard, and tests that only ever assert "some violation" would not notice.
"""

from __future__ import annotations

from datetime import timedelta

from krama.domain.enums import CommitmentState, ViolationCode
from krama.domain.events import VanUnavailable
from krama.domain.invariants import (
    validate_against_baseline,
    validate_plan,
)
from krama.domain.models import Stop, TimeWindow
from krama.domain.state import Unavailability
from krama.fixtures.sample_business import WEEK_START
from tests.conftest import PlanBuilder, moment

MONDAY = WEEK_START


def codes(violations: object) -> set[ViolationCode]:
    return {v.code for v in violations}  # type: ignore[attr-defined]


# ------------------------------------------------------------------ the happy path


def test_well_formed_plan_has_no_violations(plan_builder: PlanBuilder, world, travel, config):
    """Three crews, each matched to work it is certified and stocked for."""
    plan_builder.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    plan_builder.route("B", ["w-marcus", "w-priya"], "van-1", ["j-401"], day=0, start_hour=6)
    plan_builder.route("C", ["w-ken"], "van-2", ["j-405"], day=0, start_hour=8)
    violations = validate_plan(plan_builder.build(), world, travel, config)
    assert violations == (), "\n".join(str(v) for v in violations)


def test_empty_plan_is_feasible(plan_builder: PlanBuilder, world, travel, config):
    assert validate_plan(plan_builder.build(), world, travel, config) == ()


# ------------------------------------------------------------------------ references


def test_unknown_worker_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    plan_builder.route("A", ["w-ghost"], "van-2", ["j-402"])
    assert ViolationCode.UNKNOWN_ENTITY_REFERENCE in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_unknown_van_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"])
    plan = plan_builder.build()
    broken = plan.model_copy(
        update={"routes": (plan.routes[0].model_copy(update={"van_id": "van-99"}),)}
    )
    assert ViolationCode.UNKNOWN_ENTITY_REFERENCE in codes(
        validate_plan(broken, world, travel, config)
    )


def test_same_job_on_two_crews_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    plan_builder.route("B", ["w-ken"], "van-3", ["j-402"], day=1, start_hour=8)
    assert ViolationCode.DUPLICATE_JOB_ASSIGNMENT in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


# -------------------------------------------------------------------- crew fitness


def test_uncertified_crew_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    """Sofia holds auto-glass only; the Chen job needs residential glazing."""
    plan_builder.route("A", ["w-sofia"], "van-2", ["j-402"])
    assert ViolationCode.MISSING_CERTIFICATION in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_crew_certifications_pool_across_members(plan_builder: PlanBuilder, world, travel, config):
    """A certified lead plus an uncertified hand satisfies the requirement."""
    plan_builder.route("B", ["w-marcus", "w-alex"], "van-1", ["j-401"], day=0, start_hour=6)
    violations = validate_plan(plan_builder.build(), world, travel, config)
    assert ViolationCode.MISSING_CERTIFICATION not in codes(violations)


def test_two_person_job_with_one_worker_is_rejected(
    plan_builder: PlanBuilder, world, travel, config
):
    plan_builder.route("B", ["w-marcus"], "van-1", ["j-401"], day=0, start_hour=6)
    assert ViolationCode.CREW_SIZE_MISMATCH in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


# ------------------------------------------------------------------ exclusivity


def test_worker_in_two_places_at_once_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    plan_builder.route("B", ["w-dan"], "van-3", ["j-405"], day=0, start_hour=8)
    assert ViolationCode.WORKER_DOUBLE_BOOKED in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_van_in_two_places_at_once_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    plan_builder.route("B", ["w-ken"], "van-2", ["j-405"], day=0, start_hour=8)
    assert ViolationCode.VAN_DOUBLE_BOOKED in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_same_worker_on_different_days_is_fine(plan_builder: PlanBuilder, world, travel, config):
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    plan_builder.route("B", ["w-dan"], "van-2", ["j-408"], day=1, start_hour=8)
    assert validate_plan(plan_builder.build(), world, travel, config) == ()


# ------------------------------------------------------------------- availability


def test_worker_scheduled_through_an_outage_is_rejected(
    plan_builder: PlanBuilder, world, travel, config
):
    world.worker_outages["w-dan"] = [
        Unavailability(from_time=moment(0, 7), until_time=moment(0, 18), reason="sick")
    ]
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"])
    assert ViolationCode.WORKER_UNAVAILABLE in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_van_scheduled_through_an_outage_is_rejected(
    plan_builder: PlanBuilder, world, travel, config
):
    world.van_outages["van-2"] = [
        Unavailability(from_time=moment(0, 7), until_time=None, reason="wont start")
    ]
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"])
    assert ViolationCode.VAN_UNAVAILABLE in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_outage_that_ends_before_the_route_is_fine(
    plan_builder: PlanBuilder, world, travel, config
):
    world.van_outages["van-2"] = [
        Unavailability(from_time=moment(0, 2), until_time=moment(0, 6), reason="battery")
    ]
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    assert validate_plan(plan_builder.build(), world, travel, config) == ()


# ---------------------------------------------------------------------- physics


def test_teleporting_between_stops_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    """Claim an arrival that no real drive could achieve."""
    route = plan_builder.route("A", ["w-dan"], "van-2", ["j-402", "j-405"], day=0, start_hour=8)
    second = route.stops[1]
    cheated = second.model_copy(
        update={
            "arrival": route.stops[0].departure,
            "departure": route.stops[0].departure + timedelta(minutes=45),
            "travel_minutes_from_prev": 0,
        }
    )
    plan = plan_builder.build()
    broken = plan.model_copy(
        update={"routes": (route.model_copy(update={"stops": (route.stops[0], cheated)}),)}
    )
    assert ViolationCode.TRAVEL_TIME_INCONSISTENT in codes(
        validate_plan(broken, world, travel, config)
    )


def test_too_little_time_on_site_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    """The Chen job needs 120 minutes; allot 30."""
    route = plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    stop = route.stops[0]
    rushed = stop.model_copy(update={"departure": stop.arrival + timedelta(minutes=30)})
    plan = plan_builder.build()
    broken = plan.model_copy(update={"routes": (route.model_copy(update={"stops": (rushed,)}),)})
    assert ViolationCode.TRAVEL_TIME_INCONSISTENT in codes(
        validate_plan(broken, world, travel, config)
    )


# ---------------------------------------------------------------------- windows


def test_hard_window_must_contain_the_whole_service(
    plan_builder: PlanBuilder, world, travel, config
):
    """j-401 must *finish* before the store opens at 09:00, not merely start before it."""
    route = plan_builder.route(
        "B", ["w-marcus", "w-priya"], "van-1", ["j-401"], day=0, start_hour=6
    )
    stop = route.stops[0]
    late = Stop(
        job_id=stop.job_id,
        arrival=moment(0, 8, 30),
        departure=moment(0, 11, 0),
        travel_minutes_from_prev=stop.travel_minutes_from_prev,
        travel_miles_from_prev=stop.travel_miles_from_prev,
    )
    plan = plan_builder.build()
    broken = plan.model_copy(update={"routes": (route.model_copy(update={"stops": (late,)}),)})
    assert ViolationCode.HARD_WINDOW_VIOLATED in codes(validate_plan(broken, world, travel, config))


def test_moving_a_confirmed_window_is_reported_distinctly(
    plan_builder: PlanBuilder, world, travel, config
):
    """A promised slot that moves is a different failure from an infeasible one."""
    job = world.jobs["j-402"]
    world.jobs["j-402"] = job.model_copy(
        update={
            "commitment_state": CommitmentState.CONFIRMED,
            "commitment_cost": 180.0,
            "windows": (TimeWindow(start=moment(0, 9), end=moment(0, 11)),),
        }
    )
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=13)
    assert ViolationCode.CONFIRMED_WINDOW_MOVED in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


# ---------------------------------------------------------------- working hours


def test_starting_before_the_shift_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    """Dan's shift begins at 08:00; a 06:00 departure means he is not there yet."""
    plan_builder.route("A", ["w-dan"], "van-2", ["j-405"], day=0, start_hour=6)
    assert ViolationCode.OUTSIDE_WORKING_HOURS in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_overtime_for_an_ineligible_worker_is_rejected(
    plan_builder: PlanBuilder, world, travel, config
):
    """A worker with overtime_eligible=False cannot be routed past their shift.

    The sample crew is all-eligible now, so the ineligible worker is constructed here:
    the invariant belongs to the model, not to any particular roster.
    """
    ken = world.workers["w-ken"]
    world.workers["w-ken"] = ken.model_copy(update={"overtime_eligible": False})
    plan_builder.route("A", ["w-ken"], "van-2", ["j-402"], day=0, start_hour=15, start_minute=30)
    violations = validate_plan(plan_builder.build(), world, travel, config)
    assert ViolationCode.OUTSIDE_WORKING_HOURS in codes(violations)


# -------------------------------------------------------------------- van stock


def test_van_without_the_part_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    """van-2 stocks no shower kits; the Patel install needs one."""
    plan_builder.route("A", ["w-alex"], "van-2", ["j-403"], day=0, start_hour=8)
    assert ViolationCode.VAN_CAPACITY_EXCEEDED in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_exceeding_rack_slots_is_rejected(plan_builder: PlanBuilder, world, travel, config):
    small_van = world.vans["van-2"].model_copy(update={"rack_slots": 1})
    world.vans["van-2"] = small_van
    plan_builder.route("A", ["w-dan"], "van-2", ["j-402", "j-405"], day=0, start_hour=8)
    assert ViolationCode.VAN_CAPACITY_EXCEEDED in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


# --------------------------------------------------------------------- materials


def test_scheduling_before_materials_arrive_is_rejected(
    plan_builder: PlanBuilder, world, travel, config
):
    """j-406 waits on a tempered unit with a three-day lead time."""
    plan_builder.route("A", ["w-dan"], "van-3", ["j-406"], day=0, start_hour=8)
    assert ViolationCode.MATERIALS_UNAVAILABLE in codes(
        validate_plan(plan_builder.build(), world, travel, config)
    )


def test_scheduling_after_materials_arrive_is_fine(
    plan_builder: PlanBuilder, world, travel, config
):
    plan_builder.route("A", ["w-dan"], "van-3", ["j-406"], day=3, start_hour=8)
    violations = validate_plan(plan_builder.build(), world, travel, config)
    assert ViolationCode.MATERIALS_UNAVAILABLE not in codes(violations)


# ------------------------------------------------------------- baseline comparison


def test_moving_a_dispatched_job_is_rejected(plan_builder: PlanBuilder, world, travel):
    """A crew already on site cannot be retroactively rescheduled."""
    baseline_builder = PlanBuilder(world, travel)
    baseline_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    baseline = baseline_builder.build("plan-0")

    world.jobs["j-402"] = world.jobs["j-402"].model_copy(
        update={"commitment_state": CommitmentState.DISPATCHED}
    )

    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=13)
    moved = plan_builder.build("plan-1", parent_id="plan-0")

    assert ViolationCode.LOCKED_JOB_MOVED in codes(
        validate_against_baseline(moved, baseline, world)
    )


def test_dropping_a_dispatched_job_is_rejected(plan_builder: PlanBuilder, world, travel):
    baseline_builder = PlanBuilder(world, travel)
    baseline_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    baseline = baseline_builder.build("plan-0")

    world.jobs["j-402"] = world.jobs["j-402"].model_copy(
        update={"commitment_state": CommitmentState.DISPATCHED}
    )
    empty = plan_builder.build("plan-1", parent_id="plan-0")

    assert ViolationCode.LOCKED_JOB_MOVED in codes(
        validate_against_baseline(empty, baseline, world)
    )


def test_moving_a_provisional_job_is_allowed(plan_builder: PlanBuilder, world, travel):
    """The cold path is free to reshuffle anything nobody has been promised."""
    baseline_builder = PlanBuilder(world, travel)
    baseline_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    baseline = baseline_builder.build("plan-0")

    plan_builder.route("A", ["w-dan"], "van-2", ["j-402"], day=1, start_hour=9)
    moved = plan_builder.build("plan-1", parent_id="plan-0")

    assert validate_against_baseline(moved, baseline, world) == ()


# ------------------------------------------------------------------ event plumbing


def test_events_drive_the_checks_end_to_end(plan_builder: PlanBuilder, travel, config):
    """A van-breakdown event alone must make a previously valid plan infeasible."""
    from krama.domain.state import fold
    from krama.fixtures.sample_business import sample_world_at

    events, now = sample_world_at()
    clean = fold(events, as_of=now)
    builder = PlanBuilder(clean, travel)
    builder.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    plan = builder.build()
    assert validate_plan(plan, clean, travel, config) == ()

    events.append(
        VanUnavailable(
            event_id="e-break",
            occurred_at=moment(0, 10, 40),
            recorded_at=moment(0, 10, 52),
            dispatch_id="d-1",
            van_id="van-2",
            from_time=moment(0, 10, 40),
            reason="wont start",
        )
    )
    after = fold(events, as_of=moment(0, 11))
    assert ViolationCode.VAN_UNAVAILABLE in codes(validate_plan(plan, after, travel, config))


# --------------------------------------------------- window kinds, and promises


def test_a_soft_window_may_be_served_late(plan_builder: PlanBuilder, world, travel, config):
    """Lateness against a soft window is priced by the objective, not refused.

    Treating it as infeasible quietly turns every soft window into a hard one: a
    customer who said "Tuesday morning would suit" would become unschedulable the
    moment the morning filled up, rather than simply more expensive to serve later.
    Found by a property test over generated worlds, not by anyone reasoning about it.
    """
    job = world.jobs["j-402"]
    world.jobs["j-402"] = job.model_copy(
        update={"windows": (TimeWindow(start=moment(0, 9), end=moment(0, 11)),)}
    )
    plan_builder.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=13)
    violations = validate_plan(plan_builder.build(), world, travel, config)
    assert ViolationCode.HARD_WINDOW_VIOLATED not in codes(violations)


def test_arriving_before_any_window_opens_is_still_refused(
    plan_builder: PlanBuilder, world, travel, config
):
    """Late is a cost; early is impossible - nobody is there to let the crew in.

    The stop is constructed rather than materialized because the materializer already
    does the right thing: it waits for the window. This checks the guard behind it,
    for a plan that arrived some other way.
    """
    job = world.jobs["j-405"]
    world.jobs["j-405"] = job.model_copy(
        update={"windows": (TimeWindow(start=moment(0, 14), end=moment(0, 16)),)}
    )
    route = plan_builder.route("C", ["w-ken"], "van-2", ["j-405"], day=0, start_hour=13)
    stop = route.stops[0]
    early = stop.model_copy(update={"arrival": moment(0, 9), "departure": moment(0, 9, 45)})
    plan = plan_builder.build()
    broken = plan.model_copy(update={"routes": (route.model_copy(update={"stops": (early,)}),)})
    violations = validate_plan(broken, world, travel, config)
    assert ViolationCode.HARD_WINDOW_VIOLATED in codes(violations)
    assert any("before any window opens" in v.detail for v in violations)


def test_a_confirmed_soft_window_binds_like_a_hard_one(
    plan_builder: PlanBuilder, world, travel, config
):
    """hardness says what the customer needs; CONFIRMED says what we promised. Once
    someone has been told "between nine and eleven", that window stops being a
    preference the optimiser may spend, however soft it began."""
    job = world.jobs["j-402"]
    world.jobs["j-402"] = job.model_copy(
        update={
            "commitment_state": CommitmentState.CONFIRMED,
            "commitment_cost": 180.0,
            "windows": (TimeWindow(start=moment(0, 9), end=moment(0, 11)),),
        }
    )
    plan_builder.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=13)
    violations = validate_plan(plan_builder.build(), world, travel, config)
    assert ViolationCode.CONFIRMED_WINDOW_MOVED in codes(violations)
    # The line names the customer and both times - a sentence the dispatcher can
    # read down the phone, not a pair of ISO timestamps.
    assert any("was promised" in v.detail and "have not been told" in v.detail for v in violations)
