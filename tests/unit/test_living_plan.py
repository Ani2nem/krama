"""Diff classification, autonomy, repair and persistence.

The load-bearing assertion here is about *blast radius*. A promise is a window, not a
minute, so a crew arriving at 12:40 instead of 09:33 inside a 09:00-15:00 window has
not let anyone down. Getting that wrong in either direction breaks the product: nag
about nothing and the dispatcher stops reading the queue; stay quiet too often and
the system moves an appointment somebody booked a day off work for.
"""

from __future__ import annotations

from datetime import UTC, date, timedelta

import pytest

from krama.config import BusinessParams
from krama.domain.autonomy import AutonomyPolicy, Decision, decide
from krama.domain.diff import BlastRadius, ChangeKind, diff_plans
from krama.domain.enums import CommitmentState
from krama.domain.invariants import (
    ValidationConfig,
    summarize,
    validate_against_baseline,
    validate_plan,
)
from krama.domain.models import PlanVersion, TimeWindow
from krama.domain.state import Unavailability, fold
from krama.fixtures.sample_business import WEEK_START, _at, seed_events
from krama.persistence.log import PlanConflict, Workspace
from krama.scheduler.day_planner import SolveParams
from krama.scheduler.horizon import HorizonParams, plan_horizon
from krama.scheduler.repair import (
    RepairOptions,
    carve_out_locked,
    locked_jobs,
    repair_plan,
)
from tests.conftest import PlanBuilder

TUESDAY = date.fromordinal(WEEK_START.toordinal() + 1)


@pytest.fixture
def business() -> BusinessParams:
    return BusinessParams.load()


@pytest.fixture
def policy(business: BusinessParams) -> AutonomyPolicy:
    return AutonomyPolicy.from_business(business)


def confirm(world, job_id: str, start_hour: int, end_hour: int, cost: float = 250.0) -> None:
    """Promise a customer a window, the way a JobConfirmed event would."""
    job = world.jobs[job_id]
    world.jobs[job_id] = job.model_copy(
        update={
            "commitment_state": CommitmentState.CONFIRMED,
            "commitment_cost": cost,
            "windows": (TimeWindow(start=_at(0, start_hour), end=_at(0, end_hour)),),
        }
    )


# --------------------------------------------------------------------------- diff


def test_identical_plans_have_no_changes(world, travel):
    builder = PlanBuilder(world, travel)
    builder.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    plan = builder.build("p1")
    assert not diff_plans(plan, plan, world)


def test_a_new_job_is_an_addition(world, travel):
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-3", ["j-402", "j-405"], day=0, start_hour=8)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert [c.job_id for c in diff.by_kind(ChangeKind.ADDED)] == ["j-405"]


def test_a_different_day_is_a_reschedule(world, travel):
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-3", ["j-402"], day=1, start_hour=8)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert diff.changes[0].kind is ChangeKind.RESCHEDULED


def test_a_different_van_is_a_reassignment(world, travel):
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert diff.changes[0].kind is ChangeKind.REASSIGNED
    assert "van" in diff.changes[0].describe()


def test_reassignment_names_what_actually_moved(world, travel):
    """A van swap with the same worker once rendered as "crew Dan -> Dan", which reads
    as a bug in the plan rather than a fact about it."""
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    text = diff_plans(before.build("p1"), after.build("p2"), world).changes[0].describe()
    assert "Dan -> Dan" not in text


# ------------------------------------------------------------------ blast radius


def test_moving_provisional_work_is_invisible_to_customers(world, travel):
    """Nobody was told, so nobody needs telling. That is what makes provisional work
    the slack the optimiser is allowed to spend."""
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-3", ["j-402"], day=1, start_hour=8)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert diff.blast_radius is BlastRadius.INTERNAL
    assert diff.customer_visible_changes == ()


def test_retiming_inside_a_promised_window_needs_no_call(world, travel):
    """The central case. A customer told "between nine and three" has not been let
    down when the crew arrives at 12:40 rather than 09:33."""
    confirm(world, "j-402", 9, 15)
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=12)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert diff.changes and diff.changes[0].kind is ChangeKind.RETIMED
    assert not diff.changes[0].customer_visible
    assert diff.blast_radius is not BlastRadius.CUSTOMER_VISIBLE


def test_retiming_outside_a_promised_window_needs_a_call(world, travel):
    confirm(world, "j-402", 9, 11)
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=13)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert diff.changes[0].customer_visible
    assert diff.blast_radius is BlastRadius.CUSTOMER_VISIBLE


def test_dropping_a_promised_job_always_needs_a_call(world, travel):
    confirm(world, "j-402", 9, 15)
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    diff = diff_plans(before.build("p1"), PlanBuilder(world, travel).build("p2"), world)
    assert diff.changes[0].kind is ChangeKind.DROPPED
    assert diff.changes[0].customer_visible


def test_moving_a_promise_to_another_day_needs_a_call(world, travel):
    confirm(world, "j-402", 9, 15)
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-3", ["j-402"], day=1, start_hour=9)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert diff.changes[0].customer_visible


# ------------------------------------------------------------------- autonomy


def test_internal_changes_apply_themselves(world, travel, policy):
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert decide(diff, policy).decision is Decision.AUTO_APPLY


def test_a_customer_visible_change_escalates_however_cheap(world, travel, policy):
    """Structural rules come first. The cost of moving a promised appointment is not
    measured in dollars, so no threshold can buy it off."""
    confirm(world, "j-402", 9, 11)
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=13)
    diff = diff_plans(before.build("p1"), after.build("p2"), world, cost_delta=-500.0)
    decision = decide(diff, policy)
    assert decision.decision is Decision.ESCALATE
    assert "customer" in decision.explain()


def test_expensive_internal_churn_escalates(world, travel, policy):
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    diff = diff_plans(
        before.build("p1"),
        after.build("p2"),
        world,
        cost_delta=policy.max_cost_delta + 1,
    )
    assert decide(diff, policy).decision is Decision.ESCALATE


def test_new_overtime_escalates(world, travel, policy):
    """Nobody notices except the worker whose evening it is."""
    before = PlanBuilder(world, travel)
    before.route("A", ["w-dan"], "van-3", ["j-402"], day=0, start_hour=8)
    after = PlanBuilder(world, travel)
    after.route("A", ["w-dan"], "van-2", ["j-402"], day=0, start_hour=8)
    diff = diff_plans(before.build("p1"), after.build("p2"), world)
    assert decide(diff, policy, added_overtime_minutes=45).decision is Decision.ESCALATE


def test_doing_nothing_is_always_allowed(world, travel, policy):
    builder = PlanBuilder(world, travel)
    plan = builder.build("p1")
    assert decide(diff_plans(plan, plan, world), policy).auto


# --------------------------------------------------------------------- repair


@pytest.fixture
def committed(world, travel, business) -> PlanVersion:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(business.meta.timezone)
    params = SolveParams.from_business(business, tz)
    result = plan_horizon(
        world=world,
        travel=travel,
        start=WEEK_START,
        params=params,
        horizon_params=HorizonParams.from_business(business),
    )
    return PlanVersion(
        id="committed",
        created_at=_at(0, 6),
        horizon_start=WEEK_START,
        horizon_end=date.fromordinal(WEEK_START.toordinal() + 4),
        routes=result.routes,
        unserved=result.unserved,
    )


def dispatch(world, job_id: str) -> None:
    world.jobs[job_id] = world.jobs[job_id].model_copy(
        update={"commitment_state": CommitmentState.DISPATCHED}
    )


def test_in_flight_work_is_carved_out_not_replanned(world, travel, committed):
    """A crew on site is a fact, not a decision variable. It was also a real trap:
    pinning a materialized arrival into a model that prices travel pessimistically
    made the day unsatisfiable, because the solver correctly said it could not get
    there that early."""
    dispatch(world, "j-401")
    reduced, carried = carve_out_locked(world, committed, locked_jobs(world))
    assert "j-401" not in reduced.jobs
    assert any("j-401" in route.job_ids for route in carried)
    # The crew and van that are out are unavailable for the window they are busy.
    assert reduced.worker_outages or reduced.van_outages


def repair(world, travel, business, committed) -> RepairOptions:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(business.meta.timezone)
    return repair_plan(
        world=world,
        travel=travel,
        baseline=committed,
        start=WEEK_START,
        params=SolveParams.from_business(business, tz),
        horizon_params=HorizonParams.from_business(business),
    )


def test_repair_survives_a_breakdown_with_work_in_flight(world, travel, business, committed):
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(business.meta.timezone)
    dispatch(world, "j-401")
    confirm(world, "j-402", 9, 15)
    world.van_outages["van-1"] = [
        Unavailability(from_time=_at(0, 10, 40), until_time=None, reason="wont start")
    ]

    options = repair(world, travel, business, committed)
    assert options.candidates
    for candidate in options.candidates:
        violations = validate_plan(candidate.plan, world, travel, ValidationConfig(business_tz=tz))
        assert violations == (), f"{candidate.strategy.name}: {summarize(violations)}"
        moved = validate_against_baseline(candidate.plan, committed, world)
        assert moved == (), f"{candidate.strategy.name} moved in-flight work"


def test_repair_keeps_a_promised_window(world, travel, business, committed):
    confirm(world, "j-402", 9, 15)
    world.van_outages["van-1"] = [
        Unavailability(from_time=_at(0, 10, 40), until_time=None, reason="wont start")
    ]
    options = repair(world, travel, business, committed)
    recommended = options.best_by_fewest_calls
    assert recommended is not None

    found = recommended.plan.stop_for("j-402")
    if found is not None:
        _, stop = found
        window = world.jobs["j-402"].windows[0]
        assert window.contains(stop.arrival), "a promised window was silently moved"


def test_repair_offers_genuinely_different_trade_offs(world, travel, business, committed):
    """One answer would hide the judgement.

    Two vans down used to be enough of a squeeze; under marginal-cash economics the
    strategies all found the same cheap answer, because moving work costs fuel rather
    than wages. Three vans down used to leave one van and a genuine squeeze - then
    crews learned to re-form at the shop and run out twice, and a single van could
    quietly absorb the whole day again. So the last van also loses its morning:
    one van, one afternoon, and what to keep is a judgement once more.
    """
    for van_id in ("van-1", "van-2", "van-3"):
        world.van_outages[van_id] = [
            Unavailability(from_time=_at(0, 0), until_time=None, reason="out")
        ]
    world.van_outages["van-4"] = [
        Unavailability(from_time=_at(0, 0), until_time=_at(0, 13), reason="in the shop till one")
    ]
    options = repair(world, travel, business, committed)
    signatures = {(c.jobs_served, c.changes, c.customer_calls) for c in options.candidates}
    assert len(signatures) > 1 or len(options.candidates) == 1


# ---------------------------------------------------------------- persistence


def test_the_log_round_trips_through_disk(tmp_path):
    """Derived values must not be stored. geohash and content_hash were computed
    fields, so they serialized out and then failed to load back under extra="forbid"."""
    workspace = Workspace(tmp_path / "ws")
    workspace.seed(seed_events())
    reloaded = fold(Workspace(tmp_path / "ws").events.read())
    assert len(reloaded.workers) == 6
    assert len(reloaded.jobs) == 10


def test_state_survives_a_restart(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    workspace.seed(seed_events())
    plan = PlanVersion(
        id="v1", created_at=_at(0, 6), horizon_start=WEEK_START, horizon_end=WEEK_START
    )
    workspace.plans.commit(plan, expected_parent=None)

    fresh = Workspace(tmp_path / "ws")
    head = fresh.plans.head()
    assert head is not None and head.id == "v1"


def test_a_stale_commit_is_rejected(tmp_path):
    """Two dispatchers on two calls both solve against the same head. Without this the
    second write silently discards the first customer's booking."""
    workspace = Workspace(tmp_path / "ws")
    workspace.seed(seed_events())
    first = PlanVersion(
        id="v1", created_at=_at(0, 6), horizon_start=WEEK_START, horizon_end=WEEK_START
    )
    workspace.plans.commit(first, expected_parent=None)
    second = PlanVersion(
        id="v2",
        parent_id="v1",
        created_at=_at(0, 7),
        horizon_start=WEEK_START,
        horizon_end=WEEK_START,
    )
    workspace.plans.commit(second, expected_parent="v1")

    stale = PlanVersion(
        id="v3",
        parent_id="v1",
        created_at=_at(0, 8),
        horizon_start=WEEK_START,
        horizon_end=WEEK_START,
    )
    with pytest.raises(PlanConflict):
        workspace.plans.commit(stale, expected_parent="v1")


def test_ancestry_walks_back_through_parents(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    workspace.seed(seed_events())
    previous: str | None = None
    for index in range(3):
        plan = PlanVersion(
            id=f"v{index}",
            parent_id=previous,
            created_at=_at(0, 6) + timedelta(hours=index),
            horizon_start=WEEK_START,
            horizon_end=WEEK_START,
        )
        workspace.plans.commit(plan, expected_parent=previous)
        previous = plan.id
    assert [p.id for p in workspace.plans.ancestry()] == ["v2", "v1", "v0"]


def test_seeding_twice_is_refused(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    workspace.seed(seed_events())
    with pytest.raises(FileExistsError):
        workspace.seed(seed_events())


def test_a_retime_that_overruns_the_promised_end_is_customer_visible():
    """Arriving inside the window is not enough - the work must FINISH inside it.

    Found when the quality tier went silent: a repair moved a confirmed
    nine-to-half-eleven job to arrive at 10:49 and leave at 12:49, and the diff
    called it invisible because only the arrival was tested. The customer whose
    morning ended at half eleven would not have agreed.
    """
    from datetime import datetime

    from krama.domain.diff import ChangeKind, JobChange, Placement
    from krama.domain.enums import CommitmentState
    from krama.domain.models import TimeWindow

    tz = UTC
    window = TimeWindow(
        start=datetime(2026, 9, 21, 9, 0, tzinfo=tz), end=datetime(2026, 9, 21, 11, 30, tzinfo=tz)
    )

    def placed(arrive_h: int, arrive_m: int, leave_h: int, leave_m: int) -> Placement:
        return Placement(
            on_date=window.start.date(),
            crew_id="c",
            van_id="v",
            worker_ids=("w",),
            arrival=datetime(2026, 9, 21, arrive_h, arrive_m, tzinfo=tz),
            departure=datetime(2026, 9, 21, leave_h, leave_m, tzinfo=tz),
        )

    def change(after: Placement) -> JobChange:
        return JobChange(
            job_id="j",
            kind=ChangeKind.RETIMED,
            before=placed(9, 0, 11, 0),
            after=after,
            commitment_state=CommitmentState.CONFIRMED,
            promised_window=window,
        )

    assert change(placed(10, 49, 12, 49)).customer_visible, "overruns the promise"
    assert not change(placed(9, 20, 11, 20)).customer_visible, "fits entirely - no call"
