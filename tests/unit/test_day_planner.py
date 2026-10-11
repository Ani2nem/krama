"""Solver tests.

The central property is the last one: *whatever* the solver returns must pass the
independent invariant checker. The solver optimizes over an approximation (one
travel matrix, integer minutes, a blended labour rate); the checker re-derives
everything from the materialized route. Those two agreeing is the whole safety
argument, so it is asserted under every configuration these tests exercise.
"""

from __future__ import annotations

from datetime import date

import pytest

from krama.domain.enums import CommitmentState, UnservedReason
from krama.domain.invariants import ValidationConfig, summarize, validate_plan
from krama.domain.models import PlanVersion
from krama.domain.state import Unavailability, WorldState
from krama.fixtures.sample_business import BUSINESS_TZ, WEEK_START, _at
from krama.scheduler.day_planner import DayPlanResult, SolveParams, plan_day
from krama.scheduler.travel.synthetic import SyntheticTravelProvider

MONDAY = WEEK_START
TUESDAY = date.fromordinal(WEEK_START.toordinal() + 1)
THURSDAY = date.fromordinal(WEEK_START.toordinal() + 3)


@pytest.fixture
def params() -> SolveParams:
    return SolveParams(business_tz=BUSINESS_TZ, max_solve_seconds=20.0)


def solve(
    world: WorldState,
    travel: SyntheticTravelProvider,
    params: SolveParams,
    on_date: date = MONDAY,
    job_ids: list[str] | None = None,
) -> DayPlanResult:
    candidates = job_ids if job_ids is not None else [j.id for j in world.active_jobs()]
    return plan_day(
        world=world,
        travel=travel,
        on_date=on_date,
        candidate_job_ids=candidates,
        params=params,
    )


def as_plan(result: DayPlanResult, on_date: date = MONDAY) -> PlanVersion:
    return PlanVersion(
        id="p",
        created_at=_at(0, 5),
        horizon_start=on_date,
        horizon_end=on_date,
        routes=result.routes,
    )


def assert_feasible(result: DayPlanResult, world: WorldState, travel, on_date=MONDAY) -> None:
    violations = validate_plan(
        as_plan(result, on_date), world, travel, ValidationConfig(business_tz=BUSINESS_TZ)
    )
    assert violations == (), summarize(violations)


def reasons(result: DayPlanResult) -> dict[str, UnservedReason]:
    return {u.job_id: u.reason for u in result.unserved}


def crew_for(result: DayPlanResult, job_id: str) -> tuple[str, ...] | None:
    for route in result.routes:
        if job_id in route.job_ids:
            return route.worker_ids
    return None


# ------------------------------------------------------------------------ basics


def test_solves_monday_to_optimality(world, travel, params):
    result = solve(world, travel, params)
    assert result.status == "OPTIMAL"
    assert result.routes


def test_every_monday_job_is_scheduled(world, travel, params):
    """The five jobs whose windows fall on Monday all fit."""
    result = solve(world, travel, params)
    scheduled = {job_id for route in result.routes for job_id in route.job_ids}
    assert scheduled == {"j-401", "j-402", "j-403", "j-404", "j-405"}


def test_jobs_for_other_days_are_reported_honestly(world, travel, params):
    """Not "the day was full" - the window is simply on another date."""
    result = solve(world, travel, params)
    assert reasons(result)["j-408"] is UnservedReason.WINDOW_ON_ANOTHER_DAY


def test_back_ordered_materials_block_scheduling(world, travel, params):
    """No routing decision can conjure a tempered unit that arrives Thursday."""
    result = solve(world, travel, params)
    assert reasons(result)["j-406"] is UnservedReason.MATERIALS_NOT_AVAILABLE

    later = solve(world, travel, params, on_date=THURSDAY, job_ids=["j-406"])
    assert "j-406" in {jid for route in later.routes for jid in route.job_ids}


# ------------------------------------------------------------------- crew shaping


def test_two_person_job_gets_two_people(world, travel, params):
    result = solve(world, travel, params)
    crew = crew_for(result, "j-401")
    assert crew is not None and len(crew) == 2


def test_no_useless_riders(world, travel, params):
    """Everyone aboard must be there for a reason.

    This used to assert crew size == the largest crew the route's jobs demand, which
    was the wage-per-driving-minute economics talking: under marginal cash, two
    specialists SHOULD share a van when their solo jobs sit on one route - the ride
    saves a whole van trip and the wage is owed either way. What stays wrong is a
    rider nobody needs: every member must either be demanded by headcount or hold a
    certification some stop needs that the rest of the crew cannot cover.
    """
    result = solve(world, travel, params)
    for route in result.routes:
        for worker_id in route.worker_ids:
            worker = world.workers[worker_id]
            others = [world.workers[w] for w in route.worker_ids if w != worker_id]
            needed = any(
                world.jobs[jid].crew_size >= len(route.worker_ids)
                or any(
                    cert in worker.certifications
                    and not any(cert in o.certifications for o in others)
                    for cert in world.jobs[jid].required_certifications
                )
                for jid in route.job_ids
            )
            assert needed, f"{worker.name} rides on {route.crew_id} doing nothing"


def test_only_certified_crews_are_assigned(world, travel, params):
    result = solve(world, travel, params)
    for route in result.routes:
        held = frozenset().union(*(world.workers[w].certifications for w in route.worker_ids))
        for job_id in route.job_ids:
            assert world.jobs[job_id].required_certifications <= held


def test_storefront_needs_the_scarce_commercial_certification(world, travel, params):
    """Only Marcus and Priya hold it, so they are the only possible crew."""
    result = solve(world, travel, params)
    assert set(crew_for(result, "j-401") or ()) == {"w-marcus", "w-priya"}


# ------------------------------------------------------------------- disruptions


def test_van_outage_removes_that_van_from_the_plan(world, travel, params):
    world.van_outages["van-4"] = [
        Unavailability(from_time=_at(0, 0), until_time=_at(1, 0), reason="wont start")
    ]
    result = solve(world, travel, params)
    assert all(route.van_id != "van-4" for route in result.routes)
    assert_feasible(result, world, travel)


def test_worker_outage_removes_that_worker(world, travel, params):
    world.worker_outages["w-sofia"] = [
        Unavailability(from_time=_at(0, 0), until_time=_at(1, 0), reason="sick")
    ]
    result = solve(world, travel, params)
    assert all("w-sofia" not in route.worker_ids for route in result.routes)


def test_losing_the_only_certified_workers_is_explained_precisely(world, travel, params):
    """Both commercial-certified workers out means the storefront job is unstaffable,
    and the dispatcher should be told that rather than "no capacity"."""
    for worker_id in ("w-marcus", "w-priya"):
        world.worker_outages[worker_id] = [
            Unavailability(from_time=_at(0, 0), until_time=_at(1, 0), reason="sick")
        ]
    result = solve(world, travel, params)
    assert reasons(result)["j-401"] is UnservedReason.NO_CERTIFIED_WORKER


def test_losing_every_van_yields_no_routes(world, travel, params):
    for van_id in list(world.vans):
        world.van_outages[van_id] = [
            Unavailability(from_time=_at(0, 0), until_time=_at(1, 0), reason="depot fire")
        ]
    result = solve(world, travel, params)
    assert result.routes == ()


# ------------------------------------------------------------------- objective


def test_repeated_deferral_eventually_forces_a_job_in(world, travel):
    """Pure cost minimisation drops the same far-out, low-revenue customer forever.
    The escalating unserved penalty is what stops that."""
    far = world.jobs["j-409"]
    world.jobs["j-409"] = far.model_copy(
        update={"windows": world.jobs["j-402"].windows, "deferral_count": 0}
    )
    # Serving a job now costs only fuel and overtime, so the bar that makes
    # dropping it "cheaper" must sit below a dedicated trip's fuel - not below a
    # morning of wages, as it did when driving minutes were charged at the rate.
    tight = SolveParams(
        business_tz=BUSINESS_TZ,
        max_solve_seconds=20.0,
        unserved_penalty_base=2.0,
        deferral_escalation=0.0,
        revenue_weight=0.0,
    )
    ignored = solve(world, travel, tight, job_ids=["j-402", "j-409"])
    assert "j-409" not in {j for r in ignored.routes for j in r.job_ids}

    world.jobs["j-409"] = world.jobs["j-409"].model_copy(update={"deferral_count": 6})
    escalating = SolveParams(
        business_tz=BUSINESS_TZ,
        max_solve_seconds=20.0,
        unserved_penalty_base=40.0,
        deferral_escalation=400.0,
        revenue_weight=0.0,
    )
    forced = solve(world, travel, escalating, job_ids=["j-402", "j-409"])
    assert "j-409" in {j for r in forced.routes for j in r.job_ids}


def test_solving_is_reproducible(world, travel, params):
    """Scenario replay and eval baselines depend on identical input giving identical
    output, so the solver runs single-threaded with a fixed seed."""
    first, second = solve(world, travel, params), solve(world, travel, params)
    assert as_plan(first).content_hash == as_plan(second).content_hash


# --------------------------------------------------------- the load-bearing property


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("baseline", lambda w: None),
        (
            "van-3 down",
            lambda w: w.van_outages.setdefault("van-3", []).append(
                Unavailability(from_time=_at(0, 0), until_time=_at(1, 0))
            ),
        ),
        (
            "dan out",
            lambda w: w.worker_outages.setdefault("w-dan", []).append(
                Unavailability(from_time=_at(0, 0), until_time=_at(1, 0))
            ),
        ),
        (
            "marcus out",
            lambda w: w.worker_outages.setdefault("w-marcus", []).append(
                Unavailability(from_time=_at(0, 0), until_time=_at(1, 0))
            ),
        ),
        (
            "chen cancelled",
            lambda w: w.jobs.__setitem__(
                "j-402",
                w.jobs["j-402"].model_copy(update={"commitment_state": CommitmentState.CANCELLED}),
            ),
        ),
        (
            "patel runs long",
            lambda w: w.jobs.__setitem__(
                "j-403", w.jobs["j-403"].model_copy(update={"estimated_duration_min": 330})
            ),
        ),
    ],
)
def test_solver_output_always_passes_the_independent_checker(world, travel, params, label, mutate):
    mutate(world)
    result = solve(world, travel, params)
    violations = validate_plan(
        as_plan(result), world, travel, ValidationConfig(business_tz=BUSINESS_TZ)
    )
    assert violations == (), f"{label}: {summarize(violations)}"


def test_overtime_ban_is_respected(world, travel):
    """With overtime disabled no route may run past any member's shift end."""
    strict = SolveParams(business_tz=BUSINESS_TZ, max_solve_seconds=20.0, allow_overtime=False)
    result = solve(world, travel, strict)
    config = ValidationConfig(business_tz=BUSINESS_TZ, allow_overtime=False)
    violations = validate_plan(as_plan(result), world, travel, config)
    assert violations == (), summarize(violations)


def test_empty_candidate_list_is_handled(world, travel, params):
    result = solve(world, travel, params, job_ids=[])
    assert result.routes == () and result.unserved == ()


def test_solve_is_fast_enough_for_an_interactive_call(world, travel, params):
    """The hot path quotes slots while a customer is on the phone."""
    result = solve(world, travel, params)
    assert result.metrics["solve_seconds"] < 5.0


# ------------------------------------------------------------ deterministic ties


def test_an_equal_cost_tie_is_broken_the_same_way_every_time(world, travel, params):
    """Two plans costing exactly the same must not depend on how the search ran.

    This is not symmetry between identical workers - the fixture has none. It is
    degeneracy: the objective uses a blended labour rate by design, so on a day where
    two differently-paid, differently-certified people are both qualified, it genuinely
    has no reason to prefer either, and both answers are optimal.

    Found when golden board snapshots passed on arm64 and failed on x86_64 with the
    same OR-Tools version, the same seed, one search worker and a deterministic budget.
    Both runs returned OPTIMAL at $405.61. Nothing was wrong except that the question
    had two right answers and the suite asserted one of them.
    """
    signatures = set()
    for _ in range(3):
        result = solve(world, travel, params)
        signatures.add(
            tuple(sorted((r.crew_id, tuple(sorted(r.worker_ids))) for r in result.routes))
        )
    assert len(signatures) == 1, f"the same problem gave {len(signatures)} different plans"


def test_the_tie_break_never_outweighs_a_cent_of_real_cost(world, travel, params):
    """The tie-break is scaled below the objective, so it can only order equal plans.

    Asserted by comparing against a solve of the same problem: the cost must be exactly
    what it was before a preference for lower-numbered vans was introduced. A tie-break
    large enough to buy a worse plan would be a silent, permanent overcharge.
    """
    result = solve(world, travel, params)
    assert result.status == "OPTIMAL"
    assert_feasible(result, world, travel)
    # The value this fixture produces. 244.09 -> 374.22 when the business moved from
    # Seattle to Texas; 374.22 -> 57.14 when the objective became marginal cash;
    # 57.14 -> 55.17 when the depot pin moved ~4 road miles south onto the actual
    # building. The point of pinning it is unchanged: a number that only moves when
    # the economics or the geography do is a number that would catch a tie-break
    # quietly buying a worse plan.
    assert round(result.objective_cost, 2) == 55.17


def test_crews_re_form_at_the_shop_for_the_afternoon(world, travel, params):
    """Dan runs solo all morning in one van; Priya works another. At three the
    board has a job that needs them BOTH - and now they can have it: both runs
    come home, Dan and Priya leave the shop together in one van. Nobody swaps
    vans at a kerb, nothing is left parked anywhere but home.
    """
    from krama.domain.models import TimeWindow

    template = world.jobs["j-401"]

    def job_at(job_id: str, like: str, open_h: int, close_h: int, crew: int) -> None:
        site = world.jobs[like]
        world.jobs[job_id] = template.model_copy(
            update={
                "id": job_id,
                "customer_name": job_id,
                "location": site.location,
                "required_certifications": frozenset(),
                "crew_size": crew,
                "estimated_duration_min": 60,
                "windows": (TimeWindow(start=_at(0, open_h), end=_at(0, close_h)),),
            }
        )

    job_at("j-dan-am", "j-401", 8, 11, crew=1)
    job_at("j-priya-am", "j-402", 8, 11, crew=1)
    job_at("j-pair-pm", "j-403", 15, 17, crew=2)

    result = plan_day(
        world=world,
        travel=travel,
        on_date=MONDAY,
        candidate_job_ids=["j-dan-am", "j-priya-am", "j-pair-pm"],
        params=params,
        pinned_workers={
            "j-dan-am": ("w-dan",),
            "j-priya-am": ("w-priya",),
            "j-pair-pm": ("w-dan", "w-priya"),
        },
    )

    assert not result.unserved, reasons(result)
    pair_route = next(r for r in result.routes if any(s.job_id == "j-pair-pm" for s in r.stops))
    assert set(pair_route.worker_ids) == {"w-dan", "w-priya"}, (
        "the afternoon crew is the two people the job needs, in one van"
    )
    assert len([s for r in result.routes for s in r.stops]) == 3

    # The whole arrangement survives the independent checker: no double-booked
    # van or fitter, every leg recomputed from the road - the afternoon run is a
    # real departure from the shop, not a teleport from somebody's kerb.
    assert_feasible(result, world, travel)


def test_an_hour_out_is_honoured_not_overtimed_through(world, travel, params):
    """Priya out 10 to 11 had her 8:00 install scheduled straight through the
    hole: the usable morning ended at 10:00, and the overtime allowance - meant
    for evenings after a shift CLOSE - reached past the outage edge. Overtime
    extends shifts, not absences: work now lands clear of the hole, and the
    independent checker agrees."""
    from krama.domain.models import TimeWindow

    world.worker_outages["w-dan"] = [
        Unavailability(from_time=_at(0, 10), until_time=_at(0, 11), reason="dentist")
    ]
    template = world.jobs["j-401"]
    world.jobs["j-long"] = template.model_copy(
        update={
            "id": "j-long",
            "customer_name": "Long morning",
            "required_certifications": frozenset(),
            "crew_size": 1,
            "estimated_duration_min": 150,
            "windows": (TimeWindow(start=_at(0, 8), end=_at(0, 17)),),
        }
    )

    result = plan_day(
        world=world,
        travel=travel,
        on_date=MONDAY,
        candidate_job_ids=["j-long"],
        params=params,
        pinned_workers={"j-long": ("w-dan",)},
    )
    assert not result.unserved, reasons(result)
    stop = next(s for r in result.routes for s in r.stops if s.job_id == "j-long")
    local = stop.arrival.astimezone(BUSINESS_TZ)
    assert local.hour >= 11 or stop.departure.astimezone(BUSINESS_TZ).hour < 10, (
        f"150 minutes cannot fit before a 10:00 hole - scheduled at {local:%H:%M}"
    )
    assert_feasible(result, world, travel)
