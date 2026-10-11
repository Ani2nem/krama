"""The operations the business actually performs, in one place.

Everything above this line - CLI, MCP tools, and later the HTTP API and the agents -
is a different way of asking for the same handful of things: what does the world look
like, plan the week, price a job into it, repair after a disruption, commit the
result. Writing those once means an agent and a dispatcher cannot end up with
subtly different behaviour, which is a failure mode worth designing out rather than
testing for.

It is also where tracing belongs. The solver and the domain stay free of any
observability imports; this layer opens the spans and records what came back, so a
whole episode joins up under one dispatch id without the pure code knowing tracing
exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from krama.config import BusinessParams, Provenance
from krama.domain.autonomy import AutonomyDecision, AutonomyPolicy, decide
from krama.domain.diff import PlanDiff, diff_plans
from krama.domain.events import Event
from krama.domain.invariants import ValidationConfig, Violation, validate_plan
from krama.domain.models import (
    CostBreakdown,
    CrewRoute,
    Job,
    JobId,
    Location,
    PlanVersion,
    WorkerId,
)
from krama.domain.state import TrafficOverride, WorldState, fold
from krama.obs.correlation import current_dispatch_id, dispatch
from krama.obs.tracing import record, span
from krama.persistence.log import PlanConflict, Workspace
from krama.scheduler.booking import (
    BaselineCache,
    BookingOptions,
    suggest_booking_slots,
)
from krama.scheduler.costing import RouteCost, cost_plan
from krama.scheduler.day_planner import SolveParams
from krama.scheduler.horizon import HorizonParams, HorizonResult, plan_horizon
from krama.scheduler.repair import (
    RepairOptions,
    carve_out_locked,
    locked_jobs,
    repair_plan,
    underway_jobs,
)
from krama.scheduler.travel.base import OverrideAdjustedProvider, TravelProvider
from krama.scheduler.travel.factory import TravelMode, build_travel
from krama.scheduler.travel.incidents import (
    IncidentFeed,
    NoIncidentFeed,
    build_feed,
    overrides_from,
)


class ServiceError(RuntimeError):
    """Something the caller can act on: no workspace, no committed plan, a conflict."""


@dataclass(frozen=True, slots=True)
class PlanResult:
    """A plan together with everything needed to judge it."""

    plan: PlanVersion
    horizon: HorizonResult
    violations: tuple[Violation, ...]
    cost: CostBreakdown
    route_costs: tuple[RouteCost, ...]

    @property
    def feasible(self) -> bool:
        return not self.violations


class DispatchService:
    """One business, one workspace."""

    def __init__(
        self,
        workspace: Workspace,
        business: BusinessParams | None = None,
        travel_mode: TravelMode | str = TravelMode.AUTO,
        incident_feed: IncidentFeed | None = None,
    ) -> None:
        self.workspace = workspace
        self.business = business or BusinessParams.load()
        self.travel_mode = TravelMode(travel_mode)
        # Live traffic is off unless configured, because a feed makes two solves of the
        # same problem differ and the frozen snapshot exists to stop exactly that.
        self._incident_feed = incident_feed if incident_feed is not None else build_feed()
        self.tz = ZoneInfo(self.business.meta.timezone)
        # Half of a quote is re-solving the untouched day, and that answer does not
        # change between one caller and the next. The key includes the exact job set,
        # so a booking or a disruption invalidates it by construction.
        self._baseline_cache = BaselineCache()

    # ------------------------------------------------------------------ context

    def world(self, as_of: datetime | None = None) -> WorldState:
        if not self.workspace.exists:
            raise ServiceError(
                f"no workspace at {self.workspace.root}; create one with `krama init`"
            )
        state = fold(self.workspace.events.read(), as_of=as_of)
        self._absorb_rate_card(state)
        return state

    def _absorb_rate_card(self, state: WorldState) -> None:
        """Owner-set prices from the log override the config file's estimates.

        The yaml stays the shipped default; what the business actually charges is
        event-sourced. Each overridden Param flips to CONFIRMED provenance, which is
        what lets the "estimated costs" chip retire itself one real number at a
        time. Applied on every fold so the price in force is always the price of
        record - a service object never quotes from a stale card.
        """
        if not state.rate_card:
            return
        pricing = self.business.pricing
        updates = {}
        for key, value in state.rate_card.items():
            current = getattr(pricing, key, None)
            if current is None:
                continue
            updates[key] = current.model_copy(
                update={"value": value, "source": Provenance.CONFIRMED, "note": "set by the owner"}
            )
        if updates:
            self.business = self.business.model_copy(
                update={"pricing": pricing.model_copy(update=updates)}
            )

    def travel(self, world: WorldState) -> TravelProvider:
        """Travel with recorded traffic delays layered on, and live ones if configured.

        Providers stay pure; a TrafficDelay event reaches routing without any provider
        knowing the event log exists, and a live incident joins by the same door rather
        than by reaching into the matrix.
        """
        base = build_travel(self.business, self.travel_mode)
        overrides = tuple(world.traffic_overrides) + self.live_overrides(world)
        if not overrides:
            return base
        if len(overrides) == len(world.traffic_overrides):
            return OverrideAdjustedProvider(base, world.traffic_multiplier)

        def multiplier(origin_gh5: str, dest_gh5: str, at: datetime) -> float:
            product = 1.0
            for override in overrides:
                if override.applies_to(origin_gh5, dest_gh5, at):
                    product *= override.multiplier
            return product

        return OverrideAdjustedProvider(base, multiplier)

    def live_overrides(self, world: WorldState) -> tuple[TrafficOverride, ...]:
        """Whatever the traffic feed is reporting right now, as corridor multipliers.

        Empty unless a feed is configured, which is what keeps a solve reproducible by
        default. A feed that cannot be reached returns nothing rather than raising: a
        booking must not fail because a third party is down, and planning on last
        week's congestion is exactly what the system did before this existed.
        """
        feed = self._incident_feed
        if feed is None or isinstance(feed, NoIncidentFeed):
            return ()
        depot = next(iter(self.business_depots(world)), None)
        if depot is None:
            return ()
        try:
            incidents = feed.fetch(
                depot, float(self.business.service_area.radius_miles.value), world.as_of
            )
        except Exception:
            return ()
        return overrides_from(incidents, world.as_of)

    @staticmethod
    def business_depots(world: WorldState) -> tuple[Location, ...]:
        seen: dict[str, Location] = {}
        for van in world.vans.values():
            seen.setdefault(van.home_depot.geohash7, van.home_depot)
        return tuple(seen.values())

    def solve_params(self, *, allow_overtime: bool = True) -> SolveParams:
        return SolveParams.from_business(self.business, self.tz, allow_overtime=allow_overtime)

    def horizon_params(self) -> HorizonParams:
        return HorizonParams.from_business(self.business)

    def policy(self) -> AutonomyPolicy:
        return AutonomyPolicy.from_business(self.business)

    def head(self) -> PlanVersion | None:
        return self.workspace.plans.head()

    # ------------------------------------------------------------------- events

    def apply_events(self, events: Sequence[Event]) -> int:
        """Append to the log. The only way world state ever changes."""
        with span("events.apply", count=len(events), kinds=[e.type for e in events]):
            self.workspace.events.append(events)
            return len(events)

    # ------------------------------------------------------------------ planning

    def validation_config(
        self,
        *,
        allow_overtime: bool = True,
        released_job_ids: frozenset[JobId] = frozenset(),
    ) -> ValidationConfig:
        """The checker's tolerances, with the business's real overtime ceiling."""
        return ValidationConfig(
            business_tz=self.tz,
            allow_overtime=allow_overtime,
            overtime_max_minutes=int(self.business.labor.overtime_max_minutes.value),
            released_job_ids=released_job_ids,
        )

    def plan_week(
        self,
        start: date,
        *,
        allow_overtime: bool = True,
        world: WorldState | None = None,
        pinned_workers: Mapping[JobId, tuple[WorkerId, ...]] | None = None,
        now: datetime | None = None,
        authorized_releases: frozenset[JobId] = frozenset(),
    ) -> PlanResult:
        """Plan the rolling horizon and check it. Does not commit.

        Work already dispatched is carried forward rather than re-planned. Planning a
        day from scratch at eleven in the morning would otherwise silently drop the
        crew that left at six - those jobs are no longer "schedulable", so they simply
        vanish from the result. Repair already knew this; a plain re-plan did not, and
        a dispatcher pressing the button mid-morning would have erased work in
        progress without being told.
        """
        with span("plan.horizon", start=start.isoformat()) as active:
            state = world if world is not None else self.world()
            params = self.solve_params(allow_overtime=allow_overtime)
            horizon_params = self.horizon_params()

            head = self.head()
            # Locked by paperwork OR by the clock: a crew already rolling toward a
            # stop is not the solver's to reassign, dispatched event or no.
            wall = now if now is not None else datetime.now(self.tz)
            locked = sorted({*locked_jobs(state), *underway_jobs(head, state, wall)})
            carried: tuple[CrewRoute, ...] = ()
            planning_state = state
            if head is not None and locked:
                planning_state, carried = carve_out_locked(state, head, locked)

            travel = self.travel(planning_state)
            result = plan_horizon(
                world=planning_state,
                travel=travel,
                start=start,
                params=params,
                horizon_params=horizon_params,
                pinned_workers=pinned_workers,
            )
            plan = PlanVersion(
                id=self._next_plan_id(),
                parent_id=head.id if head else None,
                created_at=state.as_of,
                horizon_start=start,
                horizon_end=date.fromordinal(start.toordinal() + horizon_params.days - 1),
                routes=(*carried, *result.routes),
                unserved=result.unserved,
                label="committed",
            )
            violations = validate_plan(
                plan,
                state,
                travel,
                # A release only passes validation when somebody authorised it -
                # the checker's own rule, now actually used by the commit flow.
                self.validation_config(
                    allow_overtime=allow_overtime, released_job_ids=authorized_releases
                ),
            )
            cost, route_costs = cost_plan(plan, state, self.business, self.tz, result.unserved)

            record(
                metrics=result.metrics,
                violations=len(violations),
                content_hash=plan.content_hash,
                cost_total=round(cost.total, 2),
            )
            active.set_attribute("feasible", not violations)
            return PlanResult(plan, result, violations, cost, tuple(route_costs))

    def commit(self, plan: PlanVersion, expected_parent: str | None) -> PlanVersion:
        """Store a plan as the new head.

        Refuses an infeasible plan and refuses a stale one. Both are guards rather
        than tests: nothing that fails its own invariants should ever be stored, and a
        stale write silently discards whatever another dispatcher just committed.
        """
        with span("plan.commit", plan_id=plan.id, parent=expected_parent or ""):
            try:
                committed = self.workspace.plans.commit(plan, expected_parent)
            except PlanConflict as exc:
                raise ServiceError(str(exc)) from exc
            record(content_hash=committed.content_hash)
            return committed

    def repair(self, world: WorldState | None = None) -> tuple[RepairOptions, PlanVersion]:
        """Repair the committed plan. Returns the candidates and the baseline."""
        baseline = self.head()
        if baseline is None:
            raise ServiceError("nothing committed yet; run `krama commit` first")

        with span("plan.repair", baseline=baseline.id) as active:
            state = world if world is not None else self.world()
            options = repair_plan(
                also_locked=underway_jobs(baseline, state, datetime.now(self.tz)),
                world=state,
                travel=self.travel(state),
                baseline=baseline,
                start=baseline.horizon_start,
                params=self.solve_params(),
                horizon_params=self.horizon_params(),
            )
            best = options.best_by_fewest_calls
            record(
                candidates=len(options.candidates),
                recommended=best.strategy.name if best else "",
                customer_calls=best.customer_calls if best else -1,
            )
            active.set_attribute("blast_radius", best.diff.blast_radius.value if best else "")
            return options, baseline

    def autonomy(self, diff: PlanDiff, *, added_overtime_minutes: int = 0) -> AutonomyDecision:
        decision = decide(diff, self.policy(), added_overtime_minutes=added_overtime_minutes)
        record(autonomy=decision.decision.value)
        return decision

    # ------------------------------------------------------------------ booking

    def booking_slots(
        self,
        draft: Job,
        start: date,
        earliest_hour: int | None = None,
        latest_hour: int | None = None,
        not_before: datetime | None = None,
        allowed_weekdays: frozenset[int] | None = None,
    ) -> BookingOptions:
        """Priced options over the next working days.

        The horizon is working days, not calendar days: a five-day window that
        swallows a weekend offered three options and looked broken. ``not_before``
        keeps today honest - at nine in the evening, today is not an option, and the
        window rolls forward so the caller still sees a full hand. ``allowed_weekdays``
        is "only on Thursdays", said out loud and meant.
        """
        with span("booking.slots", service=draft.service_type.value) as active:
            state = self.world()
            head = self.head()
            # A quote mid-morning must price around the crew that is already out,
            # not silently move them: in-flight work is carved into consumed
            # capacity exactly as the committed re-plan does.
            wall = not_before if not_before is not None else datetime.now(self.tz)
            in_flight = sorted({*locked_jobs(state), *underway_jobs(head, state, wall)})
            if head is not None and in_flight:
                state, _carried = carve_out_locked(state, head, in_flight)
            horizon_params = self.horizon_params()
            rostered = {
                hours.weekday for worker in state.workers.values() for hours in worker.working_hours
            }
            wanted = allowed_weekdays if allowed_weekdays else rostered
            horizon: list[date] = []
            cursor = start
            # Walk forward until the hand is full; three weeks bounds "only Thursdays".
            for _ in range(21):
                if len(horizon) >= horizon_params.days:
                    break
                if cursor.weekday() in rostered and cursor.weekday() in wanted:
                    horizon.append(cursor)
                cursor = date.fromordinal(cursor.toordinal() + 1)
            options = suggest_booking_slots(
                world=state,
                travel=self.travel(state),
                draft=draft,
                horizon=horizon,
                params=self.solve_params(),
                business=self.business,
                cache=self._baseline_cache,
                earliest_hour=earliest_hour,
                latest_hour=latest_hour,
                not_before=not_before,
            )
            active.set_attribute("slots", len(options.slots))
            record(
                best_cost=round(options.best.marginal_cost, 2) if options.best else -1,
                spread=round(options.savings_vs_worst, 2),
                baseline_hits=self._baseline_cache.hits,
                baseline_misses=self._baseline_cache.misses,
            )
            return options

    # -------------------------------------------------------------------- diffs

    def diff(self, before_id: str, after_id: str) -> PlanDiff:
        before = self.workspace.plans.get(before_id)
        after = self.workspace.plans.get(after_id)
        if before is None or after is None:
            missing = before_id if before is None else after_id
            raise ServiceError(f"no such plan version: {missing}")
        return diff_plans(before, after, self.world())

    # ----------------------------------------------------------------- internals

    def _next_plan_id(self) -> str:
        return f"v{len(self.workspace.plans.history()) + 1:03d}"


def traced_dispatch(dispatch_id: str | None = None) -> object:
    """Convenience re-export so callers need only import from this module."""
    return dispatch(dispatch_id)


__all__ = [
    "DispatchService",
    "PlanResult",
    "ServiceError",
    "current_dispatch_id",
    "dispatch",
]
