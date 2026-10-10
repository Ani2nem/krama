"""HTTP surface for the dispatch board.

A thin layer over :class:`~krama.service.DispatchService`. Everything the board
can do, the CLI and the MCP tools can already do, which is the point of having put the
operations in one place: a dispatcher clicking a button and an agent calling a tool
take the same code path and cannot drift apart.

Solving endpoints are declared ``def`` rather than ``async def`` on purpose. CP-SAT is
CPU-bound and would block the event loop for the duration of a solve, freezing the
board and every open event stream; FastAPI runs sync handlers in a worker thread
instead. Only the stream itself is async, because that is the one thing here that is
genuinely waiting rather than working.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from krama.agents.intake import _URGENCY_NOT_ARRANGEMENT
from krama.api import views
from krama.api.models import (
    AcceptRequest,
    AskRequest,
    BookRequest,
    CrewAskSuggestion,
    DepotMoveRequest,
    DuplicateView,
    EventRequest,
    IntakeView,
    MessageView,
    NoteView,
    ParamView,
    PlanView,
    RepairView,
    SlotView,
    TextRequest,
    TriageView,
    UnavailableDayView,
    VanConfig,
    WorkerConfig,
    WorldView,
)
from krama.cli.events import EventArgumentError, build_event
from krama.config import BusinessParams
from krama.domain.invariants import ValidationConfig, validate_plan
from krama.domain.models import Job, PlanVersion, Worker
from krama.domain.state import WorldState
from krama.formatting import clock, clock_range
from krama.geocoding import GeocodeError, OutsideServiceArea
from krama.notify import Notifier, build_notifier
from krama.obs.correlation import dispatch, new_dispatch_id
from krama.obs.tracing import configure, span
from krama.persistence.log import PlanConflict, Workspace
from krama.scheduler.booking import SlotSuggestion
from krama.scheduler.pricing import quote_for
from krama.scheduler.travel.cache import CacheMiss
from krama.scheduler.travel.factory import TravelMode, build_travel
from krama.service import DispatchService, ServiceError

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"

#: A weekday mid-morning. The snapshot is keyed by day type and hour bucket, so probing
#: at wall-clock "now" asks for a weekend leg every Saturday and reports a healthy
#: container as degraded. The business does not run weekends; the probe should not
#: pretend otherwise.
_PROBE_AT = datetime(2026, 9, 21, 16, 0, tzinfo=UTC)


class Broadcaster:
    """Fans state changes out to every open board.

    Slow subscribers are dropped rather than allowed to apply back-pressure. A browser
    tab someone left open on a sleeping laptop must not be able to stall a solve.
    """

    def __init__(self, max_queue: int = 32) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._max_queue = max_queue

    def publish(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        message = json.dumps({"type": kind, **(payload or {})})
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                self._subscribers.discard(queue)

    async def stream(self) -> AsyncIterator[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(queue)
        try:
            yield 'data: {"type": "connected"}\n\n'
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=20.0)
                except TimeoutError:
                    # Keeps proxies from closing an idle connection.
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {message}\n\n"
        finally:
            self._subscribers.discard(queue)


broadcaster = Broadcaster()
app = FastAPI(title="krama", version="0.1.0")

# The board runs on Vite's dev server during development and is served from this app
# in production, so cross-origin is a development-only concern.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _owner_pin() -> str:
    """The owner's PIN: the environment first (deployment override), then the
    business config (back-office file, where the sample ships as 1234). Empty in
    both places means single-user mode and everything shows. With a pin set, the
    board is dispatcher-safe by DEFAULT - margins, week costs and the rate card
    are stripped SERVER-SIDE, so an employee's browser never even receives the
    numbers the owner considers private."""
    from_env = os.environ.get("KRAMA_OWNER_PIN", "")
    if from_env:
        return from_env
    try:
        return BusinessParams.load().meta.owner_pin
    except Exception:
        return ""


def _is_owner(request: Request) -> bool:
    pin = _owner_pin()
    if not pin:
        return True
    supplied = request.headers.get("x-owner-key", "")
    return bool(supplied) and hmac.compare_digest(supplied, pin)


def _strip_money(view: PlanView | None, request: Request) -> PlanView | None:
    """Zero the business-private money on a plan for non-owner sessions.

    The quote a customer pays is dispatcher-facing by definition - she reads it
    down the phone. What the week COSTS the business, and what each job's margin
    is, are the owner's numbers: an employee's browser never receives them.
    """
    if view is None or _is_owner(request):
        return view
    return view.model_copy(
        update={
            "redacted": True,
            "cost": view.cost.model_copy(update={key: 0.0 for key in type(view.cost).model_fields}),
        }
    )


def _strip_margins(view: IntakeView, request: Request) -> IntakeView:
    """Quotes stay - the dispatcher reads them aloud. What we KEEP does not."""
    if _is_owner(request):
        return view

    def scrub(slot: SlotView) -> SlotView:
        # The itemised build-up goes too: its labour and materials lines ARE the
        # rate card, which is owner-gated everywhere else. The total stays - the
        # dispatcher reads it down the phone, and the customer's invoice says it.
        return slot.model_copy(update={"margin": 0.0, "margin_pct": 0.0, "quote_lines": []})

    return view.model_copy(
        update={
            "redacted": True,
            "slots": [scrub(s) for s in view.slots],
            "flexible_slots": [scrub(s) for s in view.flexible_slots],
        }
    )


def _digits(phone: str) -> str:
    return "".join(ch for ch in phone if ch.isdigit())


def _active_duplicates(world: WorldState, phone: str, address: str) -> list[Job]:
    """Active bookings that look like the same caller.

    Deterministic on purpose: "this phone number already has a booking" is a
    checkable fact, so it does not go to a model - and it is only ever a WARNING,
    because a customer with two broken windows is legitimately two jobs. The
    model's opinion would add nothing but a way to be wrong in both directions.
    """
    wanted_phone = _digits(phone)
    wanted_address = address.strip().lower()
    out = []
    for job in world.active_jobs():
        same_phone = bool(wanted_phone) and _digits(job.phone) == wanted_phone
        same_address = bool(wanted_address) and job.location.address.strip().lower() == (
            wanted_address
        )
        if same_phone or same_address:
            out.append(job)
    return out


def _owner_only(request: Request) -> None:
    if not _is_owner(request):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "OwnerOnly",
                "detail": "this needs the owner's PIN",
                "remedy": "unlock with the Owner button in the top bar",
            },
        )


@lru_cache(maxsize=1)
def notifier() -> Notifier:
    """One SMS sink per process; log mode unless Twilio is configured."""
    return build_notifier()


def service() -> DispatchService:
    workspace = Workspace(os.environ.get("KRAMA_WORKSPACE", ".krama"))
    mode = TravelMode(os.environ.get("KRAMA_TRAVEL", TravelMode.AUTO.value))
    return DispatchService(workspace, BusinessParams.load(), mode)


def _fail(exc: Exception, status: int = 400, remedy: str = "") -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": type(exc).__name__, "detail": str(exc), "remedy": remedy},
    )


# --------------------------------------------------------------------------- health


# --------------------------------------------------------------------- failures
#
# Registered once for the whole app rather than caught per route. Both of these arise
# several layers below the endpoint - inside the solver, inside the travel cache - and
# a route that forgot to catch one returned "Internal Server Error" to a dispatcher for
# something with a perfectly good explanation.


@app.exception_handler(OutsideServiceArea)
def _outside_service_area(_: Request, exc: OutsideServiceArea) -> JSONResponse:
    """Not a failure to understand the address. A business answer."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "OutsideServiceArea",
            "detail": str(exc),
            "remedy": "check the address, or book it as an out-of-area job deliberately",
        },
    )


@app.exception_handler(GeocodeError)
def _geocode_failed(_: Request, exc: GeocodeError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "GeocodeError",
            "detail": str(exc),
            "remedy": "add a street number, or a city, and try again",
        },
    )


@app.exception_handler(CacheMiss)
def _travel_cache_miss(_: Request, exc: CacheMiss) -> JSONResponse:
    """A real address nobody has ever quoted before.

    The frozen snapshot covers the fixture's geography and refuses to invent a leg it
    does not have, which is right for tests and wrong mid-call. `warm` mode answers
    from the snapshot and asks OSRM for the rest.
    """
    return JSONResponse(
        status_code=409,
        content={
            "error": "CacheMiss",
            "detail": str(exc),
            "remedy": (
                "this address is not in the frozen travel snapshot. Start OSRM and run "
                "with live routing:  docker compose up -d osrm  then  "
                "KRAMA_TRAVEL=warm make api"
            ),
        },
    )


# ------------------------------------------------------------------------- auth
#
# A function URL with AuthType NONE is reachable by anyone who has the URL, and until
# now nothing behind it asked who was calling. That is fine on a laptop and not fine
# for a business's schedule, which this can read, change and book against.
#
# A shared key rather than JWTs, deliberately: there is one dispatcher and no identity
# provider, and a signing key nobody rotates is worse than a shared secret somebody
# does. When there are users, this is the seam that becomes a real dependency.


def api_key() -> str:
    return os.environ.get("KRAMA_API_KEY", "")


#: Reachable without a key. Health and readiness are how a load balancer and a deploy
#: decide whether this container works, and neither can hold a secret.
_OPEN_PATHS = frozenset({"/api/health", "/api/ready"})


@app.middleware("http")
async def require_api_key(request: Request, call_next: Any) -> Any:
    """Refuse unauthenticated calls when a key is configured.

    Unset means open, which is what makes `make dev` work with no setup. That default
    is only safe because the deployment refuses to be public without one - terraform
    will not apply a public function URL with no key, and /api/ready says which mode
    it is in, so an open deployment cannot go unnoticed.
    """
    expected = api_key()
    path = request.url.path
    if not expected or path in _OPEN_PATHS or not path.startswith("/api/"):
        return await call_next(request)

    presented = request.headers.get("x-api-key", "")
    if not presented:
        bearer = request.headers.get("authorization", "")
        presented = bearer[7:] if bearer.lower().startswith("bearer ") else ""

    # Constant time: a comparison that returns early leaks the key one character at a
    # time to anyone willing to measure.
    if not presented or not secrets.compare_digest(presented, expected):
        return JSONResponse(
            status_code=401,
            content={
                "error": "Unauthorized",
                "detail": "this deployment requires an API key",
                "remedy": "send it as X-API-Key, or as Authorization: Bearer <key>",
            },
        )
    return await call_next(request)


def stream_mode() -> str:
    """``sse`` or ``poll``. How the board should find out that something changed.

    This is a cost decision, not a technical one. Server-sent events are the better
    mechanism and are free on a server that is running anyway. On Lambda the function
    is billed for as long as the stream is held open, so one board left open for a
    working day costs about $29 a month against about $0.58 for polling every ten
    seconds - and left open overnight, more than the always-on container it replaced.
    """
    return os.environ.get("KRAMA_STREAM", "sse")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness. Deliberately does no work at all.

    A liveness probe that touches the solver restarts a healthy task whenever a solve
    is holding the worker threads, which turns a slow minute into an outage.

    It also tells the board how to watch for changes, because that is the one thing
    the client cannot work out for itself - a stream that is expensive still works,
    so there is no failure to fall back from.
    """
    return {"status": "ok", "version": app.version, "stream": stream_mode()}


@app.get("/api/ready")
def ready() -> JSONResponse:
    """Readiness. Can this container actually plan, or has it merely started?

    The failure worth catching is the process that is up and answering while missing
    something it needs, because that is the one a port check calls healthy. Each probe
    below stands for a file the image copies selectively and could stop copying:
    business parameters, the travel snapshot, the built board.

    Cheap on purpose - one parameter load and one travel leg, no solve - because a
    load balancer runs this every few seconds on every task.
    """
    checks: dict[str, str] = {}

    try:
        business = BusinessParams.load()
        checks["params"] = f"{sum(1 for _ in business.walk())} parameters"
    except Exception as exc:
        checks["params"] = f"FAILED: {exc}"
        business = None

    if business is not None:
        try:
            mode = TravelMode(os.environ.get("KRAMA_TRAVEL", TravelMode.AUTO.value))
            provider = build_travel(business, mode)
            # The fixture's own depot and first stop. Hand-typed coordinates looked
            # equivalent and were not: the addresses are geocoded, so a literal from
            # the source landed in a different geohash cell than anything frozen.
            from krama.fixtures.sample_business import DEPOT, PROBE_STOP

            leg = provider.leg(DEPOT, PROBE_STOP, _PROBE_AT)
            checks["travel"] = f"{mode.value}: depot leg {leg.minutes:.0f} min"
        except Exception as exc:
            checks["travel"] = f"FAILED: {type(exc).__name__}: {exc}"

    checks["board"] = "built" if WEB_DIST.exists() else "FAILED: no web/dist in the image"
    # Not a failure - running open is a legitimate local choice. It is reported so that
    # a deployment cannot be open without anybody being able to see that it is.
    checks["auth"] = "api key required" if api_key() else "open, no key configured"

    failed = {name: detail for name, detail in checks.items() if detail.startswith("FAILED")}
    return JSONResponse(
        status_code=503 if failed else 200,
        content={"status": "degraded" if failed else "ready", "checks": checks},
    )


# --------------------------------------------------------------------------- state


@app.get("/api/world", response_model=WorldView)
def get_world(rota_start: str | None = Query(default=None)) -> WorldView:
    svc = service()
    start = date.fromisoformat(rota_start) if rota_start else None
    try:
        return views.world_view(svc.world(), svc.business, svc.tz, rota_start=start)
    except ServiceError as exc:
        raise _fail(exc, 409, "run `krama init` to create a workspace") from exc


@app.get("/api/week")
def get_week() -> dict[str, str]:
    """The days the board should draw, whether or not anything is planned yet.

    An empty diary is not the same thing as no calendar. The board used to render
    nothing at all until a plan existed, so a business with no bookings saw a blank
    panel instead of an open week - which is exactly when you most want to see which
    days are free.
    """
    svc = service()
    try:
        start = _default_start(svc)
    except ServiceError:
        start = next_working_day_from_today()
    days = int(svc.business.horizon.days.value)
    return {"start": start.isoformat(), "end": (start + timedelta(days=days - 1)).isoformat()}


@app.get("/api/plan", response_model=PlanView | None)
def get_plan(request: Request) -> PlanView | None:
    """The committed plan, or null when nothing is committed yet."""
    svc = service()
    try:
        world = svc.world()
    except ServiceError as exc:
        raise _fail(exc, 409) from exc

    head = svc.head()
    if head is None:
        return None

    from krama.scheduler.costing import cost_plan

    travel = svc.travel(world)
    violations = validate_plan(head, world, travel, ValidationConfig(business_tz=svc.tz))
    cost, route_costs = cost_plan(head, world, svc.business, svc.tz, head.unserved)
    view = views.plan_view(head, world, cost, route_costs, violations, svc.tz)
    return _strip_money(view, request)


@app.get("/api/params", response_model=list[ParamView])
def get_params() -> list[ParamView]:
    business = BusinessParams.load()
    return [
        ParamView(path=path, value=param.value, source=param.source.value, note=param.note)
        for path, param in business.walk()
    ]


# ------------------------------------------------------------------------ planning


@app.post("/api/plan/commit", response_model=PlanView)
def commit_plan(request: Request, start_date: str | None = Query(default=None)) -> PlanView:
    svc = service()
    with dispatch(new_dispatch_id("web")), span("api.commit"):
        try:
            world = svc.world()
            start = date.fromisoformat(start_date) if start_date else _default_start(svc)
            result = svc.plan_week(start, world=world)
            if result.feasible:
                head = svc.head()
                svc.commit(result.plan, expected_parent=head.id if head else None)
                broadcaster.publish("plan", {"plan_id": result.plan.id})
        except (ServiceError, ValueError) as exc:
            raise _fail(exc, 409) from exc

        stripped = _strip_money(
            views.plan_view(
                result.plan,
                world,
                result.cost,
                list(result.route_costs),
                result.violations,
                svc.tz,
            ),
            request,
        )
        assert stripped is not None  # the input is never None here
        return stripped


@app.post("/api/repair", response_model=RepairView)
def repair() -> RepairView:
    svc = service()
    with dispatch(new_dispatch_id("web")), span("api.repair"):
        try:
            options, baseline = svc.repair()
        except ServiceError as exc:
            raise _fail(exc, 409, "commit a plan before repairing one") from exc

        best = options.best_by_fewest_calls
        return RepairView(
            baseline_plan_id=baseline.id,
            candidates=[
                views.candidate_view(c, svc.autonomy(c.diff), c is best) for c in options.candidates
            ],
            recommended=best.strategy.name if best else "",
            rationale=(
                f"{best.jobs_served} served, {best.customer_calls} call(s), "
                f"{best.changes} change(s)"
                if best
                else ""
            ),
        )


@app.post("/api/repair/apply", response_model=PlanView)
def apply_repair(strategy: str = Query(...), force: bool = Query(default=False)) -> PlanView:
    """Commit one repair candidate.

    ``force`` is what a dispatcher's approval looks like over HTTP. Without it the
    deterministic autonomy policy refuses anything customer-visible - the endpoint
    cannot be talked past, only overridden by a person who saw the diff.
    """
    svc = service()
    with dispatch(new_dispatch_id("web")), span("api.apply_repair", strategy=strategy):
        try:
            options, baseline = svc.repair()
        except ServiceError as exc:
            raise _fail(exc, 409) from exc

        candidate = next((c for c in options.candidates if c.strategy.name == strategy), None)
        if candidate is None:
            raise HTTPException(404, detail={"error": "NoSuchStrategy", "detail": strategy})

        decision = svc.autonomy(candidate.diff)
        if not decision.auto and not force:
            raise HTTPException(
                status_code=412,
                detail={
                    "error": "NeedsApproval",
                    "detail": decision.explain(),
                    "remedy": "a dispatcher must review this diff, then retry with force",
                },
            )

        world = svc.world()
        from krama.scheduler.costing import cost_plan

        travel = svc.travel(world)
        violations = validate_plan(
            candidate.plan,
            world,
            travel,
            # Promises this candidate breaks are authorised by the act of applying it;
            # `force` above is where a dispatcher agreed to that.
            ValidationConfig(
                business_tz=svc.tz,
                released_job_ids=frozenset(candidate.released_promises),
            ),
        )
        if violations:
            # A plan that fails its own invariants never reaches storage, whoever asked.
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Infeasible",
                    "detail": "; ".join(str(v) for v in violations[:3]),
                },
            )

        try:
            svc.commit(candidate.plan, expected_parent=baseline.id)
        except ServiceError as exc:
            raise _fail(exc, 409) from exc

        broadcaster.publish("plan", {"plan_id": candidate.plan.id, "strategy": strategy})
        cost, route_costs = cost_plan(
            candidate.plan, world, svc.business, svc.tz, candidate.plan.unserved
        )
        return views.plan_view(candidate.plan, world, cost, route_costs, (), svc.tz)


# -------------------------------------------------------------------------- events


@app.post("/api/events")
def record_event(request: EventRequest) -> dict[str, str]:
    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.record_event"):
        try:
            at = _clock(svc, request.at, None, default_now=True)
            assert at is not None  # default_now always yields a moment
            event = build_event(
                request.kind,
                request.target or None,
                at=at,
                dispatch_id=dispatch_id,
                until=_clock(svc, request.until, None),
                window_start=_clock(svc, request.window_start, None),
                window_end=_clock(svc, request.window_end, None),
                minutes=request.minutes,
                multiplier=request.multiplier,
                commitment_cost=request.commitment_cost,
                reason=request.reason,
            )
            svc.apply_events([event])
        except (EventArgumentError, ServiceError, ValueError) as exc:
            raise _fail(exc, 400, "check the event kind and target id") from exc

        note = ""
        if event.type in {"worker_unavailable", "van_unavailable"}:
            note = _quiet_reroute(svc)

        broadcaster.publish("world", {"event": event.type})
        return {
            "event_id": event.event_id,
            "type": event.type,
            "dispatch_id": dispatch_id,
            "note": note,
        }


def _quiet_reroute(svc: DispatchService) -> str:
    """Absorb an outage without a human in the loop, when that is safe.

    "Priya is out" used to mean a red banner and a button, even when five other
    fitters could trivially cover her day - the dispatcher was being asked to approve
    arithmetic. So: if the committed plan still validates, nothing happens. If it is
    broken, run the repair strategies, and when the best candidate keeps every
    promise - zero customer calls, zero released promises, nobody dropped - commit it
    silently and say so in one sentence. Anything that would touch a customer stays
    exactly where it was: stale banner, human decision. The autonomy policy is the
    same one the repair button enforces; this just presses the button when the policy
    says the press is free.
    """
    baseline = svc.head()
    if baseline is None:
        return ""
    world = svc.world()
    travel = svc.travel(world)
    if not validate_plan(baseline, world, travel, ValidationConfig(business_tz=svc.tz)):
        return ""  # the outage touched nothing that was planned

    try:
        options, baseline = svc.repair()
    except ServiceError:
        return ""
    best = options.best_by_fewest_calls
    if best is None or best.customer_calls:
        return ""
    served_before = {s.job_id for r in baseline.routes for s in r.stops}
    served_after = {s.job_id for r in best.plan.routes for s in r.stops}
    if not served_before <= served_after:
        return ""  # somebody's job fell out; that is a judgement, not arithmetic
    if not svc.autonomy(best.diff).auto:
        return ""
    # The decisive check. A strategy may have been AUTHORISED to release promises
    # (that is what released_promises records), but what matters is what came out:
    # validated with no release authorisation at all, so a plan that actually moved
    # or dropped a promised window fails here and stays a human decision.
    if validate_plan(best.plan, world, travel, ValidationConfig(business_tz=svc.tz)):
        return ""

    try:
        committed = svc.commit(best.plan, expected_parent=baseline.id)
    except ServiceError:
        return ""
    broadcaster.publish("plan", {"plan_id": committed.id})
    moved = best.changes
    return "rerouted automatically - every promise kept" + (
        f", {moved} internal change(s)" if moved else ""
    )


# -------------------------------------------------------------------------- agents


@app.post("/api/note", response_model=NoteView)
def read_note(
    http_request: Request, request: TextRequest, kind: str | None = Query(default=None)
) -> NoteView:
    """One box. Work out what the note is, then hand it to the right agent.

    ``kind`` overrides the classification, which is how the dispatcher corrects it
    without retyping. The override is the reason routing by model is safe here: the
    worst case costs one click, not a wrong job on the schedule.
    """
    from krama.agents.llm.factory import build_llm
    from krama.agents.router import NoteKind, route_note

    why = ""
    if kind is None:
        routed = route_note(build_llm(), request.text)
        kind = routed.value.kind if routed.value else NoteKind.BOOKING.value
        why = routed.value.why if routed.value else "could not tell, assumed a booking"

    if kind == NoteKind.DISRUPTION.value:
        return NoteView(kind=kind, why=why, disruption=run_triage(request))
    return NoteView(
        kind=NoteKind.BOOKING.value,
        why=why,
        booking=_strip_margins(run_intake(request), http_request),
    )


@app.post("/api/triage", response_model=TriageView)
def run_triage(request: TextRequest) -> TriageView:
    from krama.agents.a2a.types import submitted
    from krama.agents.llm.factory import build_llm
    from krama.agents.registry import TRIAGE_SKILL, TriageContext, build_transport

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.triage"):
        try:
            world = svc.world()
        except ServiceError as exc:
            raise _fail(exc, 409) from exc

        context = TriageContext(world=world, on_date=_default_start(svc), tz=svc.tz)
        transport = build_transport(build_llm(), triage_context=context)
        task = transport.send(
            transport.discover(TRIAGE_SKILL),
            submitted(request.text, dispatch_id=dispatch_id),
        )
        data = artifact.data if (artifact := task.artifact("events")) else {}

        return TriageView(
            state=task.state.value,
            summary=str(data.get("summary", "")),
            events=list(data.get("events", [])),
            question=(
                task.status.message.text_content if task.needs_input and task.status.message else ""
            ),
            unknown_targets=list(data.get("unknown_targets", [])),
            rejected=list(data.get("rejected", [])),
            repairs=int(data.get("repairs", 0)),
        )


@app.post("/api/triage/accept")
def accept_triage(request: AcceptRequest) -> dict[str, int]:
    """Record events a dispatcher reviewed. Nothing an agent produced is stored until
    a person has seen it."""
    from pydantic import TypeAdapter

    from krama.domain.events import Event

    svc = service()
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    with dispatch(new_dispatch_id("web")), span("api.accept_triage"):
        try:
            events = [adapter.validate_python(e) for e in request.events]
            svc.apply_events(events)
        except (ServiceError, ValueError) as exc:
            raise _fail(exc, 400) from exc
        broadcaster.publish("world", {"accepted": len(events)})
        return {"recorded": len(events)}


def _pricing(
    draft: Job | None, business: BusinessParams, driving: float, overtime: int = 0
) -> dict[str, Any]:
    """What to charge for this job in this slot, itemised.

    The driving figure differs per slot - that is the whole point of ranking them - so
    the margin does too, and a dispatcher can see that Tuesday is not just cheaper to
    serve but worth more.
    """
    if draft is None:
        return {}
    quote = quote_for(draft, business, driving_cost=driving, overtime_minutes=overtime)
    return {
        "quote_total": quote.total,
        "quote_lines": quote.explain(),
        "margin": round(quote.margin, 2),
        "margin_pct": round(quote.margin_pct, 1),
    }


_WEEKDAY = ("Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays")

#: How much warning a crew needs before a same-day job. Below this, today is not real.
_SAME_DAY_LEAD = timedelta(minutes=45)


def _booking_clock(
    svc: DispatchService, world: WorldState, duration_min: int
) -> tuple[date, datetime]:
    """Where the booking window opens, given what time it actually is.

    At 9:14 on a Wednesday evening the board offered "Wed, arrive 4:00 PM" - five
    hours gone - because the search never knew the time of day. Today stays on the
    table only while a job could still start, run, and finish inside somebody's
    reachable hours; after that the window rolls to the next working day and the
    caller still sees a full hand.
    """
    now = datetime.now(svc.tz)
    latest_end = 0
    overtime = int(svc.business.labor.overtime_max_minutes.value)
    for worker in world.workers.values():
        hours = worker.hours_for(now.weekday())
        if hours is None:
            continue
        end = hours.end.hour * 60 + hours.end.minute
        if worker.overtime_eligible:
            end += overtime
        latest_end = max(latest_end, end)
    start_minute = (now + _SAME_DAY_LEAD).hour * 60 + (now + _SAME_DAY_LEAD).minute
    today_works = latest_end > 0 and start_minute + duration_min <= latest_end
    start = next_working_day(world, now.date() if today_works else now.date() + timedelta(days=1))
    return start, now + _SAME_DAY_LEAD


def _capable_then(
    world: WorldState, draft: Job, on_date: date, start_hour: int, duration_min: int, overtime: int
) -> list[str]:
    """Who holds the certifications AND can be on site for the whole window.

    Takes the date, not just the weekday: a one-day ShiftExtended agreement changes
    the answer for that date, and a message still claiming "nobody can work these
    hours" after two people said yes is worse than no message.
    """
    names = []
    for worker in world.workers.values():
        if not draft.required_certifications <= worker.certifications:
            continue
        hours = worker.hours_for(on_date.weekday())
        if hours is None:
            continue
        end = hours.end.hour * 60 + hours.end.minute
        if worker.overtime_eligible:
            end += overtime
        extension = world.extension_for(worker.id, on_date)
        if extension is not None:
            end = max(end, extension.hour * 60 + extension.minute)
        if start_hour * 60 + duration_min <= end:
            names.append(worker.name)
    return sorted(names)


def _when_text(earliest: int | None, latest: int | None) -> str:
    """The caller's stated hours as a phrase for the intake form."""
    from datetime import time as _time

    if earliest is None and latest is None:
        return ""
    if earliest is not None and latest is not None:
        return clock_range(_time(earliest), _time(latest))
    if earliest is not None:
        return f"after {clock(_time(earliest))}"
    return f"before {clock(_time(latest or 0))}"


def _cover_note(capable: list[str], slot: SlotSuggestion) -> str:
    """Whether anyone else could take this slot. One name is a warning, not trivia.

    Worded around the actual bottleneck. A two-person crew where only Marcus holds the
    certification is not "the only fitter" - it is Marcus who cannot be replaced, and
    saying so tells the owner exactly who to train a backup for.
    """
    others = [n for n in capable if n not in slot.worker_names]
    if others:
        return f". Could also be covered by {', '.join(others)}"
    pinch = [n for n in slot.worker_names if n in capable]
    who = " and ".join(pinch) if pinch else "this crew"
    return f". No cover: {who} cannot be replaced for these hours - if they are out, this moves"


def _bottleneck_or(
    detail: str,
    world: WorldState,
    plan: PlanVersion | None,
    draft: Job,
    on_date: date,
    start_hour: int,
    overtime: int,
    tz: tzinfo,
) -> str:
    """Replace "no room at an acceptable cost" with the fact underneath it.

    Thursday was refused while three fitters and three vans sat free, and the message
    said cost. The truth was narrower: the one person certified for the work who can
    stay late enough was already booked then. A dispatcher can act on that - offer
    another day, or ask Dan - but not on "no room".

    "Already booked" comes from the COMMITTED PLAN's routes, never from job windows
    alone. The first version listed every confirmed window on the date and pinned
    them all on the bottleneck fitter - "Dan is already booked: Ani 8-12" while the
    plan on the same screen showed Marcus doing Ani's job and Dan doing nothing.
    A message the calendar can contradict is worse than no message.
    """
    capable = _capable_then(
        world, draft, on_date, start_hour, draft.estimated_duration_min, overtime
    )
    if not capable:
        needs = ", ".join(sorted(c.value.replace("_", " ") for c in draft.required_certifications))
        return f"nobody qualified for {needs} can work these hours on a {on_date:%A}"
    if len(capable) != 1 or plan is None:
        return detail

    only = capable[0]
    worker_id = next((w.id for w in world.workers.values() if w.name == only), None)
    busy = [
        f"{world.jobs[stop.job_id].customer_name if stop.job_id in world.jobs else stop.job_id}"
        f" {clock_range(stop.arrival.astimezone(tz), stop.departure.astimezone(tz))}"
        for route in plan.routes
        if route.date == on_date and worker_id in route.worker_ids
        for stop in route.stops
    ]
    if busy:
        return (
            f"{only} is the only fitter qualified who can work these hours, "
            f"and is already booked: {'; '.join(busy[:2])}"
        )
    return detail


def _crew_reason(draft: Job | None, slot: SlotSuggestion) -> str:
    """Why these people, in one line.

    Crew size and certifications come from the job catalogue, not from the model and
    not from the solver - the solver only decides *which* qualified people, and the
    answer to "why Marcus and Priya" is almost always "because two are needed and one
    of them is the only person certified for it".
    """
    if draft is None:
        return ""
    needs = ", ".join(sorted(c.value.replace("_", " ") for c in draft.required_certifications))
    people = "one fitter" if draft.crew_size == 1 else f"{draft.crew_size} fitters"
    return f"{people} needed" + (f", qualified for {needs}" if needs else "")


@app.post("/api/intake", response_model=IntakeView)
def intake_route(http_request: Request, request: TextRequest) -> IntakeView:
    return _strip_margins(run_intake(request), http_request)


def run_intake(request: TextRequest) -> IntakeView:
    from krama.agents.intake import intake
    from krama.agents.llm.factory import build_llm
    from krama.api.models import DraftView

    svc = service()
    with dispatch(new_dispatch_id("web")), span("api.intake"):
        try:
            world = svc.world()
        except ServiceError as exc:
            raise _fail(exc, 409) from exc

        from krama.geocoding import for_service_area

        live_depot = next((v.home_depot for v in world.vans.values() if v.home_depot.address), None)
        result = intake(
            build_llm(),
            request.text,
            business=svc.business,
            now=datetime.now(svc.tz),
            # Bounded around the depot the WORLD knows, not the fixture constant -
            # otherwise a moved shop would keep geocoding callers around the old
            # pin, which is exactly the silent drift the depot event exists to kill.
            geocoder=for_service_area(depot=live_depot),
        )
        call = result.call
        draft = DraftView(
            customer_name=call.customer_name if call else "",
            phone=call.phone if call else "",
            address=call.address if call else "",
            service_type=call.service_type if call else "",
            duration_minutes=result.estimate.duration_min if result.estimate else 0,
            duration_confidence=result.estimate.confidence_min if result.estimate else 0,
            crew_size=result.estimate.crew_size if result.estimate else 0,
            certifications=(
                sorted(c.value for c in result.estimate.required_certifications)
                if result.estimate
                else []
            ),
            commitment_cost=result.commitment_cost,
            commitment_quotes=list(result.commitment_quotes),
            lead_time_days=result.estimate.lead_time_days if result.estimate else 0,
            site_notes=call.site_notes if call else "",
            lat=result.draft.location.lat if result.draft else None,
            lon=result.draft.location.lon if result.draft else None,
        )

        slots: list[SlotView] = []
        flexible: list[SlotView] = []
        unavailable: list[UnavailableDayView] = []
        crew_ask: CrewAskSuggestion | None = None
        if result.bookable and result.draft is not None:
            # A caller who stated nothing gets civil hours, not the crack of dawn:
            # Marcus starts at six, so an unconstrained solve offered every customer
            # 6:00 AM. But only when *nothing* was stated - "before we open at nine"
            # deliberately keeps early starts available, because arriving before the
            # doors open is the crew's business, not the caller's.
            earliest = result.earliest_hour
            if earliest is None and result.latest_hour is None:
                earliest = 8
            start, not_before = _booking_clock(svc, world, result.draft.estimated_duration_min)
            days = frozenset(result.preferred_weekdays) or None
            options = svc.booking_slots(
                result.draft,
                start,
                earliest_hour=earliest,
                latest_hour=result.latest_hour,
                not_before=not_before,
                allowed_weekdays=days,
            )

            # Who could actually take this work at these hours. One name is a single
            # point of failure the dispatcher should see before promising anything.
            overtime = int(svc.business.labor.overtime_max_minutes.value)
            capable = _capable_then(
                world,
                result.draft,
                start,
                earliest or 8,
                result.draft.estimated_duration_min,
                overtime,
            )
            slots = [
                SlotView(
                    date=s.on_date.isoformat(),
                    day=f"{s.on_date:%a %d %b}",
                    window=clock_range(
                        s.quoted_window.start.astimezone(svc.tz),
                        s.quoted_window.end.astimezone(svc.tz),
                    ),
                    arrival=clock(s.arrival.astimezone(svc.tz)),
                    marginal_cost=round(s.marginal_cost, 2),
                    crew=" + ".join(s.worker_names),
                    crew_ids=list(s.worker_ids),
                    from_label=s.from_label,
                    leg_minutes=s.leg_minutes,
                    leg_miles=s.leg_miles,
                    crew_reason=_crew_reason(result.draft, s) + _cover_note(capable, s),
                    reason=s.reason,
                    needs_overtime=s.overtime_minutes > 0,
                    **_pricing(result.draft, svc.business, s.marginal_cost, s.overtime_minutes),
                )
                for s in options.slots
            ]

            # The stated hours cost real money when they run past a shift, and the
            # caller may not know that. When the stated hours start in the afternoon,
            # compute the in-hours alternatives too and hand them over separately -
            # the board shows them behind a tab, so the first thing on screen is what
            # was asked for and the cheaper options are one click deep, not mixed in.
            if slots and result.earliest_hour is not None and result.earliest_hour >= 12:
                relaxed = svc.booking_slots(
                    result.draft,
                    start,
                    earliest_hour=8,
                    not_before=not_before,
                    allowed_weekdays=days,
                )
                ceiling = min(s.quote_total for s in slots)
                for best in sorted(relaxed.slots, key=lambda r: r.marginal_cost)[:3]:
                    priced = _pricing(
                        result.draft, svc.business, best.marginal_cost, best.overtime_minutes
                    )
                    if priced["quote_total"] >= ceiling:
                        continue
                    flexible.append(
                        SlotView(
                            date=best.on_date.isoformat(),
                            day=f"{best.on_date:%a %d %b}",
                            window=clock_range(
                                best.quoted_window.start.astimezone(svc.tz),
                                best.quoted_window.end.astimezone(svc.tz),
                            ),
                            arrival=clock(best.arrival.astimezone(svc.tz)),
                            marginal_cost=round(best.marginal_cost, 2),
                            crew=" + ".join(best.worker_names),
                            crew_ids=list(best.worker_ids),
                            from_label=best.from_label,
                            leg_minutes=best.leg_minutes,
                            leg_miles=best.leg_miles,
                            crew_reason=_crew_reason(result.draft, best),
                            reason=best.reason,
                            needs_overtime=best.overtime_minutes > 0,
                            outside_preference=True,
                            **priced,
                        )
                    )

            # An empty list explains nothing. If the caller said "after four" and no
            # day can hold the work by then, that is the single most useful sentence
            # on the screen - it is what the dispatcher says back down the phone.
            unavailable = [
                UnavailableDayView(
                    day=f"{u.on_date:%a %d %b}",
                    reason=_with_remedy(
                        _bottleneck_or(
                            u.detail,
                            world,
                            svc.head(),
                            result.draft,
                            u.on_date,
                            earliest or 8,
                            overtime,
                            svc.tz,
                        )
                    ),
                )
                for u in options.unavailable
            ]
            if not slots and options.unavailable:
                crew_ask = _suggest_crew_ask(
                    svc, world, result.draft, options.unavailable[0].on_date, earliest or 8
                )

        return IntakeView(
            draft=draft,
            bookable=result.bookable,
            missing=list(result.missing_required),
            ask_next=list(result.ask_next),
            slots=slots,
            flexible_slots=flexible,
            unavailable=unavailable,
            repairs=result.extraction.repairs,
            note=result.geocode_note,
            when_text=_when_text(result.earliest_hour, result.latest_hour),
            # Deterministic, from the transcript itself - the same vocabulary the
            # commitment grounding uses to tell urgency from arrangement.
            asked_for_speed=bool(_URGENCY_NOT_ARRANGEMENT.search(request.text)),
            crew_ask=crew_ask,
            not_offered=result.not_offered,
            possible_duplicates=[
                DuplicateView(
                    job_id=j.id,
                    customer=j.customer_name,
                    when=(
                        f"{j.windows[0].start.astimezone(svc.tz):%a %d %b, %-I:%M %p}"
                        if j.windows
                        else "unscheduled"
                    ),
                    what=j.service_type.value.replace("_", " "),
                )
                for j in _active_duplicates(world, draft.phone, draft.address)
            ],
        )


@app.post("/api/book")
def book_slot(request: BookRequest) -> dict[str, str]:
    """Accept a quoted slot and put the job in the diary.

    The list of options was priced and then had no button, so a dispatcher could see
    that Tuesday was four times cheaper and had no way to act on it. This is that
    button.

    The draft comes back from the client rather than being held server-side between
    the two calls. Nothing an agent produced is stored until a person agrees to it,
    and a draft sitting in memory waiting to be confirmed is stored.

    The job is recorded as CONFIRMED with the window the customer was read out, which
    is what makes it expensive to move later - a promise, not a pencil mark.
    """
    from krama.domain.enums import Certification, CommitmentState, ServiceType
    from krama.domain.events import CrewAskClosed, JobConfirmed, JobRequested
    from krama.domain.models import GlassSpec, Location, Provenance, TimeWindow

    svc = service()
    draft = request.draft
    if not request.allow_duplicate:
        existing = _active_duplicates(svc.world(), draft.phone, draft.address)
        if existing:
            twins = "; ".join(
                f"{j.customer_name} ({j.windows[0].start.astimezone(svc.tz):%a %d %b, %-I:%M %p})"
                if j.windows
                else j.customer_name
                for j in existing[:3]
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "PossibleDuplicate",
                    "detail": f"this caller already has a booking: {twins}",
                    "remedy": (
                        "if it is a second job, book again with allow_duplicate; "
                        "if it is a change, edit the existing booking instead"
                    ),
                },
            )
    if draft.lat is None or draft.lon is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "NoLocation",
                "detail": "this draft has no address we could place on the map",
                "remedy": "add a street number and city, then read the slots again",
            },
        )

    with dispatch(new_dispatch_id("web")), span("api.book"):
        world = svc.world()
        # Wall time, not world.as_of. A replayed world's clock is its last event, so
        # stamping new events with it backdated every booking to the seed - and gave
        # book and cancel identical timestamps, where ordering falls to the random
        # event id and a cancellation can fold *before* the confirmation it undoes.
        now = datetime.now(svc.tz)
        arrival_probe = datetime.fromisoformat(f"{request.date}T{request.arrival}").replace(
            tzinfo=svc.tz
        )
        if arrival_probe < now:
            # The board can only offer future slots, but the API could be handed
            # yesterday - and a confirmed promise in the past is a ghost no plan can
            # ever serve, sitting on the books forever.
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "SlotInThePast",
                    "detail": f"{request.date} {request.arrival} has already happened",
                    "remedy": "read the call again for current slots",
                },
            )
        # Highest existing suffix plus one, never a recount: len() shrinks if ids
        # ever leave the dict and two same-moment bookings would mint twins.
        taken = [int(jid.split("-")[-1]) for jid in world.jobs if jid.split("-")[-1].isdigit()]
        job_id = f"j-{max([*taken, 500]) + 1}"
        arrival = datetime.fromisoformat(f"{request.date}T{request.arrival}").replace(tzinfo=svc.tz)
        # The customer is promised an *arrival* window; the stored window has to hold
        # the work as well, because the solver bounds completion by it and the checker
        # binds a confirmed job to it either way.
        #
        # Getting this wrong booked a three-hour storefront job into a two-hour window
        # and produced a plan with no routes in it at all: the job was impossible, so
        # nothing could be scheduled, and the board said "committed" and showed an
        # empty week.
        opens = arrival.replace(minute=arrival.minute // 15 * 15, second=0, microsecond=0)
        arrival_window = timedelta(minutes=svc.business.scheduling.quoted_window_minutes.value)
        quoted = arrival_window + timedelta(minutes=draft.duration_minutes)

        job = Job(
            id=job_id,
            customer_id=f"c-{job_id}",
            customer_name=draft.customer_name or "Unnamed caller",
            phone=draft.phone,
            location=Location(lat=draft.lat, lon=draft.lon, address=draft.address),
            service_type=ServiceType(draft.service_type),
            glass_spec=GlassSpec(),
            required_certifications=frozenset(Certification(c) for c in draft.certifications),
            crew_size=draft.crew_size,
            estimated_duration_min=draft.duration_minutes,
            duration_confidence_min=draft.duration_confidence,
            commitment_cost=draft.commitment_cost,
            site_notes=draft.site_notes,
            quoted_total=request.quoted_total,
            booking_note=request.booking_note,
            # The pre-tax share of the accepted price. This is the weight the
            # scheduler gives the job in every unserved penalty from now on.
            revenue=round(
                request.quoted_total / (1 + float(svc.business.pricing.tax_rate.value)), 2
            ),
            requested_at=now,
            windows=(TimeWindow(start=opens, end=opens + quoted),),
            commitment_state=CommitmentState.CONFIRMED,
            provenance=Provenance(
                source_channel="board", received_at=now, transcript=request.transcript
            ),
        )

        events: list[CrewAskClosed | JobConfirmed | JobRequested] = [
            JobRequested(
                event_id=new_dispatch_id("job"),
                occurred_at=now,
                recorded_at=now,
                dispatch_id="web",
                job=job,
            ),
            JobConfirmed(
                event_id=new_dispatch_id("confirm"),
                occurred_at=now,
                recorded_at=now,
                dispatch_id="web",
                job_id=job_id,
                window=job.windows[0],
                commitment_cost=draft.commitment_cost,
            ),
        ]
        # Booking IS the resolution of a crew ask. The red strip is a promise to
        # call the customer back; once they are in the diary, leaving it lit until
        # someone also finds the Resolve button is a chore that reads as a bug.
        events.extend(
            CrewAskClosed(
                event_id=new_dispatch_id("askc"),
                occurred_at=now,
                recorded_at=now,
                dispatch_id="web",
                ask_id=ask_id,
                outcome="booked",
                job_id=job_id,
            )
            for ask_id, ask in world.crew_asks.items()
            if _digits(ask.phone) and _digits(ask.phone) == _digits(draft.phone)
        )
        svc.apply_events(events)
        # Re-plan and commit, so the job is on the calendar the moment it is booked.
        # Recording the events alone left it in the world and invisible on the board,
        # which reads as the button having done nothing at all - the single worst
        # outcome for a button, because the next thing a dispatcher does is press it
        # again.
        placed = "unplaced"
        try:
            pin = {job_id: tuple(request.crew_ids)} if request.crew_ids else None
            plan = svc.plan_week(start=_default_start(svc), pinned_workers=pin)
            head = svc.head()
            committed = svc.commit(plan.plan, expected_parent=head.id if head else None)
            placed = (
                "scheduled"
                if any(job_id in route.job_ids for route in committed.routes)
                else "booked but not yet scheduled"
            )
            broadcaster.publish("plan", {"plan_id": committed.id})
        except (ServiceError, PlanConflict):
            # The booking is recorded either way. A failed re-plan is a scheduling
            # problem to look at, not a reason to lose the customer's appointment.
            broadcaster.publish("world", {"booked": job_id})

        offer_note = _float_overtime_offer(svc, job_id, now)

        # Echo exactly the promise the slot card made: the arrival window the
        # customer was read. The old string fused the window's open with the
        # estimate's finish ("6:00 - 8:40 AM") - a time that appeared nowhere else
        # in the system and could be contradicted by both the slot card above it
        # and the calendar beside it.
        window_opens = job.windows[0].start.astimezone(svc.tz)
        return {
            "job_id": job_id,
            "customer": job.customer_name,
            "when": (
                f"{window_opens:%a %d %b}, arriving "
                f"{clock_range(window_opens, window_opens + arrival_window)}"
            ),
            "status": placed,
            "overtime": offer_note,
        }


def _with_remedy(reason: str) -> str:
    """A refusal that names its own fix. "No van stocks 1x shower_kit" is true and
    useless alone - the dispatcher's next question is always "so what do I do"."""
    if "no van stocks" in reason:
        return reason + " - restock a van in Configure"
    return reason


def _suggest_crew_ask(
    svc: DispatchService,
    world: WorldState,
    draft: Job,
    on_date: date,
    earliest_hour: int,
) -> CrewAskSuggestion | None:
    """When no day can hold the wanted hours, name who could change that.

    "Nobody certified can work these hours" is true and useless on its own: the
    right next sentence is "...unless Marcus or Priya says yes to staying late".
    Candidates are everyone certified and rostered that day who is not marked out -
    reach is deliberately NOT checked, because reach is the thing being asked about.
    """
    start = datetime.combine(on_date, time(min(earliest_hour, 23), 0), tzinfo=svc.tz)
    # Work end PLUS the ride back to the shop: the first version asked people to
    # "stay to 8:00" for a job finishing at 8:00, and the solver - which counts the
    # drive home as shift time, correctly - kept refusing the slot their yes was
    # supposed to unlock. 45 minutes covers the service radius pessimistically.
    until = start + timedelta(minutes=draft.estimated_duration_min + 45)
    opens = datetime.combine(on_date, time(0, 0), tzinfo=svc.tz)
    candidates = sorted(
        (
            w
            for w in world.workers.values()
            if w.is_certified_for(draft.required_certifications)
            and w.hours_for(on_date.weekday()) is not None
            and world.is_worker_available(w.id, opens, opens + timedelta(days=1))
        ),
        key=lambda w: w.id,
    )
    if len(candidates) < draft.crew_size:
        return None  # not an hours problem; nobody could cover it however late

    # The hours must actually BE the blocker: the wanted span has to run past every
    # candidate's standard reach. Without this check the suggestion fired on a van
    # stock problem and asked a dispatcher to check whether Dan - free all day -
    # could "stay to 10:15 AM", which is the kind of sentence that costs a product
    # its credibility in one reading.
    overtime_allowance = timedelta(minutes=int(svc.business.labor.overtime_max_minutes.value))

    def standard_reach(worker: Worker) -> datetime | None:
        hours = worker.hours_for(on_date.weekday())
        if hours is None:
            return None
        close = datetime.combine(on_date, hours.end, tzinfo=svc.tz)
        return close + (overtime_allowance if worker.overtime_eligible else timedelta())

    reaches = [r for w in candidates if (r := standard_reach(w)) is not None]
    if not reaches or until <= max(reaches):
        return None  # somebody can already reach these hours; the blocker is elsewhere
    names = " and ".join(w.name for w in candidates)
    who = f"{draft.crew_size} of {names}" if draft.crew_size > 1 else names
    return CrewAskSuggestion(
        message=(
            f"Check with {who} whether they can stay to "
            f"{clock(until)} on {on_date:%A} - the hours are the only blocker."
        ),
        on_date=on_date.isoformat(),
        until=f"{until:%H:%M}",
        until_label=clock(until),
        candidate_ids=[w.id for w in candidates],
        candidate_names=[w.name for w in candidates],
    )


@app.post("/api/config/worker")
def configure_worker(request: WorkerConfig) -> dict[str, str]:
    """Add or edit a fitter. Registration events already upsert, so the roster is
    editable without a schema change - and when a payroll system is connected one
    day, its export lands on this same endpoint.

    What the panel does not edit survives an edit: home location and loaded cost
    stay whatever they were, and a brand-new fitter starts from the depot at the
    blended default until somebody knows better.
    """
    from krama.domain.enums import Certification
    from krama.domain.events import WorkerRegistered
    from krama.domain.models import DayHours, Worker

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.config_worker"):
        world = svc.world()
        existing = world.workers.get(request.id) if request.id else None
        worker_id = request.id or f"w-{request.name.lower().replace(' ', '-')}"
        if not request.name.strip():
            raise HTTPException(422, detail={"error": "NoName", "detail": "a fitter needs a name"})
        try:
            start = datetime.strptime(request.shift_start, "%H:%M").time()
            end = datetime.strptime(request.shift_end, "%H:%M").time()
        except ValueError as exc:
            raise HTTPException(
                422, detail={"error": "BadHours", "detail": "shift times are HH:MM"}
            ) from exc
        if end <= start:
            raise HTTPException(
                422, detail={"error": "BadHours", "detail": "a shift must end after it starts"}
            )
        try:
            certs = frozenset(Certification(c) for c in request.certifications)
        except ValueError as exc:
            raise HTTPException(
                422, detail={"error": "BadCertification", "detail": str(exc)}
            ) from exc

        depot = next(
            (van.home_depot for van in world.vans.values() if van.home_depot.address), None
        )
        home = existing.home_location if existing else depot
        if home is None:
            raise HTTPException(
                409, detail={"error": "NoDepot", "detail": "no van to copy a depot from"}
            )

        now = datetime.now(svc.tz)
        worker = Worker(
            id=worker_id,
            name=request.name.strip(),
            phone=request.phone.strip(),
            certifications=certs,
            working_hours=tuple(DayHours(weekday=d, start=start, end=end) for d in range(5)),
            home_location=home,
            overtime_eligible=request.overtime_eligible,
            loaded_cost_per_hour=existing.loaded_cost_per_hour if existing else 50.0,
        )
        svc.apply_events(
            [
                WorkerRegistered(
                    event_id=new_dispatch_id("wr"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    worker=worker,
                )
            ]
        )
        broadcaster.publish("world", {"event": "worker_registered"})
        return {"worker_id": worker_id, "status": "saved"}


@app.delete("/api/config/worker/{worker_id}")
def remove_worker(worker_id: str) -> dict[str, str]:
    from krama.domain.events import WorkerRemoved

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.remove_worker"):
        if worker_id not in svc.world().workers:
            raise HTTPException(404, detail={"error": "NoSuchWorker", "detail": worker_id})
        now = datetime.now(svc.tz)
        svc.apply_events(
            [
                WorkerRemoved(
                    event_id=new_dispatch_id("wx"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    worker_id=worker_id,
                )
            ]
        )
        broadcaster.publish("world", {"event": "worker_removed"})
        return {"worker_id": worker_id, "status": "removed"}


@app.post("/api/config/van")
def configure_van(request: VanConfig) -> dict[str, str]:
    from krama.domain.events import VanRegistered
    from krama.domain.models import Van

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.config_van"):
        world = svc.world()
        existing = world.vans.get(request.id) if request.id else None
        taken = {
            int(v.id.split("-")[-1]) for v in world.vans.values() if v.id.split("-")[-1].isdigit()
        }
        van_id = request.id or f"van-{max(taken, default=0) + 1}"
        template = existing or next(iter(world.vans.values()), None)
        if template is None:
            raise HTTPException(
                409, detail={"error": "NoDepot", "detail": "no van to copy a depot from"}
            )
        if request.stock is not None:
            bad = {k: v for k, v in request.stock.items() if v < 0 or v > 99}
            if bad:
                raise HTTPException(
                    422,
                    detail={"error": "BadStock", "detail": f"counts must be 0..99, got {bad}"},
                )
        now = datetime.now(svc.tz)
        base_stock = dict(existing.stock) if existing else dict(template.stock)
        van = Van(
            id=van_id,
            label=request.label.strip() or van_id,
            rack_slots=template.rack_slots,
            stock=(
                {k: v for k, v in request.stock.items() if v > 0}
                if request.stock is not None
                else base_stock
            ),
            home_depot=template.home_depot,
            cost_per_mile=template.cost_per_mile,
        )
        svc.apply_events(
            [
                VanRegistered(
                    event_id=new_dispatch_id("vr"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    van=van,
                )
            ]
        )
        broadcaster.publish("world", {"event": "van_registered"})
        return {"van_id": van_id, "status": "saved"}


@app.get("/api/session")
def session_info(request: Request) -> dict[str, bool]:
    """What this browser is allowed to see. The client renders accordingly; the
    server strips regardless, so the flag is a courtesy, not the security."""
    return {"owner_pin_set": bool(_owner_pin()), "owner": _is_owner(request)}


#: The pricing fields an owner may set, with sanity rails. The rails are generous -
#: they exist to catch a slipped decimal ("$9500/hour"), not to opine on pricing.
RATE_FIELDS: dict[str, tuple[float, float]] = {
    "labour_rate_per_hour": (20.0, 500.0),
    "call_out_fee": (0.0, 500.0),
    "materials_markup": (1.0, 4.0),
    "minimum_charge": (0.0, 1000.0),
    "after_hours_rate_multiplier": (1.0, 3.0),
    "emergency_uplift": (1.0, 3.0),
    "tax_rate": (0.0, 0.2),
}


@app.get("/api/config/pricing")
def read_pricing(request: Request) -> dict[str, object]:
    """The rate card, owner's eyes only."""
    _owner_only(request)
    svc = service()
    svc.world()  # absorb any owner-set prices before reading
    pricing = svc.business.pricing
    return {
        "fields": [
            {
                "key": key,
                "value": getattr(pricing, key).value,
                "source": getattr(pricing, key).source.value,
                "min": lo,
                "max": hi,
            }
            for key, (lo, hi) in RATE_FIELDS.items()
        ]
    }


@app.post("/api/config/pricing")
def set_pricing(request: Request, body: dict[str, float]) -> dict[str, str]:
    """Set real prices. Each value lands in the event log with an author and a
    timestamp, flips its parameter to CONFIRMED, and from the next quote onward is
    simply the price - the "estimated costs" chip retires one number at a time."""
    from krama.domain.events import RateCardChanged

    _owner_only(request)
    svc = service()
    entries: dict[str, float] = {}
    for key, value in body.items():
        rails = RATE_FIELDS.get(key)
        if rails is None:
            raise HTTPException(
                422, detail={"error": "UnknownRate", "detail": f"{key!r} is not a rate field"}
            )
        lo, hi = rails
        if not (lo <= float(value) <= hi):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "OutOfRails",
                    "detail": f"{key} = {value} is outside {lo}..{hi}",
                    "remedy": "check for a slipped decimal",
                },
            )
        entries[key] = float(value)
    if not entries:
        raise HTTPException(422, detail={"error": "Empty", "detail": "nothing to set"})

    now = datetime.now(svc.tz)
    svc.apply_events(
        [
            RateCardChanged(
                event_id=new_dispatch_id("rate"),
                occurred_at=now,
                recorded_at=now,
                dispatch_id=new_dispatch_id("web"),
                entries=entries,
            )
        ]
    )
    broadcaster.publish("world", {"event": "rate_card_changed"})
    return {"status": f"{len(entries)} price(s) set - quotes use them from the next call"}


@app.post("/api/config/depot")
def move_depot(http_request: Request, request: DepotMoveRequest) -> dict[str, str]:
    """Move the shop. Deliberately the hardest edit in the product - and owner-only.

    The depot was once wrong by four road miles and every route ever planned
    carried the error, so this endpoint refuses everything it possibly can: the
    address must be typed twice and match exactly, it must geocode to an actual
    building (not a road, not a town), and it must sit inside the service radius
    of the current depot - a shop does not teleport across the country by typo.
    One event rewrites every van's home in the same breath, so no stored record
    can disagree with another afterwards.
    """
    _owner_only(http_request)
    from krama.domain.events import DepotMoved
    from krama.geocoding import for_service_area

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.move_depot"):
        wanted = request.address.strip()
        if not wanted:
            raise HTTPException(422, detail={"error": "NoAddress", "detail": "type the address"})
        if wanted != request.confirm.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ConfirmMismatch",
                    "detail": "the two entries differ - retype the address to confirm",
                },
            )

        world = svc.world()
        current = next((v.home_depot for v in world.vans.values() if v.home_depot.address), None)
        geocoder = for_service_area(depot=current)
        located = geocoder.geocode(wanted)  # raises AddressTooVague / OutsideServiceArea
        if located.precision not in {"house", "poi"}:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "NotABuilding",
                    "detail": f"'{wanted}' resolved to a {located.precision}, not a building",
                    "remedy": "include the street number",
                },
            )

        moved_miles = current.haversine_miles(located) if current else 0.0
        now = datetime.now(svc.tz)
        svc.apply_events(
            [
                DepotMoved(
                    event_id=new_dispatch_id("depot"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    location=located,
                )
            ]
        )
        broadcaster.publish("world", {"event": "depot_moved"})
        return {
            "address": located.address,
            "moved_miles": f"{moved_miles:.1f}",
            "status": (
                "every van now starts here; the committed plan re-validates against "
                "the new drives on its own"
            ),
        }


@app.delete("/api/config/van/{van_id}")
def remove_van(van_id: str) -> dict[str, str]:
    from krama.domain.events import VanRemoved

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.remove_van"):
        world = svc.world()
        if van_id not in world.vans:
            raise HTTPException(404, detail={"error": "NoSuchVan", "detail": van_id})
        if len(world.vans) <= 1:
            # The last van carries the only record of the depot - every new van and
            # every new fitter copies its home from an existing one. Deleting it
            # bricked the fleet: nothing left to copy from, nothing addable, found
            # by a test that emptied the garage and could not refill it.
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "LastVan",
                    "detail": "the last van carries the depot record and cannot be removed",
                    "remedy": "add the replacement van first, then retire this one",
                },
            )
        now = datetime.now(svc.tz)
        svc.apply_events(
            [
                VanRemoved(
                    event_id=new_dispatch_id("vx"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    van_id=van_id,
                )
            ]
        )
        broadcaster.publish("world", {"event": "van_removed"})
        return {"van_id": van_id, "status": "removed"}


@app.post("/api/asks")
def open_ask(request: AskRequest) -> dict[str, str]:
    """Hold a "can anyone make these hours?" on the board until it is answered."""
    from krama.domain.events import CrewAskOpened

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.ask_open"):
        now = datetime.now(svc.tz)
        on_date = date.fromisoformat(request.on_date)
        until = datetime.combine(
            on_date, datetime.strptime(request.until, "%H:%M").time(), tzinfo=svc.tz
        )
        ask_id = new_dispatch_id("ask")
        svc.apply_events(
            [
                CrewAskOpened(
                    event_id=new_dispatch_id("ev"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    ask_id=ask_id,
                    customer_name=request.customer_name,
                    phone=request.phone,
                    transcript=request.transcript,
                    on_date=on_date,
                    until_time=until,
                    candidate_ids=tuple(request.candidate_ids),
                    detail=request.detail,
                )
            ]
        )
        broadcaster.publish("world", {"event": "crew_ask_opened"})
        return {"ask_id": ask_id}


@app.post("/api/asks/{ask_id}/extend")
def extend_for_ask(ask_id: str, worker_id: str = Query(...)) -> dict[str, str]:
    """One fitter said yes: record the one-day agreement that makes the hours real.

    The extension is a fact with a date on it - scheduling honours it for that day,
    pay treats every minute past the rostered end as overtime. The ask stays open:
    saying yes makes the slot bookable, booking the customer closes the loop.
    """
    from krama.domain.events import ShiftExtended

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.ask_extend"):
        world = svc.world()
        ask = world.crew_asks.get(ask_id)
        if ask is None:
            raise HTTPException(404, detail={"error": "NoSuchAsk", "detail": ask_id})
        if worker_id not in ask.candidate_ids:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "NotACandidate",
                    "detail": "this ask was not about that fitter",
                },
            )
        now = datetime.now(svc.tz)
        svc.apply_events(
            [
                ShiftExtended(
                    event_id=new_dispatch_id("ext"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    worker_id=worker_id,
                    on_date=ask.on_date,
                    until_time=ask.until_time,
                    reason=f"agreed to {ask.customer_name or 'a customer'}'s hours",
                    # Scoped: this yes belongs to this ask's customer. It lapses if
                    # they cancel or are told no, and no other caller may spend it.
                    ask_id=ask_id,
                    for_customer=ask.customer_name,
                )
            ]
        )
        worker = world.workers.get(worker_id)
        broadcaster.publish("world", {"event": "shift_extended"})
        return {
            "worker": worker.name if worker else worker_id,
            "until": clock(ask.until_time),
            "status": "their day now runs that late - read the call again to book it",
        }


@app.post("/api/asks/{ask_id}/close")
def close_ask(ask_id: str, outcome: str = Query(default="")) -> dict[str, str]:
    from krama.domain.events import CrewAskClosed

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.ask_close"):
        if ask_id not in svc.world().crew_asks:
            raise HTTPException(404, detail={"error": "NoSuchAsk", "detail": ask_id})
        now = datetime.now(svc.tz)
        svc.apply_events(
            [
                CrewAskClosed(
                    event_id=new_dispatch_id("ev"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    ask_id=ask_id,
                    outcome=outcome,
                )
            ]
        )
        broadcaster.publish("world", {"event": "crew_ask_closed"})
        return {"ask_id": ask_id, "status": "closed"}


def _float_overtime_offer(svc: DispatchService, job_id: str, now: datetime) -> str:
    """When a fresh booking keeps somebody past shift, put the hours up for grabs.

    The owner's policy: overtime is volunteered, not assigned. The solver has already
    proven the slot feasible and pencilled a fitter in, so the customer's promise is
    safe whatever the crew decides - the offer only settles WHO stays late. Everyone
    qualified, rostered and free that day gets the text; first yes wins; if nobody
    answers by the deadline, the pencilled-in fitter stands and is told so.

    Deterministic end to end - no agent decides anything here. The one judgement
    (how late is too late to keep asking) is a parameterless rule: five o'clock the
    evening before, or an hour from now if the booking itself arrived later than
    that, never after the job starts.
    """
    from krama.domain.events import OvertimeOffered

    world = svc.world()
    plan = svc.head()
    job = world.jobs.get(job_id)
    if plan is None or job is None:
        return ""

    placement = next(
        ((route, stop) for route in plan.routes for stop in route.stops if stop.job_id == job_id),
        None,
    )
    if placement is None:
        return ""
    route, stop = placement

    # Does this stop actually run past the assigned crew's shift?
    closes: list[datetime] = []
    for worker_id in route.worker_ids:
        worker = world.workers.get(worker_id)
        hours = worker.hours_for(route.date.weekday()) if worker else None
        if hours is not None:
            closes.append(datetime.combine(route.date, hours.end, tzinfo=svc.tz))
    if not closes:
        return ""
    shift_close = min(closes)
    departure = stop.departure.astimezone(svc.tz)
    overtime_minutes = int((departure - shift_close).total_seconds() // 60)
    if overtime_minutes <= 0:
        return ""

    opens = datetime.combine(route.date, time(0, 0), tzinfo=svc.tz)
    closes_day = opens + timedelta(days=1)
    allowance = timedelta(minutes=int(svc.business.labor.overtime_max_minutes.value))
    arrival_local = stop.arrival.astimezone(svc.tz)

    def can_reach(worker: Worker) -> bool:
        """On shift when the job starts, and overtime stretches far enough to finish.

        Without this check the first live run offered the evening to two fitters
        whose shifts end at three - people whose yes the solver could never honour,
        which would have stranded the job the moment they claimed it.
        """
        hours = worker.hours_for(route.date.weekday())
        if hours is None:
            return False
        starts = datetime.combine(route.date, hours.start, tzinfo=svc.tz)
        reach = datetime.combine(route.date, hours.end, tzinfo=svc.tz) + allowance
        return starts <= arrival_local and reach >= departure

    candidates = sorted(
        (
            w
            for w in world.workers.values()
            if w.overtime_eligible
            and w.is_certified_for(job.required_certifications)
            and can_reach(w)
            and world.is_worker_available(w.id, opens, closes_day)
        ),
        key=lambda w: w.id,
    )
    if not candidates:
        return ""
    fallback = next((w for w in route.worker_ids if w in {c.id for c in candidates}), None)
    if fallback is None:
        fallback = candidates[0].id

    start_local = stop.arrival.astimezone(svc.tz)
    evening_before = datetime.combine(route.date - timedelta(days=1), time(17, 0), tzinfo=svc.tz)
    deadline = max(evening_before, now + timedelta(hours=1))
    deadline = min(deadline, start_local)

    svc.apply_events(
        [
            OvertimeOffered(
                event_id=new_dispatch_id("ot"),
                occurred_at=now,
                recorded_at=now,
                dispatch_id="web",
                job_id=job_id,
                on_date=route.date,
                offered_to=tuple(c.id for c in candidates),
                fallback=fallback,
                overtime_minutes=overtime_minutes,
                claim_deadline=deadline,
            )
        ]
    )

    rate = float(svc.business.labor.loaded_rate_per_minute.value) * 60
    premium = float(svc.business.labor.overtime_multiplier.value)
    pay = overtime_minutes / 60 * rate * premium
    body = (
        f"Krama: {route.date:%a %d %b} {clock(start_local)} "
        f"{job.service_type.value.replace('_', ' ')} runs ~{overtime_minutes} min past "
        f"shift (about ${pay:,.0f} at time and a half). First YES takes it - reply by "
        f"{clock(deadline)}. Otherwise it stays with the schedule."
    )
    sender = notifier()
    reached = sum(1 for c in candidates if sender.send(c.phone, body).accepted)
    names = ", ".join(c.name for c in candidates)
    return (
        f"runs {overtime_minutes} min past shift - offered to {names} "
        f"({reached} texted), first yes takes it, deadline {clock(deadline)}"
    )


@app.post("/api/overtime/{job_id}/claim")
def claim_overtime(job_id: str, worker_id: str = Query(...)) -> dict[str, str]:
    """A fitter said yes to the extra hours - usually relayed by the dispatcher.

    First reply wins: the fold ignores every later claim, so two taps in two browser
    tabs cannot double-book the evening. The claim pins the fitter in the solver and
    the week is re-planned immediately; the customer's window cannot move - it is a
    confirmed promise, and the planner is not allowed to break those silently.
    """
    from krama.domain.events import OvertimeClaimed

    svc = service()
    with dispatch(new_dispatch_id("web")) as dispatch_id, span("api.claim_ot", job=job_id):
        world = svc.world()
        offer = world.overtime_offers.get(job_id)
        if offer is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "NoOffer", "detail": f"no overtime offer on {job_id}"},
            )
        if offer.claimed_by is not None:
            claimant = world.workers.get(offer.claimed_by)
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "AlreadyClaimed",
                    "detail": f"{claimant.name if claimant else offer.claimed_by} "
                    "already took these hours",
                },
            )
        if worker_id not in offer.offered_to:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "NotOffered",
                    "detail": "these hours were not offered to that fitter",
                    "remedy": "only qualified, rostered, overtime-eligible crew may claim",
                },
            )

        now = datetime.now(svc.tz)
        svc.apply_events(
            [
                OvertimeClaimed(
                    event_id=new_dispatch_id("otc"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id=dispatch_id,
                    job_id=job_id,
                    worker_id=worker_id,
                )
            ]
        )

        # The claim changes who drives, never when the customer is seen. Re-plan and
        # commit; the confirmed window binds the solver, so if the claim cannot be
        # honoured without touching the promise the pin dissolves rather than the
        # promise moving.
        note = ""
        try:
            plan = svc.plan_week(start=_default_start(svc))
            head = svc.head()
            committed = svc.commit(plan.plan, expected_parent=head.id if head else None)
            broadcaster.publish("plan", {"plan_id": committed.id})
            crewed = next(
                (r.worker_ids for r in committed.routes for st in r.stops if st.job_id == job_id),
                (),
            )
            note = "on the route" if worker_id in crewed else "recorded; not on the route yet"
        except (ServiceError, PlanConflict):
            note = "claim recorded; re-plan it when ready"

        worker = world.workers.get(worker_id)
        broadcaster.publish("world", {"event": "overtime_claimed"})
        return {
            "job_id": job_id,
            "worker": worker.name if worker else worker_id,
            "status": note,
        }


@app.get("/api/jobs/{job_id}/crew-options")
def crew_options(job_id: str) -> dict[str, list[dict[str, object]]]:
    """Who else could take this job, and what each swap would really cost.

    One trial solve per qualified fitter, each with the candidate pinned and every
    promise still binding - so "Dan +$6.40" is a measured number, not a guess, and
    a candidate whose pin cannot be honoured without breaking a promise comes back
    infeasible with that said in words. Expensive by design (a solve per name);
    this is behind a click on one job, not on every render.
    """
    from krama.scheduler.costing import cost_plan

    svc = service()
    with dispatch(new_dispatch_id("web")), span("api.crew_options", job=job_id):
        world = svc.world()
        job = world.jobs.get(job_id)
        if job is None or not job.is_active:
            raise HTTPException(404, detail={"error": "NoSuchJob", "detail": job_id})
        head = svc.head()
        if head is None:
            raise HTTPException(409, detail={"error": "NoPlan", "detail": "nothing committed"})
        current = next(
            (tuple(r.worker_ids) for r in head.routes for st in r.stops if st.job_id == job_id),
            (),
        )
        baseline_cost, _ = cost_plan(head, world, svc.business, svc.tz, head.unserved)

        options: list[dict[str, object]] = []
        for worker in sorted(world.workers.values(), key=lambda w: w.id):
            if not worker.is_certified_for(job.required_certifications):
                options.append(
                    {
                        "worker_id": worker.id,
                        "name": worker.name,
                        "feasible": False,
                        "current": worker.id in current,
                        "note": "not qualified for this work",
                    }
                )
                continue
            if worker.id in current and len(current) == job.crew_size:
                options.append(
                    {
                        "worker_id": worker.id,
                        "name": worker.name,
                        "feasible": True,
                        "current": True,
                        "cost_delta": 0.0,
                        "note": "on it now",
                    }
                )
                continue
            trial = svc.plan_week(start=_default_start(svc), pinned_workers={job_id: (worker.id,)})
            if trial.violations or not any(
                worker.id in r.worker_ids
                for r in trial.plan.routes
                for st in r.stops
                if st.job_id == job_id
            ):
                options.append(
                    {
                        "worker_id": worker.id,
                        "name": worker.name,
                        "feasible": False,
                        "current": False,
                        "note": "cannot take it without breaking a promise",
                    }
                )
                continue
            delta = trial.cost.total - baseline_cost.total
            options.append(
                {
                    "worker_id": worker.id,
                    "name": worker.name,
                    "feasible": True,
                    "current": False,
                    "cost_delta": round(delta, 2),
                    "note": "",
                }
            )
        return {"options": options}


@app.post("/api/jobs/{job_id}/crew")
def set_crew(job_id: str, worker_id: str = Query(...)) -> dict[str, str]:
    """Put a named fitter on this job - the dispatcher's call, priced and checked.

    The swap re-plans the week with the fitter pinned and every promise binding;
    if honouring the pin would move anybody's window, the whole thing refuses
    rather than quietly costing a customer their morning.
    """
    svc = service()
    with dispatch(new_dispatch_id("web")), span("api.set_crew", job=job_id):
        world = svc.world()
        job = world.jobs.get(job_id)
        if job is None or not job.is_active:
            raise HTTPException(404, detail={"error": "NoSuchJob", "detail": job_id})
        worker = world.workers.get(worker_id)
        if worker is None:
            raise HTTPException(404, detail={"error": "NoSuchWorker", "detail": worker_id})
        if not worker.is_certified_for(job.required_certifications):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "NotQualified",
                    "detail": f"{worker.name} is not qualified for this work",
                },
            )
        trial = svc.plan_week(start=_default_start(svc), pinned_workers={job_id: (worker_id,)})
        placed = any(
            worker_id in r.worker_ids
            for r in trial.plan.routes
            for st in r.stops
            if st.job_id == job_id
        )
        if trial.violations or not placed:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "WouldBreakAPromise",
                    "detail": (
                        f"putting {worker.name} on this job cannot be done without "
                        "moving a promised window"
                    ),
                    "remedy": "pick another fitter, or edit the booking itself",
                },
            )
        head = svc.head()
        committed = svc.commit(trial.plan, expected_parent=head.id if head else None)
        broadcaster.publish("plan", {"plan_id": committed.id})
        return {"job_id": job_id, "crew": worker.name, "status": "re-planned around the swap"}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, reason: str = Query(default="customer cancelled")) -> dict[str, str]:
    """Take a booking back out of the diary, fast.

    Customers change their minds on the same call that booked them, and until now the
    only route out was the CLI. The cancellation is an event like everything else - the
    log keeps that the booking happened and then did not, which is the difference
    between undoing and pretending.

    The re-plan runs immediately, because a cancelled job still sitting on the board is
    a slot a dispatcher will not offer to the next caller.
    """
    from krama.domain.enums import CommitmentState
    from krama.domain.events import JobCancelled

    svc = service()
    with dispatch(new_dispatch_id("web")), span("api.cancel", job_id=job_id):
        world = svc.world()
        job = world.jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NoSuchJob",
                    "detail": f"{job_id} is not in the diary",
                    "remedy": "it may already be cancelled; reload the board",
                },
            )
        if job.commitment_state is CommitmentState.CANCELLED:
            # Cancelling twice used to report "freed" twice - a second event in the
            # log, a second replan, and a message claiming an action that changed
            # nothing. Saying so is cheaper than pretending.
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "AlreadyCancelled",
                    "detail": f"{job.customer_name or job_id} was already cancelled",
                    "remedy": "nothing to do - reload the board",
                },
            )
        now = datetime.now(svc.tz)
        svc.apply_events(
            [
                JobCancelled(
                    event_id=new_dispatch_id("cancel"),
                    occurred_at=now,
                    recorded_at=now,
                    dispatch_id="web",
                    job_id=job_id,
                    reason=reason,
                )
            ]
        )
        freed = "freed"
        try:
            plan = svc.plan_week(start=_default_start(svc))
            head = svc.head()
            svc.commit(plan.plan, expected_parent=head.id if head else None)
        except (ServiceError, PlanConflict):
            freed = "cancelled, but the re-plan needs attention"
        broadcaster.publish("plan", {"cancelled": job_id})
        return {"job_id": job_id, "customer": job.customer_name, "status": freed}


@app.post("/api/comms", response_model=list[MessageView])
def draft_messages(strategy: str = Query(...)) -> list[MessageView]:
    """Draft customer messages for one repair candidate, with grounding already checked."""
    from krama.agents.comms import draft_customer_messages
    from krama.agents.llm.factory import build_llm

    svc = service()
    with dispatch(new_dispatch_id("web")), span("api.comms", strategy=strategy):
        try:
            options, _ = svc.repair()
        except ServiceError as exc:
            raise _fail(exc, 409) from exc

        candidate = next((c for c in options.candidates if c.strategy.name == strategy), None)
        if candidate is None:
            raise HTTPException(404, detail={"error": "NoSuchStrategy", "detail": strategy})

        result = draft_customer_messages(
            build_llm(),
            candidate.diff,
            tz=svc.tz,
            reason=f"repair: {candidate.strategy.description}",
        )
        by_job: dict[str, list[str]] = {}
        for issue in result.issues:
            by_job.setdefault(issue.job_id, []).append(f"{issue.phrase}: {issue.detail}")

        return [
            MessageView(
                job_id=d.job_id,
                channel=d.channel,
                body=d.body,
                grounded=not by_job.get(d.job_id),
                issues=by_job.get(d.job_id, []),
            )
            for d in result.drafts
        ]


# --------------------------------------------------------------------------- diffs


@app.get("/api/diff")
def get_diff(before: str = Query(...), after: str = Query(...)) -> dict[str, Any]:
    svc = service()
    try:
        diff = svc.diff(before, after)
    except ServiceError as exc:
        raise _fail(exc, 404) from exc
    return {
        "summary": diff.summary(),
        "blast_radius": diff.blast_radius.value,
        "changes": [c.model_dump() for c in views.change_views(diff)],
    }


@app.get("/api/history")
def get_history() -> list[dict[str, Any]]:
    svc = service()
    return [
        {
            "plan_id": plan.id,
            "content_hash": plan.content_hash,
            "created_at": plan.created_at.astimezone(svc.tz).isoformat(),
            "label": plan.label,
            "jobs": sum(len(r.stops) for r in plan.routes),
        }
        for plan in svc.workspace.plans.ancestry()
    ]


# -------------------------------------------------------------------------- stream


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    """Server-sent events, so a board reflects a change made from the CLI or an agent."""
    if stream_mode() != "sse":
        # Refused rather than quietly served. An old tab that kept its connection would
        # go on billing for it, and a bill is a bad way to find out.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "StreamingDisabled",
                "detail": "this deployment is billed per second of open connection",
                "remedy": "the board polls instead; reload it to pick up the right mode",
            },
        )
    return StreamingResponse(
        broadcaster.stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------------- helpers


def next_working_day(world: WorldState, on_or_after: date) -> date:
    """The first day anyone is rostered on, starting from the given date.

    Asked of the roster rather than assumed to be Monday to Friday, because the roster
    is where working days are actually written down and a business that starts opening
    Saturdays should not need a code change to be planned for.
    """
    rostered = {
        hours.weekday for worker in world.workers.values() for hours in worker.working_hours
    }
    if not rostered:
        return on_or_after
    for offset in range(14):
        candidate = on_or_after + timedelta(days=offset)
        if candidate.weekday() in rostered:
            return candidate
    return on_or_after


def next_working_day_from_today() -> date:
    """Fallback for a workspace that does not exist yet: assume a Monday-to-Friday
    week, which is what the roster would have said anyway."""
    today = datetime.now(UTC).date()
    for offset in range(7):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() < 5:
            return candidate
    return today


def _default_start(svc: DispatchService) -> date:
    """Where the horizon begins.

    A committed plan keeps its own start, so re-reading the board does not silently
    slide the week out from under a dispatcher. Otherwise it is the next day anybody
    is working - which on a Saturday afternoon means Monday, and which is the whole
    point of a rolling horizon rather than a fixed one.

    This used to return a date written into the fixture, so on any day other than the
    week of 21 September 2026 the board showed a week that had already happened.
    """
    today_start = next_working_day(svc.world(), datetime.now(svc.tz).date())
    head = svc.head()
    if head is not None:
        # The committed start holds WITHIN its own week, so re-reading the board
        # does not slide the week under a dispatcher - but it must never lag the
        # calendar. Pinned forever, the horizon went stale: on Thursday the board
        # was still planning Monday-to-Friday of LAST week, so a booking for next
        # Monday - offered and priced by the quote, which walks real future days -
        # fell outside the commit's window and vanished from every screen.
        return max(head.horizon_start, today_start)
    return today_start


def _clock(
    svc: DispatchService, value: Any, fallback: time | None, *, default_now: bool = False
) -> datetime | None:
    """A stated HH:MM, or - for the moment an event happened - when it was reported.

    Not 08:00. A van reported off the road at two in the afternoon was recorded as
    unavailable since breakfast, retroactively invalidating the work it had already
    done that morning.
    """
    if not value:
        if default_now:
            # Wall time. The fold clock is the LAST EVENT'S time, and stamping "now"
            # with it is the same bug that let a cancellation sort before the booking
            # it undid. "Dan is sick now" means now.
            return datetime.now(svc.tz)
        if fallback is None:
            return None
        return datetime.combine(_default_start(svc), fallback, tzinfo=svc.tz)
    # Two spellings. "17:30" is a time on the default day - what the CLI and the event
    # forms have always sent. A full ISO datetime is for a fact with a date in it:
    # "Marcus is out until Thursday" is not expressible as a clock reading, and forcing
    # it through one is how a one-day absence became a one-week absence on the board.
    text = str(value)
    try:
        parsed = datetime.strptime(text, "%H:%M").time()
    except ValueError:
        stamped = datetime.fromisoformat(text)
        return stamped if stamped.tzinfo else stamped.replace(tzinfo=svc.tz)
    return datetime.combine(_default_start(svc), parsed, tzinfo=svc.tz)


# --------------------------------------------------------------------- static site


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        """Serve the built board, letting the client router own every other path."""
        candidate = WEB_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")


def main() -> None:
    import uvicorn

    configure(service="krama-api")
    # Loopback by default: running `krama-api` on a laptop should not put an
    # unauthenticated dispatch board on the coffee-shop wifi. A container has to opt in
    # by setting the host, which its own Dockerfile does - and must, or nothing outside
    # the container can reach it, including a load balancer's health check.
    uvicorn.run(
        app,
        host=os.environ.get("KRAMA_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("KRAMA_API_PORT", "8000")),
    )


with suppress(ImportError):  # pragma: no cover - only for `python -m`
    pass
