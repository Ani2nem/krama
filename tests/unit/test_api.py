"""HTTP surface.

The board is a thin client over these endpoints, so anything asserted here is
something a dispatcher would otherwise have to notice on screen. The regression at
the bottom is the one that matters most: it was found by clicking, not by reasoning.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from krama.api.models import TriageView
from krama.fixtures.sample_business import WEEK_START, seed_events
from krama.persistence.log import Workspace


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    workspace = Workspace(tmp_path / "ws")
    workspace.seed(seed_events())
    monkeypatch.setenv("KRAMA_WORKSPACE", str(tmp_path / "ws"))
    # Real road distances from the committed snapshot: offline and identical to what a
    # deployment would compute.
    monkeypatch.setenv("KRAMA_TRAVEL", "frozen")

    from krama.api.main import app

    # The fixture config ships a sample owner PIN, so the suite runs as the owner
    # by default - tests that probe the dispatcher view override the pin and the
    # header themselves.
    return TestClient(app, headers={"X-Owner-Key": "1234"})


#: The fixture's jobs live in a fixed week, and the horizon now starts from whatever
#: today is - correctly, since a board showing last week is the bug this fixed. Tests
#: that want the fixture's work have to say which week they mean.
COMMIT = f"/api/plan/commit?start_date={WEEK_START.isoformat()}"


def scheduled(plan: dict[str, Any]) -> set[str]:
    return {stop["job_id"] for route in plan["routes"] for stop in route["stops"]}


# ---------------------------------------------------------------------- reading


def test_world_reports_the_roster(client: TestClient):
    world = client.get("/api/world").json()
    assert len(world["workers"]) == 6
    assert len(world["vans"]) == 4
    assert len(world["jobs"]) == 10


def test_the_board_is_told_the_costs_are_uncalibrated(client: TestClient):
    """Every figure on screen rests on numbers nobody has validated. That belongs in
    front of the reader, not in a config file they will never open."""
    assert "estimate" in client.get("/api/world").json()["calibration_warning"]


def test_no_plan_yet_is_null_not_an_error(client: TestClient):
    response = client.get("/api/plan")
    assert response.status_code == 200
    assert response.json() is None


def test_params_carry_their_provenance(client: TestClient):
    params = client.get("/api/params").json()
    assert params
    assert all(p["source"] and p["note"] for p in params)


# --------------------------------------------------------------------- planning


def test_committing_returns_a_feasible_plan(client: TestClient):
    plan = client.post(COMMIT).json()
    assert plan["feasible"] and plan["violations"] == []
    assert len(scheduled(plan)) == 10


def test_the_board_gets_what_it_needs_to_draw_a_bar(client: TestClient):
    """Minutes from midnight rather than timestamps, so the client positions bars
    without parsing dates or guessing a timezone."""
    plan = client.post(COMMIT).json()
    stop = plan["routes"][0]["stops"][0]
    assert 0 <= stop["start_minute"] < stop["end_minute"] <= 24 * 60
    assert stop["lat"] and stop["lon"]
    assert stop["commitment_state"]


def test_routes_carry_utilisation_and_slack(client: TestClient):
    """The two columns that make a technically valid but obviously wrong plan look
    wrong. Neither is something an invariant check can judge."""
    plan = client.post(COMMIT).json()
    route = plan["routes"][0]
    assert 0.0 <= route["utilization"] <= 1.0
    assert route["idle_minutes"] >= 0


def test_unserved_separates_failures_from_routine(client: TestClient):
    plan = client.post(COMMIT).json()
    for item in plan["unserved"]:
        assert isinstance(item["is_failure"], bool)
        assert item["detail"]


# ----------------------------------------------------------------------- events


def test_recording_an_event_changes_the_world(client: TestClient):
    # No "at": the outage starts now. A bare HH:MM lands on the NEXT WORKING day,
    # which made this test a calendar bomb - green on weekday mornings, red from a
    # Saturday, because a Monday outage leaves the van correctly available today.
    client.post("/api/events", json={"kind": "van-unavailable", "target": "van-1"})
    vans = {v["id"]: v for v in client.get("/api/world").json()["vans"]}
    assert vans["van-1"]["available"] is False


def test_an_incomplete_event_is_refused_with_a_remedy(client: TestClient):
    response = client.post("/api/events", json={"kind": "job-overran", "target": "j-403"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "--minutes" in detail["detail"]
    assert detail["remedy"]


def test_an_unknown_event_kind_is_refused(client: TestClient):
    assert client.post("/api/events", json={"kind": "abduction"}).status_code == 400


# ----------------------------------------------------------------------- repair


def test_repairing_without_a_plan_says_what_to_do(client: TestClient):
    response = client.post("/api/repair")
    assert response.status_code == 409
    assert "commit" in response.json()["detail"]["remedy"]


def test_repair_offers_priced_options_with_autonomy_verdicts(client: TestClient):
    client.post(COMMIT)
    client.post("/api/events", json={"kind": "van-unavailable", "target": "van-1", "at": "10:40"})

    repair = client.post("/api/repair").json()
    assert repair["candidates"]
    assert repair["recommended"]
    for candidate in repair["candidates"]:
        assert candidate["autonomy"] in {"auto_apply", "escalate"}
        assert candidate["blast_radius"] in {"internal", "crew_only", "customer_visible"}


def test_applying_an_unknown_strategy_is_a_404(client: TestClient):
    client.post(COMMIT)
    assert client.post("/api/repair/apply", params={"strategy": "wing_it"}).status_code == 404


def test_applying_a_repair_moves_the_head(client: TestClient):
    first = client.post(COMMIT).json()
    client.post("/api/events", json={"kind": "van-unavailable", "target": "van-1", "at": "10:40"})
    repair = client.post("/api/repair").json()
    applied = client.post("/api/repair/apply", params={"strategy": repair["recommended"]}).json()

    assert applied["plan_id"] != first["plan_id"]
    newest = next(h["plan_id"] for h in client.get("/api/history").json())
    assert newest == applied["plan_id"]


def test_a_customer_visible_repair_cannot_be_applied_without_approval(
    client: TestClient, monkeypatch
):
    """`force` is what a dispatcher's approval looks like over HTTP. The endpoint
    cannot be talked past, only overridden by a person who saw the diff."""
    from krama.domain import autonomy as autonomy_module

    client.post(COMMIT)
    client.post("/api/events", json={"kind": "van-unavailable", "target": "van-1", "at": "10:40"})
    repair = client.post("/api/repair").json()

    forced = autonomy_module.AutonomyDecision(
        autonomy_module.Decision.ESCALATE, ("a customer would need telling",)
    )
    monkeypatch.setattr("krama.service.decide", lambda *a, **k: forced)

    response = client.post("/api/repair/apply", params={"strategy": repair["recommended"]})
    assert response.status_code == 412
    assert "dispatcher" in response.json()["detail"]["remedy"]

    ok = client.post("/api/repair/apply", params={"strategy": repair["recommended"], "force": True})
    assert ok.status_code == 200


# -------------------------------------------------------------------- regression


def test_replanning_does_not_erase_work_already_in_flight(client: TestClient):
    """Found by clicking, not by reasoning.

    A job that has been dispatched is no longer "schedulable", so planning the week
    from scratch dropped it silently - a dispatcher pressing Re-plan at eleven would
    have erased the crew that left at six. Repair already carried in-flight work
    forward; a plain re-plan did not.
    """
    first = client.post(COMMIT).json()
    assert "j-401" in scheduled(first)

    client.post("/api/events", json={"kind": "job-dispatched", "target": "j-401", "at": "06:05"})

    again = client.post(COMMIT).json()
    assert "j-401" in scheduled(again), "dispatched work vanished from the re-planned board"
    assert len(scheduled(again)) == 10
    assert again["feasible"]


def test_every_commitment_state_can_reach_the_board(client: TestClient):
    """Each state is a different colour on the Gantt, so a state that never arrives is
    a colour nobody has ever seen."""
    client.post(COMMIT)
    # Full ISO days, not bare HH:MM: a bare time lands on the CURRENT week's
    # rolling anchor (right for a dispatcher typing it live), but this test is
    # dressing jobs in the fixture's own week.
    day = WEEK_START.isoformat()
    client.post(
        "/api/events", json={"kind": "job-dispatched", "target": "j-401", "at": f"{day}T06:05"}
    )
    client.post(
        "/api/events",
        json={
            "kind": "job-confirmed",
            "target": "j-402",
            "window_start": f"{day}T09:00",
            "window_end": f"{day}T15:00",
            "commitment_cost": 250,
        },
    )
    plan = client.post(COMMIT).json()
    states = {stop["commitment_state"] for r in plan["routes"] for stop in r["stops"]}
    assert {"provisional", "confirmed", "dispatched"} <= states


# ---------------------------------------------------------------- health probes


def test_liveness_does_no_work(client: TestClient):
    """A probe that touched the solver would restart healthy tasks mid-solve."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_each_thing_the_image_could_have_failed_to_ship(client: TestClient):
    """The three checks stand for three things the image copies selectively."""
    body = client.get("/api/ready").json()
    assert set(body["checks"]) == {"params", "travel", "board", "auth"}
    assert body["checks"]["params"] == "42 parameters"
    assert body["checks"]["travel"].startswith("frozen:")


def test_readiness_is_ready_when_the_board_is_there(client: TestClient, monkeypatch, tmp_path):
    """Asserted against a directory this test creates.

    An earlier version asserted `ready` against whatever happened to be on disk, which
    passed locally - where the board had been built - and failed in CI, where the job
    that runs the tests has no reason to build it. The check was right and the test was
    reading the developer's working tree.
    """
    from krama.api import main

    monkeypatch.setattr(main, "WEB_DIST", tmp_path)
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readiness_is_degraded_without_the_built_board(client: TestClient, monkeypatch, tmp_path):
    """An image that shipped no board serves a blank page and answers 200 on every API
    route. The load balancer should not call that a healthy target."""
    from krama.api import main

    monkeypatch.setattr(main, "WEB_DIST", tmp_path / "never-built")
    response = client.get("/api/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["board"].startswith("FAILED")


def test_readiness_fails_loudly_when_travel_cannot_answer(client: TestClient, monkeypatch):
    """The failure worth catching: the process is up and cannot do the job.

    A port check calls this container healthy. It is not - a task wired to a routing
    backend that is not there would serve errors for every solve while the load
    balancer kept sending it traffic.
    """
    monkeypatch.setenv("KRAMA_TRAVEL", "osrm")
    monkeypatch.setenv("KRAMA_OSRM_URL", "http://127.0.0.1:1")

    response = client.get("/api/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["travel"].startswith("FAILED")
    assert body["checks"]["params"] == "42 parameters", "unrelated checks still report"


def test_the_readiness_probe_asks_for_a_leg_the_snapshot_holds():
    """Regression: the probe used hand-typed coordinates and a wall-clock timestamp.

    Both were wrong in a way that only showed up sometimes. The addresses are geocoded,
    so a literal copied from the source sat in a different geohash cell than anything
    frozen; and the snapshot is keyed by day type, so probing at "now" asked for a
    weekend leg every Saturday, and a healthy container reported degraded.

    Asserted against the snapshot directly rather than through the endpoint, because
    the endpoint answers correctly on six days out of seven either way.
    """
    from krama.api.main import _PROBE_AT
    from krama.config import BusinessParams
    from krama.fixtures.sample_business import DEPOT, PROBE_STOP
    from krama.scheduler.travel.factory import TravelMode, build_travel

    assert _PROBE_AT.weekday() < 5, "the business does not run weekends"

    frozen = build_travel(BusinessParams.load(), TravelMode.FROZEN)
    # Raises CacheMiss if this pair and bucket were never frozen, which is the bug.
    leg = frozen.leg(DEPOT, PROBE_STOP, _PROBE_AT)
    assert leg.minutes > 0


def test_the_osrm_url_is_configurable(monkeypatch):
    """Deployed, the routing backend is another host. localhost is a laptop default."""
    from krama.config import BusinessParams
    from krama.scheduler.travel.factory import TravelMode, build_travel

    monkeypatch.setenv("KRAMA_OSRM_URL", "http://osrm.internal:5000")
    provider = build_travel(BusinessParams.load(), TravelMode.OSRM)
    assert "osrm.internal" in repr(provider.__dict__), "the env var was not honoured"


def test_the_board_is_told_how_to_watch_for_changes(client: TestClient, monkeypatch):
    """The client cannot work this out for itself.

    A stream works perfectly well on Lambda; it is just billed for every second it
    stays open, so there is no failure to detect and fall back from. The server has to
    say, and the default is the one that is right locally.
    """
    assert client.get("/api/health").json()["stream"] == "sse"

    monkeypatch.setenv("KRAMA_STREAM", "poll")
    assert client.get("/api/health").json()["stream"] == "poll"


def test_streaming_is_refused_when_it_is_billed_by_the_second(client: TestClient, monkeypatch):
    """An old tab that kept its connection would go on costing money, and a bill is a
    bad way to find out. Refused with a remedy rather than quietly served."""
    monkeypatch.setenv("KRAMA_STREAM", "poll")
    response = client.get("/api/stream")
    assert response.status_code == 409
    assert "poll" in response.json()["detail"]["remedy"]


def test_an_address_on_the_wrong_coast_is_refused_not_crashed(client: TestClient):
    """The bug a dispatcher actually hit, reported as a 500.

    "main st" typed during a call near Fort Worth resolves nationwide - a global
    geocoder ranks by prominence and has no idea where the vans are. The solver then
    asked the travel snapshot for a leg to New York, 2,403 miles away, and the cache
    miss surfaced as Internal Server Error.
    """
    from krama.domain.models import Location
    from krama.geocoding import OutsideServiceArea, for_service_area

    geocoder = for_service_area()
    manhattan = Location(lat=40.7589, lon=-73.9668, address="2nd Ave, Manhattan")
    with pytest.raises(OutsideServiceArea) as raised:
        geocoder._check_in_area("2nd ave", manhattan)
    # Far enough that no service radius could plausibly reach it. The exact figure
    # is a property of where the depot happens to be, so it is not asserted.
    assert raised.value.miles > 1000
    assert "service area" in str(raised.value)


def test_the_search_is_bounded_to_the_service_area():
    """A bias that merely prefers nearby results still returns the far one when nothing
    closer matches, which is exactly the failing case. It has to be a hard bound."""
    from krama.geocoding import for_service_area

    box = for_service_area()._viewbox()
    assert box is not None
    west, north, east, south = (float(v) for v in box.split(","))
    assert west < -97.34 < east and south < 33.00 < north, "the depot is inside its own box"
    # A degree of longitude covers less ground than a degree of latitude this far
    # north - about 47 miles against 69 - so the box has to be wider than it is tall
    # by roughly 1/cos(47.6 degrees). Treating them as equal clips real addresses.
    expected = 1 / math.cos(math.radians(33.00))
    assert (east - west) / (north - south) == pytest.approx(expected, rel=0.02)


def test_a_new_address_explains_itself_rather_than_failing(client: TestClient):
    """The frozen snapshot refuses to invent a leg, which is right for tests and wrong
    mid-call. The answer is a remedy, not a stack trace."""
    from krama.api.main import _travel_cache_miss
    from krama.scheduler.travel.cache import CacheMiss

    # The handler ignores the request; typing it honestly beats a None the checker
    # has to be told to overlook.
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    response = _travel_cache_miss(request, CacheMiss("no cached leg for x|y|weekday|early"))
    assert response.status_code == 409
    assert b"warm" in response.body, "it must say which mode fixes it"


# ------------------------------------------------------------------ one box


def test_a_note_is_routed_to_the_right_agent(client: TestClient, monkeypatch):
    """The board used to ask the dispatcher which box to type into, and they got it
    wrong on the first try. The classifier answers instead."""
    from krama.agents.router import NoteRouting

    monkeypatch.setattr(
        "krama.agents.router.route_note",
        lambda *_a, **_k: type(
            "E", (), {"value": NoteRouting(kind="disruption", why="a van is off the road")}
        )(),
    )
    monkeypatch.setattr(
        "krama.api.main.run_triage",
        lambda request: TriageView(state="ok", summary=request.text),
    )

    body = client.post("/api/note", json={"text": "van 3 won't start"}).json()
    assert body["kind"] == "disruption"
    assert body["why"] == "a van is off the road"
    assert body["booking"] is None, "only the agent that read it should answer"


def test_the_dispatcher_can_overrule_the_classifier(client: TestClient, monkeypatch):
    """What makes routing by model safe here.

    A misroute costs one click rather than a wrong job on the schedule, and the
    classifier is skipped entirely when the answer is already known - so an override
    cannot be silently re-overridden.
    """
    called = False

    def should_not_run(*_a, **_k):
        nonlocal called
        called = True
        raise AssertionError("the classifier ran despite an explicit kind")

    monkeypatch.setattr("krama.agents.router.route_note", should_not_run)
    monkeypatch.setattr(
        "krama.api.main.run_triage", lambda request: TriageView(state="ok", summary="")
    )

    body = client.post("/api/note?kind=disruption", json={"text": "anything"}).json()
    assert body["kind"] == "disruption"
    assert body["why"] == "", "nothing was classified, so there is no reason to show"
    assert called is False


# ------------------------------------------------------------------------ auth


def test_without_a_key_configured_everything_is_open(client: TestClient):
    """What makes `make dev` work with no setup. Safe only because the deployment
    refuses to be public without one."""
    assert client.get("/api/world").status_code == 200


def test_with_a_key_configured_an_unauthenticated_call_is_refused(client: TestClient, monkeypatch):
    monkeypatch.setenv("KRAMA_API_KEY", "s3cret")
    response = client.get("/api/world")
    assert response.status_code == 401
    assert "X-API-Key" in response.json()["remedy"]


@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": "s3cret"},
        {"Authorization": "Bearer s3cret"},
        {"authorization": "bearer s3cret"},
    ],
)
def test_either_header_carries_the_key(client: TestClient, monkeypatch, headers: dict[str, str]):
    monkeypatch.setenv("KRAMA_API_KEY", "s3cret")
    assert client.get("/api/world", headers=headers).status_code == 200


def test_a_wrong_key_is_refused(client: TestClient, monkeypatch):
    monkeypatch.setenv("KRAMA_API_KEY", "s3cret")
    assert client.get("/api/world", headers={"X-API-Key": "s3cre"}).status_code == 401
    assert client.get("/api/world", headers={"X-API-Key": "s3cretx"}).status_code == 401


def test_health_and_readiness_stay_open(client: TestClient, monkeypatch):
    """A load balancer and a deploy smoke test decide whether this container works,
    and neither can hold a secret."""
    monkeypatch.setenv("KRAMA_API_KEY", "s3cret")
    assert client.get("/api/health").status_code == 200
    # Not asserted as 200: readiness answers 503 wherever the board has not been
    # built, which is every CI run. The property here is that the key does not stand
    # in front of it, so what matters is that it is not a 401.
    assert client.get("/api/ready").status_code != 401


def test_readiness_says_whether_anything_is_guarding_the_door(client: TestClient, monkeypatch):
    """Running open is a legitimate local choice. Running open without knowing is not."""
    assert "open" in client.get("/api/ready").json()["checks"]["auth"]
    monkeypatch.setenv("KRAMA_API_KEY", "s3cret")
    assert client.get("/api/ready").json()["checks"]["auth"] == "api key required"


# --------------------------------------------------------------- usable board


def test_a_fresh_workspace_has_no_work_in_it():
    """A business has staff and vans on day one and no jobs until somebody rings.
    Pre-booked work nobody booked is confusing on a board somebody is trying to use."""
    from krama.domain.state import fold
    from krama.fixtures.sample_business import seed_events

    empty = fold(seed_events(with_jobs=False))
    assert empty.workers and empty.vans
    assert empty.jobs == {}

    assert fold(seed_events()).jobs, "the fixture week is still there for scenarios"


def test_the_horizon_starts_from_the_next_working_day():
    """It used to return a date written into the fixture, so on any day but the week
    of 21 September 2026 the board showed a week that had already happened."""
    from datetime import date

    from krama.api.main import next_working_day
    from krama.domain.state import fold
    from krama.fixtures.sample_business import seed_events

    world = fold(seed_events(with_jobs=False))
    saturday = date(2026, 9, 26)
    assert next_working_day(world, saturday) == date(2026, 9, 28)
    assert next_working_day(world, date(2026, 9, 28)) == date(2026, 9, 28)


def test_working_days_come_from_the_roster_not_from_an_assumption():
    """A business that starts opening Saturdays should not need a code change."""
    from datetime import date, time

    from krama.api.main import next_working_day
    from krama.domain.models import DayHours
    from krama.domain.state import fold
    from krama.fixtures.sample_business import seed_events

    world = fold(seed_events(with_jobs=False))
    only = next(iter(world.workers.values()))
    world.workers = {
        only.id: only.model_copy(
            update={"working_hours": (DayHours(weekday=5, start=time(9), end=time(13)),)}
        )
    }
    assert next_working_day(world, date(2026, 9, 26)).weekday() == 5


def test_a_quoted_window_opens_at_the_estimate_rather_than_straddling_it():
    """Centred, a 6:27 arrival was quoted as "5:27 to 7:27" - not a time anyone reads
    down a phone, and promising half an hour earlier than the crew could manage."""
    from datetime import datetime, timedelta

    from krama.domain.models import TimeWindow

    arrival = datetime.fromisoformat("2026-09-28T06:27:00-05:00")
    opens = arrival.replace(minute=arrival.minute // 15 * 15, second=0, microsecond=0)
    window = TimeWindow(start=opens, end=opens + timedelta(minutes=120))

    assert window.start.hour == 6 and window.start.minute == 15
    assert window.start <= arrival, "never promise earlier than the crew can arrive"


def test_times_a_person_reads_are_twelve_hour():
    from datetime import time

    from krama.formatting import clock, clock_range

    assert clock(time(8, 0)) == "8:00 AM"
    assert clock(time(17, 5)) == "5:05 PM"
    assert clock(time(0, 30)) == "12:30 AM"
    assert clock_range(time(8, 0), time(17, 0)) == "8:00 AM - 5:00 PM"
    # One meridiem when both sides agree; repeating it inside a range is noise.
    assert clock_range(time(9, 0), time(11, 30)) == "9:00 - 11:30 AM"


# ----------------------------------------------------------------- cancelling


def test_a_booking_can_be_taken_back_out(client: TestClient):
    """Customers change their minds on the same call that booked them. The event log
    keeps that the booking happened and then did not - undoing, not pretending."""
    client.post(COMMIT)
    world = client.get("/api/world").json()
    victim = world["jobs"][0]["id"]

    response = client.post(f"/api/jobs/{victim}/cancel").json()
    assert response["status"] == "freed"

    after = client.get("/api/world").json()
    assert victim not in {j["id"] for j in after["jobs"]}
    plan = client.get("/api/plan").json()
    assert victim not in {s["job_id"] for r in plan["routes"] for s in r["stops"]}, (
        "a cancelled job still on the board is a slot nobody will offer"
    )


def test_cancelling_a_job_that_is_not_there_says_so(client: TestClient):
    response = client.post("/api/jobs/j-nope/cancel")
    assert response.status_code == 404
    assert "reload" in response.json()["detail"]["remedy"]


def _business_today() -> date:
    """Today in the BUSINESS's timezone, never the machine's.

    CI runs in UTC; at 8 PM on a Chicago Saturday, UTC is already Sunday - and at
    7 PM on a Chicago Sunday, UTC is already Monday, which made "next Monday" jump
    a week past the planning horizon and every booking land unscheduled. The server
    thinks in America/Chicago; tests that date things must think there too.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/Chicago")).date()


def _next_monday() -> str:
    """The next WEEKDAY strictly after business-today - not literally Monday.

    "Next Monday" from a Monday is seven days out, which falls off the five-working-
    day rolling horizon and quietly unschedules every test booking - a bug that only
    fires one day a week, which is the worst kind. Any strictly-future weekday is
    always inside the horizon; the name stays for the dozens of call sites.
    """
    from datetime import timedelta

    today = _business_today()
    bump = 1 if today.weekday() < 4 else 7 - today.weekday()
    return (today + timedelta(days=bump)).isoformat()


def test_a_booking_cancelled_moments_later_stays_cancelled(client: TestClient, monkeypatch):
    """The order of same-second events was decided by random event ids.

    New events were stamped with world.as_of - the fold clock, which is the *last
    event's* time - so a booking and its cancellation carried identical timestamps a
    week in the past, and the sort tiebreak (the random event id) decided whether the
    confirmation folded after the cancellation and quietly resurrected it. Roughly a
    coin flip, live. Wall-clock stamps make the order the order it happened in.
    """
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    draft = {
        "customer_name": "Maria",
        "phone": "9132934243",
        "address": "somewhere real",
        "service_type": "residential_window_replacement",
        "duration_minutes": 120,
        "duration_confidence": 60,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    # A future Monday: the fixture week is behind the real clock, and the API now
    # (correctly) refuses to book the past.
    booked = client.post(
        "/api/book", json={"draft": draft, "date": _next_monday(), "arrival": "16:00"}
    ).json()
    assert booked["status"] in {"scheduled", "booked but not yet scheduled"}

    cancelled = client.post(f"/api/jobs/{booked['job_id']}/cancel").json()
    assert cancelled["status"] == "freed"

    world = client.get("/api/world").json()
    assert booked["job_id"] not in {j["id"] for j in world["jobs"]}, (
        "the confirmation must not fold after the cancellation and resurrect it"
    )


def test_the_transcript_survives_from_call_to_reschedule(client: TestClient, monkeypatch):
    """A reschedule starts from what the caller originally said, not from a
    dispatcher's memory of it. The words ride on the job's provenance."""
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    said = "Maria needs glass fixed at 4pm monday, took time off, 9132934243, 14400 Artisan Dr"
    draft = {
        "customer_name": "Maria",
        "phone": "9132934243",
        "address": "14400 Artisan Dr",
        "service_type": "residential_window_replacement",
        "duration_minutes": 120,
        "duration_confidence": 60,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 250,
        "lat": 32.99,
        "lon": -97.36,
    }
    booked = client.post(
        "/api/book",
        json={"draft": draft, "date": _next_monday(), "arrival": "16:00", "transcript": said},
    ).json()

    world = client.get("/api/world").json()
    stored = next(j for j in world["jobs"] if j["id"] == booked["job_id"])
    assert stored["transcript"] == said


def test_the_rota_shows_each_fitter_week_with_overtime_reach(client: TestClient):
    """ "Why does the machine keep choosing Dan" was unanswerable from the screen. The
    rota carries the answer: shift per day, and how late overtime may keep them."""
    world = client.get("/api/world").json()
    dan = next(w for w in world["workers"] if w["name"] == "Dan")
    ken = next(w for w in world["workers"] if w["name"] == "Ken")
    marcus = next(w for w in world["workers"] if w["name"] == "Marcus")

    assert len(dan["days"]) == 5, "a working week, not a single line"
    assert all(d["day"] and d["date"] for d in dan["days"])

    working = next(d for d in dan["days"] if d["shift"] != "off")
    assert "5:00 PM" in working["shift"]
    assert "7:00 PM" in working["reach"], "overtime reach explains who can take evenings"

    # Everyone is overtime-eligible now - the owner's call - so Ken reaches seven like
    # Dan, and the evening residential work stops being a one-man bottleneck.
    ken_day = next(d for d in ken["days"] if d["shift"] != "off")
    assert "7:00 PM" in ken_day["reach"]

    marcus_day = next(d for d in marcus["days"] if d["shift"] != "off")
    assert "5:00 PM" in marcus_day["reach"], "an early shift caps at five even on overtime"


def test_marking_a_fitter_out_shows_in_their_week(client: TestClient):
    client.post("/api/events", json={"kind": "worker-unavailable", "target": "w-dan"})
    world = client.get("/api/world").json()
    dan = next(w for w in world["workers"] if w["name"] == "Dan")
    assert dan["available"] is False
    client.post("/api/events", json={"kind": "worker-restored", "target": "w-dan"})
    world = client.get("/api/world").json()
    dan = next(w for w in world["workers"] if w["name"] == "Dan")
    assert dan["available"] is True


def test_a_one_day_mark_out_touches_exactly_one_day(client: TestClient):
    """Clicking a rota cell means "out THAT day". The first shipped version recorded
    "out from this moment", which at the wrong time of night either emptied the whole
    visible week or did nothing at all - both reported from the same evening."""
    world = client.get("/api/world").json()
    dan = next(w for w in world["workers"] if w["name"] == "Dan")
    target = next(d["date"] for d in dan["days"][1:] if d["shift"] != "off")

    client.post(
        "/api/events",
        json={
            "kind": "worker-unavailable",
            "target": "w-dan",
            "window_start": f"{target}T00:00",
            "until": f"{target}T23:59",
        },
    )
    dan = next(w for w in client.get("/api/world").json()["workers"] if w["name"] == "Dan")
    out = [d["date"] for d in dan["days"] if d["shift"] != "off" and not d["available"]]
    assert out == [target], "exactly the clicked day, nothing else"
    assert dan["available"] is True, "a future absence does not grey the row today"

    # Clicking the out cell again brings that day back.
    client.post(
        "/api/events",
        json={
            "kind": "worker-restored",
            "target": "w-dan",
            "window_start": f"{target}T00:00",
            "window_end": f"{target}T23:59",
        },
    )
    dan = next(w for w in client.get("/api/world").json()["workers"] if w["name"] == "Dan")
    assert all(d["available"] for d in dan["days"] if d["shift"] != "off")


def test_restore_beats_an_outage_with_an_explicit_until(client: TestClient):
    """The bug that made the restore arrow a lie: restore only closed OPEN-ENDED
    outages, so anything recorded with an until could never be brought back."""
    world = client.get("/api/world").json()
    dan = next(w for w in world["workers"] if w["name"] == "Dan")
    # Today's cell depends on what time the suite runs (a shift already over cannot
    # be "taken out"), so the assertions stick to strictly future days.
    future = [d["date"] for d in dan["days"][1:] if d["shift"] != "off"]

    client.post(
        "/api/events",
        json={"kind": "worker-unavailable", "target": "w-dan", "until": f"{future[-1]}T23:59"},
    )
    dan = next(w for w in client.get("/api/world").json()["workers"] if w["name"] == "Dan")
    assert not any(d["available"] for d in dan["days"] if d["date"] in future)

    client.post("/api/events", json={"kind": "worker-restored", "target": "w-dan"})
    dan = next(w for w in client.get("/api/world").json()["workers"] if w["name"] == "Dan")
    assert all(d["available"] for d in dan["days"] if d["date"] in future)


def test_restoring_one_day_carves_a_hole_in_a_longer_absence(client: TestClient):
    """ "He can do Wednesday after all" must not cancel the rest of the sick week."""
    world = client.get("/api/world").json()
    dan = next(w for w in world["workers"] if w["name"] == "Dan")
    week = [d["date"] for d in dan["days"][1:] if d["shift"] != "off"]
    assert len(week) >= 3
    middle = week[1]

    client.post(
        "/api/events",
        json={"kind": "worker-unavailable", "target": "w-dan", "until": f"{week[-1]}T23:59"},
    )
    client.post(
        "/api/events",
        json={
            "kind": "worker-restored",
            "target": "w-dan",
            "window_start": f"{middle}T00:00",
            "window_end": f"{middle}T23:59",
        },
    )
    dan = next(w for w in client.get("/api/world").json()["workers"] if w["name"] == "Dan")
    state_by_day = {d["date"]: d["available"] for d in dan["days"] if d["date"] in week}
    assert state_by_day[middle] is True, "the restored day is back"
    others = [v for k, v in state_by_day.items() if k != middle]
    assert not any(others), "every other day of the absence still stands"


def test_days_already_behind_the_clock_are_marked_unactionable(client: TestClient):
    """The board greys a day nothing can be done about instead of offering a click
    that silently does nothing. Strictly future days are always actionable; today
    depends on the hour the suite runs, so only its presence is asserted."""
    world = client.get("/api/world").json()
    for worker in world["workers"]:
        for d in worker["days"][1:]:
            assert d["actionable"] is True, f"{worker['name']} {d['date']} should be open"
        assert isinstance(worker["days"][0]["actionable"], bool)


def test_an_absorbable_outage_reroutes_without_a_human(client: TestClient):
    """ "Why can't it just auto adjust if possible?" It can, when every promise
    survives: mark out a fitter whose day another crew can cover, and the plan is
    recommitted silently - feasible, with the sick fitter off every route. The red
    banner is reserved for the outages where a customer would have to be called."""
    plan = client.post(COMMIT).json()
    assert plan["feasible"]
    route = next(r for r in plan["routes"] if r["stops"])
    victim = route["worker_names"][0]
    world = client.get("/api/world").json()
    worker_id = next(w["id"] for w in world["workers"] if w["name"] == victim)

    reply = client.post(
        "/api/events",
        json={
            "kind": "worker-unavailable",
            "target": worker_id,
            # A full-ISO "at" anchors the event inside the fixture week; a bare
            # HH:MM lands on the current week's rolling anchor, and the clamp that
            # forbids backdating (correctly) voids an outage recorded against a
            # plan whose dates are already behind the real clock.
            "at": f"{route['date']}T00:01",
            "window_start": f"{route['date']}T00:00",
            "until": f"{route['date']}T23:59",
        },
    ).json()

    after = client.get("/api/plan").json()
    if reply["note"]:
        # The system absorbed it: new head, still feasible, the victim is gone.
        assert "every promise kept" in reply["note"]
        assert after["feasible"], after["violations"]
        on_that_day = [
            n for r in after["routes"] if r["date"] == route["date"] for n in r["worker_names"]
        ]
        assert victim not in on_that_day
    else:
        # Not absorbable (no cover, or a promise would move): the honest stale banner.
        assert not after["feasible"]


def test_booking_overtime_floats_an_offer_and_a_claim_moves_the_crew(
    client: TestClient, monkeypatch
):
    """The whole volunteer-overtime loop over HTTP: a four-o'clock booking that runs
    past five floats an offer to every qualified fitter (texted, in log mode), the
    rota shows it, a relayed "yes" pins the claimant, and the re-plan puts them on
    the route without the customer's window moving."""
    from krama.api import main as api_main
    from krama.notify import RecordingNotifier

    recorder = RecordingNotifier()
    monkeypatch.setattr(api_main, "notifier", lambda: recorder)
    # Synthetic travel: this booking lands on a REAL future Monday, and the frozen
    # snapshot only holds the legs and buckets the fixture week exercises.
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")

    draft = {
        "customer_name": "Evening caller",
        "phone": "9132934243",
        "address": "somewhere real",
        "service_type": "residential_window_replacement",
        "duration_minutes": 120,
        "duration_confidence": 60,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    # A real future Monday: the fixture week is behind the clock by now, and a
    # booking in the past can never be placed on the rolling horizon.
    from datetime import date as _date

    monday = _date.fromisoformat(_next_monday())
    booked = client.post(
        "/api/book", json={"draft": draft, "date": monday.isoformat(), "arrival": "16:00"}
    ).json()
    assert booked["status"] == "scheduled"
    assert "past shift" in booked["overtime"], booked["overtime"]

    world = client.get("/api/world").json()
    offers = world["overtime_offers"]
    assert len(offers) == 1
    offer = offers[0]
    assert offer["status"] == "open"
    assert offer["overtime_minutes"] > 0
    # Everyone qualified got the text; nobody unqualified did.
    assert recorder.sent, "offer must be texted"
    assert set(offer["offered_to"]) >= {"Dan", "Ken"}
    assert "Sofia" not in offer["offered_to"], "not residential-certified"

    # A fitter who was never offered cannot claim.
    refused = client.post(
        f"/api/overtime/{booked['job_id']}/claim", params={"worker_id": "w-sofia"}
    )
    assert refused.status_code == 422

    # Ken says yes.
    names = dict(zip(offer["offered_to"], offer["offered_ids"], strict=True))
    claimer = names["Ken"]
    claimed = client.post(
        f"/api/overtime/{booked['job_id']}/claim", params={"worker_id": claimer}
    ).json()
    assert claimed["worker"] == "Ken"
    assert claimed["status"] == "on the route"

    plan = client.get("/api/plan").json()
    crew = next(
        r["worker_names"]
        for r in plan["routes"]
        for s in r["stops"]
        if s["job_id"] == booked["job_id"]
    )
    assert "Ken" in crew
    stop = next(s for r in plan["routes"] for s in r["stops"] if s["job_id"] == booked["job_id"])
    assert stop["past_shift"] is True, "still drawn as overtime on the calendar"

    # Second yes is politely turned away.
    second = client.post(
        f"/api/overtime/{booked['job_id']}/claim",
        params={"worker_id": offer["offered_ids"][0]},
    )
    assert second.status_code == 409
    assert "already took these hours" in second.json()["detail"]["detail"]

    # The board shows who has it.
    world = client.get("/api/world").json()
    assert world["overtime_offers"][0]["claimed_by"] == "Ken"


def test_hours_are_only_offered_to_fitters_who_can_actually_reach_them(
    client: TestClient, monkeypatch
):
    """The first live run offered a four-to-six evening to two fitters whose shifts
    end at three - people whose yes the solver could never honour. An offer must go
    only to crew whose shift plus overtime covers the job's whole span."""
    from datetime import date as _date

    from krama.api import main as api_main
    from krama.notify import RecordingNotifier

    monkeypatch.setattr(api_main, "notifier", lambda: RecordingNotifier())
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")

    draft = {
        "customer_name": "Evening caller",
        "phone": "9132934243",
        "address": "somewhere real",
        "service_type": "residential_window_replacement",
        "duration_minutes": 120,
        "duration_confidence": 60,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    monday = _date.fromisoformat(_next_monday())
    client.post("/api/book", json={"draft": draft, "date": monday.isoformat(), "arrival": "16:00"})

    offer = client.get("/api/world").json()["overtime_offers"][0]
    assert set(offer["offered_to"]) == {"Dan", "Ken"}, (
        "Marcus and Priya finish at three and cannot reach six even on overtime; "
        "Sofia and Alex are not residential-certified"
    )


def test_the_ask_endpoints_round_trip(client: TestClient):
    """Open a crew ask, record one yes, close it - the board state follows."""
    from datetime import date as _date

    monday = _date.fromisoformat(_next_monday())
    opened = client.post(
        "/api/asks",
        json={
            "customer_name": "Jimmy",
            "phone": "8175768492",
            "transcript": "walmart glass broke, come at 5pm",
            "on_date": monday.isoformat(),
            "until": "20:45",
            "candidate_ids": ["w-marcus", "w-priya"],
            "detail": "check with Marcus and Priya",
        },
    ).json()
    ask_id = opened["ask_id"]

    world = client.get("/api/world").json()
    ask = next(a for a in world["crew_asks"] if a["ask_id"] == ask_id)
    assert ask["customer"] == "Jimmy"
    assert [c["name"] for c in ask["candidates"]] == ["Marcus", "Priya"]
    assert ask["extended"] == []

    refused = client.post(f"/api/asks/{ask_id}/extend", params={"worker_id": "w-dan"})
    assert refused.status_code == 422, "Dan was never part of this ask"

    said_yes = client.post(f"/api/asks/{ask_id}/extend", params={"worker_id": "w-priya"}).json()
    assert said_yes["worker"] == "Priya"
    world = client.get("/api/world").json()
    ask = next(a for a in world["crew_asks"] if a["ask_id"] == ask_id)
    assert ask["extended"] == ["Priya"]
    priya = next(w for w in world["workers"] if w["name"] == "Priya")
    monday_cell = next(d for d in priya["days"] if d["date"] == monday.isoformat())
    # Named and scoped: the yes was for Jimmy's evening, and the chip says so -
    # a bare "agreed to stay" read as standing overtime capacity, which it is not.
    assert "staying to" in monday_cell["extended"], "the rota explains the late evening"
    assert "for Jimmy" in monday_cell["extended"], "and names who the yes was for"

    client.post(f"/api/asks/{ask_id}/close", params={"outcome": "booked"})
    world = client.get("/api/world").json()
    assert not any(a["ask_id"] == ask_id for a in world["crew_asks"])


def test_the_configure_endpoints_edit_the_roster(client: TestClient):
    """Save a fitter, trim the fleet, hire somebody new - all as roster events."""
    world = client.get("/api/world").json()
    sofia = next(w for w in world["workers"] if w["name"] == "Sofia")

    saved = client.post(
        "/api/config/worker",
        json={
            "id": sofia["id"],
            "name": "Sofia",
            "phone": "+18175550199",
            "certifications": ["auto_glass", "residential_glazing"],
            "shift_start": "07:00",
            "shift_end": "16:00",
            "overtime_eligible": False,
        },
    ).json()
    assert saved["status"] == "saved"
    sofia = next(w for w in client.get("/api/world").json()["workers"] if w["name"] == "Sofia")
    assert sofia["shift_start"] == "07:00"
    assert "residential_glazing" in sofia["certifications"]
    assert sofia["overtime_eligible"] is False

    hired = client.post(
        "/api/config/worker",
        json={"name": "Tess", "certifications": ["screen_repair"]},
    ).json()
    assert hired["status"] == "saved"
    assert any(w["name"] == "Tess" for w in client.get("/api/world").json()["workers"])

    client.delete(f"/api/config/worker/{hired['worker_id']}")
    assert not any(w["name"] == "Tess" for w in client.get("/api/world").json()["workers"])

    gone = client.delete("/api/config/van/van-4").json()
    assert gone["status"] == "removed"
    vans = {v["id"] for v in client.get("/api/world").json()["vans"]}
    assert "van-4" not in vans
    added = client.post("/api/config/van", json={"id": "", "label": ""}).json()
    assert added["van_id"] not in vans, "a fresh id, not a reused one"

    bad = client.post(
        "/api/config/worker",
        json={"name": "X", "shift_start": "18:00", "shift_end": "09:00"},
    )
    assert bad.status_code == 422, "a shift must end after it starts"


def test_the_accepted_price_is_a_fact_everywhere(client: TestClient, monkeypatch):
    """The quote used to evaporate the moment the slot card left the screen: the
    job booked with revenue 0, the scheduler weighed a nine-hundred-dollar customer
    like a zero-dollar one, and no surface could ever say what was agreed. The
    price the dispatcher pressed now lands on the job, the world view, and the
    scheduling weight - one truth, read from one place."""
    from datetime import date as _date

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    draft = {
        "customer_name": "Priced caller",
        "phone": "9132934243",
        "address": "somewhere real",
        "service_type": "residential_window_replacement",
        "duration_minutes": 120,
        "duration_confidence": 60,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    monday = _date.fromisoformat(_next_monday())
    booked = client.post(
        "/api/book",
        json={
            "draft": draft,
            "date": monday.isoformat(),
            "arrival": "10:00",
            "quoted_total": 578.50,
        },
    ).json()

    job = next(j for j in client.get("/api/world").json()["jobs"] if j["id"] == booked["job_id"])
    assert job["quoted_total"] == 578.50, "the spoken price is on the job"

    # The scheduling weight is the pre-tax share - tax was never ours to keep.
    from krama.api.main import service

    stored = service().world().jobs[booked["job_id"]]
    assert stored.quoted_total == 578.50
    assert 0 < stored.revenue < 578.50, "revenue is the ex-tax share, not zero, not the total"

    # The confirmation echoes the promise the slot card made - an arrival window
    # that exists elsewhere in the system, not a hybrid of window-open and estimate.
    assert "arriving" in booked["when"]
    assert "10:00 AM" in booked["when"]


def test_job_ids_never_recount(client: TestClient, monkeypatch):
    """Ids come from the highest existing suffix, not from len(jobs): a recount
    mints twins the moment anything ever leaves the dict, and a twin id silently
    merges two customers' histories."""
    from datetime import date as _date

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    draft = {
        "customer_name": "A",
        "phone": "9132934243",
        "address": "x",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    monday = _date.fromisoformat(_next_monday())

    def book() -> str:
        reply = client.post(
            "/api/book", json={"draft": draft, "date": monday.isoformat(), "arrival": "09:00"}
        ).json()
        return str(reply["job_id"])

    first = book()
    client.post(f"/api/jobs/{first}/cancel")
    second = book()
    assert second != first, "a cancelled booking's id must never be reissued"


def test_the_last_van_cannot_be_removed(client: TestClient):
    """The last van carries the only record of the depot - every new van and fitter
    copies its home from an existing one. A test that emptied the garage found it
    could never refill it: nothing left to copy from, fleet bricked for good."""
    for van_id in ("van-2", "van-3", "van-4"):
        assert client.delete(f"/api/config/van/{van_id}").json()["status"] == "removed"
    refused = client.delete("/api/config/van/van-1")
    assert refused.status_code == 409
    assert "depot" in refused.json()["detail"]["detail"]
    # the documented way out works: add first, then retire
    added = client.post("/api/config/van", json={"id": "", "label": ""}).json()
    assert client.delete("/api/config/van/van-1").json()["status"] == "removed"
    vans = {v["id"] for v in client.get("/api/world").json()["vans"]}
    assert vans == {added["van_id"]}


def test_the_past_cannot_be_booked_and_cancel_tells_the_truth_twice(
    client: TestClient, monkeypatch
):
    """Two honesty holes from an adversarial pass: the API accepted a booking for a
    date already behind the clock (a confirmed ghost no plan can serve), and a second
    cancel of the same job reported "freed" again - a log event and a replan for an
    action that changed nothing."""
    from datetime import date as _date

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    draft = {
        "customer_name": "Ghost",
        "phone": "9132934243",
        "address": "x",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    past = client.post("/api/book", json={"draft": draft, "date": "2026-09-01", "arrival": "10:00"})
    assert past.status_code == 422
    assert "already happened" in past.json()["detail"]["detail"]

    monday = _date.fromisoformat(_next_monday())
    booked = client.post(
        "/api/book", json={"draft": draft, "date": monday.isoformat(), "arrival": "09:00"}
    ).json()
    assert client.post(f"/api/jobs/{booked['job_id']}/cancel").json()["status"] == "freed"
    second = client.post(f"/api/jobs/{booked['job_id']}/cancel")
    assert second.status_code == 409
    assert "already cancelled" in second.json()["detail"]["detail"]


def test_the_depot_is_pinned_on_the_actual_building():
    """The invented depot pin sat ~4 road miles north of the shop, so every first
    leg of every route carried phantom miles - the owner caught it by comparing the
    board against Google Maps. 1150 Blue Mound Rd W, per the US Census geocoder."""
    from krama.fixtures.sample_business import DEPOT

    assert abs(DEPOT.lat - 32.9464) < 0.01, DEPOT.lat
    assert abs(DEPOT.lon - (-97.3799)) < 0.01, DEPOT.lon
    assert "1150 Blue Mound" in DEPOT.address


def test_every_stop_says_where_its_drive_began(client: TestClient, monkeypatch):
    """ "Drive there: 15 min" answered the wrong question - the owner wanted to know
    whether the van chained from the previous job or doubled back to the shop. The
    first stop says "the shop"; each later stop names the customer it came from."""
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    monday = _next_monday()

    def draft(name: str) -> dict[str, object]:
        return {
            "customer_name": name,
            "phone": "9132934243",
            "address": "x",
            "service_type": "residential_window_replacement",
            "duration_minutes": 60,
            "duration_confidence": 30,
            "crew_size": 1,
            "certifications": ["residential_glazing"],
            "commitment_cost": 0,
            "lat": 32.99,
            "lon": -97.36,
        }

    client.post("/api/book", json={"draft": draft("First"), "date": monday, "arrival": "09:00"})
    # Same phone on purpose - a genuine second job, so the duplicate gate needs
    # the human's explicit yes.
    client.post(
        "/api/book",
        json={
            "draft": draft("Second"),
            "date": monday,
            "arrival": "11:00",
            "allow_duplicate": True,
        },
    )

    plan = client.get("/api/plan").json()
    labels = {
        s["customer_name"]: s["from_label"]
        for r in plan["routes"]
        if r["date"] == monday
        for s in r["stops"]
    }
    assert labels.get("First") == "the shop", labels
    assert labels.get("Second") == "First's", labels

    # and each stop names where the van goes NEXT, so the popup reads the whole
    # chain: shop -> First -> Second -> back to the shop.
    nexts = {
        s["customer_name"]: s["next_label"]
        for r in plan["routes"]
        if r["date"] == monday
        for s in r["stops"]
    }
    assert nexts.get("First") == "Second's", nexts
    assert nexts.get("Second") == "back to the shop", nexts

    # The run description is LIVE: before the second booking the card called
    # First's visit a dedicated trip; now both stops say they share the run.
    # (The stale quote-time note read "a dedicated trip" forever on a three-stop
    # morning, and the owner caught it on the screen.)
    runs = {
        s["customer_name"]: s["run_note"]
        for r in plan["routes"]
        if r["date"] == monday
        for s in r["stops"]
    }
    assert "2 stops" in runs.get("First", ""), runs
    assert "Second's" in runs.get("First", ""), runs
    assert "First's" in runs.get("Second", ""), runs

    # and a cancellation updates it right back to a dedicated trip
    second_id = next(
        s["job_id"]
        for r in plan["routes"]
        if r["date"] == monday
        for s in r["stops"]
        if s["customer_name"] == "Second"
    )
    client.post(f"/api/jobs/{second_id}/cancel")
    plan = client.get("/api/plan").json()
    runs = {
        s["customer_name"]: s["run_note"]
        for r in plan["routes"]
        if r["date"] == monday
        for s in r["stops"]
    }
    assert runs.get("First") == "a dedicated trip out and back", runs


def test_moving_the_shop_is_guarded_and_total(client: TestClient, monkeypatch):
    """The depot is the most consequential coordinate in the system - it was once
    wrong by four road miles and every route carried the error. Moving it demands
    the address typed twice; a mismatch is refused; a successful move rewrites
    EVERY van's home in one event so no record can drift from another."""
    from krama.domain.models import Location
    from krama.geocoding import Geocoder

    resolved = Location(
        lat=32.9300,
        lon=-97.3660,
        address="11100 Dunlavin Ct, Haslet, TX 76052",
        precision="house",
    )
    monkeypatch.setattr(Geocoder, "geocode", lambda self, addr: resolved)

    mismatch = client.post(
        "/api/config/depot",
        json={"address": "11100 Dunlavin Ct, Haslet, TX", "confirm": "11100 Dunlavin Ct"},
    )
    assert mismatch.status_code == 422
    assert "retype" in mismatch.json()["detail"]["detail"]

    before = client.get("/api/world").json()["depot_address"]
    moved = client.post(
        "/api/config/depot",
        json={
            "address": "11100 Dunlavin Ct, Haslet, TX",
            "confirm": "11100 Dunlavin Ct, Haslet, TX",
        },
    ).json()
    assert "Dunlavin" in moved["address"]

    world = client.get("/api/world").json()
    assert world["depot_address"] == resolved.address != before

    # every van agrees - the fold rewrote them all in the same breath
    from krama.api.main import service

    vans = service().world().vans.values()
    assert all(v.home_depot.address == resolved.address for v in vans)
    assert all(abs(v.home_depot.lat - resolved.lat) < 1e-9 for v in vans)


def test_a_depot_that_is_not_a_building_is_refused(client: TestClient, monkeypatch):
    from krama.domain.models import Location
    from krama.geocoding import Geocoder

    roadish = Location(lat=32.95, lon=-97.35, address="Blue Mound Rd", precision="road")
    monkeypatch.setattr(Geocoder, "geocode", lambda self, addr: roadish)
    refused = client.post(
        "/api/config/depot",
        json={"address": "Blue Mound Rd", "confirm": "Blue Mound Rd"},
    )
    assert refused.status_code == 422
    assert "street number" in refused.json()["detail"]["remedy"]


def test_the_booking_rationale_survives_the_slot_card(client: TestClient, monkeypatch):
    """ "Why Marcus, why Tuesday" used to evaporate with the slot card. The pressed
    slot's own explanation now rides the job and comes back on the world view,
    labelled quote-time truth for the detail card to show."""
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    draft = {
        "customer_name": "Why",
        "phone": "9132934243",
        "address": "x",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    note = "already 2 stops nearby, +6 min detour · one fitter needed"
    booked = client.post(
        "/api/book",
        json={
            "draft": draft,
            "date": _next_monday(),
            "arrival": "09:00",
            "booking_note": note,
        },
    ).json()
    job = next(j for j in client.get("/api/world").json()["jobs"] if j["id"] == booked["job_id"])
    assert job["booking_note"] == note


def test_the_card_and_the_commit_name_the_same_crew(client: TestClient, monkeypatch):
    """The slot card said Marcus; the committed plan said Priya. Equal-cost crews
    made the fresh commit free to differ from the card the dispatcher had just read
    aloud. The quoted crew now rides the booking and pins its own commit - later
    re-plans stay free to reshuffle, but the plan the confirmation is read against
    must match the card."""
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    draft = {
        "customer_name": "Pinned",
        "phone": "9132934243",
        "address": "x",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    # Priya would not be the tie-break's first pick for a residential solo job, so
    # a surviving pin is proof the pin did the work.
    booked = client.post(
        "/api/book",
        json={
            "draft": draft,
            "date": _next_monday(),
            "arrival": "09:00",
            "crew_ids": ["w-priya"],
        },
    ).json()
    assert booked["status"] == "scheduled"
    plan = client.get("/api/plan").json()
    crew = next(
        r["worker_ids"] if "worker_ids" in r else r["worker_names"]
        for r in plan["routes"]
        for s in r["stops"]
        if s["job_id"] == booked["job_id"]
    )
    assert "Priya" in " ".join(map(str, crew)) or "w-priya" in crew, crew


def test_crew_swap_is_priced_checked_and_committed(client: TestClient, monkeypatch):
    """The dispatcher asks "who else could take this, and what would it cost" and
    gets measured answers: a qualified fitter with a dollar delta, an unqualified
    one refused in words, and an applied swap that re-plans without touching any
    promised window."""
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    draft = {
        "customer_name": "Swappable",
        "phone": "9132934243",
        "address": "x",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    booked = client.post(
        "/api/book", json={"draft": draft, "date": _next_monday(), "arrival": "09:00"}
    ).json()
    job_id = booked["job_id"]

    options = client.get(f"/api/jobs/{job_id}/crew-options").json()["options"]
    by_id = {o["worker_id"]: o for o in options}
    assert by_id["w-sofia"]["feasible"] is False, "auto-glass Sofia cannot take residential"
    assert "not qualified" in by_id["w-sofia"]["note"]
    current = [o for o in options if o.get("current")]
    assert current, "somebody is on it now"
    other = next(o for o in options if o["feasible"] and not o.get("current") and "cost_delta" in o)
    assert isinstance(other["cost_delta"], float)

    swapped = client.post(
        f"/api/jobs/{job_id}/crew", params={"worker_id": other["worker_id"]}
    ).json()
    assert swapped["status"] == "re-planned around the swap"
    plan = client.get("/api/plan").json()
    crew_names = next(
        r["worker_names"] for r in plan["routes"] for s in r["stops"] if s["job_id"] == job_id
    )
    assert other["name"] in crew_names

    refused = client.post(f"/api/jobs/{job_id}/crew", params={"worker_id": "w-sofia"})
    assert refused.status_code == 422


def test_owner_prices_land_in_the_log_and_in_the_next_quote(client: TestClient):
    """Setting real prices is an event: auditable, redeploy-proof, and in force from
    the very next quote. The provenance flips so the "estimated costs" warning can
    retire itself one number at a time."""
    before = client.get("/api/config/pricing").json()["fields"]
    labour = next(f for f in before if f["key"] == "labour_rate_per_hour")
    assert labour["source"] == "estimated"

    out = client.post(
        "/api/config/pricing", json={"labour_rate_per_hour": 120.0, "call_out_fee": 80.0}
    ).json()
    assert "2 price(s) set" in out["status"]

    after = client.get("/api/config/pricing").json()["fields"]
    labour = next(f for f in after if f["key"] == "labour_rate_per_hour")
    assert labour["value"] == 120.0
    assert labour["source"] == "confirmed"

    slipped = client.post("/api/config/pricing", json={"labour_rate_per_hour": 9500.0})
    assert slipped.status_code == 422
    assert "slipped decimal" in slipped.json()["detail"]["remedy"]

    unknown = client.post("/api/config/pricing", json={"evil_field": 1.0})
    assert unknown.status_code == 422


def test_dispatcher_sessions_never_receive_the_owners_numbers(client: TestClient, monkeypatch):
    """With KRAMA_OWNER_PIN set, the server STRIPS business-private money for
    sessions without the key - margins zeroed and flagged, week cost zeroed, the
    rate card and the shop's address behind 403. The right header brings it all
    back. Stripping is server-side: an employee's browser never holds the numbers."""
    monkeypatch.setenv("KRAMA_OWNER_PIN", "4242")
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")

    locked = client.get("/api/config/pricing")
    assert locked.status_code == 403
    assert client.get("/api/session").json() == {"owner_pin_set": True, "owner": False}

    draft = {
        "customer_name": "Redacted",
        "phone": "9132934243",
        "address": "x",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    client.post("/api/book", json={"draft": draft, "date": _next_monday(), "arrival": "09:00"})
    plan = client.get("/api/plan").json()
    assert plan["redacted"] is True
    assert plan["cost"]["total"] == 0.0, "week cost is the owner's number"
    assert plan["feasible"] is not None, "operational truth still flows"

    owner = {"X-Owner-Key": "4242"}
    assert client.get("/api/session", headers=owner).json()["owner"] is True
    full = client.get("/api/plan", headers=owner).json()
    assert full["redacted"] is False
    assert client.get("/api/config/pricing", headers=owner).status_code == 200

    wrong = client.get("/api/config/pricing", headers={"X-Owner-Key": "0000"})
    assert wrong.status_code == 403


def test_the_factory_pin_enforces_out_of_the_box(client: TestClient):
    """The owner split must not be a suggestion that waits for an env var: the
    shipped config carries a sample PIN, so a fresh board is dispatcher-safe the
    moment it starts. A bare client - no header - is locked out of the owner
    surfaces and gets stripped money everywhere else."""
    from fastapi.testclient import TestClient as Bare

    from krama.api.main import app

    bare = Bare(app)
    assert bare.get("/api/session").json() == {"owner_pin_set": True, "owner": False}
    assert bare.get("/api/config/pricing").status_code == 403
    refused = bare.post("/api/config/depot", json={"address": "x", "confirm": "x"})
    assert refused.status_code == 403

    # and the owner header opens it, because the pin is config, not code
    assert client.get("/api/config/pricing").status_code == 200


def test_the_horizon_never_lags_the_calendar(client: TestClient, monkeypatch):
    """The committed plan's start used to anchor every future solve - so on
    Thursday the board still planned LAST week, and a booking for next Monday,
    offered and priced by the quote, fell outside the commit's window and showed
    on no screen. The anchor now rolls: never behind today's next working day."""
    from krama.api.main import _default_start, service

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    client.post(COMMIT)  # a plan anchored to the fixture's long-gone week
    svc = service()
    head = svc.head()
    assert head is not None and head.horizon_start == WEEK_START, "the stale anchor is real"
    assert _default_start(svc) >= _business_today(), "and it must not win"

    draft = {
        "customer_name": "Rolls",
        "phone": "9132934243",
        "address": "x",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    booked = client.post(
        "/api/book", json={"draft": draft, "date": _next_monday(), "arrival": "09:00"}
    ).json()
    assert booked["status"] == "scheduled", (
        "a bookable near-future day must land on the plan, stale head or not"
    )


def test_a_promise_beyond_the_horizon_still_has_a_tile_to_draw(client: TestClient, monkeypatch):
    """A booking ten working days out is legitimate (the quote walks 21) but no
    committed plan reaches it yet. The world view now carries machine-readable
    promise bounds so the calendar draws it as a ghost - a promise with no tile
    is a promise someone will forget."""
    from datetime import date as _date
    from datetime import timedelta as _timedelta

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    far = _date.fromisoformat(_next_monday()) + _timedelta(days=14)
    draft = {
        "customer_name": "Far Out",
        "phone": "9132934243",
        "address": "x",
        "service_type": "residential_window_replacement",
        "duration_minutes": 60,
        "duration_confidence": 30,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    booked = client.post(
        "/api/book", json={"draft": draft, "date": far.isoformat(), "arrival": "09:00"}
    ).json()
    assert booked["status"] == "booked but not yet scheduled"

    job = next(j for j in client.get("/api/world").json()["jobs"] if j["id"] == booked["job_id"])
    assert job["window_start"].startswith(far.isoformat())
    assert job["window_end"] > job["window_start"]
    assert job["commitment_state"] == "confirmed"


def test_the_same_caller_cannot_be_double_booked_by_accident(client: TestClient, monkeypatch):
    """Two Jameses on one Monday, from one dispatcher reading one call twice. The
    phone number is a checkable fact, so the gate is deterministic - and it is a
    GATE, not a merge, because a customer with two broken windows is two jobs and
    only a person can tell an edit from an addition."""
    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    draft = {
        "customer_name": "James",
        "phone": "9132952960",
        "address": "11100 Dunlavin Ct, Haslet, TX 76052",
        "service_type": "residential_window_replacement",
        "duration_minutes": 120,
        "duration_confidence": 60,
        "crew_size": 1,
        "certifications": ["residential_glazing"],
        "commitment_cost": 0,
        "lat": 32.93,
        "lon": -97.366,
    }
    first = client.post(
        "/api/book", json={"draft": draft, "date": _next_monday(), "arrival": "08:00"}
    ).json()
    assert first["status"] == "scheduled"

    # the exact rebook that produced two Jameses, refused with the existing
    # booking named
    second = client.post(
        "/api/book", json={"draft": draft, "date": _next_monday(), "arrival": "10:00"}
    )
    assert second.status_code == 409
    body = second.json()["detail"]
    assert body["error"] == "PossibleDuplicate"
    assert "James" in body["detail"]
    assert "second job" in body["remedy"]

    # formatted differently, same digits - still caught
    reformatted = {**draft, "phone": "(913) 295-2960", "address": "somewhere else"}
    assert (
        client.post(
            "/api/book", json={"draft": reformatted, "date": _next_monday(), "arrival": "13:00"}
        ).status_code
        == 409
    )

    # the human says it really is a second job - allowed, distinct id
    second_job = client.post(
        "/api/book",
        json={
            "draft": draft,
            "date": _next_monday(),
            "arrival": "10:00",
            "allow_duplicate": True,
        },
    ).json()
    assert second_job["job_id"] != first["job_id"]
    assert second_job["status"] == "scheduled"

    # a cancelled booking is not a duplicate - rebooking after a cancel is routine
    client.post(f"/api/jobs/{second_job['job_id']}/cancel")
    client.post(f"/api/jobs/{first['job_id']}/cancel")
    again = client.post(
        "/api/book", json={"draft": draft, "date": _next_monday(), "arrival": "09:00"}
    ).json()
    assert again["status"] == "scheduled"


def test_van_stock_is_visible_and_editable(client: TestClient):
    """The fleet trim that retired van-3 took the business's only shower kits with
    it, and nothing on the board could even show that. Stock is now on the world
    view and editable per van: counts land, zeros are stripped rather than stored,
    and nonsense counts are refused."""
    vans = {v["id"]: v for v in client.get("/api/world").json()["vans"]}
    assert vans["van-3"]["stock"].get("shower_kit") == 2, "the fixture's only shower kits"

    saved = client.post(
        "/api/config/van",
        json={"id": "van-1", "stock": {"annealed_std": 6, "shower_kit": 3, "screen_kit": 0}},
    ).json()
    assert saved["status"] == "saved"
    van_1 = next(v for v in client.get("/api/world").json()["vans"] if v["id"] == "van-1")
    assert van_1["stock"]["shower_kit"] == 3
    assert "screen_kit" not in van_1["stock"], "a zero is an absence, not a stored zero"

    # Omitting stock edits the label without touching the racks.
    client.post("/api/config/van", json={"id": "van-1", "label": "Big red"})
    van_1 = next(v for v in client.get("/api/world").json()["vans"] if v["id"] == "van-1")
    assert van_1["stock"]["shower_kit"] == 3

    for bad in ({"shower_kit": -1}, {"shower_kit": 100}):
        refused = client.post("/api/config/van", json={"id": "van-1", "stock": bad})
        assert refused.status_code == 422, f"{bad} is not a shelf count"


def test_a_stranded_part_names_its_fix_and_restocking_cures_it(client: TestClient, monkeypatch):
    """ "No van stocks 1x shower_kit", five days running, was true and useless - the
    dispatcher's next question is always "so what do I do". The refusal now names
    the remedy, and performing it (restock any van in Configure) actually cures the
    refusal: the same draft starts getting slots."""
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import time as _time

    from krama.api.main import _with_remedy, service
    from krama.domain.enums import Certification, ServiceType
    from krama.domain.models import Job, Location, Material, TimeWindow

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    # The user's fleet trim: retire the one van that carried shower kits.
    assert client.delete("/api/config/van/van-3").json()["status"] == "removed"

    svc = service()
    monday = _date.fromisoformat(_next_monday())
    days = [_date.fromordinal(monday.toordinal() + i) for i in range(5)]
    draft = Job(
        id="draft",
        customer_id="c-draft",
        customer_name="Maria",
        location=Location(lat=32.99, lon=-97.36, address="aquarium house"),
        service_type=ServiceType.SHOWER_DOOR_INSTALL,
        required_certifications=frozenset({Certification.SHOWER_DOOR}),
        crew_size=1,
        estimated_duration_min=120,
        revenue=900.0,
        materials=(Material(part_code="shower_kit", quantity=1, in_stock=True),),
        windows=tuple(
            TimeWindow(
                start=_datetime.combine(d, _time(8), tzinfo=svc.tz),
                end=_datetime.combine(d, _time(17), tzinfo=svc.tz),
            )
            for d in days
        ),
        requested_at=_datetime.combine(monday, _time(7), tzinfo=svc.tz),
    )

    stranded = svc.booking_slots(draft, monday)
    assert stranded.slots == (), "no van carries the part, so no day may be offered"
    detail = next(u.detail for u in stranded.unavailable)
    assert "no van stocks 1x shower_kit" in detail
    assert "restock a van in Configure" in _with_remedy(detail), "the refusal names its fix"

    restocked = client.post(
        "/api/config/van",
        json={"id": "van-1", "stock": {"annealed_std": 6, "tempered_std": 4, "shower_kit": 2}},
    )
    assert restocked.json()["status"] == "saved"
    cured = service().booking_slots(draft, monday)
    assert cured.slots, "the remedy the message names actually works"


def test_the_rota_walks_weeks_like_the_calendar(client: TestClient):
    """Holiday lives in the future. The rota used to render the five days from
    today, always - a fitter's day off three weeks out had no cell to click. The
    world view now takes a start date and deals the week from there."""
    from datetime import timedelta as _timedelta

    today_days = [d["date"] for d in client.get("/api/world").json()["workers"][0]["days"]]
    start = _business_today() + _timedelta(days=21)
    ahead = client.get(f"/api/world?rota_start={start.isoformat()}").json()
    future_days = [d["date"] for d in ahead["workers"][0]["days"]]

    assert len(future_days) == 5, "a working week, wherever it starts"
    assert future_days[0] >= start.isoformat(), "the week dealt is the week asked for"
    assert set(future_days).isdisjoint(today_days), "three weeks out shares no day with now"

    # A mark-out on a future-week day lands on that day, visible when that week is dealt.
    dan_future = next(w for w in ahead["workers"] if w["name"] == "Dan")
    target = next(d["date"] for d in dan_future["days"] if d["shift"] != "off")
    client.post(
        "/api/events",
        json={
            "kind": "worker-unavailable",
            "target": "w-dan",
            "window_start": f"{target}T00:00",
            "until": f"{target}T23:59",
        },
    )
    ahead = client.get(f"/api/world?rota_start={start.isoformat()}").json()
    dan = next(w for w in ahead["workers"] if w["name"] == "Dan")
    cell = next(d for d in dan["days"] if d["date"] == target)
    assert cell["available"] is False, "the future holiday is on its cell"
    near = client.get("/api/world").json()
    dan_now = next(w for w in near["workers"] if w["name"] == "Dan")
    assert all(d["available"] for d in dan_now["days"] if d["shift"] != "off"), (
        "and nowhere near this week"
    )


def test_booking_the_caller_resolves_their_crew_ask(client: TestClient, monkeypatch):
    """The red strip is a promise to call the customer back. Booking them IS the
    call-back, so the strip must clear itself - leaving it lit until someone also
    finds the Resolve button made the whole flow read as broken ("said yes and
    nothing happened"). Phone digits match the ask to the booking; formatting
    differences between the two entries must not keep the strip alive."""
    from datetime import date as _date

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    monday = _date.fromisoformat(_next_monday())
    opened = client.post(
        "/api/asks",
        json={
            "customer_name": "Jimmy",
            "phone": "(817) 576-8492",
            "transcript": "walmart glass broke, come at 5pm",
            "on_date": monday.isoformat(),
            "until": "20:45",
            "candidate_ids": ["w-marcus", "w-priya"],
            "detail": "check with Marcus and Priya",
        },
    ).json()
    ask_id = opened["ask_id"]

    draft = {
        "customer_name": "Jimmy",
        "phone": "8175768492",
        "address": "walmart haslet",
        "service_type": "storefront_glass",
        "duration_minutes": 120,
        "duration_confidence": 60,
        "crew_size": 1,
        "certifications": ["commercial_storefront"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    booked = client.post(
        "/api/book",
        json={"draft": draft, "date": monday.isoformat(), "arrival": "10:00"},
    ).json()
    assert booked["job_id"], "the booking itself succeeded"

    asks = client.get("/api/world").json()["crew_asks"]
    assert all(a["ask_id"] != ask_id for a in asks), "booking the caller cleared their strip"

    # A different caller's booking must not resolve someone else's promise.
    other = client.post(
        "/api/asks",
        json={
            "customer_name": "Rosa",
            "phone": "8175550000",
            "transcript": "come late please",
            "on_date": monday.isoformat(),
            "until": "19:00",
            "candidate_ids": ["w-marcus"],
            "detail": "check with Marcus",
        },
    ).json()
    client.post(
        "/api/book",
        json={
            "draft": {**draft, "customer_name": "Someone else", "phone": "8170001111"},
            "date": monday.isoformat(),
            "arrival": "13:00",
            "allow_duplicate": True,
        },
    )
    asks = client.get("/api/world").json()["crew_asks"]
    assert any(a["ask_id"] == other["ask_id"] for a in asks), "Rosa is still owed her call-back"


def _open_jimmy_ask(client: TestClient, monday: str) -> str:
    opened = client.post(
        "/api/asks",
        json={
            "customer_name": "Jimmy",
            "phone": "8175768492",
            "transcript": "walmart glass broke, come at 5pm",
            "on_date": monday,
            "until": "20:45",
            "candidate_ids": ["w-marcus", "w-priya"],
            "detail": "check with Marcus and Priya",
        },
    ).json()
    return str(opened["ask_id"])


def _monday_extensions(client: TestClient, monday: str) -> dict[str, str]:
    world = client.get("/api/world").json()
    return {
        w["name"]: next((d["extended"] for d in w["days"] if d["date"] == monday), "")
        for w in world["workers"]
        if w["name"] in ("Marcus", "Priya")
    }


def test_a_yes_lapses_when_the_customer_is_told_no(client: TestClient):
    """The yes was an answer to "can you stay for Jimmy". If Jimmy is told no, the
    agreement has nothing to back - leaving it standing turned one polite yes into
    permanent evening capacity any later booking could silently spend."""
    monday = _next_monday()
    ask_id = _open_jimmy_ask(client, monday)
    client.post(f"/api/asks/{ask_id}/extend", params={"worker_id": "w-marcus"})
    assert "for Jimmy" in _monday_extensions(client, monday)["Marcus"]

    client.post(f"/api/asks/{ask_id}/close", params={"outcome": "resolved"})
    assert _monday_extensions(client, monday)["Marcus"] == "", "told no, nobody stays late"


def test_a_yes_lapses_when_its_booking_cancels(client: TestClient, monkeypatch):
    """Booking Jimmy consumes the yes; cancelling Jimmy releases it. The fitter
    agreed to this customer's evening, not to evenings in general."""

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    monday = _next_monday()
    ask_id = _open_jimmy_ask(client, monday)
    client.post(f"/api/asks/{ask_id}/extend", params={"worker_id": "w-marcus"})
    client.post(f"/api/asks/{ask_id}/extend", params={"worker_id": "w-priya"})

    draft = {
        "customer_name": "Jimmy",
        "phone": "8175768492",
        "address": "walmart haslet",
        "service_type": "storefront_glass",
        "duration_minutes": 180,
        "duration_confidence": 90,
        "crew_size": 2,
        "certifications": ["commercial_storefront"],
        "commitment_cost": 0,
        "lat": 32.99,
        "lon": -97.36,
    }
    booked = client.post(
        "/api/book",
        json={"draft": draft, "date": monday, "arrival": "17:00"},
    ).json()
    assert booked["job_id"]
    assert "for Jimmy" in _monday_extensions(client, monday)["Marcus"], (
        "the booking keeps the agreement alive"
    )

    client.post(f"/api/jobs/{booked['job_id']}/cancel")
    after = _monday_extensions(client, monday)
    assert after["Marcus"] == "" and after["Priya"] == "", (
        "the job is gone, so nobody is staying late for it"
    )


def test_a_yes_cannot_be_spent_on_another_caller(client: TestClient, monkeypatch):
    """Marcus said yes to JIMMY's evening. A different caller wanting the same
    evening must be asked-for again, not quietly booked onto Marcus's agreement -
    we ask the crew, we never assume."""
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import time as _time

    from krama.api.main import service
    from krama.domain.enums import Certification, ServiceType
    from krama.domain.models import Job, Location, TimeWindow

    monkeypatch.setenv("KRAMA_TRAVEL", "synthetic")
    monday_iso = _next_monday()
    monday = _date.fromisoformat(monday_iso)
    ask_id = _open_jimmy_ask(client, monday_iso)
    client.post(f"/api/asks/{ask_id}/extend", params={"worker_id": "w-marcus"})
    client.post(f"/api/asks/{ask_id}/extend", params={"worker_id": "w-priya"})

    svc = service()

    def evening_draft(phone: str, name: str) -> Job:
        return Job(
            id="draft",
            customer_id="c-draft",
            customer_name=name,
            phone=phone,
            location=Location(lat=32.99, lon=-97.36, address="haslet somewhere"),
            service_type=ServiceType.STOREFRONT_GLASS,
            required_certifications=frozenset({Certification.COMMERCIAL_STOREFRONT}),
            crew_size=2,
            estimated_duration_min=180,
            revenue=1700.0,
            windows=(
                TimeWindow(
                    start=_datetime.combine(monday, _time(17), tzinfo=svc.tz),
                    end=_datetime.combine(monday, _time(21), tzinfo=svc.tz),
                ),
            ),
            requested_at=_datetime.combine(monday, _time(7), tzinfo=svc.tz),
        )

    jimmy = svc.booking_slots(evening_draft("8175768492", "Jimmy"), monday, earliest_hour=17)
    assert any(s.on_date == monday for s in jimmy.slots), (
        "the caller the yes belongs to gets the evening"
    )

    rosa = svc.booking_slots(evening_draft("8170001111", "Rosa"), monday, earliest_hour=17)
    assert not any(s.on_date == monday for s in rosa.slots), (
        "someone else's yes is not Rosa's to spend"
    )
    refusal = next(u for u in rosa.unavailable if u.on_date == monday)
    assert "agreed to stay" in refusal.detail and "ask the crew again" in refusal.detail


def test_the_price_breakdown_is_the_rate_card_and_stays_with_the_owner():
    """The itemised build-up (labour rate, materials, call-out) IS the rate card
    read sideways; a dispatcher session gets the total to read down the phone and
    nothing to reverse-engineer."""
    from unittest.mock import Mock

    from krama.api.main import _strip_margins
    from krama.api.models import DraftView, IntakeView, SlotView

    slot = SlotView(
        date="2026-10-12",
        day="Mon 12 Oct",
        window="5:00 - 7:00 PM",
        arrival="5:00 PM",
        marginal_cost=120.0,
        crew="Marcus + Priya",
        crew_reason="",
        reason="",
        quote_total=1901.95,
        quote_lines=["call-out  $65.00", "labour 6h  $855.00"],
        margin=400.0,
        margin_pct=21.0,
    )
    view = IntakeView(
        bookable=True,
        missing=[],
        ask_next=[],
        draft=DraftView(customer_name="Jimmy"),
        slots=[slot],
        flexible_slots=[slot],
    )

    owner = Mock()
    owner.headers = {"x-owner-key": "1234"}  # Starlette lower-cases header names
    dispatcher = Mock()
    dispatcher.headers = {}

    kept = _strip_margins(view, owner)
    assert kept.slots[0].quote_lines, "the owner still sees how the price is built"

    stripped = _strip_margins(view, dispatcher)
    assert stripped.redacted is True
    for s in (*stripped.slots, *stripped.flexible_slots):
        assert s.quote_total == 1901.95, "the spoken price survives"
        assert s.quote_lines == [], "the build-up does not"
        assert s.margin == 0.0 and s.margin_pct == 0.0


def test_an_appointment_is_hours_out_not_a_sick_day(client: TestClient):
    """Dan at the dentist 8-10 is not Dan gone for the day. The rota cell stays a
    working day, names the hole, hands back machine times so exactly that hole can
    be cleared - and a van's breakdown gets the same grammar as a fitter's."""
    world = client.get("/api/world").json()
    dan = next(w for w in world["workers"] if w["name"] == "Dan")
    target = next(d["date"] for d in dan["days"][1:] if d["shift"] != "off")

    client.post(
        "/api/events",
        json={
            "kind": "worker-unavailable",
            "target": "w-dan",
            "window_start": f"{target}T08:00",
            "until": f"{target}T10:00",
        },
    )
    dan = next(w for w in client.get("/api/world").json()["workers"] if w["name"] == "Dan")
    cell = next(d for d in dan["days"] if d["date"] == target)
    assert cell["available"] is True, "two hours out is not a day out"
    assert cell["out_note"] == "out 8:00 - 10:00 AM"
    assert cell["out_spans"] == [["08:00", "10:00"]]

    # Clearing exactly that hole brings the clean cell back.
    client.post(
        "/api/events",
        json={
            "kind": "worker-restored",
            "target": "w-dan",
            "window_start": f"{target}T08:00",
            "window_end": f"{target}T10:00",
        },
    )
    dan = next(w for w in client.get("/api/world").json()["workers"] if w["name"] == "Dan")
    cell = next(d for d in dan["days"] if d["date"] == target)
    assert cell["out_note"] == "" and cell["out_spans"] == []

    # The van wears the same cells: down for the morning, named and clearable.
    client.post(
        "/api/events",
        json={
            "kind": "van-unavailable",
            "target": "van-1",
            "window_start": f"{target}T06:00",
            "until": f"{target}T12:00",
        },
    )
    van = next(v for v in client.get("/api/world").json()["vans"] if v["id"] == "van-1")
    van_cell = next(d for d in van["days"] if d["date"] == target)
    assert van_cell["available"] is True
    assert van_cell["out_note"] == "out 6:00 AM - 12:00 PM"
    client.post(
        "/api/events",
        json={
            "kind": "van-restored",
            "target": "van-1",
            "window_start": f"{target}T00:00",
            "window_end": f"{target}T23:59",
        },
    )
    van = next(v for v in client.get("/api/world").json()["vans"] if v["id"] == "van-1")
    assert next(d for d in van["days"] if d["date"] == target)["out_note"] == ""


def test_a_withdrawn_yes_comes_off_the_rota_and_the_evening(client: TestClient):
    """ "Marcus can't stay anymore" was unrecordable: an extension could only lapse
    through its customer's fate, so the rota chip kept promising an evening the
    fitter had already taken back. The withdrawal is an event like everything else,
    and the evening goes back to needing a fresh ask."""
    monday = _next_monday()
    ask_id = _open_jimmy_ask(client, monday)
    client.post(f"/api/asks/{ask_id}/extend", params={"worker_id": "w-marcus"})
    assert "for Jimmy" in _monday_extensions(client, monday)["Marcus"]

    withdrawn = client.post(
        "/api/events",
        json={
            "kind": "extension-withdrawn",
            "target": "w-marcus",
            "window_start": f"{monday}T00:00",
        },
    )
    assert withdrawn.status_code == 200
    assert _monday_extensions(client, monday)["Marcus"] == "", (
        "the chip stops promising an evening the fitter took back"
    )
    # Priya never said yes and Marcus took his back - the ask still knows who to ask.
    world = client.get("/api/world").json()
    ask = next(a for a in world["crew_asks"] if a["ask_id"] == ask_id)
    assert ask["ask_id"] == ask_id, "the promise to call Jimmy back is still owed"
