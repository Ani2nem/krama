"""Agent tests.

Every one runs against a scripted provider: no network, no credentials, no spend, and
the same answer each time. A suite that needed a live model would be slow, flaky and
expensive enough that people would stop running it.

The cases that matter are the misbehaving ones. "Does the happy path work" is the
least interesting question about a small model; "what happens when it returns an
invalid enum three times running, or names a van that does not exist" is the question
the whole validate-and-repair design exists to answer.
"""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ConfigDict, Field

from krama.agents.coordinator import choose_repair
from krama.agents.llm.base import LLMError, LLMResponse, LLMUsage
from krama.agents.llm.scripted import CallbackLLMProvider, ScriptedLLMProvider
from krama.agents.structured import Example, extract, json_schema_for
from krama.agents.triage import TriageOutcome, TriageResult, triage
from krama.domain.autonomy import AutonomyPolicy
from krama.domain.state import Unavailability
from krama.fixtures.sample_business import WEEK_START, _at
from krama.obs.correlation import dispatch

TZ = ZoneInfo("America/Chicago")


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    crew_size: int = Field(ge=1, le=2)


# ------------------------------------------------------------------ the schema


def test_schema_is_self_contained():
    """Constrained-decoding paths vary in how well they follow $ref, and a schema a
    provider cannot fully honour silently becomes a weaker constraint."""
    import json

    schema = json_schema_for(TriageOutcome)
    assert "$ref" not in json.dumps(schema)
    assert "$defs" not in schema


def test_schema_forbids_unknown_fields():
    assert json_schema_for(Person)["additionalProperties"] is False


def test_enums_reach_the_schema_as_enums():
    """The model picks from a list rather than inventing a string."""
    schema = json_schema_for(TriageOutcome)
    kinds = schema["properties"]["events"]["items"]["properties"]["kind"]
    assert "van-unavailable" in kinds["enum"]


# ------------------------------------------------------------ validate and repair


def test_a_valid_first_answer_costs_one_attempt():
    provider = ScriptedLLMProvider.returning({"name": "Dan", "crew_size": 1})
    result = extract(provider, Person, system="s", text="t")
    assert result.ok and result.attempts == 1 and result.repairs == 0


def test_an_invalid_answer_is_repaired():
    provider = ScriptedLLMProvider.returning(
        {"name": "Dan", "crew_size": 4},
        {"name": "Dan", "crew_size": 2},
    )
    result = extract(provider, Person, system="s", text="t")
    assert result.ok and result.attempts == 2 and result.repairs == 1


def test_the_repair_prompt_names_the_actual_error():
    """ "That was not valid, try again" gives a small model nothing to work with."""
    provider = ScriptedLLMProvider.returning(
        {"name": "Dan", "crew_size": 4},
        {"name": "Dan", "crew_size": 2},
    )
    extract(provider, Person, system="s", text="t")
    repair_prompt = provider.last_user_text()
    assert "crew_size" in repair_prompt
    assert "4" in repair_prompt


def test_a_stuck_model_escalates_instead_of_looping():
    """A model repeating one wrong answer will keep repeating it. Burning ten calls to
    arrive nowhere is worse than handing a dispatcher a clear question."""
    provider = ScriptedLLMProvider.always({"name": "Dan", "crew_size": 9})
    result = extract(provider, Person, system="s", text="t", max_attempts=3)
    assert result.escalated
    assert result.attempts == 3
    assert provider.call_count == 3
    assert "crew_size" in result.errors[0]


def test_prose_containing_json_is_rescued_once():
    """A model that ignored the schema often still has the content right."""
    provider = ScriptedLLMProvider(
        responses=[
            LLMResponse(
                text='Sure! Here you go:\n```json\n{"name": "Dan", "crew_size": 1}\n```',
                stop_reason="end_turn",
            )
        ]
    )
    result = extract(provider, Person, system="s", text="t")
    assert result.ok and result.value is not None and result.value.name == "Dan"


def test_a_provider_failure_stops_rather_than_retrying_blindly():
    provider = ScriptedLLMProvider(responses=[LLMError("bedrock unreachable")])
    result = extract(provider, Person, system="s", text="t")
    assert result.escalated
    assert "unreachable" in result.errors[0]


def test_examples_reach_the_system_prompt():
    provider = ScriptedLLMProvider.returning({"name": "Dan", "crew_size": 1})
    extract(
        provider,
        Person,
        system="base",
        text="t",
        examples=[Example(text="Dan alone", output=Person(name="Dan", crew_size=1))],
    )
    assert "Example input" in provider.requests[0].system


def test_usage_accumulates_across_repairs():
    """Repairs are not free, and the cost of a flaky model should be visible."""
    provider = ScriptedLLMProvider.returning(
        {"name": "Dan", "crew_size": 4},
        {"name": "Dan", "crew_size": 2},
    )
    result = extract(provider, Person, system="s", text="t")
    assert result.usage.input_tokens == 240
    assert result.usage.output_tokens == 80


def test_extraction_requires_a_value_or_raises():
    provider = ScriptedLLMProvider.always({"name": "Dan", "crew_size": 9})
    result = extract(provider, Person, system="s", text="t", max_attempts=2)
    with pytest.raises(LLMError, match="no valid result"):
        result.require()


# --------------------------------------------------------------------- triage


def run_triage(world, *payloads, text: str = "Dan called, van 3 won't start") -> TriageResult:
    provider = ScriptedLLMProvider.returning(*payloads)
    with dispatch("d-test"):
        return triage(provider, world, text, on_date=WEEK_START, tz=TZ)


def test_triage_builds_typed_events(world):
    result = run_triage(
        world,
        {
            "events": [
                {
                    "kind": "van-unavailable",
                    "target": "van-3",
                    "at": "10:40",
                    "reason": "wont start",
                }
            ],
            "question": "",
            "summary": "Van 3 out from 10:40.",
        },
    )
    assert len(result.events) == 1
    event = result.events[0]
    assert event.type == "van_unavailable"
    assert event.van_id == "van-3"
    assert event.from_time.astimezone(TZ).hour == 10
    assert not result.needs_dispatcher


def test_triage_catches_an_invented_id(world):
    """A hallucinated van is caught by a dictionary lookup. Nothing about the model's
    confidence would have caught it."""
    result = run_triage(
        world,
        {
            "events": [{"kind": "van-unavailable", "target": "van-7"}],
            "question": "",
            "summary": "x",
        },
    )
    assert result.events == ()
    assert result.unknown_targets == ("van-7",)
    assert "van-7" in result.question
    assert result.needs_dispatcher


def test_triage_never_invents_a_missing_required_detail(world):
    """An over-run needs a number. Guessing it would put a fabricated fact into an
    append-only log, where it would propagate into every plan that follows."""
    result = run_triage(
        world,
        {
            "events": [{"kind": "job-overran", "target": "j-403"}],
            "question": "",
            "summary": "Patel long.",
        },
        text="Patel job running long",
    )
    assert result.events == ()
    assert result.rejected
    assert result.needs_dispatcher


def test_triage_passes_through_a_clarifying_question(world):
    result = run_triage(
        world,
        {"events": [], "question": "Which van is Dan on?", "summary": "Possible fault."},
    )
    assert result.question == "Which van is Dan on?"
    assert result.needs_dispatcher


def test_triage_grounds_the_prompt_in_the_real_roster(world):
    """The model picks from a list instead of inventing from memory - the cheapest way
    to make a small model reliable at this."""
    provider = ScriptedLLMProvider.returning({"events": [], "question": "", "summary": ""})
    with dispatch("d-test"):
        triage(provider, world, "something happened", on_date=WEEK_START, tz=TZ)
    system = provider.requests[0].system
    assert "van-3" in system
    assert "w-marcus (Marcus)" in system
    assert "j-402" in system


def test_triage_escalates_when_the_model_cannot_answer(world):
    result = run_triage(world, *([{"events": [{"kind": "nonsense"}]}] * 3))
    assert result.events == ()
    assert result.extraction.escalated
    assert result.needs_dispatcher


def test_triage_events_carry_the_dispatch_id(world):
    with dispatch("d-correlated"):
        provider = ScriptedLLMProvider.returning(
            {
                "events": [{"kind": "van-unavailable", "target": "van-3"}],
                "question": "",
                "summary": "",
            }
        )
        result = triage(provider, world, "van 3 down", on_date=WEEK_START, tz=TZ)
    assert result.events[0].dispatch_id == "d-correlated"


# ---------------------------------------------------------------- coordinator


@pytest.fixture
def repair_options(world, travel, business_params):
    from krama.domain.models import PlanVersion
    from krama.scheduler.day_planner import SolveParams
    from krama.scheduler.horizon import HorizonParams, plan_horizon
    from krama.scheduler.repair import repair_plan

    params = SolveParams.from_business(business_params, TZ)
    horizon_params = HorizonParams.from_business(business_params)
    base = plan_horizon(
        world=world,
        travel=travel,
        start=WEEK_START,
        params=params,
        horizon_params=horizon_params,
    )
    baseline = PlanVersion(
        id="committed",
        created_at=_at(0, 6),
        horizon_start=WEEK_START,
        horizon_end=date.fromordinal(WEEK_START.toordinal() + 4),
        routes=base.routes,
    )
    world.van_outages["van-1"] = [
        Unavailability(from_time=_at(0, 10, 40), until_time=None, reason="wont start")
    ]
    return repair_plan(
        world=world,
        travel=travel,
        baseline=baseline,
        start=WEEK_START,
        params=params,
        horizon_params=horizon_params,
    )


@pytest.fixture
def business_params():
    from krama.config import BusinessParams

    return BusinessParams.load()


@pytest.fixture
def policy(business_params):
    return AutonomyPolicy.from_business(business_params)


def test_coordinator_picks_a_named_strategy(repair_options, policy):
    provider = ScriptedLLMProvider.returning(
        {
            "chosen_strategy": "most_jobs",
            "rationale": "two more jobs, nobody to call",
            "confidence": "high",
            "wants_human_review": False,
        }
    )
    decision = choose_repair(provider, repair_options, policy)
    assert decision.candidate is not None
    assert decision.candidate.strategy.name == "most_jobs"
    assert decision.rationale


def test_an_invented_strategy_cannot_survive_validation(repair_options, policy):
    """The answer is an enum, so a strategy the engine never offered fails to parse
    rather than propagating into a plan."""
    provider = ScriptedLLMProvider.always(
        {
            "chosen_strategy": "wing_it",
            "rationale": "trust me",
            "confidence": "high",
            "wants_human_review": False,
        }
    )
    decision = choose_repair(provider, repair_options, policy, max_attempts=2)
    assert decision.extraction.escalated
    # The engine's own recommendation stands rather than nothing happening.
    assert decision.candidate is not None
    assert "could not choose" in decision.rationale


def test_the_model_cannot_talk_its_way_past_the_autonomy_policy(
    repair_options, policy, monkeypatch
):
    """The deterministic rules decide. A model able to authorise a customer-visible
    change silently would make the whole autonomy layer decorative."""
    from krama.domain import autonomy as autonomy_module

    forced = autonomy_module.AutonomyDecision(
        autonomy_module.Decision.ESCALATE, ("customer would need telling",)
    )
    monkeypatch.setattr(autonomy_module, "decide", lambda *a, **k: forced)
    monkeypatch.setattr("krama.agents.coordinator.decide", lambda *a, **k: forced)

    provider = ScriptedLLMProvider.returning(
        {
            "chosen_strategy": "most_jobs",
            "rationale": "fine",
            "confidence": "high",
            "wants_human_review": False,
        }
    )
    decision = choose_repair(provider, repair_options, policy)
    assert decision.needs_human


def test_the_model_may_ask_for_more_review_than_the_rules_require(repair_options, policy):
    """It can only ever raise the bar, never lower it."""
    provider = ScriptedLLMProvider.returning(
        {
            "chosen_strategy": "least_disruption",
            "rationale": "options are close",
            "confidence": "low",
            "wants_human_review": True,
        }
    )
    decision = choose_repair(provider, repair_options, policy)
    assert decision.needs_human


def test_the_coordinator_is_told_the_costed_facts_not_the_plan(repair_options, policy):
    """It ranks a short list. Handing it a whole schedule would invite it to reason
    over the wrong part of it."""
    captured: list[str] = []

    def handler(request):
        captured.append(request.messages[-1].text)
        return LLMResponse(
            structured={
                "chosen_strategy": "most_jobs",
                "rationale": "r",
                "confidence": "high",
                "wants_human_review": False,
            },
            stop_reason="tool_use",
            usage=LLMUsage(),
        )

    choose_repair(CallbackLLMProvider(handler=handler), repair_options, policy)
    prompt = captured[0]
    assert "jobs served" in prompt
    assert "customers who must be phoned" in prompt
    assert "arrival" not in prompt.lower()


def test_a_reschedule_needs_a_day_not_a_time(world):
    """ "Change James's booking to Tuesday sometime" is a complete instruction - a
    fresh booking never demands a time, and a reschedule must not either. The day
    becomes a whole-working-day window and the planner picks the slot."""
    from datetime import date as _date

    tuesday = _date.fromordinal(WEEK_START.toordinal() + 1)
    result = run_triage(
        world,
        {
            "events": [
                {
                    "kind": "customer-rescheduled",
                    "target": "j-401",
                    "day": tuesday.isoformat(),
                }
            ],
            "question": "",
            "summary": "Henderson moves to Tuesday.",
        },
        text="change the Henderson booking to tuesday sometime",
    )
    from krama.domain.events import CustomerRescheduled

    assert not result.needs_dispatcher, result.question
    event = result.events[0]
    assert isinstance(event, CustomerRescheduled)
    assert event.job_id == "j-401"
    (window,) = event.new_windows
    local_start = window.start.astimezone(TZ)
    local_end = window.end.astimezone(TZ)
    assert local_start.date() == tuesday and local_start.hour == 8
    assert local_end.date() == tuesday and local_end.hour == 17

    # A stated time narrows it to the usual two-hour arrival window.
    timed = run_triage(
        world,
        {
            "events": [
                {
                    "kind": "customer-rescheduled",
                    "target": "j-401",
                    "day": tuesday.isoformat(),
                    "at": "14:00",
                }
            ],
            "question": "",
            "summary": "Henderson moves to Tuesday at two.",
        },
        text="henderson wants tuesday at 2pm instead",
    )
    timed_event = timed.events[0]
    assert isinstance(timed_event, CustomerRescheduled)
    (window,) = timed_event.new_windows
    assert window.start.astimezone(TZ).hour == 14
    assert window.end.astimezone(TZ).hour == 16


def test_a_correction_clears_the_old_absence_and_records_the_new(world):
    """ "Actually out 11 to 1" must not merely restore (forgetting the absence) or
    merely record (stacking 11-1 on the stale 11-12). It is both: clear, then
    record the corrected hours."""
    result = run_triage(
        world,
        {
            "events": [
                {"kind": "worker-restored", "target": "w-priya"},
                {
                    "kind": "worker-unavailable",
                    "target": "w-priya",
                    "at": "11:00",
                    "until": "13:00",
                },
            ],
            "question": "",
            "summary": "Priya corrected to 11-13.",
        },
        text="just asked Priya - she's actually out 11 to 1, fine the rest of the day",
    )
    from krama.domain.events import WorkerUnavailable

    assert not result.needs_dispatcher
    kinds = [e.type for e in result.events]
    assert kinds == ["worker_restored", "worker_unavailable"]
    outage = result.events[1]
    assert isinstance(outage, WorkerUnavailable)
    assert outage.from_time.astimezone(TZ).hour == 11
    assert outage.until_time is not None and outage.until_time.astimezone(TZ).hour == 13


def test_a_batch_replays_in_the_order_it_was_said(world):
    """ "Restore, then the corrected hours" came back shuffled: every event in the
    batch carried the same recorded_at, so the fold's tiebreak was the random
    event id - and the stale absence outlived its own correction. Each event now
    records its own learned-instant, and occurred_at stays the fact's own time."""

    from krama.domain.state import _apply

    result = run_triage(
        world,
        {
            "events": [
                {"kind": "worker-restored", "target": "w-priya"},
                {
                    "kind": "worker-unavailable",
                    "target": "w-priya",
                    "at": "11:00",
                    "until": "13:00",
                },
            ],
            "question": "",
            "summary": "Priya corrected to 11-13.",
        },
        text="she's actually out 11 to 1, fine the rest of the day",
    )
    first, second = result.events
    assert first.recorded_at < second.recorded_at, "said first, learned first"

    # Replayed over a stale 11-12 absence, the correction leaves exactly 11-13.
    from krama.domain.state import Unavailability

    world.worker_outages["w-priya"] = [
        Unavailability(from_time=_at(0, 11), until_time=_at(0, 12), reason="stale")
    ]
    for event in sorted(result.events, key=lambda e: (e.recorded_at, e.event_id)):
        _apply(world, event)
    remaining = world.worker_outages["w-priya"]
    assert [(o.from_time.hour, (o.until_time or o.from_time).hour) for o in remaining] == [
        (11, 13)
    ], remaining


def test_a_mis_stamped_past_event_cannot_outrank_a_later_fix():
    """One batch was once stamped with the outage's own FUTURE time (Monday
    11:00, recorded on Saturday). Under recorded_at ordering those events
    replayed after every later correction, forever - the stale absence kept
    resurrecting. The fold replays the log in append order now: what was
    appended later wins, whatever its stamp claims."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from krama.cli.events import build_event
    from krama.domain.state import fold
    from krama.fixtures.sample_business import seed_events

    tz = ZoneInfo("America/Chicago")
    monday_11 = datetime(2026, 10, 12, 11, 0, tzinfo=tz)
    saturday = datetime(2026, 10, 10, 17, 30, tzinfo=tz)

    poisoned = build_event(
        "worker-unavailable",
        "w-priya",
        at=monday_11,  # stamped with the fact's own future time - the old bug
        dispatch_id="d-old",
        until=monday_11 + timedelta(hours=1),
    )
    fix_restore = build_event(
        "worker-restored",
        "w-priya",
        at=saturday,
        dispatch_id="d-fix",
        recorded_at=saturday,
    )
    fix_outage = build_event(
        "worker-unavailable",
        "w-priya",
        at=monday_11,
        dispatch_id="d-fix",
        recorded_at=saturday + timedelta(microseconds=1),
        until=monday_11 + timedelta(hours=2),
    )

    world = fold([*seed_events(), poisoned, fix_restore, fix_outage])
    spans = [
        (o.from_time.astimezone(tz).hour, (o.until_time or o.from_time).astimezone(tz).hour)
        for o in world.worker_outages.get("w-priya", [])
        if o.from_time.astimezone(tz).date().isoformat() == "2026-10-12"
    ]
    assert spans == [(11, 13)], f"the correction stands, the stale absence stays dead: {spans}"


def test_the_roster_tells_the_model_which_day_each_booking_is(world):
    """ "Move all the Monday bookings" was unanswerable: the job list carried
    names and ids but no dates, so the model could only ask which bookings
    Monday's were. The list says now."""
    from krama.agents.triage import _roster

    listing = _roster(world)
    assert "Mon 2026-" in listing or "Mon 2025-" in listing, listing[:400]
    # every active job wears a day name
    import re

    jobs_line = next(line for line in listing.splitlines() if line.startswith("Jobs:"))
    entries = jobs_line.count("(")
    dated = len(re.findall(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-", jobs_line))
    assert dated == entries, jobs_line[:300]
