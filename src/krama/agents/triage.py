"""Triage: a sentence someone typed, turned into typed events.

"Dan called, van 3 won't start, he's stuck at the Henderson site" is exactly the kind
of input a solver cannot take and a person produces constantly. Turning it into
``VanUnavailable(van_id="van-3", from_time=...)`` is interpretation, which is what a
model is for.

Everything after the interpretation is checked. The model chooses from an enumerated
set of event kinds and must name ids that exist; a hallucinated "van-7" is caught by
a dictionary lookup, not by hoping. The world's actual roster goes into the prompt as
grounding, so the model is picking from a list rather than inventing from memory -
the single cheapest way to make a small model reliable at this.

When something material is genuinely ambiguous - "can Dan keep working if someone
collects him?" - it asks one question instead of guessing. Guessing here is expensive:
an event is a fact in an append-only log, and a wrong one propagates into every plan
that follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from krama.agents.llm.base import LLMProvider
from krama.agents.structured import Example, Extraction, extract
from krama.cli.events import EventArgumentError, build_event
from krama.domain.events import Event
from krama.domain.state import WorldState
from krama.obs.correlation import require_dispatch_id
from krama.obs.tracing import record, span

EventKind = Literal[
    "van-unavailable",
    "van-restored",
    "worker-unavailable",
    "worker-restored",
    "job-dispatched",
    "job-cancelled",
    "job-overran",
    "job-completed",
    "customer-rescheduled",
    "traffic-delay",
]


class TriagedEvent(BaseModel):
    """One event the model believes was described."""

    model_config = ConfigDict(extra="forbid")

    kind: EventKind
    target: str = Field(
        default="",
        description="The van, worker or job id this concerns. Must be an id from the roster.",
    )
    at: str = Field(default="", description="Time it happened, HH:MM, 24-hour.")
    until: str = Field(default="", description="Time it ends, HH:MM, if stated.")
    day: str = Field(
        default="",
        description=(
            "For customer-rescheduled: the new DATE, YYYY-MM-DD. Resolve weekday "
            "names against today's date. Empty for every other kind."
        ),
    )
    minutes: int | None = Field(
        default=None, description="Minutes over-run, or actual duration, when stated."
    )
    multiplier: float | None = Field(
        default=None, description="Traffic slowdown factor, e.g. 1.8 for 80% slower."
    )
    reason: str = Field(default="", description="What was said, in a few words.")


class TriageOutcome(BaseModel):
    """What the model extracted from the message."""

    model_config = ConfigDict(extra="forbid")

    events: list[TriagedEvent] = Field(
        default_factory=list, description="Every event clearly described. May be empty."
    )
    question: str = Field(
        default="",
        description=(
            "One question for the dispatcher, only when something material is ambiguous "
            "and guessing would be wrong. Empty otherwise."
        ),
    )
    summary: str = Field(default="", description="One line restating what happened.")


SYSTEM = """\
You convert a dispatcher's note about a glass-fitting business into typed events.

Rules:
- Use only the event kinds and ids listed below. Never invent an id.
- Extract only what the message actually says. Do not infer a time that was not given.
- Ask a question only when you cannot record an event without the answer: no id to
  attach it to, or a required number like an over-run's minutes. If the event can be
  recorded correctly as it stands, leave `question` empty. Colour and context - whether
  a tow is en route, how bad the damage looks - do not change what gets recorded, and
  asking about them buries the questions that do.
- A plain statement of fact - "van 3 won't start", "Priya is out sick", "traffic is
  terrible on the 287" - is complete as spoken. Record it; never ask about one.
- Times are 24-hour HH:MM. "half ten" is 10:30, "quarter to noon" is 11:45.
- job-overran needs `minutes`. traffic-delay needs `multiplier`.
- If no time is given, leave `at` empty. It defaults to now, which is almost always
  what the dispatcher meant, so there is no need to ask.
- customer-rescheduled: an existing booking moves at the customer's request. Put the
  new day in `day` (YYYY-MM-DD); a stated time goes in `at` (and `until` if an end
  was named). A day with NO time is complete - "sometime Tuesday" reschedules to
  Tuesday and the planner picks the exact slot, the same way it does for a fresh
  booking. Never ask for a time merely because none was given; ask only when even
  the day is unclear.
"""


@dataclass(frozen=True, slots=True)
class TriageResult:
    events: tuple[Event, ...]
    question: str
    summary: str
    #: Ids the model named that do not exist. Caught by lookup, never by trust.
    unknown_targets: tuple[str, ...]
    #: Events the model described that could not be built, with the reason.
    rejected: tuple[str, ...]
    extraction: Extraction[TriageOutcome]

    @property
    def ok(self) -> bool:
        return self.extraction.ok and not self.rejected

    @property
    def needs_dispatcher(self) -> bool:
        """Anything a human has to resolve before this note becomes fact.

        Rejected events count. Something was clearly described and could not be
        recorded, and silently dropping it would lose the very information the
        dispatcher took the trouble to type.
        """
        return (
            bool(self.question)
            or bool(self.unknown_targets)
            or bool(self.rejected)
            or self.extraction.escalated
        )


def _roster(world: WorldState) -> str:
    """The actual ids, so the model picks from a list instead of inventing from memory."""
    vans = ", ".join(sorted(world.vans))
    workers = ", ".join(
        f"{w.id} ({w.name})" for w in sorted(world.workers.values(), key=lambda w: w.id)
    )
    jobs = ", ".join(f"{j.id} ({j.customer_name})" for j in world.active_jobs())
    return (
        f"Vans: {vans}\nWorkers: {workers}\nJobs: {jobs}\n"
        f"Event kinds: van-unavailable, van-restored, worker-unavailable, "
        f"worker-restored, job-dispatched, job-cancelled, job-overran, job-completed, "
        f"customer-rescheduled, traffic-delay"
    )


EXAMPLES: tuple[Example, ...] = (
    Example(
        text="Priya's out sick today",
        output=TriageOutcome(
            events=[TriagedEvent(kind="worker-unavailable", target="w-priya", reason="sick")],
            summary="Priya unavailable today, sick.",
        ),
    ),
    Example(
        text="the Patel job is running about two hours long",
        output=TriageOutcome(
            events=[TriagedEvent(kind="job-overran", target="j-403", minutes=120)],
            summary="Patel shower door running 120 minutes over.",
        ),
    ),
    Example(
        text="change the Patel booking to Tuesday sometime",
        output=TriageOutcome(
            events=[TriagedEvent(kind="customer-rescheduled", target="j-403", day="2026-01-13")],
            summary="Patel moves to Tuesday; the planner will pick the slot.",
        ),
    ),
    Example(
        text="Dan says the van's making a noise, might need looking at",
        output=TriageOutcome(
            events=[],
            question=(
                "Is the van out of service now, or still driveable? And which van is Dan on?"
            ),
            summary="Possible van fault reported, not yet actionable.",
        ),
    ),
)


def _clock_or(clock: str, fallback: time) -> time:
    try:
        return datetime.strptime(clock, "%H:%M").time()
    except ValueError:
        return fallback


def _known_target(world: WorldState, kind: str, target: str) -> bool:
    if kind.startswith("van-"):
        return target in world.vans
    if kind.startswith("worker-"):
        return target in world.workers
    if kind.startswith("job-") or kind == "customer-rescheduled":
        return target in world.jobs
    return True


def triage(
    provider: LLMProvider,
    world: WorldState,
    text: str,
    *,
    on_date: date,
    tz: tzinfo,
    now: datetime | None = None,
    max_attempts: int = 3,
) -> TriageResult:
    """Turn a dispatcher's note into events the engine can act on.

    ``now`` is when the note was typed, and it is the default for any event whose time
    was not stated. That default used to be 08:00, which is wrong in a consequential
    direction: a van reported off the road at two in the afternoon was recorded as
    unavailable since breakfast, retroactively invalidating the work it had already
    done that morning. The model kept asking "when did this happen?" - which read as
    over-caution until it became clear the answer genuinely mattered and our silent
    assumption was wrong.
    """
    with span("agent.triage", chars=len(text)) as active:
        extraction = extract(
            provider,
            TriageOutcome,
            system=(
                f"{SYSTEM}\nToday is {on_date:%A}, {on_date.isoformat()}."
                f"\nRoster:\n{_roster(world)}"
            ),
            text=text,
            examples=EXAMPLES,
            max_attempts=max_attempts,
            schema_description="The events described in the dispatcher's note.",
        )

        if extraction.value is None:
            record(events=0, escalated=True, provider_failed=extraction.provider_failed)
            active.set_attribute(
                "outcome", "provider_failed" if extraction.provider_failed else "escalated"
            )
            # Blaming the dispatcher's phrasing for an unreachable model would send
            # them rewording a sentence that was never the problem.
            question = (
                f"The assistant is unavailable: {extraction.errors[0]}"
                if extraction.provider_failed and extraction.errors
                else "I could not read that. Can you rephrase what happened?"
            )
            return TriageResult(
                events=(),
                question=question,
                summary="",
                unknown_targets=(),
                rejected=(),
                extraction=extraction,
            )

        outcome = extraction.value
        dispatch_id = require_dispatch_id()
        built: list[Event] = []
        unknown: list[str] = []
        rejected: list[str] = []

        reported_at = (now or world.as_of).astimezone(tz)

        def moment(clock: str, fallback: time | None = None) -> datetime:
            """A stated time, or when the note was typed."""
            if not clock:
                return (
                    datetime.combine(on_date, fallback, tzinfo=tz)
                    if fallback is not None
                    else reported_at
                )
            try:
                parsed = datetime.strptime(clock, "%H:%M").time()
            except ValueError:
                return reported_at
            return datetime.combine(on_date, parsed, tzinfo=tz)

        for item in outcome.events:
            if item.target and not _known_target(world, item.kind, item.target):
                # The model named something that does not exist. A dictionary lookup
                # catches this; nothing about the model's confidence would have.
                unknown.append(item.target)
                continue
            try:
                if item.kind == "customer-rescheduled":
                    # "Tuesday sometime" is a whole-day window; the solver picks the
                    # slot exactly as it would for a fresh booking. A stated time
                    # becomes the usual two-hour arrival window.
                    move_day = date.fromisoformat(item.day) if item.day else on_date
                    opens = (
                        datetime.combine(move_day, _clock_or(item.at, time(8, 0)), tzinfo=tz)
                        if item.at
                        else datetime.combine(move_day, time(8, 0), tzinfo=tz)
                    )
                    if item.until:
                        closes = datetime.combine(
                            move_day, _clock_or(item.until, time(17, 0)), tzinfo=tz
                        )
                    elif item.at:
                        closes = opens + timedelta(minutes=120)
                    else:
                        closes = datetime.combine(move_day, time(17, 0), tzinfo=tz)
                    built.append(
                        build_event(
                            item.kind,
                            item.target or None,
                            at=reported_at,
                            dispatch_id=dispatch_id,
                            window_start=opens,
                            window_end=closes,
                            reason=item.reason,
                        )
                    )
                    continue
                built.append(
                    build_event(
                        item.kind,
                        item.target or None,
                        at=moment(item.at),
                        dispatch_id=dispatch_id,
                        until=moment(item.until) if item.until else None,
                        minutes=item.minutes,
                        multiplier=item.multiplier,
                        reason=item.reason,
                    )
                )
            except (EventArgumentError, ValueError) as exc:
                rejected.append(f"{item.kind} ({item.target or 'no target'}): {exc}")

        question = outcome.question
        if unknown and not question:
            question = (
                f"I could not find {', '.join(sorted(set(unknown)))} on the roster. "
                "Which van, worker or job did you mean?"
            )
        elif rejected and not question:
            # Something was described but is missing a detail the event type requires.
            # Asking beats dropping it, and beats inventing the missing number.
            question = f"I need a bit more to record that: {rejected[0]}"

        record(
            events=len(built),
            repairs=extraction.repairs,
            unknown_targets=len(unknown),
            rejected=len(rejected),
            asked_question=bool(question),
        )
        active.set_attribute("outcome", "ok" if built else "no_events")

        return TriageResult(
            events=tuple(built),
            question=question,
            summary=outcome.summary,
            unknown_targets=tuple(unknown),
            rejected=tuple(rejected),
            extraction=extraction,
        )
