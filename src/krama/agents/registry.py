"""Agents as A2A services, with the cards that advertise them.

Each agent is wrapped as a handler taking a task and returning one. Locally that is a
function call; deployed it is an HTTP service behind the same contract. Nothing about
an agent changes between the two, which is the point of defining the protocol before
the deployment rather than after.

The task lifecycle earns its keep immediately: triage that needs a clarifying question
returns ``input-required`` rather than guessing, and the caller can see the difference
between "here are your events" and "I need to know which van first".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, tzinfo

from krama.agents.a2a.transport import Handler, InProcessTransport
from krama.agents.a2a.types import (
    A2AMessage,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    DataPart,
    Role,
    Task,
    TaskState,
    TaskStatus,
)
from krama.agents.coordinator import choose_repair
from krama.agents.llm.base import LLMProvider
from krama.agents.triage import triage
from krama.domain.autonomy import AutonomyPolicy
from krama.domain.state import WorldState
from krama.scheduler.repair import RepairOptions

TRIAGE_SKILL = "disruption-triage"
COORDINATOR_SKILL = "repair-selection"


TRIAGE_CARD = AgentCard(
    name="triage",
    description=(
        "Turns a dispatcher's note about a disruption into typed events the "
        "scheduling engine can act on, or asks one question when something material "
        "is ambiguous."
    ),
    capabilities=AgentCapabilities(state_transition_history=True),
    skills=[
        AgentSkill(
            id=TRIAGE_SKILL,
            name="Disruption triage",
            description="Free text to typed domain events, grounded in the live roster.",
            tags=["extraction", "events"],
            examples=[
                "Dan called, van 3 won't start",
                "Priya's out sick today",
                "the Patel job is running two hours long",
            ],
        )
    ],
)

COORDINATOR_CARD = AgentCard(
    name="coordinator",
    description=(
        "Chooses between repair options the engine has already costed, favouring kept "
        "promises over extra jobs. Does not plan or set times."
    ),
    skills=[
        AgentSkill(
            id=COORDINATOR_SKILL,
            name="Repair selection",
            description="Ranks priced repair candidates and explains the trade-off.",
            tags=["ranking", "policy"],
        )
    ],
)


@dataclass(frozen=True, slots=True)
class TriageContext:
    world: WorldState
    on_date: date
    tz: tzinfo
    #: Wall time at the moment of the note. Without it triage falls back to
    #: world.as_of - the LAST EVENT'S clock - which is the exact stale-stamp trap
    #: triage's own docstring warns about, quietly reintroduced by this wrapper.
    now: datetime | None = None


def _completed(task: Task, artifact: Artifact, text: str) -> Task:
    message = A2AMessage.text(Role.AGENT, text)
    return task.model_copy(
        update={
            "status": TaskStatus(state=TaskState.COMPLETED, message=message),
            "history": [*task.history, message],
            "artifacts": [*task.artifacts, artifact],
        }
    )


def _needs_input(task: Task, question: str, artifact: Artifact | None = None) -> Task:
    """The state that makes "ask rather than guess" expressible across an agent boundary."""
    message = A2AMessage.text(Role.AGENT, question)
    return task.model_copy(
        update={
            "status": TaskStatus(state=TaskState.INPUT_REQUIRED, message=message),
            "history": [*task.history, message],
            "artifacts": [*task.artifacts, artifact] if artifact else task.artifacts,
        }
    )


def triage_handler(provider: LLMProvider, context: TriageContext) -> Handler:
    def handle(task: Task) -> Task:
        note = task.history[-1].text_content if task.history else ""
        result = triage(
            provider,
            context.world,
            note,
            on_date=context.on_date,
            tz=context.tz,
            now=context.now,
        )
        artifact = Artifact(
            name="events",
            parts=[
                DataPart(
                    data={
                        "events": [e.model_dump(mode="json") for e in result.events],
                        "summary": result.summary,
                        "unknown_targets": list(result.unknown_targets),
                        "rejected": list(result.rejected),
                        "repairs": result.extraction.repairs,
                    }
                )
            ],
        )
        if result.needs_dispatcher:
            return _needs_input(
                task, result.question or "I need a dispatcher to look at this.", artifact
            )
        return _completed(task, artifact, result.summary or "Recorded.")

    return handle


def coordinator_handler(
    provider: LLMProvider, options: RepairOptions, policy: AutonomyPolicy
) -> Handler:
    def handle(task: Task) -> Task:
        decision = choose_repair(provider, options, policy)
        artifact = Artifact(
            name="repair-choice",
            parts=[
                DataPart(
                    data={
                        "strategy": (
                            decision.candidate.strategy.name if decision.candidate else ""
                        ),
                        "rationale": decision.rationale,
                        "confidence": decision.confidence,
                        "needs_human": decision.needs_human,
                        "autonomy": (decision.autonomy.decision.value if decision.autonomy else ""),
                        "overrode_default": decision.overrode_default,
                    }
                )
            ],
        )
        if decision.needs_human:
            reasons = "; ".join(decision.autonomy.reasons) if decision.autonomy else "no decision"
            return _needs_input(task, f"{decision.rationale}. Needs review: {reasons}", artifact)
        return _completed(task, artifact, decision.rationale)

    return handle


def build_transport(
    provider: LLMProvider,
    *,
    triage_context: TriageContext | None = None,
    repair_options: RepairOptions | None = None,
    policy: AutonomyPolicy | None = None,
) -> InProcessTransport:
    """Register whichever agents this caller has the context to run."""
    transport = InProcessTransport()
    if triage_context is not None:
        transport.register(TRIAGE_CARD, triage_handler(provider, triage_context))
    if repair_options is not None and policy is not None:
        transport.register(COORDINATOR_CARD, coordinator_handler(provider, repair_options, policy))
    return transport
