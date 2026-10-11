"""Which kind of note is this?

The board used to ask the dispatcher. Two text boxes sat in the same column looking
identical - one booked jobs, one recorded disruptions - and the first person to use it
typed "Dan called, van 3 won't start" into the booking box and got back a customer
named Dan who wanted auto glass fitted.

That is not a labelling problem to solve with better labels. A dispatcher writing down
what they just heard should not have to know which of two agents wants it; working out
what a sentence *is* before acting on it is interpretation, which is the half of this
system that agents are for. So there is one box, and this decides.

Deliberately the smallest possible task: one of two labels, plus a sentence saying why.
The project's own argument for a small model is that classification sits well within
its range while open-ended planning does not, and this is that argument's cleanest
instance. Nothing here decides what to *do* - it only picks which specialist reads the
note next, and the dispatcher can override it with one click.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from krama.agents.llm.base import LLMProvider
from krama.agents.structured import Example, Extraction, extract
from krama.obs.tracing import span


class NoteKind(StrEnum):
    """What a note turned out to be about."""

    #: Somebody wants work done. Ends in a job on the schedule.
    BOOKING = "booking"
    #: Something broke, or somebody is unavailable. Ends in a repaired plan.
    DISRUPTION = "disruption"


class NoteRouting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["booking", "disruption"] = Field(
        description=(
            "booking when someone wants work done. "
            "disruption when something has gone wrong with work already planned."
        )
    )
    why: str = Field(description="One short sentence. What decided it.")


SYSTEM = """\
You sort notes a dispatcher at a glass-fitting business types while on the phone.

Answer "booking" when someone wants work that is not yet on the schedule: a customer
describing damage, a new job, a quote request.

Answer "disruption" when something has changed about work that is already planned: a
van broken down, a fitter off sick, traffic, a job overrunning, a part not arriving, a
customer moving or cancelling an appointment they already have.

The person calling is often an employee in the second case and a customer in the first.
A name on its own decides nothing - but the roster below does: a note ABOUT one of the
business's own fitters or vans - their hours, health, availability, a correction to
when they are out - is a disruption, never a booking. Our own fitter is not a customer.
"""


EXAMPLES: tuple[Example, ...] = (
    # A reschedule is the case that decides the boundary, so it is worked rather than
    # described. It concerns a job that already exists, so it belongs to the side that
    # changes plans - and it fails safely there. Sent to booking instead, the intake
    # agent would happily invent a second job for a customer who already has one.
    Example(
        text="Mrs Alvarez rang, wants to move her Thursday appointment to Friday",
        output=NoteRouting(
            kind="disruption", why="An appointment that already exists is changing."
        ),
    ),
    Example(
        text="Dan called, van 3 won't start, he's stuck at the Henderson site",
        output=NoteRouting(
            kind="disruption", why="A van is off the road, so today's plan has to change."
        ),
    ),
    Example(
        text=(
            "hi this is Maria from Nguyen Glass on 2nd ave, storefront pane got smashed "
            "overnight, we open at 11 so any time before that works"
        ),
        output=NoteRouting(kind="booking", why="A customer with damage wants it fixed."),
    ),
    # The case that was misread in the wild: a correction about a FITTER's hours
    # became a customer named Priya wanting a window fitted 11-to-1.
    Example(
        text=(
            "hey I just asked Priya, she's actually out from 11 to 1 on Monday, "
            "correction, she's available the rest of the day including overtime"
        ),
        output=NoteRouting(
            kind="disruption",
            why="A correction to one of our own fitters' availability.",
        ),
    ),
)


def route_note(provider: LLMProvider, text: str, roster: str = "") -> Extraction[NoteRouting]:
    """Decide which specialist should read this note.

    ``roster`` names the business's own fitters and vans, so "Priya is out 11 to 1"
    reads as a crew correction instead of a customer called Priya wanting a window.
    """
    system = SYSTEM if not roster else f"{SYSTEM}\nThe business's own roster: {roster}\n"
    with span("agent.router", chars=len(text)):
        return extract(provider, NoteRouting, system=system, text=text, examples=EXAMPLES)
