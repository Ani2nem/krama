"""Intake: a phone call, turned into a job the engine can price.

The headline feature, and the clearest case for having an agent at all. A caller says
"my front window's smashed, I'm around Tuesday but I'd have to take the morning off" -
and three quite different things need to happen to that sentence.

**Classify.** Which service is this? A model does that well.

**Estimate.** How long will it take, what certifications, what parts? A model does
that *badly*, so it does not: :mod:`krama.domain.catalog` answers from the
service type. A schedule built on an invented duration is wrong in a way no invariant
check can catch, because every arrival time is internally consistent with the fiction.

**Price the promise.** "I'd have to take the morning off" means this slot is expensive
to move. That is the single clearest justification for a model in this pipeline - no
dropdown captures it, and the solver has no way to infer it. But the model only
identifies *which* signal was expressed, from a fixed list. The dollar figure comes
from configuration, because asking a language model to price goodwill produces a
confident number with nothing behind it.

What is missing matters as much as what was captured. A dispatcher mid-call needs
"ask for a callback number" while the customer is still on the line, not a silently
half-filled form discovered later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from krama.agents.llm.base import LLMProvider
from krama.agents.structured import Example, Extraction, extract
from krama.config import BusinessParams
from krama.domain.catalog import Estimate, describe_for_prompt, lookup
from krama.domain.enums import GlassType, Priority, PropertyType, ServiceType
from krama.domain.models import GlassSpec, Job, Location, Material, Provenance
from krama.geocoding import GeocodeError, Geocoder, for_service_area
from krama.obs.tracing import record, span


class CommitmentSignal(StrEnum):
    """What a caller said they had arranged around this appointment.

    A closed list on purpose. The model recognises which of these was expressed; the
    cost of each is a business parameter, not the model's to invent.
    """

    TIME_OFF_WORK = "time_off_work"
    WAITING_IN = "waiting_in"
    ARRANGED_CHILDCARE = "arranged_childcare"
    BUSINESS_CLOSED = "business_closed"
    ALREADY_RESCHEDULED = "already_rescheduled"


#: Things a model writes when it means "nothing".
#:
#: Asked for a field it was not told, a model would rather answer than leave a blank,
#: so it fills in "N/A" or "unknown" - and a string like that is perfectly truthy. A
#: caller who said "callback" and then never gave the number came back with phone set
#: to "N/A", which counted as answered: not in the missing list, not in "still to ask",
#: shown on screen as a value. The dispatcher hangs up without the number.
#:
#: Normalised on the model rather than at the one place that noticed, because every
#: consumer downstream has the same problem and none of them should have to know.
_NOT_AN_ANSWER = frozenset(
    {
        "",
        "-",
        "--",
        "n/a",
        "na",
        "none",
        "null",
        "nil",
        "unknown",
        "unspecified",
        "not given",
        "not provided",
        "not stated",
        "not available",
        "missing",
        "tbd",
        "tba",
        "?",
    }
)


def stated(value: str) -> str:
    """The value, or empty if the model was really saying it did not know."""
    return "" if value.strip().lower().strip(".") in _NOT_AN_ANSWER else value.strip()


def phone_or_blank(value: str) -> str:
    """The value if somebody could actually be rung on it, otherwise nothing.

    Two lessons, in order. Told to produce a phone and given no number, the model
    writes the company name in - so anything without enough digits is blanked. Then a
    nine-digit number sailed through a seven-digit floor, was accepted, displayed, and
    would have been dialled: a digit count that low verifies nothing.

    A US number is ten digits, eleven with a leading 1, and this business is in Texas.
    An international caller (+44...) keeps their number: the + is an explicit claim,
    not a truncation.
    """
    digits = "".join(ch for ch in value if ch.isdigit())
    if value.strip().startswith("+") and len(digits) >= 10:
        return value
    if len(digits) == 10 or (len(digits) == 11 and digits[0] == "1"):
        return value
    return ""


def email_or_blank(value: str) -> str:
    """Same argument, same reason. An address without an @ is not one."""
    local, _, domain = value.partition("@")
    return value if local and "." in domain else ""


class CallExtraction(BaseModel):
    """What the model heard. Nothing here is a scheduling decision."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _blank_out_non_answers(cls, value: object) -> object:
        return stated(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _a_window_must_be_possible(self) -> CallExtraction:
        """Drop an upper bound that contradicts the lower one.

        "Free after 4pm" came back as earliest 16 *and* latest 16, which asks for work
        that starts after four and finishes by four. A model reaching for both ends of
        a range it was given one end of is a predictable slip, and it is checkable, so
        it is checked rather than prompted away.
        """
        if (
            self.earliest_hour is not None
            and self.latest_hour is not None
            and self.latest_hour <= self.earliest_hour
        ):
            object.__setattr__(self, "latest_hour", None)
        return self

    @field_validator("address", mode="after")
    @classmethod
    def _an_address_is_a_place_not_a_phrase(cls, value: str) -> str:
        """ "Ani in haslet" produced the address "in haslet", which then failed to
        geocode for the preposition rather than for the vagueness."""
        lowered = value.lower()
        for prefix in ("in ", "at ", "on ", "over in ", "out in "):
            if lowered.startswith(prefix):
                return value[len(prefix) :].strip()
        return value

    @field_validator("phone", mode="after")
    @classmethod
    def _only_something_that_could_be_dialled(cls, value: str) -> str:
        return phone_or_blank(value)

    @field_validator("email", mode="after")
    @classmethod
    def _only_something_that_could_be_emailed(cls, value: str) -> str:
        return email_or_blank(value)

    customer_name: str = Field(default="", description="The caller's name, if given.")
    phone: str = Field(default="", description="Callback number, digits as spoken.")
    email: str = Field(default="")
    address: str = Field(default="", description="The address as the caller said it.")

    service_type: str = Field(
        default="",
        description="One service type from the catalogue. Empty if genuinely unclear.",
    )
    description: str = Field(default="", description="What they said is wrong, briefly.")
    pane_count: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many panes of glass, only if the caller counted them. A measurement "
            "is not a count: 'a 6 ft glass' is one pane, not six."
        ),
    )
    glass_type: str = Field(
        default="", description="annealed, tempered, laminated or insulated_unit, if known."
    )
    property_type: Literal["residential", "commercial", "vehicle", ""] = ""

    urgency: Literal["emergency", "high", "normal", "low"] = Field(
        default="normal",
        description="emergency only when the property is insecure or unsafe right now.",
    )
    preferred_timing: str = Field(
        default="", description='In their words, e.g. "Tuesday afternoon", "any morning".'
    )
    # The same preference as a number, because prose cannot constrain a search.
    # "Free anytime after 4pm" was extracted perfectly into preferred_timing and then
    # ignored by everything downstream, so every slot came back at six in the morning.
    earliest_hour: int | None = Field(
        default=None,
        ge=0,
        le=23,
        description=(
            "Earliest hour they will accept, 24-hour. 16 for 'after 4pm', "
            "12 for 'afternoons'. Null if they did not say."
        ),
    )
    preferred_weekdays: list[int] = Field(
        default_factory=list,
        description=(
            "Days they can do, 0=Monday..6=Sunday, ONLY when they restricted days. "
            "'Only on Thursdays' is [3]. 'Weekdays' is no restriction - leave empty."
        ),
    )
    latest_hour: int | None = Field(
        default=None,
        ge=0,
        le=23,
        description=(
            "Latest hour the work may still be going on, 24-hour. 12 for 'mornings "
            "only', 9 for 'before we open at nine'. Null if they did not say."
        ),
    )
    hard_constraint: str = Field(
        default="",
        description=(
            'A constraint that cannot be broken, e.g. "must be finished before we open '
            'at nine". Empty when there is none.'
        ),
    )
    customer_must_be_present: bool | None = None

    commitment_signals: list[CommitmentSignal] = Field(
        default_factory=list,
        description=(
            "Which of the listed arrangements the caller said they had made around "
            "this appointment. Only what was actually said."
        ),
    )
    commitment_quotes: list[str] = Field(
        default_factory=list,
        description="The caller's own words for each signal, verbatim and short.",
    )

    site_notes: str = Field(
        default="",
        description='Access details worth writing down: gate codes, parking, "alley only".',
    )
    site_access: Literal["easy", "street_parking", "difficult", ""] = Field(
        default="",
        description=(
            "How close the van can get, ONLY if the caller described it. "
            "'easy': driveway, loading dock, or parking right at the door. "
            "'street_parking': park on the street and carry to the door. "
            "'difficult': alley only, stairs, upper floor, or a long carry. "
            "Empty when they did not say."
        ),
    )
    details_covered: list[str] = Field(
        default_factory=list,
        description="Which of the catalogue's ask-about items the caller already answered.",
    )


SYSTEM = """\
You are helping a dispatcher take a call at a glass-fitting business. Turn what the
caller said into structured details.

Rules:
- Record only what was said. Never infer an address, a phone number, or a time.
- Pick one service type from the catalogue below, or leave it empty if genuinely unclear.
- Do not estimate how long the job will take. That is not your job and the system
  already knows.
- A named place IS the address when that is what the caller gave. Managers say "the
  Walmart here in Haslet", not a street number - put "Walmart Supercenter, Haslet, TX"
  in address, never just the town.
- pane_count is how many panes, not how big. "a 6ft glass", "a 3 by 4 pane" and "the
  big window" are all one pane. Six panes means the caller said six. Getting this
  wrong triples the duration and the crew, and the customer is quoted for a day's work
  they did not ask for.
- Put any time preference in numbers as well as words. "Free after 4pm" is
  earliest_hour 16. "Mornings only" is latest_hour 12. "Before we open at nine" is
  latest_hour 9. Set only the end the caller actually gave: "after 4pm" has an
  earliest and no latest; "before nine" has a latest and no earliest. Leave them null
  when the caller did not say - a guessed constraint is worse than none, because it
  silently rules out slots they would have taken.
- commitment_signals: only arrangements the caller actually mentioned making. Quote
  their words. Say nothing if they said nothing. The difference between these matters,
  because they are priced very differently:
    time_off_work      - they are giving up work or leave for this. "I'd have to take
                         the morning off", "I'm booking a day's holiday".
    waiting_in         - they will simply be at home anyway. "I'll be in all day",
                         "I'm around Friday whenever". Much weaker than taking leave.
    arranged_childcare - cover has been booked that would have to be rebooked.
    business_closed    - a business is closing, opening late, or losing trade for us.
    already_rescheduled - we have moved this appointment before.
- urgency is `emergency` only when the property is insecure or someone is unsafe now.

Catalogue:
{catalogue}
"""


EXAMPLES: tuple[Example, ...] = (
    # The address is a place name, because that is how a manager gives it. Extracting
    # only the town threw away the one word that let the map find the right door.
    Example(
        text=(
            "This is James, a manager at the Walmart Supercenter here in Haslet - "
            "someone ran into our sliding doors, need it fixed asap, 913-777-2727."
        ),
        output=CallExtraction(
            customer_name="James",
            phone="913-777-2727",
            address="Walmart Supercenter, Haslet, TX",
            service_type="storefront_glass",
            description="sliding door glass smashed by a vehicle",
            property_type="commercial",
        ),
    ),
    # A size, not a count. "6 ft" became six panes, which tripled the duration to five
    # and three quarter hours and put two fitters on it - a day's work the caller never
    # asked for, quoted at a day's price. A rule alone did not fix it; a worked example
    # did, which is the documented lever for a small model.
    Example(
        text=(
            "Ani here, 913-295-2960. Backyard door glass is broken at 300 W Byron "
            "Nelson Blvd, Roanoke. About a 6 ft glass, ground floor. Free after 4pm."
        ),
        output=CallExtraction(
            customer_name="Ani",
            phone="913-295-2960",
            address="300 W Byron Nelson Blvd, Roanoke",
            service_type="residential_window_replacement",
            description="backyard door glass broken, about 6 ft",
            pane_count=1,
            property_type="residential",
            preferred_timing="after 4pm",
            earliest_hour=16,
            site_notes="ground floor",
        ),
    ),
    Example(
        text=(
            "Hi, it's Sarah Chen, 206-555-0142. Two windows went in the front room, "
            "4410 Ballard Ave. I can do Tuesday but I'd have to take the morning off work."
        ),
        output=CallExtraction(
            customer_name="Sarah Chen",
            phone="206-555-0142",
            address="4410 Ballard Ave",
            service_type="residential_window_replacement",
            description="Two front-room windows broken",
            pane_count=2,
            property_type="residential",
            preferred_timing="Tuesday morning",
            latest_hour=12,
            commitment_signals=[CommitmentSignal.TIME_OFF_WORK],
            commitment_quotes=["I'd have to take the morning off work"],
            details_covered=["how many panes"],
        ),
    ),
    Example(
        text="Someone's put the shop window through overnight. We open at nine.",
        output=CallExtraction(
            description="Storefront window smashed overnight",
            service_type="storefront_glass",
            property_type="commercial",
            urgency="emergency",
            hard_constraint="must be finished before the shop opens at 09:00",
            commitment_signals=[CommitmentSignal.BUSINESS_CLOSED],
            commitment_quotes=["We open at nine"],
        ),
    ),
)


#: What a job cannot be scheduled without. Everything else can follow on a second call.
REQUIRED_FIELDS: tuple[tuple[str, str], ...] = (
    ("customer_name", "the customer's name"),
    ("phone", "a callback number"),
    ("address", "the address"),
    ("service_type", "what kind of work it is"),
)


@dataclass(frozen=True, slots=True)
class IntakeResult:
    """A draft job, what it will cost to move, and what is still missing."""

    draft: Job | None
    extraction: Extraction[CallExtraction]
    estimate: Estimate | None
    commitment_cost: float
    commitment_quotes: tuple[str, ...]
    #: Things a dispatcher should ask before hanging up.
    ask_next: tuple[str, ...]
    missing_required: tuple[str, ...]
    #: The caller's stated hours, as numbers the search can use.
    earliest_hour: int | None = None
    latest_hour: int | None = None
    preferred_weekdays: tuple[int, ...] = ()
    geocode_note: str = ""
    #: Set when the caller wants something this business does not sell. The call
    #: still extracts cleanly - the model found "auto glass" correctly - but the
    #: answer down the phone is a referral, not a quote.
    not_offered: str = ""

    @property
    def bookable(self) -> bool:
        """Whether there is enough to price this into the schedule."""
        return self.draft is not None and not self.missing_required and not self.not_offered

    @property
    def call(self) -> CallExtraction | None:
        return self.extraction.value


def _said_on_the_call(quote: str, text: str) -> bool:
    """Did these words, near enough, actually come out of the caller's mouth?

    Normalised to letters so punctuation and case cannot hide a genuine quote, and
    matched as a substring so trimming counts as quoting. "time off work" against a
    call that never mentions work is what this exists to catch.
    """

    def letters(value: str) -> str:
        kept = "".join(ch for ch in value.lower() if ch.isalnum() or ch == " ")
        return " ".join(kept.split())

    return bool(quote.strip()) and letters(quote) in letters(text)


#: Digit runs that look like somebody tried to give a number: groups of digits with
#: optional separators, seven or more digits in total.
_PHONEISH = re.compile(r"(?:\d[\s().-]?){7,}")


def _a_number_was_attempted(text: str) -> bool:
    """Did the caller try to give a phone number that we then refused?

    "His number is 894892894" and silence are different situations, and "Ask for a
    callback number" answers only the second. For the first, the dispatcher needs to
    know a number was given and did not survive checking, or they will ask for
    something they believe they already have.
    """
    return bool(_PHONEISH.search(text))


_DAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
#: "only on Thursdays", "Thursdays only", "just Fridays" - a restriction, not a mention.
_ONLY_DAYS = re.compile(
    r"\bonly(?:\s+\w+){0,2}\s+(" + "|".join(_DAY_NAMES) + r")s?\b"
    r"|\b(" + "|".join(_DAY_NAMES) + r")s?\s+only\b",
    re.IGNORECASE,
)


def _restricted_days(call: CallExtraction, text: str) -> CallExtraction:
    """Keep a day restriction the caller actually made, and only then.

    "Free after 4pm weekdays" must not become Monday-to-Friday-as-a-constraint - that
    is just the week. "Only on Thursdays" must survive, because offering Wednesday to
    a man who said only Thursdays is the fastest way to sound like a machine.
    """
    said = {_DAY_NAMES[(m.group(1) or m.group(2)).lower()] for m in _ONLY_DAYS.finditer(text)}
    if said:
        return call.model_copy(update={"preferred_weekdays": sorted(said)})
    if call.preferred_weekdays and len(call.preferred_weekdays) >= 5:
        # The model restated the working week as a restriction. That is no restriction.
        return call.model_copy(update={"preferred_weekdays": []})
    return call


def _civil_floor(call: CallExtraction, text: str) -> CallExtraction:
    """Nobody said six in the morning, so do not offer it.

    "Morning" made the model guess earliest_hour 6, and the search dutifully offered
    every customer the crack of Marcus's shift. A lower bound below eight stands only
    when the caller typed the early hour themselves - "after 6am" is a request, "the
    morning" is not. Upper bounds are untouched: "before we open at nine" legitimately
    wants the crew there early, and that is the crew's business, not the caller's.
    """
    if call.earliest_hour is not None and call.earliest_hour < 8 and not _AFTER.search(text):
        return call.model_copy(update={"earliest_hour": 8})
    return call


#: Phrases that express urgency, not arrangement. "As soon as possible" was priced as
#: $90 of waiting-in goodwill: the quote was genuinely in the transcript, so grounding
#: passed, but wanting it soon is not the same as having arranged your day around it.
#: "Whenever" is deliberately absent - "I'm around Friday whenever" IS waiting in.
_URGENCY_NOT_ARRANGEMENT = re.compile(
    r"as soon as possible|\basap\b|right away|immediately|urgent", re.IGNORECASE
)

#: Words that mark a genuine arrangement - something given up or organised. A quote
#: with none of these is availability, urgency, or politeness: "I'm free after 4pm
#: weekdays" states a constraint we already captured as hours, and pricing it as $90
#: of goodwill charges the planner for a fact. Every worked example of a real signal
#: contains one of these, which is what makes the list a contract rather than a vibe.
_ARRANGEMENT = re.compile(
    r"\b(?:off work|off\b|leave|holiday|vacation|childcare|sitter|nanny|closed|closing"
    r"|reschedul\w*|moved|be in\b|be home|be around|around\b|waiting|stay(?:ing)? home"
    r"|took|taking|booked)",
    re.IGNORECASE,
)


def grounded_commitments(call: CallExtraction, text: str) -> CallExtraction:
    """Keep only the commitment evidence the transcript supports.

    "James needs the window fixed on monday morning" came back with time_off_work and
    the quote "time off work" - words James never said. The model was asked to price
    goodwill and invented the receipt. Every quote must appear in the call, and a
    signal with no surviving quote is not priced: an unpriced real commitment costs a
    slightly-too-cheap move later, while a priced invented one is a $250 lie with the
    customer's name on it.
    """
    quotes = [
        q
        for q in call.commitment_quotes
        if _said_on_the_call(q, text)
        and not _URGENCY_NOT_ARRANGEMENT.search(q)
        and _ARRANGEMENT.search(q)
    ]
    if len(quotes) == len(call.commitment_quotes):
        return call
    return call.model_copy(
        update={
            "commitment_quotes": quotes,
            "commitment_signals": [] if not quotes else call.commitment_signals,
        }
    )


def commitment_cost_for(signals: list[CommitmentSignal], business: BusinessParams) -> float:
    """Turn recognised signals into a number, capped.

    Deliberately not the model's decision. The cap matters too: without it a caller
    listing every inconvenience could make one job effectively immovable and distort
    the whole week around it.
    """
    weights = {
        CommitmentSignal.TIME_OFF_WORK: business.commitment.time_off_work.value,
        CommitmentSignal.WAITING_IN: business.commitment.waiting_in.value,
        CommitmentSignal.ARRANGED_CHILDCARE: business.commitment.arranged_childcare.value,
        CommitmentSignal.BUSINESS_CLOSED: business.commitment.business_closed.value,
        CommitmentSignal.ALREADY_RESCHEDULED: business.commitment.already_rescheduled.value,
    }
    total = sum(weights.get(signal, 0.0) for signal in set(signals))
    return min(total, business.commitment.max_total.value)


#: A measurement, however it is written: a pair like "3 by 4" or "3' x 4\"", or a
#: single one like "6 ft". Both are sizes, and neither is a quantity of panes.
_DIMENSIONS = re.compile(
    r"\b\d+\s*(?:'|ft|foot|feet|\"|in|inch(?:es)?|cm|m|mm)?\s*(?:x|by|\*)\s*\d+"
    r"|\b\d+\s*(?:'|ft|foot|feet|\"|inch(?:es)?|cm|mm)\b",
    re.IGNORECASE,
)
#: An explicit count: "six panes", "2 windows", "three sheets".
_COUNTED = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:pane|window|sheet|unit|glass)e?s\b",
    re.IGNORECASE,
)


#: "after 4pm", "from 4", "not before 4" - a lower bound on when we may turn up.
#: "come at 5pm", "arrive at 5", "be there at 17:00" - an appointment TIME, which
#: is a lower bound in disguise: the caller wants the visit to start then. Anchored
#: to a verb of arrival so "we open at 9am" (a fact about the shop) stays untouched.
_COME_AT = re.compile(
    r"\b(?:come|arrive|be (?:t?here|at the \w+)|show up|stop by|swing by|fix(?:\s+\w+){0,3}?)"
    r"\s+at\s+(\d{1,2})\s*(?::\s*(\d{2}))?\s*(am|pm|o'?clock)?",
    re.IGNORECASE,
)

_AFTER = re.compile(
    r"\b(?:after|from|not before|no earlier than|past)\s+(\d{1,2})\s*(?::\s*(\d{2}))?\s*"
    r"(am|pm|o'?clock)?",
    re.IGNORECASE,
)
#: "before 9", "by 9", "until 9" - an upper bound.
_BEFORE = re.compile(
    # "not before 8" is a lower bound wearing an upper bound's words, so the negated
    # forms are excluded here and caught by _AFTER instead. The number must not wear
    # a unit: "8ft by 8ft" put "before 8:00 AM" on a caller who never mentioned the
    # clock, because "by 8" is both a deadline and half of a measurement.
    r"(?<!not )(?<!no )(?<!earlier than )"
    # A digit or unit right before the keyword means a measurement ("3 by 4",
    # "8ft by 8ft"), not a deadline.
    r"(?<![0-9] )(?<![0-9])(?<!ft )(?<!feet )(?<!inch )(?<!inches )"
    r"\b(?:before|by|until|till|til|no later than)\s+(\d{1,2})\s*"
    r"(?!\s*(?:ft|feet|foot|in\b|inch|inches|mm|cm|m\b|meter|metre|x\b|by\b|'|\"))"
    r"(?::\s*(\d{2}))?\s*(am|pm|o'?clock)?",
    re.IGNORECASE,
)


def _hour_24(digits: str, meridiem: str | None) -> int | None:
    hour = int(digits)
    if hour > 23:
        return None
    marker = (meridiem or "").lower()
    if marker.startswith("pm") and hour < 12:
        hour += 12
    elif marker.startswith("am") and hour == 12:
        hour = 0
    elif not marker and hour <= 7:
        # "after 4" from a customer talking about their day means the afternoon. A
        # glass fitter is not turning up at four in the morning, and the caller knows
        # that, which is why they did not bother saying pm.
        hour += 12
    return hour


def _fix_the_direction(call: CallExtraction, text: str) -> CallExtraction:
    """Put a stated bound on the side the caller actually put it.

    "Free anytime of the week after 4pm" came back as latest_hour 16 - finish by four -
    and every slot offered was six in the morning, which is the exact opposite of what
    was asked for and reads as the system ignoring the customer entirely.

    Which side of a range a word puts you on is not a matter of interpretation, so it
    is checked. The model still does the hard part: finding the time in the prose.
    """
    after = _AFTER.search(text) or _COME_AT.search(text)
    before = _BEFORE.search(text)
    update: dict[str, int | None] = {}

    if after:
        hour = _hour_24(after.group(1), after.group(3))
        if hour is not None:
            update["earliest_hour"] = hour
            if call.latest_hour is not None and call.latest_hour <= hour and not before:
                update["latest_hour"] = None
    if before:
        hour = _hour_24(before.group(1), before.group(3))
        if hour is not None:
            update["latest_hour"] = hour

    return call.model_copy(update=update) if update else call


#: Deterministic classification of what the caller said about getting the van in.
#: Checked rather than prompted, like every other backstop here: Rosa said "you
#: have to park on the street out front", the model marked the access question
#: answered and left the classification blank - so the question vanished AND the
#: carry-minutes never landed. Hard words first: an alley with a dock is an alley.
_ACCESS_HARD = re.compile(r"\balley\b|long carry|walk[- ]?up|no way to park", re.I)
_ACCESS_STREET = re.compile(r"street parking|park (?:on|in) the street|no (?:parking )?lot\b", re.I)
_ACCESS_EASY = re.compile(
    r"loading dock|\bdriveway\b|park (?:right )?(?:at|by|in front of) the door", re.I
)


def _hear_the_parking_situation(call: CallExtraction, text: str) -> CallExtraction:
    """When the caller described access and the model left the box unticked."""
    if call.site_access:
        return call
    for level, pattern in (
        ("difficult", _ACCESS_HARD),
        ("street_parking", _ACCESS_STREET),
        ("easy", _ACCESS_EASY),
    ):
        if pattern.search(text):
            return call.model_copy(update={"site_access": level})
    return call


def _distrust_a_size_read_as_a_count(call: CallExtraction, text: str) -> CallExtraction:
    """A measurement is not a quantity.

    "About a 6 ft glass" came back as six panes, which took a two-hour job to five and
    three quarter hours and put two fitters on it - a day's work the caller never asked
    for, at a day's price. A worked example fixed that phrasing; "a 3 by 4 pane" still
    multiplied the dimensions.

    So it is checked rather than prompted. If the caller wrote a measurement and never
    wrote a count, the count is one. Both halves matter: "six panes, each 3 by 4" says
    a number out loud and keeps it.
    """
    if call.pane_count is None or call.pane_count <= 1:
        return call
    if _DIMENSIONS.search(text) and not _COUNTED.search(text):
        return call.model_copy(update={"pane_count": 1})
    return call


#: What answers each catalogue question, when the model forgets to say so itself.
#: Deterministic backstop for `details_covered`: the mechanism exists, the model uses
#: it unreliably, and asking a caller for the size they just gave reads as not
#: listening.
_ANSWERED_BY: dict[str, re.Pattern[str]] = {
    "how many panes": re.compile(
        r"\b(?:\d+|one|two|three|four|five|six)\s+(?:big\s+)?pane|\bpane\b", re.I
    ),
    # A measurement, or a comparison to a thing with a known size. "About the size of
    # a door" tells a glazier more than most numbers would.
    "rough size": re.compile(
        _DIMENSIONS.pattern + r"|size of (?:a |the )?\w+|door[- ]sized?|full[- ]length",
        re.IGNORECASE,
    ),
    "ground floor or upstairs": re.compile(
        r"\bground floor|\bupstairs|\bdownstairs|\bfirst floor|\bsecond floor", re.I
    ),
    "street or alley access": re.compile(
        r"\balley\b|street parking|park (?:on|in) the street|loading dock|\bdriveway\b"
        r"|park (?:right )?(?:at|by) the door",
        re.I,
    ),
}


def _still_unanswered(asks: tuple[str, ...], call: CallExtraction, text: str) -> tuple[str, ...]:
    out = []
    for ask in asks:
        pattern = _ANSWERED_BY.get(ask)
        if ask == "how many panes" and call.pane_count is not None:
            continue
        if pattern is not None and pattern.search(text):
            continue
        out.append(ask)
    return tuple(out)


#: Words in a road name that carry no identity. "Road" appears in every road.
_ROAD_NOISE = frozenset(
    {
        "road",
        "rd",
        "street",
        "st",
        "avenue",
        "ave",
        "boulevard",
        "blvd",
        "lane",
        "ln",
        "drive",
        "dr",
        "highway",
        "hwy",
        "parkway",
        "pkwy",
        "court",
        "ct",
        "way",
        "north",
        "south",
        "east",
        "west",
        "n",
        "s",
        "e",
        "w",
    }
)


def _the_map_guessed(spoken: str, location: Location) -> str:
    """The street the map matched, when the caller never said it.

    Nominatim is helpful to a fault: "16 Haslet, Texas" resolves cleanly to 16
    Avondale Haslet Road - a real front door that nobody asked for a van at. If the
    matched road has identifying words the caller did not say, this returns what to
    confirm; booking against a guessed street is how a crew spends an hour at the
    wrong house.
    """
    # A named place the caller themselves named needs no street check - "the Walmart
    # in Haslet" matching Walmart Supercenter IS the confirmation, and challenging the
    # road it happens to sit on ("you never said Avondale") would be the system
    # arguing with a correct answer.
    name_words = [w for w in location.matched_name.lower().split() if len(w) > 3]
    if name_words and any(w in spoken.lower() for w in name_words):
        return ""
    road = location.matched_road.lower()
    if not road:
        return ""
    said = spoken.lower()
    distinctive = [w for w in road.replace("-", " ").split() if w not in _ROAD_NOISE]
    missing = [w for w in distinctive if w not in said]
    return location.matched_road if missing else ""


def intake(
    provider: LLMProvider,
    text: str,
    *,
    business: BusinessParams,
    now: datetime,
    geocoder: Geocoder | None = None,
    job_id: str = "draft",
    max_attempts: int = 3,
) -> IntakeResult:
    """Turn call notes into a draft job, or say what is still needed."""
    with span("agent.intake", chars=len(text)) as active:
        extraction = extract(
            provider,
            CallExtraction,
            system=SYSTEM.format(catalogue=describe_for_prompt()),
            text=text,
            examples=EXAMPLES,
            max_attempts=max_attempts,
            schema_description="Details heard on the call.",
        )

        call = extraction.value
        if call is not None:
            call = _distrust_a_size_read_as_a_count(call, text)
            call = _hear_the_parking_situation(call, text)
            call = _fix_the_direction(call, text)
            call = grounded_commitments(call, text)
            call = _civil_floor(call, text)
            call = _restricted_days(call, text)
        if call is None:
            record(escalated=True, provider_failed=extraction.provider_failed)
            active.set_attribute("outcome", "escalated")
            return IntakeResult(
                draft=None,
                extraction=extraction,
                estimate=None,
                commitment_cost=0.0,
                commitment_quotes=(),
                ask_next=("Could you take those details again by hand?",),
                missing_required=tuple(label for _, label in REQUIRED_FIELDS),
            )

        missing = tuple(
            label for field_name, label in REQUIRED_FIELDS if not getattr(call, field_name)
        )
        # A number that was given and refused is a different conversation from a number
        # that was never given. Asking for something the caller believes they already
        # provided reads as not listening.
        if not call.phone and _a_number_was_attempted(text):
            missing = tuple(
                "the callback number again - what was given does not look dialable"
                if label == "a callback number"
                else label
                for label in missing
            )

        estimate: Estimate | None = None
        if call.service_type:
            try:
                service = ServiceType(call.service_type)
            except ValueError:
                # The model named something outside the catalogue. Treat the service
                # type as unknown rather than guessing the nearest match - a wrong
                # service type silently produces a wrong duration and wrong crew.
                missing = (*missing, "what kind of work it is")
            else:
                estimate = lookup(
                    service,
                    pane_count=call.pane_count or 1,
                    glass_type=_glass_type(call.glass_type),
                    known_details=frozenset(call.details_covered),
                    site_access=call.site_access,
                )

        commitment = commitment_cost_for(call.commitment_signals, business)

        location: Location | None = None
        geocode_note = ""
        if call.address:
            try:
                location = (geocoder or for_service_area()).geocode(call.address)
            except GeocodeError as exc:
                geocode_note = str(exc)
                missing = (*missing, "an address we can find on the map")
            else:
                guessed = _the_map_guessed(call.address, location)
                if guessed:
                    geocode_note = (
                        f"the map matched {location.address.split(',')[0]}, "
                        f"{guessed} - a street the caller did not say"
                    )
                    missing = (*missing, f"the street - the map guessed {guessed!r}")
                    location = None

        draft = (
            _build_draft(
                job_id=job_id,
                call=call,
                estimate=estimate,
                location=location,
                commitment=commitment,
                now=now,
                extraction=extraction,
            )
            if estimate is not None and location is not None
            else None
        )

        follow_ups = _still_unanswered(estimate.missing_details if estimate else (), call, text)
        # No stated time is a question, not a default. Every caller has one, and a
        # dispatcher who books six in the morning without asking finds out about it
        # from the doorbell.
        if call.earliest_hour is None and call.latest_hour is None and not call.preferred_timing:
            follow_ups = (*follow_ups, "when suits them - morning, afternoon, or a day")
        ask_next = tuple([f"Ask for {label}." for label in missing] + list(follow_ups))

        record(
            repairs=extraction.repairs,
            missing=len(missing),
            commitment_cost=commitment,
            bookable=draft is not None and not missing,
            service=call.service_type,
        )
        active.set_attribute("outcome", "bookable" if draft is not None else "incomplete")

        # Deterministic, from config: the model's extraction can be perfect and the
        # answer still be "we don't do that". Checked here so no slot search, no
        # pricing and no crew ask ever runs for work this business does not sell.
        not_offered = ""
        if call.service_type and not business.meta.offers(call.service_type):
            what = call.service_type.replace("_", " ")
            not_offered = (
                f"Not something we do: {what}. Suggest a dedicated {what} shop "
                "and take their next call."
            )

        return IntakeResult(
            draft=draft,
            extraction=extraction,
            estimate=estimate,
            commitment_cost=commitment,
            commitment_quotes=tuple(call.commitment_quotes),
            ask_next=ask_next,
            missing_required=missing,
            earliest_hour=call.earliest_hour,
            latest_hour=call.latest_hour,
            preferred_weekdays=tuple(call.preferred_weekdays),
            geocode_note=geocode_note,
            not_offered=not_offered,
        )


def _glass_type(value: str) -> GlassType | None:
    try:
        return GlassType(value) if value else None
    except ValueError:
        return None


def _build_draft(
    *,
    job_id: str,
    call: CallExtraction,
    estimate: Estimate,
    location: Location,
    commitment: float,
    now: datetime,
    extraction: Extraction[CallExtraction],
) -> Job:
    """Assemble the job. Every scheduling number comes from the catalogue, not the call."""
    materials = tuple(
        Material(
            part_code=part,
            quantity=max(1, call.pane_count or 1),
            in_stock=not estimate.needs_ordering,
            lead_time_days=estimate.lead_time_days,
        )
        for part in estimate.parts
    )
    return Job(
        id=job_id,
        customer_id=f"c-{job_id}",
        customer_name=call.customer_name or "Unnamed caller",
        phone=call.phone,
        email=call.email,
        location=location,
        property_type=PropertyType(call.property_type)
        if call.property_type
        else PropertyType.RESIDENTIAL,
        service_type=estimate.entry.service_type,
        description_raw=call.description,
        glass_spec=GlassSpec(
            pane_count=max(1, call.pane_count or 1),
            glass_type=_glass_type(call.glass_type),
        ),
        required_certifications=estimate.required_certifications,
        crew_size=estimate.crew_size,
        estimated_duration_min=estimate.duration_min,
        duration_confidence_min=estimate.confidence_min,
        materials=materials,
        priority=Priority(call.urgency),
        commitment_cost=commitment,
        requires_customer_present=(
            call.customer_must_be_present
            if call.customer_must_be_present is not None
            else estimate.requires_customer_present
        ),
        site_notes=call.site_notes,
        requested_at=now,
        provenance=Provenance(
            source_channel="phone",
            received_at=now,
            missing_required=(),
            extractor_retries=extraction.repairs,
        ),
    )
