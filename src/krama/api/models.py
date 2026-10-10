"""Response shapes for the dispatch board.

Richer than the MCP models on purpose. Those are read by a small language model, where
every extra field is a chance to weigh the wrong one; these are read by a dispatcher
looking at a screen, where the extra context is the point. A person wants to see the
gap between two jobs and judge whether it looks wrong - a model would just be
distracted by it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Api(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StopView(Api):
    job_id: str
    customer_name: str
    service_type: str
    arrival: str
    departure: str
    #: Minutes from midnight, so the board can position bars without parsing dates.
    start_minute: int
    end_minute: int
    travel_minutes: int
    travel_miles: float
    crew_size: int
    commitment_state: str
    #: Slack before this stop. Surfaced because a large gap usually means the sequence
    #: is wrong, and that is a judgement a person makes by looking.
    gap_minutes: int = 0
    lat: float = 0.0
    lon: float = 0.0
    #: True when the work runs past the crew's shift - the overtime portion of the
    #: day. The board draws these dashed so a late booking is visibly a late stay.
    past_shift: bool = False
    #: Where this drive begins: "the shop", or the previous customer's name. The
    #: owner read "15 min - 5.83 mi" and could not tell whether the van chained from
    #: the last job or doubled back to the depot - a question the data always knew
    #: the answer to and the screen never said.
    from_label: str = ""
    #: And where the van goes afterwards - the next customer, or "back to the
    #: shop". With both ends on the card, a stop reads as a link in the day's
    #: chain instead of an isolated trip.
    next_label: str = ""
    #: The day's shape as it stands NOW - "a dedicated trip out and back" or
    #: "one of 3 stops on this run, with James's and Ani's". Recomputed from the
    #: committed plan on every read, because the quote-time note went stale the
    #: moment a later booking joined the run and the card kept calling a
    #: three-stop morning a dedicated trip.
    run_note: str = ""


class RouteView(Api):
    crew_id: str
    date: str
    worker_names: list[str]
    van_id: str
    stops: list[StopView]
    travel_minutes: int
    travel_miles: float
    idle_minutes: int
    utilization: float
    overtime_minutes: int


class UnservedView(Api):
    job_id: str
    customer_name: str
    reason: str
    detail: str
    is_failure: bool


class CostView(Api):
    vehicle: float
    overtime: float
    lateness: float
    unserved: float
    total: float


class PlanView(Api):
    #: True when business-private money was stripped for a dispatcher session. The
    #: stripping itself happens server-side; this flag just tells the UI not to
    #: render empty money rows.
    redacted: bool = False
    plan_id: str
    content_hash: str
    horizon_start: str
    horizon_end: str
    routes: list[RouteView]
    unserved: list[UnservedView]
    cost: CostView
    feasible: bool
    violations: list[str] = Field(default_factory=list)
    depot: list[float] = Field(default_factory=list)


class WorkerDayView(Api):
    """One fitter's availability on one day, in words a dispatcher reads at a glance."""

    date: str
    day: str
    #: "8:00 AM - 5:00 PM", or "off".
    shift: str
    #: How late they may legally be kept with overtime - "can stay to 7:00 PM". The
    #: single fact that explains why an after-four job keeps landing on Dan.
    reach: str = ""
    #: "agreed to stay to 8:00 PM" - a one-day ShiftExtended, shown in the cell so
    #: the rota explains why an evening booking suddenly became possible.
    extended: str = ""
    available: bool = True
    #: False once the day can no longer be changed - the shift plus any overtime
    #: reach is already behind the clock. The board greys these instead of offering
    #: a click that silently does nothing, which is the worst thing a click can do.
    actionable: bool = True
    #: Hour-scoped absences inside an otherwise working day - "out 8:00 - 10:00 AM".
    #: A dentist appointment is not a sick day, and showing it as one made the rota
    #: lie in both directions: the fitter looked gone, and the real two-hour hole
    #: was invisible. Recorded through the same events the call box writes.
    out_note: str = ""
    #: The same holes as machine times, [["08:00", "10:00"], ...], so the cell can
    #: offer "bring back exactly these hours" without parsing its own label.
    out_spans: list[list[str]] = []


class WorkerView(Api):
    id: str
    name: str
    certifications: list[str]
    shift: str
    available: bool
    overtime_eligible: bool
    #: Raw HH:MM pair + phone, so the configure panel edits what is actually stored
    #: instead of parsing the display string back apart.
    shift_start: str = ""
    shift_end: str = ""
    phone: str = ""
    #: The week ahead, one entry per working day the board shows.
    days: list[WorkerDayView] = Field(default_factory=list)


class VanDayView(Api):
    """One van's availability on one rota day - the same grammar as a fitter's."""

    date: str
    day: str
    available: bool = True
    actionable: bool = True
    out_note: str = ""
    out_spans: list[list[str]] = []


class VanView(Api):
    id: str
    label: str
    available: bool
    stock: dict[str, int]
    #: The rota week for this van, so a breakdown is a visible, clickable fact on
    #: the same table as the crew - not something only the solver knows.
    days: list[VanDayView] = []


class JobView(Api):
    id: str
    customer_name: str
    service_type: str
    duration_minutes: int
    crew_size: int
    certifications: list[str]
    commitment_state: str
    commitment_cost: float
    window: str
    lat: float
    lon: float
    phone: str = ""
    address: str = ""
    quoted_total: float = 0.0
    booking_note: str = ""
    #: Machine-readable promise bounds, so the calendar can draw a confirmed
    #: booking even when no committed plan has reached its week yet - a promise
    #: with no tile is a promise someone will forget.
    window_start: str = ""
    window_end: str = ""
    transcript: str = ""


class OvertimeOfferView(Api):
    """Hours past shift, up for grabs - rendered as a strip on the crew panel."""

    job_id: str
    customer: str
    day: str
    arrival: str
    overtime_minutes: int
    #: "open", "claimed" or "expired" (deadline passed; the fallback stands).
    status: str
    offered_to: list[str] = Field(default_factory=list)
    offered_ids: list[str] = Field(default_factory=list)
    claimed_by: str = ""
    fallback: str = ""
    deadline: str = ""


class WorldView(Api):
    as_of: str
    workers: list[WorkerView]
    vans: list[VanView]
    jobs: list[JobView]
    overtime_offers: list[OvertimeOfferView] = Field(default_factory=list)
    crew_asks: list[CrewAskView] = Field(default_factory=list)
    committed_plan_id: str = ""
    #: Shown on screen. Every cost here rests on numbers nobody has validated.
    calibration_warning: str = ""
    depot_address: str = ""


class ChangeView(Api):
    job_id: str
    customer_name: str
    kind: str
    description: str
    needs_customer_call: bool


class CandidateView(Api):
    strategy: str
    description: str
    jobs_served: int
    changes: int
    customer_calls: int
    blast_radius: str
    autonomy: str
    autonomy_reasons: list[str]
    diff: list[ChangeView]
    recommended: bool = False


class RepairView(Api):
    baseline_plan_id: str
    candidates: list[CandidateView]
    recommended: str = ""
    rationale: str = ""
    #: Present when an agent, rather than the engine's default, made the choice.
    chosen_by: str = "engine"


class DraftView(Api):
    """A job taking shape during a call, including what is still missing."""

    customer_name: str = ""
    phone: str = ""
    address: str = ""
    service_type: str = ""
    duration_minutes: int = 0
    duration_confidence: int = 0
    crew_size: int = 0
    certifications: list[str] = Field(default_factory=list)
    commitment_cost: float = 0.0
    commitment_quotes: list[str] = Field(default_factory=list)
    lead_time_days: int = 0
    site_notes: str = ""
    lat: float | None = None
    lon: float | None = None


class SlotView(Api):
    """One bookable option, in the terms a dispatcher would say out loud.

    ``window`` is what the customer is promised; ``arrival`` is when the crew expects
    to be there. They differ because a two-hour promise around a 6:56 arrival reads as
    "5:56 to 7:56", which looks arbitrary until you know which number is the estimate
    and which is the commitment.
    """

    date: str
    day: str
    window: str
    arrival: str
    marginal_cost: float
    crew: str
    #: Worker ids behind ``crew`` - the booking pins these so the committed plan
    #: cannot quietly differ from the card the dispatcher just read aloud.
    crew_ids: list[str] = Field(default_factory=list)
    #: Where the drive begins and what it costs - on the card, not just the detail
    #: popup after booking.
    from_label: str = "the shop"
    leg_minutes: int = 0
    leg_miles: float = 0.0
    crew_reason: str
    reason: str

    #: What the customer pays, and what this placement leaves us. Two different
    #: questions, both on screen, because a price with no cost beside it cannot be
    #: judged and a cost with no price is what the board used to show.
    quote_total: float = 0.0
    quote_lines: list[str] = Field(default_factory=list)
    margin: float = 0.0
    margin_pct: float = 0.0

    #: True when this slot keeps a fitter past their shift. The price already carries
    #: the after-hours rate; this flag is for the *crew* conversation - the dispatcher
    #: can promise the customer the slot and then ask who wants the time-and-a-half,
    #: rather than needing to know before answering the phone.
    needs_overtime: bool = False

    #: True for a slot outside the caller's stated hours, offered because it is
    #: cheaper for them - usually by the after-hours premium. The dispatcher decides
    #: whether to float it; the system's job is to make the saving visible.
    outside_preference: bool = False


class BookRequest(Api):
    """A draft plus the slot a customer accepted. Nothing is stored before this."""

    draft: DraftView
    date: str
    arrival: str
    #: What the caller said, kept on the job so a reschedule can start from it.
    transcript: str = ""
    #: The tax-inclusive total on the slot card the dispatcher pressed - the price
    #: the customer just heard. Stored on the job; without it the agreed price
    #: evaporated the moment the quote left the screen, and the scheduler weighed a
    #: nine-hundred-dollar booking exactly like a zero-dollar one.
    quoted_total: float = 0.0
    #: The pressed slot's own explanation (reason + crew_reason), stored on the job
    #: so "why Marcus, why Tuesday" survives the slot card it was written on.
    booking_note: str = ""
    #: A human said "yes, this caller really wants a SECOND job" - required when
    #: an active booking already carries the same phone or address, because the
    #: alternative was two silent Jameses on one Monday.
    allow_duplicate: bool = False
    #: The quoted crew, pinned for THIS booking's own commit only. Later re-plans
    #: stay free to reshuffle crews inside promises - that flexibility is a feature
    #: - but the plan the customer's confirmation is read against must match the
    #: card that was just on screen.
    crew_ids: list[str] = Field(default_factory=list)


class UnavailableDayView(Api):
    """A day we cannot offer, and why. An empty slot list explains nothing."""

    day: str
    reason: str


class CrewAskSuggestion(Api):
    """Nobody can work the wanted hours - but somebody COULD, if they said yes.

    The red sentence on the intake: "check with Marcus and Priya whether they can
    stay to 8:00 PM on Monday". Holding it stores a CrewAsk so the promise to call
    the customer back survives a reload and a shift change.
    """

    message: str
    on_date: str
    until: str  # "20:00", for the extension event
    until_label: str  # "8:00 PM", for people
    candidate_ids: list[str] = Field(default_factory=list)
    candidate_names: list[str] = Field(default_factory=list)


class CrewAskView(Api):
    ask_id: str
    customer: str
    phone: str
    day: str
    until_label: str
    detail: str
    transcript: str
    candidates: list[dict[str, str]] = Field(default_factory=list)
    #: Candidates who have already said yes (a ShiftExtended exists for that day).
    extended: list[str] = Field(default_factory=list)


class AskRequest(Api):
    customer_name: str = ""
    phone: str = ""
    transcript: str
    on_date: str
    until: str
    candidate_ids: list[str]
    detail: str = ""


class WorkerConfig(Api):
    """One fitter as the configure panel edits them. Today this is typed by hand;
    the shape is deliberately the subset a payroll/HR export would carry, so when
    one is connected it feeds the same endpoint instead of a new one."""

    id: str = ""  # blank = new fitter; the server assigns an id
    name: str
    phone: str = ""
    certifications: list[str] = Field(default_factory=list)
    shift_start: str = "08:00"
    shift_end: str = "17:00"
    overtime_eligible: bool = True


class DepotMoveRequest(Api):
    """Both fields must carry the same text, typed - not pasted state, a retype."""

    address: str
    confirm: str


class VanConfig(Api):
    id: str = ""  # blank = new van
    label: str = ""
    #: What the van carries, by part code. None keeps what it has; a dict replaces
    #: it. Editable because the fleet trim quietly stranded two whole service
    #: types - the only shower kits in the business retired with van-3.
    stock: dict[str, int] | None = None


class DuplicateView(Api):
    """An active booking that looks like the same caller - same digits, or the
    same door. Surfaced, never auto-merged: a customer with two broken windows is
    two jobs, and only a person can tell an edit from an addition."""

    job_id: str
    customer: str
    when: str
    what: str


class IntakeView(Api):
    draft: DraftView
    bookable: bool
    missing: list[str]
    ask_next: list[str]
    slots: list[SlotView] = Field(default_factory=list)
    #: Cheaper options outside the stated hours, for the "if they're flexible" tab.
    flexible_slots: list[SlotView] = Field(default_factory=list)
    unavailable: list[UnavailableDayView] = Field(default_factory=list)
    #: The caller's stated hours, as words for the fixed intake form.
    when_text: str = ""
    #: True when the caller used urgency words - "as soon as possible", "urgent".
    #: The board then leads with the soonest slot; otherwise it leads with the
    #: cheapest, because a caller who named no hurry is being sold overtime they
    #: never asked for when the priciest day happens to sort first.
    asked_for_speed: bool = False
    #: Present when no day works but an extended shift would make one work.
    crew_ask: CrewAskSuggestion | None = None
    #: Set when the caller wants something this business does not sell. Red on the
    #: intake; the right answer is a referral, not a quote.
    not_offered: str = ""
    #: Margins stripped for a dispatcher session (quotes stay - the dispatcher
    #: reads prices aloud; what we KEEP is the owner's business).
    redacted: bool = False
    #: Active bookings that share this caller's phone or address. The dispatcher
    #: sees them BEFORE quoting; booking over one requires an explicit "it's a
    #: second job" from a human.
    possible_duplicates: list[DuplicateView] = Field(default_factory=list)
    repairs: int = 0
    note: str = ""


class TriageView(Api):
    state: str
    summary: str
    events: list[dict[str, object]] = Field(default_factory=list)
    question: str = ""
    unknown_targets: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    repairs: int = 0


class NoteView(Api):
    """One note, read by whichever agent it turned out to be for.

    Exactly one of ``booking`` and ``disruption`` is populated. The board used to ask
    the dispatcher to choose before typing, with two identical-looking boxes, and the
    first person to use it put a broken van into the one that books appointments.
    """

    kind: str
    why: str = ""
    booking: IntakeView | None = None
    disruption: TriageView | None = None


class MessageView(Api):
    job_id: str
    channel: str
    body: str
    grounded: bool
    issues: list[str] = Field(default_factory=list)


class ParamView(Api):
    path: str
    value: float
    source: str
    note: str


class ApiError(Api):
    error: str
    detail: str
    remedy: str = ""


# ------------------------------------------------------------------- requests


class EventRequest(Api):
    """One recorded fact. Validated here so a malformed body never reaches the log."""

    kind: str
    target: str | None = None
    at: str | None = Field(default=None, description="HH:MM")
    until: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    minutes: int | None = None
    multiplier: float | None = None
    commitment_cost: float = 0.0
    reason: str = ""


class TextRequest(Api):
    """Free text from a dispatcher, for triage or intake."""

    text: str


class AcceptRequest(Api):
    """Events a dispatcher reviewed and agreed to record."""

    events: list[dict[str, object]] = Field(default_factory=list)
