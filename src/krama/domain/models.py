"""Core entities for the dispatch domain.

Conventions used throughout:

* Every ``datetime`` is timezone-aware UTC. Local business hours are expressed as
  naive ``time`` values plus the business timezone, and resolved at solve time.
* Durations are whole minutes (``int``). Distances are miles (``float``).
* Money is dollars (``float``) at the domain boundary. The solver converts to
  integer cents internally because CP-SAT objectives must be integral.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time
from typing import Annotated

import pygeohash
from pydantic import BaseModel, ConfigDict, Field, model_validator

from krama.domain.enums import (
    Certification,
    CommitmentState,
    GlassType,
    Priority,
    PropertyType,
    ServiceType,
    UnservedReason,
    WindowHardness,
)

EARTH_RADIUS_MILES = 3958.7613

WorkerId = Annotated[str, Field(min_length=1)]
VanId = Annotated[str, Field(min_length=1)]
JobId = Annotated[str, Field(min_length=1)]
CustomerId = Annotated[str, Field(min_length=1)]


class Frozen(BaseModel):
    """Base for immutable domain values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Location(Frozen):
    """A geocoded point. ``geohash7`` is the travel-matrix cache key.

    Two addresses inside the same ~150m geohash-7 cell have effectively identical
    drive times, which is what keeps the travel cache small enough to be free.
    """

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    address: str = ""
    geocode_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    #: How precisely this was resolved: "house", "road", "area", or "" when unknown.
    #: A van can be sent to a house. "Haslet" is a town, and sending a crew to its
    #: centroid is not a booking, it is a guess with a price on it.
    precision: str = ""
    #: The street the geocoder matched, so a caller's words can be checked against
    #: what the map actually found. "16 Haslet, Texas" resolves to 16 Avondale Haslet
    #: Road - a real house the caller never named.
    matched_road: str = ""
    #: The named place the geocoder matched, when it matched one. "The Walmart in
    #: Haslet" is how a manager actually gives their address, and the name is what
    #: proves the match is the place they meant.
    matched_name: str = ""

    @property
    def geohash7(self) -> str:
        return str(pygeohash.encode(self.lat, self.lon, precision=7))

    @property
    def geohash5(self) -> str:
        """Coarse ~5km cell, used as the traffic-calibration corridor key."""
        return str(pygeohash.encode(self.lat, self.lon, precision=5))

    def haversine_miles(self, other: Location) -> float:
        """Great-circle distance. Used for k-nearest pruning and as a fallback bound."""
        lat1, lon1, lat2, lon2 = map(math.radians, (self.lat, self.lon, other.lat, other.lon))
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


class TimeWindow(Frozen):
    """A period during which a job may be started.

    ``hardness`` is the whole point of this type: "before we open at 9" is HARD and
    the solver must respect it; "any time Tuesday" is SOFT and may be violated for
    a per-minute lateness penalty.
    """

    start: datetime
    end: datetime
    hardness: WindowHardness = WindowHardness.SOFT

    @model_validator(mode="after")
    def _ordered(self) -> TimeWindow:
        if self.end <= self.start:
            raise ValueError(f"time window end {self.end} must be after start {self.start}")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("time windows must be timezone-aware")
        return self

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end


class DayHours(Frozen):
    """Local working hours for one weekday. ``weekday`` is 0=Monday .. 6=Sunday."""

    weekday: int = Field(ge=0, le=6)
    start: time
    end: time

    @model_validator(mode="after")
    def _ordered(self) -> DayHours:
        if self.end <= self.start:
            raise ValueError("working-hours end must be after start")
        return self


class Worker(Frozen):
    id: WorkerId
    name: str
    certifications: frozenset[Certification] = frozenset()
    working_hours: tuple[DayHours, ...] = ()
    home_location: Location
    overtime_eligible: bool = True
    loaded_cost_per_hour: float = Field(default=55.0, gt=0)
    #: Where the overtime offer lands. Empty means "no SMS for this person" - the
    #: offer still shows on the board, it just cannot be texted.
    phone: str = ""

    def hours_for(self, weekday: int) -> DayHours | None:
        return next((h for h in self.working_hours if h.weekday == weekday), None)

    def is_certified_for(self, required: frozenset[Certification]) -> bool:
        return required <= self.certifications


class Van(Frozen):
    id: VanId
    label: str
    rack_slots: int = Field(default=8, gt=0)
    stock: dict[str, int] = Field(default_factory=dict)
    home_depot: Location
    cost_per_mile: float = Field(default=0.65, ge=0)

    def has_stock(self, part_code: str, quantity: int) -> bool:
        return self.stock.get(part_code, 0) >= quantity


class Material(Frozen):
    """A part a job consumes. Lead time can make a job unschedulable regardless of routing."""

    part_code: str
    quantity: int = Field(default=1, gt=0)
    in_stock: bool = True
    lead_time_days: int = Field(default=0, ge=0)

    def available_from(self, ordered_on: date) -> date:
        if self.in_stock:
            return ordered_on
        return date.fromordinal(ordered_on.toordinal() + self.lead_time_days)


class GlassSpec(Frozen):
    pane_count: int = Field(default=1, gt=0)
    width_inches: float | None = None
    height_inches: float | None = None
    glass_type: GlassType | None = None
    frame_material: str | None = None


class Provenance(Frozen):
    """Where a job record came from and how much of it the extractor was sure about.

    ``field_confidence`` and ``missing_required`` drive the dispatcher's
    "ask the customer for X" prompts, and are the raw signal for extraction evals.
    """

    source_channel: str = "manual"
    received_at: datetime | None = None
    #: The words the booking came from, verbatim. A reschedule starts from what the
    #: caller originally said rather than from a dispatcher's memory of it, and when
    #: the source becomes an email or a recording, this is where it already lives.
    transcript: str = ""
    field_confidence: dict[str, float] = Field(default_factory=dict)
    missing_required: tuple[str, ...] = ()
    extractor_retries: int = 0


class Job(Frozen):
    id: JobId
    customer_id: CustomerId
    customer_name: str
    phone: str = ""
    email: str = ""
    location: Location
    property_type: PropertyType = PropertyType.RESIDENTIAL

    service_type: ServiceType
    description_raw: str = ""
    glass_spec: GlassSpec | None = None

    # Derived by the intake agent from the job catalog, never typed by a human.
    required_certifications: frozenset[Certification] = frozenset()
    crew_size: int = Field(default=1, ge=1, le=2)
    estimated_duration_min: int = Field(gt=0)
    duration_confidence_min: int = Field(default=0, ge=0)

    materials: tuple[Material, ...] = ()
    windows: tuple[TimeWindow, ...] = ()
    priority: Priority = Priority.NORMAL
    revenue: float = Field(default=0.0, ge=0)
    #: The tax-inclusive total read to the customer when they accepted the slot.
    #: ``revenue`` above is the pre-tax share and is what scheduling weighs; this is
    #: what the phone call actually said, kept verbatim because a price that exists
    #: only in the moment of quoting is a disconnection waiting for an invoice.
    quoted_total: float = Field(default=0.0, ge=0)
    #: Why this slot looked the way it did when the dispatcher pressed it - "already
    #: 3 stops nearby, +8 min detour; one fitter needed, qualified for residential
    #: glazing". A decision whose reasons evaporate with the slot card cannot be
    #: audited a week later; this is the quote-time rationale, labelled as such on
    #: the board because a later re-plan may legitimately choose differently.
    booking_note: str = ""

    commitment_state: CommitmentState = CommitmentState.DRAFT
    #: Dollar weight resisting a reschedule. This is where "I'll take off work that
    #: day" lands - a fact no dropdown captures and the solver cannot infer.
    commitment_cost: float = Field(default=0.0, ge=0)
    deferral_count: int = Field(default=0, ge=0)

    requires_customer_present: bool = False
    site_notes: str = ""
    requested_at: datetime
    provenance: Provenance = Provenance()

    @model_validator(mode="after")
    def _tz_aware(self) -> Job:
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")
        return self

    @property
    def is_active(self) -> bool:
        return self.commitment_state not in {
            CommitmentState.COMPLETED,
            CommitmentState.CANCELLED,
        }

    @property
    def hard_windows(self) -> tuple[TimeWindow, ...]:
        return tuple(w for w in self.windows if w.hardness is WindowHardness.HARD)

    def earliest_material_date(self, ordered_on: date) -> date:
        if not self.materials:
            return ordered_on
        return max(m.available_from(ordered_on) for m in self.materials)


class Stop(Frozen):
    """One job visit inside a crew's route, with concrete arrival and departure."""

    job_id: JobId
    arrival: datetime
    departure: datetime
    travel_minutes_from_prev: int = Field(ge=0)
    travel_miles_from_prev: float = Field(ge=0)
    #: True when the crew went back to the shop before this stop instead of
    #: idling hours at the previous customer's kerb. travel_*_from_prev then
    #: measures shop -> here, and run_break_* carries the trip home that ended
    #: the previous run - real driving, so it is real money.
    fresh_run: bool = False
    run_break_minutes: int = Field(default=0, ge=0)
    run_break_miles: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Stop:
        if self.departure < self.arrival:
            raise ValueError("stop departure precedes arrival")
        return self

    @property
    def service_minutes(self) -> int:
        return int((self.departure - self.arrival).total_seconds() // 60)


class CrewRoute(Frozen):
    """A crew (1-2 workers + one van) working one day, as an ordered list of stops."""

    crew_id: str
    date: date
    worker_ids: tuple[WorkerId, ...] = Field(min_length=1, max_length=2)
    van_id: VanId
    stops: tuple[Stop, ...] = ()
    return_to_depot_minutes: int = Field(default=0, ge=0)
    return_to_depot_miles: float = Field(default=0.0, ge=0)

    @property
    def total_travel_minutes(self) -> int:
        legs = sum(s.travel_minutes_from_prev + s.run_break_minutes for s in self.stops)
        return legs + self.return_to_depot_minutes

    @property
    def total_travel_miles(self) -> float:
        legs = sum(s.travel_miles_from_prev + s.run_break_miles for s in self.stops)
        return legs + self.return_to_depot_miles

    @property
    def job_ids(self) -> tuple[JobId, ...]:
        return tuple(s.job_id for s in self.stops)


class UnservedJob(Frozen):
    """A job the solver could not place, with a reason the dispatcher can act on."""

    job_id: JobId
    reason: UnservedReason
    detail: str = ""


class CostBreakdown(Frozen):
    """Objective decomposition in dollars, so a plan's price is explainable.

    Marginal cash: fuel, overtime, penalties. Rostered wages are deliberately
    absent - they are owed whether the plan runs or not, and a cost view that
    counted them taught the solver to hoard work onto one fitter's overtime.
    """

    vehicle: float = 0.0
    overtime: float = 0.0
    unserved_penalty: float = 0.0
    lateness_penalty: float = 0.0
    reschedule_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.vehicle
            + self.overtime
            + self.unserved_penalty
            + self.lateness_penalty
            + self.reschedule_penalty
        )


class PlanVersion(Frozen):
    """An immutable schedule snapshot.

    Every mutation produces a new version with a ``parent_id``, which gives a full
    audit chain and makes optimistic concurrency on the plan head straightforward.
    """

    id: str
    parent_id: str | None = None
    created_at: datetime
    horizon_start: date
    horizon_end: date
    routes: tuple[CrewRoute, ...] = ()
    unserved: tuple[UnservedJob, ...] = ()
    cost: CostBreakdown = CostBreakdown()
    label: str = ""
    solver_metrics: dict[str, float] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Stable digest of the scheduling decisions only (not metadata).

        Lets evals and the cold path detect "nothing actually changed" without
        comparing timestamps or ids.
        """
        payload = [
            {
                "crew": r.crew_id,
                "date": r.date.isoformat(),
                "workers": sorted(r.worker_ids),
                "van": r.van_id,
                "stops": [
                    {"job": s.job_id, "arr": s.arrival.isoformat(), "dep": s.departure.isoformat()}
                    for s in r.stops
                ],
            }
            for r in sorted(self.routes, key=lambda r: (r.date, r.crew_id))
        ]
        payload_unserved = sorted(u.job_id for u in self.unserved)
        blob = json.dumps(
            {"routes": payload, "unserved": payload_unserved}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def scheduled_job_ids(self) -> set[JobId]:
        return {job_id for route in self.routes for job_id in route.job_ids}

    def stop_for(self, job_id: JobId) -> tuple[CrewRoute, Stop] | None:
        for route in self.routes:
            for stop in route.stops:
                if stop.job_id == job_id:
                    return route, stop
        return None
