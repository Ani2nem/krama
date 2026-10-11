"""What each kind of job actually involves.

The division of labour that makes intake trustworthy: the model reads a caller's
description and picks a *service type*; this supplies the duration, the
certifications, the crew size and the parts. Asking a model "how long does a
shower-door install take" invites a plausible number with nothing behind it, and a
schedule built on invented durations is wrong in a way no invariant check can catch,
because every arrival time is internally consistent with a fiction.

Classification is something a small model is good at. Estimating is not. So it
classifies, and a table estimates.

Today this is a hand-written catalogue. The shape is deliberately the one a learned
version would have: once there are completed jobs, ``typical_duration_min`` becomes a
measured percentile of actuals per service type, and the confidence band narrows on
its own. Nothing above this module changes when that happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krama.domain.enums import Certification, GlassType, ServiceType


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """The known shape of one kind of work."""

    service_type: ServiceType
    typical_duration_min: int
    #: Plausible spread. Quoting the midpoint while knowing the range is what lets a
    #: dispatcher say "about two hours, maybe three" honestly.
    duration_range: tuple[int, int]
    required_certifications: frozenset[Certification] = frozenset()
    crew_size: int = 1
    #: Added per pane beyond the first. The dominant driver of duration in practice.
    per_extra_pane_min: int = 0
    typical_parts: tuple[str, ...] = ()
    requires_customer_present: bool = False
    #: Details worth asking about on the call, because getting them wrong costs a visit.
    ask_about: tuple[str, ...] = ()
    notes: str = ""

    #: What the glass and consumables cost us for a typical job of this kind, and per
    #: extra pane. Separate from labour because materials are a pass-through with a
    #: markup and labour is sold by the hour - mixing them makes a quote impossible to
    #: explain to a customer who asks why.
    materials_cost: float = 0.0
    materials_cost_per_extra_pane: float = 0.0


#: Tempered and laminated glass is usually made to measure rather than stocked, which
#: is what turns a routing problem into a lead-time problem.
MADE_TO_ORDER: frozenset[GlassType] = frozenset({GlassType.TEMPERED, GlassType.LAMINATED})
DEFAULT_LEAD_TIME_DAYS = 3


CATALOG: dict[ServiceType, CatalogEntry] = {
    ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT: CatalogEntry(
        service_type=ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT,
        materials_cost=180.0,
        materials_cost_per_extra_pane=120.0,
        typical_duration_min=120,
        duration_range=(75, 210),
        required_certifications=frozenset({Certification.RESIDENTIAL_GLAZING}),
        per_extra_pane_min=45,
        typical_parts=("annealed_std",),
        requires_customer_present=True,
        ask_about=("how many panes", "rough size", "ground floor or upstairs"),
        notes="Upstairs or oversized units may need a second pair of hands.",
    ),
    ServiceType.STOREFRONT_GLASS: CatalogEntry(
        service_type=ServiceType.STOREFRONT_GLASS,
        materials_cost=620.0,
        materials_cost_per_extra_pane=480.0,
        typical_duration_min=180,
        duration_range=(120, 300),
        required_certifications=frozenset({Certification.COMMERCIAL_STOREFRONT}),
        crew_size=2,
        per_extra_pane_min=90,
        typical_parts=("tempered_std",),
        ask_about=("opening hours", "pane size", "street or alley access"),
        notes="Almost always tempered, and almost always before the business opens.",
    ),
    ServiceType.AUTO_GLASS: CatalogEntry(
        service_type=ServiceType.AUTO_GLASS,
        materials_cost=240.0,
        materials_cost_per_extra_pane=0.0,
        typical_duration_min=90,
        duration_range=(60, 150),
        required_certifications=frozenset({Certification.AUTO_GLASS}),
        typical_parts=("auto_windshield",),
        ask_about=("make, model and year", "windscreen or side glass", "ADAS camera fitted"),
        notes="A camera behind the screen means recalibration and a longer slot.",
    ),
    ServiceType.EMERGENCY_BOARD_UP: CatalogEntry(
        service_type=ServiceType.EMERGENCY_BOARD_UP,
        materials_cost=45.0,
        materials_cost_per_extra_pane=25.0,
        typical_duration_min=45,
        duration_range=(30, 90),
        typical_parts=("board_up_kit",),
        ask_about=("is the property secure", "how many openings"),
        notes="Makes the property safe; the real repair is a second visit.",
    ),
    ServiceType.SCREEN_REPAIR: CatalogEntry(
        service_type=ServiceType.SCREEN_REPAIR,
        materials_cost=28.0,
        materials_cost_per_extra_pane=18.0,
        typical_duration_min=45,
        duration_range=(25, 90),
        # Deliberately ungated. A screen swap is something every fitter on the
        # roster can do; requiring a badge for it meant the board refused screen
        # jobs while three capable people sat free - scarcity the business invented
        # for itself. The SCREEN_REPAIR skill survives in the enum for old records.
        required_certifications=frozenset(),
        per_extra_pane_min=15,
        typical_parts=("screen_kit",),
        ask_about=("how many screens", "frames intact"),
    ),
    ServiceType.SHOWER_DOOR_INSTALL: CatalogEntry(
        service_type=ServiceType.SHOWER_DOOR_INSTALL,
        materials_cost=390.0,
        materials_cost_per_extra_pane=0.0,
        typical_duration_min=150,
        duration_range=(105, 240),
        required_certifications=frozenset({Certification.SHOWER_DOOR}),
        typical_parts=("shower_kit",),
        requires_customer_present=True,
        ask_about=("frameless or framed", "has it been measured yet"),
        notes="Tempered by code, so an unmeasured job needs a survey visit first.",
    ),
    ServiceType.MEASURE_QUOTE: CatalogEntry(
        service_type=ServiceType.MEASURE_QUOTE,
        materials_cost=0.0,
        materials_cost_per_extra_pane=0.0,
        typical_duration_min=30,
        duration_range=(20, 60),
        requires_customer_present=True,
        ask_about=("what is being measured",),
        notes="Cheap to serve and easy to cluster with nearby work.",
    ),
}


@dataclass(frozen=True, slots=True)
class Estimate:
    """A catalogue entry resolved against the specifics of one call."""

    entry: CatalogEntry
    duration_min: int
    confidence_min: int
    crew_size: int
    required_certifications: frozenset[Certification]
    parts: tuple[str, ...] = ()
    lead_time_days: int = 0
    requires_customer_present: bool = False
    missing_details: tuple[str, ...] = field(default_factory=tuple)
    #: Minutes added for how close the van can get - already inside duration_min,
    #: carried separately so the card can say "+15 for street parking".
    access_buffer_min: int = 0
    access_label: str = ""

    @property
    def needs_ordering(self) -> bool:
        return self.lead_time_days > 0


#: What hard access costs in crew-minutes. Carrying a storefront pane from a
#: street bay, or up from an alley, is real labour on every such job - so it is
#: scheduled and billed like the rest of the work instead of living as a prose
#: note the solver never reads. The intake agent only ever CLASSIFIES what the
#: caller said; the minutes are fixed here, where they can be argued with.
ACCESS_BUFFER_MIN: dict[str, int] = {
    "easy": 0,  # driveway, loading dock, parking at the door
    "street_parking": 15,  # park on the street, carry to the door
    "difficult": 25,  # alley only, stairs, upper floor, long carry
}

ACCESS_LABEL: dict[str, str] = {
    "street_parking": "street parking",
    "difficult": "tricky access",
}


def lookup(
    service_type: ServiceType,
    *,
    pane_count: int = 1,
    glass_type: GlassType | None = None,
    known_details: frozenset[str] = frozenset(),
    site_access: str = "",
) -> Estimate:
    """Resolve a service type into concrete scheduling facts.

    Made-to-order glass is the interesting case: it converts a routing question into
    a lead-time one, and no amount of clever scheduling can install a pane that has
    not been cut yet.
    """
    entry = CATALOG[service_type]
    extra = max(0, pane_count - 1)
    access_buffer = ACCESS_BUFFER_MIN.get(site_access, 0)
    duration = entry.typical_duration_min + extra * entry.per_extra_pane_min + access_buffer
    if site_access:
        # The caller answered the access question, whatever words they used.
        known_details = known_details | {"street or alley access"}
    low, high = entry.duration_range
    confidence = max(15, (high - low) // 2)

    certifications = set(entry.required_certifications)
    parts = list(entry.typical_parts)
    lead_time = 0
    if glass_type in MADE_TO_ORDER:
        # Lead time only - no extra badge. "Tempered safety" was a phantom
        # certification the skills cleanup retired: it gated nothing real, the
        # Configure roster can no longer grant it, and leaving it here made every
        # made-to-order job permanently unbookable on any roster built in the
        # product. Handling tempered glass is the trade, not a certificate.
        lead_time = DEFAULT_LEAD_TIME_DAYS
        parts = [f"{glass_type.value}_custom"]

    # A large multi-pane job needs a second person regardless of the base crew size.
    crew = entry.crew_size
    if pane_count >= 4 and crew < 2:
        crew = 2

    return Estimate(
        entry=entry,
        duration_min=duration,
        access_buffer_min=access_buffer,
        access_label=ACCESS_LABEL.get(site_access, ""),
        confidence_min=confidence,
        crew_size=crew,
        required_certifications=frozenset(certifications),
        parts=tuple(parts),
        lead_time_days=lead_time,
        requires_customer_present=entry.requires_customer_present,
        missing_details=tuple(d for d in entry.ask_about if d not in known_details),
    )


def describe_for_prompt() -> str:
    """The catalogue as prompt context, so the model classifies against real options."""
    lines: list[str] = []
    for entry in CATALOG.values():
        certs = ", ".join(sorted(c.value for c in entry.required_certifications)) or "none"
        lines.append(
            f"- {entry.service_type.value}: ~{entry.typical_duration_min}min, "
            f"crew {entry.crew_size}, certs {certs}" + (f". {entry.notes}" if entry.notes else "")
        )
    return "\n".join(lines)
