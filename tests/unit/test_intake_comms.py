"""Intake and comms.

Both are about the same discipline from opposite ends. Intake refuses to let the model
supply numbers; comms refuses to let it supply facts. In each case the model does the
part it is good at - classifying, phrasing - and code does the part where being
plausibly wrong is expensive.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from krama.agents.comms import (
    CommsResult,
    DraftMessage,
    draft_customer_messages,
    verify_grounding,
)
from krama.agents.intake import (
    CommitmentSignal,
    IntakeResult,
    commitment_cost_for,
    intake,
)
from krama.agents.llm.scripted import ScriptedLLMProvider
from krama.config import BusinessParams
from krama.domain.catalog import CATALOG, lookup
from krama.domain.diff import ChangeKind, JobChange, Placement, PlanDiff
from krama.domain.enums import (
    Certification,
    CommitmentState,
    GlassType,
    ServiceType,
)
from krama.domain.models import TimeWindow
from krama.fixtures.sample_business import _at
from krama.geocoding import Geocoder
from krama.obs.correlation import dispatch

#: Must match BUSINESS_TZ in the fixture, or a message saying 15:10 is checked
#: against a plan that reads 13:10 and every grounded draft is rejected.
TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 9, 21, 9, 0, tzinfo=TZ)


@pytest.fixture
def business() -> BusinessParams:
    return BusinessParams.load()


@pytest.fixture
def geocoder() -> Geocoder:
    """Cache-only: fixture addresses resolve, anything else fails. Keeps tests offline."""
    return Geocoder(allow_network=False)


CALL = {
    "customer_name": "Sarah Chen",
    "phone": "817-555-0142",
    "address": "2201 N Tarrant Pkwy, Fort Worth, TX",
    "service_type": "residential_window_replacement",
    "description": "Two front-room windows broken",
    "pane_count": 2,
    "property_type": "residential",
    "urgency": "normal",
    "preferred_timing": "Tuesday morning",
    "commitment_signals": ["time_off_work"],
    "commitment_quotes": ["I'd have to take the morning off work"],
    "details_covered": ["how many panes"],
}


def run_intake(
    payload: dict[str, Any],
    business: BusinessParams,
    geocoder: Geocoder,
    text: str = "call notes",
) -> IntakeResult:
    """Drive intake with a scripted extraction.

    ``text`` matters more than it looks: commitment quotes are now checked against the
    transcript, so a test asserting a priced promise has to put the promise in the
    caller's mouth, exactly as production would see it.
    """
    provider = ScriptedLLMProvider.returning(payload)
    with dispatch("d-intake"):
        return intake(provider, text, business=business, now=NOW, geocoder=geocoder)


# --------------------------------------------------------------------- catalogue


def test_every_service_type_is_in_the_catalogue():
    """A service the model can name but the catalogue cannot price would force a
    guessed duration back into the system."""
    assert set(CATALOG) == set(ServiceType)


def test_duration_scales_with_panes_not_with_the_model():
    one = lookup(ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT, pane_count=1)
    three = lookup(ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT, pane_count=3)
    assert three.duration_min > one.duration_min


def test_made_to_order_glass_becomes_a_lead_time():
    """The case that turns a routing problem into a procurement one. No amount of
    clever scheduling installs a pane that has not been cut."""
    estimate = lookup(ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT, glass_type=GlassType.TEMPERED)
    assert estimate.needs_ordering
    assert estimate.lead_time_days > 0
    # And ONLY a lead time. The retired "tempered safety" badge used to ride along
    # here, which made every made-to-order job unbookable on a roster built in
    # Configure - the certification no longer exists anywhere a person can grant it.
    assert Certification.TEMPERED_SAFETY not in estimate.required_certifications
    assert estimate.required_certifications == frozenset({Certification.RESIDENTIAL_GLAZING})


def test_a_big_multi_pane_job_needs_a_second_pair_of_hands():
    assert lookup(ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT, pane_count=5).crew_size == 2


def test_the_catalogue_reports_what_was_not_asked():
    estimate = lookup(
        ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT,
        known_details=frozenset({"how many panes"}),
    )
    assert "how many panes" not in estimate.missing_details
    assert estimate.missing_details


# ------------------------------------------------------------------------ intake


def test_intake_builds_a_bookable_draft(business, geocoder):
    result = run_intake(CALL, business, geocoder)
    assert result.bookable
    assert result.draft is not None
    assert result.draft.customer_name == "Sarah Chen"


def test_scheduling_numbers_come_from_the_catalogue_not_the_call(business, geocoder):
    """The model is never asked how long a job takes. A schedule built on an invented
    duration is wrong in a way no invariant check can catch, because every arrival
    time is internally consistent with the fiction."""
    result = run_intake(CALL, business, geocoder)
    expected = lookup(ServiceType.RESIDENTIAL_WINDOW_REPLACEMENT, pane_count=2)
    assert result.draft is not None
    assert result.draft.estimated_duration_min == expected.duration_min
    assert result.draft.required_certifications == expected.required_certifications
    assert result.draft.crew_size == expected.crew_size


def test_the_schema_has_no_field_for_duration():
    """Enforced by construction rather than by instruction. A model cannot supply a
    number it is given nowhere to put."""
    from krama.agents.intake import CallExtraction

    fields = set(CallExtraction.model_fields)
    assert not {"duration", "duration_minutes", "estimated_duration_min"} & fields


def test_a_commitment_signal_becomes_a_priced_promise(business, geocoder):
    """ "I'd have to take the morning off" is the phrase that motivated the whole
    field. No dropdown captures it and the solver cannot infer it."""
    result = run_intake(
        CALL,
        business,
        geocoder,
        text="Hi, it's Sarah Chen. Two windows went in - I'd have to take the morning off work.",
    )
    assert result.draft is not None
    assert result.draft.commitment_cost == business.commitment.time_off_work.value
    assert result.commitment_quotes == ("I'd have to take the morning off work",)


def test_the_model_does_not_choose_the_number(business):
    """It identifies which signal was expressed; the price is configuration."""
    assert (
        commitment_cost_for([CommitmentSignal.TIME_OFF_WORK], business)
        == business.commitment.time_off_work.value
    )


def test_commitment_cost_is_capped(business):
    """Without a ceiling, a caller listing every inconvenience could make one job
    effectively immovable and distort the whole week around it."""
    every = list(CommitmentSignal)
    assert commitment_cost_for(every, business) == business.commitment.max_total.value


def test_repeated_signals_do_not_stack(business):
    once = commitment_cost_for([CommitmentSignal.WAITING_IN], business)
    twice = commitment_cost_for(
        [CommitmentSignal.WAITING_IN, CommitmentSignal.WAITING_IN], business
    )
    assert once == twice


def test_no_signal_means_no_commitment_cost(business, geocoder):
    result = run_intake(
        {**CALL, "commitment_signals": [], "commitment_quotes": []}, business, geocoder
    )
    assert result.commitment_cost == 0.0
    assert result.draft is not None
    assert result.draft.commitment_cost == 0.0


def test_missing_details_become_questions_to_ask_on_the_call(business, geocoder):
    """A dispatcher needs "ask for a callback number" while the customer is still on
    the line, not a half-filled form discovered later."""
    result = run_intake({**CALL, "phone": ""}, business, geocoder)
    assert not result.bookable
    assert any("callback number" in q for q in result.ask_next)


def test_an_unknown_service_type_is_not_guessed_at(business, geocoder):
    result = run_intake({**CALL, "service_type": "glass_blowing"}, business, geocoder)
    assert result.draft is None
    assert any("what kind of work" in m for m in result.missing_required)


def test_an_unresolvable_address_is_reported_not_invented(business, geocoder):
    result = run_intake({**CALL, "address": "the blue house with the gate"}, business, geocoder)
    assert result.draft is None
    assert result.geocode_note


def test_an_escalated_extraction_still_tells_the_dispatcher_what_to_do(business, geocoder):
    provider = ScriptedLLMProvider.always({"urgency": "extremely"})
    with dispatch("d"):
        result = intake(provider, "notes", business=business, now=NOW, geocoder=geocoder)
    assert result.draft is None
    assert result.extraction.escalated
    assert result.ask_next


def test_repair_count_reaches_the_job_provenance(business, geocoder):
    """Extraction quality travels with the record it produced, so a later audit can
    ask which jobs came from a struggling model."""
    provider = ScriptedLLMProvider.returning({**CALL, "urgency": "catastrophic"}, CALL)
    with dispatch("d"):
        result = intake(provider, "notes", business=business, now=NOW, geocoder=geocoder)
    assert result.draft is not None
    assert result.draft.provenance.extractor_retries == 1


# ------------------------------------------------------------------------ comms


def change_at(hour: int, minute: int, *, window: TimeWindow | None = None) -> JobChange:
    return JobChange(
        job_id="j-402",
        kind=ChangeKind.RETIMED,
        before=Placement(
            on_date=_at(0, 9).date(),
            crew_id="A",
            van_id="van-1",
            worker_ids=("w-dan",),
            arrival=_at(0, 9, 33),
            departure=_at(0, 11, 33),
        ),
        after=Placement(
            on_date=_at(0, 9).date(),
            crew_id="A",
            van_id="van-1",
            worker_ids=("w-dan",),
            arrival=_at(0, hour, minute),
            departure=_at(0, hour + 2, minute),
        ),
        commitment_state=CommitmentState.CONFIRMED,
        customer_name="Sarah Chen",
        promised_window=window,
    )


def draft(body: str, diff: PlanDiff, job_id: str = "j-402") -> CommsResult:
    provider = ScriptedLLMProvider.always(
        {"messages": [{"job_id": job_id, "channel": "sms", "body": body}]}, times=4
    )
    with dispatch("d-comms"):
        return draft_customer_messages(provider, diff, tz=TZ, reason="a van broke down")


def test_a_grounded_message_is_safe_to_send():
    result = draft("Moving you to Monday 15:10.", PlanDiff(changes=(change_at(15, 10),)))
    assert result.safe_to_send
    assert result.issues == ()


def test_an_invented_time_is_caught():
    """A plausible wrong time in a text message is worse than no message: the customer
    believes it and arranges their day around it."""
    result = draft("We'll be with you Monday at 3pm sharp.", PlanDiff(changes=(change_at(15, 10),)))
    assert not result.safe_to_send
    assert any(i.phrase.lower() == "3pm" for i in result.issues)


def test_the_bare_hour_is_allowed_only_when_it_is_exact():
    """3pm means 15:00. Allowing it for 15:10 would also have allowed it for 15:55."""
    assert draft("See you Monday at 3pm.", PlanDiff(changes=(change_at(15, 0),))).safe_to_send


def test_a_precise_time_is_not_misread_as_a_shorter_one():
    """ "3:10pm" once matched as "10pm" because the colon creates a word boundary, and
    a correctly grounded message was held for a phrase it never contained."""
    result = draft("We'll be with you Monday at 3:10pm.", PlanDiff(changes=(change_at(15, 10),)))
    assert result.safe_to_send, [i.phrase for i in result.issues]


def test_an_invented_day_is_caught():
    result = draft("Moving you to Thursday 15:10.", PlanDiff(changes=(change_at(15, 10),)))
    assert any(i.phrase.lower() == "thursday" for i in result.issues)


def test_a_change_inside_the_promised_window_drafts_nothing():
    """If the visit still fits the window the customer was given - arrives after it
    opens AND finishes before it closes - they have not been let down and there is
    nothing to tell them. The original version of this test promised 14:00-16:00 and
    called a 15:10-17:10 visit "inside", which is the arrival-only reading of a
    promise that the diff layer has since been cured of."""
    window = TimeWindow(start=_at(0, 14), end=_at(0, 18))
    provider = ScriptedLLMProvider.always({"messages": []})
    with dispatch("d"):
        result = draft_customer_messages(
            provider, PlanDiff(changes=(change_at(15, 10, window=window),)), tz=TZ
        )
    assert result.drafts == () and result.issues == ()
    assert provider.call_count == 0


def test_the_window_a_customer_was_given_may_be_restated_as_context():
    """Moving outside the promised window does need a message, and that message may
    legitimately refer to the slot they were originally given."""
    window = TimeWindow(start=_at(0, 9), end=_at(0, 11))
    result = draft(
        "You were down for 09:00-11:00 on Monday; we now need to move you to 15:10.",
        PlanDiff(changes=(change_at(15, 10, window=window),)),
    )
    assert result.safe_to_send, [i.phrase for i in result.issues]


def test_a_message_to_an_unaffected_customer_is_caught():
    """Worse than a wrong time: nothing changed for them, and a message says otherwise."""
    result = draft(
        "We're running late Monday 15:10.", PlanDiff(changes=(change_at(15, 10),)), job_id="j-999"
    )
    assert not result.safe_to_send
    assert any("did not change" in i.detail for i in result.issues)


def test_a_missing_message_is_caught():
    provider = ScriptedLLMProvider.always({"messages": []}, times=4)
    with dispatch("d"):
        result = draft_customer_messages(provider, PlanDiff(changes=(change_at(15, 10),)), tz=TZ)
    assert not result.safe_to_send
    assert any("no message drafted" in i.detail for i in result.issues)


def test_nothing_is_drafted_when_no_customer_is_affected():
    """Provisional work moving is invisible by definition."""
    provider = ScriptedLLMProvider.always({"messages": []})
    with dispatch("d"):
        result = draft_customer_messages(provider, PlanDiff(changes=()), tz=TZ)
    assert result.drafts == () and result.issues == ()
    assert provider.call_count == 0, "should not have asked the model at all"


def test_grounding_is_checked_on_the_text_not_the_intent():
    """The only kind of check a model cannot argue with."""
    issues = verify_grounding(
        DraftMessage(job_id="j-1", channel="sms", body="See you at 09:00 Friday."),
        allowed={"15:10", "monday"},
    )
    assert {i.phrase.lower() for i in issues} == {"09:00", "friday"}


# ------------------------------------------------- a model that will not say no


@pytest.mark.parametrize(
    "written",
    ["N/A", "n/a", "N/A.", "none", "None", "unknown", "not given", "TBD", "-", "  ", "?"],
)
def test_a_model_writing_nothing_is_read_as_nothing(written: str):
    """Asked for a field it was not told, a model would rather answer than leave a
    blank. "N/A" is perfectly truthy, and that is the whole bug."""
    from krama.agents.intake import CallExtraction

    assert CallExtraction(phone=written).phone == ""


@pytest.mark.parametrize("written", ["817-555-0142", "Maria", "2nd ave", "0", "N/A Glass Co"])
def test_a_real_answer_survives(written: str):
    """The check has to be exact. A customer called "None Ltd" is a stretch, but a
    company with N/A in its name is not, and clipping it would be a worse bug."""
    from krama.agents.intake import CallExtraction

    assert CallExtraction(customer_name=written).customer_name == written


def test_a_missing_number_is_asked_for_rather_than_filled_in():
    """The failure as a dispatcher met it.

    A caller said "callback" and never gave the number. The model wrote "N/A", which
    counted as answered: not in the missing list, not in "still to ask", shown on
    screen as a value. The dispatcher hangs up without the number and the job is
    unbookable for a reason nobody was told.
    """
    from krama.agents.intake import REQUIRED_FIELDS, CallExtraction

    call = CallExtraction(
        customer_name="Maria",
        phone="N/A",
        address="2nd ave",
        service_type="storefront_glass",
    )
    missing = [label for field, label in REQUIRED_FIELDS if not getattr(call, field)]
    assert missing == ["a callback number"]


@pytest.mark.parametrize(
    "written",
    ["Nguyen Glass", "ask Maria", "call the shop", "12345", "N/A", "the mobile"],
)
def test_a_phone_field_that_could_not_be_dialled_is_blank(written: str):
    """Blanking "N/A" was not enough, and this is what came next.

    Told to produce a phone and given no number, the model does not give up - it
    reaches for the nearest string and writes the company name in. Phone: Nguyen Glass
    is worse than Phone: N/A, because it looks like data.
    """
    from krama.agents.intake import CallExtraction

    assert CallExtraction(phone=written).phone == ""


@pytest.mark.parametrize(
    "written",
    ["817-555-0142", "(206) 555 0142", "+44 20 7946 0958", "1-913-295-2960", "9132952960"],
)
def test_a_number_someone_could_actually_ring_survives(written: str):
    """The bar is "could this be dialled", not "does it match a format". A validator
    strict enough to reject a real customer is a worse bug than the one it fixes."""
    from krama.agents.intake import CallExtraction

    assert CallExtraction(phone=written).phone == written


@pytest.mark.parametrize("written", ["894892894", "555-0142", "123456789012345"])
def test_a_number_nobody_could_dial_is_blank(written: str):
    """Nine digits passed a seven-digit floor, was displayed, and would have been
    dialled. A US number is ten digits; a floor that verifies less verifies nothing."""
    from krama.agents.intake import CallExtraction

    assert CallExtraction(phone=written).phone == ""


@pytest.mark.parametrize("written", ["maria", "maria@", "@glass.com", "maria@glass", "n/a"])
def test_an_address_without_an_at_is_not_an_email(written: str):
    from krama.agents.intake import CallExtraction

    assert CallExtraction(email=written).email == ""


def test_a_real_email_survives():
    from krama.agents.intake import CallExtraction

    assert CallExtraction(email="maria@nguyenglass.com").email == "maria@nguyenglass.com"


def test_the_twelve_hour_form_the_brief_uses_is_grounded():
    """Caught by the eval gate, which is what it is for.

    Converting the board to twelve-hour time also changed the brief the comms agent
    reads. The model duly echoed "9:00", the allowed set held only "09:00" and
    "9:00am", and tier 4 fell from 97.9% to 72.9% - a grounded message reported as a
    fabrication because the checker and the brief had drifted apart.
    """
    from krama.agents.comms import verify_grounding

    draft = DraftMessage(
        job_id="j-402", channel="sms", body="We can be with you at 9:00 on Monday."
    )
    allowed = {"09:00", "9:00am", "9:00", "monday"}
    assert verify_grounding(draft, allowed) == ()


def test_a_bare_hour_is_still_refused_for_a_time_that_is_not_on_the_hour():
    """The looser twelve-hour form must not reopen the rounding hole: "3pm" for a
    15:10 slot would also have passed for 15:55."""
    from krama.agents.comms import (
        _facts_for,  # noqa: F401  (import guard)
        verify_grounding,
    )

    draft = DraftMessage(job_id="j-402", channel="sms", body="See you at 3pm.")
    assert verify_grounding(draft, {"15:10", "3:10pm", "3:10"}) != ()


def test_a_service_the_business_does_not_sell_is_a_referral_not_a_quote(business, geocoder):
    """ "He needs his car glass repaired" was drafted, priced and nearly booked.
    Krama fits glass in buildings; the config says so, and the gate is
    deterministic - a perfect extraction of a service we do not sell must come
    back unbookable with a referral message, never a slot list."""
    result = run_intake(
        {**CALL, "service_type": "auto_glass", "property_type": "commercial"},
        business,
        geocoder,
    )
    assert result.not_offered, "the refusal must be explicit"
    assert "auto glass" in result.not_offered
    assert not result.bookable, "no pricing, no slots, no crew ask for a referral"

    offered = run_intake(CALL, business, geocoder)
    assert not offered.not_offered, "residential work books exactly as before"


def test_access_is_minutes_on_the_schedule_not_a_prose_note():
    """ "Street parking only" means someone carries a storefront pane down the
    block - real crew-minutes, so they land in the duration the solver schedules
    and the quote bills, not in a note nobody prices. And the answer counts as
    answered: the board must not ask about access the caller just described."""
    from krama.domain.catalog import lookup
    from krama.domain.enums import ServiceType

    plain = lookup(ServiceType.STOREFRONT_GLASS)
    kerb = lookup(ServiceType.STOREFRONT_GLASS, site_access="street_parking")
    alley = lookup(ServiceType.STOREFRONT_GLASS, site_access="difficult")
    dock = lookup(ServiceType.STOREFRONT_GLASS, site_access="easy")

    assert kerb.duration_min == plain.duration_min + 15
    assert kerb.access_buffer_min == 15 and kerb.access_label == "street parking"
    assert alley.duration_min == plain.duration_min + 25
    assert dock.duration_min == plain.duration_min, "a loading dock costs nothing extra"

    assert "street or alley access" in plain.missing_details
    for answered in (kerb, alley, dock):
        assert "street or alley access" not in answered.missing_details


def test_the_access_answer_is_heard_even_when_the_model_forgets():
    """The deterministic backstop: a caller who said "it's alley access only"
    answered the access question, whether or not the model ticked the box."""

    from krama.agents.intake import _ANSWERED_BY

    pattern = _ANSWERED_BY["street or alley access"]
    for said in (
        "it's alley access only",
        "you can park on the street out front",
        "there's a loading dock around back",
        "park right at the door",
        "we have a driveway",
    ):
        assert pattern.search(said), said
    assert not pattern.search("the glass is broken and we open at nine")


def test_the_parking_situation_is_heard_even_when_the_box_is_unticked():
    """Rosa said "you have to park on the street out front"; the model marked the
    access question answered but left site_access blank, so the question vanished
    AND the minutes never landed. The classification is checkable, so it is
    checked."""
    from krama.agents.intake import CallExtraction, _hear_the_parking_situation

    blank = CallExtraction()
    said = {
        "you have to park on the street out front": "street_parking",
        "there's no lot here": "street_parking",
        "it's alley access only": "difficult",
        "there's a loading dock around back": "easy",
        "park right at the door": "easy",
        # An alley with a dock is still an alley - the hard read wins.
        "alley in the back with a loading dock": "difficult",
    }
    for text, expected in said.items():
        assert _hear_the_parking_situation(blank, text).site_access == expected, text
    assert _hear_the_parking_situation(blank, "the glass is cracked").site_access == ""
    # The model's own answer is never second-guessed.
    told = CallExtraction(site_access="easy")
    assert _hear_the_parking_situation(told, "alley only").site_access == "easy"
