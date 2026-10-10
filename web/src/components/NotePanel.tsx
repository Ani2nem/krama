import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { type Dictation, dictationSupported, startDictation } from "../dictation";
import { ApiError, api } from "../api";
import type { Note } from "../types";

/** The board speaks 12-hour; the API speaks unambiguous. Converting here keeps the
 * wire format machine-readable and the screen human-readable, rather than asking one
 * of them to compromise. */
function to24h(label: string): string {
  const [, rawHour = "0", minute = "00", meridiem = ""] =
    label.trim().match(/^(\d{1,2}):(\d{2})\s*([AaPp][Mm])?$/) ?? [];
  let hour = Number(rawHour) % 12;
  if (meridiem.toLowerCase() === "pm") hour += 12;
  return `${String(hour).padStart(2, "0")}:${minute}`;
}

/**
 * One box. Type what you just heard.
 *
 * There used to be two, side by side and identical: one booked jobs, one recorded
 * disruptions. The first person to use the board typed "Dan called, van 3 won't
 * start" into the booking one and got back a customer named Dan who wanted auto glass
 * fitted. Better labels would not have fixed that - a dispatcher writing down a phone
 * call should not have to know which of two agents wants it.
 *
 * So a classifier reads the note first and hands it to the right specialist. It says
 * which it chose and why, and one click overrides it, which is what makes routing by
 * model safe here: the worst case costs a click rather than a wrong job on the
 * schedule.
 */
function Field({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value || <em>{hint ?? "needed"}</em>}</dd>
    </>
  );
}

export function NotePanel({
  onChanged,
  prefill,
}: {
  onChanged: () => void;
  /** A reschedule hands back the original transcript, so nobody retypes a call they
   * already took. The nonce distinguishes "reschedule the same job again" from
   * "nothing new". With run, the call is read immediately - "Book them" on a crew
   * ask used to paste the text and silently wait for a second click on Read it,
   * which read as the button doing nothing at all. */
  prefill: { text: string; nonce: number; run?: boolean } | null;
}) {
  const [text, setText] = useState("");
  const [note, setNote] = useState<Note | null>(null);
  const [booked, setBooked] = useState<string | null>(null);
  /** Which list shows: what they asked for, or the cheaper days outside their
   * stated hours. The ordering of the asked-for list is not a choice any more -
   * cheapest leads unless the caller used urgency words, and the two sort
   * buttons that exposed it were two decisions nobody ever wanted to make. */
  const [slotTab, setSlotTab] = useState<"asked" | "flexible" | null>(null);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const dictation = useRef<Dictation | null>(null);

  function toggleDictation() {
    if (dictation.current) {
      dictation.current.stop();
      dictation.current = null;
      setListening(false);
      setInterim("");
      return;
    }
    const session = startDictation(
      (final) => setText((t) => (t ? t + " " : "") + final.trim()),
      setInterim,
      (reason) => {
        dictation.current = null;
        setListening(false);
        setInterim("");
        if (reason) setError(new ApiError(reason, "allow the microphone and try again"));
      },
    );
    if (session) {
      dictation.current = session;
      setListening(true);
    }
  }

  // A dictation session must not outlive the panel.
  useEffect(() => () => dictation.current?.stop(), []);

  useEffect(() => {
    if (prefill) {
      setText(prefill.text);
      setNote(null);
      if (prefill.run) {
        setBooked(null);
        void run(undefined, prefill.text);
      } else {
        setBooked("Rescheduling - their original call is below. Add what changed, then Read it.");
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefill]);
  const [busy, setBusy] = useState(false);
  const [held, setHeld] = useState(false);
  const [pendingDuplicate, setPendingDuplicate] = useState<{
    date: string;
    arrival: string;
    quotedTotal: number;
    note: string;
    crewIds: string[];
    message: string;
  } | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  async function run(kind?: "booking" | "disruption", spoken: string = text) {
    setBusy(true);
    setError(null);
    setBooked(null);
    try {
      setSlotTab(null); // a fresh call decides its own leading order
      setHeld(false);
      setNote(await api.note(spoken, kind));
    } catch (exc) {
      setError(exc as ApiError);
      setNote(null);
    } finally {
      setBusy(false);
    }
  }

  async function take(
    date: string,
    arrival: string,
    quotedTotal: number,
    note: string,
    crewIds: string[],
    allowDuplicate = false,
  ) {
    if (!draft) return;
    setBusy(true);
    try {
      const result = await api.book(
        draft,
        date,
        to24h(arrival),
        text,
        quotedTotal,
        note,
        crewIds,
        allowDuplicate,
      );
      setPendingDuplicate(null);
      // Say so. The button worked before this and looked like it had not, which is
      // the worst thing a button can do: the next thing anybody does is press it again.
      setBooked(`Booked ${result.customer} for ${result.when} - ${result.status}.`);
      setNote(null);
      setText("");
      onChanged();
    } catch (exc) {
      const err = exc as ApiError;
      if (err.message.includes("already has a booking")) {
        // The duplicate gate fired. Not an error to bury in red text - a question
        // for the human: second job, or an edit of the one that exists?
        setPendingDuplicate({ date, arrival, quotedTotal, note, crewIds, message: err.message });
      } else {
        setError(err);
      }
    } finally {
      setBusy(false);
    }
  }

  async function accept() {
    const events = note?.disruption?.events;
    if (!events?.length) return;
    setBusy(true);
    try {
      await api.acceptTriage(events);
      setNote(null);
      setText("");
      onChanged();
    } catch (exc) {
      setError(exc as ApiError);
    } finally {
      setBusy(false);
    }
  }

  const draft = note?.booking?.draft;
  const booking = note?.booking;
  const activeTab = slotTab ?? "asked";
  const orderedSlots = (() => {
    if (!booking) return [];
    if (activeTab === "flexible") return booking.flexible_slots;
    const copy = [...booking.slots];
    if (booking.asked_for_speed) {
      // They asked for speed: the calendar leads, price breaks ties.
      copy.sort((a, b) => a.date.localeCompare(b.date) || a.quote_total - b.quote_total);
    } else {
      // Money first, then the calendar: among equally priced days the earlier wins.
      copy.sort((a, b) => a.quote_total - b.quote_total || a.date.localeCompare(b.date));
    }
    return copy;
  })();
  const disruption = note?.disruption;
  const other = note?.kind === "disruption" ? "booking" : "disruption";

  return (
    <section className={`panel panel--note${note ? ` panel--${note.kind}` : ""}`}>
      <h2>Take a call</h2>
      <p className="panel__hint">
        A customer calling, or something going wrong - type or dictate it either way.
        This box stands in until the phone line feeds it directly.
      </p>
      <textarea
        value={text}
        rows={3}
        placeholder="Maria at Nguyen Glass, storefront pane smashed… / Dan called, van 3 won't start…"
        onChange={(e) => setText(e.target.value)}
      />
      {interim && <p className="dictation__interim">{interim}…</p>}
      <div className="note__actions">
        <button className="primary" disabled={busy || !text.trim()} onClick={() => void run()}>
          {busy ? "Reading…" : "Read it"}
        </button>
        {/* Dictation, because typing out a phone call defeats the point of simulating
            one. Chrome ships recognition for free; where the browser has none, the
            button simply is not there rather than there and broken. */}
        {dictationSupported() && (
          <button
            className={listening ? "mic mic--live" : "mic"}
            disabled={busy}
            onClick={toggleDictation}
            title={listening ? "stop dictating" : "dictate instead of typing"}
          >
            {listening ? "◼ Stop dictating" : "🎤 Dictate"}
          </button>
        )}
      </div>

      {booked && <p className="booked">{booked}</p>}

      {error && (
        <p className="error">
          {error.message}
          {error.remedy && <span className="muted"> - {error.remedy}</span>}
        </p>
      )}

      {note?.booking?.not_offered && (
        <p className="error">{note.booking.not_offered}</p>
      )}

      {note && (
        <div className="routed">
          <span className={`tag tag--${note.kind}`}>
            {note.kind === "booking" ? "new booking" : "disruption"}
          </span>
          <span className="muted">{note.why}</span>
          {/* The override. Without it, a misrouted note is a dead end and the
              dispatcher retypes it into a box that no longer exists. */}
          <button
            className="link"
            disabled={busy}
            onClick={() => void run(other as "booking" | "disruption")}
          >
            not a {note.kind}?
          </button>
        </div>
      )}

      {draft && booking && (
        <>
          {/* The non-negotiables, one fixed row each, whether heard or still needed.
              The layout does not reflow as fields fill in, which is what makes it
              readable at a glance mid-call - and the same slots hold when the source
              is an email or a transcript instead of typing. */}
          <dl className="fields">
            <Field label="Name" value={draft.customer_name} />
            <Field label="Phone" value={draft.phone} />
            <Field label="Address" value={draft.address} />
            <Field label="Work" value={draft.service_type.replace(/_/g, " ")} />
            <Field label="When" value={booking.when_text} hint="any time - worth asking" />
            {draft.duration_minutes > 0 && (
              <>
                <dt>Duration</dt>
                <dd>
                  {draft.duration_minutes} min ±{draft.duration_confidence}
                  <span className="muted"> from the catalogue</span>
                </dd>
                <dt>Crew</dt>
                <dd>{draft.crew_size} · {draft.certifications.join(", ") || "no certs"}</dd>
              </>
            )}
            {draft.lead_time_days > 0 && (
              <>
                <dt>Glass</dt>
                <dd className="warn">made to order, {draft.lead_time_days} day lead</dd>
              </>
            )}
          </dl>

          {draft.commitment_cost > 0 && (
            <div className="commitment">
              {/* This is not a fee, a collision, or anything the customer pays. It is
                  how the planner remembers a promise has a person attached: when some
                  future disruption forces a choice about who to move, this appointment
                  looks $N more expensive to touch than one where nobody arranged
                  anything. Only shown when the caller actually said so - the quote is
                  the evidence, checked against the transcript. */}
              <strong>Worth protecting.</strong> They said:
              {draft.commitment_quotes.map((quote) => (
                <p key={quote} className="quote">“{quote}”</p>
              ))}
              <p className="commitment__why muted">
                So if a breakdown later forces us to move someone, the planner will move
                this appointment last - it treats rescheduling them as costing $
                {draft.commitment_cost.toFixed(0)} of goodwill, not as free.
              </p>
            </div>
          )}

          {booking.ask_next.length > 0 && (
            <div className="ask">
              <h3>Still to ask</h3>
              <ul>{booking.ask_next.map((q) => <li key={q}>{q}</li>)}</ul>
            </div>
          )}

          {booking.unavailable.length > 0 && (
            <div className="ask">
              <h3>Days we cannot offer, and why</h3>
              <ul>
                {booking.unavailable.map((u) => (
                  <li key={u.day}>
                    <strong>{u.day}</strong> - {u.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {booking.possible_duplicates.length > 0 && (
            <div className="dupe">
              <strong>This caller already has a booking.</strong>
              <ul>
                {booking.possible_duplicates.map((d) => (
                  <li key={d.job_id}>
                    {d.customer} · {d.when} · {d.what}
                  </li>
                ))}
              </ul>
              Booking below creates a <strong>second job</strong> for them; to change
              the existing one, open it on the calendar and edit it there.
            </div>
          )}

          {pendingDuplicate &&
            /* A modal, not a line of red text - and rendered through a PORTAL to
               the document body, because inside the rail an ancestor's styles made
               position:fixed pin to the rail instead of the screen: the "popup"
               dimmed one column and the user rightly asked where it was. Backdrop
               click means never mind. */
            createPortal(
            <div
              className="dupemodal"
              onMouseDown={(e) => {
                if (e.target === e.currentTarget) setPendingDuplicate(null);
              }}
            >
              <div className="dupemodal__card">
                <h3>Already booked?</h3>
                <p className="dupemodal__msg">{pendingDuplicate.message}.</p>
                {booking.possible_duplicates.length > 0 && (
                  <ul className="dupemodal__list">
                    {booking.possible_duplicates.map((d) => (
                      <li key={d.job_id}>
                        <strong>{d.customer}</strong> · {d.when} · {d.what}
                      </li>
                    ))}
                  </ul>
                )}
                <p className="dupemodal__hint">
                  To change the existing booking instead, close this and edit it on
                  the calendar.
                </p>
                <button
                  className="dupemodal__primary"
                  disabled={busy}
                  onClick={() =>
                    void take(
                      pendingDuplicate.date,
                      pendingDuplicate.arrival,
                      pendingDuplicate.quotedTotal,
                      pendingDuplicate.note,
                      pendingDuplicate.crewIds,
                      true,
                    )
                  }
                >
                  Book anyway - it's a second job
                </button>
                <button
                  className="dupemodal__cancel"
                  disabled={busy}
                  onClick={() => setPendingDuplicate(null)}
                >
                  Never mind
                </button>
              </div>
            </div>,
            document.body,
          )}

          {booking.crew_ask && (
            <div className="crewask">
              {/* The call does not end with "no" - it ends with "let me check and
                  call you back". Holding the ask stores it on the board, so the
                  promise to call back survives a reload and a shift change. */}
              <p className="crewask__msg">{booking.crew_ask.message}</p>
              <button
                className="crewask__hold"
                disabled={busy || held}
                onClick={async () => {
                  if (!booking.crew_ask) return;
                  setBusy(true);
                  try {
                    await api.openAsk({
                      customer_name: draft.customer_name,
                      phone: draft.phone,
                      transcript: text,
                      on_date: booking.crew_ask.on_date,
                      until: booking.crew_ask.until,
                      candidate_ids: booking.crew_ask.candidate_ids,
                      detail: booking.crew_ask.message,
                    });
                    setHeld(true);
                    onChanged();
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                {held ? "On the board - tell them you'll call back" : "Hold while I check with the crew"}
              </button>
            </div>
          )}

          {booking.slots.length > 0 && (
            <div className="slots">
              <h3>Offer them</h3>
              <p className="slots__how">
                What they asked for, priced. The big number is the quote, tax included
                {booking.redacted
                  ? "."
                  : "; underneath is what we keep once the glass, wages and driving are paid."}
              </p>
              {booking.flexible_slots.length > 0 && (
                <div className="slots__tabs">
                  <button
                    className={activeTab === "asked" ? "on" : ""}
                    onClick={() => setSlotTab("asked")}
                  >
                    Their hours
                  </button>
                  <button
                    className={activeTab === "flexible" ? "on" : ""}
                    onClick={() => setSlotTab("flexible")}
                  >
                    Cheaper outside their hours ({booking.flexible_slots.length})
                  </button>
                </div>
              )}
              {booking.asked_for_speed && activeTab === "asked" && (
                <p className="muted">They asked for speed, so the earliest day leads.</p>
              )}
              {orderedSlots.map((slot, index) => (
                <div
                  key={slot.date + slot.window + slot.arrival}
                  className={`slot${index === 0 ? " slot--best" : ""}${slot.outside_preference ? " slot--flex" : ""}`}
                >
                  {slot.outside_preference && (
                    <div className="slot__flex">
                      outside their stated hours - saves them $
                      {(
                        Math.min(...booking.slots.map((other) => other.quote_total)) -
                        slot.quote_total
                      ).toFixed(2)}
                    </div>
                  )}
                  <div className="slot__day">
                    {slot.day}
                    {slot.needs_overtime && (
                      <span
                        className="slot__ot"
                        title="runs past a shift - the price covers it; ask the crew who wants the overtime"
                      >
                        overtime
                      </span>
                    )}
                  </div>
                  <div className="slot__cost" title="the quote: what the customer pays, tax included">
                    <span className="slot__cost-label">customer pays</span>
                    ${slot.quote_total.toFixed(2)}
                  </div>
                  <div className="slot__arrival">
                    arrive about <strong>{slot.arrival}</strong>
                    <span className="muted"> · promise {slot.window}</span>
                  </div>
                  <div className="slot__leg muted">
                    drive: {slot.leg_minutes} min · {slot.leg_miles} mi from {slot.from_label}
                  </div>
                  {!booking.redacted && (
                    <div className="slot__margin">
                      we keep <strong>${slot.margin.toFixed(2)}</strong>
                      <span className="muted">
                        {" "}
                        ({slot.margin_pct.toFixed(0)}%) after glass, wages and driving
                      </span>
                    </div>
                  )}
                  {slot.quote_lines.length > 0 && (
                    /* Owner sessions only: the labour and materials lines ARE the
                       rate card. A dispatcher quotes the total; the build-up stays
                       behind the same PIN as the prices it is made of. */
                    <details className="slot__breakdown">
                      <summary>how that price is built</summary>
                      <pre>{slot.quote_lines.join("\n")}</pre>
                    </details>
                  )}
                  <div className="slot__why muted">
                    {slot.reason}
                    <br />
                    {slot.crew} - {slot.crew_reason}
                  </div>
                  <button
                    className="primary slot__book"
                    disabled={busy}
                    onClick={() =>
                      void take(
                        slot.date,
                        slot.arrival,
                        slot.quote_total,
                        [slot.reason, slot.crew_reason].filter(Boolean).join(" \u00B7 "),
                        slot.crew_ids,
                      )
                    }
                  >
                    Book it
                  </button>
                </div>
              ))}
              {(() => {
                const theirs = booking.slots;
                if (booking.redacted || theirs.length < 2 || activeTab === "flexible") return null;
                const spread =
                  Math.max(...theirs.map((s) => s.margin)) -
                  Math.min(...theirs.map((s) => s.margin));
                if (spread < 1) return null;
                return (
                  <p className="muted">
                    The best of these days keeps ${spread.toFixed(2)} more than the worst.
                  </p>
                );
              })()}
            </div>
          )}

          {booking.repairs > 0 && (
            <p className="muted">The model needed {booking.repairs} repair attempt(s).</p>
          )}
          {booking.note && <p className="warn">{booking.note}</p>}
        </>
      )}

      {disruption && (
        <>
          {disruption.summary && <p>{disruption.summary}</p>}
          {disruption.events.length > 0 && (
            <ul className="events">
              {disruption.events.map((event, index) => (
                <li key={index}>
                  <code>{String(event.type)}</code>{" "}
                  {String(event.van_id ?? event.worker_id ?? event.job_id ?? "")}
                  {event.reason ? <span className="muted"> - {String(event.reason)}</span> : null}
                </li>
              ))}
            </ul>
          )}
          {disruption.question && <p className="warn">{disruption.question}</p>}
          {disruption.rejected.map((line) => (
            <p key={line} className="warn">{line}</p>
          ))}
          {disruption.repairs > 0 && (
            <p className="muted">The model needed {disruption.repairs} repair attempt(s).</p>
          )}
          {/* Nothing the agent extracted is recorded until a dispatcher agrees. An
              event is a fact in an append-only log, and a wrong one propagates into
              every plan that follows. */}
          {disruption.events.length > 0 && (
            <button className="primary" disabled={busy} onClick={() => void accept()}>
              Record {disruption.events.length} event(s)
            </button>
          )}
        </>
      )}
    </section>
  );
}
