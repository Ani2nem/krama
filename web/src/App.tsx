import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, setOwnerKey, subscribe } from "./api";
import { Calendar } from "./components/Calendar";
import { Logo } from "./components/Logo";
import { NotePanel } from "./components/NotePanel";
import { ConfigPanel } from "./components/ConfigPanel";
import { TodayPanel } from "./components/TodayPanel";
import { RouteMap } from "./components/RouteMap";
import type { Plan, Week, World } from "./types";

type View = "board" | "map" | "crew";

/**
 * A violation as a dispatcher would say it.
 *
 * The checker writes for the checker: "[van_unavailable] van is scheduled during a
 * recorded outage (crew=crew-van-3 van=van-3)". Every part of that is useful and none
 * of it is a sentence. The code in brackets is the rule that fired, which matters in
 * a test and not on a phone call, and the parenthetical is the detail that actually
 * says which van.
 */
/** "2026-10-01T16:00:00-05:00" -> "4:00 PM". Nobody here reads military time. */
function twelveHour(iso: string): string {
  const hour24 = Number(iso.slice(11, 13));
  const hour = hour24 % 12 || 12;
  return `${hour}:${iso.slice(14, 16)} ${hour24 < 12 ? "AM" : "PM"}`;
}

function readable(violation: string): string {
  const withoutCode = violation.replace(/^\[[a-z_]+\]\s*/, "");
  const [, body = withoutCode, detail = ""] = withoutCode.match(/^(.*?)\s*\((.*)\)$/) ?? [];
  const parts = detail
    .split(/\s+/)
    .filter(Boolean)
    .map((pair) => pair.replace("=", " "))
    .join(", ");
  return parts ? `${body} - ${parts}` : body;
}

export default function App() {
  const [world, setWorld] = useState<World | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [view, setView] = useState<View>("board");
  const [day, setDay] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [week, setWeek] = useState<Week | null>(null);
  const [prefill, setPrefill] = useState<{
    text: string;
    nonce: number;
    run?: boolean;
  } | null>(null);
  const [configuring, setConfiguring] = useState(false);
  const [rotaStart, setRotaStart] = useState<string | null>(null);
  const [session, setSession] = useState<{ owner_pin_set: boolean; owner: boolean } | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const detailRef = useRef<HTMLDivElement>(null);
  const [editing, setEditing] = useState(false);
  const [crewChoices, setCrewChoices] = useState<
    | {
        worker_id: string;
        name: string;
        feasible: boolean;
        current?: boolean;
        cost_delta?: number;
        note?: string;
      }[]
    | null
  >(null);
  const [showTranscript, setShowTranscript] = useState(false);

  // A different job means a fresh card: edit state must not leak between jobs.
  useEffect(() => {
    setEditing(false);
    setCrewChoices(null);
    setShowTranscript(false);
  }, [selected]);

  // A floating card should yield to a click anywhere else - reaching for its Close
  // button is a chore the rest of the screen can do for free. Calendar blocks are
  // excluded: clicking another job means "show me that one", not "dismiss".
  useEffect(() => {
    if (!selected) return;
    function onDown(e: MouseEvent) {
      const target = e.target as Element;
      if (detailRef.current?.contains(target)) return;
      if (target.closest(".block")) return;
      setSelected(null);
    }
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [selected]);

  const refresh = useCallback(async () => {
    try {
      const [nextWorld, nextPlan, nextWeek, nextSession] = await Promise.all([
        api.world(rotaStart ?? undefined),
        api.plan(),
        api.week(),
        api.session(),
      ]);
      setWorld(nextWorld);
      setPlan(nextPlan);
      setWeek(nextWeek);
      setSession(nextSession);
      setError(null);
    } catch (exc) {
      setError(exc as ApiError);
    }
  }, [rotaStart]);

  useEffect(() => {
    void refresh();
    // Live updates, so a change made from the CLI or by an agent shows up here too.
    return subscribe(() => void refresh());
  }, [refresh]);

  // Every day that holds anything - committed routes AND booked-but-unplanned
  // windows - so the map's day picker never collapses to a lonely "all days"
  // while the calendar plainly shows work.
  const days = [
    ...new Set([
      ...(plan?.routes.map((r) => r.date) ?? []),
      ...(world?.jobs ?? [])
        .filter((j) => j.commitment_state === "confirmed" && j.window_start)
        .map((j) => j.window_start.slice(0, 10)),
    ]),
  ].sort();
  const dayLabel = (iso: string) =>
    new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, {
      weekday: "short",
      day: "numeric",
      month: "short",
    });
  const selectedStop =
    plan?.routes.flatMap((r) => r.stops).find((s) => s.job_id === selected) ?? null;
  const selectedJob = world?.jobs.find((j) => j.id === selected) ?? null;
  // Promises the committed plan can no longer keep. These are the board's loudest
  // fact: a customer was read a window, and the schedule as it stands will not
  // show up. Each one is a phone call someone owes.
  const brokenPromises =
    plan && world
      ? plan.unserved
          .filter((u) => u.is_failure)
          .map((u) => world.jobs.find((j) => j.id === u.job_id))
          .filter(
            (j): j is NonNullable<typeof j> => !!j && j.commitment_state === "confirmed",
          )
      : [];
  // Promises the plan SERVES at the wrong time - moved, with the customer not yet
  // told. Served-but-late hid from the unserved list, so Maria's 8 PM surprise
  // never reached the strip that exists for exactly this call.
  const movedPromises =
    plan && world
      ? plan.routes
          .flatMap((r) => r.stops)
          .flatMap((s) => {
            const job = world.jobs.find((j) => j.id === s.job_id);
            if (!job || job.commitment_state !== "confirmed" || !job.window_start) return [];
            const arrival = s.arrival;
            if (arrival >= job.window_start && arrival <= job.window_end) return [];
            if (brokenPromises.some((b) => b.id === job.id)) return [];
            return [{ job, arrival }];
          })
      : [];

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">
          <Logo size={30} />
          <h1>Krama</h1>
        </span>
        {plan ? (
          <div
            className="topbar__plan"
            title={`version ${plan.plan_id} \u00B7 ${plan.content_hash}`}
          >
            <span className={plan.feasible ? "ok" : "error"}>
              {plan.feasible ? "schedule holds" : "schedule broken"}
            </span>
            {!plan.redacted && (
              <span
                className="muted"
                title="what the planner expects this week to cost us in fuel and overtime - not revenue"
              >
                runs the week for ~${Math.round(plan.cost.total)}
              </span>
            )}
          </div>
        ) : (
          <span className="muted">no schedule committed yet</span>
        )}
        <div className="topbar__actions">
          {world?.calibration_warning && (
            // Every cost on screen rests on numbers nobody has validated, and that
            // belongs in front of the reader. It does not belong across the full width
            // in warning yellow: it is a standing caveat, not an incident, and a banner
            // that never changes stops being read within a day.
            <span className="chip chip--warn" title={world.calibration_warning}>
              estimated costs
            </span>
          )}
          {/* Unlocking lives in ONE place - the Owner controls section inside
              Configure. Out here only the unlocked state shows: an indicator that
              the owner's numbers are currently visible, and the one-click way to
              put them away before stepping off the desk. */}
          {session?.owner_pin_set && session.owner && (
            <button
              className="ownerchip"
              title="lock the owner view - margins and the rate card disappear again"
              onClick={async () => {
                setOwnerKey("");
                await refresh();
              }}
            >
              Owner · lock
            </button>
          )}
          <button disabled={busy} onClick={() => setConfiguring(true)}>
            Configure
          </button>
        </div>
      </header>

      {plan && !plan.feasible && (
        /* "1 violation(s)" is a true statement that tells a dispatcher nothing. What
           they need is what broke and the way out - so the way out lives HERE, not
           as a standing button in the topbar. Bookings, outages and claims all
           re-plan themselves; the one moment a human presses anything is the moment
           this banner is on screen, and a button that exists at any other time is a
           button someone has to wonder about. */
        <div className="banner banner--stale">
          <strong>The committed plan no longer works.</strong>{" "}
          Something recorded since it was made contradicts it:
          <ul>
            {plan.violations.map((violation) => (
              <li key={violation}>{readable(violation)}</li>
            ))}
          </ul>
          <button
            className="primary"
            disabled={busy}
            title="Rebuild the week around what was recorded. Promised windows are kept wherever possible; one that truly cannot be kept stays visibly broken for a phone call."
            onClick={async () => {
              setBusy(true);
              try {
                setPlan(await api.commit());
                setError(null);
              } catch (exc) {
                setError(exc as ApiError);
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy ? "Solving…" : "Rebuild around it"}
          </button>
        </div>
      )}
      {error && (
        <div className="banner banner--error">
          {error.message}
          {error.remedy && <span className="muted"> - {error.remedy}</span>}
          {error.code === "PromisesAtRisk" && (
            /* Moving a customer's promise is a decision with a phone call attached.
               It takes its own press - never a side effect of a rota click or an
               automatic rebuild, which is how Maria ended up planned for 8 PM
               without anybody deciding anything. */
            <button
              className="danger"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  setPlan(await api.commit(true));
                  setError(null);
                  await refresh();
                } catch (exc) {
                  setError(exc as ApiError);
                } finally {
                  setBusy(false);
                }
              }}
            >
              Move their windows - each one then needs a call
            </button>
          )}
        </div>
      )}

      {/* The calendar is the thing a dispatcher looks at all day, so it owns the
          screen: a fixed-viewport grid, no page scroll. The call box lives in a rail
          on the left - it is a stand-in until phones and email feed this directly,
          and a stand-in does not get the hero slot. The crew rota runs as a wide
          strip under the calendar, which is the shape the table actually is. */}
      <div className="layout">
        <aside className="rail">
          <NotePanel onChanged={() => void refresh()} prefill={prefill} />
        </aside>

        <div className="boardcol">
        {(brokenPromises.length > 0 || movedPromises.length > 0) && (
          <section className="asks">
            {movedPromises.map(({ job, arrival }) => (
              <div key={job.id} className="asks__row asks__row--broken">
                <div className="asks__what">
                  <strong>{job.customer_name} has not been told.</strong>
                  <span>
                    {" "}They were promised {job.window || "a window"} and the plan now has
                    them at {new Date(arrival).toLocaleString(undefined, {
                      weekday: "short",
                      hour: "numeric",
                      minute: "2-digit",
                    })}.{" "}
                  </span>
                  <span className="asks__who">
                    Call them{job.phone ? ` - ${job.phone}` : ""} - or find them a slot they
                    actually agreed to.
                  </span>
                </div>
                <div className="asks__actions">
                  <button
                    disabled={busy || !job.transcript}
                    title="re-price their original call right now - new slots to read them"
                    onClick={() =>
                      setPrefill({ text: job.transcript, nonce: Date.now(), run: true })
                    }
                  >
                    Find them a new slot
                  </button>
                  <button
                    disabled={busy}
                    title="open the booking on the calendar"
                    onClick={() => {
                      setView("board");
                      setSelected(job.id);
                    }}
                  >
                    Open the booking
                  </button>
                </div>
              </div>
            ))}
            {brokenPromises.map((j) => (
              <div key={j.id} className="asks__row asks__row--broken">
                <div className="asks__what">
                  <strong>We promised {j.customer_name}</strong>
                  <span> {j.window || "a window"} and the schedule can no longer make it. </span>
                  <span className="asks__who">
                    Call them{j.phone ? ` - ${j.phone} - ` : " "}before they find out the hard
                    way: offer a new slot, or fix what broke and this clears itself.
                  </span>
                </div>
                <div className="asks__actions">
                  <button
                    disabled={busy || !j.transcript}
                    title="re-price their original call right now - new slots to read them"
                    onClick={() =>
                      setPrefill({ text: j.transcript, nonce: Date.now(), run: true })
                    }
                  >
                    Find them a new slot
                  </button>
                  <button
                    disabled={busy}
                    title="open the promised booking on the calendar"
                    onClick={() => {
                      setView("board");
                      setSelected(j.id);
                    }}
                  >
                    Open the booking
                  </button>
                </div>
              </div>
            ))}
          </section>
        )}
        {world && world.crew_asks.length > 0 && (
          <section className="asks">
            {/* Unresolved promises to call a customer back. Red and persistent on
                purpose: each one is a person waiting by their phone. */}
            {world.crew_asks.map((ask) => (
              <div key={ask.ask_id} className="asks__row">
                <div className="asks__what">
                  <strong>{ask.customer || "A caller"}</strong>
                  {ask.phone && <span className="muted"> · {ask.phone}</span>}
                  <span> needs {ask.day} until {ask.until_label}</span>
                  {ask.extended.length > 0 && (
                    <span className="ok"> · {ask.extended.join(" and ")} said yes</span>
                  )}
                  {ask.candidates.some((c) => !ask.extended.includes(c.name)) ? (
                    <span className="asks__who">
                      {" "}· still to ask:{" "}
                      {ask.candidates
                        .filter((c) => !ask.extended.includes(c.name))
                        .map((c) => c.name)
                        .join(", ")}
                    </span>
                  ) : (
                    <span className="ok"> - ready to book</span>
                  )}
                </div>
                <div className="asks__actions">
                  {ask.candidates
                    .filter((c) => !ask.extended.includes(c.name))
                    .map((c) => (
                      <button
                        key={c.id}
                        disabled={busy}
                        title={`${c.name} agreed to stay until ${ask.until_label} that day`}
                        onClick={async () => {
                          setBusy(true);
                          try {
                            await api.extendAsk(ask.ask_id, c.id);
                            await refresh();
                          } catch (exc) {
                            setError(exc as ApiError);
                          } finally {
                            setBusy(false);
                          }
                        }}
                      >
                        {c.name} said yes
                      </button>
                    ))}
                  <button
                    disabled={busy || ask.extended.length < ask.crew_size}
                    title={
                      ask.extended.length < ask.crew_size
                        ? `this job needs ${ask.crew_size} fitters staying late - ${ask.crew_size - ask.extended.length} more yes${ask.crew_size - ask.extended.length === 1 ? "" : "es"} first`
                        : "re-price their call with the agreed overtime - the slots land in the call box on the left"
                    }
                    onClick={() =>
                      setPrefill({ text: ask.transcript, nonce: Date.now(), run: true })
                    }
                  >
                    Book them
                  </button>
                  <button
                    className="danger"
                    disabled={busy}
                    title="takes this off the board - nobody stays late, and the customer knows"
                    onClick={async () => {
                      setBusy(true);
                      try {
                        await api.closeAsk(ask.ask_id, "resolved");
                        await refresh();
                      } catch (exc) {
                        setError(exc as ApiError);
                      } finally {
                        setBusy(false);
                      }
                    }}
                  >
                    Told them no
                  </button>
                </div>
              </div>
            ))}
          </section>
        )}
        <main className="main">
          <div className="tabs">
            <button className={view === "board" ? "on" : ""} onClick={() => setView("board")}>
              Board
            </button>
            <button className={view === "map" ? "on" : ""} onClick={() => setView("map")}>
              Map
            </button>
            <button className={view === "crew" ? "on" : ""} onClick={() => setView("crew")}>
              Crew
            </button>
            {view === "map" && (
              <select value={day ?? ""} onChange={(e) => setDay(e.target.value || null)}>
                <option value="">all days</option>
                {days.map((d) => (
                  <option key={d} value={d}>
                    {dayLabel(d)}
                  </option>
                ))}
              </select>
            )}
          </div>

          {week && view === "board" && (
            <>
              {!selectedStop && selectedJob && (
                /* A promise no plan reaches: same card grammar, honest about the
                   gap. The dashed tile used to be wallpaper - visible, dead to
                   clicks - which made the one booking most in need of attention
                   the only one you could not open. */
                <div className="jobcard detail--overlay" ref={detailRef}>
                  <header className="jobcard__head">
                    <h3>{selectedJob.customer_name}</h3>
                    <span className="jobcard__state jobcard__state--confirmed">promised</span>
                    <button
                      className="jobcard__x"
                      aria-label="close"
                      onClick={() => setSelected(null)}
                    >
                      {"\u00D7"}
                    </button>
                  </header>
                  <dl>
                    <dt>Work</dt>
                    <dd>{selectedJob.service_type.replace(/_/g, " ")}</dd>
                    <dt>Address</dt>
                    <dd>{selectedJob.address || "-"}</dd>
                    <dt>Phone</dt>
                    <dd>{selectedJob.phone || "-"}</dd>
                    <dt>Quoted</dt>
                    <dd>
                      {selectedJob.quoted_total
                        ? `$${selectedJob.quoted_total.toFixed(2)} (tax included)`
                        : "-"}
                    </dd>
                    <dt>Promised</dt>
                    <dd>{selectedJob.window || "-"}</dd>
                    <dt>The run</dt>
                    <dd>
                      {brokenPromises.some((b) => b.id === selectedJob.id)
                        ? "not on the schedule - the promise stands, the current crew cannot reach it"
                        : "not scheduled yet - the plan has not reached this week"}
                    </dd>
                    <dt>Job</dt>
                    <dd><code>{selectedJob.id}</code></dd>
                  </dl>
                  {showTranscript && (
                    <div className="jobcard__transcript">
                      <p>{selectedJob.transcript || "(no transcript stored)"}</p>
                      {selectedJob.booking_note && (
                        <p className="jobcard__quotednote">
                          Quoted as: {selectedJob.booking_note} (as things stood when booked)
                        </p>
                      )}
                    </div>
                  )}
                  <div className="jobcard__actions">
                    <button
                      className="jobcard__primary"
                      disabled={busy || !selectedJob.transcript}
                      title="re-price their original call right now - new slots to read them"
                      onClick={() => {
                        setPrefill({
                          text: selectedJob.transcript,
                          nonce: Date.now(),
                          run: true,
                        });
                        setSelected(null);
                      }}
                    >
                      Find them a new slot
                    </button>
                    <span className="jobcard__quietrow">
                      <button
                        className="jobcard__quiet"
                        onClick={() => setShowTranscript((v) => !v)}
                      >
                        {showTranscript ? "Hide the call" : "Show the call"}
                      </button>
                      <button
                        className="jobcard__cancel"
                        disabled={busy}
                        onClick={async () => {
                          setBusy(true);
                          try {
                            await api.cancel(selectedJob.id);
                            setSelected(null);
                            await refresh();
                          } catch (exc) {
                            setError(exc as ApiError);
                          } finally {
                            setBusy(false);
                          }
                        }}
                      >
                        Cancel this booking
                      </button>
                    </span>
                  </div>
                </div>
              )}
              {selectedStop && (
                <div className="jobcard detail--overlay" ref={detailRef}>
                  {/* The Uber-modal grammar: white card, bold ink title, a quiet x,
                      clean rows, one black full-width primary at the bottom. The
                      old version wore three equal pill buttons in the header and
                      the loudest of them was the destructive one. */}
                  <header className="jobcard__head">
                    <h3>{selectedStop.customer_name}</h3>
                    <span className={`jobcard__state jobcard__state--${selectedStop.commitment_state}`}>
                      {selectedStop.commitment_state}
                    </span>
                    <button
                      className="jobcard__x"
                      aria-label="close"
                      onClick={() => setSelected(null)}
                    >
                      {"\u00D7"}
                    </button>
                  </header>
                  <dl>
                    <dt>Work</dt>
                    <dd>{selectedStop.service_type.replace(/_/g, " ")}</dd>
                    <dt>Address</dt>
                    <dd>{world?.jobs.find((j) => j.id === selectedStop.job_id)?.address || "-"}</dd>
                    <dt>Phone</dt>
                    <dd>{world?.jobs.find((j) => j.id === selectedStop.job_id)?.phone || "-"}</dd>
                    <dt>Quoted</dt>
                    <dd>
                      {(() => {
                        const total = world?.jobs.find(
                          (j) => j.id === selectedStop.job_id,
                        )?.quoted_total;
                        // Jobs seeded before prices were stored show a dash, not $0 -
                        // zero would read as "free", which nobody promised.
                        return total ? `$${total.toFixed(2)} (tax included)` : "-";
                      })()}
                    </dd>
                    <dt>On site</dt>
                    <dd>
                      {twelveHour(selectedStop.arrival)} to {twelveHour(selectedStop.departure)}
                    </dd>
                    <dt>Drive there</dt>
                    <dd>
                      <strong>from {selectedStop.from_label || "the shop"}</strong> ·{" "}
                      {selectedStop.travel_minutes} min · {selectedStop.travel_miles} mi
                      {selectedStop.next_label && (
                        <span className="muted"> · then {selectedStop.next_label}</span>
                      )}
                    </dd>
                    <dt>The run</dt>
                    <dd>
                      {/* Live, recomputed from the committed plan on every read -
                          a booking, a crew swap or a cancellation changes this
                          line the moment it changes the plan. The quote-time note
                          lives under "Show the call" with the other history. */}
                      {selectedStop.run_note || "-"}
                    </dd>
                    <dt>Crew</dt>
                    <dd>
                      {(() => {
                        // Who is actually rostered, not just how many the job needs.
                        // "one fitter" next to a calendar card naming two people read
                        // as a bug; the two are the van's crew, each aboard for their
                        // own stops.
                        const route = plan?.routes.find((r) =>
                          r.stops.some((st) => st.job_id === selectedStop.job_id),
                        );
                        const needs =
                          selectedStop.crew_size === 1 ? "needs one fitter" : `needs ${selectedStop.crew_size} fitters`;
                        return route
                          ? `${route.worker_names.join(" + ")} · ${route.van_id} (${needs})`
                          : needs;
                      })()}
                    </dd>
                    <dt>Job</dt>
                    <dd><code>{selectedStop.job_id}</code></dd>
                  </dl>
                  {showTranscript && (
                    <div className="jobcard__transcript">
                      <p>
                        {world?.jobs.find((j) => j.id === selectedStop.job_id)?.transcript ||
                          "(no transcript stored)"}
                      </p>
                      {(() => {
                        const note = world?.jobs.find(
                          (j) => j.id === selectedStop.job_id,
                        )?.booking_note;
                        return note ? (
                          <p className="jobcard__quotednote">
                            Quoted as: {note} (as things stood when booked)
                          </p>
                        ) : null;
                      })()}
                    </div>
                  )}

                  {editing && (
                    <div className="jobcard__edit">
                      {/* Who else could take it, each one a measured dollar delta
                          from a real trial solve with every promise still binding.
                          "Edit" used to cancel the booking before the dispatcher
                          had decided anything - the job vanished off the calendar
                          mid-thought. Nothing here destroys anything. */}
                      <p className="jobcard__editlabel">Change the fitter</p>
                      {crewChoices === null ? (
                        <p className="muted">pricing each option…</p>
                      ) : (
                        <div className="jobcard__crewlist">
                          {crewChoices.map((c) => (
                            <button
                              key={c.worker_id}
                              className={`jobcard__crewopt${c.current ? " on" : ""}`}
                              disabled={busy || !c.feasible || c.current}
                              title={c.note || ""}
                              onClick={async () => {
                                setBusy(true);
                                try {
                                  await api.setCrew(selectedStop.job_id, c.worker_id);
                                  setEditing(false);
                                  setCrewChoices(null);
                                  await refresh();
                                } catch (exc) {
                                  setError(exc as ApiError);
                                } finally {
                                  setBusy(false);
                                }
                              }}
                            >
                              {c.name}
                              {c.current
                                ? " · on it now"
                                : c.feasible
                                  ? ` · ${(c.cost_delta ?? 0) >= 0 ? "+" : "-"}$${Math.abs(c.cost_delta ?? 0).toFixed(2)}`
                                  : ` · ${c.note}`}
                            </button>
                          ))}
                        </div>
                      )}
                      <button
                        className="jobcard__rebook"
                        disabled={busy}
                        title="cancels this booking and puts the original call back in the box to re-quote - time, address, everything"
                        onClick={async () => {
                          const transcript =
                            world?.jobs.find((j) => j.id === selectedStop.job_id)?.transcript ??
                            "";
                          setBusy(true);
                          try {
                            await api.cancel(selectedStop.job_id);
                            setPrefill({ text: transcript, nonce: Date.now() });
                            setSelected(null);
                            await refresh();
                          } catch (exc) {
                            setError(exc as ApiError);
                          } finally {
                            setBusy(false);
                          }
                        }}
                      >
                        Rebook from the call… (cancels this booking)
                      </button>
                    </div>
                  )}

                  <div className="jobcard__actions">
                    <button
                      className="jobcard__primary"
                      disabled={busy}
                      onClick={() => {
                        const next = !editing;
                        setEditing(next);
                        if (next && crewChoices === null) {
                          void api
                            .crewOptions(selectedStop.job_id)
                            .then((r) => setCrewChoices(r.options))
                            .catch((exc) => setError(exc as ApiError));
                        }
                      }}
                    >
                      {editing ? "Done editing" : "Edit booking"}
                    </button>
                    <span className="jobcard__quietrow">
                      <button
                        className="jobcard__quiet"
                        onClick={() => setShowTranscript((v) => !v)}
                      >
                        {showTranscript ? "Hide the call" : "Show the call"}
                      </button>
                      <button
                        className="jobcard__cancel"
                        disabled={busy}
                        onClick={async () => {
                          setBusy(true);
                          try {
                            await api.cancel(selectedStop.job_id);
                            setSelected(null);
                            await refresh();
                          } catch (exc) {
                            setError(exc as ApiError);
                          } finally {
                            setBusy(false);
                          }
                        }}
                      >
                        Cancel this booking
                      </button>
                    </span>
                  </div>
                </div>
              )}
              <Calendar
                plan={plan}
                jobs={world?.jobs ?? []}
                selected={selected}
                onSelect={setSelected}
              />
            </>
          )}
          {plan && view === "map" && <RouteMap plan={plan} day={day} />}
          {world && view === "crew" && (
            <TodayPanel
              world={world}
              plan={plan}
              rotaStart={rotaStart}
              onWeek={setRotaStart}
              onChanged={() => void refresh()}
            />
          )}

          {plan && plan.unserved.length > 0 && (
            <section className="unserved">
              <h3>Unserved</h3>
              {/* Split deliberately: "could not fit" is a problem to act on,
                  "not this plan's work" is routine. */}
              {plan.unserved.filter((u) => u.is_failure).map((u) => (
                <p key={u.job_id} className="warn">
                  <strong>{u.customer_name || u.job_id}</strong> - {u.reason}: {u.detail}
                </p>
              ))}
              {plan.unserved.filter((u) => !u.is_failure).map((u) => (
                <p key={u.job_id} className="muted">
                  {u.customer_name || u.job_id} - {u.detail}
                </p>
              ))}
            </section>
          )}
        </main>

        </div>
      </div>

      {/* Somewhere to land. The quiet reference facts live here - where every route
          starts, what the calendar's colours mean, which plan is on screen - instead
          of crowding the surfaces people actually work on. */}
      {configuring && world && (
        <ConfigPanel
          world={world}
          owner={session?.owner ?? true}
          onClose={() => setConfiguring(false)}
          onChanged={() => void refresh()}
          onUnlock={async (tryPin) => {
            setOwnerKey(tryPin);
            const who = await api.session();
            if (!who.owner) {
              setOwnerKey("");
              return false;
            }
            await refresh();
            return true;
          }}
        />
      )}

      <footer className="footer">
        <span className="footer__brand">Krama</span>
        {world?.depot_address && <span>routes start and end at {world.depot_address}</span>}
        <div className="footer__legend">
          <span>
            <i className="footer__swatch footer__swatch--provisional" /> scheduled, not yet
            promised
          </span>
          <span>
            <i className="footer__swatch footer__swatch--confirmed" /> promised to the customer
          </span>
          <span>
            <i className="footer__swatch footer__swatch--done" /> already done (dulled)
          </span>
          <span>
            <i className="footer__swatch footer__swatch--dispatched" /> crew on the way
          </span>
          <span>
            <i className="footer__swatch footer__swatch--ot" /> runs past shift (overtime)
          </span>
        </div>
        {plan && (
          <span className="footer__plan" title="plan version and content hash">
            {plan.plan_id} · {plan.content_hash.slice(0, 8)}
          </span>
        )}
      </footer>
    </div>
  );
}
