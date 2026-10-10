import { useState } from "react";

import { api } from "../api";
import type { Plan, World } from "../types";

/**
 * The crew, as a rota rather than a list.
 *
 * Days across, one row per fitter, a symbol per skill. The cells are the controls:
 * click a day to mark that fitter out for that day, click an out day to bring them
 * back. The first version put a popup menu on the row with choices like "rest of
 * today", which answered the wrong question - absence has a date, and the dates are
 * already on the screen. The popup also opened inside the table's scroll frame,
 * where overflow clipped it invisible. Cells cannot be clipped by their own table.
 */

/** One symbol per SKILL - the three genuine specialties this business gates work
 * on. Not certifications: the only certifying body in this trade is for auto glass,
 * which we do not do. Screen repair needs hands, not a badge, so it gates nothing;
 * "tempered safety" gated nothing and was a phantom chip three fitters wore for no
 * reason. A skill in the domain but not in this map simply does not render. */
const CERT_BADGES: Record<string, { icon: string; label: string }> = {
  residential_glazing: { icon: "\u{1F3E0}", label: "residential glazing" },
  commercial_storefront: { icon: "\u{1F3E2}", label: "commercial storefront" },
  shower_door: { icon: "\u{1F6BF}", label: "shower door" },
};

/** "12" from "2026-10-12" - the rota header shows real dates, because "Mon"
 * alone stops meaning anything the moment the week arrows exist. */
function dayOfMonth(iso: string): number {
  return Number(iso.slice(8, 10));
}

/** "12 - 16 Oct" for the bar, so the week on display is never a guess. */
function weekLabel(days: { date: string }[]): string {
  const head = days[0];
  const tail = days[days.length - 1];
  if (!head || !tail) return "";
  const first = new Date(`${head.date}T12:00:00`);
  const last = new Date(`${tail.date}T12:00:00`);
  const month = last.toLocaleDateString(undefined, { month: "short" });
  if (first.getMonth() === last.getMonth()) {
    return `${first.getDate()} - ${last.getDate()} ${month}`;
  }
  const firstMonth = first.toLocaleDateString(undefined, { month: "short" });
  return `${first.getDate()} ${firstMonth} - ${last.getDate()} ${month}`;
}

function shiftWeek(fromIso: string | null, weeks: number): string {
  const base = fromIso ? new Date(`${fromIso}T12:00:00`) : new Date();
  base.setDate(base.getDate() + weeks * 7);
  return [
    base.getFullYear(),
    String(base.getMonth() + 1).padStart(2, "0"),
    String(base.getDate()).padStart(2, "0"),
  ].join("-");
}

export function TodayPanel({
  world,
  plan,
  rotaStart,
  onWeek,
  onChanged,
}: {
  world: World;
  plan: Plan | null;
  /** First day of the rota week on display; null means "from today". Holiday
   * lives in the future, so the rota walks weeks like the calendar does. */
  rotaStart: string | null;
  onWeek: (start: string | null) => void;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  /** The cell being edited, with its editor rendered BELOW the table - a floating
   * popup inside the table's scroll frame gets clipped invisible (learned once
   * already). kind distinguishes a fitter's cell from a van's; spans are the
   * hour-scoped holes already recorded on that day. */
  const [picked, setPicked] = useState<{
    kind: "worker" | "van";
    id: string;
    name: string;
    date: string;
    day: string;
    spans: [string, string][];
    extended: string;
  } | null>(null);
  const [outFrom, setOutFrom] = useState("08:00");
  const [outTo, setOutTo] = useState("10:00");

  /** Out sick, van won't start - recorded as the same events everything else already
   * understands. When the backend absorbed the outage on its own - rerouted with
   * every promise kept - it says so, and that sentence is worth showing: the
   * difference between "the schedule broke" and "the schedule healed" is the
   * difference between a task and a notification. */
  /** "Ken texted back yes" - the dispatcher relays the reply with one tap. First
   * claim wins server-side, so two taps in two tabs cannot double-book an evening. */
  async function claim(jobId: string, workerId: string) {
    setBusy(true);
    try {
      const result = await api.claimOvertime(jobId, workerId);
      setNote(`${result.worker} took the overtime - ${result.status}`);
      onChanged();
    } catch (exc) {
      setNote((exc as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function record(event: Record<string, unknown>) {
    setBusy(true);
    try {
      const result = await api.recordEvent(event);
      let message = result.note ?? "";
      // The committed plan is stale the moment availability changes, and the next
      // thing a dispatcher did was hunt for the red banner's Rebuild button. Run
      // the same rebuild here: mark someone out - or bring them back - and the
      // board re-solves in the same click, promises binding as always.
      if (plan) {
        try {
          await api.commit();
          message = message ? `${message} - schedule rebuilt around it` : "schedule rebuilt";
        } catch (exc) {
          message = `${message ? `${message} - ` : ""}could not rebuild: ${(exc as Error).message}`;
        }
      }
      setNote(message);
      onChanged();
    } finally {
      setBusy(false);
    }
  }

  /** Mark one fitter or van out for one calendar day: midnight to midnight, so a
   * shift and any overtime reach are both covered, and no other day is touched. */
  function dayOut(kind: "worker" | "van", id: string, date: string) {
    setPicked(null);
    void record({
      kind: `${kind}-unavailable`,
      target: id,
      window_start: `${date}T00:00`,
      until: `${date}T23:59`,
    });
  }

  /** Bring one day back. The restore carries the day as a window, so an outage that
   * spans several days is carved around it rather than cancelled outright. */
  function dayBack(kind: "worker" | "van", id: string, date: string) {
    setPicked(null);
    void record({
      kind: `${kind}-restored`,
      target: id,
      window_start: `${date}T00:00`,
      window_end: `${date}T23:59`,
    });
  }

  /** "Dan is out 8 to 10" - the dentist-appointment case, typed right on the cell
   * instead of through the call box. Same event either way. */
  function hoursOut(kind: "worker" | "van", id: string, date: string, from: string, to: string) {
    setPicked(null);
    void record({
      kind: `${kind}-unavailable`,
      target: id,
      window_start: `${date}T${from}`,
      until: `${date}T${to}`,
    });
  }

  /** Clear exactly one recorded hole - "the appointment moved" - leaving any other
   * absence that day alone. */
  function hoursBack(kind: "worker" | "van", id: string, date: string, span: [string, string]) {
    setPicked(null);
    void record({
      kind: `${kind}-restored`,
      target: id,
      window_start: `${date}T${span[0]}`,
      window_end: `${date}T${span[1]}`,
    });
  }

  const days = world.workers[0]?.days ?? [];
  const byDate = new Map<string, Plan["routes"]>();
  for (const route of plan?.routes ?? []) {
    byDate.set(route.date, [...(byDate.get(route.date) ?? []), route]);
  }

  return (
    <section className="crewcard">
      <div className="rota__split">
      <div className="rota__left">
      <div className="crewcard__bar">
        <h2>Crew availability</h2>
        <span className="crewcard__range">{weekLabel(days)}</span>
        <span className="crewcard__nav">
          <button className="cal__arrow" onClick={() => onWeek(shiftWeek(rotaStart, -1))}>
            {"\u2039"}
          </button>
          <button
            className={`cal__today${rotaStart === null ? " on" : ""}`}
            onClick={() => onWeek(null)}
          >
            This week
          </button>
          <button className="cal__arrow" onClick={() => onWeek(shiftWeek(rotaStart, 1))}>
            {"\u203A"}
          </button>
        </span>
      </div>
      {note && <p className="rota__healed">{note}</p>}
      <p className="panel__hint">
        Click a day to mark a fitter (or a van) out - the whole day, or just some
        hours. Click an out day to bring them back. Anyone can stay up to two hours
        past shift on overtime; later than that we ask, never assume.
      </p>

      <div className="rota">
        <table>
          <thead>
            <tr>
              <th>fitter</th>
              {days.map((d) => (
                <th key={d.date}>
                  {d.day} <span className="rota__date">{dayOfMonth(d.date)}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {world.workers.map((worker) => (
              <tr key={worker.id}>
                <td className="rota__who">
                  <strong>{worker.name}</strong>
                  <span className="rota__certs">
                    {worker.certifications
                      .filter((cert) => CERT_BADGES[cert])
                      .map((cert) => (
                        <span key={cert} title={CERT_BADGES[cert]?.label}>
                          {CERT_BADGES[cert]?.icon}
                        </span>
                      ))}
                  </span>
                </td>
                {worker.days.map((d) =>
                  d.shift === "off" ? (
                    <td key={d.date} className="rota__cell rota__cell--rest">
                      off
                    </td>
                  ) : !d.actionable ? (
                    // The shift is already behind the clock. A click here could not
                    // change anything, so there is nothing to click.
                    <td
                      key={d.date}
                      className="rota__cell rota__cell--past"
                      title="already past - nothing left to block out"
                    >
                      {d.available ? d.shift : "out"}
                    </td>
                  ) : d.available ? (
                    <td
                      key={d.date}
                      className={`rota__cell${picked && picked.id === worker.id && picked.date === d.date ? " rota__cell--picked" : ""}`}
                    >
                      <button
                        className="rota__daybtn"
                        disabled={busy}
                        title={`${worker.name}, ${d.day} ${dayOfMonth(d.date)} - mark out (all day or hours)`}
                        onClick={() =>
                          setPicked({
                            kind: "worker",
                            id: worker.id,
                            name: worker.name,
                            date: d.date,
                            day: d.day,
                            spans: d.out_spans,
                            extended: d.extended,
                          })
                        }
                      >
                        {d.shift}
                        {d.extended && <span className="rota__extended">{d.extended}</span>}
                        {d.out_note && <span className="rota__partout">{d.out_note}</span>}
                        <span className="rota__hovermark">{"\u2715"}</span>
                      </button>
                    </td>
                  ) : (
                    <td key={d.date} className="rota__cell rota__cell--out">
                      <button
                        className="rota__daybtn rota__daybtn--out"
                        disabled={busy}
                        title={`${worker.name} is back on ${d.day} - restore`}
                        onClick={() => dayBack("worker", worker.id, d.date)}
                      >
                        out
                        <span className="rota__hovermark">{"\u21BA"}</span>
                      </button>
                    </td>
                  ),
                )}
              </tr>
            ))}
            {/* The vans, on the same rota. A breakdown is an availability fact with
                dates exactly like a sick day, and it was only recordable by typing
                into the call box - a clickable fact belongs on the clickable table. */}
            {world.vans.map((van) => (
              <tr key={van.id} className="rota__vanrow">
                <td className="rota__who">
                  <strong>{van.id}</strong>
                  <span className="rota__certs" title="van">{"\u{1F690}"}</span>
                </td>
                {van.days.map((d) =>
                  !d.actionable ? (
                    <td key={d.date} className="rota__cell rota__cell--past">
                      {d.available ? "up" : "down"}
                    </td>
                  ) : d.available ? (
                    <td
                      key={d.date}
                      className={`rota__cell${picked && picked.id === van.id && picked.date === d.date ? " rota__cell--picked" : ""}`}
                    >
                      <button
                        className="rota__daybtn"
                        disabled={busy}
                        title={`${van.id}, ${d.day} ${dayOfMonth(d.date)} - mark down (all day or hours)`}
                        onClick={() =>
                          setPicked({
                            kind: "van",
                            id: van.id,
                            name: van.id,
                            date: d.date,
                            day: d.day,
                            spans: d.out_spans,
                            extended: "",
                          })
                        }
                      >
                        up
                        {d.out_note && <span className="rota__partout">{d.out_note}</span>}
                        <span className="rota__hovermark">{"\u2715"}</span>
                      </button>
                    </td>
                  ) : (
                    <td key={d.date} className="rota__cell rota__cell--out">
                      <button
                        className="rota__daybtn rota__daybtn--out"
                        disabled={busy}
                        title={`${van.id} is running again on ${d.day} - restore`}
                        onClick={() => dayBack("van", van.id, d.date)}
                      >
                        down
                        <span className="rota__hovermark">{"\u21BA"}</span>
                      </button>
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>

        {picked && (
          <div className="rota__editor">
            <strong>
              {picked.name} · {picked.day} {dayOfMonth(picked.date)}
            </strong>
            <button
              disabled={busy}
              onClick={() => dayOut(picked.kind, picked.id, picked.date)}
            >
              {picked.kind === "van" ? "Down all day" : "Out all day"}
            </button>
            <span className="rota__hours">
              from{" "}
              <input
                type="time"
                value={outFrom}
                disabled={busy}
                onChange={(e) => setOutFrom(e.target.value)}
              />{" "}
              to{" "}
              <input
                type="time"
                value={outTo}
                disabled={busy}
                onChange={(e) => setOutTo(e.target.value)}
              />
              <button
                disabled={busy || outFrom >= outTo}
                title={outFrom >= outTo ? "the hours must run forward" : ""}
                onClick={() => hoursOut(picked.kind, picked.id, picked.date, outFrom, outTo)}
              >
                {picked.kind === "van" ? "Down these hours" : "Out these hours"}
              </button>
            </span>
            {picked.spans.map((span) => (
              <button
                key={span.join("-")}
                className="rota__spanback"
                disabled={busy}
                title="this hole only - the rest of the day stays as recorded"
                onClick={() => hoursBack(picked.kind, picked.id, picked.date, span)}
              >
                back {span[0]} - {span[1]} {"\u21BA"}
              </button>
            ))}
            {picked.extended && (
              <button
                className="rota__withdraw"
                disabled={busy}
                title={`${picked.name} said yes to staying late but can't any more - takes the agreement back; anything it was holding up will show as broken, visibly`}
                onClick={() => {
                  setPicked(null);
                  void record({
                    kind: "extension-withdrawn",
                    target: picked.id,
                    window_start: `${picked.date}T00:00`,
                  });
                }}
              >
                won't stay late after all
              </button>
            )}
            <button className="rota__editorclose" onClick={() => setPicked(null)}>
              Done
            </button>
          </div>
        )}
      </div>

      <p className="rota__legend">
        {Object.entries(CERT_BADGES).map(([key, badge]) => (
          <span key={key}>
            {badge.icon} {badge.label}
          </span>
        ))}
      </p>

      <div className="rota__vans">
        {world.vans.map((van) => (
          <span key={van.id} className={van.available ? "" : "warn"}>
            {van.id}
            {!van.available && " out"}
            <button
              className="rota__toggle"
              disabled={busy}
              title={
                van.available
                  ? `mark ${van.id} off the road until it is restored`
                  : `${van.id} is fixed - restore`
              }
              onClick={() =>
                void record({
                  kind: van.available ? "van-unavailable" : "van-restored",
                  target: van.id,
                })
              }
            >
              {van.available ? "\u2715" : "\u21BA"}
            </button>
          </span>
        ))}
      </div>

      {world.overtime_offers.length > 0 && (
        <div className="otboard">
          <h3>Overtime on offer</h3>
          {world.overtime_offers.map((offer) => (
            <div key={offer.job_id} className={`otboard__row otboard__row--${offer.status}`}>
              <span className="otboard__what">
                {offer.customer} · {offer.day} {offer.arrival} · ~
                {offer.overtime_minutes} min past shift
              </span>
              {offer.status === "claimed" ? (
                <span className="otboard__state ok">claimed by {offer.claimed_by}</span>
              ) : offer.status === "expired" ? (
                <span className="otboard__state">
                  nobody claimed by {offer.deadline} - stays with {offer.fallback}
                </span>
              ) : (
                <span className="otboard__state">
                  first yes by {offer.deadline} ·
                  {offer.offered_to.map((name, index) => (
                    <button
                      key={name}
                      className="otboard__claim"
                      disabled={busy}
                      title={`${name} texted back yes`}
                      onClick={() => void claim(offer.job_id, offer.offered_ids[index] ?? "")}
                    >
                      {name}
                    </button>
                  ))}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      </div>

      {byDate.size > 0 && (
        <div className="workload">
          <h3>How the week loads them</h3>
          {[...byDate.keys()].sort().map((date) => (
            <div key={date} className="workload__day">
              <span className="workload__dow">
                {new Date(`${date}T12:00:00`).toLocaleDateString(undefined, {
                  weekday: "short",
                })}
              </span>
              <ul>
                {(byDate.get(date) ?? []).map((route) => (
                  <li key={route.crew_id}>
                    <span className="workload__who">
                      {route.worker_names.join(" + ")}
                      <span className="muted"> {route.van_id}</span>
                    </span>
                    <span className="workload__stats">
                      <strong>{Math.round(route.utilization * 100)}%</strong> on site
                      <span className="muted"> · {route.travel_minutes}m driving</span>
                      {route.idle_minutes > 0 && (
                        <span className="warn"> · {route.idle_minutes}m idle</span>
                      )}
                      {route.overtime_minutes > 0 && (
                        <span className="warn"> · OT {route.overtime_minutes}m</span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
      </div>
    </section>
  );
}
