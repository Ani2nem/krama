import { useState } from "react";

import type { Job, Plan, Route, Stop } from "../types";

/**
 * The week, as a calendar: days across, time down.
 *
 * This replaces a horizontal Gantt, and the reason is the label problem rather than
 * taste. Laid out horizontally, a forty-five minute job is a forty-five pixel sliver
 * and "Rodriguez Storefront" cannot be written in it at any font size. I tried
 * scrolling the text on hover, which was worse: motion to read something that should
 * simply have been legible.
 *
 * Turned on its side the same job is a block forty-five pixels *tall* and the full
 * width of a day column, which is room for the customer, the time and the crew
 * without truncating any of them. The constraint was never the amount of text.
 *
 * Overlapping work inside one day splits the column, the way every calendar does it -
 * so two crews out at the same time sit side by side rather than on top of each other.
 */

//: Pixels per minute. Sets how tall the grid is, and therefore how much room the
//: shortest job has for its label. At 1.1 a forty-five minute job clears two lines.
const SCALE = 1.1;

/** 12-hour, because that is what this business says out loud. "Alex 8 to 17" makes a
 * dispatcher translate before they can use it, and translating is what a display is
 * supposed to have done already. */
const clock = (minute: number) => {
  const hour24 = Math.floor(minute / 60);
  const hour = hour24 % 12 || 12;
  return `${hour}:${String(minute % 60).padStart(2, "0")} ${hour24 < 12 ? "AM" : "PM"}`;
};

/** The same, without the meridiem - for the hour rail, where AM/PM on every line is
 * noise and the column reads top to bottom anyway. */
const hourLabel = (minute: number) => {
  const hour24 = Math.floor(minute / 60);
  return `${hour24 % 12 || 12} ${hour24 < 12 ? "am" : "pm"}`;
};

/** Local calendar date, not UTC - `toISOString` shifts the day east of UTC+12. */
function isoDate(value: Date): string {
  return [
    value.getFullYear(),
    String(value.getMonth() + 1).padStart(2, "0"),
    String(value.getDate()).padStart(2, "0"),
  ].join("-");
}

/** The Monday of the week holding `d` - the roster runs Monday to Friday. */
function mondayOf(d: Date): Date {
  const copy = new Date(d);
  copy.setDate(copy.getDate() - ((copy.getDay() + 6) % 7));
  return copy;
}

/** Where the board opens: this week on a weekday, next week on a weekend. The old
 * behaviour anchored to the committed plan's start, which on a Saturday showed last
 * Friday first - a diary opening on yesterday. */
function initialMonday(): string {
  const today = new Date();
  const monday = mondayOf(today);
  if (today.getDay() === 0 || today.getDay() === 6) monday.setDate(monday.getDate() + 7);
  return isoDate(monday);
}

/** Monday through Friday of the week starting at `monday`. */
function weekDates(monday: string): string[] {
  const dates: string[] = [];
  const cursor = new Date(`${monday}T12:00:00`);
  for (let i = 0; i < 5; i += 1) {
    dates.push(isoDate(cursor));
    cursor.setDate(cursor.getDate() + 1);
  }
  return dates;
}

/** The Mondays whose week begins inside the given month. */
function mondaysOf(year: number, month: number): string[] {
  const cursor = new Date(year, month, 1, 12);
  cursor.setDate(cursor.getDate() + ((8 - cursor.getDay()) % 7));
  const mondays: string[] = [];
  while (cursor.getMonth() === month) {
    mondays.push(isoDate(cursor));
    cursor.setDate(cursor.getDate() + 7);
  }
  return mondays;
}

/**
 * The grid spans the business's working day - 6 AM to 9 PM covers every shift, the
 * overtime reach and the pre-opening commercial starts - and widens itself if a stop
 * ever lands outside that. It renders at full height on a page that scrolls
 * naturally: an inner scrollbar on the one thing you look at all day read as a
 * patch, and "is the morning free?" is answered by looking, not by scrolling a
 * frame inside a frame.
 */
const GRID_START = 6 * 60;
const GRID_END = 21 * 60;

/** Clamp the grid to the day's actual work when it spills past the defaults.
 * Promised-but-unscheduled windows count as work: a ghost drawn for an evening
 * promise used to hang past the bottom of a grid sized only by the real stops. */
function gridBounds(routes: Route[], windows: [number, number][]): [number, number] {
  let start = GRID_START;
  let end = GRID_END;
  for (const route of routes) {
    for (const stop of route.stops) {
      start = Math.min(start, Math.floor(stop.start_minute / 60) * 60);
      end = Math.max(end, Math.ceil(stop.end_minute / 60) * 60);
    }
  }
  for (const [open, close] of windows) {
    start = Math.min(start, Math.floor(open / 60) * 60);
    end = Math.max(end, Math.ceil(close / 60) * 60);
  }
  return [start, end];
}

/** Minutes since midnight from an ISO datetime's clock part. */
function minutesOf(iso: string): number {
  return Number(iso.slice(11, 13)) * 60 + Number(iso.slice(14, 16));
}

interface Placed {
  stop: Stop;
  route: Route;
  column: number;
  columns: number;
}

/**
 * Lay a day's stops out in columns so nothing overlaps anything else.
 *
 * Events are grouped into clusters of work that actually overlaps, and only a cluster
 * splits. The first version split the whole day by its busiest moment, so three crews
 * out at nine in the morning made every block that day a third of a column wide -
 * including the lone afternoon job with nothing near it, which then had no room for
 * its label again. That is the bug this layout exists to avoid, reintroduced one
 * level down.
 */
function layout(routes: Route[]): Placed[] {
  const all = routes
    .flatMap((route) => route.stops.map((stop) => ({ stop, route })))
    .sort((a, b) => a.stop.start_minute - b.stop.start_minute);

  const out: Placed[] = [];
  let cluster: { stop: Stop; route: Route }[] = [];
  let clusterEnd = -Infinity;

  function flush() {
    if (cluster.length === 0) return;
    const ends: number[] = [];
    const placed = cluster.map(({ stop, route }) => {
      let column = ends.findIndex((end) => end <= stop.start_minute);
      if (column === -1) {
        column = ends.length;
        ends.push(stop.end_minute);
      } else {
        ends[column] = stop.end_minute;
      }
      return { stop, route, column };
    });
    const columns = Math.max(1, ends.length);
    out.push(...placed.map((p) => ({ ...p, columns })));
    cluster = [];
    clusterEnd = -Infinity;
  }

  for (const item of all) {
    // A gap with nothing running across it ends the cluster, so what follows starts
    // again at full width.
    if (cluster.length > 0 && item.stop.start_minute >= clusterEnd) flush();
    cluster.push(item);
    clusterEnd = Math.max(clusterEnd, item.stop.end_minute);
  }
  flush();
  return out;
}

function Block({
  placed,
  top,
  selected,
  onSelect,
}: {
  placed: Placed;
  top: number;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const { stop, route, column, columns } = placed;
  const height = Math.max(26, (stop.end_minute - stop.start_minute) * SCALE);
  // A visit whose departure is behind the clock is history, not an active
  // booking - it stays on the board for the record, but dulled, so the eye
  // reads the day's REMAINING work at a glance.
  const done = new Date(stop.departure).getTime() < Date.now();
  const width = 100 / columns;
  const crew = route.worker_names.join(" + ");

  return (
    <button
      className={`block block--${stop.commitment_state}${stop.past_shift ? " block--ot" : ""}${done ? " block--done" : ""}${selected ? " block--selected" : ""}`}
      style={{
        top: `${(stop.start_minute - top) * SCALE}px`,
        height: `${height}px`,
        left: `calc(${column * width}% + 2px)`,
        width: `calc(${width}% - 4px)`,
      }}
      onClick={() => onSelect(stop.job_id)}
      title={
        `${stop.customer_name} - ${stop.service_type.replace(/_/g, " ")}\n` +
        `${clock(stop.start_minute)}-${clock(stop.end_minute)}\n` +
        `${crew} · ${route.van_id}\n` +
        `${stop.travel_minutes} min drive, ${stop.travel_miles} mi` +
        (stop.past_shift ? "\nruns past shift - overtime" : "")
      }
    >
      {stop.past_shift && <span className="block__ot-tag">OT</span>}
      {done && <span className="block__done-tag">{"\u2713"} done</span>}
      <span className="block__what">{stop.customer_name}</span>
      {/* The shortest job on the fixture is forty-five minutes, which is fifty pixels
          tall and just holds a wrapped name plus the time. Below that the time goes
          first: knowing who it is matters more than knowing to the minute when, and
          both are in the tooltip and the detail panel regardless. */}
      {height > 44 && (
        <span className="block__when">
          {clock(stop.start_minute)} - {clock(stop.end_minute)}
        </span>
      )}
      {/* Only shown when the block is tall enough to hold it. A short job says who is
          on it in the tooltip and in the detail panel instead of squeezing three
          lines into two lines of space. */}
      {height > 58 && (
        <span className="block__who">
          {crew} <span className="block__van">{route.van_id}</span>
        </span>
      )}
    </button>
  );
}

export function Calendar({
  plan,
  jobs,
  selected,
  onSelect,
}: {
  /** Null before anything is committed. An empty diary still has a week in it, and a
   * blank panel is the least useful thing to show somebody whose diary is empty. */
  plan: Plan | null;
  /** Every active job - so a confirmed booking whose week no committed plan has
   * reached yet still draws, as a ghost, inside its promised window. A promise
   * with no tile is a promise someone will forget. */
  jobs: Job[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  /** The Monday on screen. Free navigation - the diary is not chained to whatever
   * week the committed plan happens to start in. */
  const [monday, setMonday] = useState(initialMonday);
  const shift = (weeks: number) => {
    const d = new Date(`${monday}T12:00:00`);
    d.setDate(d.getDate() + weeks * 7);
    setMonday(isoDate(d));
  };

  const dates = weekDates(monday);
  const anchor = new Date(`${monday}T12:00:00`);
  const monthValue = `${anchor.getFullYear()}-${anchor.getMonth()}`;
  // Two clicks to anywhere: pick a month, then one of its weeks.
  const monthOptions: { value: string; label: string }[] = [];
  {
    const base = new Date();
    base.setDate(1);
    base.setMonth(base.getMonth() - 1);
    for (let i = 0; i < 7; i += 1) {
      monthOptions.push({
        value: `${base.getFullYear()}-${base.getMonth()}`,
        label: base.toLocaleDateString(undefined, { month: "long", year: "numeric" }),
      });
      base.setMonth(base.getMonth() + 1);
    }
    if (!monthOptions.some((m) => m.value === monthValue)) {
      monthOptions.push({
        value: monthValue,
        label: anchor.toLocaleDateString(undefined, { month: "long", year: "numeric" }),
      });
    }
  }
  const weekOptions = mondaysOf(anchor.getFullYear(), anchor.getMonth());
  const today = isoDate(new Date());
  const routes = (plan?.routes ?? []).filter((r) => dates.includes(r.date));
  const planned = new Set(routes.flatMap((r) => r.stops.map((s) => s.job_id)));
  const ghosts = jobs.filter((j) => {
    if (j.commitment_state !== "confirmed" || planned.has(j.id) || !j.window_start) return false;
    return dates.includes(j.window_start.slice(0, 10));
  });
  const [start, end] = gridBounds(
    routes,
    ghosts.map((g) => [minutesOf(g.window_start), minutesOf(g.window_end)]),
  );
  const hours: number[] = [];
  for (let h = Math.ceil(start / 60) * 60; h <= end; h += 60) hours.push(h);

  const byDate = new Map<string, Route[]>();
  for (const route of routes) {
    byDate.set(route.date, [...(byDate.get(route.date) ?? []), route]);
  }

  return (
    <div className="cal">
      <div className="cal__bar">
        <h2 className="cal__month">
          {anchor.toLocaleDateString(undefined, { month: "long", year: "numeric" })}
        </h2>
        <button
          className={`cal__today${monday === initialMonday() ? " on" : ""}`}
          onClick={() => setMonday(initialMonday())}
        >
          Today
        </button>
        <button className="cal__arrow" onClick={() => shift(-1)} title="previous week">
          {"\u2039"}
        </button>
        <button className="cal__arrow" onClick={() => shift(1)} title="next week">
          {"\u203A"}
        </button>
        <span className="cal__pickers">
        <select
          value={monthValue}
          onChange={(e) => {
            const parts = e.target.value.split("-");
            const picked = mondaysOf(Number(parts[0]), Number(parts[1]));
            setMonday(picked[0] ?? monday);
          }}
          title="jump to a month"
        >
          {monthOptions.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <select value={monday} onChange={(e) => setMonday(e.target.value)} title="then a week">
          {(weekOptions.includes(monday) ? weekOptions : [...weekOptions, monday].sort()).map(
            (w) => {
              const d = new Date(`${w}T12:00:00`);
              const f = new Date(d);
              f.setDate(f.getDate() + 4);
              return (
                <option key={w} value={w}>
                  {d.getDate()} - {f.getDate()}
                </option>
              );
            },
          )}
        </select>
        </span>
      </div>
      <div className="cal__days">
        <div className="cal__corner" />
        {dates.map((date) => {
          const routes = byDate.get(date) ?? [];
          const jobs = routes.reduce((n, r) => n + r.stops.length, 0);
          const day = new Date(`${date}T12:00:00`);
          return (
            <div
              key={date}
              className={`cal__day${jobs === 0 ? " cal__day--free" : ""}${
                date === today ? " cal__day--today" : ""
              }`}
            >
              <span className="cal__dow">
                {day.toLocaleDateString(undefined, { weekday: "short" })}
              </span>
              <strong className="cal__date">{day.getDate()}</strong>
              <span className="cal__count">
                {jobs === 0 ? "free" : `${jobs} job${jobs === 1 ? "" : "s"}`}
              </span>
            </div>
          );
        })}
      </div>

      <div className="cal__grid" style={{ height: `${(end - start) * SCALE}px` }}>
        <div className="cal__gutter">
          {hours.map((h) => (
            <span key={h} style={{ top: `${(h - start) * SCALE}px` }}>
              {hourLabel(h)}
            </span>
          ))}
        </div>

        {dates.map((date) => (
          <div key={date} className="cal__column">
            {hours.map((h) => (
              <div key={h} className="cal__line" style={{ top: `${(h - start) * SCALE}px` }} />
            ))}
            {ghosts
              .filter((g) => g.window_start.slice(0, 10) === date)
              .map((g) => {
                const startMin = minutesOf(g.window_start);
                const endMin = minutesOf(g.window_end);
                // Belt and braces: never draw outside the column, whatever the data.
                const top = Math.max(0, (startMin - start) * SCALE);
                const height = Math.min(
                  (end - start) * SCALE - top,
                  Math.max(30, (endMin - startMin) * SCALE),
                );
                return (
                  <button
                    key={g.id}
                    className={`block block--ghost${selected === g.id ? " block--selected" : ""}`}
                    style={{
                      top: `${top}px`,
                      height: `${height}px`,
                      left: "2px",
                      width: "calc(100% - 4px)",
                    }}
                    title={`${g.customer_name} - promised, not yet on a committed plan`}
                    onClick={() => onSelect(g.id)}
                  >
                    <span className="block__what">{g.customer_name}</span>
                    <span className="block__when">promised · awaiting schedule</span>
                  </button>
                );
              })}
            {layout(byDate.get(date) ?? []).map((placed) => (
              <Block
                key={placed.stop.job_id}
                placed={placed}
                top={start}
                selected={selected === placed.stop.job_id}
                onSelect={onSelect}
              />
            ))}
          </div>
        ))}
      </div>

      {routes.length === 0 && (
        <p className="cal__empty">
          Nothing booked this week. Take a call on the left, or browse another week above.
        </p>
      )}
    </div>
  );
}
