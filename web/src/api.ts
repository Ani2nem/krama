// Typed client. Errors are surfaced with the server's own remedy text rather than a
// status code: the API already says what to do about each failure, and rewording it
// here would only lose that.

import type {
  Intake,
  Draft,
  Message,
  Note,
  Plan,
  Triage,
  Week,
  World,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly remedy: string = "",
    readonly status: number = 0,
    readonly code: string = "",
  ) {
    super(message);
  }
}

// The owner's PIN, held in memory only - closing the tab locks the board again.
// The server strips sensitive money regardless; this header is how the owner asks
// for it back.
let ownerKey = "";
export function setOwnerKey(key: string) {
  ownerKey = key;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(ownerKey ? { "X-Owner-Key": ownerKey } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = response.statusText;
    let remedy = "";
    let code = "";
    try {
      const body = await response.json();
      const payload = body.detail ?? body;
      detail = payload.detail ?? payload.error ?? detail;
      remedy = payload.remedy ?? "";
      code = payload.error ?? "";
    } catch {
      /* a non-JSON error body is still an error */
    }
    throw new ApiError(detail, remedy, response.status, code);
  }
  return (await response.json()) as T;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });

export const api = {
  world: (rotaStart?: string) =>
    request<World>(`/api/world${rotaStart ? `?rota_start=${rotaStart}` : ""}`),
  week: () => request<Week>("/api/week"),
  plan: () => request<Plan | null>("/api/plan"),
  commit: (movePromises = false) =>
    post<Plan>(`/api/plan/commit${movePromises ? "?move_promises=true" : ""}`),
  openAsk: (ask: Record<string, unknown>) => post<{ ask_id: string }>("/api/asks", ask),
  extendAsk: (askId: string, workerId: string) =>
    post<{ worker: string; until: string; status: string }>(
      `/api/asks/${encodeURIComponent(askId)}/extend?worker_id=${encodeURIComponent(workerId)}`,
    ),
  closeAsk: (askId: string, outcome: string) =>
    post<{ status: string }>(
      `/api/asks/${encodeURIComponent(askId)}/close?outcome=${encodeURIComponent(outcome)}`,
    ),
  configWorker: (worker: Record<string, unknown>) =>
    post<{ worker_id: string; status: string }>("/api/config/worker", worker),
  removeWorker: (workerId: string) =>
    request<{ status: string }>(`/api/config/worker/${encodeURIComponent(workerId)}`, {
      method: "DELETE",
    }),
  configVan: (van: Record<string, unknown>) =>
    post<{ van_id: string; status: string }>("/api/config/van", van),
  removeVan: (vanId: string) =>
    request<{ status: string }>(`/api/config/van/${encodeURIComponent(vanId)}`, {
      method: "DELETE",
    }),
  claimOvertime: (jobId: string, workerId: string) =>
    post<{ job_id: string; worker: string; status: string }>(
      `/api/overtime/${encodeURIComponent(jobId)}/claim?worker_id=${encodeURIComponent(workerId)}`,
    ),
  recordEvent: (event: Record<string, unknown>) =>
    post<{ event_id: string; type: string; note?: string }>("/api/events", event),
  triage: (text: string) => post<Triage>("/api/triage", { text }),
  acceptTriage: (events: Record<string, unknown>[]) =>
    post<{ recorded: number }>("/api/triage/accept", { events }),
  intake: (text: string) => post<Intake>("/api/intake", { text }),
  /** One box: the classifier picks the agent, `kind` overrides it. */
  note: (text: string, kind?: "booking" | "disruption") =>
    post<Note>(`/api/note${kind ? `?kind=${kind}` : ""}`, { text }),
  /** Accept a quoted slot. The draft goes back with it: nothing an agent produced is
   * stored until a person agrees to it, and a draft held server-side awaiting
   * confirmation is stored. */
  book: (
    draft: Draft,
    date: string,
    arrival: string,
    transcript = "",
    quotedTotal = 0,
    bookingNote = "",
    crewIds: string[] = [],
    allowDuplicate = false,
  ) =>
    post<{ job_id: string; customer: string; when: string; status: string }>("/api/book", {
      draft,
      date,
      arrival,
      transcript,
      quoted_total: quotedTotal,
      booking_note: bookingNote,
      crew_ids: crewIds,
      allow_duplicate: allowDuplicate,
    }),
  crewOptions: (jobId: string) =>
    request<{
      options: {
        worker_id: string;
        name: string;
        feasible: boolean;
        current?: boolean;
        cost_delta?: number;
        note?: string;
      }[];
    }>(`/api/jobs/${encodeURIComponent(jobId)}/crew-options`),
  setCrew: (jobId: string, workerId: string) =>
    post<{ job_id: string; crew: string; status: string }>(
      `/api/jobs/${encodeURIComponent(jobId)}/crew?worker_id=${encodeURIComponent(workerId)}`,
    ),
  session: () => request<{ owner_pin_set: boolean; owner: boolean }>("/api/session"),
  pricing: () =>
    request<{
      fields: { key: string; value: number; source: string; min: number; max: number }[];
    }>("/api/config/pricing"),
  setPricing: (entries: Record<string, number>) =>
    post<{ status: string }>("/api/config/pricing", entries),
  moveDepot: (address: string, confirm: string) =>
    post<{ address: string; moved_miles: string; status: string }>("/api/config/depot", {
      address,
      confirm,
    }),
  /** Take a booking back out of the diary. Same-call changes of mind are normal. */
  cancel: (jobId: string) =>
    post<{ job_id: string; customer: string; status: string }>(
      `/api/jobs/${encodeURIComponent(jobId)}/cancel`,
    ),
  comms: (strategy: string) =>
    post<Message[]>(`/api/comms?strategy=${encodeURIComponent(strategy)}`),
};

/** How often to re-check when streaming is off. Slow enough to be nearly free, fast
 * enough that a change made in the CLI shows up before you switch windows. */
const POLL_MS = 10_000;

function stream(onChange: () => void): () => void {
  const source = new EventSource("/api/stream");
  source.onmessage = (event) => {
    const payload = JSON.parse(event.data) as { type: string };
    if (payload.type === "plan" || payload.type === "world") onChange();
  };
  return () => source.close();
}

function poll(onChange: () => void): () => void {
  const timer = setInterval(onChange, POLL_MS);
  return () => clearInterval(timer);
}

/**
 * Live updates, so a change made from the CLI or an agent shows up here too.
 *
 * The server chooses the mechanism, because the client cannot: a stream works
 * perfectly well on Lambda, it is just billed for every second it stays open, and
 * there is no failure to detect and fall back from. Running locally it streams;
 * deployed it polls.
 */
export function subscribe(onChange: () => void): () => void {
  let stop = () => {};
  let cancelled = false;

  void (async () => {
    let mode = "sse";
    try {
      const health = await request<{ stream?: string }>("/api/health");
      mode = health.stream ?? "sse";
    } catch {
      // Unreachable server, or an older one that does not say. Streaming is the
      // right guess for both: it is what a local dev server does.
    }
    if (cancelled) return;
    stop = mode === "poll" ? poll(onChange) : stream(onChange);
  })();

  return () => {
    cancelled = true;
    stop();
  };
}
