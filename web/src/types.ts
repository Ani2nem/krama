// Mirrors krama.api.models. Hand-written rather than generated: the board
// consumes a deliberately small slice, and a generated client would pull in the whole
// surface plus its churn.

export interface Stop {
  job_id: string;
  customer_name: string;
  service_type: string;
  arrival: string;
  departure: string;
  start_minute: number;
  end_minute: number;
  travel_minutes: number;
  travel_miles: number;
  crew_size: number;
  commitment_state: string;
  gap_minutes: number;
  past_shift: boolean;
  from_label: string;
  next_label: string;
  run_note: string;
  lat: number;
  lon: number;
}

export interface Route {
  crew_id: string;
  date: string;
  worker_names: string[];
  van_id: string;
  stops: Stop[];
  travel_minutes: number;
  travel_miles: number;
  idle_minutes: number;
  utilization: number;
  overtime_minutes: number;
}

export interface Unserved {
  job_id: string;
  customer_name: string;
  reason: string;
  detail: string;
  is_failure: boolean;
}

export interface Cost {
  vehicle: number;
  overtime: number;
  lateness: number;
  unserved: number;
  total: number;
}

export interface Plan {
  redacted: boolean;
  plan_id: string;
  content_hash: string;
  horizon_start: string;
  horizon_end: string;
  routes: Route[];
  unserved: Unserved[];
  cost: Cost;
  feasible: boolean;
  violations: string[];
  depot: number[];
}

export interface WorkerDay {
  date: string;
  day: string;
  shift: string;
  reach: string;
  extended: string;
  available: boolean;
  actionable: boolean;
  out_note: string;
  out_spans: [string, string][];
}

export interface VanDay {
  date: string;
  day: string;
  available: boolean;
  actionable: boolean;
  out_note: string;
  out_spans: [string, string][];
}

export interface Worker {
  days: WorkerDay[];
  id: string;
  name: string;
  certifications: string[];
  shift: string;
  available: boolean;
  overtime_eligible: boolean;
  shift_start: string;
  shift_end: string;
  phone: string;
}

export interface Van {
  id: string;
  label: string;
  available: boolean;
  stock: Record<string, number>;
  days: VanDay[];
}

export interface Job {
  phone: string;
  address: string;
  quoted_total: number;
  booking_note: string;
  window_start: string;
  window_end: string;
  transcript: string;
  id: string;
  customer_name: string;
  service_type: string;
  duration_minutes: number;
  crew_size: number;
  certifications: string[];
  commitment_state: string;
  commitment_cost: number;
  window: string;
  lat: number;
  lon: number;
}

export interface CrewAskSuggestion {
  message: string;
  on_date: string;
  until: string;
  until_label: string;
  candidate_ids: string[];
  candidate_names: string[];
}

export interface CrewAsk {
  ask_id: string;
  customer: string;
  phone: string;
  day: string;
  until_label: string;
  detail: string;
  transcript: string;
  candidates: { id: string; name: string }[];
  extended: string[];
}

export interface OvertimeOffer {
  job_id: string;
  customer: string;
  day: string;
  arrival: string;
  overtime_minutes: number;
  status: "open" | "claimed" | "expired";
  offered_to: string[];
  offered_ids: string[];
  claimed_by: string;
  fallback: string;
  deadline: string;
}

export interface World {
  depot_address: string;
  as_of: string;
  workers: Worker[];
  vans: Van[];
  jobs: Job[];
  overtime_offers: OvertimeOffer[];
  crew_asks: CrewAsk[];
  committed_plan_id: string;
  calibration_warning: string;
}

export interface Change {
  job_id: string;
  customer_name: string;
  kind: string;
  description: string;
  needs_customer_call: boolean;
}

export interface Candidate {
  strategy: string;
  description: string;
  jobs_served: number;
  changes: number;
  customer_calls: number;
  blast_radius: string;
  autonomy: string;
  autonomy_reasons: string[];
  diff: Change[];
  recommended: boolean;
}


export interface Slot {
  crew_ids: string[];
  from_label: string;
  leg_minutes: number;
  leg_miles: number;
  needs_overtime: boolean;
  outside_preference: boolean;
  quote_total: number;
  quote_lines: string[];
  margin: number;
  margin_pct: number;
  day: string;
  arrival: string;
  crew_reason: string;
  date: string;
  window: string;
  marginal_cost: number;
  crew: string;
  reason: string;
}

export interface Draft {
  customer_name: string;
  phone: string;
  address: string;
  service_type: string;
  duration_minutes: number;
  duration_confidence: number;
  crew_size: number;
  certifications: string[];
  commitment_cost: number;
  commitment_quotes: string[];
  lead_time_days: number;
  site_notes: string;
  lat: number | null;
  lon: number | null;
}

export interface UnavailableDay {
  day: string;
  reason: string;
}

export interface Duplicate {
  job_id: string;
  customer: string;
  when: string;
  what: string;
}

export interface Intake {
  redacted: boolean;
  possible_duplicates: Duplicate[];
  draft: Draft;
  bookable: boolean;
  missing: string[];
  ask_next: string[];
  slots: Slot[];
  flexible_slots: Slot[];
  unavailable: UnavailableDay[];
  when_text: string;
  asked_for_speed: boolean;
  crew_ask: CrewAskSuggestion | null;
  not_offered: string;
  repairs: number;
  note: string;
}

export interface Triage {
  state: string;
  summary: string;
  events: Record<string, unknown>[];
  question: string;
  unknown_targets: string[];
  rejected: string[];
  repairs: number;
}

export interface Message {
  job_id: string;
  channel: string;
  body: string;
  grounded: boolean;
  issues: string[];
}

/** One note, read by whichever agent it turned out to be for. */
export interface Note {
  kind: "booking" | "disruption";
  why: string;
  booking: Intake | null;
  disruption: Triage | null;
}

/** The days the board draws, whether or not anything is planned yet. */
export interface Week {
  start: string;
  end: string;
}
