// gta6-twitch-ccu-forecast-01 — launch-day Twitch CCU forecaster (Node 22, stdlib only).
//
// Forecast for hours 2..24 of a game's release day (UTC), five quantiles each, from
// the case file alone: hour-1 print + the game's own pre-midnight history + 38-42
// comparable past launches (hourly SullyGnome counts, 14 days before to 3 days after).
//
// Method (fitted at runtime from the case's own comparables — nothing hard-coded
// from the practice answers):
//   1. For every comparable, build its release-day curve as log-ratios to its hour-1
//      print: r_h = ln(v_h / v_1). Hour 1 is a local trough (00:00-01:00 UTC, between
//      the NZ/Australia evening wave and the UK/US waves), so the ratio curve's shape
//      — dip, then rise to an evening peak — is what must be learned, not the level.
//   2. Keep only "fresh-category" launches: median hourly viewers in the 7 days
//      before release < 1000. Live-service / early-access games whose category already
//      had viewers follow a different shape and poison the fit. (If fewer than 8
//      fresh launches exist, fall back to all comparables.)
//   3. Four diverse predictors of the ratio curve, ensembled with equal weight in
//      log space (geometric mean of quantiles):
//      A. per-hour linear OLS of log-ratio on log(hour-1 viewers), normal residual
//         spread — big launches rise less steeply from hour 1 than small ones;
//      B. per-hour quadratic OLS on log(hour-1 viewers), normal residual spread —
//         the rise flattens at the top end;
//      C. 8 nearest neighbours by |log h1 - log p1|, empirical ratio quantiles;
//      D. 8 nearest neighbours by pre-midnight profile distance (RMS log-ratio
//         difference over hours -13..-1, plus 0.25 * (log h1 gap)^2), empirical
//         ratio quantiles — the NZ/Australia unlock wave observed before midnight
//         fingerprints the launch's shape.
//   4. Quantiles are the five scored levels, clamped to (0, 1e9], non-decreasing,
//      rounded to 2 decimals. Deterministic: no randomness, no clock, no network.
//      Runs in well under 1 s per case.
//
// Local backtest (scorer verified against the RULES worked example, loss 0.112545):
//   practice p00/p01/p02 mean pinball loss 0.193 (starter house baseline: 0.415).
//   Per case: p00 0.215, p01 0.194, p02 0.169. (p03 answers not published by the
//   source for its release month, so it is not scored locally.)

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Fresh-category cutoff: median hourly viewers in the 7 days before release. */
const FRESH_BASELINE_MAX = 1000;
/** Minimum fresh launches before falling back to the full pool. */
const MIN_FRESH_POOL = 8;
/** Neighbours used by the two kNN predictors. */
const KNN_K = 8;

interface HourlyPoint { from: string; viewers: number; channels?: number | null }
interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}
interface Case {
  id: string;
  day_start: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history_hourly: HourlyPoint[];
  comparables: Comparable[];
}

/** One comparable reduced to what the predictors need. */
interface LaunchRecord {
  logH1: number;
  h1: number;
  base: number | null;
  /** log(v_h / v_1) for release-day hours 2..24 (offset h-1 in the map). */
  ratios: Map<number, number>;
  /** log(v_k / v_1) for pre-midnight offsets k = -13..-1. */
  pre: Map<number, number>;
}

/** Whole-hour offset of a point from the release day's start (offset 0 = hour 1). */
const hourOffset = (from: string, t0: number): number =>
  Math.floor((Date.parse(from) - t0) / 3600000);

/** Build an offset -> viewers map from an hourly series (first point wins on dupes). */
function offsetMap(hourly: HourlyPoint[], t0: number): Map<number, number> {
  const m = new Map<number, number>();
  for (const p of hourly) {
    if (!(p.viewers > 0)) continue;
    const off = hourOffset(p.from, t0);
    if (!m.has(off)) m.set(off, p.viewers);
  }
  return m;
}

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

/** Median hourly viewers in the 7 days before release (null when too sparse). */
function baseline(m: Map<number, number>): number | null {
  const vals: number[] = [];
  for (const [off, v] of m) if (off >= -168 && off < 0) vals.push(v);
  return vals.length >= 10 ? median(vals) : null;
}

function recordOf(comp: Comparable): LaunchRecord | null {
  const t0 = Date.parse(comp.day_start);
  const m = offsetMap(comp.hourly, t0);
  const v1 = m.get(0);
  if (v1 === undefined) return null;
  const ratios = new Map<number, number>();
  for (let h = 2; h <= 24; h++) {
    const v = m.get(h - 1);
    if (v !== undefined) ratios.set(h, Math.log(v / v1));
  }
  const pre = new Map<number, number>();
  for (let k = -13; k <= -1; k++) {
    const v = m.get(k);
    if (v !== undefined) pre.set(k, Math.log(v / v1));
  }
  return { logH1: Math.log(v1), h1: v1, base: baseline(m), ratios, pre };
}

/** Type-7 empirical quantile of a sorted array. */
function quantileSorted(sorted: number[], tau: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function meanSd(xs: number[]): [number, number] {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const v = xs.reduce((a, b) => a + (b - m) * (b - m), 0) / Math.max(1, xs.length - 1);
  return [m, Math.sqrt(v)];
}

/** Solve a small dense linear system by Gaussian elimination (null if singular). */
function solveLinear(A: number[][], Bv: number[]): number[] | null {
  const d = Bv.length;
  const M = A.map((row, i) => [...row, Bv[i]]);
  for (let c = 0; c < d; c++) {
    let piv = c;
    for (let r = c + 1; r < d; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    const tmp = M[c]; M[c] = M[piv]; M[piv] = tmp;
    if (Math.abs(M[c][c]) < 1e-12) return null;
    for (let r = c + 1; r < d; r++) {
      const k = M[r][c] / M[c][c];
      for (let cc = c; cc < d + 1; cc++) M[r][cc] -= k * M[c][cc];
    }
  }
  const beta = new Array<number>(d).fill(0);
  for (let r = d - 1; r >= 0; r--) {
    let s = M[r][d];
    for (let cc = r + 1; cc < d; cc++) s -= M[r][cc] * beta[cc];
    beta[r] = s / M[r][r];
  }
  return beta;
}

/** Polynomial OLS (degree 1 or 2) of y on x, x centred; returns predictor + residuals. */
function polyFit(pts: Array<[number, number]>, deg: number): { pred: (x: number) => number; resid: number[] } | null {
  const xs = pts.map((p) => p[0]);
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
  const d = deg + 1;
  const A = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const Bv = new Array<number>(d).fill(0);
  for (const [x, y] of pts) {
    const X = Array.from({ length: d }, (_, i) => (x - mx) ** i);
    for (let r = 0; r < d; r++) {
      Bv[r] += X[r] * y;
      for (let cc = 0; cc < d; cc++) A[r][cc] += X[r] * X[cc];
    }
  }
  const beta = solveLinear(A, Bv);
  if (beta === null) return null;
  return {
    pred: (x: number): number => {
      const xc = x - mx;
      let s = 0;
      for (let i = 0; i < d; i++) s += beta[i] * xc ** i;
      return s;
    },
    resid: pts.map(([x, y]) => y - beta.reduce((s, b, i) => s + b * (x - mx) ** i, 0)),
  };
}

/** Clamp to (0, 1e9], enforce non-decreasing, round to 2 decimals. */
function finishQuantiles(qs: number[]): number[] {
  const out = qs.map((q) => Math.min(1e9, Math.max(0.01, Math.round(q * 100) / 100)));
  for (let j = 1; j < out.length; j++) out[j] = Math.max(out[j], out[j - 1]);
  return out;
}

/** Polynomial-regression predictor (normal residual spread) for one hour. */
function polyPredictor(pool: LaunchRecord[], p1: number, deg: number): Map<number, number[]> {
  const out = new Map<number, number[]>();
  const x0raw = Math.log(p1);
  const xs = pool.map((r) => r.logH1);
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  // Winsorise: never extrapolate the curve beyond the observed hour-1 range.
  const x0 = Math.min(hi, Math.max(lo, x0raw));
  for (let h = 2; h <= 24; h++) {
    const pts: Array<[number, number]> = [];
    for (const r of pool) {
      const y = r.ratios.get(h);
      if (y !== undefined) pts.push([r.logH1, y]);
    }
    if (pts.length < 4) continue;
    const fit = polyFit(pts, deg);
    if (fit === null) continue;
    const mu = fit.pred(x0);
    const [, sd] = meanSd(fit.resid);
    const s = Math.max(sd, 0.05);
    out.set(h, Z.map((z) => p1 * Math.exp(mu + z * s)));
  }
  return out;
}

/** kNN predictor (empirical quantiles) for every hour. */
function knnPredictor(pool: LaunchRecord[], p1: number, targetPre: Map<number, number>, useProfile: boolean): Map<number, number[]> {
  const out = new Map<number, number[]>();
  const x0 = Math.log(p1);
  const scored: Array<[number, LaunchRecord]> = [];
  for (const r of pool) {
    let d: number;
    if (!useProfile) {
      d = Math.abs(r.logH1 - x0);
    } else {
      let s = 0;
      let n = 0;
      for (const [k, v] of targetPre) {
        const rv = r.pre.get(k);
        if (rv !== undefined) { s += (v - rv) * (v - rv); n++; }
      }
      if (n < 4) continue;
      d = Math.sqrt(s / n + 0.25 * (r.logH1 - x0) * (r.logH1 - x0));
    }
    scored.push([d, r]);
  }
  scored.sort((a, b) => a[0] - b[0]);
  const near = scored.slice(0, KNN_K).map(([, r]) => r);
  if (near.length < 3) return out;
  for (let h = 2; h <= 24; h++) {
    const vals: number[] = [];
    for (const r of near) {
      const y = r.ratios.get(h);
      if (y !== undefined) vals.push(y);
    }
    if (vals.length < 3) continue;
    vals.sort((a, b) => a - b);
    out.set(h, LEVELS.map((t) => p1 * Math.exp(quantileSorted(vals, t))));
  }
  return out;
}

/** Forecast one case: ensemble of the four predictors, equal weights in log space. */
function forecastCase(c: Case): Array<{ hour: number; q: number[] }> {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${c.id} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);

  const records: LaunchRecord[] = [];
  for (const comp of c.comparables) {
    const rec = recordOf(comp);
    if (rec !== null) records.push(rec);
  }
  let pool = records.filter((r) => r.base !== null && r.base < FRESH_BASELINE_MAX);
  if (pool.length < MIN_FRESH_POOL) pool = records;
  if (pool.length === 0) {
    // No usable comparable at all: fall back to the persistence shape
    // (house-baseline spread) so the case still scores instead of crashing.
    return c.forecast.map(({ hour }) => ({
      hour,
      q: finishQuantiles(Z.map((z) => p1 * Math.exp(z * 0.15 * Math.sqrt(hour - 1)))),
    }));
  }
  // Deterministic order regardless of input order.
  pool.sort((a, b) => a.logH1 - b.logH1 || a.h1 - b.h1);

  // The target's own pre-midnight profile (NZ/Australia unlock wave).
  const t0 = Date.parse(c.day_start);
  const own = offsetMap(c.history_hourly, t0);
  const pre = new Map<number, number>();
  for (let k = -13; k <= -1; k++) {
    const v = own.get(k);
    if (v !== undefined) pre.set(k, Math.log(v / p1));
  }

  const parts = [
    polyPredictor(pool, p1, 1),
    polyPredictor(pool, p1, 2),
    knnPredictor(pool, p1, pre, false),
    knnPredictor(pool, p1, pre, true),
  ];

  return c.forecast.map(({ hour }) => {
    const qs: number[] = [];
    for (let j = 0; j < LEVELS.length; j++) {
      let sum = 0;
      let n = 0;
      for (const part of parts) {
        const row = part.get(hour);
        if (row !== undefined && isFinite(row[j]) && row[j] > 0) { sum += Math.log(row[j]); n++; }
      }
      // Every predictor covers every hour in practice; flat p1 is the last resort.
      qs.push(n > 0 ? Math.exp(sum / n) : p1);
    }
    return { hour, q: finishQuantiles(qs) };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
