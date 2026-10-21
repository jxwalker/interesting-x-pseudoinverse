import "./styles.css";
import {
  basisRow,
  designMatrix,
  nullBasis,
  selfTest,
  solve,
  svd,
  type Basis,
  type SelfTestResult,
  type Solution,
  type SVD,
} from "./linalg";
import {
  CURVE_N,
  PULSE_LIFE,
  Y_MAX,
  Y_MIN,
  curveXs,
  drawSpectrum,
  drawStage,
  fitCanvas,
  fromPxX,
  fromPxY,
  plotRect,
  toPxX,
  toPxY,
  type BarKind,
  type Badge,
  type Pulse,
  type Stack,
  type StagePoint,
} from "./viz";

declare global {
  interface Window {
    __pinvSelfTest?: () => SelfTestResult;
  }
}

const CREDIT_URL = "https://x.com/scievision369/status/2106644532825858439";
const MAX_POINTS = 60;
const DEG_MIN = 1;
const DEG_MAX = 6;
const GHOSTS = 4;
const GHOST_AMP = 0.85;
/** Curve morph rates (1/s): calm for control changes, snappy while dragging. */
const MORPH_RATE = 5.5;
const DRAG_RATE = 22;
const SPRING_K = 95;
const SPRING_C = 11;
const INITIAL_SEED = 20261015;
const SUB = "₀₁₂₃₄₅₆₇₈₉";

type Preset = "overdetermined" | "underdetermined" | "rank-deficient";

interface Pt {
  x: number;
  y: number;
  /** Burst offset (data units) and its velocity — decays on a spring. */
  ox: number;
  oy: number;
  vx: number;
  vy: number;
}

const HINTS: Record<Preset, string> = {
  overdetermined:
    "Drag the dots — the glowing curve is the best fit A⁺ can find. Scrub “keep” to throw away singular values and watch the fit collapse.",
  underdetermined:
    "Fewer points than coefficients: every ghost curve fits exactly. A⁺ picks the one with the smallest ‖x‖ — switch basis and that “smallest” changes.",
  "rank-deficient":
    "Stacked points share an x, so A loses rank: no curve can hit them all — A⁺ threads each stack's mean, and the null space adds free ghosts.",
};

function mustEl<T extends HTMLElement>(root: ParentNode, selector: string, ctor: new () => T): T {
  const el = root.querySelector(selector);
  if (!(el instanceof ctor)) throw new Error(`Missing or wrong element: ${selector}`);
  return el;
}

function setText(el: HTMLElement, next: string): void {
  if (el.textContent !== next) el.textContent = next;
}

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rnd: () => number): number {
  const u = Math.max(1e-12, rnd());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
}

/** Compact number format: 3.2e4 style outside [0.01, 1e4). */
function fmt(v: number): string {
  if (!Number.isFinite(v)) return "∞";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-2) return v.toExponential(1).replace("e+", "e");
  return v.toPrecision(3);
}

/** A smooth hidden target: two sines (one damped) plus a tilt, randomised per seed. */
function makeTarget(rnd: () => number): (x: number) => number {
  const a1 = 0.75 + 0.35 * rnd();
  const w1 = 2.3 + 1.5 * rnd();
  const p1 = 2 * Math.PI * rnd();
  const a2 = 0.35 + 0.3 * rnd();
  const w2 = 4.6 + 1.8 * rnd();
  const p2 = 2 * Math.PI * rnd();
  const tilt = (rnd() - 0.5) * 0.7;
  const raw = (x: number): number =>
    a1 * Math.sin(w1 * x + p1) + a2 * Math.sin(w2 * x + p2) * Math.exp(-0.9 * x * x) + tilt * x;
  let peak = 0;
  for (let i = 0; i <= 80; i++) peak = Math.max(peak, Math.abs(raw(-1 + i / 40)));
  const k = peak > 1.3 ? 1.3 / peak : 1;
  return (x) => raw(x) * k;
}

function makeData(preset: Preset, seed: number): { xs: number[]; ys: number[]; degree: number } {
  const rnd = mulberry(seed);
  const f = makeTarget(rnd);
  const xs: number[] = [];
  const ys: number[] = [];
  if (preset === "overdetermined") {
    const m = 14;
    for (let i = 0; i < m; i++) {
      const x = -0.94 + (1.88 * i) / (m - 1) + (rnd() - 0.5) * 0.07;
      xs.push(x);
      ys.push(f(x) + 0.14 * gauss(rnd));
    }
    return { xs, ys, degree: 5 };
  }
  if (preset === "underdetermined") {
    for (const x0 of [-0.74, -0.24, 0.27, 0.76]) {
      const x = x0 + (rnd() - 0.5) * 0.08;
      xs.push(x);
      ys.push(f(x) + 0.05 * gauss(rnd));
    }
    return { xs, ys, degree: 6 };
  }
  for (const x of [-0.62, 0.04, 0.66]) {
    for (let j = 0; j < 4; j++) {
      xs.push(x);
      ys.push(f(x) + 0.32 * gauss(rnd));
    }
  }
  return { xs, ys, degree: 4 };
}

interface Fit {
  m: number;
  n: number;
  svd: SVD;
  sol: Solution;
  full: Solution;
  curve: Float64Array;
  reference: Float64Array | null;
  ghostDirs: Float64Array[];
  basisCurves: Float64Array[];
  stacks: Stack[];
  lambda: number;
}

function mount(): void {
  const root = document.getElementById("app");
  if (!(root instanceof HTMLElement)) throw new Error("Missing #app");
  const recordMode = new URLSearchParams(window.location.search).has("record");

  root.innerHTML = `
    <div class="grain" aria-hidden="true"></div>
    <div class="wash" aria-hidden="true"></div>

    <header class="hero">
      <div class="hero-row">
        <p class="brand">Interesting X</p>
        <h1 class="title">Pseudoinverse playground</h1>
      </div>
      <p class="hint">
        <span class="hint-dot" aria-hidden="true"></span>
        <span id="hint">${HINTS.overdetermined}</span>
      </p>
    </header>

    <main class="layout">
      <section class="stage" id="stage" aria-label="Least-squares fit through data points">
        <canvas id="canvas" role="img" aria-label="Data points with the pseudoinverse least-squares curve and residual stalks. Click to add a point, drag to move, double-click or right-click to remove."></canvas>
        <div class="stage-help" aria-hidden="true">click add · drag move · dbl-click remove</div>
      </section>

      <aside class="panel" aria-label="Controls and measurements">
        <div class="controls">
          <div class="field">
            <label for="degree">Polynomial degree <span class="unit">→ columns of A</span></label>
            <input id="degree" type="range" min="${DEG_MIN}" max="${DEG_MAX}" step="1" value="5" />
            <div class="range-meta"><span>line</span><span id="degree-val" class="val">degree 5 · n = 6</span><span>wiggly</span></div>
          </div>
          <div class="field">
            <label for="rank-keep">Keep singular values <span class="unit">TSVD</span></label>
            <input id="rank-keep" type="range" min="1" max="6" step="1" value="6" />
            <div class="range-meta"><span>1</span><span id="rank-val" class="val val-amber">keep 6 / 6</span><span>all</span></div>
          </div>
          <div class="btn-row">
            <button type="button" class="btn btn-primary" id="btn-reseed">Reseed noise</button>
            <select id="preset" aria-label="Problem shape">
              <option value="overdetermined">Overdetermined</option>
              <option value="underdetermined">Underdetermined</option>
              <option value="rank-deficient">Rank-deficient</option>
            </select>
          </div>
          <details class="adv" id="advanced">
            <summary>Advanced</summary>
            <div class="adv-body">
              <div class="field field-inline">
                <label for="basis">Basis</label>
                <select id="basis">
                  <option value="monomial">Monomial xᵏ</option>
                  <option value="chebyshev">Chebyshev Tₖ(x)</option>
                </select>
              </div>
              <div class="field">
                <label for="ridge">Ridge λ <span class="unit">Tikhonov</span></label>
                <input id="ridge" type="range" min="0" max="100" step="1" value="0" />
                <div class="range-meta"><span>0</span><span id="ridge-val" class="val val-mag">λ off</span><span>strong</span></div>
              </div>
            </div>
          </details>
        </div>

        <div class="stats" aria-live="off">
          <div class="stat"><span class="stat-k">Shape</span><span class="stat-v" id="stat-shape">A: 14 × 6</span></div>
          <div class="stat"><span class="stat-k">Rank used</span><span class="stat-v v-amber" id="stat-rank">6 / 6</span></div>
          <div class="stat"><span class="stat-k">κ kept</span><span class="stat-v v-violet" id="stat-cond">—</span></div>
          <div class="stat"><span class="stat-k">‖Ax̂ − b‖₂</span><span class="stat-v v-hot" id="stat-resid">—</span></div>
          <div class="stat"><span class="stat-k">‖x̂‖₂</span><span class="stat-v v-cyan" id="stat-xnorm">—</span></div>
          <div class="stat"><span class="stat-k">κ full σ₁/σₘᵢₙ</span><span class="stat-v" id="stat-cond-full">—</span></div>
        </div>

        <div class="card card-spec">
          <div class="card-head">
            <span class="card-title">Singular values σᵢ <span class="unit">log</span></span>
            <span class="legend"><i class="lg-bar"></i>kept<i class="lg-hollow"></i>dropped<i class="lg-dia"></i>|uᵢᵀb|</span>
          </div>
          <canvas id="spectrum" aria-label="Singular value spectrum on a log scale"></canvas>
        </div>

        <div class="card card-energy">
          <div class="card-head">
            <span class="card-title">Energy split <span class="unit">‖b‖²</span></span>
            <span class="energy-note" id="energy-note">orthogonal ✓</span>
          </div>
          <div class="energy" aria-hidden="true">
            <i class="en-fit" id="energy-fit"></i><i class="en-res" id="energy-res"></i>
          </div>
          <div class="energy-meta"><span id="energy-fit-val">‖Ax̂‖² —</span><span id="energy-res-val">‖r‖² —</span></div>
          <p class="recipe">
            <span class="eq">x̂ = A<sup>+</sup>b = V Σ<sup>+</sup> U<sup>T</sup> b</span>
            <span class="recipe-note">Σ⁺ inverts the nonzero singular values — keep fewer and the fit gets smoother but worse.</span>
          </p>
        </div>

        <p class="credit${recordMode ? " record-hide" : ""}">
          Moore–Penrose pseudoinverse via one-sided Jacobi SVD, in your browser. Inspired by
          <a href="${CREDIT_URL}" target="_blank" rel="noreferrer">@scievision369</a>.
        </p>
      </aside>
    </main>
  `;

  const stage = mustEl(root, "#stage", HTMLElement);
  const canvas = mustEl(root, "#canvas", HTMLCanvasElement);
  const specCanvas = mustEl(root, "#spectrum", HTMLCanvasElement);
  const degreeIn = mustEl(root, "#degree", HTMLInputElement);
  const degreeVal = mustEl(root, "#degree-val", HTMLElement);
  const keepIn = mustEl(root, "#rank-keep", HTMLInputElement);
  const keepVal = mustEl(root, "#rank-val", HTMLElement);
  const btnReseed = mustEl(root, "#btn-reseed", HTMLButtonElement);
  const presetSel = mustEl(root, "#preset", HTMLSelectElement);
  const basisSel = mustEl(root, "#basis", HTMLSelectElement);
  const ridgeIn = mustEl(root, "#ridge", HTMLInputElement);
  const ridgeVal = mustEl(root, "#ridge-val", HTMLElement);
  const advanced = mustEl(root, "#advanced", HTMLDetailsElement);
  const hintEl = mustEl(root, "#hint", HTMLElement);
  const statShape = mustEl(root, "#stat-shape", HTMLElement);
  const statRank = mustEl(root, "#stat-rank", HTMLElement);
  const statCond = mustEl(root, "#stat-cond", HTMLElement);
  const statResid = mustEl(root, "#stat-resid", HTMLElement);
  const statXnorm = mustEl(root, "#stat-xnorm", HTMLElement);
  const statCondFull = mustEl(root, "#stat-cond-full", HTMLElement);
  const energyFit = mustEl(root, "#energy-fit", HTMLElement);
  const energyRes = mustEl(root, "#energy-res", HTMLElement);
  const energyFitVal = mustEl(root, "#energy-fit-val", HTMLElement);
  const energyResVal = mustEl(root, "#energy-res-val", HTMLElement);
  const energyNote = mustEl(root, "#energy-note", HTMLElement);

  if (window.matchMedia("(min-width: 960px)").matches) advanced.open = true;

  // ------------------------------------------------------------------ state
  const xs = curveXs();
  let preset: Preset = "overdetermined";
  let seed = INITIAL_SEED;
  let degree = 5;
  let basis: Basis = "monomial";
  let keep = 6;
  let keepFull = true;
  let ridgeU = 0;
  let pts: Pt[] = [];
  let fit: Fit | null = null;
  let dirty = true;

  const shown = new Float64Array(CURVE_N);
  let refAlpha = 0;
  let ghostAlpha = 0;
  let flash = 0;
  const pulses: Pulse[] = [];
  const ghostPhase = Array.from({ length: GHOSTS }, (_, g) => ({
    w: 0.9 + 0.37 * g,
    p: 1.7 * g,
  }));

  let specLog = new Float64Array(0);
  let specUb = new Float64Array(0);
  let specFilter = new Float64Array(0);
  let specLo = -2;
  let specHi = 1;
  let filterAlpha = 0;

  let dragIdx = -1;
  let hoverIdx = -1;

  // Opening shot: start flat-ish with a little jitter so the fit visibly settles.
  {
    const rnd = mulberry(7);
    for (let i = 0; i < CURVE_N; i++) shown[i] = 0.12 * Math.sin(xs[i]! * 9 + rnd() * 0.4);
  }

  function burst(prev: Pt[], strength: number): void {
    const rnd = mulberry((seed ^ 0x9e3779b9) >>> 0);
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i]!;
      const o = prev[i];
      const ang = rnd() * Math.PI * 2;
      const kick = strength * (0.25 + 0.35 * rnd());
      p.ox = (o ? o.x + o.ox - p.x : 0) + Math.cos(ang) * kick;
      p.oy = (o ? o.y + o.oy - p.y : 0) + Math.sin(ang) * kick * 1.4;
      p.vx = 0;
      p.vy = 0;
    }
    pulses.push({ x: 0, y: 0, age: 0 });
  }

  function loadPreset(next: Preset, newSeed: number, withDegree: boolean): void {
    const prev = pts;
    preset = next;
    seed = newSeed;
    const d = makeData(next, newSeed);
    pts = d.xs.map((x, i) => ({ x, y: d.ys[i]!, ox: 0, oy: 0, vx: 0, vy: 0 }));
    if (withDegree) {
      degree = d.degree;
      degreeIn.value = String(degree);
      keepFull = true;
    }
    burst(prev.length === pts.length ? prev : [], prev.length === pts.length ? 0.35 : 0.7);
    setText(hintEl, HINTS[next]);
    dirty = true;
  }

  function lambdaFor(s: SVD): number {
    if (ridgeU <= 0 || s.r === 0) return 0;
    return s.S[0]! * Math.pow(10, -4 + (4 * ridgeU) / 100);
  }

  function evalCurve(coef: Float64Array, out: Float64Array): void {
    const row = new Float64Array(degree + 1);
    for (let i = 0; i < CURVE_N; i++) {
      basisRow(xs[i]!, degree, basis, row);
      let v = 0;
      for (let k = 0; k <= degree; k++) v += row[k]! * coef[k]!;
      out[i] = v;
    }
  }

  function refit(): void {
    const m = pts.length;
    const n = degree + 1;
    const a = designMatrix(
      pts.map((p) => p.x),
      degree,
      basis,
    );
    const b = Float64Array.from(pts, (p) => p.y);
    const s = svd(a);
    const r = s.r;

    if (keepFull) keep = r;
    keep = Math.max(1, Math.min(keep, r));
    keepIn.max = String(r);
    keepIn.value = String(keep);

    const lambda = lambdaFor(s);
    const sol = solve(a, s, b, { keep, lambda });
    const full = solve(a, s, b);

    const curve = new Float64Array(CURVE_N);
    evalCurve(sol.x, curve);
    let reference: Float64Array | null = null;
    if (sol.used < full.rank || lambda > 0) {
      reference = new Float64Array(CURVE_N);
      evalCurve(full.x, reference);
    }

    // Ghosts: random mixes of null-space directions, scaled to a fixed height on [−1, 1].
    const nb = nullBasis(s);
    const ghostDirs: Float64Array[] = [];
    if (nb.length) {
      const rnd = mulberry(4242 + nb.length);
      for (let g = 0; g < GHOSTS; g++) {
        const z = new Float64Array(n);
        for (const v of nb) {
          const w = gauss(rnd);
          for (let j = 0; j < n; j++) z[j] += w * v[j]!;
        }
        const c = new Float64Array(CURVE_N);
        evalCurve(z, c);
        let peak = 0;
        for (let i = 0; i < CURVE_N; i++) if (Math.abs(xs[i]!) <= 1) peak = Math.max(peak, Math.abs(c[i]!));
        if (peak < 1e-12) continue;
        for (let i = 0; i < CURVE_N; i++) c[i] = c[i]! / peak;
        ghostDirs.push(c);
      }
    }

    const basisCurves: Float64Array[] = [];
    const row = new Float64Array(n);
    for (let k = 0; k < n; k++) basisCurves.push(new Float64Array(CURVE_N));
    for (let i = 0; i < CURVE_N; i++) {
      basisRow(xs[i]!, degree, basis, row);
      for (let k = 0; k < n; k++) basisCurves[k]![i] = row[k]!;
    }

    const groups = new Map<string, number[]>();
    for (const p of pts) {
      const key = p.x.toFixed(3);
      const g = groups.get(key);
      if (g) g.push(p.y);
      else groups.set(key, [p.y]);
    }
    const stacks: Stack[] = [];
    for (const [key, ys] of groups) {
      if (ys.length < 2) continue;
      stacks.push({ x: Number(key), mean: ys.reduce((u, v) => u + v, 0) / ys.length, count: ys.length });
    }

    fit = { m, n, svd: s, sol, full, curve, reference, ghostDirs, basisCurves, stacks, lambda };
    updateReadouts();
    dirty = false;
  }

  function updateReadouts(): void {
    if (!fit) return;
    const { m, n, sol, full, svd: s, lambda } = fit;
    setText(degreeVal, `degree ${degree} · n = ${n}`);
    setText(keepVal, `keep ${keep} / ${s.r}`);
    setText(ridgeVal, lambda > 0 ? `λ = ${fmt(lambda / s.S[0]!)}·σ₁` : "λ off");
    setText(statShape, `A: ${m} × ${n}`);
    setText(statRank, `${sol.used} / ${full.rank}`);
    setText(statCond, fmt(sol.condKept));
    setText(statCondFull, fmt(full.condFull));
    setText(statResid, fmt(sol.residNorm));
    setText(statXnorm, fmt(sol.xNorm));

    const fitSq = sol.fitNorm * sol.fitNorm;
    const resSq = sol.residNorm * sol.residNorm;
    const bSq = sol.bNorm * sol.bNorm;
    const tot = Math.max(1e-300, fitSq + resSq);
    energyFit.style.width = `${(100 * fitSq) / tot}%`;
    energyRes.style.width = `${(100 * resSq) / tot}%`;
    const pct = (v: number): string => `${(100 * v / Math.max(1e-300, bSq)).toFixed(v / bSq < 0.001 && v > 0 ? 2 : 1)}%`;
    const label = sol.used < full.rank ? "‖Axₖ‖²" : "‖Ax̂‖²";
    setText(energyFitVal, `${label} ${pct(fitSq)}`);
    setText(energyResVal, `‖r‖² ${pct(resSq)}`);
    const cross = bSq - fitSq - resSq;
    const orth = Math.abs(cross) <= 1e-9 * Math.max(1, bSq);
    setText(energyNote, orth ? (sol.used < full.rank ? "projection onto u₁…uₖ ✓" : "orthogonal ✓") : `ridge: 2⟨Ax,r⟩ = ${fmt(cross)}`);
    energyNote.classList.toggle("warn", !orth);
  }

  function badges(): Badge[] {
    if (!fit) return [];
    const { m, n, sol, full, svd: s, lambda } = fit;
    const out: Badge[] = [];
    if (m > n) out.push({ text: `OVERDETERMINED ${m}>${n}`, tone: "over" });
    else if (m < n) out.push({ text: `UNDERDETERMINED ${m}<${n}`, tone: "under" });
    else out.push({ text: `SQUARE ${m}=${n}`, tone: "over" });
    if (full.rank < s.r) out.push({ text: `RANK-DEFICIENT r=${full.rank}`, tone: "rankdef" });
    if (sol.used < full.rank) out.push({ text: `TRUNCATED ${sol.used} of ${full.rank}`, tone: "trunc" });
    if (lambda > 0) out.push({ text: `RIDGE λ=${fmt(lambda)}`, tone: "ridge" });
    return out;
  }

  // ------------------------------------------------------------------ controls
  degreeIn.addEventListener("input", () => {
    degree = Math.max(DEG_MIN, Math.min(DEG_MAX, Number(degreeIn.value)));
    flash = Math.max(flash, 0.5);
    dirty = true;
  });
  keepIn.addEventListener("input", () => {
    keep = Number(keepIn.value);
    keepFull = keep >= Number(keepIn.max);
    dirty = true;
  });
  btnReseed.addEventListener("click", () => {
    loadPreset(preset, (Math.random() * 2 ** 31) >>> 0, false);
    flash = 1;
  });
  presetSel.addEventListener("change", () => {
    loadPreset(presetSel.value as Preset, seed + 1, true);
    flash = 1;
  });
  basisSel.addEventListener("change", () => {
    basis = basisSel.value as Basis;
    flash = 1;
    dirty = true;
  });
  ridgeIn.addEventListener("input", () => {
    ridgeU = Number(ridgeIn.value);
    dirty = true;
  });

  // ------------------------------------------------------------------ pointer
  function local(ev: { clientX: number; clientY: number }): { px: number; py: number } {
    const rect = canvas.getBoundingClientRect();
    return { px: ev.clientX - rect.left, py: ev.clientY - rect.top };
  }

  function nearest(px: number, py: number, radius: number): number {
    const p = plotRect(canvas.clientWidth, canvas.clientHeight);
    let best = -1;
    let bestD = radius * radius;
    for (let i = 0; i < pts.length; i++) {
      const q = pts[i]!;
      const dx = toPxX(p, q.x + q.ox) - px;
      const dy = toPxY(p, q.y + q.oy) - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  function moveTo(i: number, px: number, py: number): void {
    const p = plotRect(canvas.clientWidth, canvas.clientHeight);
    const q = pts[i]!;
    q.x = Math.max(-1, Math.min(1, fromPxX(p, px)));
    q.y = Math.max(Y_MIN + 0.05, Math.min(Y_MAX - 0.05, fromPxY(p, py)));
    q.ox = q.oy = q.vx = q.vy = 0;
    refit();
  }

  function removeAt(px: number, py: number, radius: number): void {
    const i = nearest(px, py, radius);
    if (i < 0 || pts.length <= 1) return;
    pts.splice(i, 1);
    if (dragIdx === i) dragIdx = -1;
    hoverIdx = -1;
    pulses.push({ x: fromPxX(plotRect(canvas.clientWidth, canvas.clientHeight), px), y: fromPxY(plotRect(canvas.clientWidth, canvas.clientHeight), py), age: PULSE_LIFE * 0.55 });
    refit();
  }

  const radiusFor = (ev: PointerEvent | MouseEvent): number =>
    "pointerType" in ev && ev.pointerType === "touch" ? 26 : 16;

  canvas.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const { px, py } = local(ev);
    let i = nearest(px, py, radiusFor(ev));
    if (i < 0) {
      const p = plotRect(canvas.clientWidth, canvas.clientHeight);
      const inside = px >= p.left && px <= p.left + p.w && py >= p.top && py <= p.top + p.h;
      if (!inside || pts.length >= MAX_POINTS) return;
      pts.push({ x: 0, y: 0, ox: 0, oy: 0, vx: 0, vy: 0 });
      i = pts.length - 1;
      pulses.push({ x: fromPxX(p, px), y: fromPxY(p, py), age: PULSE_LIFE * 0.55 });
    }
    dragIdx = i;
    try {
      canvas.setPointerCapture(ev.pointerId);
    } catch {
      // synthetic events may not be capturable
    }
    moveTo(i, px, py);
    ev.preventDefault();
  });
  canvas.addEventListener("pointermove", (ev) => {
    const { px, py } = local(ev);
    if (dragIdx >= 0) {
      moveTo(dragIdx, px, py);
      return;
    }
    hoverIdx = nearest(px, py, radiusFor(ev));
    canvas.style.cursor = hoverIdx >= 0 ? "grab" : "crosshair";
  });
  const endDrag = (): void => {
    dragIdx = -1;
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => {
    if (dragIdx < 0) hoverIdx = -1;
  });
  canvas.addEventListener("dblclick", (ev) => {
    const { px, py } = local(ev);
    removeAt(px, py, 18);
  });
  canvas.addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    const { px, py } = local(ev);
    removeAt(px, py, 22);
  });

  // ------------------------------------------------------------------ sizing
  const sizes = { cw: 0, ch: 0, sw: 0, sh: 0 };
  const measure = (): void => {
    sizes.cw = canvas.clientWidth;
    sizes.ch = canvas.clientHeight;
    sizes.sw = specCanvas.clientWidth;
    sizes.sh = specCanvas.clientHeight;
  };
  const ro = new ResizeObserver(measure);
  ro.observe(stage);
  ro.observe(specCanvas);
  measure();

  // ------------------------------------------------------------------ frame
  const ghostBuf = Array.from({ length: GHOSTS }, () => new Float64Array(CURVE_N));
  const stagePts: StagePoint[] = [];

  function stepSpectrum(dt: number): void {
    if (!fit) return;
    const { n, svd: s, sol, lambda } = fit;
    if (specLog.length !== n) {
      const prev = specLog;
      specLog = new Float64Array(n).fill(specLo);
      for (let i = 0; i < Math.min(prev.length, n); i++) specLog[i] = prev[i]!;
      const pu = specUb;
      specUb = new Float64Array(n).fill(specLo);
      for (let i = 0; i < Math.min(pu.length, n); i++) specUb[i] = pu[i]!;
      const pf = specFilter;
      specFilter = new Float64Array(n);
      for (let i = 0; i < Math.min(pf.length, n); i++) specFilter[i] = pf[i]!;
    }
    const a = Math.min(1, dt * 9);
    let lo = Infinity;
    let hi = -Infinity;
    const floor = Math.log10(Math.max(1e-300, (s.S[0] ?? 1) * 1e-16));
    for (let i = 0; i < s.r; i++) {
      const sg = s.S[i]!;
      if (sg > sol.tol) {
        const l = Math.log10(sg);
        lo = Math.min(lo, l);
        hi = Math.max(hi, l);
      }
      const u = Math.abs(sol.ub[i]!);
      if (u > 0) {
        const l = Math.log10(u);
        if (l > floor) {
          lo = Math.min(lo, l);
          hi = Math.max(hi, l);
        }
      }
    }
    if (!Number.isFinite(lo)) {
      lo = -1;
      hi = 1;
    }
    let tLo = lo - 0.45;
    const tHi = hi + 0.3;
    if (tHi - tLo < 1.6) tLo = tHi - 1.6;
    specLo += (tLo - specLo) * a;
    specHi += (tHi - specHi) * a;
    for (let i = 0; i < n; i++) {
      const sg = i < s.r ? s.S[i]! : 0;
      const t = sg > sol.tol ? Math.log10(sg) : specLo;
      specLog[i] = specLog[i]! + (t - specLog[i]!) * a;
      const u = i < s.r ? Math.abs(sol.ub[i]!) : 0;
      const tu = u > 0 ? Math.max(specLo - 1, Math.log10(u)) : specLo - 1;
      specUb[i] = specUb[i]! + (tu - specUb[i]!) * a;
      const tf = i < s.r ? sol.filter[i]! : 0;
      specFilter[i] = specFilter[i]! + (tf - specFilter[i]!) * a;
    }
    const wantFilter = lambda > 0 || sol.used < sol.rank ? 1 : 0;
    filterAlpha += (wantFilter - filterAlpha) * Math.min(1, dt * 8);
    if (Math.abs(filterAlpha - wantFilter) < 0.01) filterAlpha = wantFilter;
  }

  let last = performance.now();
  let clock = 0;
  function frame(now: number): void {
    const dt = Math.min(1 / 20, Math.max(0, (now - last) / 1000));
    last = now;
    clock += dt;
    if (dirty) refit();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (!fit) {
      requestAnimationFrame(frame);
      return;
    }

    // Morph the curve toward the current solution.
    const rate = dragIdx >= 0 ? DRAG_RATE : MORPH_RATE;
    const a = 1 - Math.exp(-dt * rate);
    for (let i = 0; i < CURVE_N; i++) shown[i] = shown[i]! + (fit.curve[i]! - shown[i]!) * a;

    for (const p of pts) {
      if (p.ox === 0 && p.oy === 0) continue;
      p.vx += (-SPRING_K * p.ox - SPRING_C * p.vx) * dt;
      p.vy += (-SPRING_K * p.oy - SPRING_C * p.vy) * dt;
      p.ox += p.vx * dt;
      p.oy += p.vy * dt;
      if (Math.abs(p.ox) + Math.abs(p.oy) + Math.abs(p.vx) * 0.05 + Math.abs(p.vy) * 0.05 < 1e-4) {
        p.ox = p.oy = p.vx = p.vy = 0;
      }
    }
    for (let i = pulses.length - 1; i >= 0; i--) {
      pulses[i]!.age += dt;
      if (pulses[i]!.age >= PULSE_LIFE) pulses.splice(i, 1);
    }
    flash = Math.max(0, flash - dt * 1.6);

    const wantRef = fit.reference ? 1 : 0;
    refAlpha += (wantRef - refAlpha) * Math.min(1, dt * 6);
    const wantGhost = fit.ghostDirs.length ? 1 : 0;
    ghostAlpha += (wantGhost - ghostAlpha) * Math.min(1, dt * 3);

    const ghosts: Float64Array[] = [];
    for (let g = 0; g < fit.ghostDirs.length; g++) {
      const dir = fit.ghostDirs[g]!;
      const buf = ghostBuf[g]!;
      const ph = ghostPhase[g]!;
      const t = GHOST_AMP * Math.sin(clock * ph.w + ph.p);
      for (let i = 0; i < CURVE_N; i++) buf[i] = shown[i]! + t * dir[i]!;
      ghosts.push(buf);
    }

    stagePts.length = 0;
    let rs = 0;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i]!;
      stagePts.push({ x: p.x + p.ox, y: p.y + p.oy, hover: i === hoverIdx, drag: i === dragIdx });
      rs += fit.sol.resid[i]! ** 2;
    }
    const rms = Math.sqrt(rs / Math.max(1, pts.length));

    const ghostLabel = fit.ghostDirs.length
      ? fit.m < fit.n && fit.stacks.length === 0
        ? "infinitely many exact fits — A⁺ picks the one with the smallest ‖x‖"
        : "ghosts hit the same values at every data x — A⁺ picks the smallest ‖x‖"
      : null;

    fitCanvas(canvas, sizes.cw, sizes.ch, dpr);
    drawStage(canvas, sizes.cw, sizes.ch, dpr, xs, {
      points: stagePts,
      curve: shown,
      reference: fit.reference,
      refAlpha,
      ghosts,
      ghostAlpha,
      ghostLabel,
      basisCurves: fit.basisCurves,
      badges: badges(),
      chip: `columns φ₀…φ${SUB[degree] ?? degree} · ${basis} · κ ${fmt(fit.full.condFull)}`,
      stacks: fit.stacks,
      pulses,
      flash,
      residScale: Math.max(0.35, rms * 2.2),
    }, clock);

    stepSpectrum(dt);
    const kinds: BarKind[] = [];
    for (let i = 0; i < fit.n; i++) {
      const sg = i < fit.svd.r ? fit.svd.S[i]! : 0;
      kinds.push(sg <= fit.sol.tol ? "null" : i < fit.sol.used ? "kept" : "trunc");
    }
    fitCanvas(specCanvas, sizes.sw, sizes.sh, dpr);
    drawSpectrum(specCanvas, sizes.sw, sizes.sh, dpr, {
      logS: specLog,
      kinds,
      logUb: specUb,
      r: fit.svd.r,
      filter: specFilter,
      filterAlpha,
      lo: specLo,
      hi: specHi,
      logTol: Math.log10(Math.max(1e-300, fit.sol.tol)),
      logLambda: fit.lambda > 0 ? Math.log10(fit.lambda) : null,
    });

    requestAnimationFrame(frame);
  }

  window.__pinvSelfTest = () => {
    const r = selfTest();
    console.table(r.cases);
    return r;
  };

  loadPreset("overdetermined", INITIAL_SEED, true);
  refit();
  if (fit) {
    const f: Fit = fit;
    specHi = Math.log10(f.svd.S[0]!) + 0.3;
    specLo = specHi - 3;
  }
  requestAnimationFrame(frame);
}

mount();
