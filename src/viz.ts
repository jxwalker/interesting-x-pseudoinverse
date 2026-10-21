/** Canvas 2D drawing for the plot stage and the singular-value spectrum. */

export const BG = "#050a12";
export const CURVE_N = 300;
export const X_MIN = -1.12;
export const X_MAX = 1.12;
export const Y_MIN = -2.1;
export const Y_MAX = 2.1;

type RGB = readonly [number, number, number];
const rgba = (c: RGB, a: number): string => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

const CYAN: RGB = [126, 232, 255];
const TEAL: RGB = [94, 234, 212];
const AMBER: RGB = [251, 191, 36];
const CORAL: RGB = [255, 107, 74];
const HOT: RGB = [255, 64, 48];
const MAGENTA: RGB = [255, 79, 216];
const VIOLET: RGB = [180, 160, 255];
const MUTED: RGB = [138, 163, 184];

const MONO = "'IBM Plex Mono', ui-monospace, monospace";

/** Size the backing store for DPR; returns true if it changed (which clears the canvas). */
export function fitCanvas(canvas: HTMLCanvasElement, cssW: number, cssH: number, dpr: number): boolean {
  const bw = Math.max(1, Math.round(cssW * dpr));
  const bh = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width === bw && canvas.height === bh) return false;
  canvas.width = bw;
  canvas.height = bh;
  return true;
}

function ctx2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas unavailable");
  return ctx;
}

function mix(a: RGB, b: RGB, t: number): RGB {
  const u = Math.min(1, Math.max(0, t));
  return [
    Math.round(a[0] + (b[0] - a[0]) * u),
    Math.round(a[1] + (b[1] - a[1]) * u),
    Math.round(a[2] + (b[2] - a[2]) * u),
  ];
}

/** Residual colour: cool teal for tiny, amber, then red-hot. */
function residRGB(t: number): RGB {
  return t < 0.5 ? mix(TEAL, AMBER, t * 2) : mix(AMBER, HOT, (t - 0.5) * 2);
}

// ---------------------------------------------------------------------------
// Plot geometry

export interface Plot {
  left: number;
  top: number;
  w: number;
  h: number;
}

export function plotRect(cssW: number, cssH: number): Plot {
  const narrow = cssW < 520;
  const left = narrow ? 26 : 36;
  const right = narrow ? 8 : 14;
  const top = narrow ? 34 : 40;
  const bottom = narrow ? 22 : 26;
  return { left, top, w: Math.max(10, cssW - left - right), h: Math.max(10, cssH - top - bottom) };
}

export const toPxX = (p: Plot, x: number): number => p.left + ((x - X_MIN) / (X_MAX - X_MIN)) * p.w;
export const toPxY = (p: Plot, y: number): number => p.top + ((Y_MAX - y) / (Y_MAX - Y_MIN)) * p.h;
export const fromPxX = (p: Plot, px: number): number => X_MIN + ((px - p.left) / p.w) * (X_MAX - X_MIN);
export const fromPxY = (p: Plot, py: number): number => Y_MAX - ((py - p.top) / p.h) * (Y_MAX - Y_MIN);

/** Sample abscissae for curves (uniform over the plotted x range). */
export function curveXs(): Float64Array {
  return Float64Array.from({ length: CURVE_N }, (_, i) => X_MIN + ((X_MAX - X_MIN) * i) / (CURVE_N - 1));
}

/** Linear interpolation of a sampled curve at x. */
export function sampleAt(curve: Float64Array, x: number): number {
  const u = ((x - X_MIN) / (X_MAX - X_MIN)) * (CURVE_N - 1);
  const i = Math.max(0, Math.min(CURVE_N - 2, Math.floor(u)));
  const f = Math.min(1, Math.max(0, u - i));
  return curve[i]! * (1 - f) + curve[i + 1]! * f;
}

// ---------------------------------------------------------------------------
// Sprites

let dotSprite: { key: string; img: HTMLCanvasElement; size: number } | null = null;

function getDot(dpr: number): { img: HTMLCanvasElement; size: number } {
  const key = String(dpr);
  if (dotSprite?.key === key) return dotSprite;
  const size = 34;
  const px = Math.ceil(size * dpr);
  const cv = document.createElement("canvas");
  cv.width = px;
  cv.height = px;
  const g = ctx2d(cv);
  const m = px / 2;
  const grad = g.createRadialGradient(m, m, 0, m, m, m);
  grad.addColorStop(0, "rgba(240,255,255,1)");
  grad.addColorStop(0.12, rgba(CYAN, 1));
  grad.addColorStop(0.2, rgba(CYAN, 0.75));
  grad.addColorStop(0.32, rgba(CYAN, 0.22));
  grad.addColorStop(0.6, rgba(CYAN, 0.06));
  grad.addColorStop(1, rgba(CYAN, 0));
  g.fillStyle = grad;
  g.fillRect(0, 0, px, px);
  dotSprite = { key, img: cv, size };
  return dotSprite;
}

// ---------------------------------------------------------------------------
// Stage

export interface StagePoint {
  /** Display position (data units, includes burst offset). */
  x: number;
  y: number;
  hover: boolean;
  drag: boolean;
}

export interface Badge {
  text: string;
  tone: "over" | "under" | "rankdef" | "trunc" | "ridge";
}

export interface Pulse {
  x: number;
  y: number;
  age: number;
}

export interface Stack {
  x: number;
  mean: number;
  count: number;
}

export interface StageScene {
  points: readonly StagePoint[];
  curve: Float64Array;
  /** Untruncated, un-regularised least-squares fit (dashed reference) — shown when it differs. */
  reference: Float64Array | null;
  refAlpha: number;
  ghosts: readonly Float64Array[];
  ghostAlpha: number;
  ghostLabel: string | null;
  basisCurves: readonly Float64Array[];
  badges: readonly Badge[];
  chip: string;
  stacks: readonly Stack[];
  pulses: readonly Pulse[];
  /** 0..1 glow boost (basis switch, reseed). */
  flash: number;
  /** Typical residual scale for colouring. */
  residScale: number;
}

export const PULSE_LIFE = 0.9;

const TONE: Record<Badge["tone"], RGB> = {
  over: CYAN,
  under: VIOLET,
  rankdef: CORAL,
  trunc: AMBER,
  ridge: MAGENTA,
};

function clampPx(v: number): number {
  return Math.max(-5000, Math.min(5000, v));
}

function tracePath(ctx: CanvasRenderingContext2D, p: Plot, xs: Float64Array, ys: Float64Array): void {
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const px = toPxX(p, xs[i]!);
    const py = clampPx(toPxY(p, ys[i]!));
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
}

function pill(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  c: RGB,
  fontPx: number,
): number {
  ctx.font = `600 ${fontPx}px ${MONO}`;
  const w = ctx.measureText(text).width + fontPx * 1.2;
  const h = fontPx * 1.9;
  ctx.fillStyle = "rgba(4,12,18,0.78)";
  ctx.strokeStyle = rgba(c, 0.55);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, h / 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = rgba(c, 1);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + fontPx * 0.6, y + h / 2 + 0.5);
  return w;
}

export function drawStage(
  canvas: HTMLCanvasElement,
  cssW: number,
  cssH: number,
  dpr: number,
  xs: Float64Array,
  s: StageScene,
  time: number,
): void {
  const ctx = ctx2d(canvas);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, cssW, cssH);

  const vg = ctx.createRadialGradient(cssW * 0.5, cssH * 0.45, 0, cssW * 0.5, cssH * 0.45, Math.max(cssW, cssH) * 0.7);
  vg.addColorStop(0, "rgba(40,90,130,0.16)");
  vg.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, cssW, cssH);

  const p = plotRect(cssW, cssH);
  const narrow = cssW < 520;
  const fs = narrow ? 9.5 : 10.5;

  // Grid + axes
  ctx.lineWidth = 1;
  ctx.font = `500 ${fs}px ${MONO}`;
  ctx.textBaseline = "middle";
  for (let y = -2; y <= 2; y += 0.5) {
    const py = Math.round(toPxY(p, y)) + 0.5;
    ctx.strokeStyle = y === 0 ? "rgba(140,200,225,0.26)" : "rgba(120,190,220,0.07)";
    ctx.beginPath();
    ctx.moveTo(p.left, py);
    ctx.lineTo(p.left + p.w, py);
    ctx.stroke();
    if (Number.isInteger(y)) {
      ctx.fillStyle = rgba(MUTED, 0.6);
      ctx.textAlign = "right";
      ctx.fillText(y < 0 ? `−${-y}` : String(y), p.left - 6, py);
    }
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let x = -1; x <= 1.001; x += 0.25) {
    const px = Math.round(toPxX(p, x)) + 0.5;
    const major = Math.abs(x - Math.round(x)) < 1e-9;
    ctx.strokeStyle = Math.abs(x) < 1e-9 ? "rgba(140,200,225,0.2)" : major ? "rgba(120,190,220,0.1)" : "rgba(120,190,220,0.05)";
    ctx.beginPath();
    ctx.moveTo(px, p.top);
    ctx.lineTo(px, p.top + p.h);
    ctx.stroke();
    if (major) {
      ctx.fillStyle = rgba(MUTED, 0.6);
      const rx = Math.round(x);
      ctx.fillText(rx < 0 ? `−${-rx}` : String(rx), px, p.top + p.h + 6);
    }
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(p.left, p.top, p.w, p.h);
  ctx.clip();

  // Columns of A, faint: monomials crowd together (ill-conditioned), Chebyshev spreads out.
  ctx.lineWidth = 1;
  for (let k = 0; k < s.basisCurves.length; k++) {
    ctx.strokeStyle = rgba(VIOLET, 0.07 + 0.05 * (k / Math.max(1, s.basisCurves.length - 1)));
    tracePath(ctx, p, xs, s.basisCurves[k]!);
    ctx.stroke();
  }

  // Pulses
  for (const pu of s.pulses) {
    const t = pu.age / PULSE_LIFE;
    if (t >= 1) continue;
    const r = 12 + t * Math.max(p.w, p.h) * 0.55;
    ctx.strokeStyle = rgba(CYAN, 0.35 * (1 - t) * (1 - t));
    ctx.lineWidth = 2 + 6 * (1 - t);
    ctx.beginPath();
    ctx.arc(toPxX(p, pu.x), toPxY(p, pu.y), r, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Reference full-LS fit (dashed) when truncation / ridge changes the curve.
  if (s.reference && s.refAlpha > 0.01) {
    ctx.setLineDash([5, 6]);
    ctx.lineWidth = 1.3;
    ctx.strokeStyle = rgba(AMBER, 0.45 * s.refAlpha);
    tracePath(ctx, p, xs, s.reference);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Ghost exact fits along the null space.
  if (s.ghosts.length && s.ghostAlpha > 0.01) {
    ctx.globalCompositeOperation = "lighter";
    for (let g = 0; g < s.ghosts.length; g++) {
      const c = g % 2 ? VIOLET : CYAN;
      ctx.lineWidth = 4;
      ctx.strokeStyle = rgba(c, 0.06 * s.ghostAlpha);
      tracePath(ctx, p, xs, s.ghosts[g]!);
      ctx.stroke();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = rgba(c, 0.42 * s.ghostAlpha);
      ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";
  }

  // Residual stalks (behind the curve so the curve glow sits on top).
  ctx.lineCap = "round";
  for (const pt of s.points) {
    const cy = sampleAt(s.curve, pt.x);
    const r = pt.y - cy;
    const t = Math.min(1, Math.abs(r) / Math.max(1e-6, s.residScale));
    const px = toPxX(p, pt.x);
    const c = residRGB(t);
    const y0 = toPxY(p, pt.y);
    const y1 = clampPx(toPxY(p, cy));
    if (Math.abs(y1 - y0) < 0.5) continue;
    ctx.strokeStyle = rgba(c, 0.12 + 0.2 * t);
    ctx.lineWidth = 5 + 4 * t;
    ctx.beginPath();
    ctx.moveTo(px, y0);
    ctx.lineTo(px, y1);
    ctx.stroke();
    ctx.strokeStyle = rgba(c, 0.45 + 0.55 * t);
    ctx.lineWidth = 1.2 + 1.6 * t;
    ctx.stroke();
  }

  // Fit curve: layered additive glow with an amber → coral → magenta gradient.
  const grad = ctx.createLinearGradient(p.left, 0, p.left + p.w, 0);
  grad.addColorStop(0, rgba(AMBER, 1));
  grad.addColorStop(0.5, rgba(CORAL, 1));
  grad.addColorStop(1, rgba(MAGENTA, 1));
  const boost = 1 + s.flash * 1.4;
  tracePath(ctx, p, xs, s.curve);
  ctx.lineJoin = "round";
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = grad;
  ctx.globalAlpha = 0.07 * boost;
  ctx.lineWidth = 16 * Math.sqrt(boost);
  ctx.stroke();
  ctx.globalAlpha = 0.16 * boost;
  ctx.lineWidth = 7;
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.lineWidth = 2.6;
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,246,225,0.75)";
  ctx.lineWidth = 0.9;
  ctx.stroke();

  // Stack means (rank-deficient: the fit can only hit their average).
  for (const st of s.stacks) {
    const px = toPxX(p, st.x);
    const py = toPxY(p, st.mean);
    ctx.fillStyle = "rgba(255,240,210,0.95)";
    ctx.strokeStyle = rgba(CORAL, 0.9);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, py - 6);
    ctx.lineTo(px + 6, py);
    ctx.lineTo(px, py + 6);
    ctx.lineTo(px - 6, py);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.font = `600 ${fs}px ${MONO}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = rgba(CORAL, 0.95);
    ctx.fillText(`mean of ${st.count}`, px + 10, py);
  }

  ctx.restore();

  // Data points
  const dot = getDot(dpr);
  ctx.globalCompositeOperation = "lighter";
  for (const pt of s.points) {
    const big = pt.drag ? 1.6 : pt.hover ? 1.35 : 1;
    const sz = dot.size * big;
    const px = toPxX(p, pt.x);
    const py = toPxY(p, pt.y);
    ctx.drawImage(dot.img, px - sz / 2, py - sz / 2, sz, sz);
  }
  ctx.globalCompositeOperation = "source-over";
  for (const pt of s.points) {
    if (!pt.hover && !pt.drag) continue;
    const px = toPxX(p, pt.x);
    const py = toPxY(p, pt.y);
    ctx.strokeStyle = rgba(CYAN, pt.drag ? 0.95 : 0.7);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(px, py, 11 + Math.sin(time * 6) * 1.2, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Regime badges
  let bx = 10;
  let by = 8;
  const bfs = narrow ? 9 : 10.5;
  for (const b of s.badges) {
    ctx.font = `600 ${bfs}px ${MONO}`;
    const w = ctx.measureText(b.text).width + bfs * 1.2;
    if (bx + w > cssW - 10 && bx > 10) {
      bx = 10;
      by += bfs * 2.2;
    }
    bx += pill(ctx, bx, by, b.text, TONE[b.tone], bfs) + 6;
  }

  // Ghost explanation
  if (s.ghostLabel && s.ghostAlpha > 0.05) {
    ctx.globalAlpha = Math.min(1, s.ghostAlpha);
    ctx.font = `600 ${fs}px ${MONO}`;
    const maxW = p.w - 20;
    const lines = ctx.measureText(s.ghostLabel).width > maxW ? s.ghostLabel.split(" — ") : [s.ghostLabel];
    let ly = p.top + p.h - 10 - lines.length * fs * 2.1;
    for (const line of lines) {
      ctx.font = `600 ${fs}px ${MONO}`;
      const w = ctx.measureText(line).width + fs * 1.2;
      pill(ctx, p.left + (p.w - w) / 2, ly, line, VIOLET, fs);
      ly += fs * 2.1;
    }
    ctx.globalAlpha = 1;
  }

  // Basis / conditioning chip
  if (s.chip) {
    ctx.font = `500 ${fs}px ${MONO}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = rgba(VIOLET, 0.75);
    ctx.fillText(s.chip, cssW - 10, p.top + p.h - 6 - (s.ghostLabel && s.ghostAlpha > 0.05 ? (narrow ? 2 : 1) * fs * 2.3 + 4 : 0));
  }
  ctx.textAlign = "left";
}

// ---------------------------------------------------------------------------
// Singular-value spectrum

export type BarKind = "kept" | "trunc" | "null";

export interface SpectrumScene {
  /** Displayed log10 σᵢ (one per column of A; null-space slots have no σ). */
  logS: Float64Array;
  kinds: readonly BarKind[];
  /** Displayed log10 |uᵢᵀb|, i < r. */
  logUb: Float64Array;
  r: number;
  filter: Float64Array;
  filterAlpha: number;
  lo: number;
  hi: number;
  logTol: number;
  logLambda: number | null;
}

export function drawSpectrum(canvas: HTMLCanvasElement, cssW: number, cssH: number, dpr: number, s: SpectrumScene): void {
  const ctx = ctx2d(canvas);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const n = s.kinds.length;
  const stripH = s.filterAlpha > 0.01 ? 18 : 0;
  const left = 34;
  const right = 6;
  const top = 6;
  const bottom = 16 + stripH;
  const w = Math.max(10, cssW - left - right);
  const h = Math.max(10, cssH - top - bottom);
  const span = Math.max(0.5, s.hi - s.lo);
  const yOf = (lg: number): number => top + h * (1 - (lg - s.lo) / span);

  // Decade gridlines
  ctx.font = `500 9.5px ${MONO}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const step = span > 10 ? 3 : span > 5 ? 2 : 1;
  for (let d = Math.ceil(s.lo); d <= Math.floor(s.hi); d++) {
    if (d % step) continue;
    const y = Math.round(yOf(d)) + 0.5;
    ctx.strokeStyle = "rgba(120,190,220,0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + w, y);
    ctx.stroke();
    ctx.fillStyle = rgba(MUTED, 0.65);
    ctx.fillText(`1e${d}`, left - 4, y);
  }

  // Tolerance line
  if (s.logTol > s.lo && s.logTol < s.hi) {
    const y = yOf(s.logTol);
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = rgba(CORAL, 0.5);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + w, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.textAlign = "right";
    ctx.fillStyle = rgba(CORAL, 0.75);
    ctx.fillText("tol", left + w - 2, y - 7);
  }

  const slot = w / Math.max(1, n);
  const bw = Math.min(30, slot * 0.62);
  const base = top + h;
  for (let i = 0; i < n; i++) {
    const cx = left + slot * (i + 0.5);
    const kind = s.kinds[i]!;
    if (kind === "null") {
      const hh = Math.min(14, h * 0.12);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = rgba(CORAL, 0.6);
      ctx.lineWidth = 1;
      ctx.strokeRect(cx - bw / 2 + 0.5, base - hh, bw - 1, hh);
      ctx.setLineDash([]);
      ctx.fillStyle = rgba(CORAL, 0.85);
      ctx.font = `600 9.5px ${MONO}`;
      ctx.textAlign = "center";
      ctx.fillText("0", cx, base - hh - 7);
    } else {
      const y = Math.min(base - 1, yOf(s.logS[i]!));
      if (kind === "kept") {
        const g = ctx.createLinearGradient(0, y, 0, base);
        g.addColorStop(0, rgba(AMBER, 1));
        g.addColorStop(1, rgba(CORAL, 0.55));
        ctx.shadowColor = rgba(AMBER, 0.7);
        ctx.shadowBlur = 10;
        ctx.fillStyle = g;
        ctx.fillRect(cx - bw / 2, y, bw, base - y);
        ctx.shadowBlur = 0;
        ctx.fillStyle = "rgba(255,246,225,0.9)";
        ctx.fillRect(cx - bw / 2, y, bw, 2);
      } else {
        ctx.fillStyle = "rgba(138,163,184,0.06)";
        ctx.fillRect(cx - bw / 2, y, bw, base - y);
        ctx.strokeStyle = "rgba(138,163,184,0.45)";
        ctx.lineWidth = 1;
        ctx.strokeRect(cx - bw / 2 + 0.5, y + 0.5, bw - 1, base - y - 1);
      }
    }
    // Picard marker |uᵢᵀb|
    if (i < s.r) {
      const ly = s.logUb[i]!;
      if (Number.isFinite(ly) && ly > s.lo) {
        const y = yOf(ly);
        ctx.fillStyle = rgba(CYAN, 0.95);
        ctx.beginPath();
        ctx.moveTo(cx, y - 4);
        ctx.lineTo(cx + 4, y);
        ctx.lineTo(cx, y + 4);
        ctx.lineTo(cx - 4, y);
        ctx.closePath();
        ctx.fill();
      }
    }
    ctx.fillStyle = rgba(MUTED, kind === "kept" ? 0.9 : 0.5);
    ctx.font = `500 9.5px ${MONO}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(String(i + 1), cx, base + 3);
    ctx.textBaseline = "middle";
  }

  // Ridge λ level: σ = λ is where the filter factor crosses ½.
  if (s.logLambda !== null && s.logLambda > s.lo && s.logLambda < s.hi) {
    const y = yOf(s.logLambda);
    ctx.strokeStyle = rgba(MAGENTA, 0.85);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + w, y);
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.fillStyle = rgba(MAGENTA, 0.95);
    ctx.font = `600 9.5px ${MONO}`;
    ctx.fillText("λ", left + 3, y - 7);
  }

  // Filter factors strip
  if (stripH) {
    ctx.globalAlpha = s.filterAlpha;
    const y0 = base + 15;
    const y1 = y0 + stripH - 4;
    ctx.strokeStyle = "rgba(120,190,220,0.12)";
    ctx.beginPath();
    ctx.moveTo(left, y1 + 0.5);
    ctx.lineTo(left + w, y1 + 0.5);
    ctx.stroke();
    ctx.font = `500 9px ${MONO}`;
    ctx.textAlign = "right";
    ctx.fillStyle = rgba(MUTED, 0.7);
    ctx.fillText("fᵢ", left - 4, (y0 + y1) / 2);
    for (let i = 0; i < n; i++) {
      const cx = left + slot * (i + 0.5);
      const f = i < s.filter.length ? s.filter[i]! : 0;
      const y = y1 - f * (y1 - y0);
      ctx.fillStyle = f > 0.02 ? rgba(mix(MAGENTA, AMBER, f), 1) : "rgba(138,163,184,0.4)";
      ctx.beginPath();
      ctx.arc(cx, y, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}
