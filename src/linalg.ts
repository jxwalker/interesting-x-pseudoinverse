/**
 * Dense linear algebra for the pseudoinverse playground — no dependencies.
 * Matrices are row-major Float64Array. Sizes here are tiny (≤ 60 × 13), so
 * clarity and accuracy win over blocking tricks.
 */

export const EPS = 2.220446049250313e-16;

export interface Mat {
  rows: number;
  cols: number;
  data: Float64Array;
}

export function mat(rows: number, cols: number): Mat {
  return { rows, cols, data: new Float64Array(rows * cols) };
}

export function transpose(a: Mat): Mat {
  const t = mat(a.cols, a.rows);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) t.data[j * a.rows + i] = a.data[i * a.cols + j]!;
  }
  return t;
}

export function matmul(a: Mat, b: Mat): Mat {
  if (a.cols !== b.rows) throw new Error("matmul: shape mismatch");
  const c = mat(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let k = 0; k < a.cols; k++) {
      const aik = a.data[i * a.cols + k]!;
      if (aik === 0) continue;
      for (let j = 0; j < b.cols; j++) c.data[i * b.cols + j] += aik * b.data[k * b.cols + j]!;
    }
  }
  return c;
}

export function matvec(a: Mat, x: ArrayLike<number>): Float64Array {
  const y = new Float64Array(a.rows);
  for (let i = 0; i < a.rows; i++) {
    let s = 0;
    for (let j = 0; j < a.cols; j++) s += a.data[i * a.cols + j]! * x[j]!;
    y[i] = s;
  }
  return y;
}

export function norm2(v: ArrayLike<number>): number {
  // Scaled to dodge overflow/underflow on the extreme vectors truncation can produce.
  let scale = 0;
  for (let i = 0; i < v.length; i++) scale = Math.max(scale, Math.abs(v[i]!));
  if (scale === 0 || !Number.isFinite(scale)) return scale;
  let s = 0;
  for (let i = 0; i < v.length; i++) {
    const t = v[i]! / scale;
    s += t * t;
  }
  return scale * Math.sqrt(s);
}

function frob(a: Mat): number {
  return norm2(a.data);
}

function sub(a: Mat, b: Mat): Mat {
  const c = mat(a.rows, a.cols);
  for (let i = 0; i < c.data.length; i++) c.data[i] = a.data[i]! - b.data[i]!;
  return c;
}

// ---------------------------------------------------------------------------
// SVD — one-sided Jacobi (Hestenes)

export interface SVD {
  m: number;
  n: number;
  /** r = min(m, n) */
  r: number;
  /** m × r, orthonormal columns (columns for σ = 0 may be zero). */
  U: Mat;
  /** r singular values, descending. */
  S: Float64Array;
  /** n × r, orthonormal columns. */
  V: Mat;
  sweeps: number;
}

interface Thin {
  U: Float64Array[];
  S: Float64Array;
  V: Float64Array[];
  sweeps: number;
}

/**
 * Hestenes one-sided Jacobi for m ≥ n: orthogonalise the columns of A by plane
 * rotations accumulated into V. High relative accuracy, so the tiny singular
 * values of a cond-1e12 Vandermonde still come out right.
 */
function jacobiTall(a: Mat): Thin {
  const m = a.rows;
  const n = a.cols;
  const U: Float64Array[] = [];
  const V: Float64Array[] = [];
  for (let j = 0; j < n; j++) {
    const col = new Float64Array(m);
    for (let i = 0; i < m; i++) col[i] = a.data[i * n + j]!;
    U.push(col);
    const e = new Float64Array(n);
    e[j] = 1;
    V.push(e);
  }

  let sweeps = 0;
  for (; sweeps < 80; sweeps++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const up = U[p]!;
        const uq = U[q]!;
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let i = 0; i < m; i++) {
          const x = up[i]!;
          const y = uq[i]!;
          alpha += x * x;
          beta += y * y;
          gamma += x * y;
        }
        if (gamma === 0 || Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t =
          Math.abs(zeta) > 1e150
            ? 1 / (2 * zeta)
            : (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let i = 0; i < m; i++) {
          const x = up[i]!;
          const y = uq[i]!;
          up[i] = c * x - s * y;
          uq[i] = s * x + c * y;
        }
        const vp = V[p]!;
        const vq = V[q]!;
        for (let i = 0; i < n; i++) {
          const x = vp[i]!;
          const y = vq[i]!;
          vp[i] = c * x - s * y;
          vq[i] = s * x + c * y;
        }
      }
    }
    if (!rotated) break;
  }

  const S = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const s = norm2(U[j]!);
    S[j] = s;
    if (s > 1e-300) {
      const col = U[j]!;
      for (let i = 0; i < m; i++) col[i] = col[i]! / s;
    } else {
      U[j]!.fill(0);
    }
  }

  const order = Array.from({ length: n }, (_, j) => j).sort((i, j) => S[j]! - S[i]!);
  return {
    U: order.map((j) => U[j]!),
    S: Float64Array.from(order, (j) => S[j]!),
    V: order.map((j) => V[j]!),
    sweeps,
  };
}

function packCols(cols: Float64Array[], rows: number): Mat {
  const out = mat(rows, cols.length);
  for (let j = 0; j < cols.length; j++) {
    const c = cols[j]!;
    for (let i = 0; i < rows; i++) out.data[i * cols.length + j] = c[i]!;
  }
  return out;
}

/** Thin SVD A = U diag(S) Vᵀ for any shape (m < n goes through Aᵀ and swaps U/V). */
export function svd(a: Mat): SVD {
  const m = a.rows;
  const n = a.cols;
  if (m >= n) {
    const t = jacobiTall(a);
    return { m, n, r: n, U: packCols(t.U, m), S: t.S, V: packCols(t.V, n), sweeps: t.sweeps };
  }
  const t = jacobiTall(transpose(a));
  return { m, n, r: m, U: packCols(t.V, m), S: t.S, V: packCols(t.U, n), sweeps: t.sweeps };
}

/** Default numerical-rank threshold: max(m,n)·σ₁·eps. */
export function defaultTol(s: SVD): number {
  return Math.max(s.m, s.n) * (s.S[0] ?? 0) * EPS;
}

export function numericalRank(s: SVD, tol = defaultTol(s)): number {
  let k = 0;
  for (let i = 0; i < s.r; i++) if (s.S[i]! > tol) k++;
  return k;
}

// ---------------------------------------------------------------------------
// Pseudoinverse solves

export interface SolveOpts {
  /** Keep only the top-k singular values (TSVD). Defaults to all. */
  keep?: number;
  /** Tikhonov / ridge λ ≥ 0. */
  lambda?: number;
}

export interface Solution {
  x: Float64Array;
  /** r = b − A x */
  resid: Float64Array;
  residNorm: number;
  xNorm: number;
  /** ‖A x‖ */
  fitNorm: number;
  bNorm: number;
  /** Singular values actually inverted (min(keep, numerical rank)). */
  used: number;
  /** Numerical rank at the default tolerance. */
  rank: number;
  tol: number;
  /** σ₁ / σ_used */
  condKept: number;
  /** σ₁ / σ_min over all r singular values (∞ if rank-deficient). */
  condFull: number;
  /** uᵢᵀ b for i < r (the Picard coefficients). */
  ub: Float64Array;
  /** Filter factors: 1 for plain pinv, σ²/(σ²+λ²) for ridge, 0 if dropped. */
  filter: Float64Array;
}

/**
 * x̂ = V Σ⁺ Uᵀ b, with Σ⁺ inverting only σ > tol, honouring a top-k truncation
 * and optional ridge filter factors fᵢ = σᵢ²/(σᵢ²+λ²).
 */
export function solve(a: Mat, s: SVD, b: ArrayLike<number>, opts: SolveOpts = {}): Solution {
  const { m, n, r } = s;
  const tol = defaultTol(s);
  const rank = numericalRank(s, tol);
  const keep = Math.max(0, Math.min(opts.keep ?? r, r));
  const lambda = Math.max(0, opts.lambda ?? 0);
  const used = Math.min(keep, rank);

  const ub = new Float64Array(r);
  for (let i = 0; i < r; i++) {
    let d = 0;
    for (let k = 0; k < m; k++) d += s.U.data[k * r + i]! * b[k]!;
    ub[i] = d;
  }

  const filter = new Float64Array(r);
  const x = new Float64Array(n);
  for (let i = 0; i < used; i++) {
    const sig = s.S[i]!;
    const f = lambda > 0 ? (sig * sig) / (sig * sig + lambda * lambda) : 1;
    filter[i] = f;
    const w = (f * ub[i]!) / sig;
    for (let j = 0; j < n; j++) x[j] += w * s.V.data[j * r + i]!;
  }

  const ax = matvec(a, x);
  const resid = new Float64Array(m);
  for (let i = 0; i < m; i++) resid[i] = b[i]! - ax[i]!;

  const s1 = s.S[0] ?? 0;
  const smin = r > 0 ? s.S[r - 1]! : 0;
  return {
    x,
    resid,
    residNorm: norm2(resid),
    xNorm: norm2(x),
    fitNorm: norm2(ax),
    bNorm: norm2(b),
    used,
    rank,
    tol,
    condKept: used > 0 ? s1 / s.S[used - 1]! : Infinity,
    condFull: smin > tol ? s1 / smin : Infinity,
    ub,
    filter,
  };
}

/** Explicit A⁺ (n × m) at the default tolerance. Only for the self-test. */
export function pinv(a: Mat, s: SVD = svd(a)): Mat {
  const { m, n, r } = s;
  const tol = defaultTol(s);
  const p = mat(n, m);
  for (let i = 0; i < r; i++) {
    const sig = s.S[i]!;
    if (sig <= tol) continue;
    for (let j = 0; j < n; j++) {
      const v = s.V.data[j * r + i]! / sig;
      if (v === 0) continue;
      for (let k = 0; k < m; k++) p.data[j * m + k] += v * s.U.data[k * r + i]!;
    }
  }
  return p;
}

/**
 * Orthonormal basis of null(A): right singular vectors with σ ≤ tol, plus the
 * orthogonal complement of V's columns when n > m (those directions have no σ).
 */
export function nullBasis(s: SVD, tol = defaultTol(s)): Float64Array[] {
  const { n, r } = s;
  const range: Float64Array[] = [];
  const nul: Float64Array[] = [];
  for (let i = 0; i < r; i++) {
    const v = new Float64Array(n);
    for (let j = 0; j < n; j++) v[j] = s.V.data[j * r + i]!;
    (s.S[i]! > tol ? range : nul).push(v);
  }
  const basis = [...range, ...nul];
  for (let e = 0; e < n && basis.length < n; e++) {
    const v = new Float64Array(n);
    v[e] = 1;
    // Two passes of modified Gram–Schmidt for a clean complement.
    for (let pass = 0; pass < 2; pass++) {
      for (const q of basis) {
        let d = 0;
        for (let j = 0; j < n; j++) d += q[j]! * v[j]!;
        for (let j = 0; j < n; j++) v[j] = v[j]! - d * q[j]!;
      }
    }
    const nv = norm2(v);
    if (nv < 1e-8) continue;
    for (let j = 0; j < n; j++) v[j] = v[j]! / nv;
    basis.push(v);
    nul.push(v);
  }
  return nul;
}

// ---------------------------------------------------------------------------
// Polynomial bases on x ∈ [−1, 1]

export type Basis = "monomial" | "chebyshev";

/** φ₀(x)…φ_d(x) into out (length ≥ d + 1). */
export function basisRow(x: number, degree: number, basis: Basis, out: Float64Array): void {
  out[0] = 1;
  if (degree < 1) return;
  out[1] = x;
  if (basis === "monomial") {
    for (let k = 2; k <= degree; k++) out[k] = out[k - 1]! * x;
  } else {
    for (let k = 2; k <= degree; k++) out[k] = 2 * x * out[k - 1]! - out[k - 2]!;
  }
}

/** Design matrix A[i, k] = φ_k(xᵢ), with x already scaled to [−1, 1]. */
export function designMatrix(xs: ArrayLike<number>, degree: number, basis: Basis): Mat {
  const n = degree + 1;
  const a = mat(xs.length, n);
  const row = new Float64Array(n);
  for (let i = 0; i < xs.length; i++) {
    basisRow(xs[i]!, degree, basis, row);
    a.data.set(row, i * n);
  }
  return a;
}

// ---------------------------------------------------------------------------
// Self-test: the four Penrose conditions on a few shapes

export interface SelfTestCase {
  name: string;
  shape: string;
  rank: number;
  /** ‖A − U Σ Vᵀ‖_F / ‖A‖_F */
  recon: number;
  /** ‖A A⁺ A − A‖ / ‖A‖ */
  p1: number;
  /** ‖A⁺ A A⁺ − A⁺‖ / ‖A⁺‖ */
  p2: number;
  /** ‖(A A⁺)ᵀ − A A⁺‖ */
  p3: number;
  /** ‖(A⁺ A)ᵀ − A⁺ A‖ */
  p4: number;
  ok: boolean;
}

export interface SelfTestResult {
  ok: boolean;
  cases: SelfTestCase[];
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

export function selfTest(seed = 12345): SelfTestResult {
  const rnd = mulberry(seed);
  const randMat = (m: number, n: number): Mat => {
    const a = mat(m, n);
    for (let i = 0; i < a.data.length; i++) a.data[i] = rnd() * 2 - 1;
    return a;
  };
  const lowRank = (m: number, n: number, k: number): Mat => matmul(randMat(m, k), randMat(k, n));
  const vander = (m: number, d: number, basis: Basis): Mat =>
    designMatrix(
      Array.from({ length: m }, (_, i) => -1 + (2 * i) / (m - 1)),
      d,
      basis,
    );

  const inputs: Array<[string, Mat]> = [
    ["random tall", randMat(9, 5)],
    ["random wide", randMat(4, 7)],
    ["rank-deficient", lowRank(8, 6, 3)],
    ["wide rank-deficient", lowRank(5, 9, 2)],
    ["monomial Vandermonde d=12", vander(40, 12, "monomial")],
    ["Chebyshev Vandermonde d=12", vander(40, 12, "chebyshev")],
  ];

  const cases = inputs.map(([name, a]): SelfTestCase => {
    const s = svd(a);
    const p = pinv(a, s);
    const sig = mat(s.r, s.r);
    for (let i = 0; i < s.r; i++) sig.data[i * s.r + i] = s.S[i]!;
    const usv = matmul(matmul(s.U, sig), transpose(s.V));
    const ap = matmul(a, p);
    const pa = matmul(p, a);
    const na = frob(a);
    const np = Math.max(frob(p), 1e-300);
    const c: SelfTestCase = {
      name,
      shape: `${a.rows}×${a.cols}`,
      rank: numericalRank(s),
      recon: frob(sub(a, usv)) / na,
      p1: frob(sub(matmul(ap, a), a)) / na,
      p2: frob(sub(matmul(pa, p), p)) / np,
      p3: frob(sub(transpose(ap), ap)),
      p4: frob(sub(transpose(pa), pa)),
      ok: false,
    };
    // Ill-conditioned inputs lose ~κ·eps in A⁺-relative checks; scale the bar accordingly.
    const kappa = numericalRank(s) > 0 ? s.S[0]! / s.S[numericalRank(s) - 1]! : 1;
    const loose = 1e-10 * Math.max(1, kappa);
    c.ok = c.recon < 1e-12 && c.p1 < 1e-10 && c.p2 < loose && c.p3 < loose && c.p4 < loose;
    return c;
  });

  return { ok: cases.every((c) => c.ok), cases };
}
