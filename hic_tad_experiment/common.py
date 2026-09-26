"""Shared Hi-C TAD-boundary infrastructure: loading, Knight-Ruiz balancing,
downsampling, DP-NCut / SpectralTAD boundary calling, insulation scoring.

# hyperparameters at top of run_hic_tad.py (trained on chr1)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solvers"))
from wtv_ssc import ssc_admm_sparse_block_tv  # noqa: E402

# load hic straw
try:
    import hicstraw  # noqa: F401
except ImportError:
    _test_env = Path.home() / "miniconda3/envs/test_env/lib/python3.11/site-packages"
    if _test_env.is_dir():
        sys.path.insert(0, str(_test_env))
    import hicstraw  # noqa: E402

BINSIZE = 10_000       # 10 kbp per bin
WINDOW = 200            # 200 bins = 2 Mb sliding-window scan width
MIN_TAD_BINS = 5        # SpectralTAD min_size


# kr code: https://github.com/ay-lab/HiCKRy/blob/master/Scripts/knightRuiz.py
# kr balancing
def knight_ruiz(A, tol=1e-6):
    """Knight-Ruiz inner-outer iteration. Returns [x, n_outer, n_inner] with
    x an (n, 1) vector s.t. diag(x) A diag(x) is doubly stochastic."""
    n = A.shape[0]
    e = np.ones((n, 1), dtype=np.float64)
    Delta, delta = 3, 0.1
    x0 = np.copy(e)
    g = 0.9
    etamax = eta = 0.1
    stop_tol = tol * 0.5
    x = np.copy(x0)
    rt = tol ** 2.0
    v = x * (A.dot(x))
    rk = 1.0 - v
    rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
    rho_km2 = rho_km1
    rout = rold = rho_km1
    MVP = i = k = 0

    while rout > rt:
        i += 1
        if i > 30:
            break
        k = 0
        y = np.copy(e)
        innertol = max(eta ** 2.0 * rout, rt)
        while rho_km1 > innertol:
            k += 1
            if k == 1:
                Z = rk / v
                p = np.copy(Z)
                rho_km1 = (rk.transpose()).dot(Z)
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p
            if k > 10:
                break
            w = x * A.dot(x * p) + v * p
            alpha = rho_km1 / (((p.transpose()).dot(w))[0, 0])
            ap = alpha * p
            ynew = y + ap
            if np.amin(ynew) <= delta:
                if delta == 0:
                    break
                ind = np.where(ap < 0.0)[0]
                gamma = np.amin((delta - y[ind]) / ap[ind])
                y += gamma * ap
                break
            if np.amax(ynew) >= Delta:
                ind = np.where(ynew > Delta)[0]
                gamma = np.amin((Delta - y[ind]) / ap[ind])
                y += gamma * ap
                break
            y = np.copy(ynew)
            rk -= alpha * w
            rho_km2 = rho_km1
            Z = rk / v
            rho_km1 = ((rk.transpose()).dot(Z))[0, 0]
        x *= y
        v = x * (A.dot(x))
        rk = 1.0 - v
        rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
        rout = rho_km1
        MVP += k + 1
        rat = rout / rold
        rold = rout
        res_norm = rout ** 0.5
        eta_o = eta
        eta = g * rat
        if g * eta_o ** 2.0 > 0.1:
            eta = max(eta, g * eta_o ** 2.0)
        eta = max(min(eta, etamax), stop_tol / res_norm)

    return [x, i, k]


def kr_normalize(A, tol=1e-6):
    """KR-balance a nonnegative sparse contact matrix to doubly-stochastic
    (row/col sums ~ 1). Empty (all-zero) rows are left as zeros."""
    A = A.tocsr(copy=True)
    if A.data.size:
        A.data = np.clip(A.data, 0.0, None)
        A.eliminate_zeros()
    n = A.shape[0]
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    valid = row_sum > 0
    n_valid = int(valid.sum())
    if n_valid < 2:
        return A, dict(n=n, n_valid=n_valid, mean_row_sum=float("nan"))

    idx = np.flatnonzero(valid)
    sub = A[idx][:, idx]
    x, n_outer, n_inner = knight_ruiz(sub, tol=tol)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise SystemExit(
            f"KR failed: non-finite or non-positive bias "
            f"(n_valid={n_valid}, outer={n_outer}).")
    D = sps.diags(x, 0, format="csr")
    bal = D.dot(sub.dot(D))
    rs = np.asarray(bal.sum(axis=1)).ravel()
    mean_row_sum = float(rs.mean()) if rs.size else float("nan")
    bal = bal.tocoo()
    out = sps.csr_matrix(
        (bal.data, (idx[bal.row], idx[bal.col])), shape=(n, n), dtype=np.float64)
    info = dict(n=n, n_valid=n_valid, n_outer=int(n_outer), n_inner=int(n_inner),
                mean_row_sum=mean_row_sum)
    return out, info


# ── .hic loading (partition_hic.py) ─────────────────────────────────────────

def load_chrom_matrix(hic_path, chrom, resolution, norm):
    """One hicstraw query for a whole chromosome -> dense symmetric matrix."""
    t0 = time.time()
    chrom_len = {c.name: c.length for c in hicstraw.HiCFile(hic_path).getChromosomes()}[chrom]
    n_bins = int(np.ceil(chrom_len / resolution))
    recs = hicstraw.straw("observed", norm, hic_path, chrom, chrom, "BP", resolution)
    binX = np.fromiter((r.binX for r in recs), dtype=np.int64, count=len(recs)) // resolution
    binY = np.fromiter((r.binY for r in recs), dtype=np.int64, count=len(recs)) // resolution
    counts = np.fromiter((r.counts for r in recs), dtype=np.float64, count=len(recs))
    finite = np.isfinite(counts)
    binX, binY, counts = binX[finite], binY[finite], counts[finite]
    in_range = (binX >= 0) & (binX < n_bins) & (binY >= 0) & (binY < n_bins)
    binX, binY, counts = binX[in_range], binY[in_range], counts[in_range]

    M = np.full((n_bins, n_bins), np.nan)
    M[binX, binY] = counts
    M[binY, binX] = counts
    print(f"  loaded chr{chrom} ({n_bins} bins, {len(recs)} records) in "
          f"{time.time() - t0:.1f}s", flush=True)
    return M


def dense_to_csr(M):
    Y = np.nan_to_num(np.asarray(M, dtype=np.float64), nan=0.0)
    Y[~np.isfinite(Y)] = 0.0
    return csr_matrix(Y)


# ── Binomial downsampling (compare_replicate_dp_ncut.py) ───────────────────

def binomial_downsample_csr(M, frac, rng):
    """Independent Binomial(round(m_ij), frac) on stored contacts, symmetric.
    Intended for raw observed (NONE) counts, before KR balancing."""
    frac = float(frac)
    if frac >= 1.0:
        return M, dict(applied=False)
    A = M.tocoo()
    if A.nnz == 0:
        return M, dict(applied=False)
    upper = A.row <= A.col
    row, col = A.row[upper], A.col[upper]
    counts = np.rint(np.clip(A.data[upper], 0.0, None)).astype(np.int64)
    drawn = rng.binomial(counts, frac).astype(np.float64)
    nz = drawn > 0.0
    row, col, drawn = row[nz], col[nz], drawn[nz]
    diag = row == col
    rows = np.concatenate([row, col[~diag]])
    cols = np.concatenate([col, row[~diag]])
    data = np.concatenate([drawn, drawn[~diag]])
    out = csr_matrix((data, (rows, cols)), shape=A.shape)
    sum_before, sum_after = float(counts.sum()), float(drawn.sum())
    info = dict(applied=True, n_entries=int(upper.sum()), n_kept=int(nz.sum()),
                keep_rate=(sum_after / sum_before) if sum_before > 0 else float("nan"))
    return out, info


def downsample_seed(base, tag):
    return int(base) + 100003 * (sum((i + 1) * ord(c) for i, c in enumerate(str(tag))) + 1)


def load_chrom(hic, chrom, resolution, downsample_frac=1.0, downsample_seed_=0, tag=""):
    """Load one chromosome's observed (NONE) counts, binomial-thin to
    ``downsample_frac`` (no-op at 1.0), then Knight-Ruiz balance ourselves.

    The original run this reproduces always passes --observed-kr, i.e. it
    recomputes KR from raw observed counts even at p=1.0 rather than using
    the .hic file's own pre-balanced "KR" vectors -- matching that (not the
    file's KR) is what made this port's numbers reproduce the original's."""
    prefix = f"{tag} " if tag else ""
    frac = float(downsample_frac)
    print(f"Loading {prefix}chr{chrom} observed (p={frac:g}) ...", flush=True)
    dense = load_chrom_matrix(hic, chrom, resolution, "NONE")
    M = dense_to_csr(dense)
    del dense
    M, ds = binomial_downsample_csr(
        M, frac, np.random.default_rng(downsample_seed(downsample_seed_, f"{tag}:{chrom}")))
    if ds.get("applied"):
        print(f"  chr{chrom} binomial p={frac:g}  keep_rate={ds['keep_rate']:.4f}",
              flush=True)
    M, kr = kr_normalize(M)
    print(f"  chr{chrom} KR  n_valid={kr['n_valid']}/{kr['n']}  "
          f"mean_row_sum={kr['mean_row_sum']:.6g}", flush=True)
    return M, M.shape[0]


# ── Sliding-window scan driver (tad_common.py) ──────────────────────────────

def _scan_region(cut_fn, start_bin, end_bin, window, min_tad_bins=MIN_TAD_BINS,
                 verbose=False):
    """SpectralTAD's sliding window. cut_fn(pos, end) -> (starts, scores);
    every domain but the last is accepted and the window restarts at the last
    domain's start (re-cut with full right-hand context)."""
    pos = start_bin
    tads: list[tuple[int, int]] = []
    scores: list[float] = []
    while pos < end_bin - min_tad_bins:
        end = min(pos + window, end_bin)
        if end - pos < 2 * min_tad_bins:
            tads.append((pos, end_bin))
            scores.append(0.0)
            break
        starts, cut_scores = cut_fn(pos, end)
        if starts is None:
            pos += max(window // 4, 1)
            continue
        at_end = end >= end_bin
        n_keep = len(starts) if (at_end or len(starts) < 2) else len(starts) - 1
        bounds = list(starts) + [end]
        for i in range(n_keep):
            tads.append((bounds[i], bounds[i + 1]))
            scores.append(cut_scores[i] if cut_scores else 0.0)
        pos = bounds[n_keep] if bounds[n_keep] > pos else end
    if not tads:
        tads.append((start_bin, end_bin))
    elif tads[-1][1] < end_bin:
        tads.append((tads[-1][1], end_bin))
    if verbose:
        print(f"  scan complete: {len(tads)} TADs", flush=True)
    return tads


def silhouette_postprocess(tads, min_tad_bins=MIN_TAD_BINS):
    """Package default (SpectralTAD qual_filter=FALSE): keep the scan's
    contiguous partition, just drop domains under the minimum size."""
    return [t for t in tads if t[1] - t[0] >= min_tad_bins]


# ── SpectralTAD reference cutter (unit-circle embedding) ───────────────────

def _unit_circle_gaps(A):
    """Two largest-magnitude eigenpairs of the degree-normalized affinity,
    projected onto the unit circle; gaps[i] = dist(row i-1, row i)."""
    d = np.abs(A).sum(axis=1)
    inv = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))
    L = inv[:, None] * A * inv[None, :]
    L[~np.isfinite(L)] = 0.0
    try:
        vals, vecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.full(A.shape[0], np.nan)
    V = vecs[:, np.argsort(-np.abs(vals))[:2]]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    gaps = np.full(A.shape[0], np.nan)
    gaps[1:] = np.linalg.norm(np.diff(V, axis=0), axis=1)
    return gaps


def _greedy_cut_candidates(gaps, min_size, k_max):
    order = np.argsort(np.where(np.isfinite(gaps), -gaps, np.inf))
    cuts: list[int] = []
    for idx in order:
        if not np.isfinite(gaps[idx]):
            break
        idx = int(idx)
        if any(abs(idx - c) <= min_size for c in cuts):
            continue
        cuts.append(idx)
        if len(cuts) >= k_max:
            break
    return cuts


def window_cuts(affinity, contacts, min_size=MIN_TAD_BINS):
    """Local cut positions for one window: embed `affinity`, pick k by
    silhouette on d = 1/(1+contacts), stop at the first k that beats k+1."""
    W = affinity.shape[0]
    cand = _greedy_cut_candidates(_unit_circle_gaps(affinity), min_size,
                                  int(np.ceil(W / max(min_size, 1))))
    if not cand:
        return []
    D = 1.0 / (1.0 + np.maximum(contacts, 0.0))
    np.fill_diagonal(D, 0.0)
    best_sil, best_cuts = None, []
    for k in range(2, len(cand) + 2):
        cuts = sorted(cand[:k - 1])
        labels = np.zeros(W, dtype=int)
        for c in cuts:
            labels[c:] += 1
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(D, labels, metric="precomputed"))
        if best_sil is not None and sil < best_sil:
            break
        best_sil, best_cuts = sil, cuts
    return best_cuts


def spectral_cut_fn(M_sparse, min_tad_bins):
    def cut(pos, end):
        Y_raw = M_sparse[pos:end, pos:end].toarray()
        keep = np.flatnonzero(Y_raw.any(axis=0))
        if keep.size < 2 * min_tad_bins:
            return None, None
        Y = Y_raw[np.ix_(keep, keep)]
        if not np.isfinite(Y).all() or Y.max() <= 0:
            return None, None
        cuts = window_cuts(Y, Y, min_tad_bins)
        starts = sorted({pos} | {pos + int(keep[c]) for c in cuts})
        return starts, None
    return cut


def run_spectral_tad(M_sparse, n_bins, start_bin, end_bin, window=WINDOW,
                     min_tad_bins=MIN_TAD_BINS, verbose=True):
    cut = spectral_cut_fn(M_sparse, min_tad_bins)
    tads = _scan_region(cut, start_bin, end_bin, window, min_tad_bins, verbose=verbose)
    return silhouette_postprocess(tads, min_tad_bins)


# ── DP-NCut contiguous cutter (dp_contiguous_partition.py) ──────────────────

def dp_contiguous_ncut_all_k(C, k_max, min_size=1):
    """DP-NCut for every feasible k in 1..k_max from one DP fill. C is a
    square, symmetric, nonnegative affinity (not re-symmetrized here).
    Returns {k: (labels, boundaries, ncut_value)}."""
    C = np.asarray(C, dtype=float)
    N = C.shape[0]
    k_max = min(int(k_max), N // min_size)
    if k_max < 1:
        raise RuntimeError(f"k_max*min_size exceeds N={N}.")
    W = np.array(C, copy=True)
    np.fill_diagonal(W, 0.0)
    S = np.zeros((N + 1, N + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(W, axis=0), axis=1)
    deg_cum = np.concatenate([[0.0], np.cumsum(W.sum(axis=1))])

    def within(a, b):
        return S[b, b] - S[a, b] - S[b, a] + S[a, a]

    def vol(a, b):
        return deg_cum[b] - deg_cum[a]

    NEG_INF = -np.inf
    gain = np.full((N + 1, N + 1), NEG_INF)
    for a in range(N):
        for b in range(a + min_size, N + 1):
            v = vol(a, b)
            gain[a, b] = within(a, b) / v if v > 0 else 0.0

    dp = np.full((k_max + 1, N + 1), NEG_INF)
    dp[0, 0] = 0.0
    back = np.full((k_max + 1, N + 1), -1, dtype=int)
    for s in range(1, k_max + 1):
        lo = s * min_size
        for b in range(lo, N + 1):
            a_lo, a_hi = (s - 1) * min_size, b - min_size
            best_val, best_a = NEG_INF, -1
            for a in range(a_lo, a_hi + 1):
                prev = dp[s - 1, a]
                if prev == NEG_INF:
                    continue
                val = prev + gain[a, b]
                if val > best_val:
                    best_val, best_a = val, a
            dp[s, b] = best_val
            back[s, b] = best_a

    out = {}
    for k in range(1, k_max + 1):
        if dp[k, N] == NEG_INF:
            continue
        boundaries = [N]
        b, s = N, k
        while s > 0:
            a = back[s, b]
            boundaries.append(a)
            b, s = a, s - 1
        boundaries.reverse()
        labels = np.empty(N, dtype=int)
        for seg_idx in range(k):
            labels[boundaries[seg_idx]:boundaries[seg_idx + 1]] = seg_idx
        out[k] = (labels, boundaries, float(k - dp[k, N]))
    if not out:
        raise RuntimeError("No feasible contiguous partition found.")
    return out


def dp_ncut_silhouette_cuts(A, Y, min_size, k_min=2, k_max=4, k_select="argmax"):
    """Silhouette sweep over DP-NCut(A) for k in [k_min, k_max]. "argmax" (the
    default here, matching the run this reproduces) keeps the best-silhouette
    k in that fixed range; "first_drop" stops at the first decrease instead."""
    n = A.shape[0]
    k_lo, k_hi = max(int(k_min), 2), min(int(k_max), n // max(min_size, 1))
    if k_hi < k_lo:
        return []
    try:
        parts = dp_contiguous_ncut_all_k(A, k_hi, min_size=min_size)
    except (ValueError, RuntimeError):
        return []
    D = 1.0 / (1.0 + np.maximum(Y, 0.0))
    np.fill_diagonal(D, 0.0)
    best_sil, best_cuts = None, []
    for k in range(k_lo, k_hi + 1):
        if k not in parts:
            break
        labels, bounds, _ = parts[k]
        if len(np.unique(labels)) < 2:
            continue
        sil = float(silhouette_score(D, labels, metric="precomputed"))
        if k_select == "argmax":
            if best_sil is None or sil > best_sil:
                best_sil, best_cuts = sil, [int(b) for b in bounds[1:-1]]
            continue
        if best_sil is not None and sil < best_sil:
            break
        best_sil, best_cuts = sil, [int(b) for b in bounds[1:-1]]
    return best_cuts


# ── WTV-SSC cutter + scan ────────────────────────────────────────────────────

def block_tv_cut_fn(M_sparse, min_tad_bins, solver_kwargs, dp_k_min=2, dp_k_max=4,
                    dp_k_select="argmax"):
    """Cutter: affinity W = |Z| + |Z^T| from ssc_admm_sparse_block_tv, cut by
    DP-NCut over k in [dp_k_min, dp_k_max]."""
    def cut(pos, end):
        Y_raw = M_sparse[pos:end, pos:end].toarray()
        keep = np.flatnonzero(Y_raw.any(axis=0))
        if keep.size < 2 * min_tad_bins:
            return None, None
        Y = Y_raw[np.ix_(keep, keep)]
        if not np.isfinite(Y).all() or Y.max() <= 0:
            return None, None
        Y_sol = Y.copy()
        np.fill_diagonal(Y_sol, 0.0)
        Z, _C, _info = ssc_admm_sparse_block_tv(Y_sol, **solver_kwargs)
        A = np.abs(Z) + np.abs(Z.T)
        cuts = dp_ncut_silhouette_cuts(A, Y, min_tad_bins, dp_k_min, dp_k_max, dp_k_select)
        starts = sorted({pos} | {pos + int(keep[c]) for c in cuts})
        return starts, None
    return cut


def run_block_tv_tad(M_sparse, n_bins, start_bin, end_bin, window, min_tad_bins,
                     solver_kwargs, dp_k_min=2, dp_k_max=4, dp_k_select="argmax",
                     verbose=True):
    cut = block_tv_cut_fn(M_sparse, min_tad_bins, solver_kwargs, dp_k_min, dp_k_max,
                          dp_k_select)
    tads = _scan_region(cut, start_bin, end_bin, window, min_tad_bins, verbose=verbose)
    return silhouette_postprocess(tads, min_tad_bins)


# ── Insulation score + boundary stats (tad_common.py) ───────────────────────

def compute_insulation_score(M_sparse, n_bins, delta=25):
    """Crane et al. (2015) insulation score, log2(score / mean). Each bin's
    score sums the contact band [i-delta, i) x [i, i+delta); NaN where empty."""
    coo = M_sparse.tocoo()
    dist = coo.col.astype(np.int64) - coo.row.astype(np.int64)
    band = (dist > 0) & (dist <= 2 * delta)
    j, k, v = coo.row[band].astype(np.int64), coo.col[band].astype(np.int64), coo.data[band]
    i_lo = np.maximum(j + 1, k - delta + 1)
    i_hi = np.minimum(j + delta, k)
    valid = (i_lo <= i_hi) & (i_lo < n_bins)
    i_lo, i_hi, v = i_lo[valid], np.minimum(i_hi[valid], n_bins - 1), v[valid]
    diff = np.zeros(n_bins + 1, dtype=np.float64)
    np.add.at(diff, i_lo, v)
    np.add.at(diff, i_hi + 1, -v)
    scores = np.cumsum(diff[:-1])
    pos = scores > 0
    if not pos.any():
        return np.full(n_bins, np.nan)
    out = np.full(n_bins, np.nan)
    out[pos] = np.log2(scores[pos] / scores[pos].mean())
    return out


def insulation_at_boundaries(insulation, bounds, lo, hi, n_perm=1000, seed=0):
    """Score a boundary set by insulation against a circular-shift null (mean
    insulation improves mechanically as boundaries thin out, so the raw mean
    is only comparable via this null's z-score)."""
    span = hi - lo
    scored = [b for b in bounds if lo <= b < hi and np.isfinite(insulation[b])]
    empty = dict(n_bounds=len(bounds), n_scored=0, mean=np.nan, median=np.nan,
                 null_mean=np.nan, null_sd=np.nan, z=np.nan, p=np.nan,
                 frac_local_min=np.nan)
    if not scored or span <= 1:
        return empty
    obs = float(np.mean([insulation[b] for b in scored]))
    barr = np.asarray(bounds)
    rng = np.random.default_rng(seed)
    null = []
    for shift in rng.integers(1, span, n_perm):
        shifted = (barr - lo + shift) % span + lo
        vals = insulation[shifted]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            null.append(vals.mean())
    if not null:
        return {**empty, "n_scored": len(scored), "mean": obs}
    null = np.asarray(null)
    sd = float(null.std())
    delta_local = 3
    local = [
        insulation[b] <= np.nanmin(insulation[max(b - delta_local, 0):b + delta_local + 1])
        for b in scored
    ]
    return dict(
        n_bounds=len(bounds), n_scored=len(scored), mean=obs,
        median=float(np.median([insulation[b] for b in scored])),
        null_mean=float(null.mean()), null_sd=sd,
        z=float((obs - null.mean()) / sd) if sd > 0 else np.nan,
        p=float((null <= obs).mean()), frac_local_min=float(np.mean(local)),
    )


def boundary_agreement(bounds_a, bounds_b, tol=3):
    """Symmetric agreement between two boundary sets, matched within tol bins."""
    n_a, n_b = len(bounds_a), len(bounds_b)
    if not n_a and not n_b:
        return dict(jaccard=1.0, frac_a_matched=1.0, frac_b_matched=1.0, f1=1.0)
    if not n_a or not n_b:
        return dict(jaccard=0.0, frac_a_matched=0.0, frac_b_matched=0.0, f1=0.0)
    a_arr, b_arr = np.array(sorted(bounds_a)), np.array(sorted(bounds_b))
    matched_a = int(np.sum([np.any(np.abs(b_arr - a) <= tol) for a in a_arr]))
    matched_b = int(np.sum([np.any(np.abs(a_arr - b) <= tol) for b in b_arr]))
    frac_a, frac_b = matched_a / n_a, matched_b / n_b
    union = matched_a + (n_a - matched_a) + (n_b - matched_b)
    return dict(
        jaccard=matched_a / union if union else 0.0,
        frac_a_matched=frac_a, frac_b_matched=frac_b,
        f1=2 * frac_a * frac_b / (frac_a + frac_b) if (frac_a + frac_b) else 0.0,
    )


def tad_boundaries(tads):
    """Internal boundary positions: every interval edge, minus the two outermost."""
    if not tads:
        return []
    edges = sorted({t[0] for t in tads} | {t[1] for t in tads})
    return edges[1:-1]


def tad_statistics(tads):
    bounds = tad_boundaries(tads)
    sizes_kbp = [(e - s) * BINSIZE / 1000.0 for s, e in tads]
    return dict(
        n_tads=len(tads), n_boundaries=len(bounds),
        median_kbp=float(np.median(sizes_kbp)) if sizes_kbp else 0.0,
        mean_kbp=float(np.mean(sizes_kbp)) if sizes_kbp else 0.0,
        boundaries=bounds,
    )


def score_region(tads, insulation, lo, hi, n_perm):
    stats = tad_statistics(tads)
    ins = insulation_at_boundaries(insulation, stats["boundaries"], lo, hi, n_perm)
    return {**stats, "insulation": ins}


def boundary_is_records(chrom, method, tads, insulation, lo, hi):
    rows = []
    for b in tad_boundaries(tads):
        if lo <= b < hi and np.isfinite(insulation[b]):
            rows.append(dict(chrom=str(chrom), method=method, boundary_bin=int(b),
                             insulation=float(insulation[b])))
    return rows


def write_bed(tads, chrom, out_bed, track_name="TAD", description=""):
    desc = description or f"TAD domains ({chrom})"
    with open(out_bed, "w") as f:
        f.write(f"track name='{track_name}' description='{desc}'\n")
        for i, (s, e) in enumerate(tads):
            f.write(f"{chrom}\t{s * BINSIZE}\t{e * BINSIZE}\tTAD_{i + 1}\n")
    print(f"Wrote {out_bed}  ({len(tads)} domains)", flush=True)
