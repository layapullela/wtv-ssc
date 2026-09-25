"""Sequential-subspace SBM data generator for the N=150/K=5,
sigma=0.5, independent_weight=0.25 config

Contiguous runs on K overlapping d-dimensional subspaces: coefficients walk
on the unit sphere (step std = walk_std), cluster noise sigma is added per
column, and columns are L2-normalized. Labels are the contiguous block ids.

Each cluster's subspace basis is (1 - independent_weight) * shared_basis +
independent_weight * an independent random basis, so independent_weight=0.25
means each basis is 0.75 shared / 0.25 independent (NOT "0.25 shared" --
"independent_weight" is the weight on the INDEPENDENT draw, confirmed from
icassp_ssc_tv2/final_code/SBM_experiment/sbm.py::_cluster_bases, which this
was ported from as "mix" -- renamed here since "mix" read ambiguously as
either the shared or the independent fraction).
"""
from __future__ import annotations

import numpy as np

D = 10          # ambient dimension
D_SUB = 4       # subspace dimension
BLOCK_SIZES = [15, 20, 30, 40, 45]   # K=5 contiguous blocks, N=150
K = len(BLOCK_SIZES)
N = sum(BLOCK_SIZES)
SIGMA = 0.5     # cluster noise
INDEPENDENT_WEIGHT = 0.25   # weight on the independent draw (0 -> identical
                            # subspaces / fully shared; 1 -> fully independent)
OUTLIER_FRAC = 0.0
OUTLIER_BURST = 5
TEST_SEEDS = range(20, 40)   # 20 test sequences


def _orthonormalize(M):
    Q, _ = np.linalg.qr(M)
    return Q[:, :M.shape[1]]


def _unit(v):
    nrm = np.linalg.norm(v)
    return v if nrm < 1e-12 else v / nrm


def _cluster_bases(k, d, dim, rng, independent_weight):
    """k orthonormal bases sharing a common component.

    Each basis is (1 - independent_weight) * shared + independent_weight *
    an independent draw. independent_weight=0 -> identical subspaces (fully
    shared); independent_weight=1 -> fully independent subspaces.
    """
    shared = rng.standard_normal((d, dim))
    return [
        _orthonormalize((1.0 - independent_weight) * shared
                         + independent_weight * rng.standard_normal((d, dim)))
        for _ in range(k)
    ]


def _burst_outliers(Y, rng, frac, burst_len):
    """Replace non-overlapping bursts of columns by isotropic Gaussian."""
    d, n = Y.shape
    n_out = int(round(float(frac) * n))
    if n_out <= 0:
        return
    burst_len = max(1, min(int(burst_len), n_out, n))
    n_bursts = max(1, n_out // burst_len)
    used = np.zeros(n, dtype=bool)
    for _ in range(n_bursts):
        starts = [s for s in range(n - burst_len + 1) if not used[s:s + burst_len].any()]
        if not starts:
            break
        s = int(rng.choice(starts))
        Y[:, s:s + burst_len] = rng.standard_normal((d, burst_len))
        used[s:s + burst_len] = True


def make_sequential_instance(seed, walk_std, sigma=SIGMA, block_sizes=None,
                              dim=D, d_sub=D_SUB,
                              independent_weight=INDEPENDENT_WEIGHT,
                              outlier_frac=OUTLIER_FRAC,
                              burst_len=OUTLIER_BURST):
    """One (Y, labels) sequential-subspace instance.

    coef vectors are unit norm. Noise is scaled so
    E[||noise||^2] = sigma^2 when ||U a|| = 1. Columns of Y
    are L2-normalized. Labels are the contiguous block ids.
    """
    if block_sizes is None:
        block_sizes = BLOCK_SIZES
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(block_sizes)), block_sizes)
    n = int(labels.size)
    k = len(block_sizes)
    Us = _cluster_bases(k, dim, d_sub, rng, independent_weight)
    Y = np.empty((dim, n))
    noise_scale = float(sigma) / np.sqrt(dim)
    step_scale = float(walk_std) / np.sqrt(d_sub)
    t = 0
    for ki, size in enumerate(block_sizes):
        a = _unit(rng.standard_normal(d_sub))
        for _ in range(int(size)):
            Y[:, t] = Us[ki] @ a + noise_scale * rng.standard_normal(dim)
            a = _unit(a + step_scale * rng.standard_normal(d_sub))
            t += 1
    _burst_outliers(Y, rng, outlier_frac, burst_len)
    col_norms = np.linalg.norm(Y, axis=0, keepdims=True)
    col_norms[col_norms < 1e-12] = 1.0
    Y = Y / col_norms
    return Y, labels


def test_matrices(walk_std, seeds=TEST_SEEDS):
    """The 20 held-out test sequences for one walk_std, K=5 known."""
    return [make_sequential_instance(seed, walk_std) for seed in seeds]
