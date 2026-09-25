# same common as in video experiments
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solvers"))
from osc import cluster_from_Z, osc_exact                          # noqa: E402
from tkss import tkss_cluster                                      # noqa: E402
from wtv_ssc import sliding_block_diff_matrix, ssc_admm_sparse_block_tv  # noqa: E402

SEED = 0   # TKSS's random_state


# ari metric (same implementation as video_experiments/common.py, so this
# package has no sklearn dependency)
def adjusted_rand_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rows, cols = np.unique(y_true), np.unique(y_pred)
    table = np.zeros((rows.size, cols.size), dtype=np.int64)
    r_idx = {v: i for i, v in enumerate(rows)}
    c_idx = {v: i for i, v in enumerate(cols)}
    for t, p in zip(y_true, y_pred):
        table[r_idx[int(t)], c_idx[int(p)]] += 1
    n = table.sum()
    if n < 2:
        return 1.0

    def _comb2(x):
        x = np.asarray(x, dtype=np.float64)
        return (x * (x - 1.0) / 2.0).sum()

    sum_ij, sum_i, sum_j = _comb2(table), _comb2(table.sum(1)), _comb2(table.sum(0))
    total = n * (n - 1.0) / 2.0
    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


# test methods, one call per method, matching the hyperparameter names in
# hyperparameters_sbm.csv
def run_osc(Y, k, lambda_1, lambda_2, max_iter):
    Z = osc_exact(Y, lambda_1, lambda_2, max_iter=int(max_iter))
    return cluster_from_Z(Z, k=k)


def run_tkss(Y, k, d, lam, s):
    pred, _ = tkss_cluster(Y, k=k, d=int(d), lam=lam, s=int(s), max_iter=30,
                            random_state=SEED)
    return pred


def run_wtv_ssc(Y, k, lambda_1, lambda_2_row_ref, block_size, max_iter):
    """lambda_2 is derived from lambda_2_row_ref per matrix so that
    block_size and TV strength don't confound each other (see readme).

    For panel="walk" rows the original grid search tuned raw lambda_2
    directly (no rescaling) -- those rows' lambda_2_row_ref was
    back-computed as raw_lambda_2 / f_scale so this single call convention
    reproduces both panels identically (f_scale * lambda_2_row_ref ==
    raw_lambda_2 exactly).
    """
    N = Y.shape[1]
    D_raw, _centers, _k_eff = sliding_block_diff_matrix(N, int(block_size),
                                                          normalize=False)
    f_scale = np.linalg.norm(D_raw, 2) / np.linalg.norm(D_raw[len(D_raw) // 2])
    lambda_2 = f_scale * float(lambda_2_row_ref)
    Z, _C, _info = ssc_admm_sparse_block_tv(
        Y, lambda_1=lambda_1, lambda_2=lambda_2, block_size=int(block_size),
        max_iter=int(max_iter))
    return cluster_from_Z(Z, k=k)


def run_row(row, test_mats):
    """Mean test ARI for one hyperparameters_sbm.csv row's method + hyperparams."""
    method, k, max_iter = row["method"], int(row["k"]), int(row["max_iter"])
    aris = []
    for Y, labels in test_mats:
        if method == "OSC":
            pred = run_osc(Y, k, float(row["lambda_1"]), float(row["lambda_2"]),
                            max_iter)
        elif method == "TKSS":
            pred = run_tkss(Y, k, int(row["d"]), float(row["lam"]), int(row["s"]))
        elif method == "WTV-SSC":
            pred = run_wtv_ssc(Y, k, float(row["lambda_1"]),
                                float(row["lambda_2_row_ref"]),
                                int(row["block_size"]), max_iter)
        else:
            raise ValueError(f"unknown method: {method!r}")
        aris.append(adjusted_rand_score(labels, pred))
    return float(np.mean(aris))


def reproduce(csv_path, test_matrices_fn):
    """test_matrices_fn(walk_std) -> list of (Y, labels) test matrices."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    cache = {}
    by_panel = defaultdict(list)
    print(f"{'panel':10} {'method':10} {'walk_std':>8}  "
          f"{'reproduced':>10} {'recorded':>10}  diff", flush=True)
    results = []
    for row in rows:
        walk_std = float(row["walk_std"])
        if walk_std not in cache:
            cache[walk_std] = test_matrices_fn(walk_std)
        reproduced = run_row(row, cache[walk_std])
        recorded = float(row["test_ari"])
        diff = reproduced - recorded
        print(f"{row['panel']:10} {row['method']:10} {walk_std:>8.2f}  "
              f"{reproduced:>10.4f} {recorded:>10.4f}  {diff:+.4f}", flush=True)
        by_panel[row["panel"]].append(diff)
        results.append({**row, "reproduced_test_ari": reproduced})

    print("\n=== summary ===", flush=True)
    for panel, diffs in by_panel.items():
        diffs = np.asarray(diffs)
        print(f"{panel:10} n={len(diffs):3d}  mean|diff|={np.abs(diffs).mean():.4f}  "
              f"max|diff|={np.abs(diffs).max():.4f}", flush=True)
    return results
