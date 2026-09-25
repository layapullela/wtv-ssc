"""Reproduce SBM experiment rerunning
OSC, TKSS, and WTV-SSC (hyperparameters given in the .csv)

panel="walk"      -- OSC/TKSS/WTV-SSC, tuned lambda_1/lambda_2(/block_size
                      or d/lam/s) per walk_std in {0, 0.05, ..., 0.30}.
panel="blocksize" -- WTV-SSC only, fixed lambda_1=0.1/lambda_2_row_ref=1,
                      block_size swept, walk_std in {0, 0.10, 0.20, 0.30}.

Data: N=150, K=5 contiguous blocks [15,20,30,40,45], sigma=0.5,
independent_weight=0.25 (each cluster's subspace basis is 0.75 shared +
0.25 an independent draw), 20 held-out test sequences per walk_std
(seeds 20-39), shared across both panels since only walk_std varies the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import reproduce  # noqa: E402
from sbm_data import test_matrices  # noqa: E402

HYPERPARAMS_CSV = HERE / "hyperparameters_sbm.csv"


def main():
    reproduce(HYPERPARAMS_CSV, test_matrices)


if __name__ == "__main__":
    main()
