#!/usr/bin/env python3
""" we test effect of  WTV regularizer on vs. off experiment 
for SSC on HIC041, with SpectralTAD as a reference, inference only.
This experiment is to ablate whether window tv regularizer specifically helps
SSC for TAD detection task.

Boundaries are called on chr2-10 downsampled to p, insulation score on full depth.
Every method uses the same 2 Mb scan. DP Ncuts w/ silhouette score {2,3,4}

  WTV-SSC   column-normalized SSC + sliding-window TV (lambda_2 > 0, b >= 2)
  SSC-noTV  the same solver with lambda_2 = 0
  SpectralTAD  Python SpectralTAD as reference

  
hyperparms tuned on chr1; lambda_1 grid
{0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45}; WTV-SSC also searched
lambda_2 {0.1, 0.2, 0.5, 1, 2} and b {2, 3, 5, 8, 12}.

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from common import (MIN_TAD_BINS, WINDOW, boundary_is_records, compute_insulation_score,
                    load_chrom, run_block_tv_tad, run_spectral_tad)

HIC041 = "/nfs/turbo/umms-minjilab/lpullela/hic_data/HIC041.hic"
RESOLUTION = 10_000
TEST_CHROMS = [str(i) for i in range(2, 11)]
FRACS = [1.0, 0.75, 0.5, 0.25]
SEEDS = list(range(11))
INSULATION_DELTA = 25
OUT = Path(__file__).resolve().parent / "results" / "regularizer"

SOLVER = dict(max_iter=100, mu_max=10.0, norm_mode="spectral")
FROZEN = {
    1.0:  {"WTV-SSC": dict(block_size=8, lambda_1=0.2, lambda_2=0.2), "SSC-noTV": dict(lambda_1=0.075)},
    0.75: {"WTV-SSC": dict(block_size=12, lambda_1=0.15, lambda_2=0.2), "SSC-noTV": dict(lambda_1=0.01)},
    0.5:  {"WTV-SSC": dict(block_size=12, lambda_1=0.2, lambda_2=0.2), "SSC-noTV": dict(lambda_1=0.075)},
    0.25: {"WTV-SSC": dict(block_size=5, lambda_1=0.15, lambda_2=0.2), "SSC-noTV": dict(lambda_1=0.1)},
}
METHODS = ["WTV-SSC", "SSC-noTV", "SpectralTAD"]


def call_tads(method, M, n, frac):
    if method == "SpectralTAD":
        return run_spectral_tad(M, n, 0, n, WINDOW, MIN_TAD_BINS, verbose=False)
    cfg = FROZEN[frac][method]
    kw = dict(block_size=cfg.get("block_size", 2), lambda_1=cfg["lambda_1"],
              lambda_2=cfg.get("lambda_2", 0.0), **SOLVER)
    return run_block_tv_tad(M, n, 0, n, WINDOW, MIN_TAD_BINS, kw, 2, 4, "argmax",
                            verbose=False, col_norm=True)


def run_unit(frac, seed):
    """Pooled boundary insulation for each method on chr2-10 at (p, seed)."""
    scores = {m: [] for m in METHODS}
    for chrom in TEST_CHROMS:
        M, n = load_chrom(HIC041, chrom, RESOLUTION, downsample_frac=frac,
                          downsample_seed_=seed, tag="test")
        M_full, n_full = (M, n) if frac >= 1.0 else load_chrom(
            HIC041, chrom, RESOLUTION, downsample_frac=1.0, downsample_seed_=0, tag="test-full-depth")
        ins = compute_insulation_score(M_full, n_full, INSULATION_DELTA)
        for m in METHODS:
            recs = boundary_is_records(chrom, m, call_tads(m, M, n, frac), ins, 0, n)
            scores[m] += [r["insulation"] for r in recs]
            print(f"  chr{chrom} {m:12s} n={len(recs)}", flush=True)
        del M, M_full, ins
    out = dict(frac=frac, seed=seed, frozen=FROZEN[frac],
               median_IS={m: float(np.median(v)) for m, v in scores.items()},
               n_boundaries={m: len(v) for m, v in scores.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"p_{frac:g}_seed_{seed}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["median_IS"], indent=1), flush=True)


def summarize():
    lines = ["# HIC041: WTV regularizer ON vs OFF (column-normalized SSC)", "",
             "Median IS pooled over all chr2-10 boundaries (IS on the full-depth matrix). "
             "p=1 is a single run; p<1 is mean ± sd over downsampling seeds 0-10.", "",
             "| p | WTV-SSC median IS | SSC (no TV) median IS | SpectralTAD median IS | seeds WTV < no-TV | paired Wilcoxon p |",
             "|---|---|---|---|---|---|"]
    for frac in FRACS:
        runs = [json.loads(f.read_text()) for f in sorted(OUT.glob(f"p_{frac:g}_seed_*.json"))]
        runs = [r for r in runs if r["frozen"] == FROZEN[frac]]   # skip results from other configs
        if not runs:
            continue
        v = {m: np.array([r["median_IS"][m] for r in runs]) for m in METHODS}
        cell = (lambda x: f"{x.mean():.4f}") if len(runs) == 1 else (lambda x: f"{x.mean():.4f} ± {x.std(ddof=1):.4f}")
        d = v["WTV-SSC"] - v["SSC-noTV"]
        p = f"{wilcoxon(d).pvalue:.2g}" if len(d) > 5 else "n/a"
        lines.append(f"| {frac:.2f} | {cell(v['WTV-SSC'])} | {cell(v['SSC-noTV'])} | "
                     f"{cell(v['SpectralTAD'])} | {int((d < 0).sum())}/{len(d)} | {p} |")
    (OUT / "regularizer_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frac", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run_unit(a.frac, a.seed)
