#!/usr/bin/env python3
"""SSC-Sparse-Block-TV vs SpectralTAD: Hi-C TAD-boundary insulation, inference
only (no hyperparameter search -- see FROZEN HYPERPARAMETERS below).

Hi-C file
---------
HIC041.hic  (/nfs/turbo/umms-minjilab/lpullela/hic_data/HIC041.hic)
10 kb resolution, Knight-Ruiz balanced (KR).

Frozen hyperparameters
----------------------
Grid-searched ONCE on full-depth (p=1.0) chr1 by
icassp_ssc_tv2/final_code/hic_experiment/compare_chr1tune_resttest_dp_ncut.py
(minimize mean_IS - frac_local_min over block_size x lambda_1 x lambda_2);
see that run's
results/hic041_100win_chr1tune_chr2to10test_l1_100_dpncut_k4_downsample/
stoch_fullchr/p_1.0/tune_split.json -> "best". Reused UNCHANGED below across
every downsample fraction -- this script does no tuning of its own.

    block_size = 2
    lambda_1   = 0.001
    lambda_2   = 0.001
    max_iter   = 100
    mu_max     = 10_000.0
    norm_mode  = "spectral"   (D-normalization; the solver's default)

Cutter: DP-NCut, k in {2, 3, 4}, k_select="argmax" (dp_k_min=2, dp_k_max=4).
Sequential 2 Mb (WINDOW=200 bin) sliding-window scan, MIN_TAD_BINS=5,
insulation delta=25 bins, 1000 circular-shift permutations per boundary set.

What this runs
--------------
Test set: chr2-10 (same file/replicate as chr1). For each downsample fraction
p in {1.0, 0.75, 0.5, 0.25} (p=1.0 is "the test experiment"; p<1 is "the
downsampling" sweep), each test chromosome is binomial-thinned to p (observed
NONE counts) then KR-balanced, boundaries are called with the frozen
hyperparameters above, and insulation is scored against that SAME (matched-
depth) matrix -- exactly the original run's protocol. (A separate follow-up
in icassp_ssc_tv2 scores against the full-depth matrix instead, to check
whether apparent insulation changes with p are real or a downsampling
artifact in the metric; not reproduced here.)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import (
    BINSIZE,
    MIN_TAD_BINS,
    WINDOW,
    boundary_agreement,
    boundary_is_records,
    compute_insulation_score,
    load_chrom,
    run_block_tv_tad,
    run_spectral_tad,
    score_region,
    write_bed,
)

HIC041 = "/nfs/turbo/umms-minjilab/lpullela/hic_data/HIC041.hic"

# ── Frozen hyperparameters (see module docstring) ───────────────────────────
SOLVER_KWARGS = dict(
    block_size=2, lambda_1=0.001, lambda_2=0.001,
    max_iter=100, mu_max=10_000.0, norm_mode="spectral",
)
DP_K_MIN, DP_K_MAX, DP_K_SELECT = 2, 4, "argmax"
INSULATION_DELTA = 25
N_PERM = 1000


def _finite(x):
    return x is not None and x == x


def _mw_u(a, b):
    from scipy.stats import mannwhitneyu
    r = mannwhitneyu(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                     alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def _fmt_pvalue(pval, floor=1e-100):
    if not _finite(pval):
        return "nan"
    return f"<{floor:g}" if pval < floor else f"{pval:.3g}"


def plot_ssc_vs_st_violin(ssc, st, title, out_png, pvalue=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    parts = ax.violinplot([ssc, st], positions=[1, 2], showmedians=True, widths=0.7,
                          points=200)
    for body, c in zip(parts["bodies"], ("#c0392b", "#2c5aa0")):
        body.set_facecolor(c)
        body.set_alpha(0.75)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("0.2")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"SSC\n(n={len(ssc)}, mean={np.mean(ssc):.3f})",
        f"SpectralTAD\n(n={len(st)}, mean={np.mean(st):.3f})",
    ])
    ax.set_ylabel("log2 insulation score")
    if pvalue is not None and pvalue == pvalue:
        title = f"{title}\nMann-Whitney p={_fmt_pvalue(pvalue)}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_png}", flush=True)


def run_one_fraction(args, frac, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_chrom = []
    test_is_records = []
    print(f"\n=== p={frac:g}  test chroms {args.test_chroms} ===", flush=True)
    print(f"  {'chrom':>5} {'SSC_n':>6} {'ST_n':>6} {'SSC_z':>8} {'ST_z':>8} "
          f"{'SSC_IS':>8} {'ST_IS':>8} {'Jac':>7}")
    for chrom in args.test_chroms:
        M, n = load_chrom(args.hic, chrom, args.resolution,
                          downsample_frac=frac, downsample_seed_=args.downsample_seed,
                          tag="test")
        lo, hi = 0, n
        tv_tads = run_block_tv_tad(M, n, lo, hi, WINDOW, MIN_TAD_BINS, SOLVER_KWARGS,
                                   DP_K_MIN, DP_K_MAX, DP_K_SELECT, verbose=False)
        spec_tads = run_spectral_tad(M, n, lo, hi, WINDOW, MIN_TAD_BINS, verbose=False)
        ins = compute_insulation_score(M, n, INSULATION_DELTA)
        tv_sc = score_region(tv_tads, ins, lo, hi, N_PERM)
        st_sc = score_region(spec_tads, ins, lo, hi, N_PERM)
        agr = boundary_agreement(tv_sc["boundaries"], st_sc["boundaries"])
        tv_recs = boundary_is_records(chrom, "SSC", tv_tads, ins, lo, hi)
        st_recs = boundary_is_records(chrom, "SpectralTAD", spec_tads, ins, lo, hi)
        test_is_records.extend(tv_recs)
        test_is_records.extend(st_recs)
        row = dict(
            chrom=chrom, n_bins=n,
            tv_n_tads=tv_sc["n_tads"], spec_n_tads=st_sc["n_tads"],
            tv_z=tv_sc["insulation"]["z"], spec_z=st_sc["insulation"]["z"],
            tv_mean_is=tv_sc["insulation"]["mean"], spec_mean_is=st_sc["insulation"]["mean"],
            tv_frac=tv_sc["insulation"]["frac_local_min"],
            spec_frac=st_sc["insulation"]["frac_local_min"], jaccard=agr["jaccard"],
        )
        per_chrom.append(row)
        print(f"  {chrom:>5} {row['tv_n_tads']:6d} {row['spec_n_tads']:6d} "
              f"{row['tv_z']:8.2f} {row['spec_z']:8.2f} "
              f"{row['tv_mean_is']:8.3f} {row['spec_mean_is']:8.3f} "
              f"{row['jaccard']:7.3f}", flush=True)
        write_bed(tv_tads, chrom, str(out_dir / f"ssc_dpncut_chr{chrom}.bed"),
                  "SSC_DPNCut", f"bs={SOLVER_KWARGS['block_size']} k<={DP_K_MAX}")
        write_bed(spec_tads, chrom, str(out_dir / f"spectral_tad_chr{chrom}.bed"),
                  "SpectralTAD", "KR")
        del M, ins, tv_tads, spec_tads

    def mean_key(key):
        vals = [r[key] for r in per_chrom if r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"  unweighted mean over {len(per_chrom)} chroms")
    print(f"    SSC+DP-NCut z={mean_key('tv_z'):.2f}  mean_IS={mean_key('tv_mean_is'):.3f}")
    print(f"    SpectralTAD  z={mean_key('spec_z'):.2f}  mean_IS={mean_key('spec_mean_is'):.3f}")

    ssc_scores = [r["insulation"] for r in test_is_records if r["method"] == "SSC"]
    st_scores = [r["insulation"] for r in test_is_records if r["method"] == "SpectralTAD"]
    mw_u, mw_p = _mw_u(ssc_scores, st_scores) if ssc_scores and st_scores else (float("nan"),) * 2

    csv_path = out_dir / "test_boundary_insulation.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["chrom", "method", "boundary_bin", "insulation"])
        w.writeheader()
        w.writerows(test_is_records)
    print(f"Wrote {csv_path}", flush=True)

    summary = dict(
        hic=args.hic, downsample_frac=frac, downsample_seed=args.downsample_seed,
        test_chroms=list(args.test_chroms), solver_kwargs=SOLVER_KWARGS,
        dp_k_min=DP_K_MIN, dp_k_max=DP_K_MAX, dp_k_select=DP_K_SELECT,
        per_chrom=per_chrom, mean_tv_z=mean_key("tv_z"), mean_spec_z=mean_key("spec_z"),
        mean_tv_is=mean_key("tv_mean_is"), mean_spec_is=mean_key("spec_mean_is"),
        mean_jaccard=mean_key("jaccard"),
        test_insulation_n_ssc=len(ssc_scores), test_insulation_n_st=len(st_scores),
        test_insulation_mean_ssc=float(np.mean(ssc_scores)) if ssc_scores else float("nan"),
        test_insulation_median_ssc=float(np.median(ssc_scores)) if ssc_scores else float("nan"),
        test_insulation_mean_st=float(np.mean(st_scores)) if st_scores else float("nan"),
        test_insulation_median_st=float(np.median(st_scores)) if st_scores else float("nan"),
        test_insulation_mw_u=mw_u, test_insulation_pvalue=mw_p,
    )
    out_json = out_dir / "test_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {out_json}", flush=True)

    plot_ssc_vs_st_violin(
        ssc_scores, st_scores,
        f"Test-set boundary insulation (chr {args.test_chroms}) p={frac:g}",
        str(out_dir / "test_insulation_violin.png"), pvalue=mw_p)
    return summary


def plot_across_fractions(summaries, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fracs = [s["downsample_frac"] for s in summaries]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot(fracs, [s["test_insulation_median_ssc"] for s in summaries], "-o",
            color="#c0392b", label="SSC median IS")
    ax.plot(fracs, [s["test_insulation_median_st"] for s in summaries], "-o",
            color="#2c5aa0", label="SpectralTAD median IS")
    ax.set_xlabel("downsample fraction p")
    ax.set_ylabel("median log2 insulation score")
    ax.set_title("Test-set insulation vs downsampling (matched-depth scoring)")
    ax.legend(fontsize=9)
    ax.invert_xaxis()
    fig.tight_layout()
    out_png = out_dir / "insulation_vs_downsample.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_png}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hic", default=HIC041)
    p.add_argument("--test-chroms", nargs="+", default=[str(i) for i in range(2, 11)])
    p.add_argument("--resolution", type=int, default=10_000)
    p.add_argument("--downsample-fracs", type=float, nargs="+",
                   default=[1.0, 0.75, 0.5, 0.25])
    p.add_argument("--downsample-seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = p.parse_args()

    print(f"HIC {args.hic}  10 kb  observed (NONE) -> binomial -> Knight-Ruiz")
    print(f"SSC solver_kwargs: {SOLVER_KWARGS}")
    print(f"Test chr {args.test_chroms}  downsample fracs {args.downsample_fracs}")

    out_root = Path(args.out_dir)
    summaries = []
    for frac in args.downsample_fracs:
        out_dir = out_root / f"p_{frac:g}"
        summaries.append(run_one_fraction(args, frac, out_dir))

    if len(summaries) > 1:
        plot_across_fractions(summaries, out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "all_fractions_summary.json").write_text(
        json.dumps(summaries, indent=2, default=float))
    print(f"Wrote {out_root / 'all_fractions_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
