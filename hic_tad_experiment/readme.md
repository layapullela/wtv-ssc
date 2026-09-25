HIC TAD-boundary experiment code (inference only, no hyperparameter search)

Data: HIC041.hic (/nfs/turbo/umms-minjilab/lpullela/hic_data/HIC041.hic),
10 kb resolution. Every chromosome is loaded as raw observed (NONE) counts
and Knight-Ruiz balanced by this package itself (never the .hic file's own
pre-balanced "KR" vectors) -- matching the original run, which always passes
--observed-kr, even at p=1.0. Test set is chr2-10; boundaries are called by
SSC-Sparse-Block-TV (DP-NCut cut, k in {2,3,4}) and by a Python SpectralTAD
reference (unit-circle embedding), then both are scored by Crane et al.
insulation at their called boundaries against a circular-shift null.

Frozen hyperparameters (`run_hic_tad.py` top of file): block_size=2,
lambda_1=0.001, lambda_2=0.001, max_iter=100, mu_max=10000, norm_mode=spectral.
Grid-searched once on full-depth chr1 by the original (much larger) experiment
folder, `icassp_ssc_tv2/final_code/hic_experiment/`
(`compare_chr1tune_resttest_dp_ncut.py`); see that script's
`results/hic041_100win_chr1tune_chr2to10test_l1_100_dpncut_k4_downsample/
stoch_fullchr/p_1.0/tune_split.json` -> `"best"`. This package does not
re-tune -- it reuses that one frozen set unchanged across every downsample
fraction below.

`run_hic_tad.py` runs both the base test (p=1.0) and the downsampling sweep
(p=0.75, 0.5, 0.25) in one job: each test chromosome is binomial-thinned to p
(applied to observed/NONE counts) then KR-balanced, boundaries are called at
the frozen hyperparameters, and insulation is scored against that same
matched-depth matrix (the original run's protocol). Per-p outputs land in
`results/p_<frac>/` (BED files, boundary-insulation CSV/JSON, SSC-vs-
SpectralTAD violin); `results/insulation_vs_downsample.png` and
`all_fractions_summary.json` summarize across p.

`common.py` is the shared, self-contained infrastructure (`.hic` loading,
Knight-Ruiz balancing, binomial downsampling, the DP-NCut and SpectralTAD
cutters, the sliding-window scan, insulation scoring), ported and trimmed
from the original messy `icassp_ssc_tv2/final_code/hic_experiment/` +
`559/559_spectralTAD/TV_algorithm/tad_common.py`. The SSC solver itself is
`../solvers/wtv_ssc.py` (shared with `sbm_experiments/` and
`video_experiments/`), not duplicated here.

Not reproduced here (see the original folder for these): the noise-
robustness variants (gaussian/poisson), the HIC042 replicate comparison,
block-size sweeps, the Col21 and row-norm solver variants, the full-depth-
scoring diagnostic, and chr18/19 experiments.

Run: `sbatch run_hic_tad.sbatch` (env: conda `ssc_559`).
