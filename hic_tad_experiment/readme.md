HIC TAD-boundary experiment code inference notes:

Data: HIC041.hic (/nfs/turbo/umms-minjilab/lpullela/hic_data/HIC041.hic),
10 kb resolution. Test set is chr2-10; boundaries are called by 
DP-NCut cut, k in {2,3,4} for WTV-SSC and by a Python SpectralTAD
reference which uses unit-circle embed. Then we score by Crane et al.
insulation.

Frozen hyperparams: block_size=2,lambda_1=0.001, lambda_2=0.001, max_iter=100, mu_max=10000
Grid search on chromosome 1, test on 2 - 10.

We do binomial downsampling for the sequencing depth experiments, then do KR-norm on
each downsampled matrix. Boundaries called at frozen hyperparameters, and insulation 
score is against the original p = 1 matrix.



## Regularizer ON vs OFF (column-normalized SSC): `run_regularizer_table.py`

This is inference only: the chosen hyperparameters below are frozen in `FROZEN` at the top of the script, so no tuning
is needed to reproduce the table. Data: `wtv-ssc/archived_wtv_ssc/hic_data/HIC041.hic`.

Methods (all use the same 2 Mb scan and DP-NCut with k in {2,3,4} per window, chosen by silhouette):
- WTV-SSC: column-normalized SSC + sliding-window TV (lambda_2 > 0, b >= 2).
- SSC-noTV: the same solver with lambda_2 = 0.
- SpectralTAD: Python SpectralTAD reference (no hyperparams).
- DP-NCut: DP-NCut without SSC on the column-norm window, W = |Yn| + |Yn^T| (no hyperparams).

### Frozen hyperparameters (selected on chr1-3)

| p | WTV-SSC: b | WTV-SSC: lambda_1 | WTV-SSC: lambda_2 | SSC-noTV: lambda_1 (lambda_2 = 0) |
|---|---|---|---|---|
| 1.00 | 12* | 0.1  | 0.2 | 0.02  |
| 0.75 | 8   | 0.15 | 0.5 | 0.02  |
| 0.50 | 12* | 0.15 | 0.2 | 0.035 |
| 0.25 | 12* | 0.15 | 0.2 | 0.1   |

\* selected value is on the edge of its search grid (largest b).

Settings (shared at each depth p):
- solver: 200-bin windows, KR balanced, diagonal zeroed; columns are unit-normalized so the hyerparams are
  consistent across regions; `ssc_admm_sparse_block_tv` with max_iter=100, mu_max=10, D normalized by its spectral norm.
- DP-NCut with k in {2,3,4} per window (chosen by best silhouette score on the raw contacts), min TAD size 5 bins.
- Crane insulation with delta=25 bins, always computed on the full-depth (p=1) matrix. Median IS is pooled over all
  chr4-10 boundaries.
- data is HIC041.hic at 10 kb, binomially downsampled to p (seed s), then KR balanced.

### How the hyperparameters were selected (tuning is not needed to reproduce)

For each depth p, both SSC arms were tuned by grid search on chr1, chr2 and chr3, downsampled to p (downsample seed 0).
Each tuning chromosome contributes its first 8,309 bins (one third of chr1's 24,926 bins). Selection minimized median IS pooled
over the boundaries, with IS computed on the matched-depth matrices.
- The lambda_1 grid was shared by both arms: {0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45}.
- WTV-SSC also searched lambda_2 in {0.1, 0.2, 0.5, 1, 2} and b in {2, 3, 5, 8, 12} (250 configs). b >= 1. chosen (tuned) b usually around 12.
- SSC-noTV fixes lambda_2 = 0 (10 configs).

Best config per arm and depth was frozen and tested on chr4-10 (never used in tuning) at the same p, over
downsampling seeds 0-10 (p=1 is a single run/seed, since there is no downsampling). Tuning code saved here:
(`archived_wtv_ssc/_sandbox_hic_multichr_tune/`).

# new result
Tuned on chr1-3, tested on chr4-10. Median IS pooled over all chr4-10 boundaries (IS on the full-depth matrix); p<1 is
mean ± sd over downsampling seeds 0-10. Seeds TV better = seeds where WTV-SSC < SSC without TV (paired Wilcoxon
p = 0.00098 at every p<1).

|p|WTV-SSC|SSC without TV|SpectralTAD|DP-NCut (col-norm)|Seeds TV better|
|---|---|---|---|---|---|
|1.00|**−0.3770**|−0.3271|−0.1766|−0.3761|single run|
|0.75|**−0.3801 ± 0.0041**|−0.3262 ± 0.0038|−0.1722 ± 0.0051|−0.3693 ± 0.0032|11/11|
|0.50|**−0.3689 ± 0.0051**|−0.3260 ± 0.0053|−0.1607 ± 0.0073|−0.3584 ± 0.0044|11/11|
|0.25|**−0.3461 ± 0.0033**|−0.3184 ± 0.0034|−0.1514 ± 0.0084|−0.3411 ± 0.0045|11/11|

WTV-SSC vs DP-NCut, paired by seed: WTV-SSC lower in 11/11 (p=0.75), 10/11 (p=0.5) and 10/11 (p=0.25) seeds.
