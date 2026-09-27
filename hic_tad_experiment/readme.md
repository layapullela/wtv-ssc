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

Inference only, the chosen hyperparameters for this experiment are given 
in the comment above.

### Frozen hyperparameters (selected on chr1)

| p | WTV-SSC: b | WTV-SSC: lambda_1 | WTV-SSC: lambda_2 | SSC-noTV: lambda_1 (lambda_2 = 0) |
|---|---|---|---|---|
| 1.00 | 8  | 0.2  | 0.5 | 0.01*  |
| 0.75 | 12* | 0.15 | 0.2 | 0.02  |
| 0.50 | 12* | 0.15 | 0.2 | 0.035 |
| 0.25 | 3  | 0.15 | 0.5 | 0.1   |

\* selected value is on the edge of its search grid (smallest lambda_1 / largest b).


Settings (Shared at each depth p value):
- solver: 200bin windows, kr balanced. columns are unit normalized to keep hyperparams consistent across regions 
- DP ncuts with k in {2,3,4} per window (chosen by best silhouette score)
- Crane insulation with delta=25 bins, always computed on the full-depth (p=1) matrix. Median IS is pooled over all
- data is HiC041.hic, downsampled with binomial, then kr balanced

### How the hyperparameters were selected

For each depth p, both SSC arms were tuned on chr1 downsampled to p (downsample seed 0) by grid search. Selection minimized
the median IS of the called boundaries, with IS computed on the matched-depth chr1.
- The lambda_1 grid was shared by both arms: {0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45}.
- WTV-SSC also searched lambda_2 in {0.1, 0.2, 0.5, 1, 2} and b in {2, 3, 5, 8, 12} (250 configs). b = 1 was
  excluded, so WTV-SSC always uses a window.
- SSC-noTV fixes lambda_2 = 0 (10 configs).

The best config per arm and depth was frozen and tested on chr2-10 at the same p, over downsampling seeds 0-10
(p=1 is a single run, since there is no downsampling). The tuning code lives outside this repo (sandbox).



# new result
Median IS pooled over all chr2-10 boundaries (IS on the full-depth matrix); p<1 is mean ± sd over downsampling seeds 0-10.
Seeds TV better = seeds where WTV-SSC < SSC without TV (paired Wilcoxon p = 0.00098 at every p<1).

|p|WTV-SSC|SSC without TV|SpectralTAD|Seeds TV better|
|---|---|---|---|---|
|1.00|**−0.3979**|−0.3343|−0.1913|single run|
|0.75|**−0.3909 ± 0.0027**|−0.3403 ± 0.0035|−0.1860 ± 0.0047|11/11|
|0.50|**−0.3839 ± 0.0032**|−0.3413 ± 0.0043|−0.1764 ± 0.0069|11/11|
|0.25|**−0.3412 ± 0.0049**|−0.3321 ± 0.0040|−0.1656 ± 0.0066|11/11|
