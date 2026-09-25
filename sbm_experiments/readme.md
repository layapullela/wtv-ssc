SBM experiment code
Searched hyperparameters reported in: `hyperparameters_sbm.csv`

Data: N=150, K=5 contiguous blocks [15,20,30,40,45], D=10, subspace dim 4,
sigma=0.5 (cluster noise), independent_weight=0.25 (each cluster's subspace
basis is 0.75 shared + 0.25 an independent draw), no burst outliers.
See `sbm_data.py`. 20 held-out test sequences per walk_std (seeds 20-39),
shared across both panels since only walk_std varies the data.

`hyperparameters_sbm.csv` has two panels (see its `panel` column):
- `walk`: OSC/TKSS/WTV-SSC, tuned lambda_1/lambda_2(/block_size or d/lam/s)
  per walk_std in {0, 0.05, ..., 0.30} (left plot panel).
- `blocksize`: WTV-SSC only, FIXED lambda_1=0.1/lambda_2_row_ref=1 (not
  tuned), block_size swept, walk_std in {0, 0.10, 0.20, 0.30} (right plot
  panel).


