HIC TAD-boundary experiment code inference notes:

Data: HIC041.hic (/nfs/turbo/umms-minjilab/lpullela/hic_data/HIC041.hic),
10 kb resolution. Test set is chr2-10; boundaries are called by 
DP-NCut cut, k in {2,3,4} for WTV-SSC and by a Python SpectralTAD
reference (unit-circle embedding). Then we score by Crane et al.
insulation at their called boundaries against a circular-shift null.

Frozen hyperparams: block_size=2,lambda_1=0.001, lambda_2=0.001, max_iter=100, mu_max=10000
Grid search on chromosome 1, test on 2 - 10.

We do binomial downsampling for the sequencing depth experiments, then do KR-norm on
each downsampled matrix. Boundaries called at frozen hyperparameters, and insulation 
score is against the original p = 1 matrix.


