#!/bin/bash

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
echo Set OMP_NUM_THREADS to one

echo The headless sessionn has launched, beginning the MCMC_v3 oned and twod runs!
conda run -n pk_env_38 python /arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/2d_blob_version/run_mcmc_fit_pfk.py $1 $2 $3 $4 $5