#!/bin/bash

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
echo Set OMP_NUM_THREADS to one

echo The headless sessionn has launched, current flex cpu test!
conda run -n pk_env_38 python /arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/general_pfk_version/run_mcmc_fit_pfk.py -op /arc/home/pknowlton/uv_product_dir_new/test_cpu/gaussing-2d-16c-pfk twodgaussring -fp /arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/general_pfk_version