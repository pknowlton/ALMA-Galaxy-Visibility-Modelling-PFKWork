#!/bin/bash
echo The headless sessionn has launched, beginning the MCMC run!
conda run -n pk_env_38 python /arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/radial_version/run_mcmc_fit_gaussring_dev.py $1 $2 $3