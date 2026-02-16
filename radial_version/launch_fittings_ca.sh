#!/bin/bash
echo hi! we found the right bash script
conda run -n pk_env_38 python /arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/radial_version/run_fittings.py $1 $2 $3 $4 $5