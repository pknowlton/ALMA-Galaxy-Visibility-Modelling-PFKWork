#!/bin/bash

echo BASHLOG BASH SCRIPT

IN_DIR=$1
echo "$IN_DIR"
OUT_DIR=$2
echo "$OUT_DIR"
NAME=$3
echo "$NAME"
DATA_PATH=$4
echo "$DATA_PATH"
FITTYPE=$5
echo "$FITTYPE"

PSR_CMD="/arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dynesty_version/launch_fittings_scratch_psrec_pfk.sh"

SCRATCH_ROOT="/scratch"

# 1. Create unique workspace & staging output directory first so logging writes directly to STAGING_DIR/output
export STAGING_DIR=$(mktemp -d "${SCRATCH_ROOT}/mcmc_${FITTYPE}_XXXXXX")
mkdir -p "${STAGING_DIR}/output"
echo "Staging area created for logging: ${STAGING_DIR}"

# 2. Run psrecord bash script, capturing stdout and stderr to bashlog.txt inside STAGING_DIR/output
"${PSR_CMD}" "${IN_DIR}" "${OUT_DIR}" "${NAME}" "${DATA_PATH}" "${FITTYPE}" 2>&1 | tee "${STAGING_DIR}/output/bashlog.txt"
EXIT_CODE=${PIPESTATUS[0]}

exit ${EXIT_CODE}
