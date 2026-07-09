#!/bin/bash

echo CORE BASH SCRIPT

IN_DIR=$1
echo "$IN_DIR"
OUT_DIR=$2
echo "$OUT_DIR"
DATA_PATH=$3
echo "$DATA_PATH"
FITTYPE=$4
echo "$FITTYPE"

SCRATCH_ROOT="/scratch"

# 1. Create a unique workspace
STAGING_DIR=$(mktemp -d "${SCRATCH_ROOT}/mcmc_${FITTYPE}_XXXXXX")
echo "Created staging area: ${STAGING_DIR}"

# 2. Setup the "Exit Trap" 
# This runs automatically when the script finishes or is killed
cleanup() {
    echo "Syncing results back to ${OUT_DIR}..."
    mkdir -p "${OUT_DIR}"
    cp -r "${STAGING_DIR}/output/"* "${OUT_DIR}/"
    
    echo "Cleaning up scratch..."
    rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT

# 3. Stage In: Copy data and scripts to scratch
mkdir -p "${STAGING_DIR}/output"
cp "${DATA_PATH}" "${STAGING_DIR}/uvtable.txt"
cp -r "${IN_DIR}/"* "${STAGING_DIR}/"

# 4. Run the MCMC
# We 'cd' into staging so all relative file paths in your Python script 
# (like logs or plots) happen locally in scratch.
cd "${STAGING_DIR}"

pwd
ls
ls casa_dir

echo "Starting MCMC at $(date)"
conda run -n pk_env_38 python run_mcmc_fit_scratch_pfk_radex.py $4
conda run -n pk_env_38 python visualize_fit_final_radex.py $4
# The exit code of the python script is captured here
EXIT_CODE=$?

echo "MCMC finished with code ${EXIT_CODE} at $(date)"
exit $EXIT_CODE