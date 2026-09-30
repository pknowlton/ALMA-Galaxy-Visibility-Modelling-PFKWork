#!/bin/bash
# ==============================================================================
# Script Name: launch_fittings_scratch_pfk.sh
# Purpose:     Core execution script for ALMA visibility modeling using ADVI.
#              Copies input scripts and visibility data to node-local scratch
#              storage to optimize I/O performance, executes Automatic
#              Differentiation Variational Inference (ADVI) and post-fit
#              visualization in a dedicated Conda environment, and automatically
#              syncs all results back to persistent storage.
#
# Usage:
#   ./launch_fittings_scratch_pfk.sh <IN_DIR> <OUT_DIR> <DATA_PATH> <FITTYPE>
#
# Arguments:
#   $1 - IN_DIR:    Path to directory containing input Python scripts and modules.
#   $2 - OUT_DIR:   Persistent destination directory for output logs, checkpoints, and plots.
#   $3 - DATA_PATH: Absolute path to the calibrated UV visibility table (ASCII format).
#   $4 - FITTYPE:   Model identifier ('twod_gaussring' or 'twod_gauss1blob').
#
# Environment:
#   Requires the 'pk_env_38' Conda environment (Python <= 3.8) containing Galario,
#   CASA (casatasks/casatools), Astropy, and NumPy.
# ==============================================================================

echo CORE ADVI BASH SCRIPT

# Parse positional input arguments
IN_DIR=$1
echo "Input Directory:  $IN_DIR"
OUT_DIR=$2
echo "Output Directory: $OUT_DIR"
DATA_PATH=$3
echo "Data Path:        $DATA_PATH"
FITTYPE=$4
echo "Fit Type:         $FITTYPE"

SCRATCH_ROOT="/scratch"

# ------------------------------------------------------------------------------
# 1. Workspace Initialization
# ------------------------------------------------------------------------------
if [ -z "${STAGING_DIR}" ]; then
    STAGING_DIR=$(mktemp -d "${SCRATCH_ROOT}/advi_${FITTYPE}_XXXXXX")
fi
echo "Created staging area: ${STAGING_DIR}"

# ------------------------------------------------------------------------------
# 2. Exit Trap & Cleanup Handler
# ------------------------------------------------------------------------------
cleanup() {
    echo "Syncing results back to ${OUT_DIR}..."
    mkdir -p "${OUT_DIR}"
    cp -r "${STAGING_DIR}/output/"* "${OUT_DIR}/"
    
    echo "Cleaning up scratch..."
    rm -rf "${STAGING_DIR}"
}
trap cleanup EXIT

# ------------------------------------------------------------------------------
# 3. Stage In: Data & Script Preparation
# ------------------------------------------------------------------------------
mkdir -p "${STAGING_DIR}/output"
cp "${DATA_PATH}" "${STAGING_DIR}/uvtable.txt"
cp -r "${IN_DIR}/"* "${STAGING_DIR}/"

# ------------------------------------------------------------------------------
# 4. Execution: ADVI Variational Optimization & Post-Processing
# ------------------------------------------------------------------------------
cd "${STAGING_DIR}"

pwd
ls
ls casa_dir_freq

echo "Starting ADVI run at $(date)"

# Step 4a: Run ADVI optimization to fit the model to UV visibilities
conda run -n pk_env_38 python run_advi.py $4

# Step 4b: Run post-processing to generate diagnostic plots and PDFs
conda run -n pk_env_38 python visualize_advi.py $4

EXIT_CODE=$?

echo "Run finished with code ${EXIT_CODE} at $(date)"
exit $EXIT_CODE
