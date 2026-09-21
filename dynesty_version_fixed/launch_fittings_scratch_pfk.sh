#!/bin/bash
# ==============================================================================
# Script Name: launch_fittings_scratch_pfk.sh
# Purpose:     Core execution script for ALMA visibility modeling using Dynesty.
#              Copies input scripts and visibility data to node-local scratch
#              storage to optimize I/O performance, executes the Bayesian
#              sampling and post-fit visualization in a dedicated Conda environment,
#              and automatically syncs all results back to persistent storage.
#
# Usage:
#   ./launch_fittings_scratch_pfk.sh <IN_DIR> <OUT_DIR> <DATA_PATH> <FITTYPE>
#
# Arguments:
#   $1 - IN_DIR:    Path to directory containing input Python scripts and modules.
#   $2 - OUT_DIR:   Persistent destination directory for output logs, models, and plots.
#   $3 - DATA_PATH: Absolute path to the calibrated UV visibility table (ASCII format).
#   $4 - FITTYPE:   Model identifier defining the geometry (e.g., 'twod_gaussring',
#                   'twod_gauss1blob', 'twod_gauss1blob_2peak', 'twod_gauss1blob_2peak_dp',
#                   'twod_gauss3blob', 'simgauss').
#
# Environment:
#   Requires the 'pk_env_38' Conda environment (Python <= 3.8) containing Galario,
#   Dynesty, CASA (casatasks/casatools), Astropy, and NumPy.
# ==============================================================================

echo CORE BASH SCRIPT

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
# Create a unique temporary directory on fast local scratch storage if not already
# defined by a parent wrapper script (such as launch_fittings_scratch_bshlog_pfk.sh).
if [ -z "${STAGING_DIR}" ]; then
    STAGING_DIR=$(mktemp -d "${SCRATCH_ROOT}/mcmc_${FITTYPE}_XXXXXX")
fi
echo "Created staging area: ${STAGING_DIR}"

# ------------------------------------------------------------------------------
# 2. Exit Trap & Cleanup Handler
# ------------------------------------------------------------------------------
# Function: cleanup
# Purpose:  Ensures that generated data products in scratch storage are copied back
#           to persistent storage (OUT_DIR) and temporary scratch directories are
#           deleted, even if the script terminates prematurely or encounters an error.
cleanup() {
    echo "Syncing results back to ${OUT_DIR}..."
    mkdir -p "${OUT_DIR}"
    cp -r "${STAGING_DIR}/output/"* "${OUT_DIR}/"
    
    echo "Cleaning up scratch..."
    rm -rf "${STAGING_DIR}"
}
# Register the cleanup function to trigger automatically upon shell exit (EXIT signal)
trap cleanup EXIT

# ------------------------------------------------------------------------------
# 3. Stage In: Data & Script Preparation
# ------------------------------------------------------------------------------
# Create output folder within scratch staging area and copy visibility data and code.
mkdir -p "${STAGING_DIR}/output"
cp "${DATA_PATH}" "${STAGING_DIR}/uvtable.txt"
cp -r "${IN_DIR}/"* "${STAGING_DIR}/"

# ------------------------------------------------------------------------------
# 4. Execution: Dynesty Nested Sampling & Visualization
# ------------------------------------------------------------------------------
# Change working directory into the scratch staging folder so that relative file paths
# (e.g. outputs, temporary CASA tables, logs) are confined to local scratch disk.
cd "${STAGING_DIR}"

pwd
ls
ls casa_dir_freq

echo "Starting run at $(date)"

# Step 4a: Run dynamic nested sampling with Dynesty to fit the model to UV visibilities
conda run -n pk_env_38 python agy_run_dynesty.py $4

# Step 4b: Run post-processing to generate diagnostic plots, CASA CLEAN images, and PDFs
conda run -n pk_env_38 python visualize_dynesty.py $4

# Capture the exit status of the Python execution pipeline
EXIT_CODE=$?

echo "Run finished with code ${EXIT_CODE} at $(date)"
exit $EXIT_CODE