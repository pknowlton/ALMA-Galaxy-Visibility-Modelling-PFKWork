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
#   $4 - FITTYPE:   Model identifier (see input_dir/model_defs.py for the full list,
#                   e.g. 'twod_gaussring', 'twod_gauss3blob_nonuni', 'resid_gauss3blob').
#
# CASA folder:
#   The folder of CASA products used by the visualiser is derived from the data
#   file name (everything after 'uvtable_'), and must exist inside IN_DIR:
#       M95_C5+C2_cont93_uvtable_freq.txt           ->  casa_dir_freq
#       M95_C5+C2_cont93_uvtable_freq_residiii.txt  ->  casa_dir_freq_residiii
#
# Exit code: the first failure from run_dynesty.py / visualize_dynesty.py (0 if both succeed).
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

# CASA folder for this data table: 'casa_dir_' + whatever follows 'uvtable_' in the file name
#   M95_C5+C2_cont93_uvtable_freq_residiii.txt -> casa_dir_freq_residiii
DATA_STEM=$(basename "${DATA_PATH%.*}")
CASA_DIR="casa_dir_${DATA_STEM#*uvtable_}"
echo "CASA folder:      $CASA_DIR"

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
ls "${CASA_DIR}"

echo "Starting run at $(date)"

# Step 4a: Run dynamic nested sampling with Dynesty to fit the model to UV visibilities
conda run -n pk_env_38 python run_dynesty.py "${FITTYPE}"
RUN_CODE=$?

# Step 4b: Run post-processing to generate diagnostic plots, CASA CLEAN images, and PDFs
# (still runs if 4a failed, so a partial checkpoint can be inspected)
conda run -n pk_env_38 python visualize_dynesty.py "${FITTYPE}" --casa-dir "${CASA_DIR}"
VIS_CODE=$?

# Report the first failure (previously only the visualiser's status was reported)
EXIT_CODE=${RUN_CODE}
if [ "${EXIT_CODE}" -eq 0 ]; then
    EXIT_CODE=${VIS_CODE}
fi

echo "Run finished with code ${EXIT_CODE} at $(date)"
exit $EXIT_CODE