#!/bin/bash
# ==============================================================================
# Script Name: launch_fittings_scratch_bshlog_pfk.sh
# Purpose:     Logging wrapper for the ALMA visibility modeling workflow.
#              Pre-allocates the scratch staging area so that console output
#              (stdout and stderr) from the entire pipeline is captured and
#              streamed live to a text logfile (bashlog.txt) in addition to
#              the terminal console.
#
# Usage:
#   ./launch_fittings_scratch_bshlog_pfk.sh <IN_DIR> <OUT_DIR> <NAME> <DATA_PATH> <FITTYPE>
#
# Arguments:
#   $1 - IN_DIR:    Path to directory containing Python scripts and models.
#   $2 - OUT_DIR:   Target persistent directory for all run outputs.
#   $3 - NAME:      Identifier/label for the fitting run (used in filenames).
#   $4 - DATA_PATH: Absolute path to the calibrated UV visibility table.
#   $5 - FITTYPE:   Model geometry identifier (e.g., 'twod_gaussring').
#
# Pipeline Architecture:
#   launch_headless_scratch.py (CANFAR session)
#     -> launch_fittings_scratch_bshlog_pfk.sh (this script: console log capture)
#       -> launch_fittings_scratch_psrec_pfk.sh (resource profiling)
#         -> launch_fittings_scratch_pfk.sh (core execution & scratch staging)
# ==============================================================================

echo BASHLOG BASH SCRIPT

# Parse positional arguments
IN_DIR=$1
echo "Input Directory:  $IN_DIR"
OUT_DIR=$2
echo "Output Directory: $OUT_DIR"
NAME=$3
echo "Run Name:         $NAME"
DATA_PATH=$4
echo "Data Path:        $DATA_PATH"
FITTYPE=$5
echo "Fit Type:         $FITTYPE"

# Path to the next stage in the pipeline (resource usage profiler)
PSR_CMD="/arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dynesty_version/launch_fittings_scratch_psrec_pfk.sh"

SCRATCH_ROOT="/scratch"

# ------------------------------------------------------------------------------
# 1. Pre-allocate Staging Area
# ------------------------------------------------------------------------------
# Create the scratch directory in advance and export STAGING_DIR so downstream
# scripts share the exact same staging location.
export STAGING_DIR=$(mktemp -d "${SCRATCH_ROOT}/mcmc_${FITTYPE}_XXXXXX")
mkdir -p "${STAGING_DIR}/output"
echo "Staging area created for logging: ${STAGING_DIR}"

# ------------------------------------------------------------------------------
# 2. Execute Downstream Profiler with Output Logging
# ------------------------------------------------------------------------------
# Run psrecord script. 
# '2>&1' redirects stderr to stdout.
# 'tee' writes the combined stream to both standard output and bashlog.txt.
# PIPESTATUS[0] captures the exit code of PSR_CMD rather than the exit code of tee.
"${PSR_CMD}" "${IN_DIR}" "${OUT_DIR}" "${NAME}" "${DATA_PATH}" "${FITTYPE}" 2>&1 | tee "${STAGING_DIR}/output/bashlog.txt"
EXIT_CODE=${PIPESTATUS[0]}

exit ${EXIT_CODE}
