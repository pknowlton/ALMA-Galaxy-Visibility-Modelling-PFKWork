#!/bin/bash
# ==============================================================================
# Script Name: launch_fittings_scratch_psrec_pfk.sh
# Purpose:     System resource profiling wrapper for the visibility fitting workflow.
#              Uses `psrecord` to monitor and record CPU activity and RAM consumption
#              over time for the execution of launch_fittings_scratch_pfk.sh (and its
#              child processes), saving both a time-series text log and a PNG plot.
#
# Usage:
#   ./launch_fittings_scratch_psrec_pfk.sh <IN_DIR> <OUT_DIR> <NAME> <DATA_PATH> <FITTYPE>
#
# Arguments:
#   $1 - IN_DIR:    Path to directory containing input Python scripts and modules.
#   $2 - OUT_DIR:   Persistent destination directory for all run outputs.
#   $3 - NAME:      Run name/label used for tagging resource profiling output files.
#   $4 - DATA_PATH: Absolute path to the calibrated UV visibility table.
#   $5 - FITTYPE:   Model geometry identifier (e.g., 'twod_gaussring').
#
# Output Products:
#   - ${NAME}_psrecord.txt: Tabular time series of timestamp, CPU %, and memory (MB).
#   - ${NAME}_psrecord.png: Line plot visualizing CPU and RAM usage over run duration.
# ==============================================================================

echo PSRECORD BASH SCRIPT

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

# Path to the core execution script being profiled
BSH_CMD="/arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dynesty_version/launch_fittings_scratch_pfk.sh"

# Directory dedicated to accumulating resource monitoring records across runs
PSR_DIR="/arc/home/pknowlton/dynest_product_dir/resource_usage"

mkdir -p "${PSR_DIR}"

txtfile="${PSR_DIR}/${NAME}_psrecord.txt"
plotfile="${PSR_DIR}/${NAME}_psrecord.png"

# ------------------------------------------------------------------------------
# Execute Core Script Under psrecord Profiling
# ------------------------------------------------------------------------------
# --log:              File to store CPU and memory numbers sampled at each interval
# --plot:             File to save generated graphical plot upon completion
# --include-children: Tracks resources of all subprocesses spawned by the shell (e.g. Python, CASA)
# --interval 60:      Sample CPU/RAM every 60 seconds
psrecord "${BSH_CMD} ${IN_DIR} ${OUT_DIR} ${DATA_PATH} ${FITTYPE}" \
  --log "$txtfile" \
  --plot "$plotfile" \
  --include-children \
  --interval 60

# ------------------------------------------------------------------------------
# Copy Resource Usage Reports to the Run's Output Directory
# ------------------------------------------------------------------------------
if [ -n "${OUT_DIR}" ]; then
    mkdir -p "${OUT_DIR}"
    [ -f "$txtfile" ] && cp "$txtfile" "${OUT_DIR}/"
    [ -f "$plotfile" ] && cp "$plotfile" "${OUT_DIR}/"
fi