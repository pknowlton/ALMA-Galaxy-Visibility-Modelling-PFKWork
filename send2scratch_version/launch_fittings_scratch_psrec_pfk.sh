#!/bin/bash

echo PSRECORD BASH SCRIPT

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

BSH_CMD="/arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/send2scratch_version/launch_fittings_scratch_pfk.sh"

PSR_DIR="/arc/home/pknowlton/uv_product_dir_new/test_scratch/resource_usage"

txtfile="${PSR_DIR}/${NAME}_${FITTYPE}_psrecord.txt"
plotfile="${PSR_DIR}/${NAME}_${FITTYPE}_psrecord.png"

psrecord "${BSH_CMD} ${IN_DIR} ${OUT_DIR} ${DATA_PATH} ${FITTYPE}" \
  --log "$txtfile" \
  --plot "$plotfile" \
  --include-children \
  --interval 60