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

BSH_CMD="/arc/home/pknowlton/git_repo/ALMA-Galaxy-Visibility-Modelling-PFK/dynesty_version/launch_fittings_scratch_pfk.sh"

PSR_DIR="/arc/home/pknowlton/dynest_product_dir/resource_usage"

mkdir -p "${PSR_DIR}"

txtfile="${PSR_DIR}/${NAME}_psrecord.txt"
plotfile="${PSR_DIR}/${NAME}_psrecord.png"

psrecord "${BSH_CMD} ${IN_DIR} ${OUT_DIR} ${DATA_PATH} ${FITTYPE}" \
  --log "$txtfile" \
  --plot "$plotfile" \
  --include-children \
  --interval 60

# Copy psrecord outputs to OUT_DIR as well
if [ -n "${OUT_DIR}" ]; then
    mkdir -p "${OUT_DIR}"
    [ -f "$txtfile" ] && cp "$txtfile" "${OUT_DIR}/"
    [ -f "$plotfile" ] && cp "$plotfile" "${OUT_DIR}/"
fi