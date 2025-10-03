#!/bin/bash

USAGE="Usage: bash scripts/inspect_elite.sh MODE LOGDIR"

MODE="$1"
LOGDIR="$2"

shift 2
while getopts "p:s:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      s) SURROGATE=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done


if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi


if [ -z "${MODE}" ]
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
if [ -z "$SURROGATE" ]; then
  SURROGATE="True"
fi

singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/inspect_elite.py \
        "$MODE" \
        --logdir "$LOGDIR" \
        --surrogate "$SURROGATE"