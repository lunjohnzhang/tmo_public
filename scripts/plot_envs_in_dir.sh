#!/bin/bash

TO_PLOT="$1"
DOMAIN="$2"

mkdir -p "${TO_PLOT}/img"

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
for ENV in ${TO_PLOT}/*.json;
do
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/visualize_env.py \
            --map-filepath "${ENV}" \
            --store_dir "${TO_PLOT}/img" \
            --domain "${DOMAIN}"
done