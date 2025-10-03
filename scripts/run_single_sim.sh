#!/bin/bash

USAGE="Usage: bash scripts/run_single_sim.sh SIM_CONFIG MAP_FILE AGENT_NUM AGENT_NUM_STEP_SIZE N_EVALS MODE N_SIM N_WORKERS DOMAIN SIM_ALGO -r RELOAD -p PROJECT_DIR"

SIM_CONFIG="$1"
MAP_FILE="$2"
CHUTE_MAPPING_FILE="$3"
AGENT_NUM="$4"
AGENT_NUM_STEP_SIZE="$5"
N_EVALS="$6"
MODE="$7"
N_SIM="$8"
N_WORKERS="$9"
DOMAIN="${10}"
SIM_ALGO="${11}"

shift 11
while getopts "p:r:m:t:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      r) RELOAD=$OPTARG;;
      m) G_POLICY_FILE=$OPTARG;;
      t) TASK_ASSIGN_FILE=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi

G_POLICY_ARG=""
if [ -n "$G_POLICY_FILE" ]; then
  G_POLICY_ARG="--model_param_file ${G_POLICY_FILE}"
fi

TASK_ASSIGN_ARG=""
if [ -n "$TASK_ASSIGN_FILE" ]; then
  TASK_ASSIGN_ARG="--task_assign_file ${TASK_ASSIGN_FILE}"
fi


if [ "${DOMAIN}" = "kiva" ]
then
    if [ "${MODE}" = "inc_agents" ]
    then
        singularity exec ${SINGULARITY_OPTS} --cleanenv singularity/ubuntu_warehouse.sif \
            python env_search/warehouse/module.py \
                --warehouse-config "$SIM_CONFIG" \
                --map-filepath "$MAP_FILE" \
                --chute_mapping_file "$CHUTE_MAPPING_FILE" \
                --simulation_algo "$SIM_ALGO" \
                --num-agent "$AGENT_NUM" \
                --num-agent-step-size "$AGENT_NUM_STEP_SIZE" \
                --n_evals "$N_EVALS" \
                --mode "$MODE" \
                --n_sim "$N_SIM" \
                --n_workers "$N_WORKERS" \
                --reload "$RELOAD" \
                ${G_POLICY_ARG} \
                ${TASK_ASSIGN_ARG}
        sleep 2
    fi

    if [ "${MODE}" = "constant" ]
    then
        singularity exec ${SINGULARITY_OPTS} --cleanenv singularity/ubuntu_warehouse.sif \
        python env_search/warehouse/module.py \
            --warehouse-config "$SIM_CONFIG" \
            --map-filepath "$MAP_FILE" \
            --simulation_algo "$SIM_ALGO" \
            --num-agent "$AGENT_NUM" \
            --n_evals "$N_EVALS" \
            --mode "$MODE" \
            --n_workers "$N_WORKERS"
    fi
fi

if [ "${DOMAIN}" = "manufacture" ]
then
    if [ "${MODE}" = "inc_agents" ]
    then
        singularity exec ${SINGULARITY_OPTS} --cleanenv singularity/ubuntu_warehouse.sif \
            python env_search/manufacture/module.py \
                --manufacture-config "$SIM_CONFIG" \
                --map-filepath "$MAP_FILE" \
                --num-agent "$AGENT_NUM" \
                --num-agent-step-size "$AGENT_NUM_STEP_SIZE" \
                --n_evals "$N_EVALS" \
                --mode "$MODE" \
                --n_sim "$N_SIM" \
                --n_workers "$N_WORKERS" \
                --reload "$RELOAD"
        sleep 2
    fi

    if [ "${MODE}" = "constant" ]
    then
        singularity exec ${SINGULARITY_OPTS} --cleanenv singularity/ubuntu_warehouse.sif \
        python env_search/manufacture/module.py \
            --manufacture-config "$SIM_CONFIG" \
            --map-filepath "$MAP_FILE" \
            --num-agent "$AGENT_NUM" \
            --n_evals "$N_EVALS" \
            --mode "$MODE" \
            --n_workers "$N_WORKERS"
    fi
fi

