import time
import random
import logging
import pathlib

import traceback
import numpy as np

from typing import List
from env_search.warehouse.warehouse_result import WarehouseResult
from env_search.utils.worker_state import get_warehouse_module
from env_search.warehouse.module import WarehouseModule

logger = logging.getLogger(__name__)


def gen_chute_mapping_from_capacity(
    chute_locs: np.ndarray,
    chute_capacities: np.ndarray,
    destination_volumes: np.ndarray,
    repair_seed: int,
):
    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(repair_seed)
    random.seed(repair_seed)

    warehouse_module = get_warehouse_module()
    chute_mapping = warehouse_module.gen_chute_mapping_from_capacity(
        chute_locs=chute_locs,
        chute_capacities=chute_capacities,
        destination_volumes=destination_volumes,
        repair_seed=repair_seed,
    )
    logger.info("gen_chute_mapping_from_capacity done after %f sec",
                time.time() - start)

    return chute_mapping


def repair_chute_mapping(
    chute_locs: np.ndarray,
    unrepaired_chute_mapping: dict,
    destination_volumes: np.ndarray,
    repair_seed: int,
):
    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(repair_seed)
    random.seed(repair_seed)

    warehouse_module = get_warehouse_module()
    chute_mapping = warehouse_module.repair_chute_mapping(
        chute_locs=chute_locs,
        unrepaired_chute_mapping=unrepaired_chute_mapping,
        destination_volumes=destination_volumes,
        repair_seed=repair_seed,
    )
    logger.info("repair_chute_mapping done after %f sec", time.time() - start)

    return chute_mapping


def repair_warehouse(
    unrepaired_env_int: np.ndarray,
    parent_repaired_env: np.ndarray,
    repair_seed: int,
    w_mode: bool,
    min_n_shelf: int,
    max_n_shelf: int,
    partial_repair: bool,
):
    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(repair_seed)
    random.seed(repair_seed)

    logger.info("repair warehouse with seed %d", repair_seed)
    warehouse_module = get_warehouse_module()

    try:
        if partial_repair:
            repaired_env_int = warehouse_module.partial_repair(
                unrepaired_env_int=unrepaired_env_int,
                repair_seed=repair_seed,
                min_n_shelf=min_n_shelf,
                max_n_shelf=max_n_shelf,
            )
            map_json = None  # Doesn't matter
        else:
            (
                map_json,
                unrepaired_env_int,
                repaired_env_int,
            ) = warehouse_module.repair(
                unrepaired_env_int=unrepaired_env_int,
                parent_repaired_env=parent_repaired_env,
                repair_seed=repair_seed,
                w_mode=w_mode,
                min_n_shelf=min_n_shelf,
                max_n_shelf=max_n_shelf,
            )
    except TimeoutError as e:
        logger.warning(f"repair failed")
        logger.info(f"The map was {map}")
        (
            map_json,
            unrepaired_env_int,
            repaired_env_int,
        ) = [None] * 3

    logger.info("repair_warehouse done after %f sec", time.time() - start)

    return map_json, unrepaired_env_int, repaired_env_int


def run_warehouse(
    map_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    map_id: int,
    eval_id: int,
    simulation_algo: str,
    model_params: np.ndarray = None,
    chute_mapping_json: str = None,
    package_dist_weight_json: str = None,
    task_assignment_params_json: str = None,
) -> WarehouseResult:
    """
    Repair map and run simulation

    Args:
        map (np.ndarray): input map in integer format
        eval_logdir (str): log dir of simulation
        sim_seed (int): random seed for simulation. Should be different for
                        each solution
        agentNum (int): number of drives
        map_id (int): id of the current map to be evaluated. The id
                      is only unique to each batch, NOT to the all the
                      solutions. The id can make sure that each simulation
                      gets a different log directory.
        eval_id (int): id of evaluation
    """

    if map_json is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result

    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run warehouse with seed %d", sim_seed)
    warehouse_module = get_warehouse_module()

    output_dir = str(eval_logdir /
                     f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

    try:
        if simulation_algo == "RHCR":
            result = warehouse_module.evaluate_rhcr(
                map_json=map_json,
                output_dir=output_dir,
                sim_seed=sim_seed,
            )
        elif simulation_algo == "PIBT":
            result = warehouse_module.evaluate_pibt(
                map_json=map_json,
                output_dir=output_dir,
                sim_seed=sim_seed,
                model_params=model_params,
                chute_mapping_json=chute_mapping_json,
                package_dist_weight_json=package_dist_weight_json,
                task_assignment_params_json=task_assignment_params_json,
            )
    except TimeoutError as e:
        layout = map_json["layout"]
        logger.warning(f"evaluate failed")
        logger.info(f"The map was {layout}")
        result = {"success": False}

    logger.info("run_warehouse done after %f sec", time.time() - start)

    return result


def run_warehouse_iterative_update(
    env_np: np.ndarray,
    map_json_str: str,
    n_valid_edges: int,
    n_valid_vertices: int,
    model_params: np.ndarray,
    output_dir: pathlib.Path,
    seed: int,
    chute_mapping_json: str = None,
    task_assignment_params_json: str = None,
):
    if map_json_str is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result, None

    start = time.time()

    warehouse_module = get_warehouse_module()

    result, all_weights = warehouse_module.evaluate_iterative_update(
        env_np,
        map_json_str,
        model_params,
        n_valid_edges,
        n_valid_vertices,
        str(output_dir),
        seed,
        chute_mapping_json=chute_mapping_json,
        task_assignment_params_json=task_assignment_params_json,
    )
    result["success"] = True

    logger.info("run_warehouse_iterative_update done after %f sec",
                time.time() - start)

    return result, all_weights


def process_warehouse_eval_result(
    curr_result_json: List[dict],
    n_evals: int,
    unrepaired_env_int: np.ndarray,
    repaired_env_int: np.ndarray,
    edge_weights: np.ndarray,
    wait_costs: np.ndarray,
    w_mode: bool,
    max_n_shelf: int,
    map_id: int,
    simulation_algo: str,
    chute_mapping: dict = None,
    chute_capacities: np.ndarray = None,
):
    start = time.time()

    warehouse_module = get_warehouse_module()

    if simulation_algo == "RHCR":
        results = warehouse_module.process_eval_result_rhcr(
            curr_result_json=curr_result_json,
            n_evals=n_evals,
            unrepaired_env_int=unrepaired_env_int,
            repaired_env_int=repaired_env_int,
            edge_weights=edge_weights,
            wait_costs=wait_costs,
            w_mode=w_mode,
            max_n_shelf=max_n_shelf,
            map_id=map_id,
        )
    elif simulation_algo == "PIBT":
        results = warehouse_module.process_eval_result_pibt(
            curr_result_json=curr_result_json,
            n_evals=n_evals,
            unrepaired_env_int=unrepaired_env_int,
            repaired_env_int=repaired_env_int,
            edge_weights=edge_weights,
            wait_costs=wait_costs,
            w_mode=w_mode,
            max_n_shelf=max_n_shelf,
            map_id=map_id,
            chute_mapping=chute_mapping,
            chute_capacities=chute_capacities,
        )
    logger.info("process_warehouse_eval_result done after %f sec",
                time.time() - start)

    return results
