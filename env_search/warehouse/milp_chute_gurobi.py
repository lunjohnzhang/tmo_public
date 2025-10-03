import os
import gc
import json
import time
import fire
import logging
import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB, Env, Var

from env_search.utils.logging import setup_logging
from env_search.warehouse import get_packages
from env_search.utils import (kiva_env_str2number, read_in_kiva_map,
                              sortation_env_str2number, read_in_sortation_map,
                              get_chute_loc, get_workstation_loc,
                              compute_dist_matrix)

logger = logging.getLogger(__name__)


def _gen_gurobi_model(n_threads, time_limit, seed, outputflag=0):
    ######### TODO: Delete when making code public #########
    options = {
        "WLSACCESSID": "c5d7d8b6-ad88-4f4e-b41f-ea3a62323682",
        "WLSSECRET": "f5a5dec7-2f4c-4f33-8dbf-adf2277a9999",
        "LICENSEID": 2559434,
        "OutputFlag": outputflag,
    }
    ######### End Delete when making code public #########
    env = Env(params=options)

    mdl = Model("chute_mapping", env=env)
    mdl.setParam('Threads', n_threads)
    mdl.setParam('TimeLimit', time_limit)
    if seed is not None:
        mdl.setParam('Seed', seed)
    return mdl, env


def define_variables(mdl: Model,
                     chute_locs,
                     n_destinations,
                     warm_start=False,
                     warm_start_sol=None):
    edges = []
    edge_by_chutes = {c_loc: [] for c_loc in chute_locs}
    edge_by_destinations = {j: [] for j in range(n_destinations)}
    for c_loc in chute_locs:
        for j in range(n_destinations):
            var_name = f"edge_c_{c_loc}_d_{j}"
            edge_var: Var = mdl.addVar(name=var_name, vtype=GRB.BINARY)
            if warm_start_sol is not None:
                edge_var.Start = warm_start_sol[var_name]
            edges.append(edge_var)
            edge_by_chutes[c_loc].append(edge_var)
            edge_by_destinations[j].append(edge_var)

    inv_dest_degs = None
    if not warm_start:
        # Auxiliary variables for degrees of the destinations nodes
        inv_dest_degs = []
        for j in range(n_destinations):
            inv_dest_degree = mdl.addVar(name=f"inv_dest_degree_{j}",
                                         vtype=GRB.CONTINUOUS)
            if warm_start_sol is not None:
                # Infer the value of inv_dest_degree from the warm start
                # solution
                start_val = 0
                for c_loc in chute_locs:
                    start_val += warm_start_sol[f"edge_c_{c_loc}_d_{j}"]
                inv_dest_degree.Start = start_val
                # print(f"start val of destination {j}", start_val)
            inv_dest_degs.append(inv_dest_degree)

    return edge_by_chutes, edge_by_destinations, edges, inv_dest_degs


def limit_chute_degrees(mdl: Model, chute_locs, edge_by_chutes):
    for c_loc in chute_locs:
        mdl.addConstr(sum(edge_by_chutes[c_loc]) == 1,
                      name=f"chute_{c_loc}_serves_1_dest")


def limit_destination_degrees(
    mdl: Model,
    n_destinations,
    edge_by_destinations,
    n_chutes,
    destination_volumes,
):
    for j in range(n_destinations):
        mdl.addConstr(sum(edge_by_destinations[j]) >= 1,
                      name=f"dest_{j}_served_by_1_chute")
        U_j = int(np.ceil(max(1, 1.5 * n_chutes * destination_volumes[j])))
        mdl.addConstr(sum(edge_by_destinations[j]) <= U_j,
                      name=f"dest_{j}_served_by_U_chutes")


def constrain_inv_dest_degs(mdl: Model, n_destinations, inv_dest_degs,
                            edge_by_destinations):
    for j in range(n_destinations):
        mdl.addConstr(
            inv_dest_degs[j] * sum(edge_by_destinations[j]) == 1,
            name=f"inv_dest_degs_{j}",
        )


def milp_chute_mapping_warm_up(
    env: Env,
    chute_locs,
    chute_capacities,
    destination_volumes,
    n_threads,
    seed,
    time_limit,
):
    mdl = Model("chute_mapping", env=env)
    mdl.setParam('Threads', n_threads)
    mdl.setParam('TimeLimit', time_limit)
    if seed is not None:
        mdl.setParam('Seed', seed)

    n_chutes = len(chute_locs)
    n_destinations = len(destination_volumes)

    # When generating the warm start solution, normalize chute capacities such
    # that the sum = n_chutes/n_destinations
    chute_capacities *= n_chutes / n_destinations

    (
        edge_by_chutes,
        edge_by_destinations,
        edges,
        inv_dest_degs,
    ) = define_variables(mdl, chute_locs, n_destinations, warm_start=True)

    # Constraints
    # 1. Each chute can serve exactly 1 destination
    limit_chute_degrees(mdl, chute_locs, edge_by_chutes)

    # 2. Each destination is served by at least 1 and at most U_j chutes
    limit_destination_degrees(mdl, n_destinations, edge_by_destinations,
                              n_chutes, destination_volumes)

    # Objective
    diff_volume = []
    for i, c_loc in enumerate(chute_locs):
        c_loc_volume = [
            destination_volumes[j] * edge_by_chutes[c_loc][j]
            for j in range(n_destinations)
        ]
        diff_volume.append(
            mdl.addVar(name=f"diff_volume_{i}", vtype=GRB.CONTINUOUS))
        mdl.addConstr(diff_volume[i] >= sum(c_loc_volume) -
                      chute_capacities[i])
        mdl.addConstr(
            diff_volume[i] >= -(sum(c_loc_volume) - chute_capacities[i]))

    mdl.setObjective(sum(diff_volume), GRB.MINIMIZE)

    # Solve
    mdl.optimize()

    # Check for any feasible solution if optimal solution is not found
    if mdl.status != GRB.OPTIMAL:
        if mdl.status == GRB.INF_OR_UNBD:
            logger.info("Warm up: Model is infeasible or unbounded.")
            return None
        else:
            logger.info(
                "Warm up: No optimal solution found. Returning the best found solution."
            )
    logger.info(f"Warm up: Objective value: {mdl.ObjVal}")
    logger.info(f"Warm up: Objective gap: {mdl.MIPGap}")
    chute_mapping = extract_solution(mdl, chute_locs, n_destinations)
    # print(f"warm up solution: {chute_mapping}")
    solution_vals = mdl.getVars()
    solution_vals = {str(v.varName): v.X for v in solution_vals}

    # Dispose model
    mdl.dispose()
    gc.collect()

    return solution_vals


def milp_chute_mapping(
    chute_locs,
    chute_capacities,
    destination_volumes,
    n_threads,
    seed,
    time_limit,
    warm_start=False,
):
    """MIQP program that generate the chute mapping given chute capacities and
    destination volumes.

    Args:
        chute_locs (list): list of locations of the chutes
        chute_capacities (list[float]): chute capacities
        destination_volumes (list[float]): destination volumes
        n_threads (int): number of threads for MILP
        seed (int): random seed
        time_limit (float): time limit in seconds

    Returns:
        chute_mapping: dict(int, int) where key is destination_id, value is
            chute_location
    """
    mdl, env = _gen_gurobi_model(n_threads, time_limit, seed)

    # Warm start
    if warm_start is not None:
        # Generate warm_start solution
        warm_start_sol = milp_chute_mapping_warm_up(
            env,
            chute_locs,
            chute_capacities,
            destination_volumes,
            n_threads,
            seed,
            time_limit,
        )

    n_chutes = len(chute_locs)
    n_destinations = len(destination_volumes)

    (
        edge_by_chutes,
        edge_by_destinations,
        edges,
        inv_dest_degs,
    ) = define_variables(
        mdl,
        chute_locs,
        n_destinations,
        warm_start=False,
        warm_start_sol=warm_start_sol,
    )

    # Constraints
    # 1. Each chute can serve exactly 1 destination
    limit_chute_degrees(mdl, chute_locs, edge_by_chutes)

    # 2. Each destination is served by at least 1 and at most U_j chutes
    limit_destination_degrees(mdl, n_destinations, edge_by_destinations,
                              n_chutes, destination_volumes)

    # 3. Inverse of the degree of the destination nodes
    constrain_inv_dest_degs(mdl, n_destinations, inv_dest_degs,
                            edge_by_destinations)

    # Objective
    diff_volume = []
    for i, c_loc in enumerate(chute_locs):
        c_loc_volume = [
            destination_volumes[j] * edge_by_chutes[c_loc][j] *
            inv_dest_degs[j] for j in range(n_destinations)
        ]
        diff_volume.append(
            mdl.addVar(name=f"diff_volume_{i}", vtype=GRB.CONTINUOUS))
        mdl.addConstr(diff_volume[i] >= sum(c_loc_volume) -
                      chute_capacities[i])
        mdl.addConstr(
            diff_volume[i] >= -(sum(c_loc_volume) - chute_capacities[i]))

    mdl.setObjective(sum(diff_volume), GRB.MINIMIZE)

    # Solve
    mdl.optimize()

    # Check for any feasible solution if optimal solution is not found
    if mdl.status != GRB.OPTIMAL:
        if mdl.status == GRB.INF_OR_UNBD:
            logger.info("Model is infeasible or unbounded.")
            return None
        else:
            logger.info(
                "No optimal solution found. Returning the best found solution."
            )
    logger.info(f"Objective value: {mdl.ObjVal}")
    logger.info(f"Objective gap: {mdl.MIPGap}")
    chute_mapping = extract_solution(mdl, chute_locs, n_destinations)

    # Dispose model and env
    mdl.dispose()
    env.dispose()
    gp.disposeDefaultEnv()
    gc.collect()
    return chute_mapping


def miqp_chute_mapping_no_agent_collision(
    chute_locs,
    workstation_locs,
    destination_volumes,
    dist_matrix,
    n_threads,
    seed,
    time_limit,
    warm_start=False,
    outputflag=1,
):
    """MIQP program that generate the chute mapping given (1) distance between
    every workstation and chute, (2) destination volumes.

    The objective is to minimize the expected distance travelled by the agents
    ignoring the collisions

    Args:
        chute_locs (list): list of locations of the chutes
        destination_volumes (list[float]): destination volumes
        dist_matrix (`list[dict]`): distance between every
            locations in the map. e.g. `dist_matrix[loc1][loc2]` is the
            distance between loc1 and loc2
        n_threads (int): number of threads for MILP
        seed (int): random seed
        time_limit (float): time limit in seconds

    Returns:
        chute_mapping: dict(int, int) where key is destination_id, value is
            chute_location
    """
    mdl, env = _gen_gurobi_model(n_threads, time_limit, seed, outputflag)

    # Warm start
    # if warm_start is not None:
    # # Generate warm_start solution
    # warm_start_sol = milp_chute_mapping_warm_up(
    #     env,
    #     chute_locs,
    #     chute_capacities,
    #     destination_volumes,
    #     n_threads,
    #     seed,
    #     time_limit,
    # )

    n_chutes = len(chute_locs)
    n_destinations = len(destination_volumes)
    n_workstations = len(workstation_locs)

    (
        edge_by_chutes,
        edge_by_destinations,
        edges,
        inv_dest_degs,
    ) = define_variables(
        mdl,
        chute_locs,
        n_destinations,
        warm_start=False,
        # warm_start_sol=warm_start_sol,
    )

    # Constraints
    # 1. Each chute can serve exactly 1 destination
    limit_chute_degrees(mdl, chute_locs, edge_by_chutes)

    # 2. Each destination is served by at least 1 and at most U_j chutes
    limit_destination_degrees(mdl, n_destinations, edge_by_destinations,
                              n_chutes, destination_volumes)

    # 3. Inverse of the degree of the destination nodes
    constrain_inv_dest_degs(mdl, n_destinations, inv_dest_degs,
                            edge_by_destinations)

    # Objective
    # Weighted distance for every destination from all workstations to the
    # chutes that are assigned to the destination.
    weighted_dists = []
    for j in range(n_destinations):
        dists_all_w_to_all_chutes = []
        for w_loc in workstation_locs:
            # Compute the average distance of one package of destination j from
            # workstation w to its assigned chutes
            dists_w_to_all_chutes = []
            for c_loc in chute_locs:
                dists_w_to_all_chutes.append(dist_matrix[w_loc][c_loc] *
                                             edge_by_chutes[c_loc][j])
            dists_all_w_to_all_chutes.append(inv_dest_degs[j] *
                                             sum(dists_w_to_all_chutes))
        weighted_dists.append(destination_volumes[j] *
                              sum(dists_all_w_to_all_chutes) / n_workstations)

    # Regularize with the sum of pairwise distance between any two chutes
    # mapped to destination
    pairwise_dist_chutes = []
    for j in range(n_destinations):
        per_dest_pairwise_dist = []
        for i1 in range(n_chutes):
            for i2 in range(i1 + 1, n_chutes):
                c_loc1 = chute_locs[i1]
                c_loc2 = chute_locs[i2]
                per_dest_pairwise_dist.append(edge_by_chutes[c_loc1][j] *
                                              edge_by_chutes[c_loc2][j] *
                                              dist_matrix[c_loc1][c_loc2])
        pairwise_dist_chutes.append(sum(per_dest_pairwise_dist))

    mdl.setObjective(
        sum(weighted_dists) - sum(pairwise_dist_chutes), GRB.MINIMIZE)

    # Solve
    mdl.optimize()

    # Check for any feasible solution if optimal solution is not found
    if mdl.status != GRB.OPTIMAL:
        if mdl.status == GRB.INF_OR_UNBD:
            logger.info("Model is infeasible or unbounded.")
            return None
        else:
            logger.info(
                "No optimal solution found. Returning the best found solution."
            )
    logger.info(f"Objective value: {mdl.ObjVal}")
    logger.info(f"Objective gap: {mdl.MIPGap}")
    chute_mapping = extract_solution(mdl, chute_locs, n_destinations)

    # Dispose model and env
    mdl.dispose()
    env.dispose()
    gp.disposeDefaultEnv()
    gc.collect()
    return chute_mapping


def extract_solution(mdl: Model, chute_locs, n_destinations):
    chute_mapping = {j: [] for j in range(n_destinations)}
    for j in range(n_destinations):
        for c_loc in chute_locs:
            # Check for binary variable
            if mdl.getVarByName(f"edge_c_{c_loc}_d_{j}").x >= 0.5:
                chute_mapping[j].append(int(c_loc))
    return chute_mapping


def test_chute_capacity_milp(map_filepath, n_threads=1, seed=0, time_limit=60):
    setup_logging(on_worker=False)
    np.random.seed(seed)
    env_str, _ = read_in_sortation_map(map_filepath)
    env_np = sortation_env_str2number(env_str)
    env_np = env_np.astype(np.uint8)

    # chute_locs = [0, 1, 2, 3]
    chute_locs = get_chute_loc(env_np)
    # n_chutes = len(chute_locs)
    n_chutes = len(chute_locs)
    n_destinations = 100
    logger.info(f"{n_chutes} chutes and {n_destinations} destinations")
    # chute_capacities = [0.4, 0.1, 0.4, 0.1]
    chute_capacities = np.random.rand(n_chutes)
    chute_capacities /= np.sum(chute_capacities)
    # chute_capacities *= n_chutes / n_destinations
    # destination_volumes = [0.8, 0.2]
    destination_volumes = np.random.rand(n_destinations)
    destination_volumes /= np.sum(destination_volumes)

    chute_mapping = milp_chute_mapping(
        chute_locs,
        chute_capacities,
        destination_volumes,
        n_threads,
        seed,
        time_limit,
        warm_start=True,
    )
    print(chute_mapping)


def test_no_agent_collision_miqp(
    map_filepath,
    package_mode="dist",
    package_dist_type="721",
    package_path=None,
    n_destinations=100,
    n_threads=1,
    seed=0,
    time_limit=60,
):
    # setup_logging(on_worker=False)
    np.random.seed(seed)
    env_str, _ = read_in_sortation_map(map_filepath)
    env_np = sortation_env_str2number(env_str)
    env_np = env_np.astype(np.uint8)

    chute_locs = get_chute_loc(env_np)
    workstation_locs = get_workstation_loc(env_np)
    n_chutes = len(chute_locs)
    n_workstations = len(workstation_locs)
    logger.info(
        f"{n_chutes} chutes, {n_destinations} destinations, {n_workstations} workstations"
    )

    destination_volumes, _ = get_packages(
        package_mode,
        package_dist_type,
        package_path,
        n_destinations,
    )

    destination_volumes /= np.sum(destination_volumes)

    dist_matrix = compute_dist_matrix(env_np)

    chute_mapping = miqp_chute_mapping_no_agent_collision(
        chute_locs,
        workstation_locs,
        destination_volumes,
        dist_matrix,
        n_threads,
        seed,
        time_limit,
        warm_start=False,
        outputflag=1,
    )
    print(chute_mapping)


if __name__ == '__main__':
    fire.Fire({
        "miqp": test_no_agent_collision_miqp,
        "milp": test_chute_capacity_milp,
    })
