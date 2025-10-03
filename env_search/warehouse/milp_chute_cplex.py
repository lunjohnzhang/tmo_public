import os
import gc
import json
import time
import fire
import math
import logging
import numpy as np
import docplex

from docplex.mp.model import Context, Model

from env_search.utils.logging import setup_logging
from env_search.utils import (kiva_env_str2number, read_in_kiva_map,
                              sortation_env_str2number, read_in_sortation_map,
                              get_chute_loc)

logger = logging.getLogger(__name__)


def define_variables(chute_locs, n_destinations, mdl: Model):
    edges = []
    # dict from chute to all edges incident to the chute
    edge_by_chutes = {c_loc: [] for c_loc in chute_locs}
    # dict from destination to all edges incident to the destination
    edge_by_destinations = {j: [] for j in range(n_destinations)}
    for c_loc in chute_locs:
        for j in range(n_destinations):
            edge_var = mdl.integer_var(name=f"edge_c_{c_loc}_d_{j}",
                                       lb=0,
                                       ub=1)
            edges.append(edge_var)
            edge_by_chutes[c_loc].append(edge_var)
            edge_by_destinations[j].append(edge_var)
    return edge_by_chutes, edge_by_destinations, edges


def milp_chute_mapping_min_pairwise_dist(chute_locs, package_dist_weight):
    # ----------------------------------------------------
    # 1) Input Data
    # ----------------------------------------------------
    # Suppose we have N 2D points:
    points = [tuple(loc) for loc in chute_locs]
    N = len(points)

    # Number of clusters (M < N)
    M = len(package_dist_weight)

    # Precompute distances d(i,j) for i < j
    dist = {}
    for i in range(N):
        for j in range(i + 1, N):
            x_i, y_i = points[i]
            x_j, y_j = points[j]
            d_ij = math.dist((x_i, y_i), (x_j, y_j))  # Euclidean distance
            dist[(i, j)] = d_ij
            # If you prefer, you can store dist also as dist[(j, i)] = d_ij,
            # but we'll just enforce i<j in constraints.

    # Compute cluster sizes
    n_chutes = len(chute_locs)
    max_chute_per_dest = []
    for v in package_dist_weight:
        max_chute_per_dest.append(int(v * n_chutes) + 1)

    remain_chutes = n_chutes
    cluster_sizes = []
    for d in range(M):
        max_chute = max_chute_per_dest[d]
        remain_dest = M - d - 1
        # We must remain at least `remain_dest` chutes for the rest of the
        # destinations
        n_chutes_to_use = min(max_chute, max(remain_chutes - remain_dest, 0))
        square_a = int(np.sqrt(n_chutes_to_use))
        square_b = n_chutes_to_use // square_a
        remain = n_chutes_to_use - square_a * square_b
        # print(f"Destination {d}: {square_a} x {square_b} + {remain}")
        cluster_sizes.append(n_chutes_to_use)
        remain_chutes -= n_chutes_to_use
        # print(f"Destination {d}: {n_chutes_to_use} chutes")

    assert np.sum(cluster_sizes) == n_chutes

    # ----------------------------------------------------
    # 2) Create Model
    # ----------------------------------------------------
    context = Context.make_default_context()
    context.cplex_parameters.threads = 32
    mdl = Model(name="Minimize_intra_cluster_distance", context=context)
    mdl.parameters.mip.pool.relgap = 0.05
    # ----------------------------------------------------
    # 3) Decision Variables
    # ----------------------------------------------------
    # x[i,k] = 1 if point i is assigned to cluster k, 0 otherwise.
    x = {
        (i, k): mdl.binary_var(name=f"x_{i}_{k}")
        for i in range(N)
        for k in range(M)
    }

    # z[i,j,k] = 1 if points i and j are both in cluster k, 0 otherwise.
    # We'll only define z for i < j to avoid duplication.
    z = {
        (i, j, k): mdl.binary_var(name=f"z_{i}_{j}_{k}")
        for (i, j) in dist.keys()  # i < j
        for k in range(M)
    }

    # ----------------------------------------------------
    # 4) Constraints
    # ----------------------------------------------------
    # 4.1) Each point is in exactly one cluster
    for i in range(N):
        mdl.add_constraint(mdl.sum(x[i, k] for k in range(M)) == 1,
                           ctname=f"assign_{i}")

    # 4.2) Linking constraints for z[i,j,k]
    #      z_{i,j,k} = 1 iff x_{i,k} = 1 and x_{j,k} = 1
    for (i, j) in dist.keys():  # i < j
        for k in range(M):
            # z_{i,j,k} >= x_{i,k} + x_{j,k} - 1
            mdl.add_constraint(z[i, j, k] >= x[i, k] + x[j, k] - 1)
            # z_{i,j,k} <= x_{i,k}
            mdl.add_constraint(z[i, j, k] <= x[i, k])
            # z_{i,j,k} <= x_{j,k}
            mdl.add_constraint(z[i, j, k] <= x[j, k])

    # 4.3) (Optional) Ensure each cluster is non-empty
    for k in range(M):
        mdl.add_constraint(mdl.sum(x[i, k] for i in range(N)) >= 1,
                           ctname=f"non_empty_cluster_{k}")

    # 4.3) Fixed cluster sizes: sum of x[i,k] == cluster_sizes[k]
    for k in range(M):
        mdl.add_constraint(mdl.sum(x[i, k]
                                   for i in range(N)) == cluster_sizes[k],
                           ctname=f"cluster_size_{k}")

    # ----------------------------------------------------
    # 5) Objective: Minimize sum of distances among points in the same cluster
    # ----------------------------------------------------
    # sum_{k} sum_{i<j} d(i,j) * z_{i,j,k}
    mdl.minimize(
        mdl.sum(dist[(i, j)] * z[i, j, k] for (i, j) in dist.keys()  # i<j
                for k in range(M)))

    # ----------------------------------------------------
    # 6) Solve
    # ----------------------------------------------------
    solution = mdl.solve(log_output=True)

    # ----------------------------------------------------
    # 7) Retrieve and Print Solution
    # ----------------------------------------------------
    if solution is not None:
        logger.info(
            f"Objective value (total intra-cluster distance) = {solution.objective_value:.3f}\n"
        )
        logger.info(f"MILP Gap: {mdl.solve_details.mip_relative_gap}")

        # Get cluster assignments
        clusters = [[] for _ in range(M)]
        for i in range(N):
            for k in range(M):
                if x[i, k].solution_value > 0.5:
                    clusters[k].append(i)
                    break

        # Print clusters
        for k, members in enumerate(clusters):
            print(f"Cluster {k}: {members} -> {[points[i] for i in members]}")

    else:
        print("No solution found.")


def milp_chute_mapping(
    chute_locs,
    chute_capacities,
    destination_volumes,
    n_threads,
    seed,
    time_limit,
    ub_chutes_per_dest=True,
):
    """MILP program that generate the chute mapping given chute capacities and
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
    n_chutes = len(chute_locs)
    n_destinations = len(destination_volumes)
    context = Context.make_default_context()
    context.cplex_parameters.threads = n_threads
    context.cplex_parameters.dettimelimit = time_limit * 1000
    if seed is not None:
        context.cplex_parameters.randomseed = seed

    with Model(context=context) as mdl:
        edge_by_chutes, edge_by_destinations, edges = define_variables(
            chute_locs, n_destinations, mdl)

        # Constraints
        # 1. Each chute can serve exactly 1 destination
        for c_loc in chute_locs:
            mdl.add_constraint(
                mdl.sum(edge_by_chutes[c_loc]) == 1,
                ctname=f"chute_{c_loc}_serves_1_dest",
            )
        # 2. Each destination is assigned to [1, U_j] chutes
        # Compute U_j = max(1, 1.5 * n_chutes * destination_volumes[j])
        for j in range(n_destinations):
            mdl.add_constraint(
                mdl.sum(edge_by_destinations[j]) >= 1,
                ctname=f"dest_{j}_served_by_1_chute",
            )
            if ub_chutes_per_dest:
                U_j = int(
                    np.ceil(max(1, 1.5 * n_chutes * destination_volumes[j])))
                # print(U_j)
                mdl.add_constraint(
                    mdl.sum(edge_by_destinations[j]) <= U_j,
                    ctname=f"dest_{j}_served_by_U_chutes",
                )

        # Objective
        # diff_volume = []
        # for i, c_loc in enumerate(chute_locs):
        #     # compute the total volume that is allocated to chute c_loc
        #     c_loc_volume = []
        #     for j in range(n_destinations):
        #         c_loc_volume.append(destination_volumes[j] *
        #                             edge_by_chutes[c_loc][j])
        #     diff_volume.append(
        #         mdl.abs(mdl.sum(c_loc_volume) - chute_capacities[i]))
        # mdl.minimize(mdl.sum(diff_volume))

        diff_volume = []
        for j in range(n_destinations):
            # Compute the total capacity assigned to destination j
            dest_volume = []
            for i, c_loc in enumerate(chute_locs):
                dest_volume.append(edge_by_destinations[j][i] *
                                   chute_capacities[i])
            diff_volume.append(
                mdl.abs(mdl.sum(dest_volume) - destination_volumes[j]))
        mdl.minimize(mdl.sum(diff_volume))

        # Solve
        solution = mdl.solve()

        # Return None if no solution
        if solution is None:
            print("No solution")
            return None

        logger.info(f"Objective value: {solution.objective_value}")
        logger.info(f"MILP Gap: {mdl.solve_details.mip_relative_gap}")
        chute_mapping = extract_solution(solution, chute_locs, n_destinations)
        del solution
        del mdl
        gc.collect()
        return chute_mapping


def milp_chute_mapping_edit_dist(
    chute_locs,
    unrepaired_chute_mapping,
    destination_volumes,
    n_threads,
    seed,
    time_limit,
):
    """MILP program that repair a given chute mapping by minimizing the number
    of edits.

    Args:
        chute_locs (list): list of locations of the chutes
        unrepaired_chute_mapping (list[list[int]]): unrepaired chute mapping
        destination_volumes (list[float]): destination volumes
        n_threads (int): number of threads for MILP
        seed (int): random seed
        time_limit (float): time limit in seconds

    Returns:
        chute_mapping: dict(int, int) where key is destination_id, value is
            chute_location
    """
    n_chutes = len(chute_locs)
    n_destinations = len(destination_volumes)
    context = Context.make_default_context()
    context.cplex_parameters.threads = n_threads
    context.cplex_parameters.dettimelimit = time_limit * 1000
    if seed is not None:
        context.cplex_parameters.randomseed = seed

    with Model(context=context) as mdl:
        edge_by_chutes, edge_by_destinations, edges = define_variables(
            chute_locs, n_destinations, mdl)

        # Constraints
        # 1. Each chute can serve exactly 1 destination
        for c_loc in chute_locs:
            mdl.add_constraint(
                mdl.sum(edge_by_chutes[c_loc]) == 1,
                ctname=f"chute_{c_loc}_serves_1_dest",
            )
        # 2. Each destination is assigned to [1, U_j] chutes
        # Compute U_j = max(1, 1.5 * n_chutes * destination_volumes[j])
        for j in range(n_destinations):
            mdl.add_constraint(
                mdl.sum(edge_by_destinations[j]) >= 1,
                ctname=f"dest_{j}_served_by_1_chute",
            )
            U_j = int(np.ceil(max(1, 1.5 * n_chutes * destination_volumes[j])))
            # print(U_j)
            mdl.add_constraint(
                mdl.sum(edge_by_destinations[j]) <= U_j,
                ctname=f"dest_{j}_served_by_U_chutes",
            )

        # Objective: minimize the number of edits
        edit_costs = []
        for j in range(n_destinations):
            for i, c_loc in enumerate(chute_locs):
                if c_loc in unrepaired_chute_mapping[j]:
                    edit_costs.append(1 - edge_by_destinations[j][i])
                else:
                    edit_costs.append(edge_by_destinations[j][i])
        mdl.minimize(mdl.sum(edit_costs))

        # Solve
        solution = mdl.solve()

        # Return None if no solution
        if solution is None:
            print("No solution")
            return None

        logger.info(f"Objective value: {solution.objective_value}")
        logger.info(f"MILP Gap: {mdl.solve_details.mip_relative_gap}")
        chute_mapping = extract_solution(solution, chute_locs, n_destinations)
        del solution
        del mdl
        gc.collect()
        return chute_mapping


def extract_solution(solution, chute_locs, n_destinations):
    chute_mapping = {j: [] for j in range(n_destinations)}
    for j in range(n_destinations):
        for c_loc in chute_locs:
            if np.abs(solution[f"edge_c_{c_loc}_d_{j}"] - 1) <= 1e-5:
                chute_mapping[j].append(int(c_loc))
    return chute_mapping


def test_milp_chute_mapping(map_filepath,
                            warehouse_config,
                            n_threads=1,
                            seed=0,
                            time_limit=180):
    # Read in envs
    setup_logging(on_worker=False)
    np.random.seed(seed)
    env_str, _ = read_in_sortation_map(map_filepath)
    env_np = sortation_env_str2number(env_str)
    env_np = env_np.astype(np.uint8)

    import gin
    from env_search.warehouse.config import WarehouseConfig
    from env_search.warehouse import get_packages
    gin.parse_config_file(warehouse_config)
    warehouse_config = WarehouseConfig()
    destination_volumes, _ = get_packages(
        warehouse_config.package_mode,
        warehouse_config.package_dist_type,
        warehouse_config.package_path,
        warehouse_config.n_destinations,
    )

    with open(
            "chute_mapping/scatter_sleep/sortation_33_57_105-chutes_heuristic_baseline_algo=cluster.json",
            "r") as f:
        chute_mapping = json.load(f)

    chute_locs = get_chute_loc(env_np)

    # Compute chute capacities
    chute_capacities = {c: 0 for c in chute_locs}
    for d in chute_mapping:
        for c in chute_mapping[d]:
            chute_capacities[c] += destination_volumes[int(d)] / len(
                chute_mapping[d])

    chute_capacities = [chute_capacities[c] for c in chute_locs]

    # # chute_locs = [0, 1, 2, 3]
    # n_destinations = 42
    # n_chutes = len(chute_locs)
    # exp_deg = n_chutes / n_destinations
    # chute_capacities = np.random.rand(n_chutes)
    # # chute_capacities = [0.4, 0.1, 0.4, 0.1]
    # # Normalize the chute capacities
    # chute_capacities /= np.sum(chute_capacities)
    # # chute_capacities *= exp_deg

    # destination_volumes = np.random.rand(n_destinations)
    # # destination_volumes = [0.8, 0.2]
    # # Normalize the destination volumes
    # destination_volumes /= np.sum(destination_volumes)
    breakpoint()
    chute_mapping_sol = milp_chute_mapping(
        chute_locs,
        chute_capacities,
        destination_volumes,
        n_threads,
        seed,
        time_limit,
    )

    # Compare with original
    for d in chute_mapping:
        print(d, set(chute_mapping[d]) == set(chute_mapping_sol[int(d)]))

    # print(chute_mapping_sol)


def test_milp_chute_mapping_edit_dist(
    map_filepath,
    n_threads=1,
    seed=0,
    time_limit=180,
):
    # Read in envs
    setup_logging(on_worker=False)
    np.random.seed(seed)
    env_str, _ = read_in_sortation_map(map_filepath)
    env_np = sortation_env_str2number(env_str)
    env_np = env_np.astype(np.uint8)

    chute_locs = get_chute_loc(env_np)
    n_destinations = 100
    n_chutes = len(chute_locs)

    destination_volumes = np.random.rand(n_destinations)
    destination_volumes /= np.sum(destination_volumes)

    sol = np.random.randint(low=0, high=n_destinations, size=n_chutes)
    unrepaired_chute_mapping = {j: [] for j in range(n_destinations)}
    for i, c_loc in enumerate(chute_locs):
        unrepaired_chute_mapping[sol[i]].append(c_loc)
    print(unrepaired_chute_mapping)
    chute_mapping = milp_chute_mapping_edit_dist(
        chute_locs,
        unrepaired_chute_mapping,
        destination_volumes,
        n_threads,
        seed,
        time_limit,
    )
    print(chute_mapping)


def test_milp_chute_mapping_min_pairwise_dist(map_filepath, warehouse_config):
    import gin
    from env_search.warehouse.config import WarehouseConfig
    from env_search.warehouse import get_packages
    gin.parse_config_file(warehouse_config)
    warehouse_config = WarehouseConfig()

    # Read in package distribution
    package_dist_weight, _ = get_packages(
        warehouse_config.package_mode,
        warehouse_config.package_dist_type,
        warehouse_config.package_path,
        warehouse_config.n_destinations,
    )

    # Read in envs
    setup_logging(on_worker=False)
    env_str, _ = read_in_sortation_map(map_filepath)
    env_np = sortation_env_str2number(env_str)
    env_np = env_np.astype(np.uint8)

    chute_locs = get_chute_loc(env_np, flatten=False)
    milp_chute_mapping_min_pairwise_dist(chute_locs, package_dist_weight)


if __name__ == '__main__':
    fire.Fire(test_milp_chute_mapping)
