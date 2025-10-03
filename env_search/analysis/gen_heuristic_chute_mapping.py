import json
import fire
import gin
import numpy as np
from queue import Queue
from tqdm import tqdm

from env_search.warehouse import get_packages
from env_search.warehouse.config import WarehouseConfig
from env_search.warehouse.module import BFS_path_len_one_goal
from env_search.utils import (read_in_sortation_map, sortation_env_number2str,
                              sortation_env_str2number, get_chute_loc,
                              get_workstation_loc, sortation_obj_types, DIRS)


def compute_avg_dist(x, Y):
    """Compute average L2 distance between x and all elements in Y.
    """
    X = np.array([x] * len(Y))
    all_dists = np.linalg.norm(np.array(X) - np.array(Y), axis=1)
    return np.mean(all_dists)


def gen_heuristic_chute_mapping(
        map_np: np.ndarray,
        map_name: str,
        package_dist_weight: np.ndarray,
        n_destinations: int,
        algo="min_dist",  # min_dist, cluster
):
    # print(package_dist_weight)
    h, w = map_np.shape
    # Get location of chutes/workstations
    chute_locs = get_chute_loc(map_np)
    chute_coor = [(loc // w, loc % w) for loc in chute_locs]
    n_chutes = len(chute_locs)
    workstation_locs = get_workstation_loc(map_np)
    workstation_coor = [(loc // w, loc % w) for loc in workstation_locs]
    block_indexs = [
        sortation_obj_types.index("@"),
        sortation_obj_types.index("T"),
    ]

    chute_mapping = {d: [] for d in range(n_destinations)}

    # Get max chute per destination based on package distribution
    max_chute_per_dest = []
    for v in package_dist_weight:
        max_chute_per_dest.append(int(v * n_chutes) + 1)

    if algo == "min_dist":
        # Compute the distance between each chute and the closest workstation
        chute_len_to_w = []  # List of (chute_loc, path_len)
        print("Computing distance between chutes and workstations")
        for i, c in enumerate(tqdm(chute_coor)):
            map_np_c = map_np.copy()
            map_np_c[c] = sortation_obj_types.index(".")
            path_len = BFS_path_len_one_goal(c, workstation_coor, map_np_c,
                                             block_indexs)
            chute_len_to_w.append((chute_locs[i], path_len))
            # print(c, path_len)

        # Sort chute_len_to_w by path_len
        chute_len_to_w = sorted(chute_len_to_w, key=lambda x: x[1])
        print(chute_len_to_w)

        # Algorithm 1
        # chute_idx = n_chutes - 1
        # while chute_idx >= 0:
        #     # Loop through destination and assign chutes
        #     for d in reversed(range(n_destinations)):
        #         max_chute = max_chute_per_dest[d]
        #         # Not enough chutes, assign the currently furthest chute
        #         if len(chute_mapping[d]) < max_chute:
        #             chute_mapping[d].append(int(chute_len_to_w[chute_idx][0]))
        #             chute_idx -= 1
        #             if chute_idx < 0:
        #                 break

        # Algorithm 2: Assign chutes to destinations based on distance
        chute_idx = 0
        remain_chutes = n_chutes
        # Assign chutes until all chutes is assigned
        for d in range(n_destinations):
            max_chute = max_chute_per_dest[d]
            # Keep assigning until max chute is reached or there are not enough
            # chutes for the rest of the destinations
            remain_dest = n_destinations - d - 1
            while len(chute_mapping[d]) < max_chute and \
                  remain_chutes > remain_dest and remain_chutes > 0:
                chute_mapping[d].append(int(chute_len_to_w[chute_idx][0]))
                chute_idx += 1
                remain_chutes -= 1

    # Algorithm 3: Assign chutes such that chutes of same destinations are
    # clustered together
    elif algo == "cluster":
        # obtain the rectangle of the chutes
        min_chute_idx = min(chute_locs)
        min_chute_coor = (min_chute_idx // w, min_chute_idx % w)

        unoccupied_chutes = set(chute_coor)
        existing_centroids = []

        remain_chutes = n_chutes
        for d in range(n_destinations):
            max_chute = max_chute_per_dest[d]
            remain_dest = n_destinations - d - 1
            # We must remain at least `remain_dest` chutes for the rest of the
            # destinations
            n_chutes_to_use = min(max_chute, max(remain_chutes - remain_dest,
                                                 0))
            # print(f"Destination {d}: {square_a} x {square_b} + {remain}")

            # Put the first chute
            # Put the chutes such that they are clustered together
            if len(existing_centroids) == 0:
                # If no centroid exists, put the first chute in top left corner
                chute_mapping[d].append(min_chute_coor)
                unoccupied_chutes.remove(min_chute_coor)
            else:
                # Otherwise, put it in the coordinate furthest from the existing
                # centroids
                max_dist = -1
                max_chute_coor = -1
                for c in unoccupied_chutes:
                    # compute the average distance from c to all existing
                    # centroids
                    avg_dist = compute_avg_dist(c, existing_centroids)
                    if avg_dist > max_dist:
                        max_dist = avg_dist
                        max_chute_coor = c
                chute_mapping[d].append(max_chute_coor)
                unoccupied_chutes.remove(max_chute_coor)

            # Put the rest of the chutes
            for i in range(1, n_chutes_to_use):
                # Find the chute in occupied_chutes that is the closest to the
                # existing chutes in chute_mapping[d]
                min_dist = np.inf
                min_chute_coor = -1
                for c in unoccupied_chutes:
                    avg_dist = compute_avg_dist(c, chute_mapping[d])
                    if avg_dist < min_dist:
                        min_dist = avg_dist
                        min_chute_coor = c
                chute_mapping[d].append(min_chute_coor)
                unoccupied_chutes.remove(min_chute_coor)

            # Compute the new centroid and add to existing_centroids
            new_centroid = np.mean(chute_mapping[d], axis=0)
            existing_centroids.append(new_centroid)

            remain_chutes -= n_chutes_to_use

        assert remain_chutes == 0, "Not all chutes are assigned"
        assert len(unoccupied_chutes) == 0, "There are unoccupied chutes"

        # Change coordinates to indices
        for d in chute_mapping:
            for i, c in enumerate(chute_mapping[d]):
                chute_mapping[d][i] = int(c[0] * w + c[1])

    # Check if chutes are double assigned
    all_chutes = []
    for v in chute_mapping.values():
        all_chutes.extend(v)
    assert len(all_chutes) == len(
        set(all_chutes)), "Chutes are double assigned"

    with open(f"{map_name}_heuristic_baseline_algo={algo}.json", "w") as f:
        json.dump(chute_mapping, f, indent=4)


def gen_heuristic_chute_mapping_cmd(
        map_filepath,
        warehouse_config,
        algo="min_dist",  # min_dist, cluster
):
    gin.parse_config_file(warehouse_config)
    warehouse_config = WarehouseConfig()

    # Read in map
    map_str, map_name = read_in_sortation_map(map_filepath)
    map_np = sortation_env_str2number(map_str)
    h, w = map_np.shape

    # Read in package distribution
    package_dist_weight, _ = get_packages(
        warehouse_config.package_mode,
        warehouse_config.package_dist_type,
        warehouse_config.package_path,
        warehouse_config.n_destinations,
    )
    gen_heuristic_chute_mapping(
        map_np=map_np,
        map_name=map_name,
        package_dist_weight=package_dist_weight,
        n_destinations=warehouse_config.n_destinations,
        algo=algo,
    )


if __name__ == '__main__':
    fire.Fire(gen_heuristic_chute_mapping_cmd)
