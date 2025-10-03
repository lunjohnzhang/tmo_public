"""Miscellaneous project-wide utilities."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing
import copy
from queue import Queue
from env_search import MAP_DIR
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from typing import Dict

# 6 object types for kiva map:
# '.' (0) : empty space
# '@' (1): obstacle (shelf)
# 'e' (2): endpoint (point around shelf)
# 'r' (3): robot start location (not searched)
# 's' (4): one of 'r'
# 'w' (5): workstation
# NOTE:
# 1: only the first 2 or 3 objects are searched by QD
# 2: s (r_s) is essentially one of r s.t. in milp can make the graph
# connected
kiva_obj_types = ".@ersw"
KIVA_ROBOT_BLOCK_WIDTH = 4
KIVA_WORKSTATION_BLOCK_WIDTH = 2
KIVA_ROBOT_BLOCK_HEIGHT = 4
MIN_SCORE = 0

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# right, up, left, down (in that direction because it is the same
# of the simulator!!)
kiva_directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]

kiva_color_map = {
    kiva_obj_types.index("."): "white",
    kiva_obj_types.index("@"): "black",
    kiva_obj_types.index("e"): "deepskyblue",
    kiva_obj_types.index("r"): "orange",
    kiva_obj_types.index("w"): "fuchsia",
}

kiva_dir_map = {
    (0, 1): 0,  # right
    (-1, 0): 1,  # up
    (0, -1): 2,  # left
    (1, 0): 3,  # down
}

kiva_rev_d_map = {
    (0, 1): (0, -1),  # right to left
    (-1, 0): (1, 0),  # up to down
    (0, -1): (0, 1),  # left to right
    (1, 0): (-1, 0),  # down to up
}

# 6 object types for kiva map:
# '.' (0) : empty space
# '0' (1): manufacture station type 0 (obstacle)
# '1' (2): manufacture station type 1 (obstacle)
# '2' (3): manufacture station type 2 (obstacle)
# 'e' (4): endpoint (point around manufacture station)
# 's' (5): special 'e'.
# NOTE:
# 1: only the first 5 objects are searched by QD
# 2: s (e_s) is essentially one of e s.t. in milp can make the graph connected
manufacture_obj_types = ".012es"

# 2 object types for maze map
# ' ' (0): empty space
# 'X' (1): obstacle
maze_obj_types = " X"

# 5 object types for sortation map:
# '.' (0) : empty space
# '@' (1): chutes (obstacle)
# 'e' (2): endpoint (point around chutes)
# 'o' (3): obstacles other than chutes
# 'w' (4): workstation
sortation_obj_types = ".@eTw"


def format_env_str(env_str):
    """Format the env from List[str] to pure string separated by \n """
    return "\n".join(env_str)


def env_str2number(env_str, obj_types):
    env_np = []
    for row_str in env_str:
        # print(row_str)
        row_np = [obj_types.index(tile) for tile in row_str]
        env_np.append(row_np)
    return np.array(env_np, dtype=int)


def env_number2str(env_np, obj_types):
    env_str = []
    n_row, n_col = env_np.shape
    for i in range(n_row):
        curr_row = []
        for j in range(n_col):
            curr_row.append(obj_types[env_np[i, j]])
        env_str.append("".join(curr_row))
    return env_str


def kiva_env_str2number(env_str):
    """
    Convert kiva env in string format to np int array format.

    Args:
        env_str (List[str]): kiva env in string format

    Returns:
        env_np (np.ndarray)
    """
    return env_str2number(env_str, kiva_obj_types)


def kiva_env_number2str(env_np):
    """
    Convert kiva env in np int array format to str format.

    Args:
        env_np (np.ndarray): kiva env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, kiva_obj_types)


def manufacture_env_str2number(env_str):
    """
    Convert manufacture env in string format to np int array format.

    Args:
        env_str (List[str]): manufacture env in string format

    Returns:
        env_np (np.ndarray)
    """
    return env_str2number(env_str, manufacture_obj_types)


def manufacture_env_number2str(env_np):
    """
    Convert manufacture env in np int array format to str format.

    Args:
        env_np (np.ndarray): manufacture env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, manufacture_obj_types)


def maze_env_str2number(env_str):
    """
    Convert maze env in string format to np int array format.

    Args:
        env_str (List[str]): maze env in string format

    Returns:
        env_np (np.ndarray)
    """
    return env_str2number(env_str, maze_obj_types)


def maze_env_number2str(env_np):
    """
    Convert maze env in np int array format to str format.

    Args:
        env_np (np.ndarray): maze env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, maze_obj_types)


def sortation_env_str2number(env_str):
    """
    Convert sortation env in string format to np int array format.

    Args:
        env_str (List[str]): sortation env in string format

    Returns:
        env_np (np.ndarray)
    """
    return env_str2number(env_str, sortation_obj_types)


def sortation_env_number2str(env_np):
    """
    Convert sortation env in np int array format to str format.

    Args:
        env_np (np.ndarray): sortation env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, sortation_obj_types)


def flip_one_r_to_s(env_np, obj_types=kiva_obj_types):
    """
    Change one of 'r' in the env to 's' for milp
    """
    all_r = np.argwhere(env_np == obj_types.index("r"))
    if len(all_r) == 0:
        raise ValueError("No 'r' found")
    to_replace = all_r[0]
    env_np[tuple(to_replace)] = obj_types.index('s')
    return env_np


def flip_one_e_to_s(env_np, obj_types=kiva_obj_types):
    """
    Change one of 'e' in the env to 's' for milp
    """
    all_e = np.argwhere(env_np == obj_types.index("e"))
    if len(all_e) == 0:
        raise ValueError("No 'e' found")
    to_replace = all_e[0]
    env_np[tuple(to_replace)] = obj_types.index('s')
    return env_np


def flip_tiles(env_np, from_tile, to_tile, obj_types=kiva_obj_types):
    """Replace ALL occurance of `from_tile` to `flip_target` in `to_tile"""
    all_from_tiles = np.where(env_np == obj_types.index(from_tile))
    if len(all_from_tiles[0]) == 0:
        raise ValueError(f"No '{from_tile}' found")
    env_np[all_from_tiles] = obj_types.index(to_tile)
    return env_np


def flip_tiles_torch(env_torch, from_tile, to_tile, obj_types=kiva_obj_types):
    """Replace ALL occurance of `from_tile` to `flip_target` in `to_tile"""
    all_from_tiles = torch.where(env_torch == obj_types.index(from_tile))
    if len(all_from_tiles[0]) == 0:
        raise ValueError(f"No '{from_tile}' found")
    env_torch[all_from_tiles] = obj_types.index(to_tile)
    return env_torch


def _read_in_map(map_filepath):
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
        raw_env = raw_env_json["layout"]
        name = raw_env_json["name"]
    return raw_env, name


# The following functions are currently implemented the same for possible
# future extensions
def read_in_kiva_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    return _read_in_map(map_filepath)


def read_in_manufacture_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    return _read_in_map(map_filepath)


def read_in_maze_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    return _read_in_map(map_filepath)


def read_in_sortation_map(map_filepath):
    """
    Read in sortation map and return in str format
    """
    return _read_in_map(map_filepath)


def write_map_str_to_json(
    map_filepath,
    repaired_env_str,
    name,
    domain,
    weight=False,
    weights=None,
    nca_runtime=None,
    milp_runtime=None,
    nca_milp_runtime=None,
    sim_score=None,
    piu_n_agent=None,
    scenario="KIVA",
):
    """Write specified map to disk.

    Args:
        map_filepath (str or pathlib.Path): filepath to write
        repaired_env_str (List[str]): map in str format
        name (str): name of the map
        domain (str): domain of the map
        weight (bool, optional): Whether the map is weighted. Defaults to False.
        weights (List[float], optional): the edge weights of the map. Defaults
            to None.
        nca_runtime (float, optional): runtime of NCA. Defaults to None.
        milp_runtime (float, optional): runtime of MILP. Defaults to None.
        nca_milp_runtime (float, optional): runtime of NCA + MILP. Defaults to
            None.
        sim_score (float, optional): similarity score. Default to None.
        piu_n_agent (int, optional): number of agent the weights are generated
            with PIU algorithm.
    """
    to_write = {
        "name": name,
        "layout": repaired_env_str,
        "nca_runtime": nca_runtime,
        "milp_runtime": milp_runtime,
        "nca_milp_runtime": nca_milp_runtime,
        "sim_score": sim_score,
        "piu_n_agent": piu_n_agent,
        "weight": weight,
        "weights": weights,
    }
    if domain == "manufacture":
        map_np = manufacture_env_str2number(repaired_env_str)
        # to_write["weight"] = False
        to_write["n_row"] = map_np.shape[0]
        to_write["n_col"] = map_np.shape[1]
        to_write["n_0"] = sum(row.count('0') for row in repaired_env_str)
        to_write["n_1"] = sum(row.count('1') for row in repaired_env_str)
        to_write["n_2"] = sum(row.count('2') for row in repaired_env_str)
        to_write["n_stations"] = \
            to_write["n_0"] + to_write["n_1"] + to_write["n_2"]
    elif domain in ["kiva", "sortation"]:
        if scenario == "KIVA":
            map_np = kiva_env_str2number(repaired_env_str)
        elif scenario == "SORTING":
            map_np = sortation_env_str2number(repaired_env_str)
        # to_write["weight"] = False
        to_write["n_row"] = map_np.shape[0]
        to_write["n_col"] = map_np.shape[1]
        to_write["n_endpoint"] = sum(
            row.count('e') for row in repaired_env_str)
        to_write["n_agent_loc"] = sum(
            row.count('r') for row in repaired_env_str)
        to_write["n_shelf"] = sum(row.count('@') for row in repaired_env_str)
        to_write["maxtime"] = 5000
    elif domain == "maze":
        map_np = maze_env_str2number(repaired_env_str)
        to_write["n_row"] = map_np.shape[0]
        to_write["n_col"] = map_np.shape[1]

    with open(map_filepath, "w") as json_file:
        json.dump(to_write, json_file, indent=4)


def set_spines_visible(ax: plt.Axes):
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(True)


def n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # params_ = 0
    # state_dict = model.state_dict()
    # for _, param in state_dict.items():
    #     params_ += np.prod(param.shape)
    # print("validate: ", params_)

    return params


def rewrite_map(path, domain):
    if domain == "manufacture":
        env, name = read_in_manufacture_map(path)
    elif domain == "kiva":
        env, name = read_in_kiva_map(path)
    write_map_str_to_json(path, env, name, domain)


def load_pibt_default_config():
    """Return default PIBT config as json str.
    """
    config_path = "WPPL/configs/pibt_default_no_rot.json"
    with open(config_path) as f:
        config = json.load(f)
        config_str = json.dumps(config)
    return config_str


def single_sim_done(result_dir_full):
    """Check if previous single simulation is done.
    """
    config_file = os.path.join(result_dir_full, "config.json")
    result_file = os.path.join(result_dir_full, "result.json")

    if os.path.exists(config_file) and os.path.exists(result_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            with open(result_file, "r") as f:
                result = json.load(f)
        except json.decoder.JSONDecodeError:
            return False
        return True
    return False


def get_n_valid_edges(map_np, bi_directed, domain):
    n_valid_edges = 0
    if domain == "kiva":
        block_idxs = [
            kiva_obj_types.index("@"),
        ]
    elif domain == "sortation":
        block_idxs = [
            sortation_obj_types.index("@"),
            sortation_obj_types.index("T"),
        ]
    h, w = map_np.shape
    for x in range(h):
        for y in range(w):
            # Skip all obstacles
            if map_np[x, y] in block_idxs:
                continue
            # Check if neighbor is obstacle or out of range
            for dx, dy in DIRS:
                n_x = x + dx
                n_y = y + dy
                if n_x < h and n_x >= 0 and \
                    n_y < w and n_y >= 0 and \
                    map_np[n_x,n_y] not in block_idxs:
                    n_valid_edges += 1
    assert n_valid_edges % 2 == 0
    # If graph is uni-directed, we have half the number of the edges
    # compared to bi-directed counterpart.
    if not bi_directed:
        n_valid_edges = n_valid_edges // 2

    return n_valid_edges


def get_n_valid_vertices(map_np, domain):
    if domain == "kiva":
        return np.sum(map_np != kiva_obj_types.index("@"), dtype=int)
    elif domain == "sortation":
        return np.sum(
            np.logical_and(
                map_np != sortation_obj_types.index("@"),
                map_np != sortation_obj_types.index("T"),
            ),
            dtype=int,
        )


def min_max_normalize(arr, lb, ub):
    """Min-Max normalization on 1D array `arr`

    Args:
        arr (array-like): array to be normalized
        lb (float): lower bound
        ub (float): upper bound
    """
    arr = np.asarray(arr)
    min_ele = np.min(arr)
    max_ele = np.max(arr)
    if max_ele - min_ele < 1e-3:
        # Clip and then return
        return np.clip(arr, lb, ub)
    arr_norm = lb + (arr - min_ele) * (ub - lb) / (max_ele - min_ele)
    if np.any(np.isnan(arr_norm)):
        print(arr)
    return arr_norm


def write_iter_update_model_to_json(filepath, model_param, model_type):
    to_write = {
        "type": str(model_type),
        "params": model_param,
    }
    with open(filepath, "w") as f:
        json.dump(to_write, f, indent=4)


def get_tile_loc(map_np, tile_type, obj_type, flatten=True):
    if flatten:
        tile_loc = np.argwhere(map_np.flatten() == obj_type.index(tile_type))
        return tile_loc.squeeze().astype(int)
    else:
        return np.argwhere(map_np == obj_type.index(tile_type))


def get_chute_loc(map_np, flatten=True):
    return get_tile_loc(map_np, "@", sortation_obj_types, flatten)


def get_workstation_loc(map_np, flatten=True):
    return get_tile_loc(map_np, "w", sortation_obj_types, flatten)


def get_endpoint_loc(map_np, flatten=True):
    return get_tile_loc(map_np, "e", sortation_obj_types, flatten)


def create_grid_graph_with_obstacles(grid):
    """
    Create an adjacency matrix for a grid graph with obstacles.
    grid: 2D numpy array where 0 represents free space and 1 represents obstacles.
    Returns:
        - adjacency: CSR sparse matrix representing the adjacency graph.
        - pos_to_node: Dictionary mapping (i, j) grid positions to node indices.
        - node_to_pos: Dictionary mapping node indices to (i, j) grid positions.
    """
    N, M = grid.shape
    free_positions = [(i, j) for i in range(N) for j in range(M)
                      if grid[i, j] == 0]
    num_nodes = len(free_positions)

    # Maps between grid positions and node indices
    pos_to_node = {pos: idx for idx, pos in enumerate(free_positions)}
    node_to_pos = {idx: pos for idx, pos in enumerate(free_positions)}

    row = []
    col = []
    data = []

    for idx, (i, j) in enumerate(free_positions):
        neighbors = []
        # Check neighboring positions (up, down, left, right)
        # Up, Down, Left, Right
        for di, dj in DIRS:
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < M and grid[ni, nj] == 0:
                neighbor_idx = pos_to_node[(ni, nj)]
                row.append(idx)
                col.append(neighbor_idx)
                data.append(1)  # Edge weight (1 for unweighted graph)

    adjacency = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adjacency, pos_to_node, node_to_pos


def compute_dist_matrix(map_np):
    """Compute a distance matrix between workstations and chutes.
    """
    chute_loc = get_chute_loc(map_np, flatten=False)
    workstation_loc = get_workstation_loc(map_np, flatten=False)
    endpoint_loc = get_endpoint_loc(map_np, flatten=False)
    n_chutes = len(chute_loc)
    n_workstations = len(workstation_loc)
    map_grid = np.zeros(map_np.shape)
    map_grid[np.nonzero(map_np == sortation_obj_types.index("@"))] = 1

    # Create csgraph from the grid
    adjacency, loc_to_node, node_to_pos = create_grid_graph_with_obstacles(
        map_grid)

    def compute_distances(adjacency):
        """
        Compute shortest path distances between nodes in set A and nodes in
        set B.
        """
        # Compute shortest paths from nodes in A to all other nodes
        dist_matrix = shortest_path(csgraph=adjacency, directed=False)
        # Extract distances to nodes in B
        # distances = dist_matrix[:, B_indices]
        return dist_matrix

    w_indices = [loc_to_node[tuple(loc)] for loc in workstation_loc]
    e_indices = [loc_to_node[tuple(loc)] for loc in endpoint_loc]

    distances = compute_distances(adjacency)

    # Convert distances to a dictionary
    # Note: Chutes are considered as obstacles, so we need to find the closest
    # endpoints and +2 as the distance between them
    # First find distance from workstations to chutes
    dist_matrix = {tuple(loc): {} for loc in workstation_loc}
    for i, w_index in enumerate(w_indices):
        w_loc = node_to_pos[w_index]
        for c_loc in chute_loc:
            # Find the endpoint around the chute that is closest to the
            # workstation w_loc
            min_dist = np.inf
            for di, dj in DIRS:
                e_loc = (c_loc[0] + di, c_loc[1] + dj)
                e_index = loc_to_node[e_loc]
                dist = distances[w_index, e_index]
                if dist < min_dist:
                    min_dist = dist
            dist_matrix[tuple(w_loc)][tuple(c_loc)] = min_dist

    # Then find distance from chutes to chutes
    for i in range(n_chutes):
        c1 = chute_loc[i]
        endpt_c1 = [(c1[0] + di, c1[1] + dj) for di, dj in DIRS]
        for j in range(i + 1, n_chutes):
            c2 = chute_loc[j]
            endpt_c2 = [(c2[0] + di, c2[1] + dj) for di, dj in DIRS]
            min_dist = np.inf
            for e1 in endpt_c1:
                for e2 in endpt_c2:
                    e1_index = loc_to_node[tuple(e1)]
                    e2_index = loc_to_node[tuple(e2)]
                    dist = distances[e1_index, e2_index]
                    if dist < min_dist:
                        min_dist = dist
            if tuple(c1) not in dist_matrix:
                dist_matrix[tuple(c1)] = {}
            if tuple(c2) not in dist_matrix:
                dist_matrix[tuple(c2)] = {}
            # breakpoint()
            dist_matrix[tuple(c1)][tuple(c2)] = min_dist + 2
            dist_matrix[tuple(c2)][tuple(c1)] = min_dist + 2

    chute_loc_1d = get_chute_loc(map_np, flatten=True)
    workstation_loc_1d = get_workstation_loc(map_np, flatten=True)
    dist_matrix_flatten = {
        loc: {}
        for loc in [*workstation_loc_1d, *chute_loc_1d]
    }
    _, w = map_np.shape
    for loc in dist_matrix.keys():
        i, j = loc
        loc_flatten = i * w + j
        for c_loc in chute_loc:
            c_i, c_j = c_loc
            c_loc_flatten = c_i * w + c_j

            if loc_flatten == c_loc_flatten:
                dist_matrix_flatten[loc_flatten][loc_flatten] = 0
            else:
                dist_matrix_flatten[loc_flatten][c_loc_flatten] = \
                    dist_matrix[tuple(loc)][tuple(c_loc)]
    del dist_matrix
    return dist_matrix_flatten


def BFS_path_len(start_loc, goal_locs, env_np, block_idxs):
    """
    Find shortest path from start_loc to all goal_locs
    """
    # Set goal loc as none-blocking tile, otherwise it cannot be reached.
    # env_np[goal_locs] = kiva_obj_types.index(".")
    result_path_len = {}
    n_goals = len(goal_locs)
    q = Queue()
    q.put(start_loc)
    seen = set()
    m, n = env_np.shape
    dist_matrix = np.full((m, n), np.inf)
    dist_matrix[start_loc] = 0

    while not q.empty():
        curr = q.get()
        x, y = curr
        around_goal, goals = reaches_goal(curr, goal_locs)
        if around_goal:
            shortest = dist_matrix[x, y] + 1
            for goal_reached in goals:
                result_path_len[goal_reached] = shortest
                goal_locs.remove(goal_reached)

            # print(f"Found goal {goal_reached}")
            # print(f"Remaining number of goals {len(goal_locs)}")

            # All goals found?
            if len(goal_locs) == 0:
                assert len(result_path_len) == n_goals
                return result_path_len

        seen.add(curr)
        for dx, dy in DIRS:
            n_x = x + dx
            n_y = y + dy
            if n_x < m and n_x >= 0 and \
               n_y < n and n_y >= 0 and \
               env_np[n_x,n_y] not in block_idxs and\
               (n_x, n_y) not in seen:
                q.put((n_x, n_y))
                dist_matrix[n_x, n_y] = dist_matrix[x, y] + 1
    raise ValueError(f"Start loc: {start_loc}. Remaining goal: {goal_locs}")


def reaches_goal(loc, goal_locs):
    """
    A `loc` reaches goal if any loc in `goal_locs` is around `loc`.
    """
    x, y = loc
    around_goal = False
    goals = []
    for goal_loc in goal_locs:
        for dx, dy in DIRS:
            n_x = x + dx
            n_y = y + dy
            if tuple([n_x, n_y]) == goal_loc:
                goals.append(goal_loc)
                around_goal = True
    return around_goal, goals


def kiva_uncompress_edge_weights(
    map_np,
    edge_weights,
    block_idxs,
    fill_value=np.nan,
):
    """Transform the raw list of edge weights to np array of size (h, w, 4),
    where the order in the last dimension is right, up, left, down (in that
    direction because it is the same as the simulator!!)
    Args:
        map_np (np.ndarray): the map in numpy format
        edge_weights (list of float): raw edge weights
        block_idxs: the indices of the map that should be considered as
            obstacles.

    Returns:
        edge_weights_matrix (np.ndarray of size(# nodes, 4)): edge weights of
            the bi-directed graph.
    """
    h, w = map_np.shape
    edge_weights_matrix = np.zeros((h, w, 4))
    edge_weights_matrix[:, :, :] = fill_value
    weight_cnt = 0
    move = [1, -w, -1, w]

    for i in range(h * w):
        x = i // w
        y = i % w
        assert i == w * x + y
        if map_np[x, y] in block_idxs:
            continue
        for dir in range(4):
            n_x = (i + move[dir]) // w
            n_y = (i + move[dir]) % w
            if 0 <= i + move[dir] < h * w and get_Manhattan_distance(
                    i, i + move[dir],
                    w) <= 1 and map_np[n_x, n_y] not in block_idxs:
                edge_weights_matrix[x, y, dir] = edge_weights[weight_cnt]
                weight_cnt += 1
    assert weight_cnt == len(edge_weights)
    return edge_weights_matrix

    # right, up, left, down (in that direction because it is the same
    # as the simulator!!)
    # for x in range(h):
    #     for y in range(w):
    #         # Skip all obstacles
    #         if map_np[x,y] in block_idxs:
    #             continue
    #         for idx, (dx, dy) in enumerate(d):
    #             n_x = x + dx
    #             n_y = y + dy
    #             if n_x < h and n_x >= 0 and \
    #                n_y < w and n_y >= 0 and \
    #                map_np[n_x,n_y] not in block_idxs:
    #                 optimized_graph[x,y,idx] = weights[weight_cnt]
    #                 weight_cnt += 1
    # assert weight_cnt == len(weights)
    # return optimized_graph


def kiva_uncompress_wait_costs(
    map_np,
    wait_costs,
    block_idxs,
    fill_value=np.nan,
):
    """Transform the raw list of wait costs to np array of size (h, w)
    Args:
        map_np (np.ndarray): the map in numpy format
        wait_costs (list of float): raw edge weights
        block_idxs: the indices of the map that should be considered as
            obstacles.

    Returns:
        optimized_graph (np.ndarray of size(# nodes, 4)): edge weights of the
            bi-directed graph.
    """
    h, w = map_np.shape
    wait_costs_matrix = np.zeros((h, w))
    wait_costs_matrix[:, :] = fill_value
    i = 0

    for x in range(h):
        for y in range(w):
            if map_np[x, y] in block_idxs:
                continue
            wait_costs_matrix[x, y] = wait_costs[i]
            i += 1
    return wait_costs_matrix


def kiva_compress_edge_weights(map_np, edge_weights, block_idxs):
    """Transform the edge weigths (np array of size (h, w, 4)) to raw weights.
    The order in the last dimension is right, up, left, down (in that
    direction because it is the same as the simulator!!)

    Args:
        edge_weights (np.ndarray of size(h, w, 4)): edge weights of the
            bi-directed graph.

    Returns:
        compress_edge_weights (list of float): raw edge weights
    """
    h, w = map_np.shape
    compress_edge_weights = []
    for x in range(h):
        for y in range(w):
            # Skip all obstacles
            if map_np[x, y] in block_idxs:
                continue
            # Check if neighbor is obstacle or out of range
            # node_idx = w * x + y
            for d_idx, (dx, dy) in enumerate(kiva_directions):
                n_x = x + dx
                n_y = y + dy
                if n_x < h and n_x >= 0 and \
                    n_y < w and n_y >= 0 and \
                    map_np[n_x,n_y] not in block_idxs:
                    # Edge should be valid
                    assert not np.isnan(edge_weights[x][y][d_idx])
                    compress_edge_weights.append(edge_weights[x][y][d_idx])
    return compress_edge_weights


def kiva_compress_wait_costs(map_np, wait_costs, block_idxs):
    """Transform the wait costs (np array of size (h, w, 4)) to raw weights.
    The order in the last dimension is right, up, left, down (in that
    direction because it is the same as the simulator!!)

    Args:
        wait_costs (np.ndarray of size(h, w, 4)): wait costs of the
            bi-directed graph.

    Returns:
        compress_wait_costs (list of float): raw wait costs
    """
    h, w = map_np.shape
    compress_wait_costs = []
    for x in range(h):
        for y in range(w):
            # Skip all obstacles
            if map_np[x, y] in block_idxs:
                continue
            compress_wait_costs.append(wait_costs[x, y])
    return compress_wait_costs


def get_Manhattan_distance(loc1, loc2, w):
    return abs(loc1 // w - loc2 // w) + abs(loc1 % w - loc2 % w)


def get_Manhattan_distance_coor(coordinate1, coordinate2):
    x1, y1 = coordinate1
    x2, y2 = coordinate2
    return abs(x1 - x2) + abs(y1 - y2)


def min_max_normalize_2d(arr, lb, ub):
    """Min-Max normalization on 2D array `arr`

    Args:
        arr (array-like): array to be normalized
        lb (float): lower bound
        ub (float): upper bound
    """
    arr = np.asarray(arr)
    min_sols = np.min(arr, axis=1, keepdims=True)
    max_sols = np.max(arr, axis=1, keepdims=True)
    arr_norm = lb + (arr - min_sols) * (ub - lb) / (max_sols - min_sols)
    return arr_norm


def chute_mapping_to_array(chute_mapping: Dict, chute_locs: np.ndarray):
    chute_mapping_arr = np.zeros(len(chute_locs), dtype=int)
    for d, chutes in chute_mapping.items():
        for chute in chutes:
            chute_mapping_arr[np.where(chute_locs == chute)[0][0]] = int(d)
    return chute_mapping_arr


def chute_mapping_list_to_dict(
    chute_mapping: np.ndarray,
    chute_locs: np.ndarray,
    n_destinations: int,
):
    chute_mapping_dict = {d: [] for d in range(n_destinations)}
    for d in range(len(chute_mapping)):
        chute_mapping_dict[d].append(chute_locs[d])
    return chute_mapping_dict
