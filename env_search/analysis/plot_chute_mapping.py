import fire
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from env_search.warehouse.module import (compute_avg_dist_to_centroid,
                                         compute_avg_min_dist_to_ws)
from env_search.utils import (read_in_sortation_map, sortation_env_number2str,
                              sortation_env_str2number, get_chute_loc,
                              get_workstation_loc, sortation_obj_types, DIRS)


def plot_chute_mapping(map_np,
                       map_name,
                       chute_mapping,
                       chute_mapping_name,
                       save_dir=".",
                       mapping="all"):
    h, w = map_np.shape

    # Plot all destinations for only the top `mapping` destinations
    if mapping == "all":
        n_destinations = len(chute_mapping.keys())
    else:
        n_destinations = mapping

    # Initialize the grid map with -1 (uncolored)
    grid_map = -1 * np.ones((h, w), dtype=int)

    # Fill the grid map based on the dictionary
    for d, chutes in list(chute_mapping.items())[:n_destinations]:
        for c in chutes:
            x, y = c // w, c % w
            grid_map[x, y] = d

    # Create a custom colormap with white for undefined locations
    num_colors = n_destinations + 1  # +1 for white
    # Generate colormap for dictionary keys
    base_cmap = plt.cm.get_cmap("tab20", num_colors - 1)
    # Add white
    new_colors = np.vstack((
        [1, 1, 1, 1],
        base_cmap(np.linspace(0, 1, num_colors - 1)),
    ))
    custom_cmap = ListedColormap(new_colors)

    # fig, ax = plt.subplots(figsize=(10, 10))
    plt.figure(figsize=(10, 5))
    plt.imshow(grid_map, cmap=custom_cmap)
    cbar = plt.colorbar()
    spaces = np.linspace(
        0,
        n_destinations - 1,
        n_destinations,
        endpoint=False,
    )
    cbar.set_ticks(spaces + spaces[1] / 2)
    cbar.set_ticklabels([f"{i}" for i in range(n_destinations)])
    plt.tight_layout()
    path = pathlib.Path(save_dir)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    save_path = path / f"{chute_mapping_name}.png"
    plt.savefig(save_path)

    # Compute some stats
    avg_centroid_dist = compute_avg_dist_to_centroid(map_np, chute_mapping)
    print(f"Average centroid distance: {avg_centroid_dist}")
    block_idxs = [
        sortation_obj_types.index("@"),
        sortation_obj_types.index("T"),
    ]
    avg_min_dist_to_ws = compute_avg_min_dist_to_ws(map_np, block_idxs,
                                                    chute_mapping)
    print(f"Average min distance to workstation: {avg_min_dist_to_ws}")


def plot_chute_mapping_cmd(map_filepath,
                           chute_mapping_filepath,
                           mapping="all"):
    # Read in map
    map_str, map_name = read_in_sortation_map(map_filepath)
    map_np = sortation_env_str2number(map_str)
    h, w = map_np.shape

    chute_mapping_name = pathlib.Path(chute_mapping_filepath).name
    chute_mapping_name = chute_mapping_name.split(".")[0]

    with open(chute_mapping_filepath, "r") as f:
        chute_mapping = json.load(f)
    chute_mapping = {int(k): v for k, v in chute_mapping.items()}
    plot_chute_mapping(map_np, map_name, chute_mapping, chute_mapping_name,
                       mapping)


if __name__ == "__main__":
    fire.Fire(plot_chute_mapping_cmd)
