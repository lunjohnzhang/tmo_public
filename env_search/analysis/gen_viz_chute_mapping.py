import fire
import json

from env_search.utils import (
    kiva_env_str2number, get_chute_loc, read_in_kiva_map,
    read_in_sortation_map, sortation_env_str2number, get_n_valid_edges,
    get_n_valid_vertices, DIRS, read_in_sortation_map, sortation_obj_types,
    get_Manhattan_distance_coor, load_pibt_default_config, get_workstation_loc)

def gen_viz_chute_mapping(map_filepath, n_destinations):
    # Read in map
    map_str, map_name = read_in_sortation_map(map_filepath)
    map_np = sortation_env_str2number(map_str)

    # Get chute locations
    chute_locs = get_chute_loc(map_np)
    full_chute_mapping = {
        d: chute_locs.tolist() for d in range(n_destinations)
    }
    with open(f"chute_mapping/{map_name}_full.json", "w") as f:
        json.dump(full_chute_mapping, f)

    one_chute_mapping = {
        d: [int(chute_locs[0])] for d in range(n_destinations)
    }

    with open(f"chute_mapping/{map_name}_one.json", "w") as f:
        json.dump(one_chute_mapping, f)


if __name__ == "__main__":
    fire.Fire(gen_viz_chute_mapping)