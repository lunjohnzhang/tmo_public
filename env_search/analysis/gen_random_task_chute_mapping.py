import os
import gin
import json
import multiprocessing
from typing import Union
from logdir import LogDir
from pathlib import Path
from tqdm import tqdm
from env_search.warehouse import get_packages
from env_search.warehouse.config import WarehouseConfig
from env_search.utils import (DIRS, sortation_env_str2number,
                              read_in_sortation_map, sortation_obj_types,
                              get_Manhattan_distance_coor)
from env_search.analysis.gen_heuristic_chute_mapping import gen_heuristic_chute_mapping
import seaborn as sns
import numpy as np
import fire
import matplotlib.pyplot as plt


def plot_package_distribution(
    package_dist_weight: np.ndarray,
    save_dir: Path,
    id: Union[int, str],
):
    n_destinations = len(package_dist_weight)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(np.arange(n_destinations), package_dist_weight)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("Package ID")
    ax.set_ylabel("Probability")
    ax.set_title("Package Distribution")
    fig.savefig(save_dir / f"package_distribution_{id}.png")
    plt.close(fig)


def gen_random_task_chute_mapping(
    warehouse_config,
    map_filepath,
    chute_mapping_file,
    T=5000,
    gaussian_noise=None,
):
    gin.parse_config_file(warehouse_config)
    warehouse_config = WarehouseConfig()

    # Read in map
    map_str, map_name = read_in_sortation_map(map_filepath)
    map_np = sortation_env_str2number(map_str)
    h, w = map_np.shape
    logdir = LogDir(name=f"gen_random_task_{map_name}_sigma={gaussian_noise}",
                    uuid=8)

    # Read in packages, chute mapping
    package_dist_weight, _ = get_packages(
        warehouse_config.package_mode,
        warehouse_config.package_dist_type,
        warehouse_config.package_path,
        warehouse_config.n_destinations,
    )
    package_dist_weight_init = np.array(package_dist_weight)

    with open(chute_mapping_file, "r") as f:
        chute_mapping_json = json.load(f)
        # chute_mapping_json = json.dumps(chute_mapping_json)

    n_destinations = len(chute_mapping_json.keys())
    task_chutes = []
    task_endpoints = []

    overall_package_dist = np.zeros(n_destinations)
    per_time_step_dist = []
    # Sampling random packages
    for t in range(T):
        if len(package_dist_weight_init.shape) > 1:
            dist = package_dist_weight_init[t].copy()
            dist /= np.sum(dist)
        else:
            if gaussian_noise is not None:
                if t % 100 == 0:
                    # Add random Gaussian noise
                    if gaussian_noise == "adaptive":
                        noise = np.random.normal(0,
                                                 package_dist_weight_init,
                                                 size=n_destinations)
                    else:
                        noise = np.random.normal(0,
                                                 gaussian_noise,
                                                 size=n_destinations)
                    dist = package_dist_weight_init + noise
                    dist = np.clip(dist, 0, 1)
                    dist /= np.sum(dist)
                    per_time_step_dist.append(dist)
                    # plot_package_distribution(dist, task_save_dir, id)
                    # print(f"task id: {id}, dist tuned")
            else:
                dist = package_dist_weight_init

        # Sample 20 packages (assuming tp = 20)

        per_time_step_dist.append(dist)
        for _ in range(20):

            package = np.random.choice(np.arange(n_destinations), p=dist)
            chutes = chute_mapping_json[str(package)]
            task_chute = np.random.choice(chutes, 1, replace=False)
            task_chutes.append(task_chute)
            overall_package_dist[package] += 1

            # Get endpoints around chutes
            endpoints = []
            for chute in chutes:
                x = chute // w
                y = chute % w
                for dir in DIRS:
                    dx, dy = dir
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and \
                        map_np[nx, ny] == sortation_obj_types.index("e") and \
                        get_Manhattan_distance_coor((x, y), (nx, ny)) == 1:
                        endpoints.append(nx * w + ny)
                        # break
            task_endpoints.append(np.random.choice(endpoints, 1))

    # Plot the init and sampling task distribution
    if len(package_dist_weight_init.shape) > 1:
        original_dist = np.sum(package_dist_weight_init, axis=0)
    else:
        original_dist = package_dist_weight_init
    plot_package_distribution(original_dist, logdir.logdir, "init")
    overall_package_dist /= np.sum(overall_package_dist)
    plot_package_distribution(overall_package_dist, logdir.logdir, "overall")

    gen_heuristic_chute_mapping(map_np, map_name + "_sampling",
                                overall_package_dist, n_destinations)

    # Plot the all task distributions using multiprocessing and store in
    # task_save_dir
    task_save_dir = logdir.pdir("task_distributions", touch=True)
    pool = multiprocessing.Pool(16)
    pool.starmap(
        plot_package_distribution,
        zip(
            per_time_step_dist,
            [task_save_dir] * len(per_time_step_dist),
            range(len(per_time_step_dist)),
        ),
    )

    endpoint_task_map = np.zeros((h, w), dtype=int)
    for e in task_endpoints:
        x = e // w
        y = e % w
        endpoint_task_map[x, y] += 1
    chute_task_map = np.zeros((h, w), dtype=int)
    for c in task_chutes:
        x = c // w
        y = c % w
        chute_task_map[x, y] += 1

    endpoint_task_map = endpoint_task_map / np.sum(endpoint_task_map)
    chute_task_map = chute_task_map / np.sum(chute_task_map)

    print(endpoint_task_map)
    print(chute_task_map)

    # Plot task distribution on the map
    fig_chute, (ax_chute, ax_endpoint) = plt.subplots(1, 2, figsize=(15, 5))

    sns.heatmap(
        endpoint_task_map,
        square=True,
        cmap="Reds",
        ax=ax_endpoint,
        cbar=True,
        rasterized=False,
        annot_kws={"size": 30},
        # linewidths=0.1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=0.05,
    )

    # ax_chute.imshow(chute_task_map, cmap="hot", interpolation="nearest")
    ax_chute.set_title("Chute Task Distribution")
    # ax_endpoint.imshow(endpoint_task_map, cmap="hot", interpolation="nearest")
    sns.heatmap(
        chute_task_map,
        square=True,
        cmap="Reds",
        ax=ax_chute,
        cbar=True,
        rasterized=False,
        annot_kws={"size": 30},
        # linewidths=0.1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=0.05,
    )

    ax_endpoint.set_title("Endpoint Task Distribution")
    fig_chute.tight_layout()
    fig_chute.savefig(logdir.logdir / "task_distribution.png")


if __name__ == '__main__':
    fire.Fire(gen_random_task_chute_mapping)
