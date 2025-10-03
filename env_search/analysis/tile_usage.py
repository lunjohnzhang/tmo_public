from operator import index
import os
from typing import List
import gin
import json
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import fire
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing

from tqdm import tqdm
from logdir import LogDir
from pathlib import Path
from pprint import pprint
from cma.constraints_handler import BoundTransform
from env_search.warehouse import get_packages, QUAD_TASK_ASSIGN_N_PARAM
from env_search.warehouse.config import WarehouseConfig
from env_search.analysis.utils import (load_experiment, load_metrics,
                                       load_archive_gen, get_extreme_pt)
from env_search.analysis.visualize_env import (visualize_kiva,
                                               visualize_manufacture)
from env_search.utils.logging import setup_logging
from env_search.utils.enum import SearchSpace
from env_search.mpl_styles.utils import mpl_style_file
from env_search.analysis.visualize_highway import plot_highway_edge_weights
from env_search.analysis.plot_chute_mapping import plot_chute_mapping
from env_search.utils import (
    set_spines_visible, KIVA_ROBOT_BLOCK_WIDTH, KIVA_WORKSTATION_BLOCK_WIDTH,
    KIVA_ROBOT_BLOCK_HEIGHT, kiva_env_number2str, kiva_env_str2number,
    read_in_kiva_map, read_in_manufacture_map, read_in_sortation_map,
    manufacture_env_str2number, sortation_env_str2number,
    sortation_env_number2str, write_map_str_to_json, min_max_normalize,
    write_iter_update_model_to_json, n_params, get_chute_loc,
    get_n_valid_vertices, min_max_normalize_2d)

mpl.use("agg")

FIG_HEIGHT = 10

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def get_tile_usage_vmax(env_h, env_w):
    if env_h * env_w == 9 * 16 or env_h * env_w == 9 * 20:
        vmax = 0.03
    elif env_h * env_w == 17 * 16 or env_h * env_w == 17 * 20:
        vmax = 0.02
    elif env_h * env_w == 33 * 36:
        vmax = 0.005
    elif env_h * env_w == 101 * 102:
        vmax = 0.002
    elif env_h * env_w == 33 * 57:
        vmax = 0.002
    else:
        vmax = None
    return vmax


def plot_capacity(
    logdir,
    env_np,
    chute_mapping,
    package_dist,
    chute_capacities=None,
    filenames: List = ["capacities.pdf", "capacities.svg", "capacities.png"],
    dpi=300,
):
    """Plot allocated and predicted capacities of each chute on map
    """
    chute_locs = get_chute_loc(env_np)
    allocated_capacities_map = np.zeros(env_np.shape).flatten().tolist()

    for j, mapped_chutes in chute_mapping.items():
        for c_loc in mapped_chutes:
            allocated_capacities_map[c_loc] += package_dist[int(j)] / len(
                mapped_chutes)
    allocated_capacities_map = np.array(allocated_capacities_map).reshape(
        env_np.shape)

    fig, (ax_alloc, ax_predicted) = plt.subplots(
        2,
        1,
        figsize=get_figsize_sim(env_np),
    )

    sns.heatmap(
        allocated_capacities_map,
        square=True,
        cmap="Reds",
        ax=ax_alloc,
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
    ax_alloc.set_title("Allocated Capacities", fontsize=20)

    cbar = ax_alloc.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    set_spines_visible(ax_alloc)
    ax_alloc.figure.tight_layout()

    if chute_capacities is not None:
        predicted_capacities_map = np.zeros(env_np.shape).flatten().tolist()
        for i, c_loc in enumerate(chute_locs):
            predicted_capacities_map[c_loc] = chute_capacities[i]
        predicted_capacities_map = np.array(predicted_capacities_map).reshape(
            env_np.shape)
        sns.heatmap(
            predicted_capacities_map,
            square=True,
            cmap="Reds",
            ax=ax_predicted,
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
        ax_predicted.set_title("Predicted Capacities", fontsize=20)

        cbar = ax_predicted.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)

        set_spines_visible(ax_predicted)
        ax_predicted.figure.tight_layout()

    for filename in filenames:
        fig.savefig(os.path.join(logdir, filename), dpi=dpi)


def plot_heatmap(
    tile_usage,
    env_h,
    env_w,
    fig,
    ax_tile_use,
    ax_tile_use_cbar,
    logdir,
    filenames: List = ["tile_usage.pdf", "tile_usage.svg", "tile_usage.png"],
    vmin=0,
    vmax=500,
    dpi=300,
):
    # Plot tile usage
    tile_usage = tile_usage.reshape(env_h, env_w)

    sns.heatmap(
        tile_usage,
        square=True,
        cmap="Reds",
        ax=ax_tile_use,
        cbar_ax=ax_tile_use_cbar,
        cbar=True,
        rasterized=False,
        annot_kws={"size": 30},
        # linewidths=0.1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={
            "orientation": "horizontal",
            "shrink": 0.7,
        },
    )

    cbar = ax_tile_use.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    set_spines_visible(ax_tile_use)
    ax_tile_use.figure.tight_layout()

    for filename in filenames:
        fig.savefig(logdir.file(filename), dpi=dpi)


def get_figsize_qd(w_mode=True, domain="kiva"):
    # Decide figsize based on size of map
    if domain in ["kiva", "sortation"]:
        env_h = gin.query_parameter("WarehouseManager.env_height")
        env_w = gin.query_parameter("WarehouseManager.env_width")
    elif domain == "manufacture":
        env_h = gin.query_parameter("ManufactureManager.env_height")
        env_w = gin.query_parameter("ManufactureManager.env_width")

    if env_h * env_w == 9 * 12 or env_h * env_w == 9 * 16:
        if w_mode:
            figsize = (8, 8)
        else:
            figsize = (8, 12)
    elif env_h * env_w == 17 * 12 or env_h * env_w == 17 * 16:
        figsize = (8, 16)
    elif env_h * env_w == 33 * 32 or env_h * env_w == 33 * 36:
        figsize = (8, 16)
    elif env_h * env_w == 33 * 57:
        figsize = (8, 10)
    else:
        figsize = (8, 16)

    return figsize


def get_figsize_sim(env_np):
    # Decide figsize based on size of map
    env_h, env_w = env_np.shape
    return (FIG_HEIGHT * env_w / env_h / 2, FIG_HEIGHT)

    # if env_h * env_w == 9 * 16 or env_h * env_w == 9 * 20:
    #     figsize = (8, 8)
    # elif env_h * env_w == 17 * 16 or env_h * env_w == 17 * 20:
    #     figsize = (8, 16)
    # elif env_h * env_w == 33 * 36:
    #     figsize = (8, 16)
    # elif env_h * env_w == 33 * 57:
    #     figsize = (8, 10)
    # elif env_h * env_w == 140 * 500:
    #     figsize = (15, 15)
    # else:
    #     figsize = (8, 16)

    # return figsize


def _tile_usage_single_run_plot_one(
    fig,
    ax_tile_use,
    ax_tile_use_cbar,
    results_dir,
    sim_dir,
    env_np,
    map_name,
    dpi,
):
    sim_dir_comp = os.path.join(results_dir, sim_dir)
    result_filepath = os.path.join(sim_dir_comp, "result.json")
    with open(result_filepath, "r") as f:
        result_json = json.load(f)
    tile_usage = result_json["vertex_wait_matrix"]
    # if np.sum(tile_usage) - 1 > 1e-6:
    #     tile_usage = tile_usage / np.sum(tile_usage)
    plot_heatmap(
        np.array(tile_usage),
        env_np.shape[0],
        env_np.shape[1],
        fig,
        ax_tile_use,
        ax_tile_use_cbar,
        LogDir(map_name, custom_dir=sim_dir_comp),
        filenames=[
            f"tile_usage_{map_name}.pdf",
            f"tile_usage_{map_name}.svg",
            f"tile_usage_{map_name}.png",
        ],
        vmin=0,
        vmax=1000,
        dpi=dpi,
    )


def _finished_tasks_single_run_plot_one(
    fig,
    ax_tile_use,
    ax_tile_use_cbar,
    results_dir,
    sim_dir,
    env_np,
    map_name,
    dpi,
):
    sim_dir_comp = os.path.join(results_dir, sim_dir)
    result_filepath = os.path.join(sim_dir_comp, "result.json")
    with open(result_filepath, "r") as f:
        result_json = json.load(f)
    finished_tasks_map = result_json["finished_tasks"]

    # finished_tasks_map = np.zeros(env_np.shape).flatten().tolist()
    # for task in finished_tasks:
    #     loc, t = task
    #     finished_tasks_map[loc] += 1
    finished_tasks_map /= np.sum(finished_tasks_map)

    # if np.sum(finished_tasks) - 1 > 1e-6:
    #     finished_tasks = finished_tasks / np.sum(finished_tasks)
    plot_heatmap(
        np.array(finished_tasks_map),
        env_np.shape[0],
        env_np.shape[1],
        fig,
        ax_tile_use,
        ax_tile_use_cbar,
        LogDir(map_name, custom_dir=sim_dir_comp),
        filenames=[
            f"finished_tasks_{map_name}.pdf",
            f"finished_tasks_{map_name}.svg",
            f"finished_tasks_{map_name}.png",
        ],
        vmin=0,
        vmax=0.01,
        dpi=dpi,
    )


def _chute_sleep_count_single_run_plot_one(
    fig,
    ax_tile_use,
    ax_tile_use_cbar,
    results_dir,
    sim_dir,
    env_np,
    map_name,
    dpi,
):
    sim_dir_comp = os.path.join(results_dir, sim_dir)
    result_filepath = os.path.join(sim_dir_comp, "result.json")
    with open(result_filepath, "r") as f:
        result_json = json.load(f)
    chute_sleep_count = result_json["chute_sleep_count"]

    chute_sleep_count_map = np.zeros(env_np.shape).flatten().tolist()
    for chute, sleep_count in chute_sleep_count:
        chute_sleep_count_map[chute] = sleep_count
    # chute_sleep_count_map /= np.sum(chute_sleep_count_map)

    # if np.sum(chute_sleep_count) - 1 > 1e-6:
    #     chute_sleep_count = chute_sleep_count / np.sum(chute_sleep_count)
    plot_heatmap(
        np.array(chute_sleep_count_map),
        env_np.shape[0],
        env_np.shape[1],
        fig,
        ax_tile_use,
        ax_tile_use_cbar,
        LogDir(map_name, custom_dir=sim_dir_comp),
        filenames=[
            f"chute_sleep_count_{map_name}.pdf",
            f"chute_sleep_count_{map_name}.svg",
            f"chute_sleep_count_{map_name}.png",
        ],
        vmin=0,
        vmax=20,
        dpi=dpi,
    )


def _chute_n_sleep_single_run_plot_one(
    fig,
    ax_tile_use,
    ax_tile_use_cbar,
    results_dir,
    sim_dir,
    env_np,
    map_name,
    dpi,
):
    sim_dir_comp = os.path.join(results_dir, sim_dir)
    result_filepath = os.path.join(sim_dir_comp, "result.json")
    with open(result_filepath, "r") as f:
        result_json = json.load(f)
    total_chute_sleep_time = result_json["total_chute_sleep_time"]

    total_chute_sleep_time_map = np.zeros(env_np.shape).flatten().tolist()
    for chute, sleep_count in total_chute_sleep_time:
        total_chute_sleep_time_map[chute] = sleep_count
    # total_chute_sleep_time_map /= np.sum(total_chute_sleep_time_map)

    # if np.sum(total_chute_sleep_time) - 1 > 1e-6:
    #     total_chute_sleep_time = total_chute_sleep_time / np.sum(total_chute_sleep_time)
    plot_heatmap(
        np.array(total_chute_sleep_time_map),
        env_np.shape[0],
        env_np.shape[1],
        fig,
        ax_tile_use,
        ax_tile_use_cbar,
        LogDir(map_name, custom_dir=sim_dir_comp),
        filenames=[
            f"total_chute_sleep_time_{map_name}.pdf",
            f"total_chute_sleep_time_{map_name}.svg",
            f"total_chute_sleep_time_{map_name}.png",
        ],
        vmin=0,
        vmax=5000,
        dpi=dpi,
    )


def tile_usage_heatmap_from_single_run(
    logdir: str,
    dpi=300,
    domain="kiva",
    n_workers=32,
):
    """
    Plot tile usage with map layout from a single run of warehouse simulation.
    """
    # Read in config (to get the scenario)
    results_dir = os.path.join(logdir, "results")
    with open(
            os.path.join(
                results_dir,
                os.listdir(results_dir)[0],
                "config.json",
            ), "r") as f:
        config = json.load(f)

    # Read in map
    map_filepath = os.path.join(logdir, "map.json")
    if domain in ["kiva", "sortation"]:
        if config["scenario"] == "KIVA":
            map, map_name = read_in_kiva_map(map_filepath)
            env_np = kiva_env_str2number(map)
        elif config["scenario"] == "SORTING":
            map, map_name = read_in_sortation_map(map_filepath)
            env_np = sortation_env_str2number(map)
    elif domain == "manufacture":
        map, map_name = read_in_manufacture_map(map_filepath)
        env_np = manufacture_env_str2number(map)

    # Read in chute capacities, if available, and plot allocated and predicted
    # chute capacities
    chute_mapping = None
    chute_mapping_filepath = os.path.join(logdir, "chute_mapping.json")
    if os.path.exists(chute_mapping_filepath):
        with open(chute_mapping_filepath, "r") as f:
            chute_mapping = json.load(f)

        # Read in one config.json from results_dir to get the configurations
        config_filepath = os.path.join(results_dir,
                                       os.listdir(results_dir)[0],
                                       "config.json")
        with open(config_filepath, "r") as f:
            config = json.load(f)
        package_dist, _ = get_packages(
            config["package_mode"],
            config["package_dist_type"],
            config["package_path"],
            config["n_destinations"],
        )
        plot_capacity(logdir, env_np, chute_mapping, package_dist)

    # Create plot
    grid_kws = {"height_ratios": (0.475, 0.475, 0.05)}
    fig, (ax_map, ax_tile_use, ax_tile_use_cbar) = plt.subplots(
        3,
        1,
        figsize=get_figsize_sim(env_np),
        gridspec_kw=grid_kws,
    )

    # Plot map
    if domain in ["kiva", "sortation"]:
        visualize_kiva(env_np, ax=ax_map, dpi=300)
    elif domain == "manufacture":
        visualize_manufacture(env_np, ax=ax_map, dpi=300)

    env_h, env_w = env_np.shape

    # Read in result and plot tile usage/wait usage
    pool = multiprocessing.Pool(n_workers)
    pool.starmap(
        _tile_usage_single_run_plot_one,
        [(
            fig,
            ax_tile_use,
            ax_tile_use_cbar,
            results_dir,
            sim_dir,
            env_np,
            map_name,
            dpi,
        ) for sim_dir in os.listdir(results_dir)],
    )

    grid_kws = {"height_ratios": (0.95, 0.05)}
    fig, (ax_finished_tasks, ax_finished_tasks_cbar) = plt.subplots(
        2,
        1,
        figsize=(FIG_HEIGHT * env_w / env_h, FIG_HEIGHT),
        gridspec_kw=grid_kws,
    )
    pool.starmap(
        _finished_tasks_single_run_plot_one,
        [(
            fig,
            ax_finished_tasks,
            ax_finished_tasks_cbar,
            results_dir,
            sim_dir,
            env_np,
            map_name,
            dpi,
        ) for sim_dir in os.listdir(results_dir)],
    )

    fig, (ax_chute_sleep_time, ax_chute_sleep_time_cbar) = plt.subplots(
        2,
        1,
        figsize=(FIG_HEIGHT * env_w / env_h, FIG_HEIGHT),
        gridspec_kw=grid_kws,
    )
    pool.starmap(
        _chute_n_sleep_single_run_plot_one,
        [(
            fig,
            ax_chute_sleep_time,
            ax_chute_sleep_time_cbar,
            results_dir,
            sim_dir,
            env_np,
            map_name,
            dpi,
        ) for sim_dir in os.listdir(results_dir)],
    )

    fig, (ax_chute_sleep_count, ax_chute_sleep_count_cbar) = plt.subplots(
        2,
        1,
        figsize=(FIG_HEIGHT * env_w / env_h, FIG_HEIGHT),
        gridspec_kw=grid_kws,
    )
    pool.starmap(
        _chute_sleep_count_single_run_plot_one,
        [(
            fig,
            ax_chute_sleep_count,
            ax_chute_sleep_count_cbar,
            results_dir,
            sim_dir,
            env_np,
            map_name,
            dpi,
        ) for sim_dir in os.listdir(results_dir)],
    )


def tile_usage_heatmap_from_qd(
    logdir: str,
    gen: int = None,
    index_0: int = None,
    index_1: int = None,
    dpi: int = 300,
    mode: str = None,
    domain: str = "kiva",
):
    """
    Plot tile usage with map layout from a QD experiment.
    """
    logdir: LogDir = load_experiment(logdir)
    gen = load_metrics(logdir).total_itrs if gen is None else gen
    archive = load_archive_gen(logdir, gen)
    df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))
    global_opt_env = None
    scenario = gin.query_parameter("WarehouseConfig.scenario")
    search_space = SearchSpace(
        gin.query_parameter("WarehouseManager.search_space"))

    # Convert index to index_0 and index_1 --> pyribs 0.4.0/0.5.0 compatibility
    # issue
    if "index_0" not in df and "index" in df:
        all_grid_index = archive.int_to_grid_index(df["index"])
        df["index_0"] = all_grid_index[:, 0]
        df["index_1"] = all_grid_index[:, 1]

    if index_0 is not None and index_1 is not None:
        to_plots = df[(df["index_0"] == index_0) & (df["index_1"] == index_1)]
        if to_plots.empty:
            raise ValueError("Specified index has no solution")
    elif mode == "extreme":

        # Add global optimal env
        global_opt = df["objective"].idxmax()
        best_sol = df.filter(regex=("solution.*")).iloc[global_opt].to_list()
        meta = df.iloc[global_opt]["metadata"]["raw_metadata"]
        global_opt_env = meta["repaired_env_str"]
        global_opt_env_unrepaired = meta["unrepaired_env_int"]
        # WARNING: This is a temporary fix for the issue with global opt env
        global_opt_env = sortation_env_number2str(global_opt_env_unrepaired)
        sim_score = meta["similarity_score"]
        chute_mapping = meta["chute_mapping"]
        chute_locs = get_chute_loc(np.array(global_opt_env_unrepaired))

        # try:
        #     chute_capacities = meta["chute_capacities"]
        # except:
        #     ####### Temporary, delete later #######
        #     bound_transform = BoundTransform([0.001, None])
        #     chute_capacities = bound_transform.repair(best_sol)

        #     # Normalize the chute capacities s.t. they sum up to
        #     # n_chutes/n_destinations
        #     #
        #     # exp_deg = len(chute_locs) / len(package_dist)
        #     chute_capacities /= np.sum(chute_capacities)
        #     # chute_capacities *= exp_deg
        #     ####### Temporary, delete later #######

        # Chute mapping related
        package_dist, _ = get_packages(
            gin.query_parameter("%package_mode"),
            gin.query_parameter("%package_dist_type"),
            gin.query_parameter("%package_path"),
            gin.query_parameter("%n_destinations"),
        )

        if search_space in [
                SearchSpace.CHUTE_CAPACITIES,
                SearchSpace.CHUTE_MAPPING,
                SearchSpace.G_POLICY_CHUTE_CAPACITIES,
                SearchSpace.G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY,
                SearchSpace.G_GRAPH_CHUTE_CAPACITIES,
                SearchSpace.G_GRAPH_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY,
        ]:
            chute_capacities = None
            if "chute_capacities" in meta:
                chute_capacities = meta["chute_capacities"]
            if chute_capacities is None:
                chute_capacities = best_sol[:len(chute_locs)]
                bound_transform = BoundTransform([0.001, None])
                chute_capacities = bound_transform.repair(chute_capacities)
                chute_capacities /= np.sum(chute_capacities)
            plot_capacity(
                logdir.logdir,
                global_opt_env_unrepaired,
                chute_mapping,
                package_dist,
                chute_capacities=chute_capacities,
            )

        # Guidance graph related
        # edge_weights = meta["edge_weights"]
        # wait_costs = meta["wait_costs"]
        weight = False
        weights = None
        try:
            iterative_update = gin.query_parameter("WarehouseManager.is_piu")
        except ValueError:
            iterative_update = False
        # if edge_weights is not None and wait_costs is not None:
        #     weight = True
        #     lb, ub = gin.query_parameter("%bounds")
        #     edge_weights = min_max_normalize(edge_weights, lb, ub)
        #     wait_costs = min_max_normalize(wait_costs, lb, ub)
        #     weights = np.concatenate([wait_costs, edge_weights]).tolist()

        # Guidance policy related
        if search_space in [SearchSpace.G_POLICY]:
            np.save(logdir.file("optimal_g_policy.npy"), np.array(best_sol))
        elif search_space in [SearchSpace.G_POLICY_CHUTE_CAPACITIES]:
            np.save(logdir.file("optimal_g_policy.npy"),
                    np.array(best_sol[len(chute_capacities):]))
        elif search_space in [SearchSpace.G_POLICY_TASK_ASSIGN_POLICY]:
            np.save(logdir.file("optimal_g_policy.npy"),
                    np.array(best_sol[:-QUAD_TASK_ASSIGN_N_PARAM]))
            np.save(logdir.file("optimal_task_assign_policy.npy"),
                    np.array(best_sol[-QUAD_TASK_ASSIGN_N_PARAM:]))
        elif search_space in [
                SearchSpace.G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY
        ]:
            np.save(
                logdir.file("optimal_g_policy.npy"),
                np.array(
                    best_sol[len(chute_capacities):-QUAD_TASK_ASSIGN_N_PARAM]))
            np.save(logdir.file("optimal_task_assign_policy.npy"),
                    np.array(best_sol[-QUAD_TASK_ASSIGN_N_PARAM:]))
        elif search_space in [SearchSpace.TASK_ASSIGN_POLICY]:
            np.save(logdir.file("optimal_task_assign_policy.npy"),
                    np.array(best_sol))
        elif search_space in [SearchSpace.G_GRAPH]:
            iterative_update = gin.query_parameter("WarehouseManager.is_piu")
            weight = True
            lb, ub = gin.query_parameter("WarehouseConfig.bounds")
            if not iterative_update:
                base_map_np = np.array(global_opt_env_unrepaired)
                n_valid_vertices = get_n_valid_vertices(base_map_np,
                                                        domain=domain)
                wait_costs = best_sol[:n_valid_vertices]
                edge_weights = best_sol[n_valid_vertices:]
            else:
                edge_weights = meta["edge_weights"]
                wait_costs = meta["wait_costs"]

            wait_costs = min_max_normalize(wait_costs, lb, ub)
            edge_weights = min_max_normalize(edge_weights, lb, ub)
            weights = np.concatenate([wait_costs, edge_weights]).tolist()

        # Read in map
        # if domain in ["kiva", "sortation"]:

        if scenario == "KIVA":
            global_opt_env_unrepaired = kiva_env_number2str(
                global_opt_env_unrepaired)
        elif scenario == "SORTING":
            global_opt_env_unrepaired = sortation_env_number2str(
                global_opt_env_unrepaired)

        # # Save global_optimal solution
        global_opt_map_dir = logdir.pdir("map_viz", touch=True)
        write_map_str_to_json(
            global_opt_map_dir / "global_optimal.json",
            global_opt_env,
            "global_optimal",
            domain,
            sim_score=sim_score,
            weight=weight,
            weights=weights,
            scenario=scenario,
        )
        # write_map_str_to_json(
        #     global_opt_map_dir / "global_optimal_unrepaired.json",
        #     global_opt_env_unrepaired,
        #     "global_optimal_unrepaired",
        #     domain,
        #     sim_score=sim_score,
        #     scenario=scenario,
        # )

        # Write chute mapping to disk
        with open(global_opt_map_dir / "chute_mapping.json", "w") as f:
            json.dump(chute_mapping, f)

        # Visualize global optimal map
        if domain in ["kiva", "sortation"]:
            # visualize_kiva(
            #     kiva_env_str2number(global_opt_env),
            #     ["global_optimal.png"],
            #     store_dir=str(global_opt_map_dir),
            # )
            # visualize_kiva(
            #     kiva_env_str2number(global_opt_env_unrepaired),
            #     ["global_optimal_unrepaired.png"],
            #     store_dir=str(global_opt_map_dir),
            # )
            if scenario == "KIVA":
                map_np = kiva_env_str2number(global_opt_env_unrepaired)
            elif scenario == "SORTING":
                map_np = sortation_env_str2number(global_opt_env_unrepaired)
            # Visualize guidance graph
            if weights is not None:
                global_opt_g_graph_dir = logdir.pdir("g_graph_viz", touch=True)
                plot_highway_edge_weights(
                    map_np,
                    weights,
                    map_name="global_optimal",
                    store_dir=str(global_opt_g_graph_dir),
                    domain=domain,
                )

                # Save update model
                if iterative_update:
                    # Write optimal trained update model param
                    config = WarehouseConfig()
                    update_model_type = gin.query_parameter(
                        "WarehouseConfig.iter_update_model_type")
                    tmp_update_model = config.iter_update_model_type(
                        None, None, None, config,
                        **config.iter_update_mdl_kwargs)
                    piu_n_param = n_params(tmp_update_model.model)
                    opt_update_model_param = best_sol[-piu_n_param:]

                    write_iter_update_model_to_json(
                        global_opt_g_graph_dir / "optimal_update_model.json",
                        opt_update_model_param,
                        update_model_type,
                    )

        # Get the maps to plot
        extreme_pts = get_extreme_pt(df)

        to_plots = df.iloc[[
            *extreme_pts,
            global_opt,
        ]]

    elif mode == "extreme-3D":
        # In 3D case, we fix the third dimension and plot the "extreme" points
        # in the archive in first/second dimensions
        partial_df = df[df["behavior_2"] == 20]
        index_0_max = partial_df["index_0"].idxmax()
        index_0_min = partial_df["index_0"].idxmin()
        index_1_max = partial_df["index_1"].idxmax()
        index_1_min = partial_df["index_1"].idxmin()

        # Add global optimal env
        global_opt = partial_df["objective"].idxmax()
        global_opt_env = partial_df.iloc[global_opt]["metadata"][
            "raw_metadata"]["repaired_env_str"]
        to_plots = partial_df.loc[[
            index_0_max,
            index_0_min,
            index_1_max,
            index_1_min,
            global_opt,
        ]]
    elif mode == "compare_human":
        selected_inds = df[(df["behavior_1"] == 20)]
        to_plots = []
        if not selected_inds.empty:
            curr_opt_idx_20 = selected_inds["objective"].idxmax()
            to_plots.append(df.iloc[curr_opt_idx_20])
        else:
            print("No map with 20 shelves in the archive!")
        selected_inds = df[(df["behavior_1"] == 24)]
        if not selected_inds.empty:
            curr_opt_idx_24 = selected_inds["objective"].idxmax()
            to_plots.append(df.iloc[curr_opt_idx_24])
        else:
            print("No map with 24 shelves in the archive!")

        to_plots = pd.DataFrame(to_plots)

    if global_opt_env is not None:
        print("Global optima: ")
        print("\n".join(global_opt_env))
        print()

    with mpl_style_file("tile_usage_heatmap.mplstyle") as f:
        with plt.style.context(f):
            for _, to_plot in to_plots.iterrows():
                index_0 = to_plot["index_0"]
                index_1 = to_plot["index_1"]
                obj = to_plot["objective"]
                metadata = to_plot["metadata"]["raw_metadata"]
                throughput = np.mean(metadata["throughput"])
                sim_score = metadata["similarity_score"]
                print(
                    f"Index ({index_0}, {index_1}): objective = {obj}, throughput = {throughput}, sim_score = {sim_score}"
                )
                if domain in ["kiva", "sortation"]:
                    w_mode = gin.query_parameter("WarehouseManager.w_mode")
                elif domain == "manufacture":
                    w_mode = False

                grid_kws = {"height_ratios": (0.475, 0.475, 0.05)}
                fig, (ax_map, ax_tile_use, ax_tile_use_cbar) = plt.subplots(
                    3,
                    1,
                    figsize=get_figsize_qd(w_mode, domain),
                    gridspec_kw=grid_kws,
                )

                # # Log unrepaired env
                # unrepaired_env_int = metadata["unrepaired_env_int"]
                # print("\n".join(kiva_env_number2str(unrepaired_env_int)))
                # print()

                # Plot repaired env
                repaired_env_str = metadata["repaired_env_str"]
                print("\n".join(repaired_env_str))
                print()
                if domain in ["kiva"]:
                    repaired_env = kiva_env_str2number(repaired_env_str)
                    visualize_kiva(repaired_env, ax=ax_map, dpi=300)
                if domain == "sortation":
                    repaired_env = sortation_env_str2number(repaired_env_str)
                    visualize_kiva(repaired_env, ax=ax_map, dpi=300)
                elif domain == "manufacture":
                    repaired_env = manufacture_env_str2number(repaired_env_str)
                    visualize_manufacture(repaired_env, ax=ax_map, dpi=300)

                tile_usage = np.array(metadata["tile_usage"])
                if domain in ["kiva", "sortation"]:
                    env_h = gin.query_parameter("WarehouseManager.env_height")
                    env_w = gin.query_parameter("WarehouseManager.env_width")
                    if scenario == "KIVA":
                        ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if w_mode else KIVA_ROBOT_BLOCK_WIDTH
                        ADDITION_BLOCK_HEIGHT = 0 if w_mode else KIVA_ROBOT_BLOCK_HEIGHT
                        env_w += 2 * ADDITION_BLOCK_WIDTH
                        env_h += 2 * ADDITION_BLOCK_HEIGHT
                elif domain == "manufacture":
                    env_h = gin.query_parameter(
                        "ManufactureManager.env_height")
                    env_w = gin.query_parameter("ManufactureManager.env_width")

                if len(tile_usage.shape) == 1:
                    tile_usage = tile_usage[np.newaxis, ...]

                # mkdir for tileusage
                tile_usage_dir = logdir.dir("tile_usages")
                for i in range(tile_usage.shape[0]):
                    curr_tile_usage = tile_usage[i]
                    plot_heatmap(
                        curr_tile_usage,
                        env_h,
                        env_w,
                        fig,
                        ax_tile_use,
                        ax_tile_use_cbar,
                        logdir,
                        filenames=[
                            f"tile_usage/{index_0}_{index_1}-{i}.pdf",
                            f"tile_usage/{index_0}_{index_1}-{i}.svg",
                            f"tile_usage/{index_0}_{index_1}-{i}.png",
                        ],
                        dpi=dpi,
                    )

                # Plot chute mapping
                if search_space in [SearchSpace.CHUTE_MAPPING]:
                    chute_mapping = metadata.get("chute_mapping", None)
                    plot_chute_mapping(
                        repaired_env,
                        None,
                        chute_mapping,
                        chute_mapping_name=f"{index_0}_{index_1}",
                        save_dir=logdir.dir("chute_mapping", touch=True),
                        mapping=10,
                    )

                plt.close('all')


def main(
    logdir: str,
    logdir_type: str = "qd",  # "qd" or "sim"
    gen: int = None,
    index_0: int = None,
    index_1: int = None,
    dpi: int = 300,
    mode: str = None,
    domain: str = "kiva",
):
    if logdir_type == "qd":
        tile_usage_heatmap_from_qd(
            logdir=logdir,
            gen=gen,
            index_0=index_0,
            index_1=index_1,
            dpi=dpi,
            mode=mode,
            domain=domain,
        )
    elif logdir_type == "sim":
        tile_usage_heatmap_from_single_run(logdir, dpi=dpi, domain=domain)


if __name__ == "__main__":
    fire.Fire(main)
