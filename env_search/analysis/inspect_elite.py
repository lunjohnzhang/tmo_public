import fire
import gin
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from env_search.archives import GridArchive
from env_search.analysis.utils import (load_experiment, load_metrics,
                                       load_archive_gen,
                                       load_archive_from_reload,
                                       grid_archive_heatmap, get_extreme_pt)
from env_search.manufacture.module import cal_similarity_score
from env_search.utils import (manufacture_obj_types,
                              manufacture_env_number2str,
                              manufacture_env_str2number, format_env_str,
                              read_in_manufacture_map, n_params,
                              write_map_str_to_json)
from env_search.analysis.heatmap import post_process_figure
from env_search.analysis.visualize_env import visualize_kiva
from env_search.manager import SearchAlgo


def plot_archive(logdir, archive, filenames, vmax=0.5):
    # Plot the archive
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    dim_x, dim_y = gin.query_parameter("GridArchive.dims")
    ax.set_box_aspect(dim_x / dim_y)
    plot_kwargs = {
        "square": False,
        "cmap": "viridis",
        "pcm_kwargs": {
            # Looks much better in PDF viewers because the heatmap is not drawn
            # as individual rectangles. See here:
            # https://stackoverflow.com/questions/27092991/white-lines-in-matplotlibs-pcolor
            "rasterized": True,
        },
        "vmin": 0,
        "vmax": vmax,
        "plot_color_bar": True,
    }
    transpose_bcs = True
    grid_archive_heatmap(
        archive,
        ax,
        transpose_bcs=transpose_bcs,
        **plot_kwargs,
    )
    post_process_figure(
        ax,
        fig,
        kiva=False,
        maze=False,
        manufacture=True,
        heatmap_only=False,
        transpose_bcs=transpose_bcs,
    )

    for filename in filenames:
        fig.savefig(
            logdir.file(filename),
            dpi=300,
            bbox_inches='tight',
        )


def create_sim_score_archive(result_archive, station_same_weight, logdir):
    """Take elites from the result archive, calculate their similarity score,
    and resert into a new archive by their measures with the similarity score
    as the objective.
    """
    archive_type = str(gin.query_parameter("Manager.archive_type"))
    if archive_type == "@GridArchive":
        # Same manufacture as in Manager.
        # pylint: disable = no-value-for-parameter
        sim_score_archive = GridArchive(seed=42, dtype=np.float64)
    else:
        raise TypeError(f"Cannot handle archive type {archive_type}")

    sim_score_archive.new_history_gen()

    df_res = result_archive.as_pandas(include_solutions=True,
                                      include_metadata=True)
    for index, row in df_res.iterrows():
        meta = row["metadata"]["manufacture_metadata"]
        map_np_unrepaired = meta["unrepaired_env_int"]
        map_np_repaired = meta["repaired_env_int"]
        curr_sim_score = cal_similarity_score(
            map_np_unrepaired,
            map_np_repaired,
            station_same_weight,
        )
        measures = tuple(row.filter(regex=("measure_*")))
        # Dummy solutions, no metadata.
        sim_score_archive.add_single(
            np.zeros(result_archive._solution_dim),
            curr_sim_score,
            measures,
            None,
        )

    filenames = [
        "heatmap_sim_score.pdf",
        "heatmap_sim_score.png",
        "heatmap_sim_score.svg",
    ]

    plot_archive(logdir, sim_score_archive, filenames)


def elite_of_fixed_measure(
    logdir: str,
    is_cma_mae=True,
    min_obj=6.3,
    idx_0=0,
    gen=None,
):
    """Inspect elites in the result archive.
    """
    logdir = load_experiment(logdir)
    gen = load_metrics(logdir).total_itrs if gen is None else gen
    station_same_weight = gin.query_parameter(
        "ManufactureConfig.station_same_weight")
    # archive = load_archive_gen(logdir, gen)

    optimize_archive, result_archive = load_archive_from_reload(
        logdir,
        is_cma_mae=is_cma_mae,
    )
    plot_archive(logdir,
                 optimize_archive,
                 filenames=[
                     "heatmap_optimize.pdf",
                     "heatmap_optimize.png",
                     "heatmap_optimize.svg",
                 ],
                 vmax=8)
    create_sim_score_archive(result_archive, station_same_weight, logdir)
    df_res = result_archive.as_pandas(include_solutions=True,
                                      include_metadata=True)

    # Convert index to index_0 and index_1 --> pyribs 0.4.0/0.5.0 compatibility
    # issue
    if "index_0" not in df_res and "index" in df_res:
        all_grid_index = result_archive.int_to_grid_index(df_res["index"])
        df_res["index_0"] = all_grid_index[:, 0]
        df_res["index_1"] = all_grid_index[:, 1]

    # Get all elites in the result archive with throughput larger than
    # `min_obj`
    good_elites = df_res[df_res["objective"] > min_obj]

    # Find the one with the highest similarity score
    max_sim_score = -np.inf
    max_sim_elite_row = None
    for index, row in good_elites.iterrows():
        meta = row["metadata"]["manufacture_metadata"]
        map_np_unrepaired = meta["unrepaired_env_int"]
        map_np_repaired = meta["repaired_env_int"]
        curr_sim_score = cal_similarity_score(
            map_np_unrepaired,
            map_np_repaired,
            station_same_weight,
        )
        if curr_sim_score > max_sim_score:
            max_sim_elite_row = row
            max_sim_score = curr_sim_score

    elite_meta = max_sim_elite_row["metadata"]["manufacture_metadata"]
    elite_throughput = elite_meta["throughput"]
    elite_measures = tuple(max_sim_elite_row.filter(regex=("measure_*")))
    map_np_unrepaired = elite_meta["unrepaired_env_int"]
    map_np_repaired = elite_meta["repaired_env_int"]

    print(
        f"Obtained elite with throughoput {elite_throughput} and sim_score {max_sim_score}, measures {elite_measures}"
    )
    write_map_str_to_json(
        logdir.file("max_sim_elite_repaired.json"),
        manufacture_env_number2str(map_np_repaired),
        "max_sim_elite_repaired",
        "manufacture",
    )
    write_map_str_to_json(
        logdir.file("max_sim_elite_unrepaired.json"),
        manufacture_env_number2str(map_np_unrepaired),
        "max_sim_elite_unrepaired",
        "manufacture",
    )

    # Get the elites with idx_1 and maximum throughput.
    df_idx_0 = df_res[df_res["index_0"] == idx_0]
    idx_0_opt = df_idx_0.loc[df_idx_0["objective"].idxmax()]
    idx_0_opt_measures = tuple(idx_0_opt.filter(regex=("measure_*")))
    idx_0_opt_meta = idx_0_opt["metadata"]["manufacture_metadata"]
    idx_0_opt_map_np_repaired = idx_0_opt_meta["repaired_env_int"]
    idx_0_opt_map_np_unrepaired = idx_0_opt_meta["unrepaired_env_int"]
    idx_0_opt_throughput = idx_0_opt_meta["throughput"]
    print(
        f"Optimal env with idx_0={idx_0}, measures={idx_0_opt_measures}, throughput={idx_0_opt_throughput}"
    )
    write_map_str_to_json(
        logdir.file(f"max_idx_0={idx_0}_unrepaired.json"),
        manufacture_env_number2str(idx_0_opt_map_np_unrepaired),
        f"max_idx_0={idx_0}_unrepaired.json",
        "manufacture",
    )
    write_map_str_to_json(
        logdir.file(f"max_idx_0={idx_0}_repaired.json"),
        manufacture_env_number2str(idx_0_opt_map_np_repaired),
        f"max_idx_0={idx_0}_repaired.json",
        "manufacture",
    )


def elites_of_extreme_pt(
    logdir: str,
    surrogate: bool = True,
    gen: int = None,
):
    """Read in archive/surrogate archive and log the solutions in the extreme points of the measure space (max/min measures)

    Args:
        logdir (str): log dir path
        surrogate (bool, optional): if True, plot for surrogate archive.
            Defaults to True.
    """
    logdir = load_experiment(logdir)
    archive, _ = load_archive_from_reload(
        logdir,
        search_algo=SearchAlgo(gin.query_parameter("Manager.search_algo")),
    )
    gen = load_metrics(logdir).total_itrs if gen is None else gen
    if surrogate:
        df = pd.read_pickle(
            logdir.file(f"surrogate_archive/archive_{gen-1}.pkl"))
    else:
        df = pd.read_pickle(logdir.file(f"archive/archive_{gen}.pkl"))

    # Convert index to index_0 and index_1 --> pyribs 0.4.0/0.5.0 compatibility
    # issue
    if "index_0" not in df and "index" in df:
        all_grid_index = archive.int_to_grid_index(df["index"])
        df["index_0"] = all_grid_index[:, 0]
        df["index_1"] = all_grid_index[:, 1]

    # Create save dir
    savedir = logdir.dir("surrogate_archive_sols", touch=True)

    extreme_pts = get_extreme_pt(df)
    to_plots = df.iloc[extreme_pts]
    for _, to_plot in to_plots.iterrows():
        # obj = to_plot["objective"]
        index_0 = to_plot["index_0"]
        index_1 = to_plot["index_1"]
        metadata = to_plot["metadata"]["raw_metadata"]
        unrepaired_env_int = metadata["unrepaired_env_int"]
        repaired_env_int = metadata["repaired_env_int"]
        sim_score = metadata["similarity_score"]
        fig, (ax_unrepaired, ax_repaired) = plt.subplots(1, 2, figsize=(16, 8))
        visualize_kiva(unrepaired_env_int, ax=ax_unrepaired)
        visualize_kiva(repaired_env_int, ax=ax_repaired)
        ax_unrepaired.set_title("Unrepaired", fontsize=30)
        ax_repaired.set_title("Partially repaired", fontsize=30)
        fig.tight_layout()
        fig.savefig(f"{savedir}/{index_0}_{index_1}.png")


if __name__ == "__main__":
    fire.Fire({
        "fixed": elite_of_fixed_measure,
        "extreme": elites_of_extreme_pt,
    })
