import os
import json

import fire
import numpy as np
import yaml
import scipy.stats as st
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

THROUGHPUT = "throughput"
RUNTIME = "runtime"
SUCCESS = "success"
N_WAIT_PER_AG = "n_wait_per_ag"
SLEEP_TIME_PER_CHUTE = "sleep_time_per_chute"
SLEEP_COUNT_PER_CHUTE = "sleep_count_per_chute"
RECIRC_RATE = "recirc_rate"

STATS = [
    THROUGHPUT,
    RUNTIME,
    SUCCESS,
    N_WAIT_PER_AG,
    SLEEP_TIME_PER_CHUTE,
    SLEEP_COUNT_PER_CHUTE,
    RECIRC_RATE,
]

TO_PLOT = [
    THROUGHPUT,
    N_WAIT_PER_AG,
    SLEEP_TIME_PER_CHUTE,
    SLEEP_COUNT_PER_CHUTE,
    RECIRC_RATE,
]

STATS_TO_LABEL = {
    THROUGHPUT: "Throughput",
    N_WAIT_PER_AG: "Avg Num Waits Per Agent",
    SLEEP_TIME_PER_CHUTE: "Avg Sleep Time Per Chute",
    SLEEP_COUNT_PER_CHUTE: "Avg Sleep Count Per Chute",
    RECIRC_RATE: "Recirculation Rate",
}

TP_Y_MINMAX = {
    "33x36": (0, 12),
    "33x57": (0, 20),
    "33x57-105": (0, 20),
    "45x47": (0, 20),
    "57x58": (0, 20),
    "69x69": (0, 20),
    "81x80": (0, 20),
    "93x91": (0, 20),
    "50x86": (0, 30),
    "50x86-325": (0, 30),
    "140x500": (0, 55),
    "xxlarge": (0, 15),
}

N_WAIT_Y_MINMAX = {
    "33x57": (50, 1600),
    "33x57-105": (50, 1600),
    "50x86": (150, 1600),
    "50x86-325": (150, 1600),
}

RECIRC_RATE_Y_MINMAX = {
    "33x57": (0, 0.1),
    "33x57-105": (0, 0.16),
    "50x86": (0, 0.1),
    "50x86-325": (150, 1600),
}

SLEEP_TIME_Y_MINMAX = {
    "33x57": (50, 1600),
    "33x57-105": (50, 2200),
    "50x86": (150, 1600),
    "50x86-325": (150, 1600),
}

SLEEP_COUNT_Y_MINMAX = {
    "33x57": (0, 5),
    "33x57-105": (0, 10),
    "50x86": (0, 10),
    "50x86-325": (0, 10),
}


def add_item_to_dict_by_agent_num(to_dict, agent_num, element):
    if agent_num in to_dict:
        to_dict[agent_num].append(element)
    else:
        to_dict[agent_num] = [element]


def sort_and_get_vals_from_dict(the_dict):
    the_dict = sorted(the_dict.items())
    agent_nums = [agent_num for agent_num, _ in the_dict]
    all_vals = [vals for _, vals in the_dict]
    return agent_nums, all_vals


def compute_numerical(vals, all_success_vals):
    # Take the average, confidence interval and standard error
    all_vals = np.array(vals)
    all_success_vals = np.array(all_success_vals)
    assert all_vals.shape == all_success_vals.shape
    # breakpoint()
    mean_vals = np.mean(all_vals, axis=1)
    mean_vals_success = []
    sem_vals_success = []
    for i, curr_vals in enumerate(all_vals):
        # curr_vals = [x for x in curr_vals if x != 0]
        filtered_curr_vals = []
        for j, x in enumerate(curr_vals):
            if all_success_vals[i, j] == 1:
                filtered_curr_vals.append(x)
        mean_vals_success.append(np.mean(filtered_curr_vals))
        sem_vals_success.append(st.sem(filtered_curr_vals))

    cf_vals = st.t.interval(confidence=0.95,
                            df=all_vals.shape[1] - 1,
                            loc=mean_vals,
                            scale=st.sem(all_vals, axis=1) + 1e-8)
    sem_vals = st.sem(all_vals, axis=1)
    return mean_vals, cf_vals, sem_vals, mean_vals_success, sem_vals_success


def collect_results_one_setup(logdirs_plot: str):
    with open(os.path.join(logdirs_plot, "meta.yaml"), "r") as f:
        meta = yaml.safe_load(f)

    algo_name = meta["algorithm"]
    map_size = meta["map_size"]
    mode = meta["mode"]
    map_from = meta["map_from"]
    n_agents_opt = meta.get("n_agents_opt", None)

    all_stats = {stat: {} for stat in STATS}

    # y_min, y_max = MAP_SIZE_TO_Y_MINMAX[map_size]

    for logdir_f in os.listdir(logdirs_plot):
        logdir = os.path.join(logdirs_plot, logdir_f)
        if not os.path.isdir(logdir):
            continue
        results_dir = os.path.join(logdir, "results")
        # agent_nums = []
        # throughputs = []
        for sim_dir in os.listdir(results_dir):
            sim_dir_comp = os.path.join(results_dir, sim_dir)
            config_file = os.path.join(sim_dir_comp, "config.json")
            result_file = os.path.join(sim_dir_comp, "result.json")

            if os.path.exists(config_file) and os.path.exists(result_file):

                with open(config_file, "r") as f:
                    config = json.load(f)

                with open(result_file, "r") as f:
                    result = json.load(f)

                congested = result[
                    "congested"] if "congested" in result else False
                agent_num = config[
                    "agentNum"] if "agentNum" in config else config[
                        "num_agents"]

                # Number of wait actions
                if "vertex_wait_matrix" in result:
                    n_wait_per_ag = np.sum(
                        result["vertex_wait_matrix"]) / agent_num
                else:
                    n_wait_per_ag = 0

                # Only consider the uncongested simulations
                throughput = result["throughput"]  # if not congested else 0
                runtime = result.get("cpu_runtime",
                                     0)  # if not congested else 0
                success = 1 if not congested else 0
                # agent_nums.append(agent_num)
                # throughputs.append(throughput)

                # Sleep count per chute
                chute_sleep_count = result.get("chute_sleep_count",
                                               np.zeros((2, 2)))
                chute_sleep_count = np.array(chute_sleep_count)
                sleep_count_per_chute = np.mean(chute_sleep_count[:, 1])

                # Total sleep time per chute
                total_chute_sleep_time = result.get("total_chute_sleep_time",
                                                    np.zeros((2, 2)))
                total_chute_sleep_time = np.array(total_chute_sleep_time)
                sleep_time_per_chute = np.mean(total_chute_sleep_time[:, 1])

                recirc_rate = result.get("recirc_rate", 0)

                add_item_to_dict_by_agent_num(
                    all_stats[THROUGHPUT],
                    agent_num,
                    throughput,
                )
                add_item_to_dict_by_agent_num(
                    all_stats[RUNTIME],
                    agent_num,
                    runtime,
                )
                add_item_to_dict_by_agent_num(
                    all_stats[SUCCESS],
                    agent_num,
                    success,
                )
                add_item_to_dict_by_agent_num(
                    all_stats[N_WAIT_PER_AG],
                    agent_num,
                    n_wait_per_ag,
                )
                add_item_to_dict_by_agent_num(
                    all_stats[SLEEP_TIME_PER_CHUTE],
                    agent_num,
                    sleep_time_per_chute,
                )
                add_item_to_dict_by_agent_num(
                    all_stats[SLEEP_COUNT_PER_CHUTE],
                    agent_num,
                    sleep_count_per_chute,
                )
                add_item_to_dict_by_agent_num(
                    all_stats[RECIRC_RATE],
                    agent_num,
                    recirc_rate,
                )

            else:
                print(f"Result of {sim_dir} is missing")

    # Nothing in the logdir, skip
    if not all_stats[THROUGHPUT]:
        return None

    # Assuming success rate of always included
    agent_nums, success_vals = sort_and_get_vals_from_dict(all_stats[SUCCESS])

    success_vals = np.array(success_vals)
    success_rates = np.sum(success_vals, axis=1) / success_vals.shape[1]

    numerical_result = {}
    numerical_result["agent_num"] = agent_nums
    numerical_result[f"success_rate"] = success_rates

    # all_stats_vals = {}
    for stat in STATS:
        _, vals = sort_and_get_vals_from_dict(all_stats[stat])
        # Compute numerical and add to the dictionary
        mean, cf, sem, mean_success, sem_success = compute_numerical(
            vals, success_vals)
        # Create numerical result
        numerical_result[f"mean_{stat}"] = mean
        numerical_result[f"mean_{stat}_success"] = mean_success
        numerical_result[f"cf_lb_{stat}"] = cf[0]
        numerical_result[f"cf_ub_{stat}"] = cf[1]
        numerical_result[f"sem_{stat}"] = sem
        numerical_result[f"sem_{stat}s_success"] = sem_success

    numerical_result_df = pd.DataFrame(numerical_result)
    numerical_result_df.to_csv(os.path.join(logdirs_plot, "numerical.csv"))


def collect_eval_results(all_logdirs_plot: str):
    for logdirs_plot_dir in os.listdir(all_logdirs_plot):
        logdirs_plot = os.path.join(all_logdirs_plot, logdirs_plot_dir)
        if not os.path.isdir(logdirs_plot):
            continue
        collect_results_one_setup(logdirs_plot)


# from env_search.analysis.throughput_vs_n_agents import throughput_vs_n_agents

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def collect_compare_eval_results(
    all_logdirs_plot: str,
    ax=None,
    add_legend=True,
    save_fig=True,
):
    collect_eval_results(all_logdirs_plot)
    plot_eval_results(all_logdirs_plot, ax, add_legend, save_fig)


def plot_eval_results(
    all_logdirs_plot,
    ax=None,
    add_legend=True,
    save_fig=True,
):
    min_agent_num = np.inf
    max_agent_num = -1

    all_axs = [plt.subplots(1, 1, figsize=(25, 10)) for _ in TO_PLOT]

    # if ax is None:
    #     fig, ax =
    for i, logdirs_plot_dir in enumerate(sorted(os.listdir(all_logdirs_plot))):
        logdirs_plot = os.path.join(all_logdirs_plot, logdirs_plot_dir)
        if not os.path.isdir(logdirs_plot):
            continue

        # Skip if there is no numerical.csv
        result_csv = os.path.join(logdirs_plot, "numerical.csv")
        if not os.path.exists(result_csv):
            continue

        with open(os.path.join(logdirs_plot, "meta.yaml"), "r") as f:
            meta = yaml.safe_load(f)

        result_df = pd.read_csv(result_csv)

        for i, stat in enumerate(TO_PLOT):
            fig, ax = all_axs[i]
            color = ax._get_lines.get_next_color()
            mean_stat = result_df[f"mean_{stat}"].to_numpy()
            cf_lb = result_df[f"cf_lb_{stat}"].to_numpy()
            cf_ub = result_df[f"cf_ub_{stat}"].to_numpy()
            agent_nums = result_df["agent_num"].to_numpy()

            ax.plot(
                agent_nums,
                mean_stat,
                marker=".",
                color=color,
                label=meta["map_from"],
            )
            ax.fill_between(
                agent_nums,
                cf_lb,
                cf_ub,
                alpha=0.5,
                color=color,
            )

        if agent_nums[-1] > max_agent_num:
            max_agent_num = agent_nums[-1]
        if agent_nums[0] < min_agent_num:
            min_agent_num = agent_nums[0]
    # ax.set_xlim(agent_nums[0], agent_nums[-1])

    # Set plot properties and save fig
    y_min, y_max = None, None
    for i, stat in enumerate(TO_PLOT):
        fig, ax = all_axs[i]
        if stat == THROUGHPUT:
            y_min, y_max = TP_Y_MINMAX[meta["map_size"]]
        elif stat == N_WAIT_PER_AG:
            y_min, y_max = N_WAIT_Y_MINMAX[meta["map_size"]]
        elif stat == RECIRC_RATE:
            y_min, y_max = RECIRC_RATE_Y_MINMAX[meta["map_size"]]
        elif stat == SLEEP_COUNT_PER_CHUTE:
            y_min, y_max = SLEEP_COUNT_Y_MINMAX[meta["map_size"]]
        elif stat == SLEEP_TIME_PER_CHUTE:
            y_min, y_max = SLEEP_TIME_Y_MINMAX[meta["map_size"]]
        ax.set_ylabel(STATS_TO_LABEL[stat], fontsize=45)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Number of Agents", fontsize=45)
        # ax.grid()
        # ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.tick_params(axis='both', which='minor', labelsize=15)

        # Add a vertical line at the number of agents used to optimize the layout
        map_size = meta["map_size"]
        mode = meta["mode"]

        n_agent_vertical = None
        if map_size == "33x36":
            n_agent_vertical = [600]

        # Draw vertical line
        if n_agent_vertical is not None:
            for v in n_agent_vertical:
                ax.axvline(x=v, color="black", linewidth=2)

        range_x = [min_agent_num, max_agent_num]
        range_y = [y_min, y_max]
        if map_size in ["small", "medium"]:
            ax.set_xticks([
                range_x[0],
                np.mean([range_x]),
                n_agent_vertical,
                range_x[1],
            ])
            ax.set_xticklabels(
                [
                    range_x[0],
                    np.mean([range_x], dtype=int),
                    n_agent_vertical,
                    range_x[1],
                ],
                rotation=0,
                fontsize=35,
            )
        elif map_size == "large":
            ax.set_xticks([
                range_x[0],
                n_agent_vertical,
                range_x[1],
            ])
            ax.set_xticklabels(
                [
                    range_x[0],
                    n_agent_vertical,
                    range_x[1],
                ],
                rotation=0,
                fontsize=35,
            )
        else:
            range_x_mid = np.mean(range_x)
            if range_x_mid.is_integer():
                range_x_mid = int(range_x_mid)
            ax.set_xticks([range_x[0], range_x_mid, range_x[1]])
            ax.set_xticklabels(
                [range_x[0], range_x_mid, range_x[1]],
                fontsize=35,
            )
        range_y_mid = np.mean(range_y)
        if range_y_mid.is_integer():
            range_y_mid = int(range_y_mid)
        ax.set_yticks([range_y[0], range_y_mid, range_y[1]])
        ax.set_yticklabels(
            [range_y[0], range_y_mid, range_y[1]],
            fontsize=35,
        )

        # Legend
        if add_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(
                handles,
                labels,
                loc="lower left",
                ncol=4,
                fontsize=30,
                mode="expand",
                bbox_to_anchor=(0, 1.02, 1, 0.2),  # for ncols=2
                # borderaxespad=0,)
            )

            # For front fig
            # order = [1, 0]
            # ax.legend(
            #     [handles[idx] for idx in order],
            #     [labels[idx] for idx in order],
            #     # loc="lower left",
            #     ncol=1,
            #     fontsize=35,
            #     frameon=False,
            #     # mode="expand",
            #     # bbox_to_anchor=(0, 1.02, 1, 0.2),  # for ncols=2
            #     # borderaxespad=0,)
            # )

            # For r-mode less agents
            # legend = ax.legend(
            #     handles,
            #     labels,
            #     # loc="lower left",
            #     ncol=1,
            #     fontsize=35,
            #     borderaxespad=0,
            #     bbox_to_anchor=(1.04, 1),  # for ncols=2
            # )

            for line in legend.get_lines():
                line.set_linewidth(4.0)

        if save_fig:
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    all_logdirs_plot,
                    f"{stat}_agent_num_{mode}_{map_size}.pdf",
                ),
                dpi=300,
                bbox_inches='tight',
            )

            fig.savefig(
                os.path.join(
                    all_logdirs_plot,
                    f"{stat}_agent_num_{mode}_{map_size}.png",
                ),
                dpi=300,
                bbox_inches='tight',
            )


def convert_float(number):
    return "%.2f" % number


def table(manifest_file):
    with open(manifest_file, "r") as f:
        manifest = yaml.safe_load(f)

    table_data = [
        "\\toprule",
        # "Setup & TMO & Throughput & Recirc Rate & CPU Runtime (s) \\\\",
        "Setup & TMO & Throughput & Recirc Rate \\\\",
        "\\midrule",
    ]
    n_columns = len(table_data[1].split("&"))

    for i, setup_name in enumerate(manifest["experiment_setups"]):
        setup = manifest["experiment_setups"][setup_name]
        dir_path = setup["dir"]
        n_agent_show = setup["n_agent_show"]
        row_data_show = []
        all_success = []
        all_throughput_mean = []
        all_runtime_mean = []
        all_throughput_sem = []
        all_runtime_sem = []
        all_labels = []
        all_recirc_rate = []
        all_recirc_rate_sem = []
        for exp in os.listdir(dir_path):
            exp_full = os.path.join(dir_path, exp)
            if not os.path.isdir(exp_full):
                continue
            exp_meta_file = os.path.join(exp_full, "meta.yaml")
            with open(exp_meta_file, "r") as f:
                meta = yaml.safe_load(f)
            algo = meta["algorithm"]
            map_from = meta["map_from"]
            map_size = meta["map_size"]
            mode = meta["mode"]
            data = pd.read_csv(os.path.join(exp_full, f"numerical.csv"))
            data_row = data[data["agent_num"] == n_agent_show]
            throughput_mean = convert_float(
                data_row["mean_throughput"].iloc[0])
            throughput_sem = convert_float(
                data_row["sem_throughputs_success"].iloc[0])
            runtime_mean = convert_float(
                data_row["mean_runtime_success"].iloc[0])
            runtime_sem = convert_float(
                data_row["sem_runtimes_success"].iloc[0])
            success = int(data_row["success_rate"].iloc[0] * 100)
            recirc_rate = convert_float(
                data_row["mean_recirc_rate_success"].iloc[0] * 100)
            recirc_rate_sem = convert_float(
                data_row["sem_recirc_rates_success"].iloc[0] * 100)
            label = f"{map_from}"
            row_data_show.append([
                " ",  # first row is empty to place the multirow setup index
                label,
                f"${throughput_mean} \pm {throughput_sem}$"
                if success > 0 else "N/A",
                f"${recirc_rate}\% \pm {recirc_rate_sem}\%$"
                if success > 0 else "N/A",
                # f"${runtime_mean} \pm {runtime_sem}$"
                # if success > 0 else "N/A",
            ])

            # Add data for comparison
            all_labels.append(label)
            all_success.append(success)
            all_throughput_mean.append(float(throughput_mean))
            all_throughput_sem.append(float(throughput_sem))
            all_runtime_mean.append(float(runtime_mean))
            all_runtime_sem.append(float(runtime_sem))
            all_recirc_rate.append(float(recirc_rate))
            all_recirc_rate_sem.append(float(recirc_rate_sem))

        # Bold the results that are the best
        best_success = np.argmax(all_success)
        best_throughput = np.nanargmax(all_throughput_mean)
        best_runtime = np.nanargmin(all_runtime_mean)
        best_reirc_rate = np.nanargmin(all_recirc_rate)

        row_data_show[best_throughput][
            1] = f"\\textbf{{{all_labels[best_throughput]}}}"

        row_data_show[best_throughput][
            2] = f"$\\textbf{{{all_throughput_mean[best_throughput]}}} \pm \\textbf{{{all_throughput_sem[best_throughput]}}}$" if all_success[
                best_throughput] > 0 else "N/A"
        row_data_show[best_reirc_rate][
            3] = f"$\\textbf{{{all_recirc_rate[best_reirc_rate]}\%}} \pm \\textbf{{{all_recirc_rate_sem[best_reirc_rate]}\%}}$"
        # row_data_show[best_runtime][
        #     4] = f"$\\textbf{{{all_runtime_mean[best_runtime]}}} \pm \\textbf{{{all_runtime_sem[best_runtime]}}}$" if all_success[
        #         best_runtime] > 0 else "N/A"

        # We have gotten the data for one setup.
        n_exps = len(row_data_show)

        # sort entries based on label.
        label_row_data_show = list(zip(all_labels, row_data_show))
        label_row_data_show.sort(key=lambda i: i[0])
        row_data_show = [row for _, row in label_row_data_show]

        # row_data_show.sort(key=lambda i: i[1])
        row_data_show[0][0] = f"\\multirow{{{n_exps}}}{{*}}{{{i+1}}}"

        for row in row_data_show:
            table_data.append(str(" & ".join(row) + "\\\\"))
        table_data.append("\\midrule")

    table_data[-1] = "\\bottomrule"
    with open("table.txt", "w") as f:
        f.writelines(line + '\n' for line in table_data)


if __name__ == "__main__":
    fire.Fire({
        "collect": collect_eval_results,
        "comparison": plot_eval_results,
        "collect_comparison": collect_compare_eval_results,
        "table": table,
    })
