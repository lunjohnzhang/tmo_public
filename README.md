# Destination-to-Chutes Task Mapping Optimization for Multi-Robot Coordination in Robotic Sorting Systems

This repository is the official implementation of **Destination-to-Chutes Task Mapping Optimization for Multi-Robot Coordination in Robotic Sorting Systems**. The repository builds on top of the repository of [Arbitrarily Scalable Environment Generators via Neural Cellular Automata](https://github.com/lunjohnzhang/warehouse_env_gen_nca_public).

## Installation

This is a hybrid C++/Python project. The simulation environment is written in C++ and the rests are in Python. We use [pybind11](https://pybind11.readthedocs.io/en/stable/) to bind the two languages.

1. **Initialize pybind11:** After cloning the repo, initialize the pybind11 submodule

   ```bash
   git submodule init
   git submodule update
   ```

1. **Install Singularity:** All of our code runs in a Singularity container.
   Singularity is a container platform (similar in many ways to Docker). Please
   see the instructions
   [here](https://sylabs.io/singularity/) for installing SingularityCE. As a reference, we use version 3.11.1.


2. **Install CPLEX:** CPLEX is used for repairing the generated warehouse maps.

   1. Download the free academic version [here](https://www.ibm.com/products/ilog-cplex-optimization-studio).
   2. Download the installation file for Linux.
   3. Follow this [guide](https://www.ibm.com/docs/en/icos/22.1.0?topic=2210-installing-cplex-optimization-studio) to install it. Basically:

   ```
   chmod u+x INSTALLATION_FILE
   ./INSTALLATION_FILE
   ```

   During installation, set the installation directory to `CPLEX_Studio2210/` in the repo.

3. **Build Singularity container:** Run the provided script to build the container. Note that this need `sudo` permission on your system.
   ```
   bash build_container.sh
   ```
   The script will first build a container as a sandbox, compile the C++ simulator, then convert that to a regular `.sif` Singularity container.

### Troubleshooting

1. If you encounter the following error while running experiments in the Singularity container:

   ```
   container creation failed: mount /proc/self/fd/3->/usr/local/var/singularity/mnt/session/rootfs error: while mounting image`/proc/self/fd/3: failed to find loop device: no loop devices available
   ```

   Please try downgrading/upgrading the Linux kernel version to `5.15.0-67-generic`, as suggested in [this Github issue](https://github.com/sylabs/singularity/issues/1499).

## Optimizing Task Mappings

### Training Logging Directory Manifest

Regardless of where the script is run, the log files and results are placed in a
logging directory in `logs/`. The directory's name is of the form
`%Y-%m-%d_%H-%M-%S_<dashed-name>_<uuid>`, e.g.
`2020-12-01_15-00-30_experiment-1_ff1dcb2b`. Inside each directory are the
following files:

```text
- config.gin  # All experiment config variables, lumped into one file.
- seed  # Text file containing the seed for the experiment.
- reload.pkl  # Data necessary to reload the experiment if it fails.
- reload_em.pkl  # Pickle data for EmulationModel.
- reload_em.pth  # PyTorch models for EmulationModel.
- metrics.json  # Data for a MetricLogger with info from the entire run, e.g. QD score.
- hpc_config.sh  # Same as the config in the Slurm dir, if Slurm is used.
- archive/  # Snapshots of the full archive, including solutions and metadata,
            # in pickle format.
- archive_history.pkl  # Stores objective values and behavior values necessary
                       # to reconstruct the archive. Solutions and metadata are
                       # excluded to save memory.
- dashboard_status.txt  # Job status which can be picked up by dashboard scripts.
                        # Only used during execution.
- evaluations # Output logs of LMAPF simulator
```

### Running Locally

#### Single Run

To run one experiment locally, use:

```bash
bash scripts/run_local.sh CONFIG SEED NUM_WORKERS
```

For instance, with 4 workers:

```bash
bash scripts/run_local.sh config/foo.gin 42 4
```

`CONFIG` is the [gin](https://github.com/google/gin-config) experiment config
for `env_search/main.py`.

### Running on Slurm

Use the following command to run an experiment on an HPC with Slurm (and
Singularity) installed:

```bash
bash scripts/run_slurm.sh CONFIG SEED HPC_CONFIG
```

For example:

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh
```

`CONFIG` is the experiment config for `env_search/main.py`, and `HPC_CONFIG` is a shell
file that is sourced by the script to provide configuration for the Slurm
cluster. See `config/hpc` for example files.

Once the script has run, it will output commands like the following:

- `tail -f ...` - You can use this to monitor stdout and stderr of the main
  experiment script. Run it.
- `bash scripts/slurm_cancel.sh ...` - This will cancel the job.

### Reloading

While the experiment is running, its state is saved to `reload.pkl` in the
logging directory. If the experiment fails, e.g. due to memory limits, time
limits, or network connection issues, `reload.pkl` may be used to continue the
experiment. To do so, execute the same command as before, but append the path to
the logging directory of the failed experiment.

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh -r logs/.../
```

The experiment will then run to completion in the same logging directory. This
works with `scripts/run_local.sh` too.

## Reproducing Paper Results

The `config/` directory contains the config files required to run the experiments shown in the paper.

| Config file                                                                  | Map                 | Num Chutes | Num Destinations (including recirculation) | Num Agents | Method                      | Optimized Task Mapping                                                                       |
| ---------------------------------------------------------------------------- | ------------------- | ---------- | ------------------------------------------ | ---------- | --------------------------- | -------------------------------------------------------------------------------------------- |
| config/Genetic_Algo_600_agents_sortation-33-57_sim-5000.gin                  | sortation-33-57-253 | 253        | 100                                        | 600        | EA                          | chute_mapping/optimized/sortation-33-57_ga_100-destinations_600-agents.json                  |
| config/Genetic_Algo_Plus_600_agents_sortation-33-57_sim-5000.gin             | sortation-33-57-253 | 253        | 100                                        | 600        | EA w/ Greedy Initialization | chute_mapping/optimized/sortation-33-57_ga-plus_100-destinations_600-agents.json             |
| config/Genetic_Algo_600_agents_sortation-33-57-105-chutes_sim-5000.gin       | sortation-33-57-105 | 105        | 42                                         | 600        | EA                          | chute_mapping/optimized/sortation-33-57-105-chutes_ga_100-destinations_600-agents.json       |
| config/Genetic_Algo_Plus_600_agents_sortation-33-57-105-chutes_sim-5000.gin  | sortation-33-57-105 | 105        | 42                                         | 600        | EA w/ Greedy Initialization | chute_mapping/optimized/sortation-33-57-105-chutes_ga-plus_100-destinations_600-agents.json  |
| config/Genetic_Algo_1200_agents_sortation-50-86_sim-5000.gin                 | sortation-50-86-703 | 703        | 300                                        | 1200       | EA                          | chute_mapping/optimized/sortation-50-86_ga_100-destinations_1200-agents.json                 |
| config/Genetic_Algo_Plus_1200_agents_sortation-50-86_sim-5000.gin            | sortation-50-86-703 | 703        | 300                                        | 1200       | EA w/ Greedy Initialization | chute_mapping/optimized/sortation-50-86_ga-plus_100-destinations_1200-agents.json            |
| config/Genetic_Algo_1200_agents_sortation-50-86-325-chutes_sim-5000.gin      | sortation-50-86-325 | 325        | 139                                        | 1200       | EA                          | chute_mapping/optimized/sortation-50-86-325-chutes_ga_100-destinations_1200-agents.json      |
| config/Genetic_Algo_Plus_1200_agents_sortation-50-86-325-chutes_sim-5000.gin | sortation-50-86-325 | 325        | 139                                        | 1200       | EA w/ Greedy Initialization | chute_mapping/optimized/sortation-50-86-325-chutes_ga-plus_100-destinations_1200-agents.json |

## Baseline Task Mappings

| Baseline Task Mapping                                                                    | Map                 | Num Chutes | Num Destinations | Num Agents | Method          |
| ---------------------------------------------------------------------------------------- | ------------------- | ---------- | ---------------- | ---------- | --------------- |
| chute_mapping/baselines/sortation_33_57_heuristic_baseline_algo=cluster.json             | sortation-33-57-253 | 253        | 100              | 600        | Cluster Greedy  |
| chute_mapping/baselines/sortation_50_86-325-chutes_heuristic_baseline_algo=min_dist.json | sortation-33-57-253 | 253        | 100              | 600        | Min-dist Greedy |
| chute_mapping/baselines/sortation_33_57_105-chutes_heuristic_baseline_algo=cluster.json  | sortation-33-57-105 | 105        | 42               | 600        | Cluster Greedy  |
| chute_mapping/baselines/sortation_33_57_105-chutes_heuristic_baseline_algo=min_dist.json | sortation-33-57-105 | 105        | 42               | 600        | Min-dist Greedy |
| chute_mapping/baselines/sortation_50_86_heuristic_baseline_algo=cluster.json             | sortation-50-86-703 | 703        | 300              | 1200       | Cluster Greedy  |
| chute_mapping/baselines/sortation_50_86_heuristic_baseline_algo=min_dist.json            | sortation-50-86-703 | 703        | 300              | 1200       | Min-dist Greedy |
| chute_mapping/baselines/sortation_50_86-325-chutes_heuristic_baseline_algo=cluster.json  | sortation-50-86-325 | 325        | 139              | 1200       | Cluster Greedy  |
| chute_mapping/baselines/sortation_50_86-325-chutes_heuristic_baseline_algo=min_dist.json | sortation-50-86-325 | 325        | 139              | 1200       | Min-dist Greedy |
