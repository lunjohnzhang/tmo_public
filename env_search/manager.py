"""Provides a class for running each QD algorithm."""
import dataclasses
import itertools
import logging
import pickle as pkl
import time
from typing import Callable, List, Tuple

import cloudpickle
import gin
import numpy as np
import wandb

from dask.distributed import Client
from logdir import LogDir
from ribs.archives import ArchiveBase, Elite

from env_search.archives import GridArchive
from env_search.emitters import EvolutionStrategyEmitter, MapElitesBaselineWarehouseEmitter, MapElitesBaselineManufactureEmitter, MapElitesBaselineMazeEmitter, RandomEmitter, IsoLineEmitter
from env_search.schedulers import Scheduler
from env_search.utils.logging import worker_log
from env_search.utils.metric_logger import MetricLogger
from env_search.utils.enum import SearchAlgo
from env_search.warehouse import QUAD_TASK_ASSIGN_N_PARAM
from env_search.warehouse.warehouse_manager import WarehouseManager
from env_search.warehouse.config import WarehouseConfig
from env_search.maze.maze_manager import MazeManager
from env_search.manufacture.manufacture_manager import ManufactureManager
from env_search.utils.enum import SearchSpace

# Just to get rid of pylint warning about unused import (adding a comment after
# each line above messes with formatting).
IMPORTS_FOR_GIN = (
    GridArchive,
    WarehouseManager,
    MazeManager,
    ManufactureManager,
)

EMITTERS_WITH_RESTARTS = (
    EvolutionStrategyEmitter,
    MapElitesBaselineWarehouseEmitter,
    MapElitesBaselineManufactureEmitter,
    MapElitesBaselineMazeEmitter,
    RandomEmitter,
    IsoLineEmitter,
)

logger = logging.getLogger(__name__)


@gin.configurable
class Manager:  # pylint: disable = too-many-instance-attributes
    """Runs an (emulation model) QD algorithm on distributed compute.

    If you are trying to understand this code, first refer to how the general
    pyribs loop works (https://pyribs.org). Essentially, the execute() method of
    this class runs this loop but in a more complicated fashion, as we want to
    distribute the solution evaluations, log various performance metrics, save
    various pieces of data, support reloading / checkpoints, etc.

    Main args:
        client: Dask client for distributed compute.
        logdir: Directory for saving all logging info.
        seed: Master seed. The seed is not passed in via gin because it needs to
            be flexible.
        reload: If True, reload the experiment from the given logging directory.
        env_manager_class: This class calls a separate manager based on the
            environment, such as MazeManager. Pass this class using this
            argument.

    Algorithm args:
        search_algo: Search algorithm, one of ["classic", "em", "extinct"]
            classic: runs classic QD search with not emulation model
            em: run DSAGE with emulation model (or surrogate model)
            extinct: runs classic QD search to optimize inner objective in
                inner loop, outer objective in outer loop
        max_evals: Total number of evaluations of the true objective.
        initial_sols: Number of initial solutions to evaluate.
        inner_itrs: Number of times to run the inner loop.
        archive_type: Archive class for both main and emulation archives.
            Intended for gin configuration.
        sol_size: Size of the solution that the emitter should emit and the
            archive should store.
        emitter_types: List of tuples of (class, n); where each tuple indicates
            there should be n emitters with the given class. If search algo is
            "em" or "extinct", these emitters are only used in the inner loop;
            otherwise, they are maintained for the entire run. Intended for gin
            configuration.
        num_elites_to_eval: Number of elites in the surrogate archive to
            evaluate. Pass None to evaluate all elites. (default: None)
        random_sample_em: True if num_elites_to_eval should be selected
            randomly. If num_elites_to_eval is None, this argument is
            ignored. (default: False)
        downsample_surrogate: Whether to downsample the surrogate archive for
            EM and EXTINCT search.
        downsample_archive_type: Archive type for downsampling. Used for Gin.
        extinct_inner_archive_type: Archive type for inner loop of extinct
            search.
        extinct_p: portion of elites to select from archive in outer loop for
            extinct search

    Logging args:
        archive_save_freq: Number of outer itrs to wait before saving the full
            archive (i.e. including solutions and metadata). Set to None to
            never save (the archive will still be available in the reload file).
            Set to -1 to only save on the final iter.
        save_surrogate_archive: Whether to save surrogate archive or not.
        reload_save_freq: Number of outer itrs to wait before saving
            reload data.
        plot_metrics_freq: Number of outer itrs to wait before displaying text
            plot of metrics. Plotting is not expensive, but the output can be
            pretty large.
    """

    def __init__(
        self,
        ## Main args ##
        client: Client,
        logdir: LogDir,
        seed: int,
        reload: bool = False,
        env_manager_class: Callable = gin.REQUIRED,
        ## Algorithm args ##
        search_algo: str = gin.REQUIRED,
        is_cma_mae: bool = gin.REQUIRED,
        max_evals: int = gin.REQUIRED,
        initial_sols: int = gin.REQUIRED,
        inner_itrs: int = gin.REQUIRED,
        archive_type: Callable = gin.REQUIRED,
        initial_mean: int = 0,
        # sol_size: int = gin.REQUIRED,
        emitter_types: List[Tuple] = gin.REQUIRED,
        num_elites_to_eval: int = None,
        random_sample_em: bool = False,
        downsample_surrogate: bool = False,
        downsample_archive_type: Callable = None,
        extinct_inner_archive_type: Callable = None,
        extinct_p: float = 0.2,
        # Cooperative coevolution args
        cc_search_spaces: List[str] = None,
        cc_per_iter_evals: int = 1000,
        ## Logging args ##
        archive_save_freq: int = None,
        save_surrogate_archive: bool = True,
        reload_save_freq: int = 5,
        plot_metrics_freq: int = 5,
        wandb_mode: str = "offline",
    ):  # pylint: disable = too-many-arguments, too-many-branches

        # Main.
        self.client = client
        self.logdir = logdir

        # Algorithm.
        self.search_algo = SearchAlgo(search_algo)
        self.is_cma_mae = is_cma_mae
        self.max_evals = max_evals
        self.inner_itrs = inner_itrs
        self.initial_sols = initial_sols
        self.archive_type = archive_type
        self.initial_mean = initial_mean
        # self.sol_size = sol_size
        self.emitter_types = emitter_types
        self.n_emitters = np.sum([x[1] for x in emitter_types], dtype=int)
        self.num_elites_to_eval = num_elites_to_eval
        self.random_sample_em = random_sample_em
        self.downsample_surrogate = downsample_surrogate
        self.downsample_archive_type = downsample_archive_type
        self.extinct_inner_archive_type = extinct_inner_archive_type
        self.extinct_p = extinct_p
        self.result_archive = None
        self.env_manager_class = env_manager_class

        # Logging.
        self.archive_save_freq = archive_save_freq
        self.save_surrogate_archive = save_surrogate_archive
        self.reload_save_freq = reload_save_freq
        self.plot_metrics_freq = plot_metrics_freq
        self.wandb_mode = wandb_mode

        # Set up the environment manager.
        # NOTE: For CC, we need to reinitialize the environment manager in
        # every iteration.
        if self.search_algo != SearchAlgo.CC:
            self.env_manager = env_manager_class(self.client, self.logdir)
            # Get solution space
            self.sol_size = self.env_manager.get_sol_size()
            logger.info(f"Solution space size: {self.sol_size}")
            self.env_manager.seed = seed
        # Running CC algorithm
        else:
            assert cc_search_spaces is not None
            # Create a list of env managers for each search space
            self.cc_search_spaces = [SearchSpace(s) for s in cc_search_spaces]
            self.env_managers = [
                env_manager_class(self.client, self.logdir, search_space=space)
                for space in cc_search_spaces
            ]
            # Sum up the search space of the sub problems to get the solution
            # space
            self.sol_size = np.sum(
                [env.get_sol_size() for env in self.env_managers])
            for env_m in self.env_managers:
                env_m.seed = seed
            # Number of CC iteration.
            # Total should be max_evals // cc_per_iter_evals
            self.cc_n_iter = 0
            # Number of evaluations per CC iteration
            self.cc_per_iter_evals = cc_per_iter_evals
            # Number of finished evaluations in each CC iteration
            self.cc_curr_iter_evals = 0
            # current search space of CC
            self.cc_curr_search_space = None
            # Current context vector of CC
            self.cc_context_vector = None
        # Remember master seed
        self.seed = seed

        # The attributes below are either reloaded or created fresh. Attributes
        # added below must be added to the _save_reload_data() method.
        if not reload:
            logger.info("Setting up fresh components")
            self.rng = np.random.default_rng(seed)
            self.outer_itrs_completed = 0
            self.evals_used = 0

            metric_list = [
                ("Total Evals", True),
                ("Mean Evaluation", False),
                ("Actual QD Score", True),
                ("Archive Size", True),
                ("Archive Coverage", True),
                ("Best Objective", False),
                ("Worst Objective", False),
                ("Mean Objective", False),
                ("Overall Min Objective", False),
                ("Similarity Score of Env w/ Best Objective", False),
            ]

            self.metrics = MetricLogger(metric_list)
            self.total_evals = 0
            self.overall_min_obj = np.inf

            self.metadata_id = 0
            self.cur_best_id = None  # ID of most recent best solution.

            self.failed_levels = []

            if self.search_algo == SearchAlgo.EM:
                logger.info("Setting up emulation model and archive")
                # Archive must be initialized since there is no scheduler.
                self.env_manager.em_init(seed)
                self.archive: ArchiveBase = archive_type(
                    solution_dim=self.sol_size,
                    seed=seed,
                    dtype=np.float64,
                )
                logger.info("Archive: %s", self.archive)
            elif self.search_algo == SearchAlgo.CLASSIC:
                logger.info("Setting up scheduler for classic pyribs")
                _, self.scheduler = self.build_emitters_and_scheduler(
                    archive_type(solution_dim=self.sol_size,
                                 seed=seed,
                                 dtype=np.float64),
                    sol_size=self.sol_size)
                logger.info("Scheduler: %s", self.scheduler)
                # Set self.archive too for ease of reference.
                self.archive = self.scheduler.archive
                logger.info("Archive: %s", self.archive)
                if self.is_cma_mae:
                    self.result_archive = self.scheduler.result_archive
            elif self.search_algo == SearchAlgo.EXTINCT:
                logger.info("Setting up main archive")
                self.archive: ArchiveBase = archive_type(
                    solution_dim=self.sol_size,
                    learning_rate=1.0,
                    threshold_min=-np.inf,
                    seed=self.seed,
                    dtype=np.float64,
                )
                logger.info("Archive: %s", self.archive)
            elif self.search_algo == SearchAlgo.CC:
                logger.info("Setting up main archive")
                self.archive: ArchiveBase = archive_type(
                    solution_dim=self.sol_size,
                    learning_rate=1.0,
                    threshold_min=-np.inf,
                    seed=self.seed,
                    dtype=np.float64,
                )
                # NOTE: In case of CC, archive should record the context
                # vectors, aka the current found best solution(s).
                logger.info("Archive: %s", self.archive)
                # Initial CC iteration setup
                self._cc_per_iter_setup(space_idx=0)
        else:
            logger.info("Reloading scheduler and other data from logdir")

            with open(self.logdir.pfile("reload.pkl"), "rb") as file:
                data = pkl.load(file)
                self.rng = data["rng"]
                self.outer_itrs_completed = data["outer_itrs_completed"]
                self.total_evals = data["total_evals"]
                self.metrics = data["metrics"]
                self.overall_min_obj = data["overall_min_obj"]
                self.metadata_id = data["metadata_id"]
                self.cur_best_id = data["cur_best_id"]
                self.failed_levels = data["failed_levels"]
                if self.search_algo in [
                        SearchAlgo.EM,
                        SearchAlgo.EXTINCT,
                        SearchAlgo.CC,
                ]:
                    self.archive = data["archive"]
                    self.result_archive = None
                elif self.search_algo == SearchAlgo.CLASSIC:
                    self.scheduler = data["scheduler"]
                    self.archive = self.scheduler.archive
                    if self.is_cma_mae:
                        self.result_archive: ArchiveBase = self.scheduler.result_archive
                    else:
                        self.result_archive = None

            if self.search_algo == SearchAlgo.EM:
                self.env_manager.em_init(seed,
                                         self.logdir.pfile("reload_em.pkl"),
                                         self.logdir.pfile("reload_em.pth"))

            logger.info("Outer itrs already completed: %d",
                        self.outer_itrs_completed)
            logger.info("Execution continues from outer itr %d (1-based)",
                        self.outer_itrs_completed + 1)
            logger.info("Reloaded archive: %s", self.archive)

        logger.info("solution_dim: %d", self.archive.solution_dim)

        # Set the rng of the env manager
        if self.search_algo != SearchAlgo.CC:
            self.env_manager.rng = self.rng
        else:
            for env_m in self.env_managers:
                env_m.rng = self.rng

    def msg_all(self, msg: str):
        """Logs msg on master, on all workers, and in dashboard_status.txt."""
        logger.info(msg)
        self.client.run(worker_log, msg)
        with self.logdir.pfile("dashboard_status.txt").open("w") as file:
            file.write(msg)

    def finished(self):
        """Whether execution is done."""
        return self.total_evals >= self.max_evals

    def save_reload_data(self):
        """Saves data necessary for a reload.

        Current reload files:
        - reload.pkl
        - reload_em.pkl
        - reload_em.pth

        Since saving may fail due to memory issues, data is first placed in
        reload-tmp.pkl. reload-tmp.pkl then overwrites reload.pkl.

        We use gin to reference emitter classes, and pickle fails when dumping
        things constructed by gin, so we use cloudpickle instead. See
        https://github.com/google/gin-config/issues/8 for more info.
        """
        logger.info("Saving reload data")

        logger.info("Saving reload-tmp.pkl")
        with self.logdir.pfile("reload-tmp.pkl").open("wb") as file:
            reload_data = {
                "rng": self.rng,
                "outer_itrs_completed": self.outer_itrs_completed,
                "total_evals": self.total_evals,
                "metrics": self.metrics,
                "overall_min_obj": self.overall_min_obj,
                "metadata_id": self.metadata_id,
                "cur_best_id": self.cur_best_id,
                "failed_levels": self.failed_levels,
            }
            if self.search_algo in [SearchAlgo.EM, SearchAlgo.EXTINCT]:
                reload_data["archive"] = self.archive
            elif self.search_algo == SearchAlgo.CLASSIC:
                # Do not save self.archive again here even though it is set.
                reload_data["scheduler"] = self.scheduler

            cloudpickle.dump(reload_data, file)

        if self.search_algo == SearchAlgo.EM:
            logger.info("Saving reload_em-tmp.pkl and reload_em-tmp.pth")
            self.env_manager.emulation_model.save(
                self.logdir.pfile("reload_em-tmp.pkl"),
                self.logdir.pfile("reload_em-tmp.pth"))

        logger.info("Renaming tmp reload files")
        self.logdir.pfile("reload-tmp.pkl").rename(
            self.logdir.pfile("reload.pkl"))
        if self.search_algo == SearchAlgo.EM:
            self.logdir.pfile("reload_em-tmp.pkl").rename(
                self.logdir.pfile("reload_em.pkl"))
            self.logdir.pfile("reload_em-tmp.pth").rename(
                self.logdir.pfile("reload_em.pth"))

        logger.info("Finished saving reload data")

    def save_archive(self):
        """Saves dataframes of the archive.

        The archive, including solutions and metadata, is saved to
        logdir/archive/archive_{outer_itr}.pkl

        Note that the archive is saved as an ArchiveDataFrame storing common
        Python objects, so it should be stable (at least, given fixed software
        versions).
        """
        itr = self.outer_itrs_completed
        if self.is_cma_mae and self.search_algo == SearchAlgo.CLASSIC:
            df = self.result_archive.as_pandas(include_solutions=True,
                                               include_metadata=True)
        else:
            df = self.archive.as_pandas(include_solutions=True,
                                        include_metadata=True)
        df.to_pickle(self.logdir.file(f"archive/archive_{itr}.pkl"))

    def save_archive_history(self):
        """Saves the archive's history.

        We are okay with a pickle file here because there are only numpy arrays
        and Python objects, both of which are stable.
        """
        with self.logdir.pfile("archive_history.pkl").open("wb") as file:
            if self.is_cma_mae and self.search_algo == SearchAlgo.CLASSIC:
                pkl.dump(self.result_archive.history(), file)
            else:
                pkl.dump(self.archive.history(), file)

    def save_data(self):
        """Saves archive, reload data, history, and metrics if necessary.

        This method must be called at the _end_ of each outer itr. Otherwise,
        some things might not be complete. For instance, the metrics may be in
        the middle of an iteration, so when we reload, we get an error because
        we did not end the iteration.
        """
        if self.archive_save_freq is None:
            save_full_archive = False
        elif self.archive_save_freq == -1 and self.finished():
            save_full_archive = True
        elif (self.archive_save_freq > 0
              and self.outer_itrs_completed % self.archive_save_freq == 0):
            save_full_archive = True
        else:
            save_full_archive = False

        logger.info("Saving metrics")
        self.metrics.to_json(self.logdir.file("metrics.json"))

        logger.info("Saving archive history")
        self.save_archive_history()

        if save_full_archive:
            logger.info("Saving full archive")
            self.save_archive()
        if ((self.outer_itrs_completed % self.reload_save_freq == 0)
                or self.finished()):
            self.save_reload_data()
        if self.finished():
            logger.info("Saving failed envs")
            self.logdir.save_data(self.failed_levels, "failed_levels.pkl")

    def plot_metrics(self):
        """Plots metrics every self.plot_metrics_freq itrs or on final itr."""
        if (self.outer_itrs_completed % self.plot_metrics_freq == 0
                or self.finished()):
            logger.info("Metrics:\n%s", self.metrics.get_plot_text())

    def add_performance_metrics(self):
        """Calculates various performance metrics at the end of each iter."""
        if self.result_archive is not None:
            df = self.result_archive.as_pandas(include_solutions=False,
                                               include_metadata=True)
            stats = self.result_archive.stats
        else:
            df = self.archive.as_pandas(include_solutions=False,
                                        include_metadata=True)
            stats = self.archive.stats

        objs = df.objective_batch()
        actual_qd_score = self.env_manager.module.actual_qd_score(objs)
        self.metrics.add(
            "Total Evals",
            self.total_evals,
            logger,
        )
        self.metrics.add(
            "Actual QD Score",
            actual_qd_score,
            logger,
        )
        self.metrics.add(
            "Archive Size",
            stats.num_elites,
            logger,
        )
        self.metrics.add(
            "Archive Coverage",
            stats.coverage,
        )
        self.metrics.add(
            "Best Objective",
            np.max(objs),
            logger,
        )
        self.metrics.add(
            "Worst Objective",
            np.min(objs),
            logger,
        )
        self.metrics.add(
            "Mean Objective",
            np.mean(objs),
            logger,
        )
        self.metrics.add(
            "Overall Min Objective",
            self.overall_min_obj,
            logger,
        )

        # Extract other relevant stats from metadata
        metas = df.metadata_batch()
        if "similarity_score" in metas[0]["raw_metadata"]:
            sim_scores = [
                metas[i]["raw_metadata"]["similarity_score"]
                for i in range(len(metas))
            ]
            self.metrics.add(
                "Similarity Score of Env w/ Best Objective",
                sim_scores[np.argmax(objs)],
                logger,
            )

        if wandb.run is not None:
            self.stats["overal_max_obj"] = np.max(objs)
            self.stats["QD_score"] = actual_qd_score
            self.stats["Archive Coverage"] = stats.coverage
            wandb.log(self.stats)

    def extract_metadata(self, r) -> dict:
        """Constructs metadata object from results of an evaluation."""
        meta = dataclasses.asdict(r)

        # Remove unwanted keys.
        none_keys = [key for key in meta if meta[key] is None]
        for key in itertools.chain(none_keys, []):
            try:
                meta.pop(key)
            except KeyError:
                pass

        meta["metadata_id"] = self.metadata_id
        self.metadata_id += 1

        return meta

    def build_emitters_and_scheduler(self,
                                     archive,
                                     initial_solutions=None,
                                     sol_size=None):
        """Builds pyribs components with the config params and given archive."""
        # Makes sense to initialize at zero since these are latent vectors.
        if initial_solutions is None:
            # initial_solutions = np.zeros(
            #     (self.n_emitters, sol_size)) + self.initial_mean
            n_emitters = np.sum([x[1] for x in self.emitter_types], dtype=int)
            initial_solutions = self.env_manager.get_cma_initial_mean(
                n_emitters, self.initial_mean)

        emitters = []
        k = 0  # cnt of emitters
        for emitter_class, n_curr_emitters in self.emitter_types:
            emitter_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                              size=n_curr_emitters,
                                              endpoint=True)
            # For EvolutionStrategyEmitter
            emitters.extend([
                emitter_class(archive, x0=initial_solutions[k + i], seed=s)
                for i, s in enumerate(emitter_seeds)
            ])
            # For GaussianEmitter
            # emitters.extend([
            #     emitter_class(archive, initial_solutions=initial_solutions, seed=s)
            #     for i, s in enumerate(emitter_seeds)
            # ])
            logger.info("Constructed %d emitters of class %s - seeds %s",
                        n_curr_emitters, emitter_class, emitter_seeds)
            k += n_curr_emitters
        logger.info("Emitters: %s", emitters)

        # Create result archive for CMA-MAE
        result_archive = None
        if self.is_cma_mae:
            # Only record history in result archive if we are running classic
            # search
            record_history = (self.search_algo == SearchAlgo.CLASSIC)
            result_archive = self.archive_type(
                solution_dim=sol_size,
                learning_rate=1.0,
                threshold_min=-np.inf,
                seed=self.seed,
                dtype=np.float64,
                record_history=record_history,
            )
            logger.info("Constructed result archive for CMA-MAE")

        scheduler = Scheduler(
            archive,
            emitters,
            result_archive=result_archive,
        )
        logger.info("scheduler: %s", scheduler)

        return emitters, scheduler

    def _inner_loop_setup(self, initial_solutions=None):
        """Util function to setup archive and scheduler for inner loop"""
        logger.info("Setting up pyribs components")
        seed = self.rng.integers(np.iinfo(np.int32).max / 2, endpoint=True)

        if self.search_algo == SearchAlgo.EM:
            surrogate_archive: ArchiveBase = self.archive_type(
                solution_dim=self.sol_size,
                seed=seed,
                dtype=np.float64,
                record_history=False)
        elif self.search_algo == SearchAlgo.EXTINCT:
            surrogate_archive: ArchiveBase = self.extinct_inner_archive_type(
                solution_dim=self.sol_size,
                seed=seed,
                dtype=np.float64,
                record_history=False)

        logger.info("Archive: %s", surrogate_archive)

        _, scheduler = self.build_emitters_and_scheduler(
            surrogate_archive, initial_solutions, self.sol_size)

        return seed, scheduler

    def _downsample_archive(self, surrogate_archive, seed):
        """
        In downsampling, we create a smaller archive where the elite in each
        cell is sampled from a corresponding region of cells in the main
        archive.
        """
        downsample_archive: ArchiveBase = self.downsample_archive_type(
            solution_dim=self.sol_size,
            seed=seed,
            dtype=np.float64,
            record_history=False)
        # downsample_archive.initialize(archive.solution_dim)
        scales = np.array(surrogate_archive.dims) // np.array(
            downsample_archive.dims)

        # Iterate through every index in the downsampled archive.
        for downsample_idx in itertools.product(
                *map(range, downsample_archive.dims)):

            # In each index, retrieve the corresponding elites in the main
            # archive.
            elites = []
            archive_ranges = [
                range(scales[i] * downsample_idx[i],
                      scales[i] * (downsample_idx[i] + 1))
                for i in range(surrogate_archive.measure_dim)
            ]
            for idx in itertools.product(*archive_ranges):
                # pylint: disable = protected-access
                idx = surrogate_archive.grid_to_int_index(
                    np.array(idx).reshape(1, 2))
                if surrogate_archive._occupied_arr[idx]:
                    elites.append(
                        Elite(surrogate_archive._solution_arr[idx],
                              surrogate_archive._objective_arr[idx],
                              surrogate_archive._measures_arr[idx], idx,
                              surrogate_archive._metadata_arr[idx]))

            # Choose one of the elites to insert into the archive.
            if len(elites) > 0:
                sampled_elite = elites[self.rng.integers(len(elites))]
                downsample_archive.add(sampled_elite.solution,
                                       sampled_elite.objective,
                                       sampled_elite.measures,
                                       sampled_elite.metadata)

        # surrogate_archive = downsample_archive

        # Save downsampled surrogate archive
        if self.save_surrogate_archive:
            save_dir = self.logdir.dir("surrogate_archive", touch=True)
            df = downsample_archive.as_pandas(include_solutions=True,
                                              include_metadata=True)
            df.to_pickle(
                f"{save_dir}/downsample_archive_{self.outer_itrs_completed}.pkl"
            )

        logger.info(
            "Downsampled emulation archive has %d elites (%f coverage)",
            downsample_archive.stats.num_elites,
            downsample_archive.stats.coverage)

        return downsample_archive

    def build_emulation_archive(self) -> ArchiveBase:
        """Builds an archive which optimizes the emulation model."""

        seed, scheduler = self._inner_loop_setup()

        # Obtain surrogate archive
        if self.is_cma_mae:
            surrogate_archive = scheduler.result_archive
        else:
            surrogate_archive = scheduler.archive

        log_step = 1000
        if self.inner_itrs <= 1000:
            log_step = 100

        for inner_itr in range(1, self.inner_itrs + 1):
            self.em_evaluate(scheduler)
            if inner_itr % log_step == 0 or inner_itr == self.inner_itrs:
                logger.info("Completed inner iteration %d", inner_itr)

        # Use result archive to build emulation archive for CMA-MAE
        # if self.is_cma_mae:
        #     surrogate_archive = self.result_archive

        logger.info("Generated emulation archive with %d elites (%f coverage)",
                    surrogate_archive.stats.num_elites,
                    surrogate_archive.stats.coverage)

        # Save surrogate archive
        if self.save_surrogate_archive:
            save_dir = self.logdir.dir("surrogate_archive", touch=True)

            df = surrogate_archive.as_pandas(include_solutions=True,
                                             include_metadata=True)
            df.to_pickle(f"{save_dir}/archive_{self.outer_itrs_completed}.pkl")

        # In downsampling, we create a smaller archive where the elite in each
        # cell is sampled from a corresponding region of cells in the main
        # archive.
        if self.downsample_surrogate:
            surrogate_archive = self._downsample_archive(
                surrogate_archive, seed)

        return surrogate_archive

    def em_evaluate(self, scheduler):
        """
        Asks for solutions from the scheduler, evaluates using the emulation
        model, and tells the objective and measures
        Args:
            scheduler: Scheduler to use
        """
        sols, _ = scheduler.ask()
        map_comps, objs, measures, success_mask = \
            self.env_manager.emulation_pipeline(sols)

        all_objs = np.full(len(map_comps), np.nan)
        all_measures = np.full((len(map_comps), self.archive.measure_dim),
                               np.nan)
        all_objs[success_mask] = objs
        all_measures[success_mask] = measures

        # Need to add map_comps to metadata
        scheduler.tell(
            all_objs,
            all_measures,
            success_mask=success_mask,
            # metadata_batch=map_comps,
            result_archive_objective_batch=all_objs,
        )

        return sols, map_comps, objs, measures

    def build_inner_obj_archive(self) -> ArchiveBase:
        """
        Build archive with inner objective (usually similarity score) by running
        classic pyribs.
        """
        if self.outer_itrs_completed == 0:
            initial_solutions = None
        else:
            # After first outer loop iteration, select high-objective elites as
            # the initial solutions of the next iterations
            # initial_solutions = self.archive.best_elite.solution
            best_elite = self.archive.best_elite.solution
            if self.n_emitters <= 1:
                initial_solutions = [best_elite]
            else:
                initial_solutions = self.archive.sample_top_elites(
                    self.n_emitters - 1, self.extinct_p).solution_batch
                initial_solutions = np.concatenate(
                    [best_elite[np.newaxis, ...], initial_solutions])
        seed, scheduler = self._inner_loop_setup(initial_solutions)

        # Obtain surrogate archive
        if self.is_cma_mae:
            inner_obj_archive = scheduler.result_archive
        else:
            inner_obj_archive = scheduler.archive

        log_step = 10

        for inner_itr in range(1, self.inner_itrs + 1):
            sols, parent_sols = scheduler.ask()
            self.extinct_inner_evaluate(sols, parent_sols, scheduler)
            if inner_itr % log_step == 0 or inner_itr == self.inner_itrs:
                logger.info("Completed inner iteration %d", inner_itr)

        logger.info("Generated inner obj archive with %d elites (%f coverage)",
                    inner_obj_archive.stats.num_elites,
                    inner_obj_archive.stats.coverage)

        # Save surrogate archive
        if self.save_surrogate_archive:
            save_dir = self.logdir.dir("surrogate_archive", touch=True)

            df = inner_obj_archive.as_pandas(include_solutions=True,
                                             include_metadata=True)
            df.to_pickle(f"{save_dir}/archive_{self.outer_itrs_completed}.pkl")

        if self.downsample_surrogate:
            inner_obj_archive = self._downsample_archive(
                inner_obj_archive, seed)

        return inner_obj_archive

    def extinct_inner_evaluate(self, sols, parent_sols, scheduler):
        """
        Evaluate a batch of solutions on the inner loop objective and add to
        surrogate archive
        """
        logger.info("Evaluating solutions on inner objective")
        results = self.env_manager.inner_obj_pipeline(
            sols,
            parent_sols=parent_sols,
        )
        objs, result_objs, measures, metadata, success_mask = [], [], [], [], []
        for r in results:
            if not r.failed:
                objs.append(r.agg_obj)
                result_objs.append(r.agg_result_obj)
                measures.append(r.agg_measures)
                success_mask.append(True)
                metadata.append(self.extract_metadata(r))
            else:
                objs.append(np.nan)
                result_objs.append(np.nan)
                measures.append(np.full(scheduler.archive.measure_dim, np.nan))
                success_mask.append(False)
                metadata.append(None)

        scheduler.tell(
            objs,
            measures,
            metadata,
            success_mask=success_mask,
            result_archive_objective_batch=result_objs,
        )

    def _cc_per_iter_setup(self, space_idx: int):
        search_space = self.cc_search_spaces[space_idx]
        env_manager = self.env_managers[space_idx]
        curr_sol_size = env_manager.get_sol_size()

        logger.info(
            f"CC Iter {self.cc_n_iter}, Current search space: {search_space}")

        # Change the config if necessary
        if isinstance(env_manager, WarehouseManager):
            config = WarehouseConfig()

        if self.outer_itrs_completed == 0:
            config.online_update = False
            config.task_assignment_cost = "heuristic+num_agents"
            # Fake elite
            best_elite = Elite(solution=self.rng.random(self.sol_size),
                               objective=None,
                               measures=None,
                               index=None,
                               metadata=None)
        else:
            # Get the context vector, for now we greedyly get the best solution
            best_elite = self.archive.best_elite

        # We are recreating env manager everytime.
        # TODO: This is not efficient, find alternate way.
        env_manager = self.env_manager_class(self.client,
                                             self.logdir,
                                             search_space=search_space,
                                             config=config)
        env_manager.seed = self.seed
        env_manager.rng = self.rng

        # Set up the context vector inside env_manager
        # Searching for CCS with currently best g-policy and t-policy
        initial_solution = None
        sol = best_elite.solution
        if search_space == SearchSpace.CHUTE_CAPACITIES:
            # Assuming that the second and third parts of the solutions are
            # the g-policy and t-policy parameters respectively
            env_manager.g_policy_params = sol[
                env_manager.n_chutes:-QUAD_TASK_ASSIGN_N_PARAM]
            env_manager.t_policy_params = sol[
                -QUAD_TASK_ASSIGN_N_PARAM:].tolist()

            # Initial solution is best chute capacities from the previous CC
            # iter
            if self.outer_itrs_completed > 0:
                initial_solution = sol[:env_manager.n_chutes]

        elif search_space == SearchSpace.G_POLICY_TASK_ASSIGN_POLICY:
            # No need to worry about the archive not having solution because we
            # SHOULD always optimize g/t-policy AFTER chute capacities
            env_manager.base_chute_mapping_json = best_elite.metadata[
                "raw_metadata"]["chute_mapping"]
            if self.outer_itrs_completed > 0:
                initial_solution = sol[env_manager.n_chutes:]

        initial_solutions = np.repeat(
            initial_solution[np.newaxis, ...], self.n_emitters,
            axis=0) if initial_solution is not None else None

        logger.info("Setting up scheduler for CC pyribs")
        _, scheduler = self.build_emitters_and_scheduler(
            self.archive_type(solution_dim=curr_sol_size,
                              seed=self.seed,
                              dtype=np.float64,
                              record_history=False),
            # initial_solutions=initial_solutions,
            sol_size=curr_sol_size)
        logger.info("Scheduler: %s", scheduler)

        # Update relevant attributes for CC
        self.cc_curr_search_space = search_space
        self.env_manager = env_manager
        self.scheduler = scheduler
        self.cc_context_vector = best_elite.solution

    def _cc_form_complete_solution(self, sol, context_sol):
        """
        Form the complete solution with the context vector
        Args:
            sols: List of solutions from the per iteration scheduler
        Returns:
            np.ndarray: Complete solution
        """
        # Form the complete solution with the context vector
        if self.cc_curr_search_space == SearchSpace.CHUTE_CAPACITIES:
            # Replace the first part of the context vector to form the complete
            # solution
            return np.concatenate(
                [sol, context_sol[self.env_manager.n_chutes:]])
        elif self.cc_curr_search_space == SearchSpace.G_POLICY_TASK_ASSIGN_POLICY:
            # Replace the last part of the context vector to form the complete
            # solution
            return np.concatenate(
                [context_sol[:self.env_manager.n_chutes], sol])

    def evaluate_solutions(self, sols, parent_sols=None, context_sol=None):
        """Evaluates a batch of solutions and adds them to the archive.

        Args:
            sols (np.ndarray): solutions to evaluate
            parent_sols (np.ndarray, optional): parent solutions.
            context_sol (np.ndarray, optional): context vector for CC.

        """
        logger.info("Evaluating solutions")

        skipped_sols = 0
        if self.total_evals + len(sols) > self.max_evals:
            remaining_evals = self.max_evals - self.total_evals
            remaining_sols = remaining_evals
            skipped_sols = len(sols) - remaining_sols
            sols = sols[:remaining_sols]
            if parent_sols is not None:
                parent_sols = parent_sols[:remaining_sols]
            logger.info(
                "Unable to evaluate all solutions; will evaluate %d instead",
                remaining_sols,
            )

        logger.info("total_evals (old): %d", self.total_evals)
        self.total_evals += len(sols)
        logger.info("total_evals (new): %d", self.total_evals)

        logger.info("Distributing evaluations")

        results = self.env_manager.eval_pipeline(
            sols,
            parent_sols=parent_sols,
            batch_idx=self.outer_itrs_completed,
        )

        if self.search_algo == SearchAlgo.EM:
            logger.info(
                "Adding solutions to main archive and emulation dataset")
        elif self.search_algo == SearchAlgo.CLASSIC:
            logger.info("Adding solutions to the scheduler")
        elif self.search_algo == SearchAlgo.EXTINCT:
            logger.info("Outer loop: Adding solutions to the main archive")
        elif self.search_algo == SearchAlgo.CC:
            logger.info(
                "Outer loop: Adding solutions to the main archive and per iteration scheduler"
            )

        objs = []
        result_archive_objs = []
        if self.search_algo in [SearchAlgo.CLASSIC, SearchAlgo.CC]:
            measures, metadata, success_mask = [], [], []

        for sol, r in zip(sols, results):
            if not r.failed:
                obj = r.agg_obj
                obj_result = r.agg_result_obj
                objs.append(obj)  # Always insert objs.
                # objs used for the result archive, if applicable
                result_archive_objs.append(obj_result)
                meas = r.agg_measures
                meta = self.extract_metadata(r)

                # For DSAGE and QD-Extinct, we add the evaluated solutions to
                # the main archive
                if self.search_algo in [SearchAlgo.EM, SearchAlgo.EXTINCT]:
                    # Use obj_result for ground-truth archive in DSAGE
                    self.archive.add_single(sol,
                                            obj_result,
                                            meas,
                                            metadata=meta)
                    # For DSAGE, add evaluated sols to data buffer
                    if self.search_algo == SearchAlgo.EM:
                        self.env_manager.add_experience(sol, r)
                # For classic QD, we batch all the results and `tell` the main
                # scheduler
                elif self.search_algo == SearchAlgo.CLASSIC:
                    measures.append(meas)
                    metadata.append(meta)
                    success_mask.append(True)
                # For CC, we do two things:
                # 1. form the complete solution with the context vector and add
                #    to the main archive.
                # 2. add the solution to the per iteration scheduler.
                elif self.search_algo == SearchAlgo.CC:
                    comp_sol = self._cc_form_complete_solution(
                        sol, context_sol)
                    self.archive.add_single(comp_sol,
                                            obj_result,
                                            meas,
                                            metadata=meta)
                    measures.append(meas)
                    metadata.append(meta)
                    success_mask.append(True)
            else:
                failed_level_info = self.env_manager.add_failed_info(sol, r)
                self.failed_levels.append(failed_level_info)
                if self.search_algo in [SearchAlgo.CLASSIC, SearchAlgo.CC]:
                    objs.append(np.nan)
                    measures.append(np.full(self.archive.measure_dim, np.nan))
                    metadata.append(None)
                    success_mask.append(False)
                    result_archive_objs.append(np.nan)

        # Tell results to scheduler.
        # For Classic, the scheduler contains the archive.
        # For CC, the scheduler is the per iteration scheduler that only
        # contains the current search space. self.archive is the main archive
        # that stores the context vectors.
        if self.search_algo in [SearchAlgo.CLASSIC, SearchAlgo.CC]:
            logger.info("Filling in null values for skipped sols: %d",
                        skipped_sols)
            for _ in range(skipped_sols):
                objs.append(np.nan)
                measures.append(np.full(self.archive.measure_dim, np.nan))
                metadata.append(None)
                success_mask.append(False)
                result_archive_objs.append(np.nan)

            self.scheduler.tell(
                objs,
                measures,
                metadata,
                success_mask=success_mask,
                result_archive_objective_batch=result_archive_objs,
            )

        self.metrics.add("Mean Evaluation", np.nanmean(objs), logger)
        self.overall_min_obj = min(self.overall_min_obj, np.nanmin(objs))
        self.stats["per_iter_obj_mean"] = np.nanmean(objs)
        measures = np.array(measures)
        self.stats["per_iter_obj_std"] = np.nanstd(objs)
        self.stats["per_iter_mes1_mean"] = np.nanmean(measures[:, 0])
        self.stats["per_iter_mes2_mean"] = np.nanmean(measures[:, 1])

    def evaluate_initial_emulation_solutions(self):
        logger.info("Evaluating initial solutions")
        initial_solutions, _ = self.env_manager.get_initial_sols(
            (self.initial_sols, self.sol_size))
        self.evaluate_solutions(initial_solutions)

    def _extract_sols_from_archive(self, archive: ArchiveBase):
        if self.num_elites_to_eval is None:
            sols = [elite.solution for elite in archive]
            logger.info("%d solutions in archive", len(sols))
        else:
            num_sols = len(archive)
            sols = []
            sol_values = []
            rands = self.rng.uniform(0, 1e-8, size=num_sols)  # For tiebreak

            for i, elite in enumerate(archive):
                sols.append(elite.solution)
                if self.random_sample_em:
                    new_elite = 1
                else:
                    new_elite = int(
                        self.archive.retrieve_single(elite.measures).objective
                        is None)
                sol_values.append(new_elite + rands[i])

            _, sorted_sols = zip(*sorted(
                zip(sol_values, sols), reverse=True, key=lambda x: x[0]))
            sols = sorted_sols[:self.num_elites_to_eval]
            logger.info(
                f"{np.sum(np.array(sol_values) > 1e-6)} solutions predicted to "
                f"improve.")
            logger.info(
                f"Evaluating {len(sols)} out of {num_sols} solutions in "
                f"emulation_archive")
        return sols

    def evaluate_emulation_archive(self, emulation_archive: ArchiveBase):
        """
        Evaluate solutions in `emulation_archive`

        Args:
            emulation_archive (ArchiveBase): archive with solutions from inner
                loop
        """
        logger.info("Evaluating solutions in emulation_archive")
        sols = self._extract_sols_from_archive(emulation_archive)
        self.evaluate_solutions(sols)

    def evaluate_inner_obj_archive(self, inner_obj_archive):
        """
        Evaluate solutions in `inner_obj_archive` for outer objective (usually
        throughput)

        Args:
            inner_obj_archive (ArchiveBase): archive with solutions from inner
                loop
        """
        logger.info("Evaluating solutions in inner_obj_archive")
        sols = self._extract_sols_from_archive(inner_obj_archive)
        self.evaluate_solutions(sols)

    def execute(self):
        """Runs the entire algorithm."""
        wandb.login(key="")
        run = wandb.init(
            name=f"{self.logdir.logdir.name}",
            project="chute_mapping",
            id=f"{self.logdir.logdir.name}",
            resume="allow",
            mode=self.wandb_mode,
        )
        while not self.finished():
            self.msg_all(f"----- Outer Itr {self.outer_itrs_completed + 1} "
                         f"({self.total_evals} evals) -----")
            self.metrics.start_itr()
            self.stats = {}
            self.archive.new_history_gen()

            if self.search_algo == SearchAlgo.EM:
                if self.outer_itrs_completed == 0:
                    self.evaluate_initial_emulation_solutions()
                else:
                    logger.info("Running inner loop")
                    self.env_manager.em_train()
                    emulation_archive = self.build_emulation_archive()
                    self.evaluate_emulation_archive(emulation_archive)
            elif self.search_algo == SearchAlgo.CLASSIC:
                if self.result_archive is not None:
                    self.result_archive.new_history_gen()
                logger.info("Running classic pyribs")
                sols, parent_sols = self.scheduler.ask()
                self.evaluate_solutions(sols, parent_sols=parent_sols)
            elif self.search_algo == SearchAlgo.EXTINCT:
                # The inner loop is the same as running classic multiple times,
                # except that we optimize for inner objective
                logger.info("Running inner loop")
                # Start inner loop with selected solutions from the previous
                # outer loop
                inner_obj_archive = self.build_inner_obj_archive()
                # The outer loop simply evaluate for outer objective, similar to
                # outer loop of DSAGE.
                self.evaluate_inner_obj_archive(inner_obj_archive)
            elif self.search_algo == SearchAlgo.CC:
                # Reset counter if changing search space
                if self.cc_curr_iter_evals >= self.cc_per_iter_evals:
                    self.cc_curr_iter_evals = 0
                    self.cc_n_iter += 1
                    # Alternate between the given list of search spaces
                    # Determine the current search space
                    space_idx = self.cc_n_iter % len(self.cc_search_spaces)
                    self._cc_per_iter_setup(space_idx)
                # Run a round of search with the current search space
                # breakpoint()
                sols, parent_sols = self.scheduler.ask()
                self.evaluate_solutions(sols,
                                        parent_sols=parent_sols,
                                        context_sol=self.cc_context_vector)
                self.cc_curr_iter_evals += len(sols)
            # Restart worker to clean up memory leak
            self.client.restart()

            logger.info("Outer itr complete - now logging and saving data")
            self.outer_itrs_completed += 1
            self.add_performance_metrics()
            self.metrics.end_itr(logger=logger)
            self.plot_metrics()
            self.save_data()  # Keep at end of loop (see method docstring).

        repair_runtime = round(self.env_manager.repair_runtime, 2)
        sim_runtime = round(self.env_manager.sim_runtime, 2)
        self.msg_all(f"----- Done! {self.outer_itrs_completed} itrs, "
                     f"{self.total_evals} evals, "
                     f"Repair takes {repair_runtime} s, "
                     f"Sim takes {sim_runtime} s -----")
