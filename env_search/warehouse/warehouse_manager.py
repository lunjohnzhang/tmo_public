"""Provides WarehouseManager."""
import logging
from pathlib import Path
from typing import List, Tuple

import time
import gin
import numpy as np
import copy
import torch
import json
# import multiprocessing
import torch.multiprocessing as multiprocessing
from dask.distributed import Client
from logdir import LogDir
from tqdm import tqdm
from itertools import repeat
from scipy.special import softmax

from cma.constraints_handler import BoundTransform

from env_search.utils.network import int_preprocess_onehot
from env_search.utils.enum import SearchSpace
from env_search.device import DEVICE
from env_search.warehouse import (get_package_dist, get_packages,
                                  QUAD_TASK_ASSIGN_N_PARAM)
from env_search.warehouse.emulation_model.buffer import Experience
from env_search.warehouse.emulation_model.aug_buffer import AugExperience
from env_search.warehouse.emulation_model.double_aug_buffer import DoubleAugExperience
from env_search.warehouse.emulation_model.emulation_model import WarehouseEmulationModel
from env_search.warehouse.emulation_model.networks import (
    WarehouseAugResnetOccupancy, WarehouseAugResnetRepairedMapAndOccupancy)
from env_search.warehouse.module import (WarehouseModule, WarehouseConfig,
                                         get_comp_map, get_additional_h_blocks,
                                         get_additional_v_blocks,
                                         compute_chute_capacities)
from env_search.warehouse.generator.nca_generator import WarehouseNCA
from env_search.warehouse.run import (run_warehouse, repair_warehouse,
                                      process_warehouse_eval_result,
                                      run_warehouse_iterative_update,
                                      gen_chute_mapping_from_capacity,
                                      repair_chute_mapping)
from env_search.utils.worker_state import init_warehouse_module

from env_search.utils import (
    kiva_obj_types, KIVA_ROBOT_BLOCK_WIDTH, KIVA_WORKSTATION_BLOCK_WIDTH,
    KIVA_ROBOT_BLOCK_HEIGHT, MIN_SCORE, kiva_env_number2str,
    kiva_env_str2number, read_in_sortation_map, sortation_env_str2number,
    format_env_str, read_in_kiva_map, flip_tiles, flip_tiles_torch,
    get_n_valid_edges, get_n_valid_vertices, n_params, get_chute_loc,
    min_max_normalize_2d)

logger = logging.getLogger(__name__)


def _nca_generate_one_slice_env(sols, seed_map_torch, nca_iter):
    """Helper function to generate environments with NCA model
    """
    warehouseNCA = WarehouseNCA()
    envs = []
    for sol in sols:
        warehouseNCA.set_params(sol)
        env, _ = warehouseNCA.generate(
            seed_map_torch,
            n_iter=nca_iter,
        )
        envs.append(env)
    return torch.cat(envs)


@gin.configurable(denylist=["client", "rng"])
class WarehouseManager:
    """Manager for the warehouse environments.

    Args:
        client: Dask client for distributed compute.
        rng: Random generator. Can be set later. Uses `np.random.default_rng()`
            by default.
        n_evals: Number of times to evaluate each solution during real
            evaluation.
        env_width: Width of the level.
        env_height: Height of the level.
        num_objects: Number of objects in the level to generate.
        min_n_shelf (int): min number of shelves
        max_n_shelf (int): max number of shelves
        w_mode (bool): whether to run with w_mode, which replace 'r' with 'w' in
                       generated map layouts, where 'w' is a workstation.
                       Under w_mode, robots will start from endpoints and their
                       tasks will alternate between endpoints and workstations.
        n_endpt (int): number of endpoint around each obstacle
        search_mode (str): search space, one of ["layout", "guidance-graph",
            "layout_guidance-graph"]
    """

    def __init__(
        self,
        client: Client,
        logdir: LogDir,
        rng: np.random.Generator = None,
        n_evals: int = gin.REQUIRED,
        simulation_algo: str = gin.REQUIRED,
        search_space: str = gin.REQUIRED,
        # Layout optimization related
        env_width: int = 33,
        env_height: int = 32,
        num_objects: int = 3,
        min_n_shelf: int = 240,
        max_n_shelf: int = 240,
        w_mode: bool = True,
        n_endpt: bool = 1,
        is_nca: bool = False,
        nca_iter: int = 200,
        # GGO related
        is_piu: bool = False,
        seed_env_path: str = None,
        guidance_policy_file: str = None,
        # Shared by GGO and Chute mapping
        task_assign_policy_file: str = None,
        base_map_path: str = None,
        bound_handle: str = None,
        bounds: tuple = None,
        # Chute mapping related
        package_mode: str = "dist",
        package_dist_type: str = "721",
        package_path: str = None,
        n_destinations: int = 100,
        chute_mapping_file: str = None,
        heuristic_chute_mapping: List[str] = None,
        # Warehouse config
        config: WarehouseConfig = None,
    ):
        self.client = client
        self.rng = rng or np.random.default_rng()

        self.n_evals = n_evals
        self.eval_batch_idx = 0  # index of each batch of evaluation

        self.logdir = logdir

        self.env_width = env_width
        self.env_height = env_height

        self.num_objects = num_objects

        self.min_n_shelf = min_n_shelf
        self.max_n_shelf = max_n_shelf

        self.w_mode = w_mode
        self.n_endpt = n_endpt
        self.simulation_algo = simulation_algo
        self.search_space = SearchSpace(search_space)

        self.emulation_model = None

        self.is_nca = is_nca
        self.nca_iter = nca_iter
        self.nca_n_param = -1

        # bounds
        self.bounds = bounds
        self.bound_handle = bound_handle
        self.bound_transform = BoundTransform(list(
            self.bounds)) if self.bounds is not None else None

        # Dummy package dist for layout opt and GGO
        self.package_dist_weight = None
        self.package_dist_weight_json = None

        self.is_piu = is_piu
        self.update_mdl_n_param = 0
        self.heuristic_chute_mapping = heuristic_chute_mapping

        # Runtime
        self.repair_runtime = 0
        self.sim_runtime = 0

        # Set up a module locally and on workers. During evaluations,
        # the run functions retrieves this module and uses it to
        # evaluate the function. Configuration is done with gin (i.e. the
        # params are in the config file).
        if config is None:
            config = WarehouseConfig()
        self.module = WarehouseModule(config)
        client.register_worker_callbacks(lambda: init_warehouse_module(config))

        # NCA related
        if self.is_nca:
            self.warehouseNCA = WarehouseNCA().to(DEVICE)
            self.nca_n_param = n_params(self.warehouseNCA)
            seed_map_str, _ = read_in_kiva_map(seed_env_path)
            seed_map_int = kiva_env_str2number(seed_map_str)
            self.seed_map_torch = torch.tensor(seed_map_int[np.newaxis, :, :],
                                               # device=DEVICE,
                                               )

        # For GGO and chute mapping, we need a base map
        # NOTE: for offline GGO, the edge weights are assumed to be included in
        # the base map json
        if self.search_space in [
                SearchSpace.G_GRAPH,
                SearchSpace.G_POLICY,
                SearchSpace.CHUTE_CAPACITIES,
                SearchSpace.CHUTE_MAPPING,
                SearchSpace.G_POLICY_CHUTE_CAPACITIES,
                SearchSpace.G_POLICY_TASK_ASSIGN_POLICY,
                SearchSpace.TASK_ASSIGN_POLICY,
                SearchSpace.G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY,
                SearchSpace.G_GRAPH_CHUTE_CAPACITIES,
                SearchSpace.G_GRAPH_TASK_ASSIGN_POLICY,
                SearchSpace.G_GRAPH_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY,
        ]:
            assert base_map_path is not None

            # Read in map as json, str, and np
            with open(base_map_path, "r") as f:
                self.base_map_json = json.load(f)
            if self.module.config.scenario == "KIVA":
                self.base_map_str, _ = read_in_kiva_map(base_map_path)
                self.base_map_np = kiva_env_str2number(self.base_map_str)
                self.domain = "kiva"
            elif self.module.config.scenario == "SORTING":
                self.base_map_str, _ = read_in_sortation_map(base_map_path)
                self.base_map_np = sortation_env_str2number(self.base_map_str)
                self.domain = "sortation"

            self.n_valid_edges = get_n_valid_edges(
                self.base_map_np,
                bi_directed=True,
                domain=self.domain,
            )
            self.n_valid_vertices = get_n_valid_vertices(self.base_map_np,
                                                         domain=self.domain)

            # Record the number of valid weights in the base map
            self.n_g_graph_weights = None
            if self.simulation_algo == "PIBT":
                self.n_g_graph_weights = self.n_valid_vertices + self.n_valid_edges
            elif self.simulation_algo == "RHCR":
                self.n_g_graph_weights = self.n_valid_edges + 1
            else:
                logger.error(
                    f"Unknown simulation algorithm {self.simulation_algo}")

        # In sortation systems, we need to generate package distribution
        if self.module.config.scenario == "SORTING":
            self.n_destinations = n_destinations
            self.chute_locs = get_chute_loc(self.base_map_np)
            self.n_chutes = len(self.chute_locs)
            # NOTE: For now one chute can serve exactly one destination.
            # Therefore we must have more chutes can destinations
            assert self.n_chutes >= n_destinations

            # Read in the package distribution
            self.package_dist_weight, self.package_dist_weight_json = get_packages(
                package_mode,
                package_dist_type,
                package_path,
                n_destinations,
            )

            # Under sortation system but not searching for chute mapping, we
            # need to read in a fixed chute mapping!
            if self.search_space not in [
                    SearchSpace.CHUTE_CAPACITIES,
                    SearchSpace.CHUTE_MAPPING,
                    SearchSpace.G_POLICY_CHUTE_CAPACITIES,
                    SearchSpace.G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY,
                    SearchSpace.G_GRAPH_CHUTE_CAPACITIES,
                    SearchSpace.G_GRAPH_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY,
            ]:
                assert chute_mapping_file is not None
                with open(chute_mapping_file, "r") as f:
                    self.base_chute_mapping_json = json.load(f)

            # Under sortation system, using online update, and not searching
            # for guidance policy, we need to read in a fixed guidance policy
            if self.module.config.online_update and self.search_space not in [
                    SearchSpace.G_POLICY,
                    SearchSpace.G_POLICY_CHUTE_CAPACITIES,
                    SearchSpace.G_POLICY_TASK_ASSIGN_POLICY,
                    SearchSpace.G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY,
            ]:
                assert guidance_policy_file is not None
                # np.ndarray
                self.g_policy_params: np.ndarray = np.load(
                    guidance_policy_file)
            else:
                self.g_policy_params = None

            # Under sortation system, using online update, using quadratic task
            # assignment policy, and not searching for task assignment policy
            # parameters, need to read in a fixed task assignment policy!
            if self.module.config.online_update and \
                self.module.config.task_assignment_cost == "opt_quadratic_f" \
                and self.search_space not in [
                    SearchSpace.TASK_ASSIGN_POLICY,
                    SearchSpace.G_POLICY_TASK_ASSIGN_POLICY,
                    SearchSpace.G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY,
            ]:
                assert task_assign_policy_file is not None
                # List
                self.t_policy_params: List = np.load(
                    task_assign_policy_file).tolist()
            else:
                self.t_policy_params = None

        # PIU/Guidance policy related
        if self.is_piu or self.module.config.online_update:
            # Create a fake update model to calculate the number of params
            tmp_update_model = self.module.config.iter_update_model_type(
                env_np=None,
                n_valid_vertices=None,
                n_valid_edges=None,
                model_params=None,
                config=self.module.config,
                **self.module.config.iter_update_mdl_kwargs,
            )
            self.update_mdl_n_param = n_params(tmp_update_model.model)

    def em_init(self,
                seed: int,
                pickle_path: Path = None,
                pytorch_path: Path = None):
        """Initialize the emulation model and optionally load from saved state.

        Args:
            seed: Random seed to use.
            pickle_path: Path to the saved emulation model data (optional).
            pytorch_path: Path to the saved emulation model network (optional).
        """
        self.emulation_model = WarehouseEmulationModel(seed=seed + 420)
        if pickle_path is not None:
            self.emulation_model.load(pickle_path, pytorch_path)
        logger.info("Emulation Model: %s", self.emulation_model)

    def get_initial_sols(self, size: Tuple):
        """Returns random solutions with the given size.

        Args:
            size: Tuple with (n_solutions, sol_size).

        Returns:
            Randomly generated solutions.
        """
        batch_size, solution_dim = size

        # For NCA, initial solutions are parameters of NCA generators.
        if self.is_nca:
            return self.rng.normal(0, 100, size=size), None

        # Otherwise, initial solutions are actual environments.
        if self.num_objects == 2:
            # If we know the exact number of shelves and we have one only
            # two objects (floor or shelf), we can generate solutions
            # directly.
            if self.min_n_shelf == self.max_n_shelf:
                n_shelf = self.min_n_shelf
                idx_array = np.tile(np.arange(solution_dim), (batch_size, 1))
                shelf_idxs = self.rng.permuted(idx_array, axis=1)[:, :n_shelf]
                sols = np.zeros((batch_size, solution_dim), dtype=int)
                for i in range(batch_size):
                    sols[i, shelf_idxs[i]] = 1
                assert np.sum(sols) == batch_size * n_shelf
            else:
                # If we still have only 2 objects, we can generate
                # solutions in a biased fashion and keep generate until we
                # have a the specified number of shelves.
                if self.num_objects == 2:
                    sols = []
                    for _ in range(batch_size):
                        # Keep generate new solutions until we get desired
                        # number of shelves
                        sol = np.ones(solution_dim, dtype=int)
                        while not (self.min_n_shelf <= np.sum(sol) <=
                                   self.max_n_shelf):
                            sol = self.rng.choice(
                                np.arange(self.num_objects),
                                size=solution_dim,
                                p=[
                                    1 - self.max_n_shelf / solution_dim,
                                    self.max_n_shelf / solution_dim
                                ],
                            )
                        sols.append(sol)
        # If we have more than 2 objects, we just generate new
        # solutions directly
        else:
            sols = self.rng.integers(self.num_objects,
                                     size=(batch_size, solution_dim))

        return np.array(sols), None

    def get_cma_initial_mean(self, n_emitters: int, initial_mean: float):
        """Return initial mean for CMA-based optimizers.
        """
        if self.search_space == SearchSpace.CHUTE_CAPACITIES:
            if self.heuristic_chute_mapping is None:
                return np.zeros(
                    (n_emitters, self.get_sol_size())) + initial_mean
            else:
                all_chute_capacities = []
                # assert n_emitters == len(self.heuristic_chute_mapping)
                for chute_mapping_file in self.heuristic_chute_mapping:
                    with open(chute_mapping_file, "r") as f:
                        chute_mapping = json.load(f)
                    chute_capacities = compute_chute_capacities(
                        chute_mapping,
                        self.chute_locs,
                        self.package_dist_weight,
                    )
                    all_chute_capacities.append(chute_capacities)
                return np.array(all_chute_capacities)
        else:
            return np.zeros((n_emitters, self.get_sol_size())) + initial_mean

    def em_train(self):
        self.emulation_model.train()

    def _nca_generate_envs(self, sols):
        """Helper function to generate environments using NCA parameters.

        Args:
            sols: Emitted solutions.

        Returns:
            envs: NCA generated environments.
        """
        # nca_start_time = time.time()
        n_envs = len(sols)

        # Splitting into 5 workers empirically brings the optimal balance
        # between the overhead of creating NCA model and parallelization of NCA
        # generation
        n_workers = min(5, n_envs)
        sols_split = np.array_split(sols, n_workers)
        pool = multiprocessing.Pool(n_workers)
        envs = pool.starmap(
            _nca_generate_one_slice_env,
            zip(
                sols_split,
                repeat(self.seed_map_torch, n_workers),
                repeat(self.nca_iter, n_workers),
            ),
        )
        envs = torch.cat(envs)
        # envs = np.array(envs).reshape(
        #     (n_envs, self.env_height, self.env_width)).astype(int)
        # nca_time_lapsed = time.time() - nca_start_time
        # logger.info(f"NCA takes {round(nca_time_lapsed, 3)} seconds")

        ########## Single-process version ##########
        # n_envs = len(sols)
        # nca_start_time = time.time()
        # # warehouseNCA = WarehouseNCA()
        # envs = []
        # # for i in tqdm(range(n_envs)):
        # for i in range(n_envs):
        #     self.warehouseNCA.set_params(sols[i])
        #     env, _ = self.warehouseNCA.generate(
        #         self.seed_map_torch,
        #         n_iter=self.nca_iter,
        #     )
        #     envs.append(env)
        # envs = torch.cat(envs)
        # nca_time_lapsed = time.time() - nca_start_time
        # logger.info(f"NCA takes {round(nca_time_lapsed, 3)} seconds")
        ########## Single-process version ##########
        return envs

    def emulation_pipeline(self, sols):
        """Pipeline that takes solutions and uses the emulation model to predict
        the objective and measures.

        Args:
            sols: Emitted solutions.

        Returns:
            envs: Generated envs.
            objs: Predicted objective values.
            measures: Predicted measure values.
            success_mask: Array of size `len(envs)`. An element in the array is
                False if some part of the prediction pipeline failed for the
                corresponding solution.
        """
        n_maps = len(sols)

        # For NCA, use NCA model to generate actual envs
        if self.is_nca:
            maps = self._nca_generate_envs(sols)
        else:
            maps = torch.tensor(sols, dtype=int).reshape(
                (n_maps, self.env_height, self.env_width))

        # Add l and r block in a batched fashion
        ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if self.w_mode \
                            else KIVA_ROBOT_BLOCK_WIDTH
        ADDITION_BLOCK_HEIGHT = 0 if self.w_mode else KIVA_ROBOT_BLOCK_HEIGHT

        l_block, r_block = get_additional_h_blocks(ADDITION_BLOCK_WIDTH,
                                                   self.env_height,
                                                   self.w_mode)

        # Repeat boths blocks by n_maps times
        l_blocks = torch.tensor(
            np.tile(l_block, reps=(n_maps, 1, 1)),
            dtype=int,
            # device=DEVICE,
        )
        r_blocks = torch.tensor(
            np.tile(r_block, reps=(n_maps, 1, 1)),
            dtype=int,
            # device=DEVICE,
        )
        map_comps = torch.cat((l_blocks, maps, r_blocks), dim=2)

        if ADDITION_BLOCK_HEIGHT > 0:
            n_col_comp = self.env_width + 2 * ADDITION_BLOCK_WIDTH
            t_block, b_block = \
                get_additional_v_blocks(ADDITION_BLOCK_HEIGHT,
                                        n_col_comp, self.w_mode)
            t_blocks = torch.tensor(
                np.tile(t_block, reps=(n_maps, 1, 1)),
                dtype=int,
                # device=DEVICE,
            )
            b_blocks = torch.tensor(
                np.tile(b_block, reps=(n_maps, 1, 1)),
                dtype=int,
                # device=DEVICE,
            )
            map_comps = torch.cat((t_blocks, map_comps, b_blocks), dim=1)

        # Same as MILP, in the surrogate model, we replace 'w' with 'r' under
        # w_mode to use 'r' internally.
        if self.w_mode:
            map_comps = flip_tiles_torch(
                map_comps,
                'w',
                'r',
            )

        # futures = [
        #     self.client.submit(
        #         get_comp_map,
        #         map=map,
        #         seed=self.seed, # This is the master seed
        #         w_mode=self.w_mode,
        #         n_endpt=self.n_endpt,
        #         env_height=self.env_height,
        #     ) for map in maps
        # ]

        # map_comps = self.client.gather(futures)
        # map_comps = np.array(map_comps)

        assert map_comps.shape == (
            n_maps,
            self.env_height + 2 * ADDITION_BLOCK_HEIGHT,
            self.env_width + 2 * ADDITION_BLOCK_WIDTH,
        )

        success_mask = np.ones(len(map_comps), dtype=bool)
        objs, measures = self.emulation_model.predict(map_comps)
        return map_comps, objs, measures, success_mask

    def _repair_sols(self, sols, parent_sols=None, partial_repair=False):
        """Helper function to perform MILP repair

        Args:
            sols: Emitted solution.
            parent_sols: Parent solution of sols.

        Returns:
            List[tuple(map_json, unrepaired_env_int, repaired_env_int)].
        """
        n_envs = len(sols)

        # For NCA, use NCA model to generate actual envs
        if self.is_nca:
            unrepaired_envs_int = self._nca_generate_envs(sols)
            unrepaired_envs_int = unrepaired_envs_int.cpu().numpy()
        else:
            unrepaired_envs_int = np.array(sols).reshape(
                (n_envs, self.env_height, self.env_width)).astype(int)

        if parent_sols is not None:
            parent_envs = np.array(parent_sols)
        else:
            parent_envs = [None] * n_envs

        repair_start_time = time.time()
        # First, repair the maps
        repair_futures = [
            self.client.submit(
                repair_warehouse,
                unrepaired_env_int=unrepaired_env_int,
                parent_repaired_env=parent_env,
                repair_seed=self.seed,
                w_mode=self.w_mode,
                min_n_shelf=self.min_n_shelf,
                max_n_shelf=self.max_n_shelf,
                partial_repair=partial_repair,
            ) for unrepaired_env_int, parent_env in zip(
                unrepaired_envs_int, parent_envs)
        ]

        repair_results = self.client.gather(repair_futures)
        self.repair_runtime += time.time() - repair_start_time
        return repair_results

    def _repair_chute_mapping(self, sols):
        """Helper function to generate chute mapping using MILP

        Args:
            sols: Emitted solution.

        Returns:
            List of chute mappings
        """
        repair_start_time = time.time()
        # Convert sols to chute mapping
        unrepaired_chute_mappings = []
        for sol in sols:
            unrepaired_chute_mapping = {
                j: []
                for j in range(self.n_destinations)
            }
            for i, c_loc in enumerate(self.chute_locs):
                unrepaired_chute_mapping[sol[i]].append(c_loc)
            unrepaired_chute_mappings.append(unrepaired_chute_mapping)

        # Repair the chute mapping
        chute_mapping_futures = [
            self.client.submit(
                repair_chute_mapping,
                chute_locs=self.chute_locs,
                unrepaired_chute_mapping=unrepaired_chute_mapping,
                destination_volumes=self.package_dist_weight,
                repair_seed=self.seed,
            ) for unrepaired_chute_mapping in unrepaired_chute_mappings
        ]

        chute_mappings = self.client.gather(chute_mapping_futures)

        # ###### Debug failure handling ######
        # for i in range(len(chute_mappings)):
        #     rnd = self.rng.random()
        #     if rnd < 0.5:
        #         chute_mappings[i] = None
        # ###### END Debug failure handling ######

        self.repair_runtime += time.time() - repair_start_time
        return chute_mappings

    def _gen_chute_mapping_capacity(self, sols, batch_idx):
        """Helper function to generate chute mapping using MILP

        Args:
            sols: Emitted solution.

        Returns:
            List of chute mappings
        """
        repair_start_time = time.time()

        if self.bound_handle == "transformation":
            chute_capacities = np.array(
                [self.bound_transform.repair(sol) for sol in sols])
        elif self.bound_handle == "softmax":
            chute_capacities = softmax(sols, axis=1)
        elif self.bound_handle == "clip":
            lb, ub = self.bounds
            chute_capacities = np.clip(sols, lb, ub)
        else:
            logger.error(f"Unknown bound handling method {self.bound_handle}")

        # Normalize the chute capacities s.t. they sum up to
        # n_chutes/n_destinations
        # exp_deg = self.n_chutes / len(self.package_dist_weight)
        # if batch_idx > 0:
        chute_capacities /= np.sum(chute_capacities, axis=1, keepdims=True)
        # chute_capacities *= exp_deg
        # Generate the chute mapping
        chute_mapping_futures = [
            self.client.submit(
                gen_chute_mapping_from_capacity,
                chute_locs=self.chute_locs,
                chute_capacities=sol,
                destination_volumes=self.package_dist_weight,
                repair_seed=self.seed,
            ) for sol in chute_capacities
        ]

        chute_mappings = self.client.gather(chute_mapping_futures)

        # ###### Debug: plotting repaired chute mapping ######
        # from env_search.analysis.plot_chute_mapping import plot_chute_mapping
        # for i, chute_mapping in enumerate(chute_mappings):
        #     plot_chute_mapping(self.base_map_np, f"debug_chute_mapping_{i}",
        #                        chute_mapping, 10)
        # breakpoint()

        # ###### Debug failure handling ######
        # for i in range(len(chute_mappings)):
        #     rnd = self.rng.random()
        #     if rnd < 0.5:
        #         chute_mappings[i] = None
        # ###### END Debug failure handling ######

        self.repair_runtime += time.time() - repair_start_time
        return chute_mappings, chute_capacities

    def _run_sims(self,
                  n_envs,
                  repair_results,
                  batch_idx,
                  model_params=None,
                  chute_mappings=None,
                  chute_capacities=None,
                  task_assignment_params=None):
        """Run MAPF simulation with the repaired maps

        Args:
            sols (np.ndarray): solutions
            repair_results (list[tuple]): repaired results returned by
                `_repair_sols`

        Returns:
            list[WarehouseResult]: evaluated results
        """
        if batch_idx is None:
            batch_idx = self.eval_batch_idx
        eval_logdir = self.logdir.pdir(f"evaluations/eval_batch_{batch_idx}")

        evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                             size=(n_envs, self.n_evals),
                                             endpoint=True)

        logger.info("Collecting evaluations")

        sim_start_time = time.time()
        sim_futures = []
        for i in range(n_envs):
            (
                map_json,
                unrepaired_env_int,
                repaired_env_int,
            ) = repair_results[i]
            for j in range(self.n_evals):
                future = self.client.submit(
                    run_warehouse,
                    map_json=map_json,
                    eval_logdir=eval_logdir,
                    sim_seed=int(evaluation_seeds[i, j]),
                    map_id=i,
                    eval_id=j,
                    simulation_algo=self.simulation_algo,
                    model_params=model_params[i]
                    if model_params is not None else None,
                    chute_mapping_json=json.dumps(chute_mappings[i])
                    if chute_mappings is not None else None,
                    package_dist_weight_json=self.package_dist_weight_json,
                    task_assignment_params_json=json.dumps(
                        task_assignment_params[i])
                    if task_assignment_params is not None else None,
                )
                sim_futures.append(future)
                # future = self.module.evaluate_pibt(
                #     map_json=map_json,
                #     output_dir=eval_logdir,
                #     sim_seed=int(evaluation_seeds[i, j]),
                #     chute_mapping_json=json.dumps(chute_mappings[i])
                #     if chute_mappings else None,
                #     package_dist_weight_json=self.package_dist_weight_json,
                # )

        results_json = self.client.gather(sim_futures)
        self.sim_runtime += time.time() - sim_start_time

        results_json_sorted = []
        for i in range(n_envs):
            curr_eval_results = []
            for j in range(self.n_evals):
                curr_eval_results.append(results_json[i * self.n_evals + j])
            results_json_sorted.append(curr_eval_results)

        logger.info("Processing eval results")
        process_futures = []
        for i in range(n_envs):
            (
                map_json,
                unrepaired_env_int,
                repaired_env_int,
            ) = repair_results[i]
            future = self.client.submit(
                process_warehouse_eval_result,
                curr_result_json=results_json_sorted[i],
                n_evals=self.n_evals,
                unrepaired_env_int=unrepaired_env_int,
                repaired_env_int=repaired_env_int,
                edge_weights=None,
                wait_costs=None,
                w_mode=self.w_mode,
                max_n_shelf=self.max_n_shelf,
                map_id=i,
                simulation_algo=self.simulation_algo,
                chute_mapping=chute_mappings[i]
                if chute_mappings is not None else None,
                chute_capacities=chute_capacities[i]
                if chute_capacities is not None else None,
            )
            process_futures.append(future)

        results = self.client.gather(process_futures)
        return results

    def _run_piu(self,
                 piu_sols,
                 repair_results,
                 batch_idx,
                 chute_mappings=None,
                 task_assignment_params=None):
        n_sols = len(piu_sols)
        eval_logdir = self.logdir.pdir(f"evaluations/eval_batch_{batch_idx}")
        evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                             size=n_sols,
                                             endpoint=True)
        sim_start_time = time.time()

        sim_futures = []
        all_n_valid_vertices = []
        all_n_valid_edges = []
        for i in range(n_sols):
            map_json_str, _, repaired_env_int = repair_results[i]

            # If repair is failed, consider the simulation as failed
            n_valid_vertices = -1
            n_valid_edges = -1
            if repaired_env_int is not None:
                n_valid_vertices = get_n_valid_vertices(repaired_env_int,
                                                        domain=self.domain)
                n_valid_edges = get_n_valid_edges(
                    repaired_env_int,
                    bi_directed=True,
                    domain=self.domain,
                )
            all_n_valid_vertices.append(n_valid_vertices)
            all_n_valid_edges.append(n_valid_edges)
            # self.module.evaluate_iterative_update(
            #     repaired_env_int,
            #     map_json_str,
            #     piu_sols[0],
            #     n_valid_edges,
            #     n_valid_vertices,
            #     str(eval_logdir),
            #     evaluation_seeds[0],
            #     chute_mapping_json=json.dumps(chute_mappings[i])
            #     if chute_mappings is not None else None,
            #     task_assignment_params_json=json.dumps(
            #         task_assignment_params[i])
            #     if task_assignment_params is not None else None,
            # )
            sim_futures.append(
                self.client.submit(
                    run_warehouse_iterative_update,
                    env_np=repaired_env_int,
                    map_json_str=map_json_str,
                    n_valid_edges=n_valid_edges,
                    n_valid_vertices=n_valid_vertices,
                    model_params=piu_sols[i],
                    output_dir=eval_logdir,
                    seed=evaluation_seeds[i],
                    chute_mapping_json=json.dumps(chute_mappings[i])
                    if chute_mappings is not None else None,
                    task_assignment_params_json=json.dumps(
                        task_assignment_params[i])
                    if task_assignment_params is not None else None,
                ))

        logger.info("Collecting evaluations")
        results_and_weights = self.client.gather(sim_futures)
        self.sim_runtime += time.time() - sim_start_time
        results_json = []
        # all_weights = []
        all_edge_weights = []
        all_wait_costs = []
        for i in range(n_sols):
            result_json, curr_weights = results_and_weights[i]
            results_json.append(result_json)
            # all_weights.append(curr_weights)
            if curr_weights is not None:
                all_edge_weights.append(curr_weights[all_n_valid_vertices[i]:])
                all_wait_costs.append(curr_weights[:all_n_valid_vertices[i]])
            else:
                all_edge_weights.append(None)
                all_wait_costs.append(None)

        logger.info("Processing eval results")

        process_futures = []
        for i in range(n_sols):
            _, unrepaired_env_int, repaired_env_int = repair_results[i]
            process_futures.append(
                self.client.submit(
                    process_warehouse_eval_result,
                    curr_result_json=[results_json[i]],
                    unrepaired_env_int=unrepaired_env_int,
                    repaired_env_int=repaired_env_int,
                    edge_weights=all_edge_weights[i],
                    wait_costs=all_wait_costs[i],
                    w_mode=self.w_mode,
                    max_n_shelf=self.max_n_shelf,
                    n_evals=1,  # For PIU, n_eval should be fixed to 1
                    map_id=i,
                    simulation_algo=self.simulation_algo,
                ))
        results = self.client.gather(process_futures)
        return results

    def _fabricate_repair_results(self, sols):
        # Fabricate `repair_results` to be used for Chute capacity search
        repair_results = [
            (
                json.dumps(self.base_map_json),
                self.base_map_np,  # No unrepaired map
                self.base_map_np,
            ) for _ in range(len(sols))
        ]
        return repair_results

    def _create_g_graph_maps(self, sols):
        """Create map jsons with edge weights for offline GGO

        Args:
            sols (np.ndarray): edge weights. Assumed to be of size
                (n_sols, `self.n_valid_vertices` + `self.n_valid_edges`)
        """
        # Apply transformation to edge weights
        lb, ub = self.module.config.bounds
        wait_costs = sols[:, :self.n_valid_vertices]
        edge_weights = sols[:, self.n_valid_vertices:]
        wait_costs = min_max_normalize_2d(wait_costs, lb, ub)
        edge_weights = min_max_normalize_2d(edge_weights, lb, ub)
        sols = np.concatenate([wait_costs, edge_weights], axis=1)

        # Construct the map_jsons with edge weights
        repair_results = []
        for sol in sols:
            map_json = copy.deepcopy(self.base_map_json)
            map_json["weights"] = sol.tolist()
            map_json["weight"] = True
            repair_results.append(
                (json.dumps(map_json), self.base_map_np, self.base_map_np))
        return repair_results

    def eval_pipeline(self, sols, parent_sols=None, batch_idx=None):
        """Pipeline that takes a solution and evaluates it.

        Args:
            sols: Emitted solution.
            parent_sols: Parent solution of sols.

        Returns:
            Results of the evaluation.
        """
        # Make each solution evaluation have a different seed. Note that we
        # assign seeds to solutions rather than workers, which means that we
        # are agnostic to worker configuration.
        n_sols = len(sols)
        # evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
        #                                      size=n_sols,
        #                                      endpoint=True)

        # Split repair and evaluate.
        # Since evaluation might take a lot longer than repair, and each
        # evaluation might includes several simulations, we want to distribute
        # all the simulations to the workers instead of evaluations to fully
        # exploit the available compute

        # Co-optimize guidance graph and layout
        if self.search_space == SearchSpace.LAYOUT_G_GRAPH:
            # Currently only works while searching for NCA and PIU concurrently
            assert self.is_nca and self.is_piu
            assert self.nca_n_param + self.update_mdl_n_param == sols.shape[1]
            nca_sols = sols[:, :self.nca_n_param]
            piu_sols = sols[:, self.nca_n_param:]

            repair_results = self._repair_sols(nca_sols, parent_sols)
            # if self.is_piu:
            results = self._run_piu(piu_sols, repair_results, batch_idx)

        # Optimize layout only
        elif self.search_space == SearchSpace.LAYOUT:
            repair_results = self._repair_sols(sols, parent_sols)
            results = self._run_sims(n_sols, repair_results, batch_idx)
        # Optimize piu only
        elif self.search_space == SearchSpace.G_GRAPH:
            # Fabricate `repair_results` to be used for GGO
            chute_mappings = [self.base_chute_mapping_json] * n_sols

            if not self.is_piu:
                repair_results = self._create_g_graph_maps(sols)
                results = self._run_sims(n_sols,
                                         repair_results,
                                         batch_idx,
                                         chute_mappings=chute_mappings)
            else:
                repair_results = self._fabricate_repair_results(sols)
                results = self._run_piu(sols,
                                        repair_results,
                                        batch_idx,
                                        chute_mappings=chute_mappings)
        elif self.search_space == SearchSpace.G_POLICY:
            # Fabricate `repair_results` to be used for guidance policy
            # optimization
            repair_results = self._fabricate_repair_results(sols)
            chute_mappings = [self.base_chute_mapping_json] * n_sols

            # Evaluate with the parameters of guidance policy
            results = self._run_sims(
                n_sols,
                repair_results,
                batch_idx,
                model_params=sols,
                chute_mappings=chute_mappings,
                chute_capacities=None,
            )

        elif self.search_space == SearchSpace.CHUTE_CAPACITIES:
            # Generate the chute mappings
            chute_mappings, chute_capacities = self._gen_chute_mapping_capacity(
                sols, batch_idx)

            g_policy_params = [self.g_policy_params] * n_sols
            t_policy_params = [self.t_policy_params] * n_sols

            # Fabricate `repair_results` to be used for Chute capacity search
            repair_results = self._fabricate_repair_results(sols)

            results = self._run_sims(
                n_sols,
                repair_results,
                batch_idx,
                chute_mappings=chute_mappings,
                chute_capacities=chute_capacities,
                model_params=g_policy_params,
                task_assignment_params=t_policy_params,
            )
        elif self.search_space == SearchSpace.CHUTE_MAPPING:
            # Repair the chute mapping
            chute_mappings = self._repair_chute_mapping(sols)

            g_policy_params = [self.g_policy_params] * n_sols
            t_policy_params = [self.t_policy_params] * n_sols

            # Fabricate `repair_results` to be used for Chute capacity search
            repair_results = self._fabricate_repair_results(sols)

            results = self._run_sims(
                n_sols,
                repair_results,
                batch_idx,
                chute_mappings=chute_mappings,
                model_params=g_policy_params,
                task_assignment_params=t_policy_params,
            )
        elif self.search_space == SearchSpace.G_POLICY_CHUTE_CAPACITIES:
            # Generate the chute mappings
            # The first `self.n_chutes` params are for chute capacities
            chute_mappings, chute_capacities = self._gen_chute_mapping_capacity(
                sols[:, :self.n_chutes], batch_idx)

            # Fabricate `repair_results` to be used for Chute capacity search
            repair_results = self._fabricate_repair_results(sols)

            # Evaluate with the parameters of guidance policy
            results = self._run_sims(
                n_sols,
                repair_results,
                batch_idx,
                model_params=sols[:, self.n_chutes:],
                chute_mappings=chute_mappings,
                chute_capacities=chute_capacities,
            )
        elif self.search_space == SearchSpace.G_POLICY_TASK_ASSIGN_POLICY:
            # Fabricate `repair_results` to be used for guidance policy
            # optimization
            repair_results = self._fabricate_repair_results(sols)
            chute_mappings = [self.base_chute_mapping_json] * n_sols

            # Evaluate with the parameters of guidance policy
            results = self._run_sims(
                n_sols,
                repair_results,
                batch_idx,
                model_params=sols[:, :self.update_mdl_n_param],
                chute_mappings=chute_mappings,
                chute_capacities=None,
                task_assignment_params=sols[:,
                                            self.update_mdl_n_param:].tolist(),
            )
        elif self.search_space == SearchSpace.TASK_ASSIGN_POLICY:
            # Fabricate `repair_results` to be used for task assignment policy
            # optimization
            repair_results = self._fabricate_repair_results(sols)
            chute_mappings = [self.base_chute_mapping_json] * n_sols
            g_policy_params = [self.g_policy_params] * n_sols

            # Evaluate with the parameters of guidance policy
            results = self._run_sims(
                n_sols,
                repair_results,
                batch_idx,
                model_params=g_policy_params,
                chute_mappings=chute_mappings,
                chute_capacities=None,
                task_assignment_params=sols.tolist(),
            )

        elif self.search_space == SearchSpace.G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY:
            # Generate the chute mappings
            # The first `self.n_chutes` params are for chute
            # capacities
            chute_mappings, chute_capacities = self._gen_chute_mapping_capacity(
                sols[:, :self.n_chutes], batch_idx)

            g_policy_params = sols[:, self.n_chutes:self.n_chutes +
                                   self.update_mdl_n_param]
            task_assignment_params = sols[:, self.n_chutes +
                                          self.update_mdl_n_param:]

            # Fabricate `repair_results` to be used for Chute capacity search
            repair_results = self._fabricate_repair_results(sols)

            results = self._run_sims(
                n_sols,
                repair_results,
                batch_idx,
                model_params=g_policy_params,
                chute_mappings=chute_mappings,
                chute_capacities=chute_capacities,
                task_assignment_params=task_assignment_params.tolist(),
            )

        elif self.search_space == SearchSpace.G_GRAPH_CHUTE_CAPACITIES:
            # The first `self.n_chutes` params are for chute capacities
            chute_mappings, chute_capacities = self._gen_chute_mapping_capacity(
                sols[:, :self.n_chutes], batch_idx)
            if not self.is_piu:
                repair_results = self._create_g_graph_maps(
                    sols[:, self.n_chutes:])
                results = self._run_sims(n_sols,
                                         repair_results,
                                         batch_idx,
                                         chute_mappings=chute_mappings,
                                         chute_capacities=chute_capacities)
            else:
                repair_results = self._fabricate_repair_results(sols)
                results = self._run_piu(sols[:, self.n_chutes:],
                                        repair_results,
                                        batch_idx,
                                        chute_mappings=chute_mappings)

        elif self.search_space == SearchSpace.G_GRAPH_TASK_ASSIGN_POLICY:
            chute_mappings = [self.base_chute_mapping_json] * n_sols
            if not self.is_piu:
                # The first `self.n_g_graph_weights` params are for g graph
                repair_results = self._create_g_graph_maps(
                    sols[:, :self.n_g_graph_weights])
                results = self._run_sims(
                    n_sols,
                    repair_results,
                    batch_idx,
                    chute_mappings=chute_mappings,
                    task_assignment_params=sols[:, self.
                                                n_g_graph_weights:].tolist())
            else:
                repair_results = self._fabricate_repair_results(sols)
                # The first `self.update_mdl_n_param` params are for PIU
                results = self._run_piu(
                    sols[:, :self.update_mdl_n_param],
                    repair_results,
                    batch_idx,
                    chute_mappings=chute_mappings,
                    task_assignment_params=sols[:, self.
                                                update_mdl_n_param:].tolist())

        elif self.search_space == SearchSpace.G_GRAPH_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY:
            chute_mappings, chute_capacities = self._gen_chute_mapping_capacity(
                sols[:, :self.n_chutes], batch_idx)
            g_graph_params = sols[:, self.n_chutes:self.n_chutes +
                                  self.n_g_graph_weights]
            task_assignment_params = sols[:, self.n_chutes +
                                          self.n_g_graph_weights:]

            if not self.is_piu:
                repair_results = self._create_g_graph_maps(g_graph_params)
                results = self._run_sims(
                    n_sols,
                    repair_results,
                    batch_idx,
                    chute_mappings=chute_mappings,
                    chute_capacities=chute_capacities,
                    task_assignment_params=task_assignment_params.tolist())
            else:
                repair_results = self._fabricate_repair_results(sols)
                results = self._run_piu(
                    g_graph_params,
                    repair_results,
                    batch_idx,
                    chute_mappings=chute_mappings,
                    task_assignment_params=task_assignment_params.tolist())

        return results

    def inner_obj_pipeline(self, sols, parent_sols=None):
        """Evaluate the solutions on inner objective and return the results

        Args:
            sols: Emitted solution.
            parent_sols: Parent solution of sols.

        Returns:
            Results of the evaluation.
        """
        repair_results = self._repair_sols(sols,
                                           parent_sols,
                                           partial_repair=True)
        # Just calculating sim score and measures here, no need for
        # parallelization
        results = []
        for repair_result in repair_results:
            _, unrepaired_env_int, repaired_env_int = repair_result
            result = self.module.process_inner_obj_results(
                unrepaired_env_int,
                repaired_env_int,
                self.w_mode,
            )
            results.append(result)
        return results

    def add_experience(self, sol, result):
        """Add required experience to the emulation model based on the solution
        and the results.

        Args:
            sol: Emitted solution.
            result: Evaluation result.
        """
        obj = result.agg_obj
        meas = result.agg_measures
        input_env = result.raw_metadata["unrepaired_env_int"]
        repaired_env = result.raw_metadata["repaired_env_int"]

        # Same as MILP, we replace 'w' with 'r' and use 'r' internally in
        # emulation model
        if self.w_mode:
            input_env = flip_tiles(input_env, 'w', 'r')
            repaired_env = flip_tiles(repaired_env, 'w', 'r')

        if self.emulation_model.pre_network is not None:
            # Mean of tile usage over n_evals
            avg_tile_usage = np.mean(result.raw_metadata["tile_usage"], axis=0)
            if isinstance(self.emulation_model.pre_network,
                          WarehouseAugResnetOccupancy):
                self.emulation_model.add(
                    AugExperience(sol, input_env, obj, meas, avg_tile_usage))
            elif isinstance(self.emulation_model.pre_network,
                            WarehouseAugResnetRepairedMapAndOccupancy):
                self.emulation_model.add(
                    DoubleAugExperience(sol, input_env, obj, meas,
                                        avg_tile_usage, repaired_env))
        else:
            self.emulation_model.add(Experience(sol, input_env, obj, meas))

    @staticmethod
    def add_failed_info(sol, result) -> dict:
        """Returns a dict containing relevant information about failed envs.

        Args:
            sol: Emitted solution.
            result: Evaluation result.

        Returns:
            Dict with failed level information.
        """
        failed_level_info = {
            "solution": sol,
            "unrepaired_env_int": result.raw_metadata["unrepaired_env_int"],
            "log_message": result.log_message,
        }
        return failed_level_info

    def get_sol_size(self):
        """Get number of parameters to optimize.
        """
        logger.info(f"Search space: {self.search_space}")

        # Co-optimize guidance graph and layout
        if self.search_space == SearchSpace.LAYOUT_G_GRAPH:
            assert self.is_nca and self.is_piu
            return self.nca_n_param + self.update_mdl_n_param

        # Optimize layout only
        elif self.search_space == SearchSpace.LAYOUT:
            if self.is_nca:
                return self.nca_n_param
            else:
                return self.env_height * self.env_width

        # Optimize guidance graph only
        elif self.search_space == SearchSpace.G_GRAPH:
            if self.is_piu:
                return self.update_mdl_n_param
            else:
                return self.n_g_graph_weights

        # Optimize guidance policy only
        elif self.search_space == SearchSpace.G_POLICY:
            return self.update_mdl_n_param

        # Optimize chute capacities or chute mapping directly
        elif self.search_space in [
                SearchSpace.CHUTE_CAPACITIES,
                SearchSpace.CHUTE_MAPPING,
        ]:
            return self.n_chutes

        # Co-optimize guidance policy and chute capacities
        elif self.search_space == SearchSpace.G_POLICY_CHUTE_CAPACITIES:
            return self.update_mdl_n_param + self.n_chutes

        # Co-optimize guidance policy and a quadratic task assignment policy
        elif self.search_space == SearchSpace.G_POLICY_TASK_ASSIGN_POLICY:
            return self.update_mdl_n_param + QUAD_TASK_ASSIGN_N_PARAM

        # Optimize task assignment policy
        elif self.search_space == SearchSpace.TASK_ASSIGN_POLICY:
            if self.module.config.task_assignment_cost == "opt_quadratic_f":
                return QUAD_TASK_ASSIGN_N_PARAM
            else:
                raise NotImplementedError(
                    f"Task assignment {self.module.config.task_assignment_cost} is not optimizable"
                )

        # Co-optimize guidance policy, chute capacities, and task assignment
        # policy
        elif self.search_space == SearchSpace.G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY:
            return self.update_mdl_n_param + self.n_chutes + QUAD_TASK_ASSIGN_N_PARAM

        # Optimize guidance graph and task assignment policy
        elif self.search_space == SearchSpace.G_GRAPH_TASK_ASSIGN_POLICY:
            if self.is_piu:
                return self.update_mdl_n_param + QUAD_TASK_ASSIGN_N_PARAM
            else:
                return self.n_g_graph_weights + QUAD_TASK_ASSIGN_N_PARAM

        # Optimize guidance graph and chute capacities
        elif self.search_space == SearchSpace.G_GRAPH_CHUTE_CAPACITIES:
            if self.is_piu:
                return self.update_mdl_n_param + self.n_chutes
            else:
                return self.n_g_graph_weights + self.n_chutes

        # Optimize chute capacities, guidance graph, and task assignment policy
        elif self.search_space == SearchSpace.G_GRAPH_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY:
            if self.is_piu:
                return self.n_chutes + self.update_mdl_n_param + QUAD_TASK_ASSIGN_N_PARAM
            else:
                return self.n_chutes + self.n_g_graph_weights + QUAD_TASK_ASSIGN_N_PARAM

        else:
            raise NotImplementedError(
                f"Search space {self.search_space} is not supported yet")
