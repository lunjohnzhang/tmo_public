import copy
import json
from typing import Optional, List

import gin
import numpy as np
import ribs
from ribs.emitters import EmitterBase
from env_search.warehouse import get_packages
from env_search.warehouse.module import compute_chute_capacities
from env_search.utils import (read_in_sortation_map, sortation_env_str2number,
                              get_chute_loc, chute_mapping_to_array)


@gin.configurable(denylist=["archive", "x0", "seed"])
class MapElitesBaselineChuteMappingEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    map layout.

    Args:
        archive: Archive to store the solutions.
        x0: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound. (default: None)
        seed: Random seed. (default None)
        num_objects: Solutions will be generated as ints between
            [0, num_objects)
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
        geometric_k: Whether to vary k geometrically. If it is True,
            `mutation_k` will be ignored.
        max_n_shelf: max number of shelves(index 1).
        min_n_shelf: min number of shelves(index 1).
    """

    def __init__(
        self,
        archive: ribs.archives.ArchiveBase,
        x0: np.ndarray,
        bounds: Optional["array-like"] = None,  # type: ignore
        seed: int = None,
        initial_chute_mappings: List[str] = None,
        base_map_path: str = gin.REQUIRED,
        num_objects: int = gin.REQUIRED,
        batch_size: int = gin.REQUIRED,
        initial_population: int = gin.REQUIRED,
        mutation_k: int = gin.REQUIRED,
        geometric_k: bool = gin.REQUIRED,
        package_mode: str = "dist",
        package_path: str = None,
        package_dist_type: str = "721",
    ):
        solution_dim = len(x0)  # n_chutes
        super().__init__(
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.num_objects = num_objects  # n_destinations
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        self.geometric_k = geometric_k
        self.initial_chute_mappings = initial_chute_mappings

        # Read in map as json, str, and np
        with open(base_map_path, "r") as f:
            self.base_map_json = json.load(f)
            self.base_map_str, _ = read_in_sortation_map(base_map_path)
            self.base_map_np = sortation_env_str2number(self.base_map_str)
            self.domain = "sortation"
            self.chute_locs = get_chute_loc(self.base_map_np)
            self.n_chutes = len(self.chute_locs)

        # Read in package distribution
        self.package_dist_weight, _ = get_packages(
            package_mode,
            package_dist_type,
            package_path,
            self.num_objects,
        )

        if not self.geometric_k:
            assert solution_dim >= self.mutation_k

        self.sols_emitted = 0

    def ask(self):
        if self.sols_emitted < self.initial_population:
            # Read in chute mappings from the initial list, if any
            init_mappings = []
            if self.initial_chute_mappings is not None:
                for file in self.initial_chute_mappings:
                    with open(file, "r") as f:
                        chute_mapping = json.load(f)
                        init_mappings.append(
                            chute_mapping_to_array(chute_mapping,
                                                   self.chute_locs))
                init_mappings = np.array(init_mappings)

            # Generate extra random solutions, if necessary
            extra_sols = []
            if self.batch_size - len(init_mappings) > 0:
                extra_sols = self.rng.choice(
                    np.arange(self.num_objects),
                    size=(self.batch_size - len(init_mappings),
                          self.solution_dim),
                    p=self.package_dist_weight,
                )
            # sols = self.rng.integers(
            #     self.num_objects,
            #     size=(self.batch_size - len(init_mappings), self.solution_dim))

            # Combine the initial mappings and extra solutions
            if len(init_mappings) > 0 and len(extra_sols) > 0:
                sols = np.vstack((init_mappings, extra_sols))
            elif len(init_mappings) > 0:
                sols = init_mappings
            else:
                sols = extra_sols

            # Clip the number of solutions to the initial population
            if self.initial_population < self.batch_size:
                self.sols_emitted += self.initial_population
                sols = sols[:self.initial_population]
            else:
                self.sols_emitted += self.batch_size
            return np.array(sols), None

        # Mutate current solutions
        else:
            sols = []
            # parent_sols = []
            repaired_parent_sols = []

            # select k spots randomly without replacement
            # and calculate the random replacement values
            curr_k = self.sample_k(self.solution_dim)
            idx_array = np.tile(np.arange(self.solution_dim),
                                (self.batch_size, 1))
            mutate_idxs = self.rng.permuted(idx_array, axis=1)[:, :curr_k]
            mutate_vals = self.rng.integers(self.num_objects,
                                            size=(self.batch_size, curr_k))

            parent_sols = self.archive.sample_elites(self.batch_size)
            for i in range(self.batch_size):
                parent_sol = parent_sols.solution_batch[i]
                meta = parent_sols.metadata_batch[i]
                sol = copy.deepcopy(parent_sol.astype(int))

                # # Mutate with the goal of balancing the chute capacity
                # # 1. Compute chute capacities
                # curr_mapping = copy.deepcopy(
                #     meta["raw_metadata"]["chute_mapping"])
                # chute_capacities = compute_chute_capacities(
                #     curr_mapping, self.chute_locs, self.package_dist_weight)
                # chute_capacities = chute_capacities / np.sum(chute_capacities)

                # # 2. Sample k chutes to mutate based on the capacity
                # mutate_chutes = self.rng.choice(
                #     self.chute_locs,
                #     size=curr_k,
                #     p=chute_capacities,
                #     replace=False,
                # )
                # for mutate_chute in mutate_chutes:
                #     # 3. Get the destination of the current chute
                #     d = parent_sol[np.where(
                #         self.chute_locs == mutate_chute)[0][0]]
                #     d = int(d)

                #     # 4. This destination requires more chutes, sample a chute
                #     # based on inverse of chute capacities and assign d to it.
                #     # This is to balance the chute capacities.
                #     inv_chute_capacities = 1 / (chute_capacities + 1e-6)
                #     inv_chute_capacities = inv_chute_capacities / np.sum(
                #         inv_chute_capacities)
                #     new_chute = self.rng.choice(
                #         self.chute_locs,
                #         p=inv_chute_capacities,
                #     )
                #     new_chute_curr_d = parent_sol[np.where(
                #         self.chute_locs == new_chute)[0][0]]
                #     if new_chute in curr_mapping[new_chute_curr_d]:
                #         curr_mapping[new_chute_curr_d].remove(new_chute)
                #     curr_mapping[d].append(new_chute)

                # sol = chute_mapping_to_array(curr_mapping, self.chute_locs)
                # sols.append(sol)

                # # Replace with random values
                sol[mutate_idxs[i]] = mutate_vals[i]
                sols.append(sol)

                # Get the repaired parent sol
                if meta is not None and "raw_metadata" in meta:
                    repaired_parent_sols.append(
                        meta["raw_metadata"]["repaired_env_int"])

            self.sols_emitted += self.batch_size
            return np.array(sols), np.array(repaired_parent_sols)

    def sample_k(self, max_k):
        if self.geometric_k:
            curr_k = self.rng.geometric(p=0.5)
            # Clip k if necessary
            if curr_k > max_k:
                curr_k = max_k
        else:
            curr_k = self.mutation_k
        return curr_k


@gin.configurable(denylist=["archive", "x0", "seed"])
class MapElitesBaselineWarehouseEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    map layout.

    Args:
        archive: Archive to store the solutions.
        x0: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound. (default: None)
        seed: Random seed. (default None)
        num_objects: Solutions will be generated as ints between
            [0, num_objects)
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
        geometric_k: Whether to vary k geometrically. If it is True,
            `mutation_k` will be ignored.
        max_n_shelf: max number of shelves(index 1).
        min_n_shelf: min number of shelves(index 1).
    """

    def __init__(
        self,
        archive: ribs.archives.ArchiveBase,
        x0: np.ndarray,
        bounds: Optional["array-like"] = None,  # type: ignore
        seed: int = None,
        num_objects: int = gin.REQUIRED,
        batch_size: int = gin.REQUIRED,
        initial_population: int = gin.REQUIRED,
        mutation_k: int = gin.REQUIRED,
        geometric_k: bool = gin.REQUIRED,
    ):
        solution_dim = len(x0)  # n_chutes
        super().__init__(
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.num_objects = num_objects  # n_destinations
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        self.geometric_k = geometric_k

        if not self.geometric_k:
            assert solution_dim >= self.mutation_k

        self.sols_emitted = 0

    def ask(self):
        if self.sols_emitted < self.initial_population:
            sols = self.rng.integers(self.num_objects,
                                     size=(self.batch_size, self.solution_dim))

            self.sols_emitted += self.batch_size
            return np.array(sols), None

        # Mutate current solutions
        else:
            sols = []
            # parent_sols = []
            repaired_parent_sols = []

            # select k spots randomly without replacement
            # and calculate the random replacement values
            curr_k = self.sample_k(self.solution_dim)
            idx_array = np.tile(np.arange(self.solution_dim),
                                (self.batch_size, 1))
            mutate_idxs = self.rng.permuted(idx_array, axis=1)[:, :curr_k]
            mutate_vals = self.rng.integers(self.num_objects,
                                            size=(self.batch_size, curr_k))

            parent_sols = self.archive.sample_elites(self.batch_size)
            for i in range(self.batch_size):
                parent_sol = parent_sols.solution_batch[i]
                meta = parent_sols.metadata_batch[i]
                sol = copy.deepcopy(parent_sol.astype(int))
                # Replace with random values
                sol[mutate_idxs[i]] = mutate_vals[i]
                sols.append(sol)

                # Get the repaired parent sol
                if meta is not None and "raw_metadata" in meta:
                    repaired_parent_sols.append(
                        meta["raw_metadata"]["repaired_env_int"])

            self.sols_emitted += self.batch_size
            return np.array(sols), np.array(repaired_parent_sols)

    def sample_k(self, max_k):
        if self.geometric_k:
            curr_k = self.rng.geometric(p=0.5)
            # Clip k if necessary
            if curr_k > max_k:
                curr_k = max_k
        else:
            curr_k = self.mutation_k
        return curr_k


@gin.configurable(denylist=["archive", "x0", "seed"])
class MapElitesBaselineMazeEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    mazes.

    Args:
        archive: Archive to store the solutions.
        x0: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound. (default: None)
        seed: Random seed. (default None)
        num_objects: Solutions will be generated as ints between
            [0, num_objects)
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
    """

    def __init__(
            self,
            archive: ribs.archives.ArchiveBase,
            x0: np.ndarray,
            bounds: Optional["array-like"] = None,  # type: ignore
            seed: int = None,
            num_objects: int = gin.REQUIRED,
            batch_size: int = gin.REQUIRED,
            initial_population: int = gin.REQUIRED,
            mutation_k: int = gin.REQUIRED):
        solution_dim = len(x0)
        super().__init__(
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.num_objects = num_objects
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        assert solution_dim >= self.mutation_k

        self.sols_emitted = 0

    def ask(self):
        if self.sols_emitted < self.initial_population:
            self.sols_emitted += self.batch_size
            return self.rng.integers(self.num_objects,
                                     size=(self.batch_size,
                                           self.solution_dim)), None
        else:
            sols = []

            # select k spots randomly without replacement
            # and calculate the random replacement values
            idx_array = np.tile(np.arange(self.solution_dim),
                                (self.batch_size, 1))
            mutate_idxs = self.rng.permuted(idx_array,
                                            axis=1)[:, :self.mutation_k]
            mutate_vals = self.rng.integers(self.num_objects,
                                            size=(self.batch_size,
                                                  self.mutation_k))

            parent_sols = self.archive.sample_elites(self.batch_size)
            for i in range(self.batch_size):
                parent_sol = parent_sols.solution_batch[i]
                sol = copy.deepcopy(parent_sol.astype(int))
                # Replace with random values
                sol[mutate_idxs[i]] = mutate_vals[i]
                sols.append(sol)

            self.sols_emitted += self.batch_size
            return np.array(sols), parent_sols.solution_batch


@gin.configurable(denylist=["archive", "x0", "seed"])
class MapElitesBaselineManufactureEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    map layout.

    Args:
        archive: Archive to store the solutions.
        x0: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound. (default: None)
        seed: Random seed. (default None)
        num_objects: Solutions will be generated as ints between
            [0, num_objects)
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
        geometric_k: Whether to vary k geometrically. If it is True,
            `mutation_k` will be ignored.
        max_n_shelf: max number of shelves(index 1).
        min_n_shelf: min number of shelves(index 1).
    """

    def __init__(
        self,
        archive: ribs.archives.ArchiveBase,
        x0: np.ndarray,
        bounds: Optional["array-like"] = None,  # type: ignore
        seed: int = None,
        num_objects: int = gin.REQUIRED,
        batch_size: int = gin.REQUIRED,
        initial_population: int = gin.REQUIRED,
        mutation_k: int = gin.REQUIRED,
        geometric_k: bool = gin.REQUIRED,
        max_n_shelf: float = gin.REQUIRED,
        min_n_shelf: float = gin.REQUIRED,
    ):
        solution_dim = len(x0)
        super().__init__(
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.num_objects = num_objects
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        self.geometric_k = geometric_k
        self.max_n_shelf = max_n_shelf
        self.min_n_shelf = min_n_shelf

        if not self.geometric_k:
            assert solution_dim >= self.mutation_k

        self.sols_emitted = 0

    def ask(self):
        if self.sols_emitted < self.initial_population:
            sols = self.rng.integers(self.num_objects,
                                     size=(self.batch_size, self.solution_dim))

            self.sols_emitted += self.batch_size
            return np.array(sols), None

        # Mutate current solutions
        else:
            sols = []
            # parent_sols = []
            repaired_parent_sols = []

            # select k spots randomly without replacement
            # and calculate the random replacement values
            curr_k = self.sample_k(self.solution_dim)
            idx_array = np.tile(np.arange(self.solution_dim),
                                (self.batch_size, 1))
            mutate_idxs = self.rng.permuted(idx_array, axis=1)[:, :curr_k]
            mutate_vals = self.rng.integers(self.num_objects,
                                            size=(self.batch_size, curr_k))

            parent_sols = self.archive.sample_elites(self.batch_size)
            for i in range(self.batch_size):
                parent_sol = parent_sols.solution_batch[i]
                meta = parent_sols.metadata_batch[i]
                sol = copy.deepcopy(parent_sol.astype(int))
                # Replace with random values
                sol[mutate_idxs[i]] = mutate_vals[i]
                sols.append(sol)

                # Get the repaired parent sol
                if meta is not None and "manufacture_metadata" in meta:
                    repaired_parent_sols.append(
                        meta["manufacture_metadata"]["repaired_env_int"])

            self.sols_emitted += self.batch_size
            return np.array(sols), np.array(repaired_parent_sols)

    def sample_k(self, max_k):
        if self.geometric_k:
            curr_k = self.rng.geometric(p=0.5)
            # Clip k if necessary
            if curr_k > max_k:
                curr_k = max_k
        else:
            curr_k = self.mutation_k
        return curr_k
