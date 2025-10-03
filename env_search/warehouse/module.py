"""WarehouseConfig and WarehouseModule.

Usage:
    # Run as a script to demo the WarehouseModule.
    python env_search/warehouse/module.py
"""

import os
import gin
import copy
import json
import time
import fire
import logging
import pathlib
import warnings
import warehouse_sim  # type: ignore # ignore pylance warning
import py_driver  # type: ignore # ignore pylance warning
import numpy as np
import shutil
import multiprocessing
import dataclasses
import shortuuid
import subprocess
import hashlib
import gc

from scipy.stats import entropy
from pprint import pprint
from typing import List
from itertools import repeat, product

from queue import Queue
from env_search import LOG_DIR, RUN_DIR
from env_search.iterative_update import WarehouseIterUpdateEnv, PIBTWarehouseOnlineEnv
from env_search.utils.logging import setup_logging, get_current_time_str
from env_search.warehouse import get_packages
from env_search.warehouse.config import WarehouseConfig
from env_search.warehouse.milp_repair import repair_env, partial_repair_env
from env_search.warehouse.milp_chute_cplex import milp_chute_mapping, milp_chute_mapping_edit_dist
# from env_search.warehouse.milp_chute_gurobi import milp_chute_mapping
from env_search.warehouse.warehouse_result import (WarehouseResult,
                                                   WarehouseMetadata)
from env_search.warehouse.update_model import WarehouseBaseUpdateModel, WarehouseCNNUpdateModel
from env_search.utils import (
    kiva_obj_types, KIVA_ROBOT_BLOCK_WIDTH, KIVA_WORKSTATION_BLOCK_WIDTH,
    MIN_SCORE, KIVA_ROBOT_BLOCK_HEIGHT, kiva_env_number2str,
    kiva_env_str2number, format_env_str, read_in_kiva_map, flip_tiles,
    load_pibt_default_config, get_n_valid_vertices, get_n_valid_edges,
    sortation_env_str2number, sortation_env_number2str, BFS_path_len,
    get_workstation_loc, DIRS, sortation_obj_types)

logger = logging.getLogger(__name__)


class WarehouseModule:

    def __init__(self, config: WarehouseConfig):
        self.config = config

    def gen_chute_mapping_from_capacity(
        self,
        chute_locs: np.ndarray,
        chute_capacities: np.ndarray,
        destination_volumes: np.ndarray,
        repair_seed: int,
    ):
        """Generate chute mapping using MILP

        Args:
            chute_locs (np.ndarray): locations of the chutes
            chute_capacities (np.ndarray): capacities of the chutes
            destination_volumes (np.ndarray): volumes of the destinations
            repair_seed (int): seed of MILP
        """
        chute_mapping = milp_chute_mapping(
            chute_locs,
            chute_capacities,
            destination_volumes,
            seed=repair_seed,
            time_limit=self.config.milp_time_limit,
            n_threads=self.config.milp_n_threads,
            ub_chutes_per_dest=self.config.ub_chutes_per_dest,
            # warm_start=self.config.use_warm_up,
        )
        return chute_mapping

    def repair_chute_mapping(
        self,
        chute_locs: np.ndarray,
        unrepaired_chute_mapping: dict,
        destination_volumes: np.ndarray,
        repair_seed: int,
    ):
        """Generate chute mapping using MILP

        Args:
            chute_locs (np.ndarray): locations of the chutes
            chute_capacities (np.ndarray): capacities of the chutes
            destination_volumes (np.ndarray): volumes of the destinations
            repair_seed (int): seed of MILP
        """
        chute_mapping = milp_chute_mapping_edit_dist(
            chute_locs,
            unrepaired_chute_mapping,
            destination_volumes,
            seed=repair_seed,
            time_limit=self.config.milp_time_limit,
            n_threads=self.config.milp_n_threads,
        )
        return chute_mapping

    def repair(
        self,
        unrepaired_env_int: np.ndarray,
        parent_repaired_env: np.ndarray,
        repair_seed: int,
        w_mode: bool,
        min_n_shelf: int,
        max_n_shelf: int,
    ):
        """Add non-storage area AND enforce domain specific constraints

        Args:
            map (np.ndarray): unrepaired storage area
            parent_repaired_env (np.ndarray): paranet repaired env, if
                applicable
            repair_seed (int): seed of repairing
            sim_seed (int): seed of simulation
            w_mode (bool): whether in w mode
            min_n_shelf (int): min number of shelf
            max_n_shelf (int): max number of shelf

        Returns:
            tuple: (
                map_json,
                unrepaired_env_int,
                repaired_env_int,
            )
        """

        # Create json string for the map
        if self.config.scenario == "KIVA":
            if self.config.obj_type == "throughput_plus_n_shelf":
                assert max_n_shelf == min_n_shelf

            unrepaired_env_int, n_row_comp, n_col_comp = add_non_storage_area(
                unrepaired_env_int, w_mode)

            # Repair environment here
            format_env = format_env_str(
                kiva_env_number2str(unrepaired_env_int))

            logger.info(f"Repairing generated environment:\n{format_env}")

            # Limit n_shelf?
            limit_n_shelf = True
            if self.config.obj_type == "throughput_plus_n_shelf":
                limit_n_shelf = False
            # Warm start schema
            warm_up_sols = None
            if self.config.use_warm_up:
                if parent_repaired_env is not None:
                    parent_env_str = format_env_str(
                        kiva_env_number2str(parent_repaired_env))
                    logger.info(f"Parent warm up solution:\n{parent_env_str}")
                    warm_up_sols = [parent_repaired_env]
                # Get the solution from hamming distance objective
                hamming_repaired_env = repair_env(
                    unrepaired_env_int,
                    self.config.num_agents,
                    # agentNum,
                    add_movement=False,
                    min_n_shelf=min_n_shelf,
                    max_n_shelf=max_n_shelf,
                    seed=repair_seed,
                    w_mode=w_mode,
                    warm_envs_np=warm_up_sols,
                    limit_n_shelf=limit_n_shelf,
                    n_threads=self.config.milp_n_threads,
                )
                ##############################################################
                ## For testing purpose, randomly force some layout to fail ###
                # rnd = np.random.rand()
                # if rnd > 0.5:
                #     hamming_repaired_env = None
                ##############################################################

                # If the repair is failed (which happens very rarely), we
                # return None and remember the unrepaired layout.
                if hamming_repaired_env is None:
                    failed_unrepaired_env = format_env_str(
                        kiva_env_number2str(unrepaired_env_int))
                    logger.info(
                        f"Hamming repair failed! The layout is:\n{failed_unrepaired_env}"
                    )
                    return None, unrepaired_env_int, None
                hamming_warm_env_str = format_env_str(
                    kiva_env_number2str(hamming_repaired_env))
                logger.info(
                    f"Hamming warm up solution:\n{hamming_warm_env_str}")

                if parent_repaired_env is None:
                    warm_up_sols = [hamming_repaired_env]
                else:
                    warm_up_sols = [hamming_repaired_env, parent_repaired_env]

            # If hamming only, we just use hamming_repaired_env as the result
            # env
            if self.config.hamming_only:
                repaired_env_int = hamming_repaired_env
            else:
                repaired_env_int = repair_env(
                    unrepaired_env_int,
                    # agentNum,
                    add_movement=True,
                    warm_envs_np=warm_up_sols,
                    min_n_shelf=min_n_shelf,
                    max_n_shelf=max_n_shelf,
                    seed=repair_seed,
                    w_mode=w_mode,
                    limit_n_shelf=limit_n_shelf,
                    n_threads=self.config.milp_n_threads,
                )
                if repaired_env_int is None:
                    failed_unrepaired_env = format_env_str(
                        kiva_env_number2str(unrepaired_env_int))
                    logger.info(
                        f"Repair failed! The layout is:\n{failed_unrepaired_env}"
                    )
                    return None, unrepaired_env_int, None

            # Convert map layout to str format
            map_str_repaired = kiva_env_number2str(repaired_env_int)

            format_env = format_env_str(map_str_repaired)
            logger.info(f"\nRepaired result:\n{format_env}")

            # Create a random name for the map
            uuid_str = f"_{shortuuid.ShortUUID().random(length=8)}"

            # Create json string to map layout
            map_json_str = json.dumps({
                "name":
                f"repaired_map-{uuid_str}",
                "weight":
                False,
                "n_row":
                n_row_comp,
                "n_col":
                n_col_comp,
                "n_endpoint":
                sum(row.count('e') for row in map_str_repaired),
                "n_agent_loc":
                sum(row.count('r') for row in map_str_repaired),
                "maxtime":
                self.config.simulation_time,
                "layout":
                map_str_repaired,
            })

        else:
            NotImplementedError("Other warehouse types not supported yet.")

        return (
            map_json_str,
            unrepaired_env_int,
            repaired_env_int,
        )

    def partial_repair(
        self,
        unrepaired_env_int: np.ndarray,
        repair_seed: int,
        min_n_shelf: int,
        max_n_shelf: int,
    ):
        repaired_env_int = partial_repair_env(
            unrepaired_env_int,
            add_movement=False,
            min_n_shelf=min_n_shelf,
            max_n_shelf=max_n_shelf,
            seed=repair_seed,
            limit_n_shelf=True,
            n_threads=self.config.milp_n_threads,
        )
        return repaired_env_int

    def evaluate_rhcr(
        self,
        map_json: str,
        output_dir: str,
        sim_seed: int,
        model_params: List = None,
        chute_mapping_json: str = None,
        package_dist_weight_json: str = None,
        task_assignment_params_json: str = None,
    ):
        """
        Run simulation with RHCR

        Args:
            map (np.ndarray): input map in integer format
            parent_repaired_env (np.ndarray): parent solution of the map. Will be None if
                                     current sol is the initial population.
            output_dir (str): log dir of simulation
            n_evals (int): number of evaluations
            sim_seed (int): random seed for simulation. Should be different for
                            each solution
            repair_seed (int): random seed for repairing. Should be the same as
                               master seed
            w_mode (bool): whether to run with w_mode, which replace 'r' with
                           'w' in generated map layouts, where 'w' is a
                           workstation. Under w_mode, robots will start from
                           endpoints and their tasks will alternate between
                           endpoints and workstations.
            n_endpt (int): number of endpoint around each obstacle
            min_n_shelf (int): min number of shelves
            max_n_shelf (int): max number of shelves
            agentNum (int): number of drives
            map_id (int): id of the current map to be evaluated. The id
                          is only unique to each batch, NOT to the all the
                          solutions. The id can make sure that each simulation
                          gets a different log directory.
        """
        # output = str(output_dir /
        #              f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        # We need to construct kwargs manually because some parameters
        # must NOT be passed in in order to use the default values
        # defined on the C++ side.
        # It is very dumb but works.

        kwargs = {
            "map": map_json,
            "output": output_dir,
            "scenario": self.config.scenario,
            "task": self.config.task,
            "agentNum": self.config.num_agents,
            "cutoffTime": self.config.cutoffTime,
            "seed": sim_seed,
            "screen": self.config.screen,
            "solver": self.config.solver,
            "id": self.config.id,
            "single_agent_solver": self.config.single_agent_solver,
            "lazyP": self.config.lazyP,
            "simulation_time": self.config.simulation_time,
            "simulation_window": self.config.simulation_window,
            "travel_time_window": self.config.travel_time_window,
            "potential_function": self.config.potential_function,
            "potential_threshold": self.config.potential_threshold,
            "rotation": self.config.rotation,
            "robust": self.config.robust,
            "CAT": self.config.CAT,
            "hold_endpoints": self.config.hold_endpoints,
            "dummy_paths": self.config.dummy_paths,
            "prioritize_start": self.config.prioritize_start,
            "suboptimal_bound": self.config.suboptimal_bound,
            "log": self.config.log,
            "test": self.config.test,
            "force_new_logdir": True,
            "save_result": self.config.save_result,
            "save_solver": self.config.save_solver,
            "save_heuristics_table": self.config.save_heuristics_table,
            "stop_at_traffic_jam": self.config.stop_at_traffic_jam,
            "left_w_weight": self.config.left_w_weight,
            "right_w_weight": self.config.right_w_weight,
            # Chute mapping related
            "package_dist_weight": package_dist_weight_json,
            "package_mode": self.config.package_mode,
            "chute_mapping": chute_mapping_json,
            "task_assignment_cost": self.config.task_assignment_cost,
            "assign_C": self.config.assign_C,
            # recirc and task wait
            "recirc_mechanism": self.config.recirc_mechanism,
            "task_waiting_time": self.config.task_waiting_time,
            "workstation_waiting_time": self.config.workstation_waiting_time,
        }

        # For some of the parameters, we do not want to pass them in here
        # to the use the default value defined on the C++ side.
        # We are not able to define the default value in python because values
        # such as INT_MAX can be tricky in python but easy in C++.
        planning_window = self.config.planning_window
        if planning_window is not None:
            kwargs["planning_window"] = planning_window

        one_sim_result_jsonstr = warehouse_sim.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        result_json["success"] = True
        return result_json

    def _run_pibt_single_offline(
        self,
        kwargs,
        manually_clean_memory=True,
        save_in_disk=True,
    ):
        if not manually_clean_memory:
            one_sim_result_jsonstr = py_driver.run(**kwargs)
            result_json = json.loads(one_sim_result_jsonstr)
            return result_json
        else:
            if save_in_disk:
                os.makedirs(RUN_DIR, exist_ok=True)
                hash_obj = hashlib.sha256()
                raw_name = get_current_time_str().encode() + os.urandom(16)
                hash_obj.update(raw_name)
                file_name = hash_obj.hexdigest()
                file_path = os.path.join(RUN_DIR, file_name)
                with open(file_path, 'w') as f:
                    json.dump(kwargs, f)
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                # delimiter2 = "----DELIMITER2----DELIMITER2----"
                output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
import py_driver
import json
import time

file_path='{file_path}'
with open(file_path, 'r') as f:
    kwargs_ = json.load(f)

one_sim_result_jsonstr = py_driver.run(**kwargs_)
result_json = json.loads(one_sim_result_jsonstr)

print("{delimiter1}")
print(result_json)
print("{delimiter1}")

                """
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE).stdout.decode('utf-8')
                t2 = time.time()
                if os.path.exists(file_path):
                    os.remove(file_path)
                else:
                    raise NotImplementedError

            else:
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
import py_driver
import json

kwargs_ = {kwargs}
one_sim_result_jsonstr = py_driver.run(**kwargs_)
result_json = json.loads(one_sim_result_jsonstr)
np.set_printoptions(threshold=np.inf)
print("{delimiter1}")
print(result_json)
print("{delimiter1}")
                    """
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE).stdout.decode('utf-8')
                t2 = time.time()
            # print("================")
            # if self.verbose >= 2:
            #     print("run_sim time = ", t2-t1)

            # if self.verbose >= 4:
            #     o = output.split(delimiter2)
            #     for t in o[1:-1:2]:
            #         time_s = t.replace('\n', '')
            #         print("inner sim time =", time_s)
            #     print(self.config.iter_update_n_sim)
            outputs = output.split(delimiter1)
            if len(outputs) <= 2:
                print(output)
                # with open("error_output.txt", "w") as f:
                #     f.write(output)
                # with open("error_kwargs.txt", "w") as f:
                #     json.dump(kwargs, f)
                # Sometimes output does not capture the output json. In that
                # case we attempt to run it again.
                # Not the best solution but it works for now.
                logger.info("Running again without manually cleaning memory.")
                one_sim_result_jsonstr = py_driver.run(**kwargs)
                results = json.loads(one_sim_result_jsonstr)
            else:
                results_str = outputs[1].replace('\n', '').replace(
                    'array', 'np.array')
                # print(collected_results_str)
                results = eval(results_str)

            gc.collect()
            return results

    def _run_pibt_single_online(
        self,
        map_np: np.ndarray,
        map_json: str,
        model_params: List,
        sim_seed: int,
        chute_mapping_json: str,
        task_assignment_params_json: str,
    ):
        if self.config.scenario == "KIVA":
            domain = "kiva"
        elif self.config.scenario == "SORTING":
            domain = "sortation"
        n_valid_vertices = get_n_valid_vertices(map_np, domain)
        n_valid_edges = get_n_valid_edges(map_np,
                                          bi_directed=True,
                                          domain=domain)

        online_env = PIBTWarehouseOnlineEnv(
            map_np,
            map_json,
            n_valid_vertices,
            n_valid_edges,
            self.config,
            seed=sim_seed,
            chute_mapping_json=chute_mapping_json,
            task_assignment_params_json=task_assignment_params_json,
        )

        if self.config.iter_update_mdl_kwargs is None:
            self.config.iter_update_mdl_kwargs = {}
        update_model: WarehouseBaseUpdateModel = \
            self.config.iter_update_model_type(
                map_np,
                n_valid_vertices,
                n_valid_edges,
                model_params=model_params,
                config=self.config,
                **self.config.iter_update_mdl_kwargs,
            )

        obs, info = online_env.reset()

        done = False
        while not done:
            wait_cost_update_vals, edge_weight_update_vals = \
                update_model.get_update_values_from_obs(obs)
            # action = np.random.rand(n_valid_vertices + n_valid_edges)
            obs, reward, terminated, truncated, info = online_env.step(
                np.concatenate([
                    wait_cost_update_vals,
                    edge_weight_update_vals,
                ]))
            done = terminated or truncated

        curr_result = info["result"]
        return curr_result

    def evaluate_pibt(
        self,
        map_json: str,
        output_dir: str,
        sim_seed: int,
        model_params: List = None,
        chute_mapping_json: str = None,
        package_dist_weight_json: str = None,
        task_assignment_params_json: str = None,
        # map_id: int,
        # eval_id: int,
    ):
        """
        Run simulation with PIBT

        Args:
            edge_weights_json (str): json string of the edge weights
            wait_costs_json (str): json string of the wait costs
            output_dir (str): log dir of simulation
            n_evals (int): number of evaluations
            sim_seed (int): random seed for simulation. Should be different for
                            each solution
            map_id (int): id of the current map to be evaluated. The id
                          is only unique to each batch, NOT to the all the
                          solutions. The id can make sure that each simulation
                          gets a different log directory.
            eval_id (int): id of evaluation.
        """
        # output = str(output_dir /
        #              f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        # Read in chute mapping, if any
        if self.config.scenario == "SORTING":
            assert chute_mapping_json is not None
            assert package_dist_weight_json is not None

            # Chute mapping MILP returns no solution
            if chute_mapping_json == "null":
                logger.info("Chute mapping MILP returns no solution.")
                return {"success": False}

        # Get map_np
        map_json = json.loads(map_json)
        if self.config.scenario == "KIVA":
            map_np = kiva_env_str2number(map_json["layout"])
            domain = "kiva"
        elif self.config.scenario == "SORTING":
            map_np = sortation_env_str2number(map_json["layout"])
            domain = "sortation"

        # Offline PIBT
        if not self.config.online_update:
            kwargs = {
                # "cmd": cmd,
                "scenario": self.config.scenario,
                "map_json_str": json.dumps(map_json),
                "simulation_steps": self.config.simulation_time,
                "gen_random": self.config.gen_random,
                "num_tasks": self.config.num_tasks,
                "num_agents": self.config.num_agents,
                "left_w_weight": self.config.left_w_weight,
                "right_w_weight": self.config.right_w_weight,
                # "weights": json.dumps(edge_weights),
                # "wait_costs": json.dumps(wait_costs),
                "plan_time_limit": self.config.plan_time_limit,
                "seed": sim_seed,
                "preprocess_time_limit": self.config.preprocess_time_limit,
                "file_storage_path": "large_files",
                "task_assignment_strategy":
                self.config.task_assignment_strategy,
                "num_tasks_reveal": self.config.num_tasks_reveal,
                "assign_C": self.config.assign_C,
                "recirc_mechanism": self.config.recirc_mechanism,
                "task_waiting_time": self.config.task_waiting_time,
                "workstation_waiting_time":
                self.config.workstation_waiting_time,
                "task_change_time": self.config.task_change_time,
                "task_gaussian_sigma": self.config.task_gaussian_sigma,
                "time_dist": self.config.time_dist,
                "time_sigma": self.config.time_sigma,
                "config":
                load_pibt_default_config(),  # Use PIBT default config
                # Chute mapping related
                "package_dist_weight": package_dist_weight_json,
                "package_mode": self.config.package_mode,
                "chute_mapping": chute_mapping_json,
                "sleep_time_factor": self.config.sleep_time_factor,
                "sleep_time_noise_std": self.config.sleep_time_noise_std,
                # Task assignment related
                "task_assignment_cost": self.config.task_assignment_cost,
                "task_assignment_params": task_assignment_params_json,
            }

            # weighted?
            weighted = map_json["weight"]
            if weighted:
                raw_weights = map_json["weights"]
                n_valid_vertices = get_n_valid_vertices(map_np, domain=domain)
                kwargs["weights"] = json.dumps(raw_weights[n_valid_vertices:])
                kwargs["wait_costs"] = json.dumps(
                    raw_weights[:n_valid_vertices])

            result_json = self._run_pibt_single_offline(
                kwargs,
                manually_clean_memory=True,
                save_in_disk=True,
            )
        # Online PIBT
        else:
            result_json = self._run_pibt_single_online(
                map_np,
                json.dumps(map_json),
                model_params,
                sim_seed,
                chute_mapping_json,
                task_assignment_params_json,
            )
        result_json["success"] = True
        return result_json

    def evaluate_iterative_update(
        self,
        env_np: np.ndarray,
        map_json_str: str,
        model_params: List,
        n_valid_edges: int,
        n_valid_vertices: int,
        output_dir: str,
        seed: int,
        chute_mapping_json: str = None,
        task_assignment_params_json: str = None,
    ):
        """Run PIU. Currently only works with PIBT

        Args:
            model_params (List): parameters of the update model
            n_valid_edges (int): number of valid edges
            n_valid_vertices (int): number of valid vertices
            seed (int): random seed

        """
        # Read in chute mapping, if any
        if self.config.scenario == "SORTING":
            assert chute_mapping_json is not None

            # Chute mapping MILP returns no solution
            if chute_mapping_json == "null":
                logger.info("No chute mapping.")
                return {"success": False}

        iter_update_env = WarehouseIterUpdateEnv(
            env_np=env_np,
            map_json_str=map_json_str,
            n_valid_vertices=n_valid_vertices,
            n_valid_edges=n_valid_edges,
            config=self.config,
            seed=seed,
            output_dir=output_dir,
            chute_mapping_json=chute_mapping_json,
            task_assignment_params_json=task_assignment_params_json,
            # init_weight_file=init_weight_file,
        )

        if self.config.iter_update_mdl_kwargs is None:
            self.config.iter_update_mdl_kwargs = {}
        update_model: WarehouseBaseUpdateModel = \
            self.config.iter_update_model_type(
                env_np,
                n_valid_vertices,
                n_valid_edges,
                model_params=model_params,
                config=self.config,
                **self.config.iter_update_mdl_kwargs,
            )
        all_throughputs = []
        obs, info = iter_update_env.reset()
        # curr_wait_costs = info["curr_wait_costs"]
        # curr_edge_weights = info["curr_edge_weights"]
        curr_result = info["result"]
        curr_throughput = curr_result["throughput"]
        all_throughputs.append(curr_throughput)
        done = False
        while not done:
            edge_usage_matrix = np.moveaxis(obs[:4], 0, 2)
            wait_usage_matrix = obs[4]
            curr_edge_weights_matrix = np.moveaxis(obs[5:9], 0, 2)
            curr_wait_costs_matrix = obs[9]

            # Get update value
            wait_cost_update_vals, edge_weight_update_vals = \
                update_model.get_update_values(
                    wait_usage_matrix,
                    edge_usage_matrix,
                    curr_wait_costs_matrix,
                    curr_edge_weights_matrix,
                )

            # Perform update
            obs, imp_throughput, done, _, info = iter_update_env.step(
                np.concatenate([
                    wait_cost_update_vals,
                    edge_weight_update_vals,
                ]))

            curr_throughput += imp_throughput
            all_throughputs.append(curr_throughput)
            curr_result = info["result"]

        # print(all_throughputs)
        # print(np.max(all_throughputs))
        # breakpoint()
        curr_wait_costs = info["curr_wait_costs"]
        curr_edge_weights = info["curr_edge_weights"]
        # curr_edge_weights = comp_compress_edge_matrix(
        #     comp_map, np.moveaxis(obs[5:9], 0, 2))
        # curr_wait_costs = comp_compress_vertex_matrix(obs[9])

        return curr_result, np.concatenate(
            [curr_wait_costs, curr_edge_weights])

    def process_eval_result_rhcr(
        self,
        curr_result_json: List[dict],
        n_evals: int,
        unrepaired_env_int: np.ndarray,
        repaired_env_int: np.ndarray,
        edge_weights: np.ndarray,
        wait_costs: np.ndarray,
        w_mode: bool,
        max_n_shelf: int,
        map_id: int,
    ):
        """
        Process the evaluation result

        Args:
            curr_result_json (List[dict]): result json of all simulations of 1
                map.

        """

        # Deal with failed layout.
        # For now, failure only happens during MILP repair, so if failure
        # happens, all simulation json results would contain
        # {"success": False}.
        if not curr_result_json[0]["success"]:
            logger.info(f"Map ID {map_id} failed.")

            metadata = WarehouseMetadata(unrepaired_env_int=unrepaired_env_int)
            result = WarehouseResult.from_raw(
                raw_metadata=metadata,
                opts={
                    "aggregation": self.config.aggregation_type,
                    "measure_names": self.config.measure_names,
                },
            )
            result.failed = True
            return result

        # Collect the results
        collected_results = self._collect_results(curr_result_json)

        # Calculate n_shelf and n_endpoint
        tile_ele, tile_cnt = np.unique(repaired_env_int, return_counts=True)
        tile_cnt_dict = dict(zip(tile_ele, tile_cnt))
        n_shelf = tile_cnt_dict[kiva_obj_types.index("@")]
        n_endpoint = tile_cnt_dict[kiva_obj_types.index("e")]

        # Get average length of all tasks
        all_task_len_mean = collected_results.get("avg_task_len")
        # all_task_len_mean = calc_path_len_mean(repaired_env_int, w_mode)
        all_task_len_mean = all_task_len_mean[0]

        logger.info(
            f"Map ID {map_id}: Average length of all possible tasks: {all_task_len_mean}"
        )

        # Calculate number of connected shelf components
        n_shelf_components = calc_num_shelf_components(repaired_env_int)
        logger.info(
            f"Map ID {map_id}: Number of connected shelf components: {n_shelf_components}"
        )

        # Calculate layout entropy
        entropy = calc_layout_entropy(repaired_env_int, w_mode)
        logger.info(f"Map ID {map_id}: Layout entropy: {entropy}")

        # Post process result if necessary
        tile_usage = np.array(collected_results.get("tile_usage"), dtype=float)
        tile_usage = tile_usage.reshape(n_evals, *repaired_env_int.shape)
        tasks_finished_timestep = [
            np.array(x)
            for x in collected_results.get("tasks_finished_timestep")
        ]

        # Get objective
        objs, sim_score = self._compute_objective(
            collected_results,
            map_id,
            unrepaired_env_int,
            repaired_env_int,
            max_n_shelf,
            n_shelf,
        )

        if self.config.scenario == "KIVA":
            repaired_env_str = kiva_env_number2str(repaired_env_int)
        elif self.config.scenario == "SORTING":
            repaired_env_str = sortation_env_number2str(repaired_env_int)

        # Create WarehouseResult object using the mean of n_eval simulations
        # For tile_usage, num_wait, and finished_task_len, the mean is not taken
        metadata = WarehouseMetadata(
            objs=objs,
            result_objs=collected_results.get("throughput"),
            throughput=collected_results.get("throughput"),
            unrepaired_env_int=unrepaired_env_int,
            repaired_env_int=repaired_env_int,
            repaired_env_str=repaired_env_str,
            edge_weights=edge_weights,
            wait_costs=wait_costs,
            n_shelf=n_shelf,
            n_endpoint=n_endpoint,
            tile_usage=tile_usage,
            tile_usage_mean=np.mean(collected_results.get("tile_usage_mean")),
            tile_usage_std=np.mean(collected_results.get("tile_usage_std")),
            num_wait=collected_results.get("num_wait"),
            num_wait_mean=np.mean(collected_results.get("num_wait_mean")),
            num_wait_std=np.mean(collected_results.get("num_wait_std")),
            num_turns=collected_results.get("num_turns"),
            num_turns_mean=np.mean(collected_results.get("num_turns_mean")),
            num_turns_std=np.mean(collected_results.get("num_turns_std")),
            finished_task_len=collected_results.get("finished_task_len"),
            finished_len_mean=np.mean(
                collected_results.get("finished_len_mean")),
            finished_len_std=np.mean(
                collected_results.get("finished_len_std")),
            all_task_len_mean=all_task_len_mean,
            tasks_finished_timestep=tasks_finished_timestep,
            n_shelf_components=n_shelf_components,
            layout_entropy=entropy,
            similarity_score=sim_score,
        )
        result = WarehouseResult.from_raw(
            raw_metadata=metadata,
            opts={
                "aggregation": self.config.aggregation_type,
                "measure_names": self.config.measure_names,
            },
        )

        return result

    def process_eval_result_pibt(
        self,
        curr_result_json: List[dict],
        n_evals: int,
        unrepaired_env_int: np.ndarray,
        repaired_env_int: np.ndarray,
        edge_weights: np.ndarray,
        wait_costs: np.ndarray,
        w_mode: bool,
        max_n_shelf: int,
        map_id: int,
        chute_mapping: dict,
        chute_capacities: np.ndarray,
    ):
        """
        Process the evaluation result

        Args:
            curr_result_json (List[dict]): result json of all simulations of 1
                map.

        """
        collected_results = self._collect_results(curr_result_json)

        # Deal with failed layout.
        # For now, failure only happens during MILP repair, so if failure
        # happens, all simulation json results would contain
        # {"success": False}.
        if not curr_result_json[0]["success"]:
            logger.info(f"Map ID {map_id} failed.")

            metadata = WarehouseMetadata(unrepaired_env_int=unrepaired_env_int)
            result = WarehouseResult.from_raw(
                raw_metadata=metadata,
                opts={
                    "aggregation": self.config.aggregation_type,
                    "measure_names": self.config.measure_names,
                },
            )
            result.failed = True
            return result

        # Post process result if necessary
        tile_usage = np.array(collected_results.get("tile_usage"), dtype=float)
        tile_usage /= np.sum(tile_usage, axis=1, keepdims=True)
        # tile_usage = tile_usage.reshape(n_evals, *repaired_env_int.shape)
        tile_usage_mean = np.mean(tile_usage, axis=1)
        tile_usage_std = np.std(tile_usage, axis=1)

        logger.info(f"Mean of tile-usage std: {np.mean(tile_usage_std)}")

        # Calculate number of connected shelf components
        n_shelf_components = calc_num_shelf_components(repaired_env_int)
        logger.info(
            f"Map ID {map_id}: Number of connected shelf components: {n_shelf_components}"
        )

        # Calculate layout entropy
        if self.config.scenario == "KIVA":
            entropy = calc_layout_entropy(repaired_env_int, w_mode)
            logger.info(f"Map ID {map_id}: Layout entropy: {entropy}")
        elif self.config.scenario == "SORTING":
            entropy = None

        # Number of shelf
        tile_ele, tile_cnt = np.unique(repaired_env_int, return_counts=True)
        tile_cnt_dict = dict(zip(tile_ele, tile_cnt))
        n_shelf = tile_cnt_dict[kiva_obj_types.index("@")]

        # Get objective based on type
        objs, sim_score = self._compute_objective(
            collected_results,
            map_id,
            unrepaired_env_int,
            repaired_env_int,
            max_n_shelf,
            n_shelf,
        )

        # Compute average distance of the chutes assigned to the top 5% of the
        # destinations to the closest workstation
        avg_min_dist_to_ws = None
        avg_centroid_dist = None
        if chute_mapping is not None:
            assert self.config.scenario == "SORTING"
            block_idxs = [
                sortation_obj_types.index("@"),
                sortation_obj_types.index("T"),
            ]
            avg_min_dist_to_ws = compute_avg_min_dist_to_ws(
                repaired_env_int, block_idxs, chute_mapping)
            avg_centroid_dist = compute_avg_dist_to_centroid(
                repaired_env_int, chute_mapping)

        if self.config.scenario == "KIVA":
            repaired_env_str = kiva_env_number2str(repaired_env_int)
        elif self.config.scenario == "SORTING":
            repaired_env_str = sortation_env_number2str(repaired_env_int)

        metadata = WarehouseMetadata(
            objs=objs,
            result_objs=collected_results.get("throughput"),
            throughput=collected_results.get("throughput"),
            unrepaired_env_int=unrepaired_env_int,
            repaired_env_int=repaired_env_int,
            repaired_env_str=repaired_env_str,
            edge_weights=edge_weights,
            wait_costs=wait_costs,
            tile_usage=tile_usage,
            tile_usage_mean=np.mean(tile_usage_mean),
            tile_usage_std=np.mean(tile_usage_std),
            n_shelf_components=n_shelf_components,
            layout_entropy=entropy,
            similarity_score=sim_score,
            chute_mapping=chute_mapping,
            chute_capacities=chute_capacities,
            avg_min_dist_to_ws=avg_min_dist_to_ws,
            avg_centroid_dist=avg_centroid_dist,
        )
        result = WarehouseResult.from_raw(
            raw_metadata=metadata,
            opts={
                "aggregation": self.config.aggregation_type,
                "measure_names": self.config.measure_names,
            },
        )

        return result

    def process_inner_obj_results(
        self,
        unrepaired_env_int,
        repaired_env_int,
        w_mode,
    ):
        """process the results of inner loop evaluation for extinct search.
        For now the objective is the similarity score.

        Args:
            unrepaired_env_int (np.ndarray): unrepaired env
            repaired_env_int (np.ndarray): repaired env
            w_mode (bool): if True, env is w mode

        Returns:
            result: result of evaluation
        """
        # If partial repair is failed, repaired_env_int = None
        if repaired_env_int is None:
            logger.info(f"Partial repair failed.")

            metadata = WarehouseMetadata(unrepaired_env_int=unrepaired_env_int)
            result = WarehouseResult.from_raw(
                raw_metadata=metadata,
                opts={
                    "aggregation": self.config.aggregation_type,
                    "measure_names": self.config.measure_names,
                },
            )
            result.failed = True
            return result

        sim_score = cal_similarity_score(unrepaired_env_int, repaired_env_int)
        n_shelf_components = calc_num_shelf_components(repaired_env_int)
        entropy = calc_layout_entropy(repaired_env_int, w_mode)

        if self.config.scenario == "KIVA":
            repaired_env_str = kiva_env_number2str(repaired_env_int)
        elif self.config.scenario == "SORTING":
            repaired_env_str = sortation_env_number2str(repaired_env_int)

        metadata = WarehouseMetadata(
            objs=np.array([sim_score]),
            result_objs=np.array([sim_score]),
            unrepaired_env_int=unrepaired_env_int,
            repaired_env_int=repaired_env_int,
            repaired_env_str=repaired_env_str,
            n_shelf_components=n_shelf_components,
            layout_entropy=entropy,
            similarity_score=sim_score,
        )
        result = WarehouseResult.from_raw(
            raw_metadata=metadata,
            opts={
                "aggregation": self.config.aggregation_type,
                "measure_names": self.config.extinct_measure_names,
            },
        )

        return result

    def actual_qd_score(self, objs):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)

    def _compute_objective(
        self,
        collected_results,
        map_id,
        unrepaired_env_int,
        repaired_env_int,
        max_n_shelf,
        n_shelf,
    ):
        """Compute objective of QD optimization

        Args:
            collected_results (json): collected result json
            map_id (id): id of the map
            unrepaired_env_int (np.ndarray): unrepaired full env in int format
            repaired_env_int (np.ndarray): repaired full env in int format
            max_n_shelf (int): maximum number of shelves
            n_shelf (int): number of shelves in env

        Returns: (
            np.ndarray: objective values,
            float or None: similarity score
        )
        """
        # Get objective based on type
        sim_score = cal_similarity_score(
            unrepaired_env_int,
            repaired_env_int,
            self.config.shelf_weight,
        )
        throughput = np.array(collected_results.get("throughput"))

        logger.info(f"Map ID {map_id}: similarity score: {sim_score}")
        logger.info(f"Map ID {map_id}: throughput: {throughput}")

        if self.config.obj_type == "throughput":
            objs = throughput
        elif self.config.obj_type == "throughput_plus_n_shelf":
            objs = throughput - (max_n_shelf - n_shelf)**2 * 0.5
        elif self.config.obj_type == "throughput_minus_hamming_dist":
            # Normalize hamming dist "regularization" to [0, 1]
            # Essentially we maximize:
            # 1. The throughput
            # 2. The percentage of tiles that are the same in unrepaired and
            #    repaired layouts
            objs = throughput + self.config.hamming_obj_weight * sim_score
        elif self.config.obj_type == "throughput_+_sim_squre":
            objs = throughput + (sim_score + 2)**2
        else:
            return ValueError(
                f"Object type {self.config.obj_type} not supported")
        logger.info(f"Map ID {map_id}: Computed obj: {objs}")
        return objs, sim_score

    def _collect_results(self, curr_result_json):
        """Merge list of json into a single json

        Args:
            curr_result_json (List[dict]): list of jsons with results of all `n_eval` simulations.

        Returns:
            dict: one json with collected results
        """
        # Collect the results
        keys = curr_result_json[0].keys()
        collected_results = {key: [] for key in keys}
        for result_json in curr_result_json:
            for key in keys:
                collected_results[key].append(result_json[key])
        return collected_results


def compute_chute_capacities(chute_mapping, chute_locs, destination_volumes):
    # Compute chute capacities
    chute_capacities = {c: 0 for c in chute_locs}
    for d in chute_mapping:
        for c in chute_mapping[d]:
            chute_capacities[c] += destination_volumes[int(d)] / len(
                chute_mapping[d])

    chute_capacities = [chute_capacities[c] for c in chute_locs]
    return np.array(chute_capacities)


def cal_similarity_score(unrepaired_env_int, repaired_env_int, shelf_weight=1):
    # Calculate hamming distance
    assert unrepaired_env_int.shape == repaired_env_int.shape

    # Get the weight of each tile, shelf has higher weight
    W = np.ones(unrepaired_env_int.shape)
    W[np.where(repaired_env_int == kiva_obj_types.index("@"))] = shelf_weight

    same_matrix = (unrepaired_env_int == repaired_env_int).astype(float)

    sim_score = np.sum(same_matrix * W)

    # Normalize: repaired map is guranteed to have `N_s` shelves, therefore max
    # unnormalized weight is fixed
    N_s = np.count_nonzero(repaired_env_int == kiva_obj_types.index("@"))
    n_non_shelf = np.prod(unrepaired_env_int.shape) - N_s
    max_score = N_s * shelf_weight + n_non_shelf
    sim_score /= max_score

    return sim_score


def has_endpoint_around(env_np, i, j, n_endpt=2):
    endpoint_cnt = 0
    n_row, n_col = env_np.shape
    for dx, dy in DIRS:
        n_i, n_j = i + dx, j + dy
        if 0 <= n_i < n_row and 0 <= n_j < n_col:
            if env_np[n_i, n_j] == kiva_obj_types.index("e"):
                endpoint_cnt += 1
                if endpoint_cnt >= n_endpt:
                    return True
    return False


def put_endpoints(env_np, n_endpt=2):
    # Use a new order of putting endpoints everytime
    cur_d = copy.deepcopy(DIRS)
    # np.random.shuffle(cur_d)

    # Put endpoint around the obstacles
    n_row, n_col = env_np.shape
    for i in range(n_row):
        for j in range(n_col):
            if env_np[i, j] == kiva_obj_types.index("@"):
                for dx, dy in cur_d:
                    n_i, n_j = i + dx, j + dy
                    # if in range and the tile is empty space, add endpoint
                    if 0 <= n_i < n_row and \
                        0 <= n_j < n_col and \
                        env_np[n_i, n_j] == kiva_obj_types.index(".") and \
                        not has_endpoint_around(env_np, i, j, n_endpt=n_endpt):
                        env_np[n_i, n_j] = kiva_obj_types.index("e")
    return env_np


def add_non_storage_area(storage_area, w_mode):
    ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if w_mode \
                                else KIVA_ROBOT_BLOCK_WIDTH
    ADDITION_BLOCK_HEIGHT = 0 if w_mode else KIVA_ROBOT_BLOCK_HEIGHT
    n_row, n_col = storage_area.shape

    # # Keep only max_obs_ratio of obstacles
    # n_max_obs = round(n_row * n_col * max_obs_ratio)
    # n_curr_obs = round(np.sum(storage_area))
    # n_rm_obs = n_curr_obs - n_max_obs
    # if n_rm_obs > 0:
    #     all_obs_idx = np.transpose(
    #         np.nonzero(storage_area == kiva_obj_types.index("@")))
    #     to_change = random.sample(list(all_obs_idx), k=n_rm_obs)
    #     for i, j in to_change:
    #         storage_area[i, j] = kiva_obj_types.index(".")

    # storage_area = put_endpoints(storage_area, n_endpt=n_endpt)

    # Stack left and right additional blocks
    l_block, r_block = get_additional_h_blocks(ADDITION_BLOCK_WIDTH, n_row,
                                               w_mode)
    assert l_block.shape == r_block.shape == (n_row, ADDITION_BLOCK_WIDTH)
    full_env = np.hstack((l_block, storage_area, r_block))
    n_col_comp = n_col + 2 * ADDITION_BLOCK_WIDTH

    # Stack top and bottom additional blocks (At this point, we assume
    # that left and right blocks are stacked)
    n_row_comp = n_row
    if ADDITION_BLOCK_HEIGHT > 0:
        t_block, b_block = \
            get_additional_v_blocks(ADDITION_BLOCK_HEIGHT,
                                    n_col_comp, w_mode)
        full_env = np.vstack((t_block, full_env, b_block))
        n_row_comp += 2 * ADDITION_BLOCK_HEIGHT

    return full_env, n_row_comp, n_col_comp


def calc_path_len_mean(repaired_env, w_mode):
    if w_mode:
        start_locs = np.where(repaired_env == kiva_obj_types.index("w"))
    else:
        start_locs = np.where(repaired_env == kiva_obj_types.index("e"))

    start_locs = np.stack(start_locs, axis=1)
    end_locs = np.where(repaired_env == kiva_obj_types.index("e"))
    end_locs = np.stack(end_locs, axis=1)

    block_idxs = [
        kiva_obj_types.index("@"),
        # kiva_obj_types.index("e"),
        kiva_obj_types.index("w"),
        kiva_obj_types.index("r"),
    ]
    if not w_mode:
        block_idxs.append(kiva_obj_types.index("e"))

    path_length_table = {}
    for start_loc in start_locs:
        path_length_table[tuple(start_loc)] = BFS_path_len(
            tuple(start_loc),
            [tuple(end_loc) for end_loc in end_locs],
            copy.deepcopy(repaired_env),
            block_idxs,
        )

    all_path_length = []
    for start_loc in start_locs:
        for end_loc in end_locs:
            if tuple(start_loc) != tuple(end_loc):
                all_path_length.append(
                    path_length_table[tuple(start_loc)][tuple(end_loc)])

    return np.mean(all_path_length)


def calc_layout_entropy(map_np_repaired, w_mode):
    """
    Calculate entropy of the storage area of the layout.

    We first formulate the layout as a tile pattern distribution by following
    Lucas, Simon M. M. and Vanessa Volz. “Tile pattern KL-divergence for
    analysing and evolving game envs.” Proceedings of the Genetic and
    Evolutionary Computation Conference (2019).

    Then we calculate the entropy.
    """
    # Separate the storage area
    h, w = map_np_repaired.shape
    storage_area = None
    if w_mode:
        storage_area = map_np_repaired[:, KIVA_WORKSTATION_BLOCK_WIDTH:w -
                                       KIVA_WORKSTATION_BLOCK_WIDTH]
    else:
        storage_area = map_np_repaired[KIVA_ROBOT_BLOCK_WIDTH:h -
                                       KIVA_ROBOT_BLOCK_WIDTH,
                                       KIVA_ROBOT_BLOCK_WIDTH:w -
                                       KIVA_ROBOT_BLOCK_WIDTH]

    # Generate list of patterns (we use 2 x 2)
    storage_obj_types = kiva_obj_types[:3]
    tile_patterns = {
        "".join(x): 0
        for x in product(storage_obj_types, repeat=4)
    }

    s_h, s_w = storage_area.shape
    # Iterate over 2x2 blocks
    for i in range(s_h - 1):
        for j in range(s_w - 1):
            curr_block = storage_area[i:i + 2, j:j + 2]
            curr_pattern = "".join(kiva_env_number2str(curr_block))
            tile_patterns[curr_pattern] += 1
    pattern_dist = list(tile_patterns.values())

    # Use number of patterns as the base to bound the entropy to [0, 1]
    return entropy(pattern_dist, base=len(pattern_dist))


def BFS_shelf_component(start_loc, env_np, env_visited):
    """
    Find all shelves that are connected to the shelf at start_loc.
    """
    # We must start searching from shelf
    assert env_np[start_loc] == kiva_obj_types.index("@")

    q = Queue()
    q.put(start_loc)
    seen = set()
    m, n = env_np.shape
    block_idxs = [
        kiva_obj_types.index("e"),
        kiva_obj_types.index("w"),
        kiva_obj_types.index("r"),
        kiva_obj_types.index("w"),
        kiva_obj_types.index("."),
    ]
    while not q.empty():
        curr = q.get()
        x, y = curr
        env_visited[x, y] = True
        seen.add(curr)
        for dx, dy in DIRS:
            n_x = x + dx
            n_y = y + dy
            if n_x < m and n_x >= 0 and \
               n_y < n and n_y >= 0 and \
               env_np[n_x,n_y] not in block_idxs and\
               (n_x, n_y) not in seen:
                q.put((n_x, n_y))


def calc_num_shelf_components(repaired_env):
    env_visited = np.zeros(repaired_env.shape, dtype=bool)
    n_row, n_col = repaired_env.shape
    n_shelf_components = 0
    for i in range(n_row):
        for j in range(n_col):
            if repaired_env[i,j] == kiva_obj_types.index("@") and\
                not env_visited[i,j]:
                n_shelf_components += 1
                BFS_shelf_component((i, j), repaired_env, env_visited)
    return n_shelf_components


def get_additional_h_blocks(ADDITION_BLOCK_WIDTH, n_row, w_mode):
    """
    Generate additional blocks to horizontally stack to the map on the left and
    right side
    """

    if w_mode:
        # In 'w' mode, horizontally stack the workstations
        # The workstation locations are fixed as the following:
        # 1. Stack workstations on the border of the generated map,
        #    meaning that there is no columns on the left/right side of the
        #    left/right workstations.
        # 2. The first row and last row has no workstations.
        # 3. For the rest of the rows, starting from the second row, put
        # workstations for every three rows, meaning that there are at least
        # two empty cells between each pair of workstations.
        # 4. The left and right side of workstation blocks are symmetrical
        l_block = []
        r_block = []
        for i in range(n_row):
            curr_l_row = None
            curr_r_row = None
            if i == 0 or i == n_row - 1 or (i - 1) % 3 != 0:
                curr_l_row = [
                    kiva_obj_types.index(".")
                    for _ in range(ADDITION_BLOCK_WIDTH)
                ]
                curr_r_row = copy.deepcopy(curr_l_row)
            elif (i - 1) % 3 == 0:
                curr_l_row = [
                    kiva_obj_types.index("w"),
                    kiva_obj_types.index(".")
                ]
                curr_r_row = [
                    kiva_obj_types.index("."),
                    kiva_obj_types.index("w")
                ]
            l_block.append(curr_l_row)
            r_block.append(curr_r_row)
        r_block = np.array(r_block)
        l_block = np.array(l_block)

    else:
        # In 'r' mode, horizontally stack the robot start locations
        # The robot start locations are fixed as the following:
        # 1. Stack robot location blocks on either sides of the generated map
        # 2. On each side, the length of the block is 4
        # 3. The top and bottom rows and the left and right columns have no
        #    robots
        # 4. Starting from the 2nd row, there are 2 robots in the middle column
        # 5. There are at most 3 sequential rows of robots
        # 6. For every 3 rows, append a row of empty space
        r_block = []
        n_robot_row = 0
        for i in range(n_row):
            curr_row = None
            if i == 0 or i == n_row - 1:
                curr_row = [
                    kiva_obj_types.index(".")
                    for _ in range(ADDITION_BLOCK_WIDTH)
                ]
            elif n_robot_row < 3:
                curr_row = [
                    kiva_obj_types.index("."),
                    kiva_obj_types.index("r"),
                    kiva_obj_types.index("r"),
                    kiva_obj_types.index("."),
                ]
                n_robot_row += 1
            elif n_robot_row >= 3:
                curr_row = [
                    kiva_obj_types.index(".")
                    for _ in range(ADDITION_BLOCK_WIDTH)
                ]
                n_robot_row = 0
            r_block.append(curr_row)

        # Under 'r' mode, left and right blocks are the same
        r_block = np.array(r_block)
        l_block = copy.deepcopy(r_block)

    return l_block, r_block


def get_additional_v_blocks(ADDITION_BLOCK_HEIGHT, n_col_comp, w_mode):
    """
    Generate additional blocks to vertically stack to the map on the top and
    bottom
    """
    # Only applicable for r mode
    assert not w_mode
    # We only want even # of cols to make the map symmetrical
    assert n_col_comp % 2 == 0
    t_block = []
    b_block = []
    # For r mode, we need to append additional on top and bottom of the map
    for i in range(ADDITION_BLOCK_HEIGHT):
        # We add 'r' to everywhere except for:
        # 1) the first and block column of each row
        # 2) the 2 columns in the middle of each row
        # 2) the first row for t_block and last row for b_block
        cont_r_length = (n_col_comp - 4) // 2

        if i == 0 or i == ADDITION_BLOCK_HEIGHT - 1:
            t_block.append(
                [kiva_obj_types.index(".") for _ in range(n_col_comp)])
        else:
            t_block.append([
                kiva_obj_types.index("."),
                *[kiva_obj_types.index("r") for _ in range(cont_r_length)],
                kiva_obj_types.index("."),
                kiva_obj_types.index("."),
                *[kiva_obj_types.index("r") for _ in range(cont_r_length)],
                kiva_obj_types.index("."),
            ])

    t_block = np.array(t_block)
    b_block = copy.deepcopy(t_block)
    assert t_block.shape == b_block.shape == (ADDITION_BLOCK_HEIGHT,
                                              n_col_comp)
    return t_block, b_block


def compute_avg_dist_to_centroid(map_np: np.ndarray,
                                 chute_mapping,
                                 ratio=0.05):
    """Compute the average L2 distance of the chutes of the same destinations
    to their centroids, measuring how clustered the chutes are.
    """
    h, w = map_np.shape
    n_destinations = len(chute_mapping)
    avg_dist = 0
    n_chutes = 0
    for d in range(int(n_destinations * ratio)):
        chutes = chute_mapping[d]
        # Compute the centroid by averaging the x and y coordinates of the
        # chutes
        chute_coor = [(loc // w, loc % w) for loc in chutes]
        centroid = np.mean(chute_coor, axis=0)
        for chute in chute_coor:
            n_chutes += 1
            avg_dist += np.linalg.norm(np.array(chute) - centroid)

    avg_dist /= n_chutes
    return avg_dist


def compute_avg_min_dist_to_ws(map_np: np.ndarray,
                               block_idxs,
                               chute_mapping,
                               ratio=0.02):
    """Compute the average minimal distance from each chute to the closest
    workstation, measuring how far the chutes are from the workstations.

    Note: Only works for sortation map
    """
    h, w = map_np.shape
    n_destinations = len(chute_mapping)
    workstation_locs = get_workstation_loc(map_np)
    workstation_coor = [(loc // w, loc % w) for loc in workstation_locs]
    avg_path_len = 0
    n_chutes = 0

    for d in range(max(int(n_destinations * ratio), 1)):
        chutes = chute_mapping[d]
        chutes_coor = [(loc // w, loc % w) for loc in chutes]
        for c in chutes_coor:
            n_chutes += 1
            map_np_c = map_np.copy()
            map_np_c[c] = sortation_obj_types.index(".")
            path_len = BFS_path_len_one_goal(c, workstation_coor, map_np_c,
                                             block_idxs)
            avg_path_len += path_len
    avg_path_len /= n_chutes
    return avg_path_len


def BFS_path_len_one_goal(start_loc, goal_locs, env_np, block_idxs):
    """
    Find shortest path len from start_loc to all goal_locs. Stops when the
    closest goal in `goal_locs` is reached.
    """
    q = Queue()
    q.put(start_loc)
    seen = set()
    m, n = env_np.shape
    dist_matrix = np.full((m, n), np.inf)
    dist_matrix[start_loc] = 0

    while not q.empty():
        curr = q.get()
        x, y = curr
        if curr in goal_locs:
            shortest = dist_matrix[x, y] + 1
            return shortest

        seen.add(curr)
        for dx, dy in DIRS:
            n_x = x + dx
            n_y = y + dy
            if n_x < m and n_x >= 0 and \
               n_y < n and n_y >= 0 and \
               env_np[n_x,n_y] not in block_idxs and\
               (n_x, n_y) not in seen:
                q.put((n_x, n_y))
                dist_matrix[n_x, n_y] = dist_matrix[x, y] + 1
                seen.add((n_x, n_y))
    raise ValueError(f"Start loc: {start_loc}. Remaining goal: {goal_locs}")


def get_comp_map(
    map,
    seed,
    w_mode,
    n_endpt,
    env_height,
):
    """
    Helper function that repair one map using hamming for EM-ME inner
    loop.
    """
    np.random.seed(seed // np.int32(4))

    # Put endpoints in raw maps and repair using hamming distance obj
    ADDITION_BLOCK_WIDTH = KIVA_WORKSTATION_BLOCK_WIDTH if w_mode \
                            else KIVA_ROBOT_BLOCK_WIDTH
    # map = put_endpoints(map, n_endpt=n_endpt)
    l_block, r_block = get_additional_h_blocks(ADDITION_BLOCK_WIDTH,
                                               env_height, w_mode)
    map_comp = np.hstack((l_block, map, r_block))

    # Same as MILP, in the surrogate model, we replace 'w' with 'r' under
    # w_mode to use 'r' internally.
    if w_mode:
        map_comp = flip_tiles(
            map_comp,
            'w',
            'r',
        )
    return map_comp


def single_simulation(
    warehouse_config: WarehouseConfig,
    map_json,
    seed,
    num_agents,
    results_dir,
    simulation_algo,
    model_params: List = None,
    chute_mapping_json: str = None,
    package_dist_weight_json: str = None,
    task_assignment_params_json: str = None,
):
    warehouse_config.num_agents = num_agents
    warehouse_module = WarehouseModule(warehouse_config)
    output_dir = os.path.join(results_dir,
                              f"sim-num_agent={num_agents}-seed={seed}")

    if simulation_algo == "RHCR":
        result_json = warehouse_module.evaluate_rhcr(
            map_json=map_json,
            output_dir=output_dir,
            sim_seed=seed,
            model_params=model_params,
            chute_mapping_json=chute_mapping_json,
            package_dist_weight_json=package_dist_weight_json,
            task_assignment_params_json=task_assignment_params_json,
        )
    elif simulation_algo == "PIBT":
        result_json = warehouse_module.evaluate_pibt(
            map_json=map_json,
            output_dir=output_dir,
            sim_seed=seed,
            model_params=model_params,
            chute_mapping_json=chute_mapping_json,
            package_dist_weight_json=package_dist_weight_json,
            task_assignment_params_json=task_assignment_params_json,
        )

    throughput = result_json["throughput"]

    # Write result to logdir
    # Load and then dump to format the json
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, f"result.json"), "w") as f:
        f.write(json.dumps(result_json, indent=4))

    # Write config to logdir
    # Replace callables with names
    warehouse_config_ = copy.deepcopy(warehouse_config)
    warehouse_config_.iter_update_model_type = getattr(
        warehouse_config_.iter_update_model_type, '__name__', "null")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(dataclasses.asdict(warehouse_config_), indent=4))

    return throughput


def test_calc_path_len_mean(map_filepath):
    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
    repaired_env_str = raw_env_json["layout"]
    repaired_env = kiva_env_str2number(repaired_env_str)

    avg_len = calc_path_len_mean(repaired_env, True)
    print(f"Average length (naive BFS): {avg_len}")


def test_calc_num_shelf_components(map_filepath):
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
    repaired_env_str = raw_env_json["layout"]
    repaired_env = kiva_env_str2number(repaired_env_str)
    n_shelf_components = calc_num_shelf_components(repaired_env)
    print(f"Number of connected shelf components: {n_shelf_components}")


def test_calc_layout_entropy(map_filepath):
    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
    repaired_env_str = raw_env_json["layout"]
    repaired_env = kiva_env_str2number(repaired_env_str)

    layout_entropy = calc_layout_entropy(repaired_env, True)
    print(f"Layout entropy: {layout_entropy}")


def test_calc_sim_score(unrepaired_file, repaired_file):
    unrepaired_env_str, _ = read_in_kiva_map(unrepaired_file)
    repaired_env_str, _ = read_in_kiva_map(repaired_file)
    unrepaired_env_int = kiva_env_str2number(unrepaired_env_str)
    repaired_env_int = kiva_env_str2number(repaired_env_str)
    sim_score = cal_similarity_score(unrepaired_env_int, repaired_env_int, 1)
    print(f"Sim score: {sim_score}")


def main(
    warehouse_config,
    map_filepath,
    simulation_algo,
    model_param_file=None,
    chute_mapping_file=None,
    task_assign_file=None,
    num_agent=10,
    num_agent_step_size=1,
    seed=0,
    n_evals=10,
    n_sim=2,  # Run `inc_agents` `n_sim`` times
    mode="constant",
    n_workers=32,
    reload=None,
):
    """
    For testing purposes. Graph a map and run one simulation.

    Args:
        mode: "constant", "inc_agents", or "inc_timesteps".
              "constant" will run `n_eval` simulations with the same
              `num_agent`.
              "increase" will run `n_eval` simulations with an inc_agents
              number of `num_agent`.
    """
    if simulation_algo not in ["RHCR", "PIBT"]:
        raise ValueError(f"Algorithm '{simulation_algo}' not supported")

    setup_logging(on_worker=False)

    gin.parse_config_file(warehouse_config)
    warehouse_config = WarehouseConfig()

    import torch
    torch.set_num_threads(1)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
    map_json = json.dumps(raw_env_json)

    # Create log dir
    map_name = raw_env_json["name"]
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = time_str + "_" + map_name
    log_dir = os.path.join(LOG_DIR, base_log_dir)
    results_dir = os.path.join(log_dir, "results")
    os.mkdir(log_dir)
    os.mkdir(results_dir)

    # Write map file to logdir
    with open(os.path.join(log_dir, "map.json"), "w") as f:
        f.write(json.dumps(raw_env_json, indent=4))

    # Read in packages, chute mapping
    package_dist_weight_json = None
    chute_mapping_json = None
    if warehouse_config.scenario == "SORTING":
        _, package_dist_weight_json = get_packages(
            warehouse_config.package_mode,
            warehouse_config.package_dist_type,
            warehouse_config.package_path,
            warehouse_config.n_destinations,
        )
        with open(chute_mapping_file, "r") as f:
            chute_mapping_json = json.load(f)
            chute_mapping_json = json.dumps(chute_mapping_json)

        # Store chute mapping in logdir
        with open(os.path.join(log_dir, "chute_mapping.json"), "w") as f:
            f.write(chute_mapping_json)

    # Read in model params, if any
    model_params = None
    if model_param_file is not None:
        model_params = np.load(model_param_file)

    # Read in task assign policy, if any
    task_assignment_params_json = json.dumps(None)
    if task_assign_file is not None:
        task_assignment_params_json = json.dumps(
            np.load(task_assign_file).tolist())

    # Preprocess Callables because gin configurable classes are not pickleable
    itr_update_model_type = getattr(warehouse_config.iter_update_model_type,
                                    '__name__', None)
    if itr_update_model_type == "WarehouseCNNUpdateModel":
        warehouse_config.iter_update_model_type = WarehouseCNNUpdateModel

    have_run = set()
    if reload is not None and reload != "":
        all_results_dir = os.path.join(reload, "results")
        for result_dir in os.listdir(all_results_dir):
            result_dir_full = os.path.join(all_results_dir, result_dir)
            if os.path.exists(os.path.join(result_dir_full, "result.json")) and\
               os.path.exists(os.path.join(result_dir_full, "config.json")):
                curr_configs = result_dir.split("-")
                curr_num_agent = int(curr_configs[1].split("=")[1])
                curr_seed = int(curr_configs[2].split("=")[1])
                have_run.add((curr_num_agent, curr_seed))
            else:
                # breakpoint()
                shutil.rmtree(result_dir_full)

    pool = multiprocessing.Pool(n_workers)
    if mode == "inc_agents":
        seeds = []
        num_agents = []
        num_agent_range = range(0, n_evals, num_agent_step_size)
        actual_n_evals = len(num_agent_range)
        for i in range(n_sim):
            for j in num_agent_range:
                curr_seed = seed + i
                curr_num_agent = num_agent + j
                if (curr_num_agent, curr_seed) in have_run:
                    continue
                seeds.append(curr_seed)
                num_agents.append(curr_num_agent)
        n_runs = actual_n_evals * n_sim - len(have_run)
        throughputs = [
            single_simulation(
                warehouse_config,
                map_json,
                seeds[0],
                num_agents[0],
                results_dir,
                simulation_algo,
                model_params,
                chute_mapping_json,
                package_dist_weight_json,
                task_assignment_params_json,
            )
        ]
        throughputs = pool.starmap(
            single_simulation,
            zip(
                repeat(warehouse_config, n_runs),
                repeat(map_json, n_runs),
                seeds,
                num_agents,
                repeat(results_dir, n_runs),
                repeat(simulation_algo, n_runs),
                repeat(model_params, n_runs),
                repeat(chute_mapping_json, n_runs),
                repeat(package_dist_weight_json, n_runs),
                repeat(task_assignment_params_json, n_runs),
            ),
        )
    elif mode == "constant":
        raise NotImplementedError("Constant mode not implemented")
        # num_agents = [num_agent for _ in range(n_evals)]
        # seeds = np.random.choice(np.arange(10000), size=n_evals, replace=False)

        # throughputs = pool.starmap(
        #     single_simulation,
        #     zip(
        #         repeat(warehouse_config, n_evals),
        #         repeat(map_json, n_evals),
        #         seeds,
        #         num_agents,
        #         repeat(results_dir, n_evals),
        #         repeat(simulation_algo, n_evals),
        #     ),
        # )
    avg_obj = np.mean(throughputs)
    max_obj = np.max(throughputs)
    min_obj = np.min(throughputs)

    n_evals = actual_n_evals if mode == "inc_agents" else n_evals

    print(f"Average throughput over {n_evals} simulations: {avg_obj}")
    print(f"Max throughput over {n_evals} simulations: {max_obj}")
    print(f"Min throughput over {n_evals} simulations: {min_obj}")


if __name__ == "__main__":
    fire.Fire(main)
    # fire.Fire(test_calc_sim_score)
