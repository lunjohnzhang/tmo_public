import sys

# sys.path.append("build")
import os
import gymnasium as gym
import json
import numpy as np
import copy
import py_driver  # type: ignore # ignore pylance warning
import warehouse_sim  # type: ignore # ignore pylance warning
from py_sim import py_sim  # type: ignore # ignore pylance warning
from WarehouseSimulator import WarehouseSimulator  # type: ignore # ignore pylance warning
import shutil
import hashlib
import time
import subprocess
import gc
import fire
import gin

# from abc import ABC
from gymnasium import spaces

from env_search import LOG_DIR
from env_search.warehouse import get_packages, QUAD_TASK_ASSIGN_N_PARAM
from env_search.warehouse.config import WarehouseConfig
from env_search.warehouse.update_model.utils import (
    comp_uncompress_edge_matrix,
    comp_uncompress_vertex_matrix,
)
from env_search.utils import (
    kiva_obj_types,
    sortation_obj_types,
    kiva_uncompress_edge_weights,
    kiva_uncompress_wait_costs,
    min_max_normalize,
    load_pibt_default_config,
    get_n_valid_edges,
    get_n_valid_vertices,
    read_in_sortation_map,
    sortation_env_str2number,
    read_in_kiva_map,
    kiva_env_str2number,
)

from env_search.utils.logging import get_current_time_str
# from env_search.utils.task_generator import generate_task_and_agent
# from env_search.iterative_update.envs.utils import visualize_simulation

REDUNDANT_COMPETITION_KEYS = [
    "final_pos, final_tasks",
    "actual_paths",
    "starts",
    "exec_future",
    "plan_future",
    "exec_move",
    "plan_move",
    "past_paths",
    "done",
    "agents_finish_task",
]

DIRECTION2ID = {"R": 0, "D": 3, "L": 2, "U": 1, "W": 4}


def _process_output(output, delimiter1):
    """Helper function to process the output of the subprocess

    Args:
        output (str): output of the subprocess
        delimiter1 (str): the first delimiter
    """
    outputs = output.split(delimiter1)
    if len(outputs) <= 2:
        print(output)
        with open("debug.json", "w") as f:
            json.dump(output, f)
        raise ValueError(
            "Output is not in the correct format. Maybe simulation failed")
    else:
        results_str = outputs[1].replace('\n', '').replace('array', 'np.array')
        # print(collected_results_str)
        results = eval(results_str)
    return results


class IterUpdateEnvBase(gym.Env):
    """Iterative update env base."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        n_valid_vertices,
        n_valid_edges,
        max_iter=10,
        init_weight_file=None,
    ):
        super().__init__()
        self.i = 0  # timestep
        self.n_valid_vertices = n_valid_vertices
        self.n_valid_edges = n_valid_edges
        self.max_iter = max_iter
        self.init_weight_file = init_weight_file
        self.action_space = spaces.Box(low=-100,
                                       high=100,
                                       shape=(n_valid_edges +
                                              n_valid_vertices, ))

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(n_valid_edges +
                                                   n_valid_vertices, ))

    def step(self, action):
        return NotImplementedError()

    def reset(self, seed=None, options=None):
        # return observation, info
        if self.init_weight_file is None or self.init_weight_file == "":
            self.curr_edge_weights = np.ones(self.n_valid_edges)
            self.curr_wait_costs = np.ones(self.n_valid_vertices)
        else:
            with open(self.init_weight_file, "r") as f:
                map_json = json.load(f)
                all_weights = map_json["weights"]
                self.curr_edge_weights = np.array(
                    all_weights[self.n_valid_vertices:])
                self.curr_wait_costs = np.array(
                    all_weights[:self.n_valid_vertices])

        # Get baseline throughput
        self.i = 1  # We will run 1 simulation in reset
        init_result = self._run_sim(init_weight=True)
        self.init_throughput = init_result["throughput"]
        self.curr_throughput = init_result["throughput"]
        info = {
            "result": init_result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }
        return np.concatenate(
            [
                self.curr_wait_costs,
                self.curr_edge_weights,
            ],
            dtype=np.float64,
        ), info

    def render(self):
        return NotImplementedError()

    def close(self):
        return NotImplementedError()


class WarehouseIterUpdateEnv(IterUpdateEnvBase):

    def __init__(
        self,
        env_np: np.ndarray,
        map_json_str: str,
        n_valid_vertices,
        n_valid_edges,
        config: WarehouseConfig,
        seed=0,
        init_weight_file=None,
        output_dir="logs/",
        chute_mapping_json: str = None,
        task_assignment_params_json: str = None,
    ):
        super().__init__(
            n_valid_vertices=n_valid_vertices,
            n_valid_edges=n_valid_edges,
            max_iter=config.iter_update_max_iter,
            init_weight_file=init_weight_file,
        )
        # self.input_file = input_file
        self.config = config
        # self.simulation_time = simulation_time
        self.env_np = env_np
        self.map_json_str = map_json_str
        self.output_dir = output_dir
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.chute_mapping_json = chute_mapping_json
        self.task_assignment_params_json = task_assignment_params_json

        # Read in the package distribution
        _, self.package_dist_weight_json = get_packages(
            self.config.package_mode,
            self.config.package_dist_type,
            self.config.package_path,
            self.config.n_destinations,
        )

        if self.config.scenario == "KIVA":
            self.block_idx = [kiva_obj_types.index("@")]
        elif self.config.scenario == "SORTING":
            self.block_idx = [
                sortation_obj_types.index("@"),
                sortation_obj_types.index("T"),
            ]

        # Use CNN observation
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, *self.env_np.shape))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None

    def _gen_obs(self, result):
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])
        wait_cost_matrix = np.array(
            comp_uncompress_vertex_matrix(self.env_np,
                                          self.curr_wait_costs,
                                          self.block_idx,
                                          fill_value=0))
        edge_weight_matrix = np.array(
            comp_uncompress_edge_matrix(self.env_np,
                                        self.curr_edge_weights,
                                        self.block_idx,
                                        fill_value=0))

        # Normalize
        wait_usage_matrix = min_max_normalize(wait_usage_matrix, 0, 1)
        edge_usage_matrix = min_max_normalize(edge_usage_matrix, 0, 1)
        wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)

        h, w = self.env_np.shape
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)
        input = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix,
                wait_cost_matrix,
            ],
            axis=2,
            dtype=np.float64,
        )
        input = np.moveaxis(input, 2, 0)
        return input

    def _run_sim(self,
                 init_weight=False,
                 manually_clean_memory=True,
                 save_in_disk=True):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """
        # cmd = f"./lifelong_comp --inputFile {self.input_file} --simulationTime {self.simulation_time} --planTimeLimit 1 --fileStoragePath large_files/"

        # Initial weights are assumed to be valid
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                           self.ub).tolist()

        results = []
        kwargs = {
            "scenario": self.config.scenario,
            "map_json_str": self.map_json_str,
            "simulation_steps": self.config.simulation_time,
            "gen_random": self.config.gen_random,
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_agents,
            "weights": json.dumps(edge_weights),
            "wait_costs": json.dumps(wait_costs),
            "plan_time_limit": self.config.plan_time_limit,
            # "seed": int(self.rng.integers(100000)),
            "preprocess_time_limit": self.config.preprocess_time_limit,
            "file_storage_path": self.output_dir,
            "task_assignment_strategy": self.config.task_assignment_strategy,
            "num_tasks_reveal": self.config.num_tasks_reveal,
            "recirc_mechanism": self.config.recirc_mechanism,
            "task_waiting_time": self.config.task_waiting_time,
            "workstation_waiting_time": self.config.workstation_waiting_time,
            "config": load_pibt_default_config(),  # Use PIBT default config
            "task_change_time": self.config.task_change_time,
            "task_gaussian_sigma": self.config.task_gaussian_sigma,
            "time_dist": self.config.time_dist,
            "time_sigma": self.config.time_sigma,
            # Chute mapping related
            "package_dist_weight": self.package_dist_weight_json,
            "package_mode": self.config.package_mode,
            "chute_mapping": self.chute_mapping_json,
            "sleep_time_factor": self.config.sleep_time_factor,
            "sleep_time_noise_std": self.config.sleep_time_noise_std,

            # Task assign policy related
            "task_assignment_cost": self.config.task_assignment_cost,
            "task_assignment_params": self.task_assignment_params_json,
        }

        if not manually_clean_memory:
            results = []  # List[json]
            for _ in range(self.config.iter_update_n_sim):
                kwargs["seed"] = int(self.rng.integers(100000))
                result_jsonstr = py_driver.run(**kwargs)
                result = json.loads(result_jsonstr)
                results.append(result)
        else:
            if save_in_disk:
                file_dir = os.path.join(LOG_DIR, 'run_files')
                os.makedirs(file_dir, exist_ok=True)
                hash_obj = hashlib.sha256()
                raw_name = get_current_time_str().encode() + os.urandom(16)
                hash_obj.update(raw_name)
                file_name = hash_obj.hexdigest()
                file_path = os.path.join(file_dir, file_name)
                with open(file_path, 'w') as f:
                    json.dump(kwargs, f)
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                delimiter2 = "----DELIMITER2----DELIMITER2----"
                results = []
                for _ in range(self.config.iter_update_n_sim):
                    curr_seed = int(self.rng.integers(100000))
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

t0 = time.time()
kwargs_["seed"] = int({curr_seed})
t0 = time.time()
result_jsonstr = py_driver.run(**kwargs_)
t1 = time.time()
print("{delimiter2}")
print(t1-t0)
print("{delimiter2}")
result = json.loads(result_jsonstr)
np.set_printoptions(threshold=np.inf)

print("{delimiter1}")
print(result)
print("{delimiter1}")

                    """
                        ],
                        stdout=subprocess.PIPE).stdout.decode('utf-8')
                    t2 = time.time()
                    result = _process_output(output, delimiter1)
                    results.append(result)
                    gc.collect()
                # Delete temp kwargs file
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
results = []
for _ in range({self.config.iter_update_n_sim}):
    kwargs_["seed"] = int({self.rng.integers(100000)})
    result_jsonstr = py_driver.run(**kwargs_)
    result = json.loads(result_jsonstr)
    results.append(result)
np.set_printoptions(threshold=np.inf)
print("{delimiter1}")
print(results)
print("{delimiter1}")
                    """
                    ],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
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

                results = _process_output(output, delimiter1)

            gc.collect()
        # aggregate results
        keys = results[0].keys()
        collected_results = {key: [] for key in keys}

        for result_json in results:
            for key in keys:
                collected_results[key].append(result_json[key])
        for key in keys:
            collected_results[key] = np.mean(collected_results[key], axis=0)
        return collected_results

    def step(self, action):
        self.i += 1  # increment timestep

        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs
        wait_cost_update_vals = action[:self.n_valid_vertices]
        edge_weight_update_vals = action[self.n_valid_vertices:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        # Reward is difference between new throughput and current throughput
        result = self._run_sim()
        new_throughput = result["throughput"]
        reward = new_throughput - self.curr_throughput
        self.curr_throughput = new_throughput

        # terminated/truncate only if max iter is passed
        terminated = self.i >= self.max_iter
        truncated = terminated

        # Info includes the results
        info = {
            "result": result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        init_result = info["result"]
        return self._gen_obs(init_result), info


class PIBTWarehouseOnlineEnv:

    def __init__(
        self,
        env_np: np.ndarray,
        map_json_str: str,
        n_valid_vertices: int,
        n_valid_edges: int,
        config: WarehouseConfig,
        seed: int,
        chute_mapping_json: str = None,
        task_assignment_params_json: str = None,
    ):
        self.config = config
        assert (self.config.update_interval > 0)

        self.env_np = env_np
        self.map_json_str = map_json_str
        self.n_valid_vertices = n_valid_vertices
        self.n_valid_edges = n_valid_edges
        self.chute_mapping_json = chute_mapping_json
        self.task_assignment_params_json = task_assignment_params_json

        # Read in the package distribution
        _, self.package_dist_weight_json = get_packages(
            self.config.package_mode,
            self.config.package_dist_type,
            self.config.package_path,
            self.config.n_destinations,
        )

        if self.config.scenario == "KIVA":
            self.block_idx = [kiva_obj_types.index("@")]
        elif self.config.scenario == "SORTING":
            self.block_idx = [
                sortation_obj_types.index("@"),
                sortation_obj_types.index("T"),
            ]

        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

        # Use CNN observation
        h, w = self.env_np.shape
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, h, w))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None

    # def generate_video(self):
    #     visualize_simulation(self.comp_map, self.pos_hists,
    #                          "large_files_new/results.json")

    def update_paths_with_full_past(self, pos_hists, agents_paths):
        self.pos_hists = pos_hists
        self.move_hists = []
        for agent_path in agents_paths:
            self.move_hists.append(agent_path.replace(",", ""))

    def update_paths(self, agents_paths):
        h, w = self.env_np.shape
        for agent_moves, agent_new_paths in zip(self.move_hists, agents_paths):
            for s in agent_new_paths:
                if s == ",":
                    continue
                agent_moves.append(s)

        for i, agent_pos in enumerate(self.pos_hists):
            if len(agent_pos) == 0:
                agent_pos.append(self.starts[i])

            last_h, last_w = agent_pos[-1]

            for s in agents_paths[i]:
                if s == ",":
                    continue
                elif s == "R":
                    cur_pos = [last_h, last_w + 1]
                elif s == "D":
                    cur_pos = [last_h + 1, last_w]
                elif s == "L":
                    cur_pos = [last_h, last_w - 1]
                elif s == "U":
                    cur_pos = [last_h - 1, last_w]
                elif s == "W":
                    cur_pos = [last_h, last_w]
                else:
                    print(f"s = {s}")
                    raise NotImplementedError
                assert (cur_pos[0]>=0 and cur_pos[0]<h \
                    and cur_pos[1]>=0 and cur_pos[1]<w)
                agent_pos.append(cur_pos)
                last_h, last_w = agent_pos[-1]

    def _gen_future_obs(self, results):
        "exec_future, plan_future, exec_move, plan_move"
        # 5 dim
        h, w = self.env_np.shape
        exec_future_usage = np.zeros((5, h, w))
        for aid, (agent_path, agent_m) in enumerate(
                zip(results["exec_future"], results["exec_move"])):
            if aid in results["agents_finish_task"]:
                continue
            goal_id = results["final_tasks"][aid]
            for (x, y), m in zip(agent_path[1:], agent_m[1:]):
                if x * w + y == goal_id:
                    break
                d_id = DIRECTION2ID[m]
                exec_future_usage[d_id, x, y] += 1

        plan_future_usage = np.zeros((5, h, w))
        for aid, (agent_path, agent_m) in enumerate(
                zip(results["plan_future"], results["plan_move"])):
            if aid in results["agents_finish_task"]:
                continue
            goal_id = results["final_tasks"][aid]
            for (x, y), m in zip(agent_path, agent_m):
                if x * w + y == goal_id:
                    break
                d_id = DIRECTION2ID[m]
                plan_future_usage[d_id, x, y] += 1

        if exec_future_usage.sum() != 0:
            exec_future_usage = exec_future_usage / exec_future_usage.sum()
        if plan_future_usage.sum() != 0:
            plan_future_usage = plan_future_usage / plan_future_usage.sum()

        return exec_future_usage, plan_future_usage

    def _gen_traffic_obs_new(self, mode="init"):
        """Generate edge usage and wait usage observation

        Args:
            mode (str, optional): When the observation is generated. One of
                ["init", "mid", "end"]. Defaults to "init".

        Returns:
            _type_: _description_
        """
        h, w = self.env_np.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))
        tile_usage = np.zeros((1, h, w))

        if mode == "init":
            time_range = min(self.config.past_traffic_interval,
                             self.config.simulation_time - self.left_timesteps)
        elif mode == "mid":
            time_range = min(self.config.past_traffic_interval,
                             self.config.warmup_time)
        elif mode == "end":
            time_range = len(self.pos_hists[0]) - 1

        # print("time range =", time_range)
        for t in range(time_range):
            for agent_i in range(self.config.num_agents):
                prev_x, prev_y = self.pos_hists[agent_i][-(time_range + 1 - t)]
                # cur_x, cur_y = self.pos_hists[agent_i][-(self.config.past_traffic_interval-t)]
                # print(prev_x, prev_y)
                tile_usage[0, prev_x, prev_y] += 1

                move = self.move_hists[agent_i][-(time_range - t)]
                id = DIRECTION2ID[move]
                if id < 4:
                    edge_usage[id, prev_x, prev_y] += 1
                else:
                    # if env.i > 0:
                    #     print(prev_x, prev_y)
                    wait_usage[0, prev_x, prev_y] += 1

        # Normalize tile (vertex) and edge usage
        # wait usage should not be normalized as the total number of waits is
        # undetermined.
        if tile_usage.sum() != 0:
            tile_usage = tile_usage / tile_usage.sum() * 100
        if edge_usage.sum() != 0:
            edge_usage = edge_usage / edge_usage.sum() * 100
        return wait_usage, edge_usage, tile_usage

    def _gen_task_obs(self, result):
        h, w = self.env_np.shape
        task_usage = np.zeros((1, h, w))
        for aid, goal_id in enumerate(result["final_tasks"]):
            x = goal_id // w
            y = goal_id % w
            task_usage[0, x, y] += 1
        if task_usage.sum() != 0:
            task_usage = task_usage / task_usage.sum() * 10
        return task_usage

    def _gen_curr_pos_obs(self, result):
        h, w = self.env_np.shape
        pos_usage = np.zeros((2, h, w))
        for aid, (curr_id, goal_id) in enumerate(
                zip(result["final_pos"], result["final_tasks"])):
            x = curr_id // w
            y = curr_id % w

            gx = goal_id // w
            gy = goal_id % w

            pos_usage[0, x, y] = (gx - x) / h
            pos_usage[1, x, y] = (gy - y) / w
        return pos_usage

    def _gen_obs(self, result, mode="mid"):
        h, w = self.env_np.shape
        obs = np.zeros((0, h, w))
        if self.config.has_traffic_obs:

            wait_usage_matrix, edge_usage_matrix, _ = self._gen_traffic_obs_new(
                mode)
            traffic_obs = np.concatenate(
                [edge_usage_matrix, wait_usage_matrix],
                axis=0,
                dtype=np.float64)
            obs = np.concatenate([obs, traffic_obs], axis=0, dtype=np.float64)

        if self.config.has_gg_obs:
            wait_costs = min_max_normalize(self.curr_wait_costs, 0.1, 1)
            edge_weights = min_max_normalize(self.curr_edge_weights, 0.1, 1)
            wait_cost_matrix = np.array(
                comp_uncompress_vertex_matrix(
                    self.env_np,
                    wait_costs,
                    self.block_idx,
                    fill_value=0,
                ))
            edge_weight_matrix = np.array(
                comp_uncompress_edge_matrix(
                    self.env_np,
                    edge_weights,
                    self.block_idx,
                    fill_value=0,
                ))
            edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
            wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)

            edge_weight_matrix = np.moveaxis(edge_weight_matrix, 2, 0)
            wait_cost_matrix = np.moveaxis(wait_cost_matrix, 2, 0)

            gg_obs = np.concatenate([edge_weight_matrix, wait_cost_matrix],
                                    axis=0,
                                    dtype=np.float64)
            obs = np.concatenate([obs, gg_obs], axis=0, dtype=np.float64)

        if self.config.has_future_obs:
            exec_future_usage, plan_future_usage = self._gen_future_obs(result)
            obs = np.concatenate([obs, exec_future_usage + plan_future_usage],
                                 axis=0,
                                 dtype=np.float64)
        if self.config.has_task_obs:
            task_obs = self._gen_task_obs(result)
            obs = np.concatenate([obs, task_obs], axis=0, dtype=np.float64)
        if self.config.has_curr_pos_obs:
            curr_pos_obs = self._gen_curr_pos_obs(result)
            obs = np.concatenate([obs, curr_pos_obs], axis=0, dtype=np.float64)

        return obs

    def get_base_kwargs(self):
        # TODO: SEED!!!
        kwargs = {
            "scenario": self.config.scenario,
            "map_json_str": self.map_json_str,
            "simulation_steps": self.config.simulation_time,
            "gen_random": self.config.gen_random,
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_agents,
            "left_w_weight": self.config.left_w_weight,
            "right_w_weight": self.config.right_w_weight,
            "plan_time_limit": self.config.plan_time_limit,
            "seed": int(self.rng.integers(100000)),
            # "task_dist_change_interval": self.config.task_dist_change_interval,
            "preprocess_time_limit": self.config.preprocess_time_limit,
            "file_storage_path": "large_files",
            "task_assignment_strategy": self.config.task_assignment_strategy,
            "num_tasks_reveal": self.config.num_tasks_reveal,
            "assign_C": self.config.assign_C,
            "task_change_time": self.config.task_change_time,
            "task_gaussian_sigma": self.config.task_gaussian_sigma,
            "config": load_pibt_default_config(),  # Use PIBT default config
            # Online GGO related
            "warmup_steps": self.config.warmup_time,
            "update_gg_interval": self.config.update_interval,
            "recirc_mechanism": self.config.recirc_mechanism,
            "task_waiting_time": self.config.task_waiting_time,
            "workstation_waiting_time": self.config.workstation_waiting_time,
            "time_dist": self.config.time_dist,
            "time_sigma": self.config.time_sigma,
            "sleep_time_factor": self.config.sleep_time_factor,
            "sleep_time_noise_std": self.config.sleep_time_noise_std,
            # "h_update_late": self.config.h_update_late,
            # "dist_sigma": self.config.dist_sigma,
            # "dist_K": self.config.dist_K
            # "save_paths": True

            # Chute mapping related
            "package_dist_weight": self.package_dist_weight_json,
            "package_mode": self.config.package_mode,
            "chute_mapping": self.chute_mapping_json,

            # Task assign policy related
            "task_assignment_cost": self.config.task_assignment_cost,
            "task_assignment_params": self.task_assignment_params_json,
        }
        assert self.config.gen_random
        # if not self.config.gen_random:
        #     file_dir = os.path.join(get_project_dir(), 'run_files', 'gen_task')
        #     os.makedirs(file_dir, exist_ok=True)
        #     sub_dir_name = get_hash_file_name()
        #     self.task_save_dir = os.path.join(file_dir, sub_dir_name)
        #     os.makedirs(self.task_save_dir, exist_ok=True)

        #     generate_task_and_agent(self.config.map_base_path,
        #                             total_task_num=100000,
        #                             num_agents=self.config.num_agents,
        #                             save_dir=self.task_save_dir)

        #     kwargs["agents_path"] = os.path.join(self.task_save_dir,
        #                                          "test.agent")
        #     kwargs["tasks_path"] = os.path.join(self.task_save_dir,
        #                                         "test.task")
        # if self.config.task_dist_change_interval > 0:
        #     kwargs["task_random_type"] = self.config.task_random_type
        # if self.config.base_algo == "pibt":
        #     if self.config.has_future_obs:
        #         kwargs["config"] = load_w_pibt_default_config()
        #     else:
        #         kwargs["config"] = load_pibt_default_config()
        # elif self.config.base_algo == "wppl":
        #     kwargs["config"] = load_wppl_default_config()
        # else:
        #     print(f"base algo [{self.config.base_algo}] is not supported")
        #     raise NotImplementedError
        return kwargs

    def _run_sim(self,
                 init_weight=False,
                 manually_clean_memory=True,
                 save_in_disk=True):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """
        # Initial weights are assumed to be valid
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                           self.ub).tolist()

        kwargs = self.get_base_kwargs()
        kwargs["weights"] = json.dumps(edge_weights)
        kwargs["wait_costs"] = json.dumps(wait_costs)

        result_str = self.simulator.update_gg_and_step(edge_weights,
                                                       wait_costs)
        result = json.loads(result_str)

        self.left_timesteps -= self.config.update_interval
        self.left_timesteps = max(0, self.left_timesteps)
        return result

    def step(self, action):
        self.i += 1  # increment timestep
        # print(f"[step={self.i}]")
        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs
        wait_cost_update_vals = action[:self.n_valid_vertices]
        edge_weight_update_vals = action[self.n_valid_vertices:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        start = time.time()
        result = self._run_sim()
        self.sim_elapsed += time.time() - start

        # self.last_agent_pos = result["final_pos"]
        # self.last_tasks = result["final_tasks"]
        assert self.starts is not None
        self.update_paths(result["actual_paths"])

        new_task_finished = result["num_task_finished"]
        reward = new_task_finished - self.num_task_finished
        self.num_task_finished = new_task_finished

        # terminated/truncate if no left time steps
        terminated = result["done"]
        truncated = terminated
        if terminated or truncated:
            if not self.config.gen_random:
                if os.path.exists(self.task_save_dir):
                    shutil.rmtree(self.task_save_dir)
                else:
                    raise NotImplementedError

        result[
            "throughput"] = self.num_task_finished / self.config.simulation_time

        # Info includes the results
        sub_result = {
            k: v
            for k, v in result.items() if k not in REDUNDANT_COMPETITION_KEYS
        }

        # Simulation if ending, generate additional metadata
        # The offline version of the same metadata is generated in C++. The
        # data structure is therefore transformed to match the offline version
        # to that we don't need to change the downstream analysis code
        if terminated or truncated:
            wait_usage, _, tile_usage = self._gen_traffic_obs_new(mode="end")
            sub_result["vertex_wait_matrix"] = wait_usage.flatten().tolist()
            # sub_result["edge_usage_matrix"] = edge_usage
            sub_result["tile_usage"] = tile_usage.flatten().tolist()

        info = {
            "result": sub_result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.sim_elapsed = 0
        self.i = 0
        self.num_task_finished = 0
        self.left_timesteps = self.config.simulation_time
        self.last_agent_pos = None
        # self.last_tasks = None

        self.starts = None
        self.task_save_dir = None

        self.pos_hists = [[] for _ in range(self.config.num_agents)]
        self.move_hists = [[] for _ in range(self.config.num_agents)]

        self.last_wait_usage = np.zeros(np.prod(self.env_np.shape))
        self.last_edge_usage = np.zeros(4 * np.prod(self.env_np.shape))

        if self.config.reset_weights_path is None:
            self.curr_edge_weights = np.ones(self.n_valid_edges)
            self.curr_wait_costs = np.ones(self.n_valid_vertices)
        else:
            with open(self.config.reset_weights_path, "r") as f:
                weights_json = json.load(f)
            weights = weights_json["weights"]
            self.curr_wait_costs = np.array(weights[:self.n_valid_vertices])
            self.curr_edge_weights = np.array(weights[self.n_valid_vertices:])

        kwargs = self.get_base_kwargs()
        kwargs["weights"] = json.dumps(self.curr_edge_weights.tolist())
        kwargs["wait_costs"] = json.dumps(self.curr_wait_costs.tolist())

        # Create simulator and warmup (running with no guidance)
        # with open("WPPL/demo_kwargs.json", "w") as f:
        #     json.dump(kwargs, f)
        self.simulator = py_sim(**kwargs)
        result_str = self.simulator.warmup()
        result = json.loads(result_str)
        self.starts = result["starts"]
        self.update_paths(result["actual_paths"])

        obs = self._gen_obs(result, mode="init")
        info = {"result": {}}
        return obs, info


class RHCRWarehouseOnlineEnv:

    def __init__(
        self,
        map_np,
        map_json,
        num_agents,
        eval_logdir,
        n_valid_vertices,
        n_valid_edges,
        config: WarehouseConfig,
        seed: int,
        chute_mapping_json: str = None,
        task_assignment_params_json: str = None,
    ):
        self.n_valid_vertices = n_valid_vertices
        self.n_valid_edges = n_valid_edges
        self.chute_mapping_json = chute_mapping_json
        self.task_assignment_params_json = task_assignment_params_json

        self.config = config
        self.map_np = map_np
        self.map_json = map_json
        self.num_agents = num_agents
        self.eval_logdir = eval_logdir
        if self.config.scenario == "KIVA":
            self.block_idx = [kiva_obj_types.index("@")]
        elif self.config.scenario == "SORTING":
            self.block_idx = [
                sortation_obj_types.index("@"),
                sortation_obj_types.index("T"),
            ]

        self.rng = np.random.default_rng(seed=seed)

        # Read in the package distribution
        _, self.package_dist_weight_json = get_packages(
            self.config.package_mode,
            self.config.package_dist_type,
            self.config.package_path,
            self.config.n_destinations,
        )

        # Use CNN observation
        h, w = self.map_np.shape
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, h, w))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None

    def _gen_traffic_obs_ols(self, result):
        h, w = self.map_np.shape
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])
        # Normalize
        wait_usage_matrix = min_max_normalize(wait_usage_matrix, 0, 1)
        edge_usage_matrix = min_max_normalize(edge_usage_matrix, 0, 1)
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)

        traffic_obs = np.concatenate([edge_usage_matrix, wait_usage_matrix],
                                     axis=2)
        traffic_obs = np.moveaxis(traffic_obs, 2, 0)
        return traffic_obs

    def _gen_traffic_obs(self, is_init):
        h, w = self.map_np.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))

        if not is_init:
            time_range = min(self.config.past_traffic_interval,
                             self.config.simulation_time - self.left_timesteps)
        else:
            time_range = min(self.config.past_traffic_interval,
                             self.config.warmup_time)

        for t in range(time_range):
            for agent_i in range(self.num_agents):
                prev_loc = self.pos_hists[agent_i][-(time_range + 1 - t)]
                curr_loc = self.pos_hists[agent_i][-(time_range - t)]

                prev_r, prev_c = prev_loc // w, prev_loc % w
                if prev_loc == curr_loc:
                    wait_usage[0, prev_r, prev_c] += 1
                elif prev_loc + 1 == curr_loc:  # R
                    edge_usage[0, prev_r, prev_c] += 1
                elif prev_loc + w == curr_loc:  # D
                    edge_usage[1, prev_r, prev_c] += 1
                elif prev_loc - 1 == curr_loc:  # L
                    edge_usage[2, prev_r, prev_c] += 1
                elif prev_loc - w == curr_loc:  # U
                    edge_usage[3, prev_r, prev_c] += 1
                else:
                    print(prev_loc, curr_loc)
                    print(self.pos_hists[agent_i])
                    raise NotImplementedError

        # print("max", wait_usage.max(), wait_usage.argmax(), edge_usage.max(), edge_usage.argmax())
        if wait_usage.sum() != 0:
            wait_usage = wait_usage / wait_usage.sum() * 100
        if edge_usage.sum() != 0:
            edge_usage = edge_usage / edge_usage.sum() * 100
        # print("new, wait_usage:", wait_usage.max(), "edge_usage:", edge_usage.max())
        traffic_obs = np.concatenate([edge_usage, wait_usage], axis=0)
        return traffic_obs

    def _gen_future_obs(self, result):
        h, w = self.map_np.shape
        edge_usage = np.zeros((4, h, w))
        wait_usage = np.zeros((1, h, w))

        for agent_i in range(self.num_agents):
            for t in range(len(result["future_paths"][agent_i]) - 1):
                prev_loc = result["future_paths"][agent_i][t]
                curr_loc = result["future_paths"][agent_i][t + 1]

                prev_r, prev_c = prev_loc // w, prev_loc % w
                if prev_loc == curr_loc:
                    wait_usage[0, prev_r, prev_c] += 1
                elif prev_loc + 1 == curr_loc:  # R
                    edge_usage[0, prev_r, prev_c] += 1
                elif prev_loc + w == curr_loc:  # D
                    edge_usage[1, prev_r, prev_c] += 1
                elif prev_loc - 1 == curr_loc:  # L
                    edge_usage[2, prev_r, prev_c] += 1
                elif prev_loc - w == curr_loc:  # U
                    edge_usage[3, prev_r, prev_c] += 1
                else:
                    print(prev_loc, curr_loc)
                    raise NotImplementedError

        # print("max", wait_usage.max(), wait_usage.argmax(), edge_usage.max(), edge_usage.argmax())
        if wait_usage.sum() != 0:
            wait_usage = wait_usage / wait_usage.sum() * 100
        if edge_usage.sum() != 0:
            edge_usage = edge_usage / edge_usage.sum() * 100
        # print("new, wait_usage:", wait_usage.max(), "edge_usage:", edge_usage.max())
        future_obs = np.concatenate([wait_usage, edge_usage], axis=0)
        return future_obs

    def _gen_gg_obs(self):
        edge_weight_matrix = np.array(
            kiva_uncompress_edge_weights(self.map_np,
                                         self.curr_edge_weights,
                                         self.block_idx,
                                         fill_value=0))
        # While optimizing all wait costs, all entries of `wait_cost_matrix`
        # are different.
        if self.config.optimize_wait:
            wait_cost_matrix = np.array(
                kiva_uncompress_wait_costs(self.map_np,
                                           self.curr_wait_costs,
                                           self.block_idx,
                                           fill_value=0))
        # Otherwise, `self.curr_wait_costs` is a single number and so all wait
        # costs are the same, but we need to transform it to a matrix.
        else:
            curr_wait_costs_compress = np.zeros(self.n_valid_vertices)
            curr_wait_costs_compress[:] = self.curr_wait_costs
            wait_cost_matrix = np.array(
                kiva_uncompress_wait_costs(self.map_np,
                                           curr_wait_costs_compress,
                                           self.block_idx,
                                           fill_value=0))
        wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)

        h, w = self.map_np.shape
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)

        gg_obs = np.concatenate([edge_weight_matrix, wait_cost_matrix], axis=2)
        gg_obs = np.moveaxis(gg_obs, 2, 0)
        return gg_obs

    def _gen_task_obs(self, result):
        h, w = self.map_np.shape
        task_usage = np.zeros((1, h, w))
        for aid, goal_id in enumerate(result["goal_locs"]):
            x = goal_id // w
            y = goal_id % w
            task_usage[0, x, y] += 1
        if task_usage.sum() != 0:
            task_usage = task_usage / task_usage.sum() * 10
        return task_usage

    def _gen_obs(self, result, is_init=False):
        h, w = self.map_np.shape
        obs = np.zeros((0, h, w))
        if self.config.has_traffic_obs:
            traffic_obs = self._gen_traffic_obs(is_init)
            obs = np.concatenate([obs, traffic_obs], axis=0)
        if self.config.has_gg_obs:
            gg_obs = self._gen_gg_obs()
            obs = np.concatenate([obs, gg_obs], axis=0)
        if self.config.has_future_obs:
            future_obs = self._gen_future_obs(result)
            obs = np.concatenate([obs, future_obs], axis=0)
        if self.config.has_task_obs:
            task_obs = self._gen_task_obs(result)
            obs = np.concatenate([obs, task_obs], axis=0)
        return obs

    def gen_base_kwargs(self, ):
        sim_seed = self.rng.integers(10000)
        kwargs = {
            "seed": int(sim_seed),
            "output": os.path.join(self.eval_logdir,
                                   f"online-seed={sim_seed}"),
            "scenario": self.config.scenario,
            "task": self.config.task,
            "agentNum": self.num_agents,
            "cutoffTime": self.config.cutoffTime,
            # "OverallCutoffTime": self.config.overallCutoffTime,
            "screen": self.config.screen,
            # "screen": 1,
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
            "warmup_time": self.config.warmup_time,
            "update_gg_interval": self.config.update_interval,
            # "task_dist_update_interval": self.config.task_dist_update_interval,
            # "task_dist_type": self.config.task_dist_type,
            # "dist_sigma": self.config.dist_sigma,
            # "dist_K": self.config.dist_K
            # Chute mapping related
            "package_dist_weight": self.package_dist_weight_json,
            "package_mode": self.config.package_mode,
            "chute_mapping": self.chute_mapping_json,

            # Task assign policy related
            "task_assignment_cost": self.config.task_assignment_cost,
            "task_assignment_params": self.task_assignment_params_json,
        }
        return kwargs

    def _run_sim(self, init_weight=False):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """

        # Initial weights are assumed to be valid and optimize_waits = True
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
            new_weights = [*wait_costs, *edge_weights]
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            if self.config.optimize_wait:
                wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                               self.ub).tolist()
                new_weights = [*wait_costs, *edge_weights]
            else:
                all_weights = [self.curr_wait_costs, *edge_weights]
                all_weights = min_max_normalize(all_weights, self.lb, self.ub)
                new_weights = all_weights.tolist()

        # print(new_weights[:5])
        # raise NotImplementedError
        result_jsonstr = self.simulator.update_gg_and_step(
            self.config.optimize_wait, new_weights)
        result = json.loads(result_jsonstr)

        self.left_timesteps -= self.config.update_interval
        self.left_timesteps = max(0, self.left_timesteps)
        return result

    def step(self, action):
        # self.i += 1  # increment timestep

        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs

        if self.config.optimize_wait:
            wait_cost_update_vals = action[:self.n_valid_vertices]
            edge_weight_update_vals = action[self.n_valid_vertices:]
        else:
            wait_cost_update_vals = action[0]
            edge_weight_update_vals = action[1:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        # Reward is difference between new throughput and current throughput
        result = self._run_sim()
        self.update_paths(result["all_paths"])
        new_tasks_finished = result["num_tasks_finished"]
        reward = new_tasks_finished - self.num_tasks_finished
        self.num_tasks_finished = new_tasks_finished

        return_result = {}
        return_result[
            "throughput"] = self.num_tasks_finished / self.config.simulation_time

        done = result["done"]
        timeout = result["timeout"]
        congested = result["congested"]

        terminated = done | timeout | congested
        truncated = terminated

        # Info includes the results
        info = {
            "result": return_result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.num_tasks_finished = 0
        self.left_timesteps = self.config.simulation_time
        self.pos_hists = [[] for _ in range(self.num_agents)]
        # self.move_hists = [[] for _ in range(self.num_agents)]

        kwargs = self.gen_base_kwargs()
        curr_map_json = copy.deepcopy(self.map_json)
        curr_map_json["weight"] = False
        kwargs["map"] = json.dumps(curr_map_json)
        self.simulator = WarehouseSimulator(**kwargs)
        result_s = self.simulator.warmup()
        result = json.loads(result_s)
        self.update_paths(result["all_paths"])
        obs = self._gen_obs(result, is_init=True)
        info = {"result": {}}
        return obs, info

    def update_paths(self, all_paths):
        for aid, agent_path in enumerate(all_paths):
            if len(self.pos_hists[aid]) == 0:
                self.pos_hists[aid].extend(agent_path)
            else:
                self.pos_hists[aid].extend(agent_path[1:])
            # print(self.pos_hists[aid])


def test_piu(
    cfg_file_path,
    map_path,
    chute_mapping_file,
    domain="sortation",
):
    gin.parse_config_file(cfg_file_path)
    config = WarehouseConfig()

    with open(chute_mapping_file, "r") as f:
        chute_mapping_json = json.load(f)
        chute_mapping_json = json.dumps(chute_mapping_json)

    with open(map_path, "r") as f:
        map_json_str = json.load(f)

    map_str, _ = read_in_sortation_map(map_path)
    map_np = sortation_env_str2number(map_str)
    n_valid_vertices = get_n_valid_vertices(map_np, domain)
    n_valid_edges = get_n_valid_edges(map_np, bi_directed=True, domain=domain)
    task_assignment_params = np.ones(QUAD_TASK_ASSIGN_N_PARAM).tolist()

    iter_update_env = WarehouseIterUpdateEnv(
        map_np,
        json.dumps(map_json_str),
        n_valid_vertices,
        n_valid_edges,
        config,
        seed=0,
        chute_mapping_json=chute_mapping_json,
        task_assignment_params_json=json.dumps(task_assignment_params),
    )
    obs, info = iter_update_env.reset()
    done = False
    while not done:
        # Get update value
        wait_cost_update_vals, edge_weight_update_vals = np.random.rand(
            n_valid_vertices), np.random.rand(n_valid_edges)

        # Perform update
        obs, imp_throughput, done, _, info = iter_update_env.step(
            np.concatenate([
                wait_cost_update_vals,
                edge_weight_update_vals,
            ]))

        curr_result = info["result"]

    print(curr_result["throughput"])


def test_pibt_online_env(
    cfg_file_path,
    map_path,
    chute_mapping_file,
    domain="sortation",
):
    gin.parse_config_file(cfg_file_path)
    config = WarehouseConfig()

    with open(chute_mapping_file, "r") as f:
        chute_mapping_json = json.load(f)
        chute_mapping_json = json.dumps(chute_mapping_json)

    with open(map_path, "r") as f:
        map_json_str = json.load(f)

    map_str, _ = read_in_sortation_map(map_path)
    map_np = sortation_env_str2number(map_str)
    n_valid_vertices = get_n_valid_vertices(map_np, domain)
    n_valid_edges = get_n_valid_edges(map_np, bi_directed=True, domain=domain)
    task_assignment_params = np.ones(QUAD_TASK_ASSIGN_N_PARAM).tolist()

    env = PIBTWarehouseOnlineEnv(
        map_np,
        json.dumps(map_json_str),
        n_valid_vertices,
        n_valid_edges,
        config,
        seed=0,
        chute_mapping_json=chute_mapping_json,
        task_assignment_params_json=json.dumps(task_assignment_params),
    )

    def vis_arr(arr_, mask=None, name="test"):
        arr = arr_.copy()
        save_dir = "plots"
        os.makedirs(save_dir, exist_ok=True)
        import matplotlib.pyplot as plt
        if mask is not None:
            arr = np.ma.masked_where(mask, arr)
        cmap = plt.cm.Reds
        # cmap.set_bad(color='black')
        plt.imshow(arr, cmap=cmap, interpolation="none")
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{name}.png"))
        plt.close()

    np.set_printoptions(threshold=np.inf)
    start = time.time()
    obs, info = env.reset()
    for i in range(5):
        vis_arr(obs[i], name=f"step{env.i}_traffic{i}")

    done = False
    while not done:
        # print(obs.shape)
        action = np.random.rand(n_valid_vertices + n_valid_edges)
        obs, reward, terminated, truncated, info = env.step(action)
        # for i in range(5):
        #     vis_arr(obs[i], name=f"step{env.i}_traffic{i}")
        done = terminated or truncated
    elapsed = time.time() - start
    print(f"Sim elapsed time = {env.sim_elapsed}")
    print(f"Elapsed time = {elapsed}")
    print("Throughput:", info["result"]["throughput"])


def test_rhcr_online_env(
    cfg_file_path,
    map_path,
    chute_mapping_file,
    domain="sortation",
):
    from env_search.utils import get_n_valid_edges, get_n_valid_vertices
    from env_search.warehouse.update_model.update_model import WarehouseCNNUpdateModel
    np.random.seed(0)
    gin.parse_config_file(cfg_file_path, skip_unknown=True)
    cfg = WarehouseConfig()

    with open(chute_mapping_file, "r") as f:
        chute_mapping_json = json.load(f)
        chute_mapping_json = json.dumps(chute_mapping_json)
    task_assignment_params = np.ones(QUAD_TASK_ASSIGN_N_PARAM).tolist()

    if domain == "kiva":
        base_map_str, _ = read_in_kiva_map(map_path)
        base_map_np = kiva_env_str2number(base_map_str)
    elif domain == "sortation":
        base_map_str, _ = read_in_sortation_map(map_path)
        base_map_np = sortation_env_str2number(base_map_str)
    n_valid_vertices = get_n_valid_vertices(base_map_np, domain)
    n_valid_edges = get_n_valid_edges(base_map_np,
                                      bi_directed=True,
                                      domain=domain)
    with open(map_path, "r") as f:
        map_json = json.load(f)

    def vis_arr(arr_, mask=None, name="test"):
        arr = arr_.copy()
        save_dir = "env_search/warehouse/plots"
        os.makedirs(save_dir, exist_ok=True)
        import matplotlib.pyplot as plt
        if mask is not None:
            arr = np.ma.masked_where(mask, arr)
        cmap = plt.cm.Reds
        # cmap.set_bad(color='black')
        plt.imshow(arr, cmap=cmap, interpolation="none")
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{name}.png"))
        plt.close()

    for i in range(1):
        env = RHCRWarehouseOnlineEnv(
            base_map_np,
            map_json,
            num_agents=cfg.num_agents,
            eval_logdir='test',
            n_valid_vertices=n_valid_vertices,
            n_valid_edges=n_valid_edges,
            config=cfg,
            seed=0,
            chute_mapping_json=chute_mapping_json,
            task_assignment_params_json=json.dumps(task_assignment_params),
        )
        cnt = 0
        obs, info = env.reset()
        # for i in range(4, 5):
        #     vis_arr(obs[i], name=f"step{cnt}_traffic{i}")
        # vis_arr(obs[-1], name=f"step{cnt}_task")

        done = False
        while not done:
            cnt += 1
            action = np.random.rand(1 + n_valid_edges)
            obs, reward, terminated, truncated, info = env.step(action)
            # for i in range(4, 5):
            #     vis_arr(obs[i], name=f"step{cnt}_traffic{i}")
            # vis_arr(obs[-1], name=f"step{cnt}_task")
            done = terminated or truncated

        print("tp =", info["result"]["throughput"])


if __name__ == "__main__":
    fire.Fire({
        "rhcr": test_rhcr_online_env,
        "pibt": test_pibt_online_env,
        "piu": test_piu,
    })
