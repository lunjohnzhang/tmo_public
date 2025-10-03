import os
import json
import gin
import fire
import numpy as np
import py_driver  # type: ignore # ignore pylance warning
from typing import Callable
from env_search.utils import kiva_env_str2number
from env_search.warehouse.config import WarehouseConfig
from env_search.warehouse.module import WarehouseModule


def iterative_update_cmd(
    env_np,
    map_json_str,
    n_valid_edges,
    n_valid_vertices,
    config_file,
    seed=0,
    init_weight_file=None,
    output=".",
    model_params=None,
    domain="kiva",
    num_agents=None,
):
    gin.parse_config_file(config_file)
    if domain == "kiva":
        config = WarehouseConfig()
        # overwrite num agents
        if num_agents is not None and num_agents != "":
            config.num_agents = num_agents
        module = WarehouseModule(config)
        return module.evaluate_iterative_update(
            env_np=env_np,
            map_json_str=map_json_str,
            model_params=model_params
            if model_params is not None else np.random.rand(4271),
            n_valid_edges=n_valid_edges,
            n_valid_vertices=n_valid_vertices,
            output_dir=output,
            seed=seed,
        )


if __name__ == "__main__":
    fire.Fire(iterative_update_cmd)
