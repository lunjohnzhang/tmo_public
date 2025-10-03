import gin
from typing import Collection, Optional, Tuple, Callable, Dict, ClassVar
from dataclasses import dataclass


@gin.configurable
@dataclass
class WarehouseConfig:
    """
    Config warehouse simulation

    Args:
        measure_names (list[str]): list of names of measures
        extinct_measure_names(list[str]): list of name of the measures used by
            extinct search in the inner loop
        aggregation_type (str): aggregation over `n_evals` results
        obj_type (str): type of objective
                        ("throughput",
                        "throughput_plus_n_shelf",
                        "throughput_minus_hamming_dist")
        hamming_obj_weight (float): alpha value of objective

        # Repair
        use_warm_up (bool): if True, will use the warm-up procedure. In
                            particular, for the initial population, the solution
                            returned from hamming distance objective will be
                            used. For mutated solutions, the solution of the
                            parent will be used.
        hamming_only (bool): if True, will use the warm-up repair solution
                             (repaired using hamming distance objective) as the
                             final repair result
        milp_n_threads (int): number of threads to use in MILP repair
        milp_time_limit (int): time limit for MILP repair

        # Shared by RHCR and PIBT
        scenario (str): scenario For RHCR, one of [KIVA, SORTING, ONLINE, BEE],
            for PIBT, one of [KIVA, SORTING, COMPETITION]
        left_w_weight (float): weight of the workstation on the left
        right_w_weight (float): weight of the workstation on the right
        simulation_time (int): run simulation
        num_agents (int): number of agents
        shelf_weight (float): weight of shelf while computing sim score
        package_mode (str): mode of given packages. One of ["dist", "explicit"].
            If "dist", packages are sampled from the distribution given by
            "package_dist_type". If "explicit", packages are given by
            "package_path".
        package_dist_type (str): One of ["uniform", "721", "kaggle_data"].
        package_path (str): path the list of the packages.
        n_destinations (int): number of destinations.
        # Online Update
        warm_up_time (int): number of timesteps to run with no guidance before
            the first online guidance graph update
        update_interval (int): number of timesteps between online guidance
            updates
        past_traffic_interval (int): number of timesteps in the past to use in
            the observation of guidance policy
        has_traffic_obs (bool): has traffic observation
        has_gg_obs (bool): has guidance graph observation
        has_future_obs (bool): has future observation
        has_task_obs (bool): has task observation
        has_curr_pos_obs (bool): has current position observation
        reset_weights_path (str): guidance graph weights to reset to, if any.

        # RHCR simulation
        task (str): input task file
        cutoffTime (int): cutoff time (seconds)
        screen (int): screen option (0: none; 1: results; 2:all)
        solver (str): solver (LRA, PBS, WHCA, ECBS)
        id (bool): independence detection
        single_agent_solver (str): single-agent solver (ASTAR, SIPP)
        lazyP (bool): use lazy priority
        simulation_window (int): call the planner every simulation_window
                                 timesteps
        travel_time_window (int): consider the traffic jams within the
                                  given window
        planning_window (int): the planner outputs plans with first
                                     planning_window timesteps collision-free
        potential_function (str): potential function (NONE, SOC, IC)
        potential_threshold (int): potential threshold
        rotation (bool): consider rotation
        robust (int): k-robust (for now, only work for PBS)
        CAT (bool): use conflict-avoidance table
        hold_endpoints (bool): Hold endpoints from Ma et al, AAMAS 2017
        dummy_paths (bool): Find dummy paths from Liu et al, AAMAS 2019
        prioritize_start (bool): Prioritize waiting at start locations
        suboptimal_bound (int): Suboptimal bound for ECBS
        log (bool): save the search trees (and the priority trees)
        test (bool): whether under testing mode.

        save_result (bool): Whether to allow C++ save the result of simulation
        save_solver (bool): Whether to allow C++ save the result of solver
        save_heuristics_table (bool): Whether to allow C++ save the result of
                                      heuristics table
        stop_at_traffic_jam (bool): whether stop the simulation at traffic jam

        # PIBT (or rather WPPL without LNS) simulation
        plan_time_limit (int): time limit of planning at each timestep
        preprocess_time_limit (int): time limit of preprocessing
        file_storage_path (str): path to save log files
        task_assignment_strategy (str): strategy of task assignment
        num_tasks_reveal (int): number of tasks revealed to the system in
                                advance
        gen_random (bool): if True, generate random tasks
        num_tasks (int): total number of tasks

    """
    # Measures.
    measure_names: Collection[str] = gin.REQUIRED
    extinct_measure_names: Collection[str] = None

    # Results.
    aggregation_type: str = gin.REQUIRED
    obj_type: str = gin.REQUIRED
    hamming_obj_weight: float = 1

    # Repair layouts
    use_warm_up: bool = True
    hamming_only: bool = True
    milp_n_threads: int = 1
    milp_time_limit: int = 60
    ub_chutes_per_dest: bool = True,

    # Simulation
    # Shared by PIBT and RHCR
    scenario: str = gin.REQUIRED
    left_w_weight: float = gin.REQUIRED
    right_w_weight: float = gin.REQUIRED
    simulation_time: int = gin.REQUIRED
    num_agents: int = gin.REQUIRED
    shelf_weight: float = 1
    package_mode: str = "dist"
    package_path: str = None
    package_dist_type: str = "721"
    n_destinations: int = 100
    task_assignment_cost: str = "heuristic+num_agents"
    assign_C: float = 8
    recirc_mechanism: bool = True
    task_waiting_time: int = 0
    workstation_waiting_time: int = 0
    task_change_time: int = -1 # -1 means no change
    task_gaussian_sigma: float = 0.01
    time_dist: bool = False
    time_sigma: int = 1000
    sleep_time_factor: float = 2
    sleep_time_noise_std: float = 100
    # Online update
    online_update: bool = False
    warmup_time: int = 50
    update_interval: int = 50
    past_traffic_interval: int = 50
    has_traffic_obs: bool = True
    has_gg_obs: bool = True
    has_future_obs: bool = False
    has_task_obs: bool = False
    has_curr_pos_obs: bool = False
    reset_weights_path: str = None

    # RHCR Simulation
    task: str = ""
    cutoffTime: int = 60
    screen: int = 0
    solver: str = "PBS"
    id: bool = False
    single_agent_solver: str = "SIPP"
    lazyP: bool = False
    simulation_window: int = 5
    travel_time_window: int = 0
    planning_window: int = 10
    potential_function: str = "NONE"
    potential_threshold: int = 0
    rotation: bool = False
    robust: int = 0
    CAT: bool = False
    hold_endpoints: bool = False
    dummy_paths: bool = False
    prioritize_start: bool = True
    suboptimal_bound: int = 1
    log: bool = False
    test: bool = False
    save_result: bool = False
    save_solver: bool = False
    save_heuristics_table: bool = False
    stop_at_traffic_jam: bool = True
    optimize_wait: bool = False

    # PIBT Simulation
    plan_time_limit: int = 1
    preprocess_time_limit: int = 1800
    task_assignment_strategy: str = "roundrobin"
    num_tasks_reveal: int = 1
    gen_random: bool = True
    num_tasks: int = 100000

    # PIU and Guidance policy
    bounds: Tuple = None
    iter_update_model_type: Callable = None
    iter_update_max_iter: int = None
    iter_update_n_sim: int = 1
    iter_update_mdl_kwargs: Dict = None
