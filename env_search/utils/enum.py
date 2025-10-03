from enum import Enum


class SearchAlgo(Enum):
    CLASSIC = "classic"
    EM = "em"
    EXTINCT = "extinct"
    CC = "cooperative-coevolution"


class SearchSpace(Enum):
    LAYOUT = "layout"
    G_GRAPH = "guidance-graph"
    G_POLICY = "guidance-policy"
    TASK_ASSIGN_POLICY = "task-assign-policy"
    LAYOUT_G_GRAPH = "layout_guidance-graph"
    CHUTE_CAPACITIES = "chute_capacities"
    CHUTE_MAPPING = "chute_mapping"
    G_GRAPH_CHUTE_CAPACITIES = "guidance-graph_chute_capacities"
    G_GRAPH_TASK_ASSIGN_POLICY = "guidance-graph_task-assign-policy"
    G_GRAPH_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY = "guidance-graph_chute_capacities_task-assign-policy"
    G_POLICY_CHUTE_CAPACITIES = "guidance-policy_chute_capacities"
    G_POLICY_TASK_ASSIGN_POLICY = "guidance-policy_task-assign-policy"
    G_POLICY_CHUTE_CAPACITIES_TASK_ASSIGN_POLICY = "guidance-policy_chute_capacities_task-assign-policy"
