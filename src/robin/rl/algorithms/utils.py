"""Utils module for RL algorithms."""

from robin.rl.entities import (
    RobinEnvFactory, StatsSubprocVectorEnv, VectorMultiAgentEnvNormObsReward, VectorMultiAgentGraphEnvNormObsReward
)
from robin.rl.algorithms.constants import IS_COOPERATIVE, IS_GRAPH_BASED

from typing import Union


def create_env(
    supply_config: str,
    demand_config: str,
    algorithm: str,
    seed: int,
    seed_rank_multiplier: int,
    n_workers: int,
    run_name: str,
    normalize_obs: bool,
    normalize_obs_output: bool,
    reward_scale_factor: Union[float, None] = None,
    exclude_edge_type: Union[int, None] = None
) -> StatsSubprocVectorEnv:
    """
    Create the ROBIN environment.

    Args:
        supply_config (str): Supply configuration file.
        demand_config (str): Demand configuration file.
        algorithm (str): Algorithm to use.
        seed (int): Seed of the experiment.
        seed_rank_multiplier (int): Seed rank multiplier.
        n_workers (int): Number of workers.
        run_name (str): Name of the run.
        normalize_obs (bool): Whether to normalize observations.
        normalize_obs_output (bool): Whether to normalize on-the-fly observations.
        reward_scale_factor (float, optional): Scale factor for rewards.
        exclude_edge_type (int, optional): Edge type to exclude from the graph representation.

    Returns:
        StatsSubprocVectorEnv: The ROBIN environment.
    """
    env_fns = [
        lambda: RobinEnvFactory.create(
            path_config_supply=supply_config,
            path_config_demand=demand_config,
            multi_agent=True,
            cooperative=IS_COOPERATIVE[algorithm],
            graph=IS_GRAPH_BASED[algorithm],
            discrete_action_space=False,
            exclude_edge_type=exclude_edge_type,
            seed=seed + i * seed_rank_multiplier
        ) for i in range(n_workers)
    ]
    env = StatsSubprocVectorEnv(env_fns=env_fns, log_dir=run_name)
    if normalize_obs:
        kwargs = {'reward_scale_factor': reward_scale_factor, 'normalize_obs_output': normalize_obs_output}
        if IS_GRAPH_BASED[algorithm]:
            env = VectorMultiAgentGraphEnvNormObsReward(env, **kwargs)
        else:
            env = VectorMultiAgentEnvNormObsReward(env, **kwargs)
    env.seed([seed + i * seed_rank_multiplier for i in range(n_workers)])
    return env
