"""Entities for the rl module."""

import numpy as np
import torch

from robin.kernel.entities import Kernel
from robin.supply.entities import Supply
from robin.rl.constants import (
    ACTION_FACTOR, CLIP_MAX, FLATTENED_PRICE_POSITION, FLATTENED_PROFIT_POSITION, FLATTENED_TICKETS_SOLD_POSITION,
    HIGH_ACTION, HIGH_PRICE, LOG_DIR, LOW_ACTION, LOW_PRICE, NODE_FEATURE_PROFIT_POSITION, NUMBER_ACTIONS, START_ACTION
)
from robin.rl.algorithms.constants import BASIC_FEATURES, CATEGORICAL_PER_MARKET_SEAT, FEATURES_PER_MARKET_SEAT

from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from gymnasium import ActionWrapper, Env, ObservationWrapper
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from gymnasium.wrappers import FlattenObservation
from methodtools import lru_cache
from numpy.typing import NDArray
from pathlib import Path
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import SubprocVectorEnv, VectorEnvNormObs, VectorEnvWrapper
from tianshou.env.venvs import BaseVectorEnv
from tianshou.utils import RunningMeanStd
from tianshou.env.utils import gym_new_venv_step_type
from typing import Any, Tuple, Union


class FlattenAction(ActionWrapper):
    """Action wrapper that flattens the action space."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self.action_space = flatten_space(self.env.action_space)
        
    def action(self, action) -> NDArray:
        return unflatten(self.env.action_space, action)


class FlattenMultiAction(ActionWrapper):
    """Action wrapper that flattens the action space for multiple agents."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.action_space = [flatten_space(space) for space in self.env.action_space]

    def action(self, action: list) -> list[NDArray]:
        return [unflatten(space, act) for space, act in zip(self.env.action_space, action)]


class FlattenObservation(FlattenObservation):
    pass


class FlattenMultiObservation(ObservationWrapper):
    """Observation wrapper that flattens the observation space for multiple agents."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.observation_space = [flatten_space(space) for space in self.env.observation_space]

    def observation(self, observation: list):
        return [flatten(space, obs) for space, obs in zip(self.env.observation_space, observation)]


class PyGObservationWrapper(ObservationWrapper):
    """Observation wrapper that converts graph dictionaries to PyTorch Geometric Data objects."""
    
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.observation_space = self.env.observation_space
        
    def observation(self, observation: dict) -> dict:
        observation['services_graph'] = Data(
            x=observation['services_graph']['x'], edge_index=observation['services_graph']['edge_index'],
            edge_type=observation['services_graph']['edge_type']
        )   
        return observation


class HeterogeneousRunningMeanStd(RunningMeanStd):
    """
    Calculate the running mean and std of a data stream.
    
    NOTE: This class supports heterogeneous data types.
    """
    
    def norm(self, data_array: float | np.ndarray) -> float | np.ndarray:
        """
        Normalize the data array.
        
        Args:
            data_array (float | np.ndarray): Data array to normalize.
        
        Returns:
            float | np.ndarray: Normalized data array.
        """
        var = np.array([np.sqrt(var + self.eps) for var in self.var], dtype=object)
        data_array = (data_array - self.mean) / var
        if self.clip_max:
            data_array = np.reshape(
                np.array(
                    [np.clip(agent, -self.clip_max, self.clip_max) for data in data_array for agent in data],
                    dtype=object,
                ), data_array.shape
            )
        return data_array


class VectorEnvNormReward(VectorEnvWrapper):
    """
    Vector environment with normalized rewards.
    
    Attributes:
        reward_rms (RunningMeanStd): Running mean/std for the rewards.
        update_reward_rms (bool): Whether to update the reward running mean/std.
    """
    
    def __init__(
        self,
        venv: BaseVectorEnv,
        update_reward_rms: bool = True,
        clip_max: float = CLIP_MAX,
        is_heterogeneous: bool = False
    ) -> None:
        """
        Initialize the vector environment with normalized rewards.
        
        Args:
            venv (BaseVectorEnv): Vector environment.
            update_reward_rms (bool): Whether to update the reward running mean/std.
            clip_max (float): Maximum absolute value for the data array.
            is_heterogeneous (bool): Whether the data array is heterogeneous.
        """
        super().__init__(venv)
        self.reward_rms = HeterogeneousRunningMeanStd(clip_max=clip_max) if is_heterogeneous \
            else RunningMeanStd(clip_max=clip_max)
        self.update_reward_rms = update_reward_rms
        
    def step(
        self,
        action: Union[np.ndarray, torch.Tensor],
        id: Union[int, list[int], np.ndarray, None] = None,
    ) -> gym_new_venv_step_type:
        step_results = super().step(action, id)
        # Normalize reward
        if self.reward_rms and self.update_reward_rms:
            self.reward_rms.update(step_results[1])
        return (step_results[0], self._norm_reward(step_results[1]), *step_results[2:])

    def _norm_reward(self, reward: float) -> np.ndarray:
        """Normalize the reward."""
        if self.reward_rms:
            return self.reward_rms.norm(reward)
        return reward

    def set_reward_rms(self, reward_rms: RunningMeanStd) -> None:
        """Set with given reward running mean/std."""
        self.reward_rms = reward_rms
    
    def get_reward_rms(self) -> RunningMeanStd:
        """Return reward running mean/std."""
        return self.reward_rms


class VectorEnvNormObsReward(VectorEnvNormObs):
    """
    Vector environment with normalized observations and rewards.

    Attributes:
        obs_rms (RunningMeanStd): Running mean/std for the observations.
        reward_rms (RunningMeanStd): Running mean/std for the rewards.
        update_reward_rms (bool): Whether to update the reward running mean/std.
        normalize_obs_output (bool): Whether to normalize the returned observations. When False, still updates
            obs_rms statistics but returns raw observations.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        update_reward_rms: bool = True,
        clip_max: float = CLIP_MAX,
        is_heterogeneous: bool = False,
        normalize_obs_output: bool = True
    ) -> None:
        """
        Initialize the vector environment with normalized observations and rewards.

        Args:
            venv (BaseVectorEnv): Vector environment.
            update_obs_rms (bool): Whether to update the observation running mean/std.
            update_reward_rms (bool): Whether to update the reward running mean/std.
            clip_max (float): Maximum absolute value for the data array.
            is_heterogeneous (bool): Whether the data array is heterogeneous.
            normalize_obs_output (bool): Whether to normalize the returned observations. When False, still updates
                obs_rms statistics but returns raw observations.
        """
        super().__init__(venv, update_obs_rms)
        self.obs_rms = HeterogeneousRunningMeanStd(clip_max=clip_max) if is_heterogeneous \
            else RunningMeanStd(clip_max=clip_max)
        self.update_reward_rms = update_reward_rms
        self.reward_rms = HeterogeneousRunningMeanStd(clip_max=clip_max) if is_heterogeneous \
            else RunningMeanStd(clip_max=clip_max)
        self.normalize_obs_output = normalize_obs_output

    def _extract_and_normalize_continuous_features(self, obs: np.ndarray) -> np.ndarray:
        normalized_obs = obs.copy()
        # Normalize only continuous features
        feature_length = obs.shape[-1]
        continuous_mask = self._get_continuous_feature_mask(feature_length)
        continuous_features = obs[:, continuous_mask] # [n_envs, n_services * n_continuous]
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(continuous_features)
        if self.normalize_obs_output:
            # Reconstruct observation with normalized continuous features
            normalized_continuous = self._norm_obs(continuous_features)
            normalized_obs[:, continuous_mask] = normalized_continuous
        return normalized_obs

    @lru_cache(maxsize=None)
    def _get_continuous_feature_mask(self, feature_length: int) -> np.ndarray:
        mask = np.zeros(feature_length, dtype=bool)
        # Based on alphabetical flattening: corridor, line, market_seats, profit, rolling_stock, time_slot, tsp
        # market_seats expands to: destination, origin, price, seat_type, tickets_sold
        # So price is at position 4 + seat_idx * FEATURES_PER_MARKET_SEAT
        # tickets_sold at position 6 + seat_idx * FEATURES_PER_MARKET_SEAT
        # profit is at position 7 + (num_market_seats - 1) * FEATURES_PER_MARKET_SEAT (at the end of market seat features)
        mask = np.zeros(feature_length, dtype=bool)
        supply: Supply = self.venv.get_env_attr('kernel')[0].supply
        block_start = 0
        for service in supply.services:
            num_market_seats = sum(len(seats) for seats in service.prices.values())
            for seat_idx in range(num_market_seats):
                mask[block_start + FLATTENED_PRICE_POSITION + seat_idx * FEATURES_PER_MARKET_SEAT] = True
                mask[block_start + FLATTENED_TICKETS_SOLD_POSITION + seat_idx * FEATURES_PER_MARKET_SEAT] = True
            mask[block_start + FLATTENED_PROFIT_POSITION + (num_market_seats - 1) * FEATURES_PER_MARKET_SEAT] = True
            block_start += BASIC_FEATURES + num_market_seats * FEATURES_PER_MARKET_SEAT
        return mask

    def reset(
        self,
        id: int | list[int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict | list[dict]]:
        obs, info = self.venv.reset(id, **kwargs)
        if isinstance(obs, tuple):  # type: ignore
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        obs = self._extract_and_normalize_continuous_features(obs)
        return obs, info

    def _step(
        self,
        action: np.ndarray | torch.Tensor,
        id: int | list[int] | np.ndarray | None = None,
    ) -> gym_new_venv_step_type:
        step_results = self.venv.step(action, id)
        obs = self._extract_and_normalize_continuous_features(step_results[0])
        return (obs, *step_results[1:])

    def step(
        self,
        action: Union[np.ndarray, torch.Tensor],
        id: Union[int, list[int], np.ndarray, None] = None,
    ) -> gym_new_venv_step_type:
        # Normalize observation
        step_results = self._step(action, id)
        # Normalize reward
        if self.reward_rms and self.update_reward_rms:
            self.reward_rms.update(step_results[1])
        return (step_results[0], self._norm_reward(step_results[1]), *step_results[2:])
    
    def _norm_reward(self, reward: float) -> np.ndarray:
        """Normalize the reward."""
        if self.reward_rms:
            return self.reward_rms.norm(reward)
        return reward

    def set_reward_rms(self, reward_rms: RunningMeanStd) -> None:
        """Set with given reward running mean/std."""
        self.reward_rms = reward_rms
    
    def get_reward_rms(self) -> RunningMeanStd:
        """Return reward running mean/std."""
        return self.reward_rms


class VectorMultiAgentEnvNormObsReward(VectorEnvNormObsReward):
    """
    Vector environment with normalized observations and rewards for multi-agent environments.
    
    Each agent maintains separate observation and reward running mean/std statistics or uses a reward scale factor.

    Attributes:
        num_agents (int): Number of agents in the environment.
        n_envs (int): Number of environments.
        obs_rms (list[RunningMeanStd]): Running mean/std for each agent's observations.
        reward_rms (list[RunningMeanStd]): Running mean/std for each agent's rewards.
        reward_scale_factor (float | None): Scale factor for rewards. If set, it overrides reward_rms normalization.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        update_reward_rms: bool = True,
        clip_max: float = CLIP_MAX,
        is_heterogeneous: bool = False,
        reward_scale_factor: Union[float, None] = None,
        normalize_obs_output: bool = True
    ) -> None:
        """
        Initialize the vector environment with normalized observations and rewards for multi-agent.

        Args:
            venv (BaseVectorEnv): Vector environment.
            update_obs_rms (bool): Whether to update the observation running mean/std.
            update_reward_rms (bool): Whether to update the reward running mean/std.
            clip_max (float): Maximum absolute value for the data array.
            is_heterogeneous (bool): Whether the data array is heterogeneous.
            reward_scale_factor (float | None): Scale factor for rewards. If set, it overrides reward_rms normalization.
            normalize_obs_output (bool): Whether to normalize the returned observations. When False, still updates
                obs_rms statistics but returns raw observations.
        """
        super().__init__(venv, update_obs_rms, update_reward_rms, clip_max, is_heterogeneous, normalize_obs_output)
        self.num_agents = self.venv.get_env_attr('num_agents')[0]
        self.n_envs = len(self.venv.get_env_attr('workers'))
        self.obs_rms = [deepcopy(self.obs_rms) for _ in range(self.num_agents)]
        self.reward_rms = [deepcopy(self.reward_rms) for _ in range(self.num_agents)]
        self.reward_scale_factor = reward_scale_factor

    def _extract_and_normalize_continuous_features(self, obs: np.ndarray) -> np.ndarray:
        feature_length = obs.shape[-1]
        continuous_mask = self._get_continuous_feature_mask(feature_length)
        normalized_obs = obs.copy()

        # Normalize only continuous features for each agent
        for agent_idx in range(self.num_agents):
            agent_obs = obs[:, agent_idx, :]
            continuous_features = agent_obs[:, continuous_mask] # [n_envs, n_services * n_continuous]
            if self.obs_rms[agent_idx] and self.update_obs_rms:
                self.obs_rms[agent_idx].update(continuous_features)
            if self.normalize_obs_output:
                normalized_continuous = self._norm_obs(agent_idx, continuous_features)
                normalized_obs[:, agent_idx, continuous_mask] = normalized_continuous
        return normalized_obs

    def _norm_obs(self, agent_idx: int, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms[agent_idx]:
            return self.obs_rms[agent_idx].norm(obs)
        return obs

    def _norm_reward(self, reward: np.ndarray) -> np.ndarray:
        norm_reward = np.zeros_like(reward)
        for agent_idx in range(self.num_agents):
            if self.reward_scale_factor:
                norm_reward[:, agent_idx] = reward[:, agent_idx] / self.reward_scale_factor
            elif self.reward_rms[agent_idx]:
                norm_reward[:, agent_idx] = self.reward_rms[agent_idx].norm(reward[:, agent_idx])
            else:
                norm_reward[:, agent_idx] = reward[:, agent_idx]
        return norm_reward

    def step(
        self,
        action: Union[np.ndarray, torch.Tensor],
        id: Union[int, list[int], np.ndarray, None] = None,
    ) -> gym_new_venv_step_type:
        # Normalize observation    
        step_results = self._step(action, id)
        # Normalize reward
        if self.reward_rms and self.update_reward_rms:
            for agent_idx in range(self.num_agents):
                self.reward_rms[agent_idx].update(step_results[1][:, agent_idx])
        return (step_results[0], self._norm_reward(step_results[1]), *step_results[2:])

    def set_obs_rms(self, obs_rms_list: list[RunningMeanStd]) -> None:
        """Set with given observation running mean/std for each agent."""
        if len(obs_rms_list) != self.num_agents:
            raise ValueError(f'Expected {self.num_agents} obs_rms objects, got {len(obs_rms_list)}')
        self.obs_rms = obs_rms_list

    def get_obs_rms(self) -> list[RunningMeanStd]:
        """Return observation running mean/std for each agent."""
        return self.obs_rms

    def set_reward_rms(self, reward_rms_list: list[RunningMeanStd]) -> None:
        """Set with given reward running mean/std for each agent."""
        if len(reward_rms_list) != self.num_agents:
            raise ValueError(f'Expected {self.num_agents} reward_rms objects, got {len(reward_rms_list)}')
        self.reward_rms = reward_rms_list

    def get_reward_rms(self) -> list[RunningMeanStd]:
        """Return reward running mean/std for each agent."""
        return self.reward_rms


class VectorGraphEnvNormObsReward(VectorEnvNormObsReward):
    """
    Vector graph environment with normalized observations and rewards.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        update_reward_rms: bool = True,
        clip_max: float = CLIP_MAX,
        is_heterogeneous: bool = False,
        normalize_obs_output: bool = True
    ) -> None:
        """
        Initialize the graph vector environment with normalized observations and rewards.
        """
        super().__init__(venv, update_obs_rms, update_reward_rms, clip_max, is_heterogeneous, normalize_obs_output)

    def _extract_and_normalize_continuous_features(self, obs: np.ndarray) -> np.ndarray:
        service_features = np.stack([env_obs['services_graph']['x'] for env_obs in obs])
        normalized_features = service_features.copy()
        normalized_obs = obs.copy()

        # Normalize only continuous features
        feature_length = service_features.shape[-1]
        continuous_mask = self._get_continuous_feature_mask(feature_length)
        continuous_features = service_features[:, :, continuous_mask] # [n_envs, n_services, n_continuous]
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(continuous_features)
        if self.normalize_obs_output:
            normalized_continuous = self._norm_obs(continuous_features)
            # Reconstruct full feature array (keeping categorical features unchanged)
            normalized_features[:, :, continuous_mask] = normalized_continuous
            for env_idx, env_obs in enumerate(obs):
                # Copy the dict and its services_graph sub-dict to avoid mutating the original
                new_obs = env_obs.copy()
                new_obs['services_graph'] = env_obs['services_graph'].copy()
                new_obs['services_graph']['x'] = normalized_features[env_idx]
                normalized_obs[env_idx] = new_obs
        return normalized_obs

    @lru_cache(maxsize=None)
    def _get_continuous_feature_mask(self, feature_length: int) -> np.ndarray:
        mask = np.zeros(feature_length, dtype=bool)
        # Profit is at different location in the node features due to _get_node_features method
        mask[NODE_FEATURE_PROFIT_POSITION] = True
        for offset in [CATEGORICAL_PER_MARKET_SEAT, CATEGORICAL_PER_MARKET_SEAT + 1]:
            positions = np.arange(BASIC_FEATURES + offset, feature_length, FEATURES_PER_MARKET_SEAT)
            mask[positions] = True
        return mask


class VectorMultiAgentGraphEnvNormObsReward(VectorMultiAgentEnvNormObsReward, VectorGraphEnvNormObsReward):
    """
    Vector environment with normalized observations and rewards for multi-agent graph environments.

    Each agent maintains separate observation and reward running mean/std statistics for graph-based observations.
    """

    def _extract_and_normalize_continuous_features(self, obs: np.ndarray) -> np.ndarray:
        n_envs = obs.shape[0]
        feature_length = obs[0, 0]['services_graph']['x'].shape[-1]
        continuous_mask = self._get_continuous_feature_mask(feature_length)
        normalized_obs = obs.copy()

        # Normalize only continuous features for each agent
        for agent_idx in range(self.num_agents):
            agent_obs = np.stack([obs[env_idx, agent_idx]['services_graph']['x'] for env_idx in range(n_envs)])
            continuous_features = agent_obs[:, :, continuous_mask]  # [n_envs, n_services, n_continuous]
            if self.obs_rms[agent_idx] and self.update_obs_rms:
                self.obs_rms[agent_idx].update(continuous_features)
            if self.normalize_obs_output:
                normalized_continuous = self._norm_obs(agent_idx, continuous_features)
                # Reconstruct full feature array (keeping categorical features unchanged)
                agent_obs[:, :, continuous_mask] = normalized_continuous
                for env_idx, normalized_services in enumerate(agent_obs):
                    # Copy the dict and its services_graph sub-dict to avoid mutating the original
                    new_obs = obs[env_idx, agent_idx].copy()
                    new_obs['services_graph'] = obs[env_idx, agent_idx]['services_graph'].copy()
                    new_obs['services_graph']['x'] = normalized_services
                    normalized_obs[env_idx, agent_idx] = new_obs
        return normalized_obs


class VectorDummyEnvEmbeddingWrapper:
    """A dummy environment wrapper that provides a vectorized observation space for embedding."""
    
    def __init__(self, venv: BaseVectorEnv, embedding_dim: int, num_agents: int) -> None:
        self.venv = venv
        self.n_envs = len(self.venv.get_env_attr('workers'))
        self.embedding_dim = embedding_dim
        self.num_agents = num_agents
    
    @property
    def action_space(self) -> spaces.Space:
        return self.venv.action_space

    @property
    def observation_space(self) -> spaces.Space:
        agent_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32
        )
        obs_space = spaces.Tuple([agent_obs_space for _ in range(self.num_agents)])
        return [obs_space for _ in range(self.n_envs)]

    def get_env_attr(self, attr: str) -> Any:
        return self.venv.get_env_attr(attr)


class Stats:
    """
    Stats class to log data from the environment using a SummaryWriter.
    
    Attributes:
        agents (list[str]): Agents in the environment.
        num_agents (int): Number of agents in the environment.
        is_multiagent (bool): Whether the environment is multiagent.
        episode_length (int): Length of the episode.
        logger (SummaryWriter): A TensorboardX SummaryWriter instance for logging.
    """
    
    def __init__(self, agents: list[str], num_agents: int, episode_length: int, log_dir: str = LOG_DIR) -> None:
        """
        Initializes the Stats object to log data using SummaryWriter.
        
        Args:
            agents (list[str]): List of agents in the environment.
            num_agents (int): Number of agents in the environment.
            episode_length (int): Length of the episode.
            log_dir (str): The directory to save the logs.
        """
        self.agents = agents
        self.num_agents = num_agents
        self.is_multiagent = num_agents > 1
        self.episode_length = episode_length
        self.logger = SummaryWriter(log_dir)
        
    def log_agents_to_tensorboard(self, stats: list[dict], returns: np.ndarray[float], ep_i: int) -> None:
        """
        Log the agents data to Tensorboard for a specific episode.
        
        Args:
            stats (list[dict]): List of dictionaries containing stats at the end of an episode.
            returns (np.ndarray[float]): Returns of the environments.
            ep_i (int): The episode index.
        """
        # Log mean returns
        if self.is_multiagent:
            mean_returns = np.mean(returns, axis=0)
            self._log_agent_metric_to_tensorboard(dict(zip(self.agents, mean_returns)), 'mean_return', ep_i)
        
        # Log mean profit
        mean_profits = self._calculate_mean_agent_metric([info['agents']['profit'] for info in stats])
        self._log_agent_metric_to_tensorboard(mean_profits, 'mean_profit', ep_i)

        # Calculate mean profits
        profits = np.array([list(info['agents']['profit'].values()) for info in stats], dtype=np.float32)
        mean_profits = np.mean(profits, axis=0)

        # Log efficiency
        mean_efficiency = self._calculate_mean_efficiency(mean_profits)
        self.logger.add_scalar('agents/mean_efficiency', mean_efficiency, ep_i)

        # Log equality
        num_agents = len(mean_profits) # Update to the real number of agents (single-agent case)
        mean_equality = self._calculate_mean_equality(mean_profits, num_agents)
        self.logger.add_scalar('agents/mean_equality', mean_equality, ep_i)

    def log_services_to_tensorboard(self, stats: list[dict], ep_i: int) -> None:
        """
        Logs the services data to Tensorboard for a specific episode.
        
        Args:
            stats (list[dict]): List of dictionaries containing stats at the end of an episode.
            ep_i (int): The episode index.
        """
        # Log mean total profit
        mean_total_profit = np.mean([info['services']['total_profit'] for info in stats])
        self.logger.add_scalar('services/mean_total_profit', mean_total_profit, ep_i)

        # Log mean prices and tickets sold for each service, market, and seat
        mean_prices = self._calculate_mean_service_metric([info['services']['prices'] for info in stats])
        self._log_service_metric_to_tensorboard(mean_prices, 'mean_last_prices', ep_i)
        mean_tickets_sold = self._calculate_mean_service_metric([info['services']['tickets_sold'] for info in stats])
        self._log_service_metric_to_tensorboard(mean_tickets_sold, 'mean_tickets_sold', ep_i)
        
    def log_passengers_to_tensorboard(self, stats: list[dict], ep_i: int) -> None:
        """
        Logs the passengers data to Tensorboard for a specific episode.
        
        Args:
            stats (list[dict]): List of dictionaries containing stats at the end of an episode.
            ep_i (int): The episode index.
        """
        # Log mean total passengers
        mean_total_passengers = np.mean([info['passengers']['total'] for info in stats])
        self.logger.add_scalar('passengers/mean_total_passengers', mean_total_passengers, ep_i)
        
        # Log mean total passengers travelling
        mean_passengers_travelling = np.mean([info['passengers']['travelling'] for info in stats])
        self.logger.add_scalar('passengers/mean_passengers_travelling', mean_passengers_travelling, ep_i)
        
        # Log mean total passengers not travelling
        mean_passengers_not_travelling = np.mean([info['passengers']['not_travelling'] for info in stats])
        self.logger.add_scalar('passengers/mean_passengers_not_travelling', mean_passengers_not_travelling, ep_i)
        
        # Log mean total percentage of passengers travelling
        mean_percentage_travelling = np.mean([info['passengers']['percentage_travelling'] for info in stats])
        self.logger.add_scalar('passengers/mean_percentage_travelling', mean_percentage_travelling, ep_i)
        
        # Log mean utility
        mean_utility = np.mean([info['passengers']['utility'] for info in stats])
        self.logger.add_scalar('passengers/mean_utility', mean_utility, ep_i)

        # Log mean number of transfers
        mean_n_transfers = self._calculate_mean_passenger_metric([info['passengers']['n_transfers'] for info in stats])
        self._log_passenger_metric_to_tensorboard(mean_n_transfers, 'mean_n_transfers', ep_i)

        # Log mean user patterns travelling
        mean_user_pattern_travelling = self._calculate_mean_passenger_metric([info['passengers']['user_patterns']['travelling'] for info in stats])
        self._log_passenger_metric_to_tensorboard(mean_user_pattern_travelling, 'mean_user_pattern_travelling', ep_i)

    def to_tensorboard(self, stats: list[dict], returns: np.ndarray[float], ep_i: int) -> None:
        """
        Logs the stats to Tensorboard for a specific episode.
        
        Args:
            stats (list[dict]): List of dictionaries containing stats at the end of an episode.
            returns (np.ndarray[float]): Returns of the agents.
            ep_i (int): The episode index.
        """
        self.log_agents_to_tensorboard(stats, returns, ep_i)
        self.log_services_to_tensorboard(stats, ep_i)
        self.log_passengers_to_tensorboard(stats, ep_i)

    def _calculate_mean_agent_metric(self, agent_metric_list: list[dict[str, float]]) -> dict[str, float]:
        """
        Calculates the mean metric for each agent.
        
        Args:
            agent_metric_list (list[dict[str, float]]): List of dictionaries containing the metric for each agent.
        """
        aggregated_metric = {}

        # Aggregate metric for each agent
        for env in agent_metric_list:
            for agent, value in env.items():
                if agent not in aggregated_metric:
                    aggregated_metric[agent] = []
                aggregated_metric[agent].append(value)

        # Calculate the mean for each agent
        for agent, values in aggregated_metric.items():
            aggregated_metric[agent] = np.mean(values)

        return aggregated_metric

    def _calculate_mean_efficiency(self, profits: np.ndarray[float]) -> float:
        """
        Calculates the mean efficiency of the agents.

        Args:
            profits (np.ndarray[float]): Profits of the agents.

        Returns:
            float: Mean efficiency of the agents.
        """
        return np.sum(profits) / self.episode_length

    def _calculate_mean_equality(self, profits: np.ndarray[float], num_agents: int) -> float:
        """
        Calculates the mean equality of the agents.

        Args:
            profits (np.ndarray[float]): Profits of the agents.
            num_agents (int): Number of agents.

        Returns:
            float: Mean equality of the agents.
        """
        pairwise_differences = np.sum(np.abs(profits[:, np.newaxis] - profits))
        normalization_factor = 2 * num_agents * np.sum(profits)
        return 1 - pairwise_differences / normalization_factor

    def _calculate_mean_passenger_metric(self, passenger_metric_list: list[dict[str, float]]) -> dict[str, float]:
        """
        Calculates the mean metric for each passenger.
        
        Args:
            passenger_metric_list (list[dict[str, float]]): List of dictionaries containing the metric for each passenger.
        """
        aggregated_metric = {}

        # Aggregate metric for each passenger
        for env in passenger_metric_list:
            for passenger, value in env.items():
                if passenger not in aggregated_metric:
                    aggregated_metric[passenger] = []
                aggregated_metric[passenger].append(value)

        # Calculate the mean for each passenger
        for passenger, values in aggregated_metric.items():
            aggregated_metric[passenger] = np.mean(values)

        return aggregated_metric

    def _calculate_mean_service_metric(self, service_metric_list: list[dict[str, dict[str, dict[str, float]]]]) -> dict[str, dict[str, dict[str, float]]]:
        """
        Calculates the mean service metric for each service.
        
        Args:
            service_metric_list (list[dict[str, dict[str, dict[str, float]]]): List of dictionaries containing the
                service metric for each service, market and seat.
        """
        aggregated_metric = {}

        # Aggregate metric for each service, market, and seat
        for env in service_metric_list:
            for service, markets in env.items():
                if service not in aggregated_metric:
                    aggregated_metric[service] = {}
                for market, seats in markets.items():
                    if market not in aggregated_metric[service]:
                        aggregated_metric[service][market] = {}
                    for seat, value in seats.items():
                        if seat not in aggregated_metric[service][market]:
                            aggregated_metric[service][market][seat] = []
                        aggregated_metric[service][market][seat].append(value)

        # Calculate the mean for each seat
        for service, markets in aggregated_metric.items():
            for market, seats in markets.items():
                for seat, values in seats.items():
                    aggregated_metric[service][market][seat] = np.mean(values)

        return aggregated_metric
    
    def _log_agent_metric_to_tensorboard(self, metric: dict[str, float], metric_name: str, ep_i: int):
        """
        Logs the calculated mean metric for each agent to Tensorboard.
        
        Args:
            metric (dict[str, float]): Dictionary containing the calculated mean metric for each agent.
            metric_name (str): The name of the metric to log.
            ep_i (int): The episode index.
        """
        for agent, value in metric.items():
            self.logger.add_scalar(f'agents/{metric_name}/{agent}', value, ep_i)

    def _log_passenger_metric_to_tensorboard(self, metric: dict[str, float], metric_name: str, ep_i: int):
        """
        Logs the calculated mean metric for each passenger to Tensorboard.
        
        Args:
            metric (dict[str, float]): Dictionary containing the calculated mean metric for each passenger.
            metric_name (str): The name of the metric to log.
            ep_i (int): The episode index.
        """
        for passenger, value in metric.items():
            self.logger.add_scalar(f'passengers/{metric_name}/{passenger}', value, ep_i)

    def _log_service_metric_to_tensorboard(self, metric: dict[str, dict[str, dict[str, float]]], metric_name: str, ep_i: int):
        """
        Logs the calculated mean tickets sold for each service, market, and seat to Tensorboard.
        
        Args:
            metric (dict[str, dict[str, dict[str, float]]]): Dictionary containing the calculated mean metric for each service, market, and seat.
            metric_name (str): The name of the metric to log.
            ep_i (int): The episode index.
        """
        for service, markets in metric.items():
            for market, seats in markets.items():
                for seat, value in seats.items():
                    self.logger.add_scalar(f'services/{metric_name}/{service}/{market}/{seat}', value, ep_i)


class StatsSubprocVectorEnv(SubprocVectorEnv):
    """
    Subprocess vectorized environment with stats logging.
    
    Attributes:
        stats (Stats): The Stats object to log data.
        episode_index (int): The episode index.
        n_envs (int): Number of environments.
        num_agents (int): Number of agents in the environments.
        is_multiagent (bool): Whether the environment is multiagent.
        episode_length (int): Length of the episode.
        returns (np.array[float]): Returns of the environments.
    """
    
    def __init__(self, log_dir: str = LOG_DIR, *args, **kwargs) -> None:
        """
        Initializes the StatsSubprocVectorEnv object with stats logging.
        
        Args:
            log_dir (str): The directory to save the logs.
        """
        super().__init__(*args, **kwargs)
        self.episode_index = 0
        self.n_envs = len(self.workers)
        self.agents = self.get_env_attr('agents')[0]
        self.num_agents = self.get_env_attr('num_agents')[0]
        self.is_multiagent = self.num_agents > 1
        self.episode_length = len(self.get_env_attr('simulation_days')[0])
        self.returns = np.zeros((self.n_envs, self.num_agents), dtype=np.float32)
        self.stats = Stats(self.agents, self.num_agents, self.episode_length, log_dir)
    
    def step(self, action: list, *args, **kwargs) -> Tuple[list, float, bool, bool, dict]:
        """
        Perform an action in the environment and log the stats.

        Args:
            action (list): Action to perform.
        
        Returns:
            Tuple[list, float, bool, bool, dict]: Observation, reward, termination, truncation and info of the environment.
        """
        obs, reward, terminated, truncated, info = super().step(action, *args, **kwargs)
        if self.is_multiagent:
            self.returns += reward
        if terminated.all():
            self.stats.to_tensorboard(info, self.returns, self.episode_index)
            self.returns = np.zeros((self.n_envs, self.num_agents), dtype=np.float32)
            self.episode_index += self.n_envs
        return obs, reward, terminated, truncated, info


class BaseRobinEnv(ABC):
    """
    Abstract class for the Robin simulator environment.

    Attributes:
        path_config_supply (Path): Path to the supply configuration file.
        path_config_demand (Path): Path to the demand configuration file.
        departure_time_hard_restriction (bool): Whether to apply a hard restriction to the departure time.
        kernel (Kernel): Kernel of the Robin simulator.
        action_factor (int): Factor to multiply the price action.
    """

    def __init__(
        self,
        path_config_supply: Path,
        path_config_demand: Path,
        departure_time_hard_restriction: bool = False,
        discrete_action_space: bool = False,
        action_factor: int = ACTION_FACTOR,
        seed: Union[int, None] = None
    ) -> None:
        """
        Initialize the environment.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            departure_time_hard_restriction (bool): Whether to apply a hard restriction to the departure time.
            discrete_action_space (bool): Whether the action space is discrete or continuous.
            action_factor (int): Factor to multiply the price action (discrete). If the action space is continuous
                it is adjusted by multiplying the number of actions and dividing by half.
            seed (int, None): Seed for the random number generator.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.departure_time_hard_restriction = departure_time_hard_restriction
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)
        self.idx_ids = self._get_element_idx_from_id()
        self.discrete_action_space = discrete_action_space
        self.action_factor = action_factor if discrete_action_space else action_factor * ((NUMBER_ACTIONS - 1) / 2)

    @property
    def simulation_days(self):
        return self.kernel.simulation_days

    def _get_element_idx_from_id(self) -> dict[str, dict[str, int]]:
        """
        Get the index of all elements needed in the environment by their id.
        
        Returns:
            dict[str, dict[str, int]]: Dictionary with the index of each element by its id.
        """
        idx_ids = {
            'tsps': {tsp.id: idx for idx, tsp in enumerate(self.kernel.supply.tsps)},
            'lines': {line.id: idx for idx, line in enumerate(self.kernel.supply.lines)},
            'corridors': {corridor.id: idx for idx, corridor in enumerate(self.kernel.supply.corridors)},
            'time_slots': {time_slot.id: idx for idx, time_slot in enumerate(self.kernel.supply.time_slots)},
            'rolling_stocks': {rolling_stock.id: idx for idx, rolling_stock in enumerate(self.kernel.supply.rolling_stocks)},
            'stations': {station.id: idx for idx, station in enumerate(self.kernel.supply.stations)},
            'seats': {seat.id: idx for idx, seat in enumerate(self.kernel.supply.seats)},
        }
        return idx_ids

    def _get_obs(self, supply: Supply) -> list:
        """
        Get the observation of the environment.

        Args:
            supply (Supply): Supply of the environment.

        Returns:
            list: Observation of the environment.
        """
        obs = [
            {
                'tsp': self.idx_ids['tsps'][service.tsp.id],
                'line': self.idx_ids['lines'][service.line.id],
                'corridor': self.idx_ids['corridors'][service.line.corridor.id],
                'time_slot': self.idx_ids['time_slots'][service.time_slot.id],
                'rolling_stock': self.idx_ids['rolling_stocks'][service.rolling_stock.id],
                'profit': service.total_profit,
                'market_seats': [{
                    'origin': self.idx_ids['stations'][origin],
                    'destination': self.idx_ids['stations'][destination],
                    'seat_type': self.idx_ids['seats'][seat.id],
                    'price': price,
                    'tickets_sold': service.tickets_sold_pair_seats[(origin, destination)][seat]
                } for (origin, destination), seats in service.prices.items() for seat, price in seats.items()]
            }
            for service in supply.services
        ]
        return obs

    def _get_info(self) -> dict:
        """
        Get the info of the environment.

        Returns:
            dict: Info of the environment.
        """
        profit = [service.total_profit for service in self.kernel.supply.services]
        agents_profit = {tsp.name: sum(service.total_profit for service in self.kernel.supply.services if service.tsp.id == tsp.id) for tsp in self.kernel.supply.tsps}
        total_passengers = len(self.kernel.passengers)
        traveling_passengers = len([passenger for passenger in self.kernel.passengers if passenger.journey])
        info = {
            'agents': {
                'profit': agents_profit
            },
            'services': {
                'total_profit': sum(profit),
                'profit': profit,
                'prices': {
                    service.id: {
                        '_'.join(market): {
                            seat.name: price for seat, price in seats.items()
                        } for market, seats in service.prices.items()
                    } for service in self.kernel.supply.services
                },
                'tickets_sold': {
                    service.id: {
                        '_'.join(market): {
                            seat.name: count for seat, count in seats.items()
                        } for market, seats in service.tickets_sold_pair_seats.items()
                    } for service in self.kernel.supply.services
                }
            },
            'passengers': {
                'total': total_passengers,
                'travelling': traveling_passengers,
                'not_travelling': total_passengers - traveling_passengers,
                'percentage_travelling': traveling_passengers / total_passengers * 100,
                'utility': np.mean([passenger.utility for passenger in self.kernel.passengers]),
                'n_transfers': dict(Counter(str(passenger.journey.n_transfers) for passenger in self.kernel.passengers if passenger.journey)),
                'user_patterns': {
                    'travelling': dict(Counter(passenger.user_pattern.name for passenger in self.kernel.passengers if passenger.journey))
                }
            }
        }
        return info

    @abstractmethod
    def _get_reward(self) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Returns:
            float: Reward of the environment.
        """
        # NOTE: Don't forget about if we want to add negative rewards as costs
        # We can also promote the some specific services, such as direct services by adding a weight to the reward
        raise NotImplementedError

    def _get_terminated(self) -> bool:
        """
        Get the termination of the environment.

        The environment is terminated when the simulation is finished.

        Returns:
            bool: Termination of the environment.
        """
        return self.kernel.is_simulation_finished

    def reset(self, seed: Union[int, None] = None, options: dict = None) -> Tuple[list, dict]:
        """
        Reset the environment.

        It only sets the seed and re-creates the kernel. Child classes should implement the rest of the reset.

        Args:
            seed (int, None): Seed for the random number generator.
            options (dict, None): Options for the reset.
        
        Returns:
            Tuple[list, dict]: Observation and info of the environment.
        """
        super().reset(seed=seed)
        # NOTE: If the config files are changed during the simulation, it will load the new ones, beware!
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)

    def _update_prices(self, action: list, supply: Supply) -> None:
        """
        Update the prices of the services in the kernel supply by multiplying the price action by a factor.

        Args:
            action (list): Action to perform.
            supply (Supply): Supply of the environment.
        """
        for service, action_service in zip(supply.services, action):
            for ((origin, destination), seats), price in zip(service.prices.items(), action_service['prices']):
                for seat_type, seat_price in zip(seats, price['seats']):
                    price_modification = seat_price['price'] * (self.action_factor / 100)
                    service.prices[(origin, destination)][seat_type] *= (1 + price_modification)
                    # Clip the price to its range, so, it is not possible to have negative prices
                    service.prices[(origin, destination)][seat_type] = \
                        np.clip(service.prices[(origin, destination)][seat_type], LOW_PRICE, HIGH_PRICE)

    def _step(self, action: list, supply: Union[Supply, list[Supply]]) -> None:
        """
        Private method to perform an action in the environment.

        Args:
            action (list): Action to perform.
            supply (Union[Supply, list[Supply]]): Supply or supplies of the environment.
        """
        if isinstance(supply, list):
            for supply_action, supply_ in zip(action, supply):
                self._update_prices(action=supply_action, supply=supply_)
        else:
            self._update_prices(action=action, supply=supply)
        self.kernel.simulate_a_day(departure_time_hard_restriction=self.departure_time_hard_restriction)

    @abstractmethod
    def step(self, action: list) -> Tuple[list, float, bool, bool, dict]:
        """
        Perform an action in the environment.

        Args:
            action (list): Action to perform.
        
        Returns:
            Tuple[list, float, bool, bool, dict]: Observation, reward, termination, truncation and info of the environment.
        """
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        self.kernel.set_seed(seed)

    def observation_space(self, supply: Supply) -> spaces.Space:
        """
        Observation space of the environment.

        Args:
            supply (Supply): Supply of the environment.

        Returns:
            spaces.Space: Observation space of the environment.
        """
        observation_space = spaces.Tuple([
            spaces.Dict({
                # service already departed for an action mask?
                # date time details? day of the week?
                # capacity of the rolling stock?
                'tsp': spaces.Box(low=(idx := self.idx_ids['tsps'][service.tsp.id]), high=idx, shape=(), dtype=np.int32),
                'line': spaces.Box(low=(idx := self.idx_ids['lines'][service.line.id]), high=idx, shape=(), dtype=np.int32),
                'corridor': spaces.Box(low=(idx := self.idx_ids['corridors'][service.line.corridor.id]), high=idx, shape=(), dtype=np.int32),
                'time_slot': spaces.Box(low=(idx := self.idx_ids['time_slots'][service.time_slot.id]), high=idx, shape=(), dtype=np.int32),
                'rolling_stock': spaces.Box(low=(idx := self.idx_ids['rolling_stocks'][service.rolling_stock.id]), high=idx, shape=(), dtype=np.int32),
                'profit': spaces.Box(low=LOW_PRICE, high=HIGH_PRICE, shape=(), dtype=np.float32),
                'market_seats': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Box(low=(idx := self.idx_ids['stations'][origin]), high=idx, shape=(), dtype=np.int32),
                        'destination': spaces.Box(low=(idx := self.idx_ids['stations'][destination]), high=idx, shape=(), dtype=np.int32),
                        'seat_type': spaces.Box(low=(idx := self.idx_ids['seats'][seat.id]), high=idx, shape=(), dtype=np.int32),
                        'price': spaces.Box(low=LOW_PRICE, high=HIGH_PRICE, shape=(), dtype=np.float32),
                        'tickets_sold': spaces.Box(low=0, high=service.rolling_stock.total_capacity, shape=(), dtype=np.int32)
                    }) for (origin, destination), seats in service.prices.items() for seat, _ in seats.items()
                ]),
            }) for service in supply.services
        ])
        return observation_space

    def action_space(self, supply: Supply) -> spaces.Space:
        """
        Action space of the environment.

        Args:
            supply (Supply): Supply of the environment.

        Returns:
            spaces.Space: Action space of the environment.
        """
        action_space = spaces.Tuple([
            spaces.Dict({
                'prices': spaces.Tuple([
                    spaces.Dict({
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'price': spaces.Discrete(n=NUMBER_ACTIONS, start=START_ACTION) if self.discrete_action_space \
                                    else spaces.Box(low=LOW_ACTION, high=HIGH_ACTION, shape=(), dtype=np.float32)
                            }) for _ in seats
                        ])
                    }) for _, seats in service.prices.items()
                ])
            }) for service in supply.services
        ])
        return action_space


class BaseRobinGraphEnv(BaseRobinEnv):
    """
    Abstract class for the Robin simulator environment with graph representation.

    Attributes:
        exclude_edge_type (int, None): Edge type to exclude from the graph representation.
    """
    
    def __init__(self, *args, exclude_edge_type: Union[int, None] = None, **kwargs) -> None:
        """
        Initialize the environment with graph representation.

        Args:
            exclude_edge_type (int, None): Edge type to exclude from the graph representation.
        """
        super().__init__(*args, **kwargs)
        self.exclude_edge_type = exclude_edge_type

    def get_services_graph(self, supply: Supply, agent_idx: int = None) -> Data:
        """
        Build the services graph as a PyTorch Geometric Data object.
        
        Args:
            supply (Supply): Supply of the environment.
            agent_idx (int, optional): Index of the agent to filter services.

        Returns:
            torch_geometric.data.Data: PyTorch Geometric Data object with services graph.
        """
        node_features = self._get_services_node_features(supply, agent_idx)
        edge_index, edge_type = supply.get_services_edges(self.exclude_edge_type)
        num_nodes = len(supply.services)
        return Data(x=node_features, edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)

    def _get_services_node_features(self, supply: Supply, agent_idx: int = None) -> torch.Tensor:
        """
        Get node features for the services graph.
        
        Flattens each service's observation into a feature vector containing:
        - Basic service info: [tsp, line, corridor, time_slot, rolling_stock, profit]
        - For each market-seat combination: [origin, destination, seat_type, price, tickets_sold]
        
        When agent_idx is provided, profit and tickets_sold data is excluded for services not belonging to that agent.
        
        Args:
            supply (Supply): Supply of the environment.
            agent_idx (int, optional): Index of the agent to filter services.
            
        Returns:
            torch.Tensor: Node features tensor with shape (num_services, max_feature_dim).
                Features are padded with zeros to ensure consistent dimensions.
        """
        services = super()._get_obs(supply)
        node_features = []
        max_length = 0

        # Calculate services features and find max length
        all_features = []
        for service in services:
            # Basic service features
            service_features = [
                service['tsp'],
                service['line'],
                service['corridor'],
                service['time_slot'],
                service['rolling_stock']
            ]

            # Only include profit if this is the agent's own service
            profit = service['profit']
            if agent_idx is not None and service['tsp'] != agent_idx:
                profit = 0
            service_features.append(profit)

            # Add market-seat combination features
            for market_seat in service['market_seats']:
                # Only include tickets_sold if this is the agent's own service
                tickets_sold = market_seat['tickets_sold']
                if agent_idx is not None and service['tsp'] != agent_idx:
                    tickets_sold = 0
                market_features = [
                    market_seat['origin'],
                    market_seat['destination'],
                    market_seat['seat_type'],
                    market_seat['price'],
                    tickets_sold
                ]
                service_features.extend(market_features)

            # Append service features to the list
            all_features.append(service_features)
            max_length = max(max_length, len(service_features))
        
        # Pad with zeros to reach max_length
        for features in all_features:
            padded_features = features + [0] * (max_length - len(features))
            node_features.append(padded_features)
        return torch.tensor(node_features, dtype=torch.float32)

    def _get_obs(self, supply: Supply) -> dict:
        """
        Get the observation of the environment with graph representation.

        Args:
            supply (Supply): Supply of the environment.

        Returns:
            dict: Observation of the environment with graph representation.
        """
        services_graph = self.get_services_graph(supply)
        obs = {
            'services_graph': {
                'x': services_graph.x.numpy(),
                'edge_index': services_graph.edge_index.numpy(),
                'edge_type': services_graph.edge_type.numpy()
            }
        }
        return obs

    def observation_space(self, supply: Supply) -> spaces.Space:
        """
        Observation space of the environment with graph representation.

        Args:
            supply (Supply): Supply of the environment.

        Returns:
            spaces.Space: Observation space of the environment.
        """
        services_graph = self.get_services_graph(supply)
        observation_space = spaces.Dict({
            'services_graph': spaces.Dict({
                'x': spaces.Box(low=-np.inf, high=np.inf, shape=services_graph.x.shape, dtype=np.float32),
                'edge_index': spaces.Box(low=0, high=services_graph.num_nodes - 1, shape=(2, services_graph.num_edges), dtype=np.int32),
                'edge_type': spaces.Box(low=0, high=services_graph.edge_type.max().item(), shape=(services_graph.num_edges,), dtype=np.int32)
            })
        })
        return observation_space


class RobinSingleAgentEnv(BaseRobinEnv, Env):
    """
    Reinforcement learning single-agent environment for the Robin simulator.
    
    Attributes:
        agents (list[str]): Agents in the environment.
        num_agents (int): Number of agents in the environment.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the single-agent environment.
        """
        super().__init__(*args, **kwargs)
        self.agents = [tsp.name for tsp in self.kernel.supply.tsps]
        self.num_agents = 1
        self._last_total_profit = 0

    def _get_reward(self) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Returns:
            float: Reward of the environment.
        """
        total_profit = sum(service.total_profit for service in self.kernel.supply.services)
        reward = total_profit - self._last_total_profit
        self._last_total_profit = total_profit
        return reward

    def step(self, action: list) -> Tuple[list, float, bool, bool, dict]:
        """
        Perform an action in the environment.

        Args:
            action (list): Action to perform.

        Returns:
            Tuple[list, float, bool, bool, dict]: Observation, reward, termination, truncation and info of the environment.
        """
        self._step(action=action, supply=self.kernel.supply)
        obs = self._get_obs(supply=self.kernel.supply)
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Union[int, None] = None, options: dict = None) -> Tuple[list, dict]:
        """
        Reset the environment.

        Args:
            seed (int, None): Seed for the random number generator.
            options (dict, None): Options for the reset.

        Returns:
            Tuple[list, dict]: Observation and info of the environment.
        """
        super().reset(seed=seed, options=options)
        self._last_total_profit = 0
        obs = self._get_obs(supply=self.kernel.supply)
        info = self._get_info()
        return obs, info

    @property
    def observation_space(self) -> spaces.Space:
        """
        Observation space of the environment.

        Returns:
            spaces.Space: Observation space of the environment.
        """
        return super().observation_space(supply=self.kernel.supply)

    @property
    def action_space(self) -> spaces.Space:
        """
        Action space of the environment.

        Returns:
            spaces.Space: Action space of the environment.
        """
        return super().action_space(supply=self.kernel.supply)


class RobinSingleAgentGraphEnv(RobinSingleAgentEnv, BaseRobinGraphEnv):
    """
    Graph-based single-agent environment for the Robin simulator.

    Attributes:
        services_graph (torch_geometric.data.Data): Services graph of the environment.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the graph-based single-agent environment.
        """
        super().__init__(*args, **kwargs)
    
    @property
    def services_graph(self) -> Data:
        """
        Get the services graph of the environment.

        Returns:
            torch_geometric.data.Data: Services graph of the environment.
        """
        return self.get_services_graph(self.kernel.supply)


class RobinMultiAgentEnv(BaseRobinEnv, Env):
    """
    Reinforcement learning multi-agent environment for the Robin simulator.

    Attributes:
        agents (list[str]): Agents in the environment.
        num_agents (int): Number of agents in the environment.
        supplies (list[Supply]): Supplies of the agents.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the multi-agent environment.
        """
        super().__init__(*args, **kwargs)
        self.agents = [tsp.name for tsp in self.kernel.supply.tsps]
        self.num_agents = len(self.agents)
        self.supplies = [self.kernel.filter_supply_by_tsp(tsp.id) for tsp in self.kernel.supply.tsps]
        self._last_total_profit = [0 for _ in self.agents]

    def _get_obs(self) -> list:
        """
        Get the observation of the environment.

        Returns:
            list: Observation of the environment.
        """
        observation = []
        full_observation = super()._get_obs(supply=self.kernel.supply)
        for agent in self.agents:
            obs = deepcopy(full_observation)
            for service in obs:
                if service['tsp'] != self.agents.index(agent):
                    service['profit'] = 0
                    for market_seat in service['market_seats']:
                        market_seat['tickets_sold'] = 0
            observation.append(obs)
        return np.array(observation, dtype=object)

    def _get_reward(self, agent_idx: int, supply: Supply) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Args:
            agent_idx (int): Index of an agent.
            supply (Supply): Supply of an agent.
        
        Returns:
            float: Reward of the environment.
        """
        total_profit = sum(service.total_profit for service in supply.services)
        reward = total_profit - self._last_total_profit[agent_idx]
        self._last_total_profit[agent_idx] = total_profit
        return reward

    def step(self, action: list) -> Tuple[NDArray, NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], dict]:
        """
        Perform an action in the environment.

        Args:
            action (list): Action to perform.

        Returns:
            Tuple[list, float, bool, bool, dict]: Observation, reward, termination, truncation and info of the environment.
        """
        self._step(action=action, supply=self.supplies)
        obs = self._get_obs()
        reward = np.array([self._get_reward(agent_idx, supply) for agent_idx, supply in enumerate(self.supplies)])
        _terminated = self._get_terminated() # The terminated condition is the same for all agents
        terminated = np.array([_terminated for _ in self.supplies])
        truncated = np.array([False for _ in self.supplies])
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Union[int, None] = None, options: dict = None) -> Tuple[dict, dict]:
        """
        Reset the environment.

        Args:
            seed (int, None): Seed for the random number generator.
            options (dict, None): Options for the reset.

        Returns:
            Tuple[dict, dict]: Observation and info of the environment.
        """
        super().reset(seed=seed, options=options)
        # It is necessary to re-create the supplies with the new services references as the Kernel object is re-created
        self.supplies = [self.kernel.filter_supply_by_tsp(tsp.id) for tsp in self.kernel.supply.tsps]
        self._last_total_profit = [0 for _ in self.agents]
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    @property
    def observation_space(self) -> spaces.Space:
        """
        Observation space of the environment.

        All agents share the same observation space structure. For non-owned services,
        tickets_sold values are set to 0 in the observation data.

        Returns:
            spaces.Space: Observation space of each agent in the environment.
        """
        base_observation_space = super().observation_space(supply=self.kernel.supply)
        return spaces.Tuple([base_observation_space for _ in range(self.num_agents)])

    @property
    def action_space(self) -> spaces.Space:
        """
        Action space of the environment.

        Returns:
            spaces.Space: Action space of each agent in the environment.
        """
        # List comprehension can't be used with super() method
        action_spaces = []
        for supply in self.supplies:
            action_spaces.append(super().action_space(supply=supply))
        return spaces.Tuple(action_spaces)


class RobinMultiAgentGraphEnv(RobinMultiAgentEnv, BaseRobinGraphEnv):
    """
    Graph-based multi-agent environment for the Robin simulator.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the graph-based multi-agent environment.
        """
        super().__init__(*args, **kwargs)
    
    @property
    def services_graph(self) -> list[Data]:
        """
        Get the services graph of the environment for each agent.

        Returns:
            list[torch_geometric.data.Data]: List of services graphs for each agent.
        """
        return [self.get_services_graph(self.kernel.supply, agent_idx) for agent_idx in range(self.num_agents)]

    @property
    def observation_space(self) -> spaces.Space:
        """
        Observation space of the environment for each agent.

        Returns:
            spaces.Space: Tuple of observation spaces for each agent.
        """
        base_observation_space = BaseRobinGraphEnv.observation_space(self, self.kernel.supply)
        return spaces.Tuple([base_observation_space for _ in range(self.num_agents)])

    def _get_obs(self) -> list:
        """
        Get the observation of the environment with graph representation for each agent.

        Returns:
            list: List of observations for each agent in the environment.
        """
        observations = []
        for agent_idx in range(self.num_agents):
            services_graph = self.get_services_graph(self.kernel.supply, agent_idx=agent_idx)
            obs = {
                'services_graph': {
                    'x': services_graph.x.numpy(),
                    'edge_index': services_graph.edge_index.numpy(),
                    'edge_type': services_graph.edge_type.numpy()
                }
            }
            observations.append(obs)
        return np.array(observations, dtype=object)


class RobinMultiAgentCoopEnv(RobinMultiAgentEnv):
    """
    Cooperative multi-agent environment for the Robin simulator.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the multi-agent cooperative environment.
        """
        super().__init__(*args, **kwargs)
        
    def _get_reward(self, agent_idx: int, supply: Supply) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Args:
            agent_idx (int): Index of an agent.
            supply (Supply): Supply of an agent.
        
        Returns:
            float: Reward of the environment.
        """
        total_profit = sum(service.total_profit for service in self.kernel.supply.services)
        reward = total_profit - self._last_total_profit[agent_idx]
        self._last_total_profit[agent_idx] = total_profit
        return reward


class RobinMultiAgentGraphCoopEnv(RobinMultiAgentGraphEnv, RobinMultiAgentCoopEnv):
    """
    Graph-based cooperative multi-agent environment for the Robin simulator.
    """

    def _get_reward(self, agent_idx: int, supply: Supply) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Args:
            agent_idx (int): Index of an agent.
            supply (Supply): Supply of an agent.
        
        Returns:
            float: Reward of the environment.
        """
        return RobinMultiAgentCoopEnv._get_reward(self, agent_idx, supply)


class RobinEnvFactory:

    @staticmethod
    def create(
        path_config_supply: Path,
        path_config_demand: Path,
        multi_agent: bool = False,
        cooperative: bool = False,
        graph: bool = False,
        use_pyg_wrapper: bool = False,
        departure_time_hard_restriction: bool = False,
        discrete_action_space: bool = False,
        action_factor: int = ACTION_FACTOR,
        exclude_edge_type: Union[int, None] = None,
        seed: Union[int, None] = None
    ) -> BaseRobinEnv:
        """
        Create a Robin environment.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            multi_agent (bool, optional): Whether to create a multi-agent environment.
            cooperative (bool, optional): Whether to create a cooperative multi-agent environment.
            graph (bool, optional): Whether to create a graph-based environment.
            use_pyg_wrapper (bool, optional): Whether to apply PyGObservationWrapper for graph environments.
            departure_time_hard_restriction (bool, optional): Whether to apply a hard restriction to the departure time.
            discrete_action_space (bool, optional): Whether the action space is discrete or continuous.
            action_factor (int, optional): Factor to multiply the price action.
            exclude_edge_type (int, optional): Edge type to exclude from the graph representation.
            seed (int, optional): Seed for the random number generator.

        Returns:
            RobinEnv: Robin environment.
        """
        if multi_agent:
            if graph:
                if cooperative:
                    env = RobinMultiAgentGraphCoopEnv(
                        path_config_supply=path_config_supply,
                        path_config_demand=path_config_demand,
                        departure_time_hard_restriction=departure_time_hard_restriction,
                        discrete_action_space=discrete_action_space,
                        action_factor=action_factor,
                        exclude_edge_type=exclude_edge_type,
                        seed=seed
                    )
                else:
                    env = RobinMultiAgentGraphEnv(
                        path_config_supply=path_config_supply,
                        path_config_demand=path_config_demand,
                        departure_time_hard_restriction=departure_time_hard_restriction,
                        discrete_action_space=discrete_action_space,
                        action_factor=action_factor,
                        exclude_edge_type=exclude_edge_type,
                        seed=seed
                    )
            elif cooperative:
                env = RobinMultiAgentCoopEnv(
                    path_config_supply=path_config_supply,
                    path_config_demand=path_config_demand,
                    departure_time_hard_restriction=departure_time_hard_restriction,
                    discrete_action_space=discrete_action_space,
                    action_factor=action_factor,
                    seed=seed
                )
            else:
                env = RobinMultiAgentEnv(
                    path_config_supply=path_config_supply,
                    path_config_demand=path_config_demand,
                    departure_time_hard_restriction=departure_time_hard_restriction,
                    discrete_action_space=discrete_action_space,
                    action_factor=action_factor,
                    seed=seed
                )
            if not graph:
                env = FlattenMultiObservation(env)
            env = FlattenMultiAction(env)
        else:
            if graph:
                env = RobinSingleAgentGraphEnv(
                    path_config_supply=path_config_supply,
                    path_config_demand=path_config_demand,
                    departure_time_hard_restriction=departure_time_hard_restriction,
                    discrete_action_space=discrete_action_space,
                    action_factor=action_factor,
                    exclude_edge_type=exclude_edge_type,
                    seed=seed
                )
                if use_pyg_wrapper:
                    env = PyGObservationWrapper(env)
                env = FlattenAction(env)
            else:
                env = RobinSingleAgentEnv(
                    path_config_supply=path_config_supply,
                    path_config_demand=path_config_demand,
                    departure_time_hard_restriction=departure_time_hard_restriction,
                    discrete_action_space=discrete_action_space,
                    action_factor=action_factor,
                    seed=seed
                )
                env = FlattenObservation(env)
                env = FlattenAction(env)
        return env
