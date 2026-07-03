"""Critics for the RL algorithms."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robin.rl.entities import StatsSubprocVectorEnv
from robin.rl.algorithms.constants import HIDDEN_SIZE


class BaseCritic(nn.Module):
    """
    Base critic network.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the Critic network.

        Args:
            input_dim (int): The dimension of the input (observation + action).
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the Critic network.
        
        Args:
            x (torch.Tensor): The input tensor.
            action (torch.Tensor): The action tensor.
        
        Returns:
            torch.Tensor: Q-value estimate.
        """
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(BaseCritic):
    """
    Critic network.
    """

    def __init__(self, env: StatsSubprocVectorEnv, agent_idx: int) -> None:
        """
        Initialize the Critic network.
        
        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            agent_idx (int): The index of the agent in the environment.
        """
        obs_dim = np.prod(env.observation_space[0][agent_idx].shape)
        action_dim = np.prod(env.action_space[0][agent_idx].shape)
        super().__init__(obs_dim + action_dim)


class CentralizedCritic(BaseCritic):
    """
    Centralized critic network.

    Takes observations and actions from ALL agents as input.
    """

    def __init__(self, env: StatsSubprocVectorEnv) -> None:
        """
        Initialize the centralized Critic network.

        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
        """
        total_obs_dim = sum(np.prod(obs_space.shape) for obs_space in env.observation_space[0])
        total_action_dim = sum(np.prod(action_space.shape) for action_space in env.action_space[0])
        super().__init__(total_obs_dim + total_action_dim)
