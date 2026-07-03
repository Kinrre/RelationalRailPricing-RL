"""Actors for the RL algorithms."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robin.rl.entities import StatsSubprocVectorEnv
from robin.rl.algorithms.constants import HIDDEN_SIZE, LOG_STD_MAX, LOG_STD_MIN


class StochasticActor(nn.Module):
    """
    Stochastic actor network.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc_mean (nn.Linear): Fully connected layer for the mean of the action.
        fc_logstd (nn.Linear): Fully connected layer for the log standard deviation of the action.
        action_scale (torch.Tensor): The scale of the action.
        action_bias (torch.Tensor): The bias of the action.
    """
    
    def __init__(self, env: StatsSubprocVectorEnv, agent_idx: int) -> None:
        """
        Initialize the Actor network.
        
        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            agent_idx (int): The index of the agent in the environment.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.prod(env.observation_space[0][agent_idx].shape), HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_mean = nn.Linear(HIDDEN_SIZE, np.prod(env.action_space[0][agent_idx].shape))
        self.fc_logstd = nn.Linear(HIDDEN_SIZE, np.prod(env.action_space[0][agent_idx].shape))
        # action rescaling
        self.register_buffer(
            'action_scale', torch.tensor((env.action_space[0][agent_idx].high - env.action_space[0][agent_idx].low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            'action_bias', torch.tensor((env.action_space[0][agent_idx].high + env.action_space[0][agent_idx].low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward pass of the Actor network.
        
        The Actor network outputs the mean and log standard deviation of the action.
        
        Args:
            x (torch.Tensor): The input tensor.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the action, log probability, and mean of the action.
        
        The action is sampled from a normal distribution with the mean and log standard deviation output by the Actor network.
        
        Args:
            x (torch.Tensor): The input tensor.
        """
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DeterministicActor(nn.Module):
    """
    Deterministic actor network.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc_mu (nn.Linear): Fully connected layer for the action output.
        action_scale (torch.Tensor): The scale of the action.
        action_bias (torch.Tensor): The bias of the action.
    """
    
    def __init__(self, env: StatsSubprocVectorEnv, agent_idx: int) -> None:
        """
        Initialize the deterministic Actor network.
        
        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            agent_idx (int): The index of the agent in the environment.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.prod(env.observation_space[0][agent_idx].shape), HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_mu = nn.Linear(HIDDEN_SIZE, np.prod(env.action_space[0][agent_idx].shape))
        # action rescaling
        self.register_buffer(
            'action_scale', torch.tensor((env.action_space[0][agent_idx].high - env.action_space[0][agent_idx].low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            'action_bias', torch.tensor((env.action_space[0][agent_idx].high + env.action_space[0][agent_idx].low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the deterministic Actor network.
        
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The deterministic action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the deterministic action.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The deterministic action.
        """
        return self.forward(x)
