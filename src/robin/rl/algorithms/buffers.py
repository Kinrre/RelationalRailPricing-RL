"""Buffers for storing and sampling data for RL algorithms."""

import numpy as np

from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer:
    """
    Replay Buffer for multi-agent RL with parallel rollouts.
    
    Attributes:
        max_steps (int): Maximum number of timepoints to store in buffer.
        num_agents (int): Number of agents in environment.
        obs_buffs (list[np.ndarray]): List of observation buffers for each agent.
        ac_buffs (list[np.ndarray]): List of action buffers for each agent.
        rew_buffs (list[np.ndarray]): List of reward buffers for each agent.
        next_obs_buffs (list[np.ndarray]): List of next observation buffers for each agent.
        done_buffs (list[np.ndarray]): List of done buffers for each agent.
        filled_i (int): Index of first empty location in buffer (last index when full).
        curr_i (int): Current index to write to (ovewrite oldest data).
    """
    
    def __init__(self, max_steps: int, num_agents: int, obs_dims: list[int], ac_dims: list[int]) -> None:
        """
        Initialize the Replay Buffer.
        
        Args:
            max_steps (int): Maximum number of timepoints to store in buffer.
            num_agents (int): Number of agents in environment.
            obs_dims (list[int]): List of observation dimensions for each agent.
            ac_dims (list[int]): List of action dimensions for each agent.
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        """
        Return the number of timepoints stored in the buffer.
        """
        return self.filled_i

    def push(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        Push a batch of data to the buffer.
        
        Args:
            observations (np.ndarray): Batch of observations.
            actions (np.ndarray): Batch of actions.
            rewards (np.ndarray): Batch of rewards.
            next_observations (np.ndarray): Batch of next observations.
            dones (np.ndarray): Batch of done flags.
        """
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N: int, to_gpu: bool = False, norm_rews: bool = False) -> tuple:
        """
        Sample a batch of data from the buffer.
        
        Args:
            N (int): Number of samples to take.
            to_gpu (bool): Whether to move data to GPU.
            norm_rews (bool): Whether to normalize rewards.
            
        Returns:
            tuple: Tuple of lists of observations, actions, rewards, next observations, and done flags.
        """
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             np.sqrt(self.rew_buffs[i][:self.filled_i].var() + np.finfo(np.float32).eps.item()))
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N: int) -> list:
        """
        Get the average rewards over the last N timepoints.
        
        Args:
            N (int): Number of timepoints to average over.
        
        Returns:
            list: List of average rewards for each agent.
        """
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]


class GraphReplayBuffer(ReplayBuffer):
    """
    Replay Buffer for graph-based multi-agent RL with parallel rollouts.

    Stores only node features (x) in pre-allocated float32 arrays. Edge topology
    (edge_index, edge_type) is static per scenario and stored once in the GraphProcessingMixin.

    Attributes:
        max_steps (int): Maximum number of timepoints to store in buffer.
        num_agents (int): Number of agents in environment.
        obs_buffs (list[np.ndarray]): List of observation buffers for each agent.
        ac_buffs (list[np.ndarray]): List of action buffers for each agent.
        rew_buffs (list[np.ndarray]): List of reward buffers for each agent.
        next_obs_buffs (list[np.ndarray]): List of next observation buffers for each agent.
        done_buffs (list[np.ndarray]): List of done buffers for each agent.
        filled_i (int): Index of first empty location in buffer (last index when full).
        curr_i (int): Current index to write to (ovewrite oldest data).
    """

    def __init__(self, max_steps: int, num_agents: int, ac_dims: list[int]) -> None:
        """
        Initialize the Graph Replay Buffer.

        Args:
            max_steps (int): Maximum number of timepoints to store in buffer.
            num_agents (int): Number of agents in environment.
            ac_dims (list[int]): List of action dimensions for each agent.
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = None
        self.next_obs_buffs = None
        self.ac_buffs = []
        self.rew_buffs = []
        self.done_buffs = []
        for adim in ac_dims:
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def _init_obs_buffs(self, num_services: int, feature_dim: int) -> None:
        """
        Allocate dense observation arrays once graph dimensions are known.

        Args:
            num_services (int): Number of service nodes in the graph.
            feature_dim (int): Dimension of node features.
        """
        self.obs_buffs = [
            np.zeros((self.max_steps, num_services, feature_dim), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.next_obs_buffs = [
            np.zeros((self.max_steps, num_services, feature_dim), dtype=np.float32)
            for _ in range(self.num_agents)
        ]

    def push(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        Push a batch of data to the graph buffer.

        Args:
            observations (np.ndarray): Batch of graph observations.
            actions (np.ndarray): Batch of actions.
            rewards (np.ndarray): Batch of rewards.
            next_observations (np.ndarray): Batch of next graph observations.
            dones (np.ndarray): Batch of done flags.
        """
        nentries = observations.shape[0]  # handle multiple parallel environments
    
        # Initialize observation buffers on first push when graph dimensions are known
        if self.obs_buffs is None:
            x0 = observations[0, 0]['services_graph']['x']
            self._init_obs_buffs(x0.shape[0], x0.shape[1])

        # Use circular indexing instead of np.roll for memory efficiency
        for i in range(nentries):
            idx = (self.curr_i + i) % self.max_steps
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i][idx] = observations[i, agent_i]['services_graph']['x']
                # actions are already batched by agent, so they are indexed differently
                self.ac_buffs[agent_i][idx] = actions[agent_i][i]
                self.rew_buffs[agent_i][idx] = rewards[i, agent_i]
                self.next_obs_buffs[agent_i][idx] = next_observations[i, agent_i]['services_graph']['x']
                self.done_buffs[agent_i][idx] = dones[i, agent_i]

        self.curr_i = (self.curr_i + nentries) % self.max_steps
        if self.filled_i < self.max_steps:
            self.filled_i = min(self.filled_i + nentries, self.max_steps)

    def sample(self, N: int, to_gpu: bool = False, norm_rews: bool = False) -> tuple:
        """
        Sample a batch of data from the graph buffer.

        Args:
            N (int): Number of samples to take.
            to_gpu (bool): Whether to move data to GPU.
            norm_rews (bool): Whether to normalize rewards.

        Returns:
            tuple: Tuple of lists of observations, actions, rewards, next observations, and done flags.
        """
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             np.sqrt(self.rew_buffs[i][:self.filled_i].var() + np.finfo(np.float32).eps.item()))
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([self.obs_buffs[i][inds] for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [self.next_obs_buffs[i][inds] for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])
