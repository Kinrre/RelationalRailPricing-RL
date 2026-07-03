"""MATD3 algorithm implementation."""

import numpy as np
import torch
import torch.optim as optim

from robin.rl.algorithms.constants import DEFAULT_NUM_LAYERS
from robin.rl.algorithms.critics import CentralizedCritic
from robin.rl.algorithms.maddpg import MADDPG, GraphMADDPG
from robin.rl.entities import StatsSubprocVectorEnv


class MATD3(MADDPG):
    """
    Multi-Agent Twin Delayed DDPG.

    Attributes:
        critic2 (list[CentralizedCritic]): Second centralized critic networks.
        critic2_target (list[CentralizedCritic]): Second target centralized critic networks.
        critic2_optimizer (list[torch.optim]): Optimizers for the second critic networks.
        exploration_noise (float): Std of fixed Gaussian noise used for exploration.
        target_noise (float): Std of Gaussian noise added to target actions for smoothing.
        noise_clip (float): Clip range for target policy smoothing noise.
        detach_actor_from_preprocessor (bool): Whether to detach the preprocessor outputs before passing to the actor.
    """

    def __init__(
        self,
        env: StatsSubprocVectorEnv,
        device: torch.device,
        policy_lr: float = 0.001,
        q_lr: float = 0.001,
        exploration_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        detach_actor_from_preprocessor: bool = True
    ) -> None:
        """
        Initialize the Multi-Agent Twin Delayed DDPG algorithm.

        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            device (torch.device): The device to run the algorithm on.
            policy_lr (float): The learning rate of the policy network optimizer.
            q_lr (float): The learning rate of the Q network optimizer.
            exploration_noise (float): Std of fixed Gaussian noise for exploration.
            target_noise (float): Std of Gaussian noise for target policy smoothing.
            noise_clip (float): Clip range for target policy smoothing noise.
            detach_actor_from_preprocessor (bool): Whether to detach the preprocessor outputs before passing to the actor.
        """
        super().__init__(env, device, policy_lr, q_lr, detach_actor_from_preprocessor=detach_actor_from_preprocessor)
        self.critic2 = [CentralizedCritic(env).to(device) for _ in range(self.num_agents)]
        self.critic2_target = [CentralizedCritic(env).to(device) for _ in range(self.num_agents)]
        for agent_idx in range(self.num_agents):
            self.critic2_target[agent_idx].load_state_dict(self.critic2[agent_idx].state_dict())
        self.critic2_optimizer = [optim.Adam(self.critic2[i].parameters(), lr=q_lr) for i in range(self.num_agents)]
        self.exploration_noise = exploration_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.detach_actor_from_preprocessor = detach_actor_from_preprocessor

    def _get_save_dict(self) -> dict:
        """
        Get the state dictionary for saving the model.

        Returns:
            dict: Dictionary containing the model parameters.
        """
        save_dict = super()._get_save_dict()
        save_dict.update({
            'critic2': [c.state_dict() for c in self.critic2],
            'critic2_target': [c.state_dict() for c in self.critic2_target],
            'critic2_optimizer': [o.state_dict() for o in self.critic2_optimizer]
        })
        return save_dict

    def _load_from_save_dict(self, save_dict: dict) -> None:
        """
        Load the model parameters from a save dictionary.

        Args:
            save_dict (dict): Dictionary containing the model parameters.
        """
        super()._load_from_save_dict(save_dict)
        for agent_idx in range(self.num_agents):
            self.critic2[agent_idx].load_state_dict(save_dict['critic2'][agent_idx])
            self.critic2_target[agent_idx].load_state_dict(save_dict['critic2_target'][agent_idx])
            self.critic2_optimizer[agent_idx].load_state_dict(save_dict['critic2_optimizer'][agent_idx])

    def eval(self) -> None:
        """
        Set the model to evaluation mode.
        """
        super().eval()
        for critic2, critic2_target in zip(self.critic2, self.critic2_target):
            critic2.eval()
            critic2_target.eval()

    def get_action(self, obs: torch.Tensor) -> tuple[list[torch.Tensor], None, None]:
        """
        Get the actions for each agent with Gaussian exploration noise.

        Args:
            obs (torch.Tensor): The observations for each agent.

        Returns:
            actions (list[torch.Tensor]): The actions for each agent with Gaussian exploration.
            log_probs (None): Placeholder for log probabilities (not used in MATD3).
            means (None): Placeholder for action means (not used in MATD3).
        """
        actions = []
        for agent_idx, actor in enumerate(self.actor):
            action = actor.get_action(obs[agent_idx])
            noise = torch.randn_like(action) * self.exploration_noise
            action = torch.clamp(action + noise, -1.0, 1.0)
            actions.append(action)
        return actions, None, None

    def get_agent_q_values(
        self,
        agent_idx: int,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor,
        use_target: bool = False,
        detach_preprocessor: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get Q values from both critics for a specific agent.

        Args:
            agent_idx (int): The index of the agent.
            all_obs (torch.Tensor): Concatenated observations from all agents.
            all_actions (torch.Tensor): Concatenated actions from all agents.
            use_target (bool): Whether to use target networks.
            detach_preprocessor (bool): Whether to detach the preprocessor from the graph. Not used by base MATD3.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Q values from critic1 and critic2.
        """
        if use_target:
            q1 = self.critic_target[agent_idx](all_obs, all_actions)
            q2 = self.critic2_target[agent_idx](all_obs, all_actions)
        else:
            q1 = self.critic[agent_idx](all_obs, all_actions)
            q2 = self.critic2[agent_idx](all_obs, all_actions)
        return q1, q2

    def on_episode_reset(self) -> None:
        """
        Handle episode reset. No-op for MATD3 since Gaussian noise does not depend on episode state.
        """
        pass

    def train(self) -> None:
        """
        Set the model to training mode, including both critics.
        """
        super().train()
        for critic2, critic2_target in zip(self.critic2, self.critic2_target):
            critic2.train()
            critic2_target.train()

    def update_target_networks(self, agent_idx: int, tau: float) -> None:
        """
        Update all target networks with soft updates, including critic2.

        Args:
            agent_idx (int): Index of the agent.
            tau (float): Soft update factor.
        """
        super().update_target_networks(agent_idx, tau)
        for param, target_param in zip(self.critic2[agent_idx].parameters(), self.critic2_target[agent_idx].parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class GraphMATD3(GraphMADDPG, MATD3):
    """
    Graph Multi-Agent Twin Delayed DDPG.

    Extends MATD3 with graph neural network preprocessing and attention pooling.

    Attributes:
        services_rgcns (list[torch.nn.Module]): RGCN networks for each agent.
        services_rgcn_optims (list[torch.optim.Optimizer]): Optimizers for RGCN networks.
    """

    def __init__(
        self,
        env: StatsSubprocVectorEnv,
        device: torch.device,
        policy_lr: float = 0.001,
        q_lr: float = 0.001,
        services_rgcn_lr: float = 0.001,
        exploration_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        detach_actor_from_preprocessor: bool = True,
        rgcn_num_layers: int = DEFAULT_NUM_LAYERS
    ) -> None:
        """
        Initialize the Graph Multi-Agent Twin Delayed DDPG algorithm.

        Each agent has its own ServicesRGCN network and ServiceAttentionPooling module.
        Uses regular CentralizedCritic with attention-pooled embeddings.

        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            device (torch.device): The device to run the algorithm on.
            policy_lr (float): The learning rate of the policy network optimizer.
            q_lr (float): The learning rate of the Q network optimizer.
            services_rgcn_lr (float): The learning rate of the GCN and attention pooling optimizer.
            exploration_noise (float): Std of fixed Gaussian noise for exploration.
            target_noise (float): Std of Gaussian noise for target policy smoothing.
            noise_clip (float): Clip range for target policy smoothing noise.
            detach_actor_from_preprocessor (bool): Whether to detach the preprocessor outputs before passing to
                the actor. Defaults to True, meaning only critic gradients will train the RGCN and attention pooling.
            rgcn_num_layers (int): Number of RGCN layers.
        """
        embedding_env = self._create_embedding_wrapper(env)
        MATD3.__init__(self, embedding_env, device, policy_lr, q_lr, exploration_noise, target_noise, noise_clip)
        self._init_graph_processing(env, device, services_rgcn_lr, rgcn_num_layers=rgcn_num_layers)
        self.preprocessors = self.services_rgcns
        self.preprocessor_optimizers = self.services_rgcn_optims
        self.detach_actor_from_preprocessor = detach_actor_from_preprocessor

    def get_agent_q_values(
        self,
        agent_idx: int,
        all_obs: list[np.ndarray],
        all_actions: torch.Tensor,
        use_target: bool = False,
        detach_preprocessor: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get Q values from both critics with graph-processed observations.

        Args:
            agent_idx (int): The index of the agent.
            all_obs (list[np.ndarray]): Graph observations from all agents.
            all_actions (torch.Tensor): Concatenated actions from all agents.
            use_target (bool): Whether to use target networks.
            detach_preprocessor (bool): Whether to detach the preprocessor outputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Q values from critic1 and critic2.
        """
        # Process each agent's observations through their own GCN and attention pooling
        all_embedding_obs = []
        for other_agent_idx in range(self.num_agents):
            # Ignore attention weights for Q-values
            embedding_obs, _ = self._process_graph_observations(all_obs[other_agent_idx], other_agent_idx)
            if detach_preprocessor:
                embedding_obs = embedding_obs.detach()
            all_embedding_obs.append(embedding_obs)

        # Concatenate all pooled embeddings: [batch_size, num_agents * embedding_dim]
        concatenated_obs = torch.cat(all_embedding_obs, dim=1)
        if use_target:
            q1 = self.critic_target[agent_idx](concatenated_obs, all_actions)
            q2 = self.critic2_target[agent_idx](concatenated_obs, all_actions)
        else:
            q1 = self.critic[agent_idx](concatenated_obs, all_actions)
            q2 = self.critic2[agent_idx](concatenated_obs, all_actions)
        return q1, q2
