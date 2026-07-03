"""MADDPG algorithm implementation."""

import numpy as np
import torch
import torch.optim as optim

from robin.rl.algorithms.actors import DeterministicActor
from robin.rl.algorithms.constants import DEFAULT_NUM_LAYERS
from robin.rl.algorithms.critics import CentralizedCritic
from robin.rl.algorithms.graph_mixin import GraphProcessingMixin
from robin.rl.algorithms.noise import OUNoise
from robin.rl.entities import StatsSubprocVectorEnv


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient.
    
    Attributes:
        num_agents (int): Number of agents in the environment.
        actor (list[DeterministicActor]): Actor networks for each agent.
        critic (list[CentralizedCritic]): Centralized critic networks for each agent.
        actor_target (list[DeterministicActor]): Target actor networks for each agent.
        critic_target (list[CentralizedCritic]): Target centralized critic networks for each agent.
        actor_optimizer (list[torch.optim]): Actor network optimizers for each agent.
        critic_optimizer (list[torch.optim]): Critic network optimizers for each agent.
        noise_generators (list[OUNoise]): OU noise generators for exploration for each agent.
        device (torch.device): Device to run the algorithm on.
        total_timesteps (int): Total training timesteps.
        init_noise_scale (float): Initial exploration noise scale.
        final_noise_scale (float): Final exploration noise scale.
        current_step (int): Current training step counter.
        detach_actor_from_preprocessor (bool): Whether to detach the preprocessor outputs before passing to the actor.
    """

    def __init__(
        self,
        env: StatsSubprocVectorEnv,
        device: torch.device,
        policy_lr: float = 0.001,
        q_lr: float = 0.001,
        total_timesteps: int = 1_000_000,
        init_noise_scale: float = 0.3,
        final_noise_scale: float = 0.0,
        detach_actor_from_preprocessor: bool = True
    ) -> None:
        """
        Initialize the Multi-Agent Deep Deterministic Policy Gradient algorithm.

        Each agent has its own actor and centralized critic networks.

        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            device (torch.device): The device to run the algorithm on.
            policy_lr (float): The learning rate of the policy network optimizer.
            q_lr (float): The learning rate of the Q network optimizer.
            total_timesteps (int): Total training timesteps.
            init_noise_scale (float): Initial exploration noise scale.
            final_noise_scale (float): Final exploration noise scale.
            detach_actor_from_preprocessor (bool): Whether to detach the preprocessor outputs before passing to the actor.
        """
        self.num_agents = env.get_env_attr('num_agents')[0]
        self.actor = [DeterministicActor(env, agent_idx).to(device) for agent_idx in range(self.num_agents)]
        self.actor_target = [DeterministicActor(env, agent_idx).to(device) for agent_idx in range(self.num_agents)]
        self.critic = [CentralizedCritic(env).to(device) for agent_idx in range(self.num_agents)]
        self.critic_target = [CentralizedCritic(env).to(device) for agent_idx in range(self.num_agents)]
        for agent_idx in range(self.num_agents):
            self.actor_target[agent_idx].load_state_dict(self.actor[agent_idx].state_dict())
            self.critic_target[agent_idx].load_state_dict(self.critic[agent_idx].state_dict())
        self.actor_optimizer = [optim.Adam(self.actor[i].parameters(), lr=policy_lr) for i in range(self.num_agents)]
        self.critic_optimizer = [optim.Adam(self.critic[i].parameters(), lr=q_lr) for i in range(self.num_agents)]
        self.noise_generators = []
        for agent_idx in range(self.num_agents):
            action_dim = env.action_space[0][agent_idx].shape[0]
            self.noise_generators.append(OUNoise(action_dim))
        self.device = device
        self.total_timesteps = total_timesteps
        self.init_noise_scale = init_noise_scale
        self.final_noise_scale = final_noise_scale
        self.current_step = 0
        self.detach_actor_from_preprocessor = detach_actor_from_preprocessor

    def _compute_noise_scale(self) -> float:
        progress = min(1.0, self.current_step / self.total_timesteps)
        return self.init_noise_scale + (self.final_noise_scale - self.init_noise_scale) * progress

    def _get_save_dict(self) -> dict:
        """
        Get the state dictionary for saving the model.

        Returns:
            dict: Dictionary containing the model parameters.
        """
        return {
            'num_agents': self.num_agents,
            'actor': [actor.state_dict() for actor in self.actor],
            'actor_target': [actor_target.state_dict() for actor_target in self.actor_target],
            'critic': [critic.state_dict() for critic in self.critic],
            'critic_target': [critic_target.state_dict() for critic_target in self.critic_target],
            'actor_optimizer': [actor_opt.state_dict() for actor_opt in self.actor_optimizer],
            'critic_optimizer': [critic_opt.state_dict() for critic_opt in self.critic_optimizer]
        }

    def _load_from_save_dict(self, save_dict: dict) -> None:
        """
        Load the model parameters from a save dictionary.

        Args:
            save_dict (dict): Dictionary containing the model parameters.
        """
        self.num_agents = save_dict['num_agents']
        for agent_idx in range(self.num_agents):
            self.actor[agent_idx].load_state_dict(save_dict['actor'][agent_idx])
            self.actor_target[agent_idx].load_state_dict(save_dict['actor_target'][agent_idx])
            self.critic[agent_idx].load_state_dict(save_dict['critic'][agent_idx])
            self.critic_target[agent_idx].load_state_dict(save_dict['critic_target'][agent_idx])
            self.actor_optimizer[agent_idx].load_state_dict(save_dict['actor_optimizer'][agent_idx])
            self.critic_optimizer[agent_idx].load_state_dict(save_dict['critic_optimizer'][agent_idx])

    def eval(self) -> None:
        """
        Set the model to evaluation mode.
        """
        for nets in zip(self.actor, self.critic, self.actor_target, self.critic_target):
            for net in nets:
                net.eval()

    def get_action(self, obs: torch.Tensor) -> tuple[list[torch.Tensor], None, None]:
        """
        Get the actions for each agent with exploration noise handling.

        Args:
            obs (torch.Tensor): The observations for each agent.

        Returns:
            actions (list[torch.Tensor]): The actions for each agent with appropriate exploration.
            log_probs (None): Placeholder for log probabilities (not used in MADDPG).
            means (None): Placeholder for action means (not used in MADDPG).
        """
        actions = []
        noise_scale = self._compute_noise_scale()
        for agent_idx, actor in enumerate(self.actor):
            action = actor.get_action(obs[agent_idx])
            self.noise_generators[agent_idx].scale = noise_scale
            noise = torch.from_numpy(self.noise_generators[agent_idx].noise()).float().to(self.device)
            action = torch.clamp(action + noise, -1.0, 1.0)
            actions.append(action)
        return actions, None, None

    def get_agent_action(self, agent_idx: int, obs: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """
        Get the action for a specific agent.

        Args:
            agent_idx (int): The index of the agent.
            obs (torch.Tensor): The observation for the agent.
            use_target (bool): Whether to use target actor networks.

        Returns:
            action (torch.Tensor): The action for the agent.
        """
        if use_target:
            return self.actor_target[agent_idx].get_action(obs)
        else:
            return self.actor[agent_idx].get_action(obs)

    def get_agent_q_values(
        self,
        agent_idx: int,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor,
        use_target: bool = False,
        detach_preprocessor: bool = False
    ) -> torch.Tensor:
        """
        Get the Q values for a specific agent's centralized critic.

        Args:
            agent_idx (int): The index of the agent.
            all_obs (torch.Tensor): Concatenated observations from all agents.
            all_actions (torch.Tensor): Concatenated actions from all agents.
            use_target (bool): Whether to use target networks.
            detach_preprocessor (bool): Whether to detach the preprocessor from the graph. Not used by base MADDPG.

        Returns:
            q_value (torch.Tensor): The Q values from the centralized critic.
        """
        if use_target:
            return self.critic_target[agent_idx](all_obs, all_actions)
        else:
            return self.critic[agent_idx](all_obs, all_actions)

    @classmethod
    def load_model(cls, path: str, env: StatsSubprocVectorEnv, device: torch.device) -> 'MADDPG':
        """
        Load the model parameters from a file.

        Args:
            path (str): The path to load the model parameters from.
            env (StatsSubprocVectorEnv): The environment.
            device (torch.device): The device to run the algorithm on.
        """
        save_dict = torch.load(path)
        model = cls(env, device)
        model._load_from_save_dict(save_dict)
        return model

    def on_episode_reset(self) -> None:
        """
        Handle episode reset by resetting noise generators.
        """
        self.reset_noise()

    def reset_noise(self) -> None:
        """Reset the OU noise generators for all agents."""
        for noise_gen in self.noise_generators:
            noise_gen.reset()

    def save_model(self, path: str) -> None:
        """
        Save the model parameters to a file.

        Args:
            path (str): The path to save the model parameters to.
        """
        save_dict = self._get_save_dict()
        torch.save(save_dict, path)

    def set_global_step(self, step: int) -> None:
        """
        Update the internal training step counter.

        Args:
            step (int): Current global training step.
        """
        self.current_step = step

    def train(self) -> None:
        """
        Set the model to training mode.
        """
        for nets in zip(self.actor, self.critic, self.actor_target, self.critic_target):
            for net in nets:
                net.train()

    def update_target_networks(self, agent_idx: int, tau: float) -> None:
        """
        Update the target networks with soft updates.

        Args:
            agent_idx (int): Index of the agent.
            tau (float): Soft update factor.
        """
        for param, target_param in zip(self.actor[agent_idx].parameters(), self.actor_target[agent_idx].parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic[agent_idx].parameters(), self.critic_target[agent_idx].parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class GraphMADDPG(GraphProcessingMixin, MADDPG):
    """
    Graph Multi-Agent Deep Deterministic Policy Gradient.
    
    Extends MADDPG with graph neural network preprocessing and attention pooling for each agent.

    Attributes:
        services_rgcns (list[torch.nn.Module]): List of RGCN networks for processing graph observations for each agent.
        services_rgcn_optims (list[torch.optim.Optimizer]): List of optimizers for the RGCN networks for each agent.
    """

    def __init__(self,
        env: StatsSubprocVectorEnv,
        device: torch.device,
        policy_lr: float = 0.001,
        q_lr: float = 0.001,
        services_rgcn_lr: float = 0.001,
        total_timesteps: int = 1_000_000,
        init_noise_scale: float = 0.3,
        final_noise_scale: float = 0.0,
        detach_actor_from_preprocessor: bool = True,
        rgcn_num_layers: int = DEFAULT_NUM_LAYERS
    ) -> None:
        """
        Initialize the Graph Multi-Agent Deep Deterministic Policy Gradient algorithm.

        Each agent has its own ServicesRGCN network and ServiceAttentionPooling module.
        Uses regular CentralizedCritic with attention-pooled embeddings.

        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            device (torch.device): The device to run the algorithm on.
            policy_lr (float): The learning rate of the policy network optimizer.
            q_lr (float): The learning rate of the Q network optimizer.
            services_rgcn_lr (float): The learning rate of the GCN and attention pooling optimizer.
            total_timesteps (int): Total training timesteps.
            init_noise_scale (float): Initial exploration noise scale.
            final_noise_scale (float): Final exploration noise scale.
            detach_actor_from_preprocessor (bool): Whether to detach the preprocessor outputs before passing to
                the actor. Defaults to True, meaning only critic gradients will train the RGCN and attention pooling.
            rgcn_num_layers (int): Number of RGCN layers.
        """
        embedding_env = self._create_embedding_wrapper(env)
        super().__init__(embedding_env, device, policy_lr, q_lr, total_timesteps, init_noise_scale, final_noise_scale)
        self._init_graph_processing(env, device, services_rgcn_lr, rgcn_num_layers=rgcn_num_layers)
        self.preprocessors = self.services_rgcns
        self.preprocessor_optimizers = self.services_rgcn_optims
        self.detach_actor_from_preprocessor = detach_actor_from_preprocessor

    def _get_save_dict(self) -> dict:
        """
        Get the state dictionary for saving the model including RGCN networks and optimizers.
        
        Returns:
            dict: Dictionary containing the model parameters including RGCN components.
        """
        return self._get_save_dict_with_rgcn()

    def _load_from_save_dict(self, save_dict: dict) -> None:
        """
        Load the model parameters from a save dictionary including RGCN networks and optimizers.
        
        Args:
            save_dict (dict): Dictionary containing the model parameters.
        """
        self._load_from_save_dict_with_rgcn(save_dict)

    def eval(self) -> None:
        """
        Set the model to evaluation mode including RGCN networks and attention pooling.
        """
        super().eval()
        self._eval_rgcns()

    def get_action(self, obs: list[np.ndarray]) -> tuple[list[torch.Tensor], None, None]:
        """
        Get the actions for each agent in the environment with graph preprocessing.

        Args:
            obs (list[np.ndarray]): The graph observations for each agent.

        Returns:
            actions (list[torch.Tensor]): The actions for each agent with appropriate exploration.
            log_probs (None): Placeholder for log probabilities (not used in MADDPG).
            means (None): Placeholder for action means (not used in MADDPG).
        """
        # Process graph observations for each agent using their R-GCNs
        processed_obs = []
        for agent_idx in range(self.num_agents):
            # Ignore attention weights when processing observations for action selection
            embedding_obs, _ = self._process_graph_observations(obs[agent_idx], agent_idx)
            if self.detach_actor_from_preprocessor:
                embedding_obs = embedding_obs.detach()
            processed_obs.append(embedding_obs)
        return super().get_action(processed_obs)

    def get_agent_action(self, agent_idx: int, obs: np.ndarray, use_target: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the action for a specific agent with graph preprocessing.

        Args:
            agent_idx (int): The index of the agent.
            obs (np.ndarray): The graph observation for the agent.
            use_target (bool): Whether to use target actor networks.

        Returns:
            tuple: (action, attention_weights)
        """
        embedding_obs, attention_weights = self._process_graph_observations(obs, agent_idx)
        if self.detach_actor_from_preprocessor:
            embedding_obs = embedding_obs.detach()
        if use_target:
            action = self.actor_target[agent_idx].get_action(embedding_obs)
        else:
            action = self.actor[agent_idx].get_action(embedding_obs)
        return action, attention_weights

    def get_agent_q_values(
        self,
        agent_idx: int,
        all_obs: list[np.ndarray],
        all_actions: torch.Tensor,
        use_target: bool = False,
        detach_preprocessor: bool = False
    ) -> torch.Tensor:
        """
        Get the Q values for a specific agent's centralized critic with attention pooling.

        Args:
            agent_idx (int): The index of the agent.
            all_obs (list[np.ndarray]): Graph observations from all agents.
            all_actions (torch.Tensor): Concatenated actions from all agents.
            use_target (bool): Whether to use target networks.
            detach_preprocessor (bool): Whether to detach the preprocessor outputs.

        Returns:
            q_value (torch.Tensor): The Q values from the centralized critic.
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
            return self.critic_target[agent_idx](concatenated_obs, all_actions)
        else:
            return self.critic[agent_idx](concatenated_obs, all_actions)

    def get_embeddings(self, obs: list[np.ndarray]) -> list[torch.Tensor]:
        """
        Get graph-level embeddings for each agent (for visualization/analysis).

        Args:
            obs (list[np.ndarray]): Graph observations for each agent.

        Returns:
            embeddings (list[torch.Tensor]): Graph embeddings for each agent.
        """
        embeddings = []
        for agent_idx in range(self.num_agents):
            embedding_obs, _ = self._process_graph_observations(obs[agent_idx], agent_idx)
            embeddings.append(embedding_obs.detach())  # Detach since it is only for visualization
        return embeddings

    def on_episode_reset(self) -> None:
        """Handle episode reset by resetting noise generators."""
        self.reset_noise()

    def train(self) -> None:
        """
        Set the model to training mode including RGCN networks and attention pooling.
        """
        super().train()
        self._train_rgcns()
