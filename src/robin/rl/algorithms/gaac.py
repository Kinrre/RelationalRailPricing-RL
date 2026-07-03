"""GA-AC algorithm implementation."""

import numpy as np
import torch
import torch.optim as optim

from robin.rl.algorithms.actors import StochasticActor
from robin.rl.algorithms.g2anet import G2ANetCritic
from robin.rl.entities import StatsSubprocVectorEnv


class GAAC:
    """
    Graph Attention Actor-Critic (GA-AC).

    Each agent has an independent stochastic actor and a G2ANetCritic that attends over other agents' encodings.

    Attributes:
        num_agents (int): Number of agents in the environment.
        actor (list[StochasticActor]): Per-agent stochastic actor networks.
        critic (list[G2ANetCritic]): Per-agent G2ANet critics.
        critic_target (list[G2ANetCritic]): Per-agent target critics.
        actor_optimizer (list[Adam]): Per-agent actor optimizers.
        critic_optimizer (list[Adam]): Per-agent critic optimizers.
        device (torch.device): Device to run the algorithm on.
        total_timesteps (int): Total training timesteps (for temperature annealing).
        init_temperature (float): Initial Gumbel-softmax temperature.
        min_temperature (float): Minimum Gumbel-softmax temperature.
    """

    def __init__(
        self,
        env: StatsSubprocVectorEnv,
        device: torch.device,
        policy_lr: float = 0.001,
        q_lr: float = 0.001,
        total_timesteps: int = 1_000_000,
        init_temperature: float = 1.0,
        min_temperature: float = 0.1
    ) -> None:
        """
        Initialize GA-AC.

        Args:
            env: The multi-agent environment.
            device: Device to run on.
            policy_lr: Learning rate for actor networks.
            q_lr: Learning rate for critic networks.
            total_timesteps: Total training steps, used for temperature annealing.
            init_temperature: Initial Gumbel-softmax temperature.
            min_temperature: Minimum Gumbel-softmax temperature.
        """
        self.num_agents = env.get_env_attr('num_agents')[0]
        obs_dims = [np.prod(env.observation_space[0][i].shape) for i in range(self.num_agents)]
        action_dims = [np.prod(env.action_space[0][i].shape) for i in range(self.num_agents)]
        self.actor = [StochasticActor(env, i).to(device) for i in range(self.num_agents)]
        self.critic = [
            G2ANetCritic(
                obs_dims[i], action_dims[i],
                [obs_dims[j] for j in range(self.num_agents) if j != i],
                [action_dims[j] for j in range(self.num_agents) if j != i]
            ).to(device)
            for i in range(self.num_agents)
        ]
        self.critic_target = [
            G2ANetCritic(
                obs_dims[i], action_dims[i],
                [obs_dims[j] for j in range(self.num_agents) if j != i],
                [action_dims[j] for j in range(self.num_agents) if j != i]
            ).to(device)
            for i in range(self.num_agents)
        ]
        for i in range(self.num_agents):
            self.critic_target[i].load_state_dict(self.critic[i].state_dict())
        self.actor_optimizer = [optim.Adam(self.actor[i].parameters(), lr=policy_lr) for i in range(self.num_agents)]
        self.critic_optimizer = [optim.Adam(self.critic[i].parameters(), lr=q_lr) for i in range(self.num_agents)]
        self.device = device
        self.total_timesteps = total_timesteps
        self.init_temperature = init_temperature
        self.min_temperature = min_temperature

    def _get_save_dict(self) -> dict:
        """
        Get the state dictionary for saving the model.

        Returns:
            dict: Dictionary containing the model parameters.
        """
        return {
            'num_agents': self.num_agents,
            'actor': [actor.state_dict() for actor in self.actor],
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
            self.critic[agent_idx].load_state_dict(save_dict['critic'][agent_idx])
            self.critic_target[agent_idx].load_state_dict(save_dict['critic_target'][agent_idx])
            self.actor_optimizer[agent_idx].load_state_dict(save_dict['actor_optimizer'][agent_idx])
            self.critic_optimizer[agent_idx].load_state_dict(save_dict['critic_optimizer'][agent_idx])

    def eval(self) -> None:
        """
        Set the model to evaluation mode.
        """
        for nets in zip(self.actor, self.critic, self.critic_target):
            for net in nets:
                net.eval()

    def get_action(
        self,
        obs: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """
        Get actions for each agent given their observations.

        Args:
            obs (torch.Tensor): The observation for each agent.

        Returns:
            actions (list[torch.Tensor]): The actions for each agent.
            log_prob (torch.Tensor): The log probabilities of the actions.
            mean (list[torch.Tensor]): The means of the actions.
        """
        actions, log_probs, means = zip(*[self.actor[i].get_action(obs[i]) for i in range(self.num_agents)])
        return list(actions), torch.stack(log_probs), list(means)

    def get_agent_action(
        self,
        agent_idx: int,
        obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the action for a specific agent.

        Args:
            agent_idx (int): The index of the agent.
            obs (torch.Tensor): The observation for the agent.

        Returns:
            action (torch.Tensor): The action for the agent.
            log_prob (torch.Tensor): The log probability of the action.
            mean (torch.Tensor): The mean of the action.
        """
        return self.actor[agent_idx].get_action(obs)

    def get_agent_q_values(
        self,
        agent_idx: int,
        all_obs: list[torch.Tensor],
        all_actions: list[torch.Tensor],
        use_target: bool = False
    ) -> tuple[torch.Tensor, dict]:
        """
        Get the Q values for a specific agent using G2ANetCritic.

        Args:
            agent_idx (int): The index of the agent.
            all_obs (torch.Tensor): Concatenated observations from all agents.
            all_actions (torch.Tensor): Concatenated actions from all agents.
            use_target (bool): Whether to use target networks.

        Returns:
            q_values (torch.Tensor): Q-values for the agent.
            attn_info (dict): Attention info from G2ANet with 'hard_gates' and 'soft_weights'.
        """
        obs_i = all_obs[agent_idx]
        action_i = all_actions[agent_idx]
        other_indices = [i for i in range(self.num_agents) if i != agent_idx]
        obs_others_list = [all_obs[i] for i in other_indices]
        actions_others_list = [all_actions[i] for i in other_indices]
        critic = self.critic_target[agent_idx] if use_target else self.critic[agent_idx]
        return critic(obs_i, action_i, obs_others_list, actions_others_list)

    @classmethod
    def load_model(cls, path: str, env: StatsSubprocVectorEnv, device: torch.device) -> 'GAAC':
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
        """No-op: stochastic policy requires no external noise reset."""
        pass

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
        Anneal the Gumbel-softmax temperature linearly from init to min.

        Args:
            step (int): Current global training step.
        """
        progress = min(1.0, step / self.total_timesteps)
        temperature = self.init_temperature - progress * (self.init_temperature - self.min_temperature)
        for i in range(self.num_agents):
            self.critic[i].g2anet.temperature = temperature
            self.critic_target[i].g2anet.temperature = temperature

    def train(self) -> None:
        """
        Set the model to training mode.
        """
        for nets in zip(self.actor, self.critic, self.critic_target):
            for net in nets:
                net.train()

    def update_target_networks(self, agent_idx: int, tau: float) -> None:
        """
        Update the target networks with soft updates.

        Args:
            agent_idx (int): Index of the agent.
            tau (float): Soft update factor.
        """
        for param, target_param in zip(self.critic[agent_idx].parameters(), self.critic_target[agent_idx].parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
