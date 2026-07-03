"""Trainer module for experimenting with different RL algorithms."""

import datetime
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import time
import tyro
import yaml

from dataclasses import dataclass
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Union

from robin.rl.algorithms.buffers import ReplayBuffer, GraphReplayBuffer
from robin.rl.algorithms.constants import IS_GRAPH_BASED, TRAINER_SEED_RANK_MULTIPLIER
from robin.rl.algorithms.gaac import GAAC
from robin.rl.algorithms.maddpg import MADDPG
from robin.rl.algorithms.matd3 import MATD3, GraphMATD3
from robin.rl.algorithms.utils import create_env

AlgorithmName = Union[MADDPG, MATD3, GraphMATD3, GAAC]
ALGORITHM_REGISTRY: dict[str, AlgorithmName] = {
    'maddpg': MADDPG,
    'matd3': MATD3,
    'rache': GraphMATD3,
    'gaac': GAAC
}


@dataclass
class TrainerArgs:
    algorithm: Literal['maddpg', 'matd3', 'rache', 'gaac'] = 'rache'
    """the algorithm to use"""
    output_dir: str = 'models'
    """the output directory to store the logs"""
    exp_name: str = 'default'
    """the name of this experiment"""
    supply_config: str = 'configs/rl/supply_data.yaml'
    """path to the supply data configuration file"""
    demand_config: str = 'configs/rl/demand_data_business.yaml'
    """path to the demand data configuration file"""
    seed: int = 0
    """seed of the experiment"""
    cuda: bool = True
    """whether to use cuda"""
    n_workers: int = 16
    """number of workers for training (it should be the number of cores)"""
    total_timesteps: int = 1_000_000
    """total timesteps to train the agent"""
    buffer_size: int = 1_000_000
    """size of the replay buffer"""
    learning_starts: int = 5_000
    """number of timesteps to start learning"""
    batch_size: int = 1024
    """the batch size of sample from the reply memory"""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 0.005
    """soft update factor"""
    policy_lr: float = 0.001
    """learning rate of the policy network optimizer"""
    q_lr: float = 0.001
    """learning rate of the q network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    rgcn_num_layers: int = 2
    """number of RGCN layers for graph-based algorithms"""
    normalize_obs: bool = True
    """whether to normalize observations"""
    normalize_obs_output: bool = False
    """whether to normalize on-the-fly observations"""
    reward_scale_factor: float | None = None
    """scale factor for rewards. If set, it overrides reward normalization in the environment."""
    detach_actor_from_preprocessor: bool = False
    """whether to detach the preprocessor outputs before passing to the actor. When True, only critic gradients train the RGCN and attention pooling."""
    exclude_edge_type: int | None = None
    """edge type to exclude for ablation studies (0=same_market, 1=same_agent, 2=dest_origin). None keeps all relation types."""


class Trainer:
    """
    Trainer class for training the agent.
    
    Attributes:
        args (TrainerArgs): Arguments for the trainer.
        run_name (str): Name of the experiment.
        writer (SummaryWriter): Tensorboard writer.
        device (torch.device): Device to run the experiment on.
        env (StatsSubprocVectorEnv): ROBIN environment.
        agent (AlgorithmName): The agent to train.
        replay_buffer (ReplayBuffer): Replay buffer for storing the experiences.
        critic_updates (int): Counter for critic updates, used for delayed policy updates.
    """ 
    
    def __init__(self) -> None:
        """
        Initialize the trainer.
        
        It uses the arguments from the command line to initialize the trainer,
        please refer to the TrainerArgs class.
        """
        self.args = tyro.cli(TrainerArgs)
        now = datetime.datetime.now().strftime('%d%m%y-%H%M%S')
        self.run_name = f'{self.args.output_dir}/{self.args.exp_name}/{self.args.algorithm}/{self.args.seed}/{now}'
        self.writer = SummaryWriter(self.run_name)
        self.log_args_and_git_commit()
        self.set_seed(self.args.seed)
        self.device = torch.device('cuda' if self.args.cuda and torch.cuda.is_available() else 'cpu')
        self.env = create_env(
            supply_config=self.args.supply_config,
            demand_config=self.args.demand_config,
            algorithm=self.args.algorithm,
            seed=self.args.seed,
            seed_rank_multiplier=TRAINER_SEED_RANK_MULTIPLIER,
            n_workers=self.args.n_workers,
            run_name=self.run_name,
            normalize_obs=self.args.normalize_obs,
            normalize_obs_output=self.args.normalize_obs_output,
            reward_scale_factor=self.args.reward_scale_factor,
            exclude_edge_type=self.args.exclude_edge_type
        )
        self.agent: AlgorithmName = self._get_agent()
        self.agent.eval()
        self.replay_buffer = self._create_replay_buffer()
        self.critic_updates = 0

    def _create_replay_buffer(self) -> Union[ReplayBuffer, GraphReplayBuffer]:
        """
        Create and return the appropriate buffer based on the algorithm type.

        Returns:
            Union[ReplayBuffer, GraphReplayBuffer, RolloutBuffer]: The buffer.
        """
        ac_dims = [acsp.shape[0] for acsp in self.env.action_space[0]]
        if IS_GRAPH_BASED[self.args.algorithm]:
            return GraphReplayBuffer(
                max_steps=self.args.buffer_size,
                num_agents=self.agent.num_agents,
                ac_dims=ac_dims
            )
        obs_dims = [obsp.shape[0] for obsp in self.env.observation_space[0]]
        return ReplayBuffer(
            max_steps=self.args.buffer_size,
            num_agents=self.agent.num_agents,
            obs_dims=obs_dims,
            ac_dims=ac_dims
        )

    def _get_agent(self) -> AlgorithmName:
        """
        Create and return the agent based on the algorithm type.

        Returns:
            AlgorithmName: The instantiated agent.
        """
        if self.args.algorithm not in ALGORITHM_REGISTRY:
            raise Exception(f'Algorithm {self.args.algorithm} is not supported. Please choose from {list(ALGORITHM_REGISTRY.keys())}.')
        agent_class = ALGORITHM_REGISTRY[self.args.algorithm]
        kwargs = {'env': self.env, 'device': self.device, 'policy_lr': self.args.policy_lr, 'q_lr': self.args.q_lr}
        if self.args.algorithm in ('maddpg', 'gaac'):
            kwargs.update({'total_timesteps': self.args.total_timesteps})
        if self.args.algorithm in ('maddpg', 'matd3', 'rache'):
            kwargs.update({'detach_actor_from_preprocessor': self.args.detach_actor_from_preprocessor})
        if self.args.algorithm == 'rache':
            kwargs.update({'rgcn_num_layers': self.args.rgcn_num_layers})
        return agent_class(**kwargs)

    def _log_attention_weights(self, global_step: int, i: int, attention_weights: torch.Tensor) -> None:
        """
        Log the attention weights as a heatmap and their entropy.

        Args:
            global_step (int): Current global step.
            i (int): Index of the agent.
            attention_weights (torch.Tensor): Attention weights.
        """
        attention_dist = attention_weights[:, :, 0].mean(dim=0)
        attention_normalized = (attention_dist - attention_dist.min()) / (attention_dist.max() - attention_dist.min() + 1e-8)
        attention_heatmap = attention_normalized.unsqueeze(0).unsqueeze(0)
        self.writer.add_image(f'attention_heatmap/agent_{i}', attention_heatmap, global_step)
        entropy = self.agent.compute_attention_entropy(attention_weights).mean().item()
        self.writer.add_scalar(f'attention_{i}/attention_entropy', entropy, global_step)

    def _log_yaml(self, filename: str, data: dict, use_vars: bool = False) -> None:
        """
        Log the data to a yaml file.
        
        Args:
            filename (str): The filename to log the data.
            data (dict): The data to log.
            use_vars (bool): Whether to use the vars function to log the data.
        """
        with open(os.path.join(self.run_name, filename), 'w') as f:
            if use_vars:
                yaml.dump(vars(data), f, sort_keys=False)
            else:
                yaml.dump(data, f, sort_keys=False)

    def log_stats(
        self,
        global_step: int,
        start_time: float,
        i: int,
        qf1_a_values: torch.Tensor,
        qf2_a_values: torch.Tensor | None,
        qf1_loss: torch.Tensor,
        qf2_loss: torch.Tensor | None,
        qf_loss: torch.Tensor,
        actor_loss: torch.Tensor,
        attention_weights: torch.Tensor | None = None
    ) -> None:
        """
        Log the statistics of the training.
        
        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
            i (int): Index of the agent.
            qf1_a_values (torch.Tensor): First Q function values.
            qf2_a_values (torch.Tensor or None): Second Q function values (None for single critic).
            qf1_loss (torch.Tensor): First Q function loss.
            qf2_loss (torch.Tensor or None): Second Q function loss (None for single critic).
            qf_loss (torch.Tensor): Total Q function loss.
            actor_loss (torch.Tensor): Actor loss.
            attention_weights (torch.Tensor or None): Attention weights [batch_size, num_services, 1] (None for non-graph algorithms).
        """
        if global_step % 100 == 0:
            self.writer.add_scalar(f'losses_{i}/qf1_values', qf1_a_values.mean().item(), global_step)
            if qf2_a_values is not None:
                self.writer.add_scalar(f'losses_{i}/qf2_values', qf2_a_values.mean().item(), global_step)
            self.writer.add_scalar(f'losses_{i}/qf1_loss', qf1_loss.item(), global_step)
            if qf2_loss is not None:
                self.writer.add_scalar(f'losses_{i}/qf2_loss', qf2_loss.item(), global_step)
            # For single critic, don't divide by 2
            divisor = 2.0 if qf2_loss is not None else 1.0
            self.writer.add_scalar(f'losses_{i}/qf_loss', qf_loss.item() / divisor, global_step)
            self.writer.add_scalar(f'losses_{i}/actor_loss', actor_loss.item(), global_step)
            self.writer.add_scalar('losses/alpha', self.args.alpha, global_step)
            if attention_weights is not None:
                self._log_attention_weights(global_step, i, attention_weights)
            logger.info(f'SPS: {int(global_step / (time.time() - start_time))}')
            self.writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

    def log_args_and_git_commit(self) -> None:
        """
        Log arguments and git commit to the log directory.
        """
        self._log_yaml('args.yaml', self.args, use_vars=True)
        self._log_yaml('supply_config.yaml', yaml.safe_load(open(self.args.supply_config)), use_vars=False)
        self._log_yaml('demand_config.yaml', yaml.safe_load(open(self.args.demand_config)), use_vars=False)
        os.system(f'git log -1 > {self.run_name}/commit.txt')
        os.system(f'git diff > {self.run_name}/diff.patch')

    def train(self) -> None:
        """
        Train the agent.
        """
        start_time = time.time()
        obs, _ = self.env.reset()
        self.agent.on_episode_reset()

        for global_step in range(0, self.args.total_timesteps, self.env.n_envs):
            # Sample actions from the agent and step the environment
            self.agent.set_global_step(global_step)
            actions = self.sample_actions(obs, global_step)
            next_obs, terminations = self.step_and_push(obs, actions)
            
            # Update observations
            obs = next_obs
            if terminations.all():
                obs, _ = self.env.reset()
                self.agent.on_episode_reset()

            # Critic training
            if global_step > self.args.learning_starts:
                self.agent.train()
                self.train_parameters(global_step, start_time)
                self.agent.eval()

        self.agent.save_model(f'{self.run_name}/model.pt')
        if self.args.normalize_obs:
            torch.save(self.env.obs_rms, f'{self.run_name}/obs_rms.pth')
            torch.save(self.env.reward_rms, f'{self.run_name}/reward_rms.pth')
        self.env.close()
        self.writer.close()
    
    def train_parameters(self, global_step: int, start_time: float) -> None:
        """
        Train the network parameters.
        
        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
        """
        if self.args.algorithm == 'maddpg':
            self.train_maddpg(global_step, start_time)
        elif self.args.algorithm == 'matd3' or self.args.algorithm == 'rache':
            self.train_matd3(global_step, start_time)
        elif self.args.algorithm == 'gaac':
            self.train_gaac(global_step, start_time)
    
    def train_maddpg(self, global_step: int, start_time: float) -> None:
        """
        Train the MADDPG agent.
        
        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
        """
        # Only update the network parameters every 100 steps
        _global_step = global_step // self.env.n_envs
        if _global_step % 100 != 0:
            return
        for agent_idx in range(self.agent.num_agents):
            observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.args.batch_size, to_gpu=self.args.cuda)
            observations, next_observations = self.normalize_buffer_obs(observations, next_observations)

            # Concatenate observations and actions from all agents for centralized critic
            if IS_GRAPH_BASED[self.args.algorithm]:
                all_obs = observations
                all_next_obs = next_observations
            else:
                all_obs = torch.cat([observations[i] for i in range(self.agent.num_agents)], dim=1)
                all_next_obs = torch.cat([next_observations[i] for i in range(self.agent.num_agents)], dim=1)
            all_actions = torch.cat([actions[i] for i in range(self.agent.num_agents)], dim=1)
            
            # Calculate the target Q values using centralized critic
            with torch.no_grad():
                # Get next actions from all agents using target actors
                next_actions = []
                for i in range(self.agent.num_agents):
                    if IS_GRAPH_BASED[self.args.algorithm]:
                        # Ignore attention weights
                        action, _ = self.agent.get_agent_action(i, next_observations[i], use_target=True)
                    else:
                        action = self.agent.get_agent_action(i, next_observations[i], use_target=True)
                    next_actions.append(action)
                all_next_actions = torch.cat(next_actions, dim=1)
                q_next_target = self.agent.get_agent_q_values(agent_idx, all_next_obs, all_next_actions, use_target=True)
                next_q_value = rewards[agent_idx] + (1 - dones[agent_idx]) * self.args.gamma * q_next_target.squeeze()
        
            # Calculate current Q values using centralized critic
            q_value = self.agent.get_agent_q_values(agent_idx, all_obs, all_actions)
            q_value = q_value.squeeze()
            critic_loss = F.mse_loss(q_value, next_q_value)

            # Optimize critic
            if IS_GRAPH_BASED[self.args.algorithm]:
                self.agent.preprocessor_optimizers[agent_idx].zero_grad()
            self.agent.critic_optimizer[agent_idx].zero_grad()
            critic_loss.backward()
            if IS_GRAPH_BASED[self.args.algorithm]:
                self.agent.preprocessor_optimizers[agent_idx].step()
            self.agent.critic_optimizer[agent_idx].step()

            # Calculate actor loss, get actions from current agent's actor, keep others fixed
            current_actions = list(actions)
            attention_weights = None
            if IS_GRAPH_BASED[self.args.algorithm]:
                current_agent_action, attention_weights = self.agent.get_agent_action(agent_idx, observations[agent_idx])
            else:
                current_agent_action = self.agent.get_agent_action(agent_idx, observations[agent_idx])
            
            current_actions[agent_idx] = current_agent_action
            all_current_actions = torch.cat(current_actions, dim=1)
            # Use critic for actor loss, optionally detaching the preprocessor
            q_pi = self.agent.get_agent_q_values(
                agent_idx, all_obs, all_current_actions,
                detach_preprocessor=self.agent.detach_actor_from_preprocessor
            )
            actor_loss = -q_pi.mean()

            # Optimize actor (and optionally preprocessor)
            if not self.agent.detach_actor_from_preprocessor and IS_GRAPH_BASED[self.args.algorithm]:
                self.agent.preprocessor_optimizers[agent_idx].zero_grad()
            self.agent.actor_optimizer[agent_idx].zero_grad()
            actor_loss.backward()
            if not self.agent.detach_actor_from_preprocessor and IS_GRAPH_BASED[self.args.algorithm]:
                self.agent.preprocessor_optimizers[agent_idx].step()
            self.agent.actor_optimizer[agent_idx].step()

            # Log statistics with single critic values
            self.log_stats(global_step, start_time, agent_idx, q_value, None, critic_loss, None, critic_loss, actor_loss, attention_weights)
        
        # Update all target networks after all agents have been trained
        for agent_idx in range(self.agent.num_agents):
            self.agent.update_target_networks(agent_idx, self.args.tau)

    def train_matd3(self, global_step: int, start_time: float) -> None:
        """
        Train the MATD3 agent.

        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
        """
        # Only update the network parameters every 100 steps
        _global_step = global_step // self.env.n_envs
        if _global_step % 100 != 0:
            return
        for agent_idx in range(self.agent.num_agents):
            observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.args.batch_size, to_gpu=self.args.cuda)
            observations, next_observations = self.normalize_buffer_obs(observations, next_observations)

            # Concatenate observations and actions from all agents for centralized critic
            if IS_GRAPH_BASED[self.args.algorithm]:
                all_obs = observations
                all_next_obs = next_observations
            else:
                all_obs = torch.cat([observations[i] for i in range(self.agent.num_agents)], dim=1)
                all_next_obs = torch.cat([next_observations[i] for i in range(self.agent.num_agents)], dim=1)
            all_actions = torch.cat([actions[i] for i in range(self.agent.num_agents)], dim=1)

            # Calculate the target Q values with target policy smoothing
            with torch.no_grad():
                # Get next actions from all agents using target actors
                next_actions = []
                for i in range(self.agent.num_agents):
                    if IS_GRAPH_BASED[self.args.algorithm]:
                        # Ignore attention weights
                        action, _ = self.agent.get_agent_action(i, next_observations[i], use_target=True)
                    else:
                        action = self.agent.get_agent_action(i, next_observations[i], use_target=True)
                    # Target policy smoothing: add clipped Gaussian noise
                    noise = torch.clamp(torch.randn_like(action) * self.agent.target_noise, -self.agent.noise_clip, self.agent.noise_clip)
                    action = torch.clamp(action + noise, -1.0, 1.0)
                    next_actions.append(action)
                all_next_actions = torch.cat(next_actions, dim=1)
                q1_next_target, q2_next_target = self.agent.get_agent_q_values(agent_idx, all_next_obs, all_next_actions, use_target=True)
                min_q_next_target = torch.min(q1_next_target, q2_next_target)
                next_q_value = rewards[agent_idx] + (1 - dones[agent_idx]) * self.args.gamma * min_q_next_target.squeeze()

            # Calculate current Q values from both critics
            q1_value, q2_value = self.agent.get_agent_q_values(agent_idx, all_obs, all_actions)
            q1_value, q2_value = q1_value.squeeze(), q2_value.squeeze()
            q1_loss = F.mse_loss(q1_value, next_q_value)
            q2_loss = F.mse_loss(q2_value, next_q_value)
            qf_loss = q1_loss + q2_loss

            # Optimize both critics
            if IS_GRAPH_BASED[self.args.algorithm]:
                self.agent.preprocessor_optimizers[agent_idx].zero_grad()
            self.agent.critic_optimizer[agent_idx].zero_grad()
            self.agent.critic2_optimizer[agent_idx].zero_grad()
            qf_loss.backward()
            if IS_GRAPH_BASED[self.args.algorithm]:
                self.agent.preprocessor_optimizers[agent_idx].step()
            self.agent.critic_optimizer[agent_idx].step()
            self.agent.critic2_optimizer[agent_idx].step()

            # Delayed policy updates
            if self.critic_updates % self.args.policy_frequency == 0:
                # Calculate actor loss, get actions from current agent's actor, keep others fixed
                current_actions = list(actions)
                attention_weights = None
                if IS_GRAPH_BASED[self.args.algorithm]:
                    current_agent_action, attention_weights = self.agent.get_agent_action(agent_idx, observations[agent_idx])
                else:
                    current_agent_action = self.agent.get_agent_action(agent_idx, observations[agent_idx])

                current_actions[agent_idx] = current_agent_action
                all_current_actions = torch.cat(current_actions, dim=1)
                # Use critic for actor loss, optionally detaching the preprocessor
                q1_pi, _ = self.agent.get_agent_q_values(
                    agent_idx, all_obs, all_current_actions,
                    detach_preprocessor=self.agent.detach_actor_from_preprocessor
                )
                # NOTE: We can take the min of the two critics to be more conservative
                actor_loss = -q1_pi.mean()

                # Optimize actor (and optionally preprocessor)
                if not self.agent.detach_actor_from_preprocessor and IS_GRAPH_BASED[self.args.algorithm]:
                    self.agent.preprocessor_optimizers[agent_idx].zero_grad()
                self.agent.actor_optimizer[agent_idx].zero_grad()
                actor_loss.backward()
                if not self.agent.detach_actor_from_preprocessor and IS_GRAPH_BASED[self.args.algorithm]:
                    self.agent.preprocessor_optimizers[agent_idx].step()
                self.agent.actor_optimizer[agent_idx].step()
                self.log_stats(global_step, start_time, agent_idx, q1_value, q2_value, q1_loss, q2_loss, qf_loss, actor_loss, attention_weights)

                # Update the target networks
                self.agent.update_target_networks(agent_idx, self.args.tau)

        self.critic_updates += 1

    def train_gaac(self, global_step: int, start_time: float) -> None:
        """
        Train the GA-AC agent.

        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
        """
        # Only update the network parameters every 100 steps
        _global_step = global_step // self.env.n_envs
        if _global_step % 100 != 0:
            return
        for agent_idx in range(self.agent.num_agents):
            observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.args.batch_size, to_gpu=self.args.cuda)
            observations, next_observations = self.normalize_buffer_obs(observations, next_observations)

            # Calculate the target Q value
            with torch.no_grad():
                # Get next actions from all agents
                next_action_i, next_log_pi_i, _ = self.agent.get_agent_action(agent_idx, next_observations[agent_idx])
                next_all_actions = [self.agent.get_agent_action(i, next_observations[i])[0] for i in range(self.agent.num_agents)]
                next_all_actions[agent_idx] = next_action_i
                q_next_target, _ = self.agent.get_agent_q_values(agent_idx, next_observations, next_all_actions, use_target=True)
                next_q_value = (
                    rewards[agent_idx].squeeze()
                    + (1 - dones[agent_idx].squeeze()) * self.args.gamma
                    * (q_next_target.squeeze() - self.args.alpha * next_log_pi_i.squeeze())
                )

            # Optimize critic
            q_value, attn_info = self.agent.get_agent_q_values(agent_idx, observations, list(actions))
            critic_loss = F.mse_loss(q_value.squeeze(), next_q_value)
            self.agent.critic_optimizer[agent_idx].zero_grad()
            critic_loss.backward()
            self.agent.critic_optimizer[agent_idx].step()

            # Optimize actor
            pi, log_pi, _ = self.agent.get_agent_action(agent_idx, observations[agent_idx])
            current_actions = [action.detach() for action in actions]
            current_actions[agent_idx] = pi
            q_pi, _ = self.agent.get_agent_q_values(agent_idx, observations, current_actions)
            actor_loss = (self.args.alpha * log_pi - q_pi).mean()
            self.agent.actor_optimizer[agent_idx].zero_grad()
            actor_loss.backward()
            self.agent.actor_optimizer[agent_idx].step()

            if global_step % 100 == 0:
                temperature = self.agent.critic[agent_idx].g2anet.temperature
                hard_gates = attn_info['hard_gates']
                soft_weights = attn_info['soft_weights']
                self.writer.add_scalar(f'gaac_{agent_idx}/mean_hard_gate', hard_gates.mean().item(), global_step)
                self.writer.add_scalar(f'gaac_{agent_idx}/temperature', temperature, global_step)
                self.writer.add_scalar(f'gaac_{agent_idx}/mean_soft_weight', soft_weights.mean().item(), global_step)
                soft_weights_entropy = -(soft_weights * torch.log(soft_weights + 1e-8)).sum(dim=-1).mean().item()
                self.writer.add_scalar(f'gaac_{agent_idx}/soft_weights_entropy', soft_weights_entropy, global_step)
                self.writer.add_image(f'gaac_{agent_idx}/soft_weights_heatmap', soft_weights.mean(dim=0).unsqueeze(0).unsqueeze(0), global_step)
                self.log_stats(global_step, start_time, agent_idx, q_value, None, critic_loss, None, critic_loss, actor_loss)

            # Update the target networks
            self.agent.update_target_networks(agent_idx, self.args.tau)

    def normalize_buffer_obs(self, observations: list, next_observations: list) -> tuple[list, list]:
        """
        Normalize the observations and next observations from the replay buffer.

        Args:
            observations (list): List of observations for each agent.
            next_observations (list): List of next observations for each agent.

        Returns:
            tuple[list, list]: Normalized observations and next observations.
        """
        # If normalize_obs_output is True, the environment already returns normalized observations
        if self.args.normalize_obs_output:
            return observations, next_observations

        n_agents = self.agent.num_agents
        if IS_GRAPH_BASED[self.args.algorithm]:
            # NOTE: This may not be working with latest changes, anyway, we're not using it
            batch_size = len(observations[0])
            obs_stacked = np.empty((batch_size, n_agents), dtype=object)
            next_obs_stacked = np.empty((batch_size, n_agents), dtype=object)
            for i in range(n_agents):
                obs_stacked[:, i] = observations[i]
                next_obs_stacked[:, i] = next_observations[i]
            obs_stacked = self.normalize_obs(obs_stacked)
            next_obs_stacked = self.normalize_obs(next_obs_stacked)
            observations = [obs_stacked[:, i] for i in range(n_agents)]
            next_observations = [next_obs_stacked[:, i] for i in range(n_agents)]
        else:
            obs_np = np.stack([obs.cpu().numpy() for obs in observations], axis=1)
            next_obs_np = np.stack([next_obs.cpu().numpy() for next_obs in next_observations], axis=1)
            obs_np = self.normalize_obs(obs_np)
            next_obs_np = self.normalize_obs(next_obs_np)
            observations = [torch.tensor(obs_np[:, i], dtype=torch.float32).to(self.device)
                            for i in range(n_agents)]
            next_observations = [torch.tensor(next_obs_np[:, i], dtype=torch.float32).to(self.device)
                                 for i in range(n_agents)]
        return observations, next_observations

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize the observations using the running mean and std from the environment.

        Args:
            obs (np.ndarray): Observations to normalize.

        Returns:
            np.ndarray: Normalized observations.
        """
        # If normalize_obs_output is True, the environment already returns normalized observations
        if self.args.normalize_obs_output:
            return obs
        self.env.update_obs_rms = False
        self.env.normalize_obs_output = True
        normalized_obs = self.env._extract_and_normalize_continuous_features(obs)
        self.env.update_obs_rms = True
        self.env.normalize_obs_output = False
        return normalized_obs

    def sample_actions(self, obs: np.ndarray, global_step: int) -> list[np.ndarray]:
        """
        Sample actions from the agent.
        
        If the global step is less than the learning starts, sample random actions.
        
        Args:
            obs (np.array): Observations from the environment.
            global_step (int): Current global step.
            
        Returns:
            actions (list[np.ndarray]): Actions sampled from the agent for each environment.
        """
        if global_step < self.args.learning_starts:
            actions = np.array([[self.env.action_space[0][agent_i].sample() for agent_i in range(self.agent.num_agents)]
                            for _ in range(self.env.n_envs)], dtype=object)
        else:
            with torch.no_grad():
                obs = self.normalize_obs(obs)
                if IS_GRAPH_BASED[self.args.algorithm]:
                    torch_obs = [np.stack([env_obs['services_graph']['x'] for env_obs in obs[:, i]])
                                 for i in range(self.agent.num_agents)]
                else:
                    torch_obs = [torch.tensor(np.vstack(obs[:, i]), dtype=torch.float32).to(self.device)
                                 for i in range(self.agent.num_agents)]
                agent_actions, _, _ = self.agent.get_action(torch_obs)
                agent_actions = [action.cpu().detach().numpy() for action in agent_actions]
                # rearrange actions to be per environment
                actions = [[ac[i] for ac in agent_actions] for i in range(self.env.n_envs)]
        return actions

    def step_and_push(self, obs: np.ndarray, agent_actions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Step the environment and push the data to the replay buffer.
        
        Args:
            obs (np.array): Observations from the environment.
            agent_actions (list[np.ndarray]): Actions sampled from the agent.
            global_step (int): Current global step.
            
        Returns:
            tuple[np.array, np.array]: Next observations and terminations from the environment.
        """
        next_obs, rewards, terminations, truncations, infos = self.env.step(agent_actions)
        # rearrange actions to be per agent
        actions = [[ac[i] for ac in agent_actions] for i in range(self.agent.num_agents)]
        self.replay_buffer.push(obs, actions, rewards, next_obs, terminations)
        return next_obs, terminations
    
    def set_seed(self, seed: int) -> None:
        """
        Set the seed for reproducibility of the experiment.
        
        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
