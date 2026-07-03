"""Evaluator module for evaluating the performance of a policy."""

import numpy as np
import os
import torch
import tyro

from dataclasses import dataclass
from typing import Literal

from robin.rl.algorithms.constants import EVALUATOR_SEED_RANK_MULTIPLIER, IS_GRAPH_BASED
from robin.rl.algorithms.trainer import ALGORITHM_REGISTRY, AlgorithmName
from robin.rl.algorithms.utils import create_env


@dataclass
class EvaluatorArgs:
    algorithm: Literal['maddpg', 'matd3', 'rache', 'gaac'] = 'rache'
    """the algorithm to use"""
    input_dir: str = '<output_dir>/<algorithm>/<seed>/<date>'
    """the input directory to load the model from"""
    total_timesteps: int = 20_000
    """number of timesteps to train the agent"""
    cuda: bool = True
    """whether to use cuda"""
    seed: int = 0
    """seed of the experiment"""
    n_workers: int = 16
    """number of workers for evaluation (it should be the number of cores)"""
    exclude_edge_type: int | None = None
    """edge type to exclude for ablation studies (0=same_market, 1=same_agent, 2=dest_origin). None keeps all relation types."""


class Evaluator:
    """
    Evaluator class for evaluating the performance of a policy.

    Attributes:
        args (EvaluatorArgs): Arguments for the evaluator.
        device (torch.device): Device to run the evaluator on.
        model_path (str): Path to the model to load.
        evaluation_path (str): Path to save the evaluation results.
        env (StatsSubprocVectorEnv): The environment to evaluate the agent on.
        agent (AlgorithmName): The agent to train.
        episode_length (int): The length of the episode.
        n_episodes (int): The number of episodes to evaluate.
        policy_distribution (np.array): The distribution of actions taken by the agent.
    """

    def __init__(self) -> None:
        """
        Initialize the evaluator.
        
        It uses the arguments from the command line to initialize the evaluator,
        please refer to the EvaluatorArgs class.
        """
        self.args = tyro.cli(EvaluatorArgs)
        self.device = torch.device('cuda' if self.args.cuda and torch.cuda.is_available() else 'cpu')
        self.model_path = f'{self.args.input_dir}/model.pt'
        self.evaluation_path = f'{self.args.input_dir}/evaluation'
        self.obs_rms_path = f'{self.args.input_dir}/obs_rms.pth'
        self.training_used_normalization = os.path.exists(self.obs_rms_path)
        self.env = create_env(
            supply_config=f'{self.args.input_dir}/supply_config.yaml',
            demand_config=f'{self.args.input_dir}/demand_config.yaml',
            algorithm=self.args.algorithm,
            seed=self.args.seed,
            seed_rank_multiplier=EVALUATOR_SEED_RANK_MULTIPLIER,
            n_workers=self.args.n_workers,
            run_name=self.evaluation_path,
            normalize_obs=self.training_used_normalization,
            normalize_obs_output=self.training_used_normalization,
            reward_scale_factor=1.0,
            exclude_edge_type=self.args.exclude_edge_type
        )
        if self.training_used_normalization:
            self.env.set_obs_rms(torch.load(self.obs_rms_path))
            self.env.update_obs_rms = False
        self.agent = self.load_model(self.args.algorithm, self.model_path)
        self.agent.eval()
        self.episode_length = len(self.env.get_env_attr('simulation_days')[0])
        # Initialize the policy distribution, shape: (num_agents, n_actions, n_episodes, episode_length)
        self.n_episodes = self.args.total_timesteps // self.episode_length
        self.policy_distribution = np.array(
            [np.zeros((acsp.shape[0], self.n_episodes, self.episode_length), dtype=np.float32)
             for acsp in self.env.action_space[0]],
            dtype=object
        )
        # Initialize embedding collection for graph-based algorithms
        if IS_GRAPH_BASED[self.args.algorithm]:
            num_agents = self.env.get_env_attr('num_agents')[0]
            self.graph_embeddings = [[] for _ in range(num_agents)]
            self.rewards_metadata = [[] for _ in range(num_agents)]

    def evaluate(self) -> None:
        """
        Evaluate the policy using the given model.
        """
        obs, _ = self.env.reset()

        for global_step in range(0, self.args.total_timesteps, self.env.n_envs):
            # Sample actions and optionally get embeddings
            if IS_GRAPH_BASED[self.args.algorithm]:
                actions, embeddings = self.sample_actions(obs, return_embeddings=True)
            else:
                actions = self.sample_actions(obs)

            self.update_policy_distribution(actions, global_step)
            next_obs, rewards, terminations, _, _ = self.env.step(actions)

            # Store embeddings and metadata for graph-based algorithms
            if IS_GRAPH_BASED[self.args.algorithm]:
                self.update_embeddings_metadata(embeddings, rewards)

            # Update observations
            obs = next_obs
            if terminations.all():
                obs, _ = self.env.reset()

        # Save the policy distribution
        np.save(f'{self.evaluation_path}/policy_distribution.npy', self.policy_distribution)

        # Save embeddings and metadata for graph-based algorithms
        if IS_GRAPH_BASED[self.args.algorithm]:
            # Concatenate collected embeddings per agent
            embeddings_array = [np.vstack(agent_embs) for agent_embs in self.graph_embeddings]
            rewards_array = [np.concatenate(agent_rews) for agent_rews in self.rewards_metadata]
            np.save(f'{self.evaluation_path}/graph_embeddings.npy', embeddings_array)
            np.save(f'{self.evaluation_path}/rewards_metadata.npy', rewards_array)

    def sample_actions(self, obs: np.ndarray, return_embeddings: bool = False):
        """
        Sample actions from the agent.

        If the global step is less than the learning starts, sample random actions.

        Args:
            obs (np.array): Observations from the environment.
            return_embeddings (bool): Whether to return embeddings (only for graph-based algorithms).

        Returns:
            actions (list[np.ndarray]): Actions sampled from the agent for each environment.
            embeddings (list[torch.Tensor], optional): Graph embeddings if return_embeddings=True and algorithm is graph-based.
        """
        with torch.no_grad():
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
            if return_embeddings and IS_GRAPH_BASED[self.args.algorithm]:
                embeddings = self.agent.get_embeddings(torch_obs)
                return actions, embeddings
        return actions

    def load_model(self, algorithm: str, model_path: str) -> AlgorithmName:
        """
        Load the model from the given path.

        Args:
            algorithm (str): The algorithm to use.
            model_path (str): Path to the model to load.

        Returns:
            AlgorithmName: The loaded model.
        """
        if algorithm not in ALGORITHM_REGISTRY:
            raise Exception(f'Algorithm {algorithm} is not supported. Please choose from {list(ALGORITHM_REGISTRY.keys())}.')
        return ALGORITHM_REGISTRY[algorithm].load_model(model_path, self.env, self.device)

    def update_embeddings_metadata(self, embeddings: list[torch.Tensor], rewards: np.ndarray) -> None:
        """
        Store graph embeddings and rewards for each agent.

        Args:
            embeddings (list[torch.Tensor]): Embeddings sampled from the agent.
            rewards (np.ndarray): Rewards received from the environment.
        """
        for agent_idx in range(self.agent.num_agents):
            self.graph_embeddings[agent_idx].append(embeddings[agent_idx].cpu().numpy())
            self.rewards_metadata[agent_idx].append(rewards[:, agent_idx])

    def update_policy_distribution(self, actions: list[np.ndarray], global_step: int) -> None:
        """
        Update the policy distribution with the agent actions.

        Args:
            actions (list[np.ndarray]): Actions sampled from the agent.
            global_step (int): The global step of the training.
        """
        # rearrange actions to be per agent
        agents_actions = [[ac[i] for ac in actions] for i in range(self.agent.num_agents)]
        episode = (global_step // self.env.n_envs // self.episode_length) * self.env.n_envs
        timestep = global_step % self.episode_length
        for agent_id, agent_actions in enumerate(agents_actions):
            stacked_actions = np.vstack(agent_actions).transpose()
            self.policy_distribution[agent_id][:, episode:episode+self.env.n_envs, timestep] = \
                stacked_actions
