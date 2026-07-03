import datetime
import numpy as np
import torch

from gymnasium import Env
from robin.rl.algorithms.constants import EVALUATOR_SEED_RANK_MULTIPLIER
from robin.rl.algorithms.utils import create_env

DEFAULT_CONFIG_DEMAND = 'configs/rl/demand_data_base.yaml'
DEFAULT_CONFIG_SUPPLY = 'configs/rl/supply_data_base.yaml'
MULTI_AGENT = True
COOPERATIVE = False
DEFAULT_NUM_STEPS = 70_000
SEED = 0
N_WORKERS = 16


def test_env(env: Env, num_steps: int = DEFAULT_NUM_STEPS) -> None:
    """
    Test the environment with a random agent.

    Args:
        env (Env): The environment to test.
        num_steps (int): The number of steps to run the environment.
    """
    observation, info = env.reset()
    rewards = []
    episodic_reward = 0

    for i in range(num_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episodic_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            rewards.append(episodic_reward)
            episodic_reward = 0

    env.close()
    print(f'Mean episodic reward (random agent): {np.mean(rewards)} +/- {np.std(rewards)}')


def test_multi_agent_env(env: Env, num_steps: int = DEFAULT_NUM_STEPS) -> None:
    """
    Test the multi-agent environment with random agents.

    Args:
        env (Env): The environment to test.
        num_steps (int): The number of steps to run the environment.
    """
    num_agents = env.get_env_attr('num_agents')[0]
    observation, info = env.reset()
    rewards = [[] for _ in range(num_agents)]
    episodic_reward = np.zeros((env.n_envs, num_agents), dtype=np.float32)

    for i in range(0, num_steps, env.n_envs):
        actions = np.array([[env.action_space[0][agent_i].sample() for agent_i in range(num_agents)]
                            for _ in range(env.n_envs)], dtype=object)
        observation, reward, terminated, truncated, info = env.step(actions)
        episodic_reward += reward

        if terminated.all() or truncated.all():
            observation, info = env.reset()
            for i in range(num_agents):
                rewards[i].append(episodic_reward[:, i])
            episodic_reward = np.zeros((env.n_envs, num_agents), dtype=np.float32)

    print(f'Mean episodic reward (random agents): {np.mean(np.sum(rewards, axis=0))} +/- {np.std(np.sum(rewards, axis=0))}')
    for i, agent in enumerate(env.get_env_attr('agents')[0]):
        print(f'{agent} - Mean episodic reward (random agent): {np.mean(rewards[i])} +/- {np.std(rewards[i])}')
    torch.save(env.obs_rms, f'obs_rms.pth')
    env.close()


if __name__ == '__main__':
    output_dir = 'models'
    exp_name = 'base'
    algorithm = 'random'
    seed = SEED
    now = datetime.datetime.now().strftime('%d%m%y-%H%M%S')
    run_name = f'{output_dir}/{exp_name}/{algorithm}/{seed}/{now}'
    env = create_env(
        supply_config=DEFAULT_CONFIG_SUPPLY,
        demand_config=DEFAULT_CONFIG_DEMAND,
        algorithm='rache',
        seed=SEED,
        seed_rank_multiplier=EVALUATOR_SEED_RANK_MULTIPLIER,
        n_workers=N_WORKERS,
        run_name=run_name,
        normalize_obs=True,
        normalize_obs_output=True,
        reward_scale_factor=1.0
    )
    print(f'Number of services: {len(env.get_env_attr("kernel")[0].supply.services)}')
    print(env.observation_space[0])
    print(env.action_space[0])
    
    if MULTI_AGENT:
        test_multi_agent_env(env)
    else:
        test_env(env)
