"""Noise generators for RL algorithms."""

import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck noise process.
    
    Used for exploration in continuous action spaces, particularly with deterministic policies like DDPG/MADDPG.
    
    Reference: https://github.com/songrotek/DDPG/blob/master/ou_noise.py
    """
    
    def __init__(
        self,
        action_dimension: int,
        scale: float = 0.1,
        mu: float = 0,
        theta: float = 0.15,
        sigma: float = 0.2
    ) -> None:
        """
        Initialize the Ornstein-Uhlenbeck noise process.
        
        Args:
            action_dimension (int): Dimension of the action space.
            scale (float): Scaling factor for the noise.
            mu (float): Long-running mean of the process.
            theta (float): Mean reversion rate.
            sigma (float): Volatility parameter.
        """
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self) -> None:
        """
        Reset the noise process to its initial state.
        """
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self) -> np.ndarray:
        """
        Generate noise sample.
        
        Returns:
            np.ndarray: Noise sample with shape (action_dimension,).
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
