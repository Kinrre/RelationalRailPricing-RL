"""Utils for the RL module."""
from tianshou.env.worker import EnvWorker

# Monkey patch the seed method of the EnvWorker class
# to support multiple action spaces in the environment
def seed(self, seed: int | None = None) -> list[int] | None:
    if isinstance(self.action_space, list):
        return [space.seed(seed) for space in self.action_space]
    return self.action_space.seed(seed)

EnvWorker.seed = seed
