from pettingzoo import ParallelEnv
from gymnasium import spaces


class CustomEnvironment(ParallelEnv):
    """ParallelEnv means that each agent acts simultaneously."""

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
