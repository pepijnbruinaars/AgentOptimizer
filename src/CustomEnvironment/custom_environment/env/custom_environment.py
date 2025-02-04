from pettingzoo import ParallelEnv
from gymnasium import spaces
from typing import TypedDict
import pandas as pd
import numpy as np

from .case import Case


class SimulationParameters(TypedDict):
    start_timestamp: pd.Timestamp


class CustomEnvironment(ParallelEnv):
    """ParallelEnv means that each agent acts simultaneously."""

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, data, simulation_parameters: SimulationParameters) -> None:
        super().__init__()
        self.data = data

        # Initialize the simulation time and event queue
        self.current_time = simulation_parameters["start_timestamp"]
        self.event_queue: list[Case] = []

        # Initialize the agents
        self.agents: list[str] = sorted(set(self.data["resource"]))
        self.agents_busy_until = {
            key: simulation_parameters["start_timestamp"] for key in self.agents
        }

    def _advance_time(self):
        # Advance time to the next event timestamp if available, or by a fixed delta.
        if (
            self.event_queue
            and self.event_queue[0].current_timestamp > self.current_time
        ):
            self.current_time = self.event_queue[0].current_timestamp
        else:
            self.current_time += pd.Timedelta(seconds=60)

    def _compute_reward(self):
        # Compute the reward based on the current state of the simulation.
        return 0

    def step(self, actions):
        pass
