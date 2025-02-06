from gymnasium.spaces import MultiDiscrete
from pettingzoo import ParallelEnv  # type: ignore

from typing import TypedDict
import pandas as pd

from .data_handling import (
    find_first_case_activity,
)

from .case import Case


class SimulationParameters(TypedDict):
    start_timestamp: pd.Timestamp


class ResourceAgent:
    def __init__(self, resource: int) -> None:
        self.resource: int = resource
        self.is_busy: bool = False
        self.is_busy_until: pd.Timestamp | None = None


class CustomEnvironment(ParallelEnv):
    """ParallelEnv means that each agent acts simultaneously."""

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(
        self, data: pd.DataFrame, simulation_parameters: SimulationParameters
    ) -> None:
        super().__init__()
        # Store the data
        self.data: pd.DataFrame = data

        # Total number of steps and epochs
        self.steps: int = 0
        self.epochs: int = 0

        # Initialize the simulation time and event queue
        self.num_activities: int = len(set(self.data["activity_name"]))
        self.current_time: pd.Timestamp = simulation_parameters["start_timestamp"]
        self.future_cases: list[Case] = self._initialize_future_cases()
        self.pending_cases: list[Case] = []
        self.past_cases: list[Case] = []

        # Initialize the agents
        self.resources: list[str] = sorted(set(self.data["resource"]))
        self.agents: list[int] = list(range(len(self.resources)))
        self.agents_busy_until: dict[int, pd.Timestamp] = {
            key: simulation_parameters["start_timestamp"] for key in self.agents
        }

    def _initialize_future_cases(self) -> list[Case]:
        """Function that initializes the future cases with the first event of each case in the data.

        Returns:
            list[Case]: List of pending cases
        """
        future_cases: list[Case] = []

        # Group the data by case_id and iterate over each case
        for case_id, case_data in self.data.groupby("case_id"):
            # Start timestamp of a case is the earliest timestamp of the case
            start_timestamp = case_data["start_timestamp"].min()
            case = Case(str(case_id), start_timestamp)

            future_cases.append(case)

        return future_cases

    def _advance_time(self) -> None:
        """Advance the time to the next event in the data."""
        if (
            len(self.pending_cases) > 0
            and self.current_time > self.pending_cases[0].current_timestamp
        ):
            self.current_time = self.pending_cases[0].current_timestamp
        else:
            self.current_time += pd.Timedelta(seconds=60)

    def _compute_reward(self) -> float:
        # Compute the reward based on the current state of the simulation.
        return 0

    def step(self, actions) -> tuple[dict, dict, dict, dict, dict]:
        # Increment step counter
        self.steps += 1

        # First we perform the actions as determined before this step
        for agent, action in actions.items():
            print(f"Agent {agent} performs action {action}")

        # Sort cases by timestamp
        self.pending_cases.sort(key=lambda x: x.current_timestamp)
        self.future_cases.sort(key=lambda x: x.current_timestamp)

        # If there are no active cases, but there are cases that can still be handled in the future
        if len(self.pending_cases) < 1 and len(self.future_cases) > 0:
            # Load the first activity of the case from data
            case = self.future_cases.pop(0)
            self.pending_cases.append(case)

        # Loop over each pending case and check if the case has a next activity
        for case in self.pending_cases:
            last_activity = case.get_last_activity()
            if last_activity is None:
                # We load the first activity of the case from data
                case.add_activity(
                    find_first_case_activity(self.data, int(case.case_id))
                )
                print(case)
            else:
                # In this case, agent will select next activity and who to handoff to including themselves
                print(f"Case {case} has activity {last_activity}")

        # After, we advance time and perform the step as defined in AgentSim
        print("Agents: ", self.agents)
        self._advance_time()

        # Check terminal conditions

        return {}, {}, {}, {}, {}

    def action_space(self, agent: int) -> MultiDiscrete:
        """The action space for each agent. It is a composite space of the number of agents and the number of activities.
        This means that the agent selects a new activity that should be performed after the current activity, and also who should perform it.

        Args:
            agent (int): The agent ID

        Returns:
            MultiDiscrete: The action space for the agent
        """
        return MultiDiscrete([self.num_agents, self.num_activities])
