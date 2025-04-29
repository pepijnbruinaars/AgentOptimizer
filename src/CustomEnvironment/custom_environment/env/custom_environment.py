from gymnasium.spaces import Discrete, MultiBinary, Box, Dict as GymDict
from pettingzoo import ParallelEnv  # type: ignore
import numpy as np

from typing import TypedDict
import pandas as pd

from env_config import debug_print_colored


from .reward import get_reward
from .objects import Case, Task, ResourceAgent, Status
from display import display_indented_list
from .constants import MAX_TASKS_PER_AGENT


class SimulationParameters(TypedDict):
    start_timestamp: pd.Timestamp


class AgentOptimizerEnvironment(ParallelEnv):
    """The environment representing the business process."""

    metadata = {
        "name": "agent_optimizer_environment_v0",
    }

    def __init__(
        self, data: pd.DataFrame, simulation_parameters: SimulationParameters
    ) -> None:
        super().__init__()
        print("Initializing environment...")
        # Store the data
        self.data: pd.DataFrame = data

        # Total number of steps and epochs
        self.steps: int = 0
        self.epochs: int = 0
        self.max_steps: int = 100_000
        self.max_episodes: int = 1000

        # Initialize the simulation time and event queue
        self.num_activities: int = len(set(self.data["activity_name"]))
        self.task_dict: dict[str, int] = {
            task: i for i, task in enumerate(sorted(set(self.data["activity_name"])))
        }
        self.current_time: pd.Timestamp = simulation_parameters["start_timestamp"]
        self.future_cases: list[Case] = self._initialize_future_cases()
        self.pending_cases: list[Case] = []
        self.completed_cases: list[Case] = []
        self.upcoming_case: Case | None = None

        # Initialize the agents
        self.resources: list[str] = sorted(set(self.data["resource"]))
        self.resource_dict: dict[str, int] = {
            resource: i for i, resource in enumerate(self.resources)
        }
        self.agents: list[ResourceAgent] = [
            ResourceAgent(
                self.resource_dict[resource],
                {
                    task_id: lambda: np.random.randint(1, 10)
                    for task_id in range(self.num_activities)
                },
            )
            for resource in self.resources
        ]

        print(f"Environment initialized. Start time: {self.current_time}")
        display_indented_list(self.agents, "Agents")
        print(f"# future cases: {len(self.future_cases)}")
        print(f"# tasks to be performed: {len(self.data)}")
        print("---------" * 8)

    def _initialize_future_cases(self) -> list[Case]:
        """Function that initializes the future cases with the first event of each case in the data."""
        future_cases: list[Case] = []
        # Group the data by case_id, and iterate over each case
        grouped_and_sorted = self.data.sort_values("start_timestamp").groupby("case_id")
        for case_id, case_data in grouped_and_sorted:
            # Start timestamp of a case is the earliest timestamp of the case
            start_timestamp = case_data["start_timestamp"].min()
            future_tasks = case_data["activity_name"].tolist()
            future_tasks.sort(key=lambda x: self.task_dict[x])
            case = Case(
                int(str(case_id)),
                start_timestamp,
                [
                    Task(self.task_dict[task], int(str(case_id)))
                    for task in future_tasks
                ],
            )

            future_cases.append(case)

        future_cases.sort(key=lambda x: x.assigned_timestamp)

        return future_cases

    def _filter_completed_cases(self) -> None:
        """Filter out completed cases from pending_cases and move them to completed_cases."""
        completed = []
        remaining = []

        for case in self.pending_cases:
            if case.status == Status.COMPLETED:
                completed.append(case)
            else:
                remaining.append(case)
        debug_print_colored(f"Completed: {len(completed)}, remaining: {len(remaining)}")
        debug_print_colored(
            f"Remaining case: {remaining[0]}" if remaining else "No remaining cases"
        )
        # Update the lists
        self.completed_cases.extend(completed)
        self.pending_cases = remaining

        if completed:
            debug_print_colored(
                f"Moved {len(completed)} completed cases to completed_cases list",
                "green",
            )

    def step(self, actions: dict[int, int]) -> tuple[dict, dict, dict, dict, dict]:
        """Execute one step of the environment's dynamics."""
        self.steps += 1
        self._filter_completed_cases()

        ### -------------------------------- ###
        ### HANDLE ACTION FROM CURRENT STEP  ###
        ### -------------------------------- ###
        if self.upcoming_case is not None:
            # Check which agents volunteered for the task
            available_agents = [
                agent_id for agent_id, action in actions.items() if action == 1
            ]

            # If nobody volunteered, select all agents as available
            if not available_agents:
                available_agents = [agent.id for agent in self.agents]

            # Select a random agent from volunteers
            selected_agent_id = np.random.choice(available_agents)
            selected_agent = self.agents[selected_agent_id]

            # Assign the case to the selected agent
            debug_print_colored(f"Upcoming case: {self.upcoming_case}")
            self.upcoming_case.assign_to_agent(selected_agent, self.current_time)
            # Reset upcoming_case to None after assignment to prevent duplicates
            self.upcoming_case = None

        # Check if any two agents have the same current_case
        for agent in self.agents:
            if agent.current_case is not None:
                for other_agent in self.agents:
                    if (
                        agent.id != other_agent.id
                        and agent.current_case == other_agent.current_case
                    ):
                        debug_print_colored(
                            f"Conflict between agents {agent.id} and {other_agent.id}",
                            "red",
                        )

        ### ------------------------------- ###
        ### CHECK COMPLETED TASKS/CASES     ###
        ### ------------------------------- ###
        debug_print_colored(f"Active cases: {len(self.pending_cases)}")
        for agent in self.agents:
            debug_print_colored(agent, "yellow")
        for agent in self.agents:
            case_is_done = agent.work_case(self.current_time)
            if agent.busy_until and agent.busy_until <= self.current_time:
                agent.busy_until = None

        # Filter out completed cases from pending cases
        self._filter_completed_cases()

        ### -------------------------------- ###
        ###    ADVANCE TIME TO NEXT EVENT    ###
        ### -------------------------------- ###
        self.current_time = self._get_next_time()

        ### ------------------------------- ###
        ### CHECK IF SIMULATION SHOULD STOP ###
        ### ------------------------------- ###
        # Truncations specify when to stop based on training constraints
        truncations = {agent.id: self.steps >= self.max_steps for agent in self.agents}

        # Terminations specify when to stop based on reaching terminal state
        terminations = {
            agent.id: len(self.future_cases) == 0 and len(self.pending_cases) == 0
            for agent in self.agents
        }

        # Compute reward
        reward = get_reward(self)
        rewards = {agent.id: reward for agent in self.agents}

        # Return early if simulation should stop
        if any(terminations.values()) or any(truncations.values()):
            return {}, rewards, terminations, truncations, {}

        ### --------------------------------------- ###
        ### DETERMINE CASE FOR NEXT SIMULATION STEP ###
        ### --------------------------------------- ###

        self.upcoming_case = None

        # Check if there are any pending cases that are not already being handled by agents
        if self.pending_cases:
            # Create a set of cases already being handled by any agent
            handled_cases = set()
            for agent in self.agents:
                if agent.current_case:
                    handled_cases.add(agent.current_case)
                for i in range(agent.case_queue.size()):
                    # Peek at cases in the queue without removing them
                    case = agent.case_queue.peek(i)
                    if case:
                        handled_cases.add(case)

            # Find first pending case that is ready to be worked on and not already handled
            for case in self.pending_cases:
                if case not in handled_cases and case.is_eligible_for_next_task(
                    self.current_time
                ):
                    self.upcoming_case = case
                    break

        # If no pending cases, check if there are future cases
        if self.upcoming_case is None and self.future_cases:
            # Assign the next future case to the upcoming case
            self.upcoming_case = self.future_cases.pop(0)
            self.pending_cases.append(self.upcoming_case)

        ### ------------------------------- ###
        ###      PREPARE OBSERVATIONS       ###
        ### ------------------------------- ###
        observations = {
            agent.id: {
                "task_id": (
                    self.upcoming_case.current_task.id
                    if self.upcoming_case and self.upcoming_case.current_task
                    else 0
                ),
                "agent_id": agent.id,
                "current_time": int(self.current_time.timestamp()),
                "agents_capable_tasks": np.array(
                    []
                ),  # This could be populated based on agent skills
                "agents_task_queue": np.array(
                    [case.id for case in agent.case_queue], dtype=np.int32
                ),
                "agent_is_busy": np.array([agent.is_busy()], dtype=np.int8),
            }
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, {}

    def render(self) -> None:
        """Renders the environment."""
        print("\n--- Environment State ---")
        print(f"Time: {self.current_time}, Step: {self.steps}")
        display_indented_list(self.agents, "Agents")
        display_indented_list(
            self.pending_cases[:5], f"Pending Cases ({len(self.pending_cases)})"
        )
        if len(self.pending_cases) > 5:
            print("  ...")
        if self.future_cases:
            print(f"  Next arrival: {self.future_cases[0].assigned_timestamp}")
        display_indented_list(
            self.future_cases[:5], f"Future Cases ({len(self.future_cases)})"
        )
        if len(self.future_cases) > 5:
            print("  ...")
        print(f"Completed Cases: {len(self.completed_cases)}")
        print("--- End State ---")

    def observation_space(self, agent: int) -> GymDict:
        """Returns the observation space for a single agent."""
        return GymDict(
            {
                "task_id": Discrete(self.num_activities),
                "agent_id": Discrete(len(self.agents)),
                "current_time": Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                "agents_capable_tasks": Box(
                    0,
                    self.num_activities - 1,
                    shape=(self.num_activities,),
                    dtype=np.int32,
                ),
                "agents_task_queue": Box(
                    0,
                    self.num_activities - 1,
                    shape=(MAX_TASKS_PER_AGENT,),
                    dtype=np.int32,
                ),
                "agent_is_busy": MultiBinary(1),
            }
        )

    def action_space(self, agent: int) -> Discrete:
        """Returns the action space for a single agent."""
        return Discrete(2)

    def _get_next_time(self) -> pd.Timestamp:
        """Get the next time where action is needed."""
        # Find next event time (either a task completion or new case arrival)
        next_event_times: list[pd.Timestamp] = []

        # Add agent task completion times
        for agent in self.agents:
            if agent.busy_until is not None:
                next_event_times.append(agent.busy_until)

        # Add task completion timestamps from pending cases
        if len(self.pending_cases) > 0:
            task_completion_times = [
                case.current_task.completion_timestamp
                for case in self.pending_cases
                if case.current_task and case.current_task.completion_timestamp
            ]
            # Only add times that are in the future
            task_completion_times = [
                time for time in task_completion_times if time > self.current_time
            ]
            next_event_times.extend(task_completion_times)

        # Add next case arrival time from future cases
        if len(self.future_cases) > 0:
            # Only add the arrival time if it's in the future
            arrival_time = self.future_cases[0].assigned_timestamp
            if arrival_time > self.current_time:
                next_event_times.append(arrival_time)

        # If there are events, advance time to the closest one
        if next_event_times:
            next_time = min(next_event_times)
            # Safety check: ensure we always move forward in time
            if next_time <= self.current_time:
                debug_print_colored(
                    f"⚠️ Time not progressing. Forcing small time increment.",
                    "red",
                )
                # Force a small time increment
                return self.current_time + pd.Timedelta(seconds=1)
            return next_time

        # If no events, return a fixed interval (fallback)
        return self.current_time
