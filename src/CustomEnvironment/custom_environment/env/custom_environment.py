import os
from gymnasium.spaces import Discrete, MultiBinary, Box, Dict as GymDict
from pettingzoo import ParallelEnv  # type: ignore
import numpy as np

from typing import TypedDict, Optional, Mapping, Dict
import pandas as pd

from env_config import debug_print_colored  # type: ignore


from .reward import get_reward
from .objects import Case, Task, ResourceAgent, Status
from display import display_indented_list  # type: ignore
from .data_handling import (
    compute_activity_duration_distribution_per_agent,
    compute_global_activity_medians,
)
from .duration_distribution import DurationDistribution
from .constants import MAX_TASKS_PER_AGENT


class SimulationParameters(TypedDict):
    start_timestamp: pd.Timestamp


class AgentOptimizerEnvironment(ParallelEnv):
    """The environment representing the business process."""

    metadata = {
        "name": "agent_optimizer_environment_v0",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        simulation_parameters: SimulationParameters,
        experiment_dir: str | None = None,
        pre_fitted_distributions: Optional[
            tuple[
                Mapping[str, Mapping[str, Optional[DurationDistribution]]],
                Mapping[str, Mapping[str, Optional[Dict[str, float]]]],
                Dict[str, float],
            ]
        ] = None,
    ) -> None:
        super().__init__()
        print("Initializing environment...")
        self.data: pd.DataFrame = data

        # Set up logging directory
        if experiment_dir:
            self.log_dir = os.path.join(experiment_dir, "logs")
        else:
            self.log_dir = "data/logs"

        # check that log_dir exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        current_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.log_file: str = os.path.join(self.log_dir, f"log_{current_timestamp}.csv")

        # CSV logging buffer - batch writes for performance
        self.csv_buffer: list[list] = []
        self.csv_buffer_size: int = 100  # Flush every 100 tasks
        self.csv_header_written: bool = False

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
        self.inv_task_dict: dict[int, str] = {
            i: task for i, task in enumerate(sorted(set(self.data["activity_name"])))
        }
        self.current_time: pd.Timestamp = simulation_parameters["start_timestamp"]
        self.future_cases: list[Case] = self._initialize_future_cases()
        self.pending_cases: list[Case] = []
        self.completed_cases: list[Case] = []
        self.upcoming_case: Case | None = None
        self.completed_task: Task | None = None

        # Initialize the agents
        self.resources: list[str] = sorted(set(self.data["resource"]))
        self.resource_dict: dict[str, int] = {
            resource: i for i, resource in enumerate(self.resources)
        }

        # Fit distributions for each agent and activity
        if pre_fitted_distributions is not None:
            # Use pre-fitted distributions (fitted on training data)
            activity_durations_dict, stats_dict, global_activity_medians = (
                pre_fitted_distributions
            )
            self._use_pre_fitted = True
            print("Using pre-fitted duration distributions from training data")
        else:
            # Fit distributions on the current data (original behavior)
            activity_durations_dict, stats_dict = (
                compute_activity_duration_distribution_per_agent(self.data)
            )
            # Compute global historical medians for each activity (across all agents)
            global_activity_medians = compute_global_activity_medians(self.data)
            self._use_pre_fitted = False
            print("Fitting duration distributions on current dataset")

        # Cache distributions for reuse in reset() - avoid refitting every episode
        self._cached_activity_durations_dict = activity_durations_dict
        self._cached_stats_dict = stats_dict
        self._cached_global_activity_medians = global_activity_medians

        # Store global activity medians
        self.global_activity_medians = global_activity_medians

        # Transform the global medians to use task IDs instead of activity names
        self.global_task_medians = {
            self.task_dict[activity]: median
            for activity, median in self.global_activity_medians.items()
        }

        # Transform the distributions to use task IDs instead of activity names
        transformed_activity_durations_dict = {
            self.resource_dict[resource]: {
                self.task_dict[task]: activity_durations_dict[resource][task]
                for task in sorted(set(self.data["activity_name"]))
            }
            for resource in self.resources
        }

        self.agents: list[ResourceAgent] = [
            ResourceAgent(
                self.resource_dict[resource],
                name=resource,
                capabilities={
                    self.task_dict[task]: transformed_activity_durations_dict[
                        self.resource_dict[resource]
                    ][self.task_dict[task]]
                    for task in sorted(set(self.data["activity_name"]))
                },
                stats_dict=stats_dict[resource],  # type: ignore
            )
            for resource in self.resources
        ]

        # Set environment reference for all tasks and cases
        for case in self.future_cases:
            case.environment = self
            for task in case.all_tasks:
                task.environment = self

        print(f"Environment initialized. Start time: {self.current_time}")
        display_indented_list(self.agents, "Agents")
        print(f"# future cases: {len(self.future_cases)}")
        print(f"# tasks to be performed: {len(self.data)}")
        print("---------" * 8)

    def resource_name_to_id(self, resource_name: str) -> int:
        """Convert a resource name to its corresponding ID."""
        if resource_name in self.resource_dict:
            return self.resource_dict[resource_name]
        else:
            raise ValueError(
                f"Resource '{resource_name}' not found in resource dictionary."
            )

    def _initialize_future_cases(self) -> list[Case]:
        """Function that initializes the future cases with the first event of each case in the data."""
        future_cases: list[Case] = []
        # Group the data by case_id, and iterate over each case
        grouped_and_sorted = self.data.sort_values(
            ["start_timestamp", "end_timestamp"]
        ).groupby("case_id", sort=False)
        for case_id, case_data in grouped_and_sorted:
            ordered_case_data = case_data.sort_values(["start_timestamp", "end_timestamp"])
            # Start timestamp of a case is the earliest timestamp of the case
            start_timestamp = ordered_case_data["start_timestamp"].min()
            future_tasks = ordered_case_data["activity_name"].tolist()
            case = Case(
                int(str(case_id)),
                start_timestamp,
                [
                    Task(self.task_dict[task], int(str(case_id)))
                    for task in future_tasks
                ],
            )
            case.environment = self  # Set environment reference
            for task in case.all_tasks:
                task.environment = self  # Set environment reference for all tasks

            future_cases.append(case)

        future_cases.sort(key=lambda x: x.open_timestamp)

        return future_cases

    def _filter_completed_cases(self) -> None:
        """Filter out completed cases from pending_cases and move them to completed_cases."""
        # Separate completed and pending cases using list comprehension (more efficient than pop)
        completed = [case for case in self.pending_cases if case.status == Status.COMPLETED]
        self.pending_cases = [case for case in self.pending_cases if case.status != Status.COMPLETED]
        self.completed_cases.extend(completed)

        debug_print_colored(
            f"Completed: {len(self.completed_cases)}, remaining: {len(self.pending_cases)}"
        )
        if self.pending_cases:
            debug_print_colored(f"Remaining case: {self.pending_cases[0]}")
        else:
            debug_print_colored("No remaining cases")

    def _get_handled_cases(self) -> set[Case]:
        """Collect all cases currently handled by agents or queued for them."""
        handled_cases: set[Case] = set()
        for agent in self.agents:
            if agent.current_case:
                handled_cases.add(agent.current_case)
            for i in range(agent.case_queue.size()):
                case = agent.case_queue.peek(i)
                if case:
                    handled_cases.add(case)
        return handled_cases

    def _find_unhandled_pending_case(self, at_time: pd.Timestamp) -> Case | None:
        """Find the next pending case that is ready and not already handled."""
        handled_cases = self._get_handled_cases()
        for case in self.pending_cases:
            if case not in handled_cases and case.is_eligible_for_next_task(at_time):
                return case
        return None

    def _has_immediate_assignment_work(self, at_time: pd.Timestamp) -> bool:
        """Check if there is assignment work to do without advancing simulation time."""
        if self._find_unhandled_pending_case(at_time) is not None:
            return True
        return bool(self.future_cases and self.future_cases[0].open_timestamp <= at_time)

    def _flush_csv_buffer(self, force: bool = False) -> None:
        """Flush accumulated CSV data to file if buffer is full or forced."""
        if len(self.csv_buffer) < self.csv_buffer_size and not force:
            return

        if len(self.csv_buffer) == 0:
            return

        # Create DataFrame from buffer
        csv_df = pd.DataFrame(
            self.csv_buffer,
            columns=[
                "case_id",
                "case_nr_tasks",
                "case_open_time",
                "case_completed_time",
                "task_id",
                "task_assigned_time",
                "task_started_time",
                "task_completed_time",
                "task_agent_id",
            ],
        )

        # Write to file (write header only on first write)
        csv_df.to_csv(
            self.log_file,
            mode="a",
            header=not self.csv_header_written,
            index=False,
        )
        self.csv_header_written = True
        self.csv_buffer.clear()

    def step(self, actions: dict[int, int]) -> tuple[dict, dict, dict, dict, dict]:
        """Execute one step of the environment's dynamics."""
        self.steps += 1

        ### -------------------------------- ###
        ### HANDLE ACTION FROM CURRENT STEP  ###
        ### -------------------------------- ###
        if (
            self.upcoming_case is not None
            and self.upcoming_case.current_task is not None
        ):
            task_id = self.upcoming_case.current_task.id
            # Check which agents volunteered for the task
            available_agents = [
                agent_id
                for agent_id, action in actions.items()
                if action == 1
            ]

            # Filter available agents to only those who can perform the task
            available_agents = [
                agent_id
                for agent_id in available_agents
                if self.agents[agent_id].can_perform_task(task_id)
            ]

            # If nobody volunteered, select all capable agents as available
            if not available_agents:
                available_agents = [
                    agent.id for agent in self.agents if agent.can_perform_task(task_id)
                ]
            if not available_agents:
                raise ValueError(
                    f"No capable agents found for task {task_id} in case {self.upcoming_case.id}."
                )

            # Select a random agent from volunteers
            selected_agent_id = np.random.choice(available_agents)
            selected_agent = self.agents[selected_agent_id]

            debug_print_colored(f"Upcoming case: {self.upcoming_case}")
            # Assign the case to the selected agent
            self.upcoming_case.assign_to_agent(selected_agent, self.current_time)
            # Reset upcoming_case to None after assignment to prevent duplicates
            self.upcoming_case = None

        # Check if any two agents have the same current_case (optimized with set-based lookup)
        # This should never happen if assignment logic is correct, but kept for safety
        active_cases = {}
        for agent in self.agents:
            if agent.current_case is not None:
                if agent.current_case in active_cases:
                    debug_print_colored(
                        f"Conflict between agents {agent.id} and {active_cases[agent.current_case]}",
                        "red",
                    )
                else:
                    active_cases[agent.current_case] = agent.id

        ### ------------------------------- ###
        ### CHECK COMPLETED TASKS/CASES     ###
        ### ------------------------------- ###
        debug_print_colored(f"Active cases: {len(self.pending_cases)}")
        for agent in self.agents:
            debug_print_colored(agent, "yellow")
        for agent in self.agents:
            is_finished, finished_case = agent.work_case(self.current_time)
            if is_finished and finished_case:
                if finished_case.current_task:
                    self.completed_task = finished_case.current_task
                # If the case is finished, add all its tasks to the CSV buffer
                for task in finished_case.all_tasks:
                    self.csv_buffer.append(
                        [
                            finished_case.id,
                            len(finished_case.all_tasks),
                            finished_case.open_timestamp,
                            finished_case.completion_timestamp,
                            task.id,
                            task.assigned_timestamp,
                            task.start_timestamp,
                            task.completion_timestamp,
                            task.assigned_agent.id,  # type: ignore
                        ]
                    )
                # Flush buffer if it gets large
                self._flush_csv_buffer()
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

        # First prefer pending work that can be assigned immediately.
        self.upcoming_case = self._find_unhandled_pending_case(self.current_time)

        # Otherwise, bring in the next arrived future case.
        if (
            self.upcoming_case is None
            and self.future_cases
            and self.future_cases[0].open_timestamp <= self.current_time
        ):
            # Assign the next future case to the upcoming case
            self.upcoming_case = self.future_cases.pop(0)
            self.pending_cases.append(self.upcoming_case)

        ### ------------------------------- ###
        ###      PREPARE OBSERVATIONS       ###
        ### ------------------------------- ###
        observations = {
            agent.id: self._get_observations(agent) for agent in self.agents
        }

        return observations, rewards, terminations, truncations, {}

    def _get_observations(self, agent: ResourceAgent):
        # Get the upcoming task ID
        task_id = (
            self.upcoming_case.current_task.id
            if self.upcoming_case and self.upcoming_case.current_task
            else -1
        )

        # Build task queue - pre-allocate array
        task_queue = np.zeros(MAX_TASKS_PER_AGENT, dtype=np.int32)
        queue_size = min(agent.case_queue.size(), MAX_TASKS_PER_AGENT)
        for i in range(queue_size):
            case = agent.case_queue.peek(i)
            if case:
                task_queue[i] = case.id

        # Check if agent can perform the task
        agent_can_perform_task = (
            task_id >= 0
            and task_id in agent.capabilities
            and agent.capabilities[task_id] is not None
        )

        # Get task stats (cached lookup)
        task_stats = None
        if agent_can_perform_task and task_id >= 0:
            task_name = self.inv_task_dict[task_id]
            task_stats = agent.stats_dict.get(task_name)

        # Build observation dict with cached values
        return {
            "task_id": task_id,
            "task_duration_left": agent.task_duration(self.current_time),
            "agents_task_queue": task_queue,
            "upcoming_task_mean": (
                task_stats["mean"] if task_stats else -1
            ),
            "upcoming_task_median": (
                task_stats["median"] if task_stats else -1
            ),
            "upcoming_task_std": (
                task_stats["std"] if task_stats else -1
            ),
            "agent_is_capable_of_upcoming_task": np.array(
                [int(agent_can_perform_task)], dtype=np.int8
            ),
            "agent_is_busy": np.array([agent.is_busy()], dtype=np.int8),
        }

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
            print(f"  Next arrival: {self.future_cases[0].open_timestamp}")
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
                "task_duration_left": Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                "agents_task_queue": Box(
                    0,
                    self.num_activities - 1,
                    shape=(MAX_TASKS_PER_AGENT,),
                    dtype=np.int32,
                ),
                "upcoming_task_mean": Box(
                    0,
                    np.inf,
                    shape=(),
                    dtype=np.float64,
                ),
                "upcoming_task_median": Box(
                    0,
                    np.inf,
                    shape=(),
                    dtype=np.float64,
                ),
                "upcoming_task_std": Box(
                    0,
                    np.inf,
                    shape=(),
                    dtype=np.float64,
                ),
                # "upcoming_task_min": Box(
                #     0,
                #     np.inf,
                #     shape=(),
                #     dtype=np.float64,
                # ),
                # "upcoming_task_max": Box(
                #     0,
                #     np.inf,
                #     shape=(),
                #     dtype=np.float64,
                # ),
                "agent_is_capable_of_upcoming_task": MultiBinary(1),
                "agent_is_busy": MultiBinary(1),
            }
        )

    def reset(self, seed: int | None = None, options=None) -> tuple[dict, dict]:
        """Resets the environment to its initial state."""
        # Flush any remaining buffered CSV data from the previous episode
        self._flush_csv_buffer(force=True)

        self.steps = 0
        self.epochs += 1
        self.current_time = self.data["start_timestamp"].min()
        self.future_cases = self._initialize_future_cases()
        self.pending_cases = []
        self.completed_cases = []
        self.upcoming_case = None
        current_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"log_{current_timestamp}.csv")
        self.csv_buffer.clear()
        self.csv_header_written = False

        # Reuse cached distributions instead of refitting every episode (major speedup)
        # Distributions remain constant across episodes, so caching is safe
        activity_durations_dict = self._cached_activity_durations_dict
        stats_dict = self._cached_stats_dict

        # Update agent capabilities with cached distributions
        for agent in self.agents:
            resource = self.resources[agent.id]
            agent.capabilities = {
                self.task_dict[task]: activity_durations_dict[resource][task]
                for task in sorted(set(self.data["activity_name"]))
            }
            agent.stats_dict = stats_dict[resource]  # type: ignore

        # Set environment reference for all tasks and cases
        for case in self.future_cases:
            case.environment = self
            for task in case.all_tasks:
                task.environment = self

        observations = {
            agent.id: self._get_observations(agent) for agent in self.agents
        }

        return observations, {}

    def action_space(self, agent: int) -> Discrete:
        """Returns the action space for a single agent."""
        return Discrete(2)

    def _get_next_time(self) -> pd.Timestamp:
        """Get the next time where action is needed."""
        if self._has_immediate_assignment_work(self.current_time):
            return self.current_time

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
            arrival_time = self.future_cases[0].open_timestamp
            if arrival_time > self.current_time:
                next_event_times.append(arrival_time)

        # If there are events, advance time to the closest one
        if next_event_times:
            next_time = min(next_event_times)
            # Safety check: ensure we always move forward in time
            if next_time < self.current_time:
                debug_print_colored(
                    "⚠️ Time not progressing. Forcing small time increment.",
                    "red",
                )
                # Force a small time increment
                return self.current_time + pd.Timedelta(seconds=1)
            return next_time

        # If no events, advance time by a fixed interval
        return self.current_time + pd.Timedelta(seconds=1)
