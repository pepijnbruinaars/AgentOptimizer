# Can't import CustomEnvironment because it causes circular import issues, but the selfs are of that type
from .objects import Status
import numpy as np
from env_config import debug_print_colored


def get_reward(self) -> float:
    """Return the reward of the current state.

    Returns:
        float: Reward of the current state
    """
    # If no cases are being processed, return high reward
    if not self.pending_cases and not self.future_cases:
        return 100.0

    # Get completed task by checking only the most recently completed tasks
    completed_task = self.completed_task
    
    # First check pending cases - these are more likely to have recently completed tasks
    for case in self.pending_cases:
        # Only check the current task if it exists
        if case.current_task and case.current_task.status == Status.COMPLETED and case.current_task.completion_timestamp == self.current_time:
            completed_task = case.current_task
            break
            
    # If no task found in pending cases, check completed cases
    if not completed_task:
        # Only check the most recently completed case
        if self.completed_cases:
            last_case = self.completed_cases[-1]
            if last_case.current_task and last_case.current_task.status == Status.COMPLETED and last_case.current_task.completion_timestamp == self.current_time:
                completed_task = last_case.current_task

    # If no task was completed at this timestep, return base reward
    if not completed_task:
        debug_print_colored("No task completed in this step", "yellow")
        return 0.0

    # Calculate reward based on task duration
    duration = completed_task.completion_timestamp - completed_task.assigned_timestamp
    duration_minutes = duration.total_seconds() / 60
    if duration_minutes < 0:
        return 0.0
    reward = -((0.04 * duration_minutes) ** 2)
    debug_print_colored(f"Task {completed_task.format()} completed with duration {duration_minutes:.2f} minutes, reward: {reward:.2f}", "green")
    return reward
