# Can't import CustomEnvironment because it causes circular import issues, but the selfs are of that type
from env_config import debug_print_colored


def get_reward(self) -> float:
    """Return a cooperative reward for the current simulation step.

    The reward is intentionally independent of historical medians:
    - 100.0 when all work is done
    - 0.0 when no task completed this step
    - negative throughput time in minutes for the completed task
    """
    if not self.pending_cases and not self.future_cases:
        return 100.0

    completed_task = self.completed_task
    if not completed_task:
        debug_print_colored("No task completed in this step", "yellow")
        return 0.0

    if (
        completed_task.assigned_timestamp is None
        or completed_task.completion_timestamp is None
    ):
        debug_print_colored(
            f"Task {completed_task.format()} missing timestamps for reward calculation",
            "red",
        )
        return 0.0

    duration = completed_task.completion_timestamp - completed_task.assigned_timestamp
    actual_duration_seconds = duration.total_seconds()

    if actual_duration_seconds < 0:
        debug_print_colored(
            f"Warning: Negative duration {actual_duration_seconds} for task {completed_task.format()}",
            "red",
        )
        return 0.0

    duration_minutes = actual_duration_seconds / 60.0
    reward = -duration_minutes

    return reward
