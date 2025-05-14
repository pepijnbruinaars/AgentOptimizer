# Can't import CustomEnvironment because it causes circular import issues, but the selfs are of that type
from .objects import Case


def _get_agent_active_cases_penalty(self, agent: int) -> float:
    """Gets a penalty for the number of active cases for an agent.

    Args:
        self (CustomEnvironment): The environment object
        agent (int): The agent ID

    Returns:
        float: The penalty score to be applied to the reward. This is a negative value.
    """
    current_case = self.agents[agent].current_case
    case_queue: list[Case] = self.agents[agent].case_queue
    active_cases = len([case for case in case_queue if not case.is_completed])
    if current_case is not None:
        active_cases += 1

    # Calculate the penalty based on the number of active cases
    return -0.1 * active_cases**2


def get_reward(self) -> float:
    """Return the reward of the current state.

    Returns:
        float: Reward of the current state
    """
    # Calculate the reward based on the number of completed cases
    completed_cases = [
        case
        for case in self.completed_cases
        if case.is_completed and case.completion_timestamp == self.current_time
    ]
    pending_completed_cases = [
        case
        for case in self.pending_cases
        if case.is_completed and case.completion_timestamp == self.current_time
    ]
    completed_cases.extend(pending_completed_cases)

    # Calculate the reward based on the number of completed cases
    reward = 0.0
    for case in completed_cases:
        reward += 1.0

    if len(self.future_cases) > 0 or len(self.pending_cases) > 0:
        reward -= 0.5

    if len(self.pending_cases) < 0 and len(self.future_cases) < 0:
        reward += 100

    return reward
