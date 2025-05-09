# Can't import CustomEnvironment because it causes circular import issues, but the selfs are of that type
from .objects import Case, ResourceAgent


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
    completed_cases = len([case for case in self.completed_cases if case.is_completed])
    reward = 0.1 * completed_cases
    # Calculate the penalty for the number of active cases for each agent
    for agent in range(len(self.agents)):
        reward += _get_agent_active_cases_penalty(self, agent)

    return reward
