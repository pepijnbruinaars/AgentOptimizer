# Can't import CustomEnvironment because it causes circular import issues, but the selfs are of that type


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
