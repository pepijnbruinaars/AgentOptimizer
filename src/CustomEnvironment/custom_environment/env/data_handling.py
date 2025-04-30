from functools import partial
from typing import Optional
import numpy as np
import pandas as pd


def activity_name_to_id(data: pd.DataFrame, activity_name: str) -> int:
    # Create unique set of activity names, then convert to list and get index using range
    return list(sorted(set(data["activity_name"]))).index(activity_name)


def activity_id_to_name(data: pd.DataFrame, activity_id: int) -> str:
    # Create unique set of activity names, then convert to list and get index using range
    return list(sorted(set(data["activity_name"])))[activity_id]


def find_first_case_activity(data: pd.DataFrame, case_id: int) -> int:
    """Find the first activity in a case.

    Args:
        data (pd.DataFrame): The event log data
        case_id (int): The case ID

    Returns:
        int: The ID of the first activity in the case
    """
    case_data = data[data["case_id"] == case_id]
    print(case_data)
    first_activity = case_data["activity_name"][0]
    first_activity_id = activity_name_to_id(data, first_activity)
    # Check if the first activity is missing or is not an integer
    if pd.isna(first_activity_id) or not isinstance(first_activity_id, int):
        raise ValueError(f"Case {case_id} has no first activity")
    return first_activity_id

def compute_agent_activity_durations(data: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Collects the activity durations for each agent in the event log.
    This function is an adaptation from the original code (_compute_activity_duration_distribution) made for the AgentSim paper.

    Args:
        data (pd.DataFrame): The standardized event log data

    Returns:
        dict: A dictionary with activity durations for each agent
    """
    activities = sorted(set(data['activity_name']))
    agents = sorted(set(data['resource']))
    activity_durations = {key: {k: np.array([]) for k in activities} for key in agents}

    for _, row in data.iterrows():
        agent = row['resource']
        activity = row['activity_name']
        duration = (row['end_timestamp'] - row['start_timestamp']).total_seconds()

        # Check if the duration is a valid number
        if pd.isna(duration) or not isinstance(duration, (int, float)):
            raise ValueError(f"Invalid duration for agent {agent} and activity {activity}")
        activity_durations[agent][activity] = np.append(activity_durations[agent][activity], duration)

    print(f"Activity durations for agents: {activity_durations}")
    return activity_durations

def sample_normal(mean: float, std: float) -> float:
    """Sample from a normal distribution with given mean and standard deviation.

    Args:
        mean (float): Mean of the normal distribution
        std (float): Standard deviation of the normal distribution
        size (int): Number of samples to generate

    Returns:
        np.ndarray: Array of samples from the normal distribution
    """
    value = np.random.normal(loc=mean, scale=std)
    if value < 0:
        return mean
    return value

def compute_activity_duration_distribution_per_agent(data: pd.DataFrame):
    """
    Compute the best fitting distribution of activity durations per agent.

    Args:
        df_train: Event log in pandas format

    Returns:
        dict: A dict storing for each agent the distribution for each activity.
    """
    activity_durations_dict = compute_agent_activity_durations(data)

    agents = activity_durations_dict.keys()
    activities = sorted(set(data['activity_name']))

    activity_duration_distribution_per_agent: dict[str, dict[str, Optional[partial[float]]]] = {agent: {activity: None for activity in activities} for agent in agents}

    for agent, val in activity_durations_dict.items():
        for act, duration_list in val.items():
            if len(duration_list) > 0:
                # Return callable normal distribution
                # Check if there are any negative values in the duration list
                if np.any(duration_list < 0):
                    raise ValueError(f"Negative duration found for agent {agent} and activity {act}")

                mean = float(np.mean(duration_list))
                std = float(np.std(duration_list))
                print(mean, std)
                duration_distribution = partial(sample_normal, mean, std)
                print("distribution sample", duration_distribution())
                activity_duration_distribution_per_agent[agent][act] = duration_distribution

    return activity_duration_distribution_per_agent