from functools import partial
from typing import Optional
import numpy as np
import pandas as pd


def compute_agent_activity_durations(
    data: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """Collects the activity durations for each agent in the event log.
    This function is an adaptation from the original code (_compute_activity_duration_distribution) made for the AgentSim paper.

    Args:
        data (pd.DataFrame): The standardized event log data

    Returns:
        dict: A dictionary with activity durations for each agent
    """
    activities = sorted(set(data["activity_name"]))
    agents = sorted(set(data["resource"]))
    activity_durations = {key: {k: np.array([]) for k in activities} for key in agents}

    for _, row in data.iterrows():
        agent = row["resource"]
        activity = row["activity_name"]
        duration = (row["end_timestamp"] - row["start_timestamp"]).total_seconds()

        # Check if the duration is a valid number
        if pd.isna(duration) or not isinstance(duration, (int, float)):
            raise ValueError(
                f"Invalid duration for agent {agent} and activity {activity}"
            )
        activity_durations[agent][activity] = np.append(
            activity_durations[agent][activity], duration
        )

    return activity_durations


def sample_normal(mean: float, std: float, min: float, max: float) -> float:
    """Sample from a normal distribution with given mean and standard deviation.

    Args:
        mean (float): Mean of the normal distribution
        std (float): Standard deviation of the normal distribution
        min (float): Minimum value for the sample
        max (float): Maximum value for the sample

    Returns:
        np.ndarray: Array of samples from the normal distribution
    """
    value = np.random.normal(loc=mean, scale=std)
    if value < min:
        return min
    if value > max:
        return max
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
    activities = sorted(set(data["activity_name"]))

    activity_duration_distribution_per_agent: dict[
        str, dict[str, Optional[partial[float]]]
    ] = {agent: {activity: None for activity in activities} for agent in agents}
    stats_dict: dict[str, dict[str, Optional[dict[str, float]]]] = {
        agent: {activity: None for activity in activities} for agent in agents
    }
    for agent, val in activity_durations_dict.items():
        for act, duration_list in val.items():
            if len(duration_list) > 0:
                # Return callable normal distribution
                # Check if there are any negative values in the duration list
                if np.any(duration_list < 0):
                    raise ValueError(
                        f"Negative duration found for agent {agent} and activity {act}"
                    )

                # Calculate mean, std, min, and max
                mean = float(np.mean(duration_list))
                std = float(np.std(duration_list))
                min = float(np.min(duration_list))
                max = float(np.max(duration_list))

                # Set the distribution function for the activity
                duration_distribution = partial(sample_normal, mean, std, min, max)
                activity_duration_distribution_per_agent[agent][
                    act
                ] = duration_distribution
                stats_dict[agent][act] = {
                    "mean": mean,
                    "std": std,
                    "min": min,
                    "max": max,
                }

    return activity_duration_distribution_per_agent, stats_dict
