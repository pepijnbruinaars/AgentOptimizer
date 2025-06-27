from functools import partial
from typing import Optional, Tuple, Dict, List, Mapping
import numpy as np
import pandas as pd
from .duration_distribution import DurationDistribution, get_best_fitting_distribution


def compute_agent_activity_durations(
    data: pd.DataFrame,
) -> dict[str, dict[str, List[float]]]:
    """Collects the activity durations for each agent in the event log.
    This function is an adaptation from the original code (_compute_activity_duration_distribution) made for the AgentSim paper.

    Args:
        data (pd.DataFrame): The standardized event log data

    Returns:
        dict: A dictionary with activity durations for each agent
    """
    activities = sorted(set(data["activity_name"]))
    agents = sorted(set(data["resource"]))
    activity_durations = {key: {k: [] for k in activities} for key in agents}

    for _, row in data.iterrows():
        agent = row["resource"]
        activity = row["activity_name"]
        duration = (row["end_timestamp"] - row["start_timestamp"]).total_seconds()

        # Check if the duration is a valid number
        if pd.isna(duration) or not isinstance(duration, (int, float)):
            raise ValueError(
                f"Invalid duration for agent {agent} and activity {activity}"
            )
        activity_durations[agent][activity].append(duration)

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


def compute_activity_duration_distribution_per_agent(
    data: pd.DataFrame,
) -> Tuple[Mapping[str, Mapping[str, Optional[DurationDistribution]]], Mapping[str, Mapping[str, Optional[Dict[str, float]]]]]:
    """
    Compute the best fitting distribution of activity durations per agent.

    Args:
        data: Event log in pandas format

    Returns:
        Tuple containing:
        - A dict storing for each agent the best fitting distribution for each activity
        - A dict storing for each agent the statistics for each activity
    """
    activity_durations_dict = compute_agent_activity_durations(data)

    agents = activity_durations_dict.keys()
    activities = sorted(set(data["activity_name"]))

    activity_duration_distribution_per_agent: Dict[str, Dict[str, Optional[DurationDistribution]]] = {
        agent: {activity: None for activity in activities} for agent in agents
    }
    stats_dict: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {
        agent: {activity: None for activity in activities} for agent in agents
    }

    for agent, val in activity_durations_dict.items():
        for act, duration_list in val.items():
            if len(duration_list) > 0:
                # Check if there are any negative values in the duration list
                if any(d < 0 for d in duration_list):
                    raise ValueError(
                        f"Negative duration found for agent {agent} and activity {act}"
                    )

                # Calculate statistics
                mean = float(np.mean(duration_list))
                median = float(np.median(duration_list))
                std = float(np.std(duration_list))
                min_val = float(np.min(duration_list))
                max_val = float(np.max(duration_list))

                # Store statistics
                stats_dict[agent][act] = {
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "min": min_val,
                    "max": max_val,
                }

                # Fit best distribution
                try:
                    best_distribution = get_best_fitting_distribution(duration_list, filter_outliers=False)
                    activity_duration_distribution_per_agent[agent][act] = best_distribution
                except Exception as e:
                    print(f"Warning: Could not fit distribution for agent {agent} and activity {act}: {e}")
                    # Fallback to normal distribution if fitting fails
                    activity_duration_distribution_per_agent[agent][act] = DurationDistribution(
                        "norm", mean=mean, std=std, minimum=min_val, maximum=max_val
                    )

    return activity_duration_distribution_per_agent, stats_dict
