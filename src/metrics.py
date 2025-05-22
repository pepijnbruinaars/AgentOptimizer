import pandas as pd


# ==== EFFICIENCY METRICS ====
def throughput_time(end_time: pd.Timestamp, assigned_time: pd.Timestamp) -> float:
    """
    Calculate the total throughput time in seconds.
    """
    return (end_time - assigned_time).total_seconds()


def waiting_time(start_time: pd.Timestamp, assigned_time: pd.Timestamp) -> float:
    """
    Calculate the waiting (queue) time in seconds.
    """
    return (start_time - assigned_time).total_seconds()


def processing_time(start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
    """
    Calculate the actual processing time in seconds.
    """
    return (end_time - start_time).total_seconds()


# ==== UTILIZATION METRICS ====
def calculate_busy_time(
    end_times: pd.Series[pd.Timestamp],
    assigned_times: pd.Series[pd.Timestamp],
) -> float:
    """
    Calculate the busy time in seconds.
    """
    time_differences: pd.Series[pd.Timedelta] = end_times - assigned_times
    busy_time = time_differences.dt.total_seconds().sum()

    return busy_time


def utilization(
    end_times: pd.Series[pd.Timestamp],
    assigned_times: pd.Series[pd.Timestamp],
    available_time: pd.Timedelta,
) -> float:
    """
    Calculate the utilization time as fraction of available time.
    """
    busy_time = calculate_busy_time(end_times, assigned_times)
    utilization = busy_time / available_time.total_seconds()

    return utilization


def idle_time(
    end_times: pd.Series[pd.Timestamp],
    assigned_times: pd.Series[pd.Timestamp],
    available_time: pd.Timedelta,
) -> float:
    """
    Calculate the idle time in seconds.
    """
    busy_time = calculate_busy_time(end_times, assigned_times)
    idle_time = available_time.total_seconds() - busy_time

    return idle_time
