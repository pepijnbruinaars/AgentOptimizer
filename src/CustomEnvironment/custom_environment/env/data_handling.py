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
