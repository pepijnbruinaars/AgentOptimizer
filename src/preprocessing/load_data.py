from functools import partial
from collections import defaultdict
import pandas as pd  # type: ignore
import os


def load_data(config: dict[str, str]) -> pd.DataFrame:
    """
    Loads the data based on the provided config.
    Returns a DataFrame with the data in the following columns and types:
    - case_id: str
    - resource: str
    - activity_name: str
    - start_timestamp: pd.Timestamp
    - end_timestamp: pd.Timestamp
    """
    data_dir = os.path.join("data", "input", config["input_filename"])
    df = pd.read_csv(data_dir)

    # Sort the data by case_id and start_timestamp
    df.sort_values(
        by=[config["case_id_col"], config["start_timestamp_col"]],
        inplace=True,
        ignore_index=True,
    )

    # Create new dataframe with only the necessary columns and standardized names
    processed_df = pd.DataFrame(
        columns=[
            "case_id",
            "resource",
            "activity_name",
            "start_timestamp",
            "end_timestamp",
        ]
    )
    processed_df["case_id"] = df[config["case_id_col"]]
    processed_df["resource"] = df[config["resource_id_col"]]
    processed_df["activity_name"] = df[config["activity_col"]]

    # Convert to datetime format
    processed_df[["start_timestamp", "end_timestamp"]] = df[
        ["start_timestamp", "end_timestamp"]
    ].apply(partial(pd.to_datetime, format="mixed"))
    # Convert times to UTC
    processed_df["start_timestamp"] = processed_df["start_timestamp"].dt.tz_convert(
        "UTC"
    )
    processed_df["end_timestamp"] = processed_df["end_timestamp"].dt.tz_convert("UTC")

    return processed_df


def split_data(df: pd.DataFrame, split: float = 0.8):
    # Split data into training and testing sets, taking into account the case_id and splitting such that no day is split between the sets
    # Get unique case_ids
    case_ids = df["case_id"].unique()
    case_dates = df.groupby("case_id")["start_timestamp"].min().dt.date

    # Shuffle case_ids

    # Split case_ids into training and testing sets
    train_case_ids = set()
    test_case_ids = set()

    # Date split, so that no day or case is split between the sets
    case_dates = sorted(case_dates)
    date_split_index = int(len(case_dates) * split)

    for i in range(len(case_ids)):
        if case_dates[i] in case_dates[:date_split_index]:
            train_case_ids.add(case_ids[i])
        else:
            test_case_ids.add(case_ids[i])

    # Create training and testing sets
    train_df = df[df["case_id"].isin(train_case_ids)]
    test_df = df[df["case_id"].isin(test_case_ids)]
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    print(
        f"Proportion of train set: {len(train_df) / (len(test_df) + len(train_df)):.2f}"
    )
    return train_df, test_df
