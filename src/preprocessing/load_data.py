from functools import partial
import pandas as pd
import os


def load_data(config: dict[str, str]) -> pd.DataFrame:
    """
    Load the data from the specified directory
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

    return processed_df
