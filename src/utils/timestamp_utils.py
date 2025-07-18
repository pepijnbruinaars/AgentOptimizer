"""
Timestamp utilities for consistent timestamp processing across the codebase.

This module centralizes timestamp processing logic that was previously duplicated
across preprocessing and analysis modules.
"""
import pandas as pd
from typing import Union, List
from functools import partial


def convert_to_datetime(
    timestamps: Union[pd.Series, List[str]], 
    format: str = "mixed",
    errors: str = "raise"
) -> pd.Series:
    """
    Convert timestamps to pandas datetime format.
    
    Args:
        timestamps: Series or list of timestamp strings
        format: Format string for parsing ("mixed" for automatic detection)
        errors: How to handle parsing errors ('raise', 'coerce', 'ignore')
        
    Returns:
        Series with converted datetime values
    """
    return pd.to_datetime(timestamps, format=format, errors=errors)


def convert_to_utc(timestamp_series: pd.Series) -> pd.Series:
    """
    Convert timestamp series to UTC timezone.
    
    Args:
        timestamp_series: Series with datetime values
        
    Returns:
        Series with UTC timezone converted values
    """
    return timestamp_series.dt.tz_convert("UTC")


def process_timestamp_columns(
    df: pd.DataFrame, 
    timestamp_columns: List[str],
    format: str = "mixed",
    convert_utc: bool = True
) -> pd.DataFrame:
    """
    Process multiple timestamp columns in a DataFrame.
    
    Args:
        df: DataFrame containing timestamp columns
        timestamp_columns: List of column names to process
        format: Format string for parsing
        convert_utc: Whether to convert to UTC timezone
        
    Returns:
        DataFrame with processed timestamp columns
    """
    df_copy = df.copy()
    
    # Convert to datetime format
    df_copy[timestamp_columns] = df_copy[timestamp_columns].apply(
        partial(convert_to_datetime, format=format)
    )
    
    # Convert to UTC if requested
    if convert_utc:
        for col in timestamp_columns:
            df_copy[col] = convert_to_utc(df_copy[col])
    
    return df_copy


def calculate_time_difference_minutes(
    start_time: pd.Series, 
    end_time: pd.Series
) -> pd.Series:
    """
    Calculate time difference in minutes between two timestamp series.
    
    Args:
        start_time: Series with start timestamps
        end_time: Series with end timestamps
        
    Returns:
        Series with time differences in minutes
    """
    return (end_time - start_time).total_seconds() / 60


def calculate_time_difference_seconds(
    start_time: pd.Series, 
    end_time: pd.Series
) -> pd.Series:
    """
    Calculate time difference in seconds between two timestamp series.
    
    Args:
        start_time: Series with start timestamps
        end_time: Series with end timestamps
        
    Returns:
        Series with time differences in seconds
    """
    return (end_time - start_time).total_seconds()