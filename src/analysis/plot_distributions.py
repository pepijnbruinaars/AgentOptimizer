import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns


def plot_log_metrics(experiment_dir: str) -> None:
    """Plot metrics from log files in the experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
    """
    # Initialize lists to store metrics
    throughput_per_epoch = []
    waiting_per_epoch = []
    processing_per_epoch = []
    epoch_labels = []
    
    # First, process the original dataset
    print("Processing original dataset...")
    original_log_file = "data/input/BPI_Challenge_2019.csv"  # Adjust path if needed
    if os.path.exists(original_log_file):
        original_df = pd.read_csv(original_log_file)
        case_groups = original_df.groupby('case_id')
        
        # Initialize lists for original data
        throughput_times = []
        waiting_times = []
        processing_times = []
        
        for case_id, group in case_groups:
            # Convert timestamps once for the entire group
            group_times = group[['start_timestamp', 'end_timestamp']].copy()
            for col in group_times.columns:
                group_times[col] = pd.to_datetime(group_times[col], errors='coerce')
            
            # Calculate waiting and processing times vectorized
            waiting_times_vec = (group_times['start_timestamp'] - group_times['start_timestamp']).dt.total_seconds() / 60
            processing_times_vec = (group_times['end_timestamp'] - group_times['start_timestamp']).dt.total_seconds() / 60
            throughput_times_vec = (group_times['end_timestamp'] - group_times['start_timestamp']).dt.total_seconds() / 60
            
            # Sum up the times for the case
            waiting = waiting_times_vec.sum()
            processing = processing_times_vec.sum()
            throughput = throughput_times_vec.sum()
            
            waiting_times.append(waiting)
            processing_times.append(processing)
            throughput_times.append(throughput)
        
        # Append original data as first epoch
        throughput_per_epoch.append(throughput_times)
        waiting_per_epoch.append(waiting_times)
        processing_per_epoch.append(processing_times)
        epoch_labels.append('Original')
    
    # Get all log files from the logs directory
    logs_dir = Path(experiment_dir) / "logs"
    log_files = sorted([f for f in logs_dir.glob("log_*.csv")])
    
    # Take first log file and then skip 4 for the rest
    selected_logs = [log_files[0]] + log_files[5::5]
    
    for i, log_file in enumerate(selected_logs):
        print(f"Processing log file: {log_file}")
        log_df = pd.read_csv(log_file)
        
        # Group by case_id
        case_groups = log_df.groupby('case_id')
        
        # Initialize lists for this epoch
        throughput_times = []
        waiting_times = []
        processing_times = []
        
        for case_id, group in case_groups:
            # Convert timestamps
            group_times = group[['task_assigned_time', 'task_started_time', 'task_completed_time']].copy()
            for col in group_times.columns:
                # Parse timestamps with the specific format including microseconds and timezone
                group_times[col] = pd.to_datetime(group_times[col], format='%Y-%m-%d %H:%M:%S.%f%z')
            
            # Calculate metrics (timestamps are now in UTC)
            waiting_times_vec = (group_times['task_started_time'] - group_times['task_assigned_time']).dt.total_seconds() / 60
            processing_times_vec = (group_times['task_completed_time'] - group_times['task_started_time']).dt.total_seconds() / 60
            throughput_times_vec = (group_times['task_completed_time'] - group_times['task_assigned_time']).dt.total_seconds() / 60
            
            # Sum up the times for the case
            waiting = waiting_times_vec.sum()
            processing = processing_times_vec.sum()
            throughput = throughput_times_vec.sum()
            
            waiting_times.append(waiting)
            processing_times.append(processing)
            throughput_times.append(throughput)
        
        # Append to lists for plots
        throughput_per_epoch.append(throughput_times)
        waiting_per_epoch.append(waiting_times)
        processing_per_epoch.append(processing_times)
        epoch_labels.append(f"Epoch {i}")
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot violin plots for each metric
    axes[0].violinplot(throughput_per_epoch, positions=range(len(throughput_per_epoch)))
    axes[0].set_xticks(range(len(epoch_labels)))
    axes[0].set_xticklabels(epoch_labels, rotation=45)
    axes[0].set_title("Case Throughput Time per Epoch")
    axes[0].set_ylabel("Throughput Time (minutes)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)
    
    axes[1].violinplot(waiting_per_epoch, positions=range(len(waiting_per_epoch)))
    axes[1].set_xticks(range(len(epoch_labels)))
    axes[1].set_xticklabels(epoch_labels, rotation=45)
    axes[1].set_title("Case Waiting Time per Epoch")
    axes[1].set_ylabel("Waiting Time (minutes)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)
    
    axes[2].violinplot(processing_per_epoch, positions=range(len(processing_per_epoch)))
    axes[2].set_xticks(range(len(epoch_labels)))
    axes[2].set_xticklabels(epoch_labels, rotation=45)
    axes[2].set_title("Case Processing Time per Epoch")
    axes[2].set_ylabel("Processing Time (minutes)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(experiment_dir) / 'log_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    # Example usage
    experiment_dir = "experiments/mappo_20250614_014227"  # Replace with your experiment directory
    plot_log_metrics(experiment_dir) 