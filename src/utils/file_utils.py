"""
File utilities for common file operations and directory management.

This module centralizes file operation patterns that were previously duplicated
across multiple modules.
"""
import os
import numpy as np
import torch
from pathlib import Path
from typing import Any, List, Optional, Union


def ensure_directory_exists(path: Union[str, Path], exist_ok: bool = True) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        exist_ok: Whether to ignore if directory already exists
    """
    os.makedirs(path, exist_ok=exist_ok)


def save_numpy_csv(
    filepath: Union[str, Path],
    data: np.ndarray,
    delimiter: str = ";",
    header: Optional[str] = None,
    comments: str = ""
) -> None:
    """
    Save numpy array to CSV file with consistent formatting.
    
    Args:
        filepath: Path to save the CSV file
        data: Numpy array to save
        delimiter: CSV delimiter
        header: Optional header string
        comments: Comment prefix for header
    """
    kwargs = {
        'fname': filepath,
        'X': data,
        'delimiter': delimiter,
        'comments': comments
    }
    
    # Only add header if it's provided
    if header is not None:
        kwargs['header'] = header
    
    np.savetxt(**kwargs)


def save_text_file(filepath: Union[str, Path], content: str) -> None:
    """
    Save text content to a file.
    
    Args:
        filepath: Path to save the file
        content: Text content to write
    """
    with open(filepath, 'w') as f:
        f.write(content)


def append_text_file(filepath: Union[str, Path], content: str) -> None:
    """
    Append text content to a file.
    
    Args:
        filepath: Path to the file
        content: Text content to append
    """
    with open(filepath, 'a') as f:
        f.write(content)


def save_torch_model(model_state_dict: dict, filepath: Union[str, Path]) -> None:
    """
    Save PyTorch model state dictionary.
    
    Args:
        model_state_dict: Model state dictionary to save
        filepath: Path to save the model file
    """
    torch.save(model_state_dict, filepath)


def load_torch_model(filepath: Union[str, Path], device: torch.device) -> dict:
    """
    Load PyTorch model state dictionary.
    
    Args:
        filepath: Path to the model file
        device: Device to map the model to
        
    Returns:
        Loaded model state dictionary
    """
    return torch.load(filepath, map_location=device)


def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """
    Create a directory for experiment results.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created experiment directory
    """
    experiment_dir = os.path.join(base_dir, experiment_name)
    ensure_directory_exists(experiment_dir)
    return experiment_dir


def create_timestamped_directory(base_dir: str, prefix: str) -> str:
    """
    Create a timestamped directory for experiments.
    
    Args:
        base_dir: Base directory
        prefix: Prefix for the directory name
        
    Returns:
        Path to the created timestamped directory
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    ensure_directory_exists(experiment_dir)
    return experiment_dir