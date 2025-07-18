"""
Device utilities for PyTorch device detection and management.

This module centralizes device detection logic that was previously duplicated
across main.py and MAPPO/agent.py.
"""
import torch


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Returns:
        torch.device: The optimal device (CUDA > MPS > CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_or_fallback(preferred_device: torch.device = None) -> torch.device:
    """
    Get a device, falling back to the best available if the preferred device is None.
    
    Args:
        preferred_device: The preferred device, or None to auto-detect
        
    Returns:
        torch.device: The device to use
    """
    if preferred_device is not None:
        return preferred_device
    return get_device()