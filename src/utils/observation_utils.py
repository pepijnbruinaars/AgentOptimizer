"""
Observation utilities for converting observation dictionaries to PyTorch tensors.

This module centralizes observation processing logic that was previously duplicated
in ActorNetwork and CriticNetwork classes.
"""
import torch
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Any, Union


def process_observation_to_tensor(
    obs_dict: Dict[str, Any], 
    obs_keys: List[str], 
    device: torch.device
) -> List[torch.Tensor]:
    """
    Convert observation dictionary components to tensor parts.
    
    Args:
        obs_dict: Dictionary containing observation data
        obs_keys: List of observation keys to process
        device: Target device for tensors
        
    Returns:
        List of tensor parts that can be concatenated
    """
    tensor_parts = []
    
    for key in obs_keys:
        if key in obs_dict:
            # Handle different observation components
            if isinstance(obs_dict[key], np.ndarray):
                # Use from_numpy for better performance, then move to device
                tensor_data = torch.from_numpy(
                    obs_dict[key].flatten().astype(np.float32)
                )
                tensor_parts.append(tensor_data.to(device))
            else:
                # Handle scalar values or other types
                tensor_parts.append(
                    torch.tensor(
                        [obs_dict[key]], device=device, dtype=torch.float32
                    )
                )
    
    return tensor_parts


def process_observation_batch_to_tensors(
    obs_dicts: List[Dict[str, Any]], 
    obs_keys: List[str], 
    device: torch.device
) -> torch.Tensor:
    """
    Convert a batch of observation dictionaries to a batched tensor.
    
    Args:
        obs_dicts: List of observation dictionaries
        obs_keys: List of observation keys to process
        device: Target device for tensors
        
    Returns:
        Batched tensor with shape (batch_size, feature_dim)
    """
    batch_inputs = []
    
    # Process all observations efficiently
    for obs_dict in obs_dicts:
        tensor_parts = []
        
        for key in obs_keys:
            if key in obs_dict:
                # Handle different observation components
                if isinstance(obs_dict[key], np.ndarray):
                    # Convert to tensor efficiently (keep on CPU initially)
                    tensor_data = torch.from_numpy(
                        obs_dict[key].flatten().astype(np.float32)
                    )
                    tensor_parts.append(tensor_data)
                else:
                    # Handle scalar values
                    tensor_parts.append(
                        torch.tensor([obs_dict[key]], dtype=torch.float32)
                    )
        
        # Concatenate parts for this observation (still on CPU)
        if tensor_parts:
            batch_inputs.append(torch.cat(tensor_parts))
    
    # Move entire batch to device at once for efficiency
    if batch_inputs:
        return torch.stack(batch_inputs).to(device)
    else:
        return torch.empty(0, device=device)


def calculate_observation_size(obs_space: spaces.Dict) -> int:
    """
    Calculate the total input size from an observation space.
    
    Args:
        obs_space: Gymnasium Dict observation space
        
    Returns:
        Total size needed for flattened observations
    """
    input_size = 0
    
    for key, space in obs_space.items():
        if isinstance(space, spaces.Discrete):
            input_size += 1
        elif isinstance(space, spaces.Box):
            input_size += int(np.prod(space.shape))
        elif isinstance(space, spaces.MultiBinary):
            input_size += int(np.prod(space.shape))
    
    return input_size


def get_observation_keys(obs_space: spaces.Dict) -> List[str]:
    """
    Extract observation keys from observation space.
    
    Args:
        obs_space: Gymnasium Dict observation space
        
    Returns:
        List of observation keys
    """
    return list(obs_space.keys())