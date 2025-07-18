"""
Utility functions shared across the AgentOptimizer codebase.
This module contains common helper functions to reduce code duplication.
"""
import torch
import os
from typing import Optional, Union
from display import print_colored


def get_device():
    """
    Get the best available device for PyTorch.
    
    Returns:
        torch.device: The best available device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_trained_mappo_agent(env, model_path: Optional[str] = None, create_new_if_missing: bool = True):
    """
    Load a trained MAPPO agent with configurable behavior for missing models.
    
    Args:
        env: The environment instance
        model_path: Path to saved MAPPO model (optional)
        create_new_if_missing: Whether to create a new agent if model is not found
        
    Returns:
        MAPPOAgent instance or None if model not found and create_new_if_missing is False
    """
    from MAPPO.agent import MAPPOAgent
    
    device = get_device()
    print_colored(f"Using device: {device}", "yellow")
    
    # Standard MAPPO configuration
    mappo_config = {
        "env": env,
        "hidden_size": 64,
        "lr_actor": 0.0003,
        "lr_critic": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,
        "batch_size": 1028,
        "num_epochs": 5,
        "device": device,
    }
    
    if model_path and os.path.exists(model_path):
        print_colored(f"Loading trained MAPPO model from {model_path}", "green")
        mappo_agent = MAPPOAgent(**mappo_config)
        mappo_agent.load_models(model_path)
        return mappo_agent
    else:
        if create_new_if_missing:
            print_colored("No trained model found, creating new MAPPO agent", "yellow")
            return MAPPOAgent(**mappo_config)
        else:
            print_colored("No trained MAPPO model found, skipping MAPPO evaluation", "yellow")
            return None


def load_trained_qmix_agent(env, model_path: Optional[str] = None):
    """
    Load a trained QMIX agent.
    
    Args:
        env: The environment instance
        model_path: Path to saved QMIX model (optional)
        
    Returns:
        QMIXAgent instance or None if model not found
    """
    from QMIX.agent import QMIXAgent
    
    device = get_device()
    
    if model_path and os.path.exists(model_path):
        print_colored(f"Loading trained QMIX model from {model_path}", "green")
        qmix_agent = QMIXAgent(env=env, device=device)
        qmix_agent.load_models(model_path)
        return qmix_agent
    else:
        print_colored("No trained QMIX model found, skipping QMIX evaluation", "yellow")
        return None