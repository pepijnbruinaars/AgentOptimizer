"""
Utility functions shared across the AgentOptimizer codebase.
This module contains common helper functions to reduce code duplication.
"""
import torch
import os
import numpy as np
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


def create_performance_data(env):
    """
    Create performance data for BestMedianAgent based on agent capabilities.
    In practice, this would come from historical agent performance data.

    Args:
        env: The environment with agents

    Returns:
        Dict mapping agent_id to list of performance scores
    """
    performance_data = {}

    # Generate more realistic performance data based on agent capabilities if available
    for agent in env.agents:
        # Base performance with some variability
        base_performance = np.random.uniform(0.4, 0.8)
        performance_variations = np.random.normal(0, 0.1, 10)
        performance_data[agent.id] = np.clip(
            base_performance + performance_variations, 0.0, 1.0
        ).tolist()

        # Try to check agent's capabilities for more realistic data if available
        try:
            if hasattr(agent, "capabilities") and hasattr(agent, "stats_dict"):
                # Adjust based on the number of tasks the agent can perform
                capable_tasks = sum(
                    1 for v in agent.capabilities.values() if v is not None
                )
                capability_ratio = (
                    capable_tasks / len(agent.capabilities)
                    if agent.capabilities
                    else 0.5
                )

                # Higher capability ratio = better base performance
                adjusted_base = 0.3 + (0.6 * capability_ratio)

                # Create performance data with some variability
                variations = np.random.normal(0, 0.15, 10)
                performance_data[agent.id] = np.clip(
                    adjusted_base + variations, 0.0, 1.0
                ).tolist()
        except Exception:
            # Fall back to default if there's any issue
            pass

    return performance_data


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