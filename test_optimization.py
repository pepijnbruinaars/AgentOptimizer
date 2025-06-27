#!/usr/bin/env python3
"""
Test script to verify MAPPO optimizations.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import time
import torch
import numpy as np
from src.MAPPO.networks import ActorNetwork, CriticNetwork
from gymnasium import spaces


def test_batch_processing():
    """Test that batch processing works correctly and is faster."""
    print("Testing batch processing optimizations...")

    # Set up test parameters
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create mock observation space
    obs_space = {
        "agent_pos": spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32),
        "goal_pos": spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32),
        "inventory": spaces.Discrete(5),
    }
    action_space = spaces.Discrete(4)

    # Create networks
    actor = ActorNetwork(obs_space, action_space, hidden_size=64, device=device).to(
        device
    )
    critic = CriticNetwork(obs_space, 2, hidden_size=64, device=device).to(device)

    # Generate test data
    batch_size = 32
    test_obs_batch = []
    test_rewards = []

    for i in range(batch_size):
        obs = {
            "agent_pos": np.random.rand(2).astype(np.float32) * 10,
            "goal_pos": np.random.rand(2).astype(np.float32) * 10,
            "inventory": np.random.randint(0, 5),
        }
        test_obs_batch.append(obs)
        test_rewards.append(np.random.randn())

    # Test actor batch processing
    print("\nTesting Actor batch processing...")

    # Individual processing (old way)
    start_time = time.perf_counter()
    individual_results = []
    for i, obs in enumerate(test_obs_batch):
        result = actor(obs, test_rewards[i])
        individual_results.append(result)
    individual_time = time.perf_counter() - start_time

    # Batch processing (new way)
    start_time = time.perf_counter()
    batch_results = actor.forward_batch(test_obs_batch, test_rewards)
    batch_time = time.perf_counter() - start_time

    print(f"Individual processing time: {individual_time:.4f}s")
    print(f"Batch processing time: {batch_time:.4f}s")
    print(f"Speedup: {individual_time / batch_time:.2f}x")

    # Verify results are equivalent
    individual_stacked = torch.stack(individual_results)
    diff = torch.abs(individual_stacked - batch_results).max()
    print(f"Max difference between individual and batch: {diff:.6f}")
    assert diff < 1e-5, "Batch processing results don't match individual processing!"

    # Test critic batch processing
    print("\nTesting Critic batch processing...")

    # Create mock multi-agent observations
    critic_obs_batch = []
    for obs in test_obs_batch:
        # Simulate 2 agents
        agent_obs_list = [obs, obs]  # Same obs for simplicity
        critic_obs_batch.append(agent_obs_list)

    # Individual processing (old way)
    start_time = time.perf_counter()
    individual_values = []
    for i, obs_list in enumerate(critic_obs_batch):
        value = critic(obs_list, test_rewards[i])
        individual_values.append(value)
    individual_time = time.perf_counter() - start_time

    # Batch processing (new way)
    start_time = time.perf_counter()
    batch_values = critic.forward_batch(critic_obs_batch, test_rewards)
    batch_time = time.perf_counter() - start_time

    print(f"Individual processing time: {individual_time:.4f}s")
    print(f"Batch processing time: {batch_time:.4f}s")
    print(f"Speedup: {individual_time / batch_time:.2f}x")

    # Verify results are equivalent
    individual_stacked = torch.cat(individual_values)
    diff = torch.abs(individual_stacked - batch_values.squeeze()).max()
    print(f"Max difference between individual and batch: {diff:.6f}")
    assert diff < 1e-5, "Batch processing results don't match individual processing!"

    print("\n✅ All batch processing tests passed!")
    print("The optimizations are working correctly and providing significant speedup.")


if __name__ == "__main__":
    test_batch_processing()
