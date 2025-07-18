#!/usr/bin/env python3
"""
Example script to evaluate baseline agents against trained MAPPO agent.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from src.baselines import create_baseline_agents, BaselineEvaluator
from src.MAPPO.agent import MAPPOAgent
from display import print_colored
# Import shared utilities to avoid code duplication
from src.utils import load_trained_mappo_agent


def create_performance_data_example():
    """
    Create example performance data for BestMedianAgent.
    In practice, this would come from historical agent performance.

    Returns:
        Dict mapping agent_id to list of performance scores
    """
    # Example: Agent 0 is better at certain tasks, Agent 1 at others
    performance_data = {
        "agent_0": [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.8],  # Median: 0.8
        "agent_1": [0.6, 0.5, 0.7, 0.8, 0.6, 0.5, 0.6, 0.7],  # Median: 0.6
        "agent_2": [0.4, 0.3, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4],  # Median: 0.4
    }
    return performance_data


def run_baseline_evaluation(env, num_episodes=100, model_path=None):
    """
    Run evaluation comparing MAPPO agent with baseline agents.

    Args:
        env: The environment to evaluate on
        num_episodes: Number of episodes to evaluate each agent
        model_path: Path to trained MAPPO model (optional)
    """
    print_colored("\n" + "=" * 70, "yellow")
    print_colored("BASELINE EVALUATION COMPARISON", "yellow")
    print_colored("=" * 70, "yellow")

    # Create performance data for BestMedianAgent
    performance_data = create_performance_data_example()

    # Create baseline agents
    random_agent, best_median_agent = create_baseline_agents(
        env, performance_data=performance_data, seed=42
    )

    # Load trained MAPPO agent
    mappo_agent = load_trained_mappo_agent(env, model_path, create_new_if_missing=True)

    # Create evaluator
    evaluator = BaselineEvaluator(env)

    # Define agents to compare
    agent_configs = [
        (random_agent, "Random Baseline"),
        (best_median_agent, "Best Median Baseline"),
        (mappo_agent, "MAPPO Agent"),
    ]

    # Run comparison
    results = evaluator.compare_agents(agent_configs, num_episodes=num_episodes)

    # Save results
    results_file = "data/output/baseline_comparison_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    evaluator.save_results(results_file)

    return results


def run_quick_baseline_evaluation():
    """
    Quick evaluation with a simple mock environment for testing.
    Use this if you don't have your full environment set up.
    """
    print_colored("Running quick baseline evaluation with mock environment...", "blue")

    # Mock environment for testing (replace with your actual environment)
    class MockEnv:
        def __init__(self):
            self.agents = [MockAgent("agent_0"), MockAgent("agent_1")]
            self.current_step = 0

        def observation_space(self, agent_id):
            from gymnasium import spaces

            return {
                "position": spaces.Box(low=0, high=10, shape=(2,), dtype=np.float32),
                "inventory": spaces.Discrete(5),
            }

        def action_space(self, agent_id):
            from gymnasium import spaces

            return spaces.Discrete(2)  # 0: don't raise hand, 1: raise hand

        def reset(self):
            self.current_step = 0
            return {
                "agent_0": {"position": np.array([1.0, 1.0]), "inventory": 0},
                "agent_1": {"position": np.array([2.0, 2.0]), "inventory": 1},
            }

        def step(self, actions):
            self.current_step += 1

            # Mock rewards based on actions
            rewards = {}
            for agent_id, action in actions.items():
                rewards[agent_id] = np.random.uniform(-1, 1) + (
                    0.5 if action == 1 else 0
                )

            # Mock observations
            observations = {
                "agent_0": {
                    "position": np.random.uniform(0, 10, 2),
                    "inventory": np.random.randint(0, 5),
                },
                "agent_1": {
                    "position": np.random.uniform(0, 10, 2),
                    "inventory": np.random.randint(0, 5),
                },
            }

            # Mock dones
            dones = {
                "agent_0": self.current_step >= 50,
                "agent_1": self.current_step >= 50,
            }

            # Mock infos
            infos = {
                "agent_0": {"task_success": np.random.random() > 0.3},
                "agent_1": {"task_success": np.random.random() > 0.3},
            }

            return observations, rewards, dones, infos

    class MockAgent:
        def __init__(self, agent_id):
            self.id = agent_id

    # Create mock environment
    mock_env = MockEnv()

    # Run evaluation
    results = run_baseline_evaluation(mock_env, num_episodes=10)

    print_colored("\nQuick evaluation complete! Check the results above.", "green")
    return results


if __name__ == "__main__":
    # Example usage:

    # Option 1: Quick test with mock environment
    print_colored("Running baseline evaluation...", "blue")
    run_quick_baseline_evaluation()

    # Option 2: Use with your actual environment (uncomment and modify):
    # from your_environment_module import YourEnvironment
    # env = YourEnvironment()
    # model_path = "models/mappo_final"  # Path to your trained model
    # results = run_baseline_evaluation(env, num_episodes=100, model_path=model_path)
