#!/usr/bin/env python3
"""
Script to run baseline evaluations for the Agent Optimizer project.
This script evaluates baseline agents against trained MAPPO agents.

The baseline agents include:
1. Random Agent - selects actions randomly
2. Best Median Agent - only the agent with the best median performance raises hand
3. Ground Truth Agent - follows the actual agent assignments from the data

Usage:
    python run_baseline_evaluation.py [options]

Options:
    --episodes N        Number of evaluation episodes (default: 10)
    --seed N            Random seed for reproducibility (default: 42)
    --use-test-data     Use test data split instead of training data
    --include-mappo     Include trained MAPPO agent in comparison
    --model-path PATH   Path to trained MAPPO model (default: ./models/mappo_final)
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import custom modules
from src.CustomEnvironment.custom_environment import AgentOptimizerEnvironment
from src.CustomEnvironment.custom_environment.env.custom_environment import (
    SimulationParameters,
)
from src.MAPPO.agent import MAPPOAgent
from src.baselines import create_baseline_agents, BaselineEvaluator
from src.config import config
from src.display import print_colored
# Import shared utilities to avoid code duplication
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.utils import get_device, load_trained_mappo_agent

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


def run_baseline_evaluation(args):
    """
    Run baseline evaluation with specified parameters.

    Args:
        args: Command line arguments
    """
    print_colored("=" * 70, "yellow")
    print_colored("BASELINE EVALUATION", "yellow")
    print_colored("=" * 70, "yellow")

    # Load and preprocess data
    print_colored("Loading and preprocessing data...", "blue")
    data = load_data(config)
    data = remove_short_cases(data)

    # Use test data for evaluation to ensure fair comparison
    train, test = split_data(data)
    evaluation_data = test if args.use_test_data else train

    # Create timestamp for unique directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_dir = f"./experiments/baseline_evaluation_{timestamp}"
    os.makedirs(baseline_dir, exist_ok=True)
    print_colored(f"Baseline evaluation directory: {baseline_dir}", "green")

    # Create environment
    simulation_parameters = SimulationParameters(
        {"start_timestamp": evaluation_data["start_timestamp"].min()}
    )

    env = AgentOptimizerEnvironment(
        evaluation_data,
        simulation_parameters,
        experiment_dir=baseline_dir,
    )

    # Generate performance data for BestMedianAgent
    performance_data = create_performance_data(env)

    # Create baseline agents
    baseline_agents = create_baseline_agents(
        env,
        performance_data=performance_data,
        seed=args.seed,
    )

    # Unpack agents
    if len(baseline_agents) == 3:
        random_agent, best_median_agent, ground_truth_agent = baseline_agents
    else:
        random_agent, best_median_agent = baseline_agents
        # Just in case the create_baseline_agents function doesn't return ground_truth_agent
        ground_truth_agent = None

    # Load MAPPO agent if specified
    mappo_agent = None
    if args.include_mappo and args.model_path:
        mappo_agent = load_trained_mappo_agent(env, args.model_path, create_new_if_missing=True)

    # Create evaluator
    evaluator = BaselineEvaluator(env)

    # Define agents to compare
    agent_configs = [
        (random_agent, "Random Baseline"),
        (best_median_agent, "Best Median Baseline"),
    ]

    # Add ground truth agent if available
    if ground_truth_agent:
        agent_configs.append((ground_truth_agent, "Ground Truth Baseline"))

    # Add MAPPO agent if available
    if mappo_agent:
        agent_configs.append((mappo_agent, "MAPPO Agent (Trained)"))

    # Run comparison
    results = evaluator.compare_agents(agent_configs, num_episodes=args.episodes)

    # Save results
    results_file = os.path.join(baseline_dir, "baseline_comparison_results.json")
    evaluator.save_results(results_file)

    # Close environment
    env.close()

    print_colored("\nBaseline evaluation completed!", "green")
    print_colored(f"Results saved to: {results_file}", "green")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline evaluations for Agent Optimizer"
    )

    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--use-test-data",
        action="store_true",
        help="Use test data split instead of training data",
    )

    parser.add_argument(
        "--include-mappo",
        action="store_true",
        help="Include trained MAPPO agent in comparison",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/mappo_final",
        help="Path to trained MAPPO model (required if --include-mappo is used)",
    )

    args = parser.parse_args()
    run_baseline_evaluation(args)
