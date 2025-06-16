import time
import argparse
import numpy as np
import os
from datetime import datetime

from CustomEnvironment.custom_environment import AgentOptimizerEnvironment
from CustomEnvironment.custom_environment.env.custom_environment import (
    SimulationParameters,
)
from MAPPO.agent import MAPPOAgent
from MAPPO.trainer import MAPPOTrainer

from config import config
from display import print_colored
from preprocessing.load_data import load_data, split_data
from preprocessing.preprocessing import remove_short_cases

import env_config


def train_mappo(env, experiment_dir: str, training_steps=100000):
    """Train a MAPPO agent on the environment."""
    # Initialize MAPPO agent
    mappo_agent = MAPPOAgent(
        env,
        hidden_size=64,
        lr_actor=0.0003,
        lr_critic=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
    )

    # Initialize trainer
    trainer = MAPPOTrainer(
        env,
        mappo_agent,
        total_epochs=20,
        should_eval=True,  # Enable evaluation
        experiment_dir=experiment_dir,  # Pass experiment directory
    )

    # Start training
    print_colored("Starting MAPPO training...", "green")
    trainer.train()

    return mappo_agent


def evaluate_agent(env, agent, episodes=10):
    """Evaluate a trained agent on the environment."""
    total_rewards = []

    for ep in range(episodes):
        observations, _ = env.reset()
        episode_reward = 0
        done = False

        print("Starting evaluation")
        print_colored(f"Episode {ep+1}/{episodes}", "blue")
        while not done:
            # Select actions using the policy
            actions, _ = agent.select_actions(observations, deterministic=True)

            # Execute actions
            observations, rewards, terminations, truncations, _ = env.step(actions)

            # Sum rewards across all agents
            step_reward = sum(rewards.values())
            episode_reward += step_reward

            # Check if episode is done
            done = any(list(terminations.values()) + list(truncations.values()))

            if env_config.DEBUG:
                env.render()

        env.reset()

        total_rewards.append(episode_reward)
        print_colored(f"Episode {ep+1} reward: {episode_reward:.2f}", "green")

    avg_reward = np.mean(total_rewards)
    print_colored(f"Average reward over {episodes} episodes: {avg_reward:.2f}", "green")

    return avg_reward


def main():
    """Main function to run the environment."""
    # Load and preprocess data
    data = load_data(config)
    data = remove_short_cases(data)
    train, test = split_data(data)

    simulation_parameters = SimulationParameters(
        {"start_timestamp": data["start_timestamp"].min()}
    )

    # Create timestamped directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/mappo_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize environment
    env = AgentOptimizerEnvironment(
        train,
        simulation_parameters,
        experiment_dir=experiment_dir,  # Pass experiment directory
    )

    # Train MAPPO agent
    agent = train_mappo(env, experiment_dir)

    # Save the trained agent
    agent.save_models(os.path.join(experiment_dir, "mappo_agent.pth"))

    # Test the agent
    test_env = AgentOptimizerEnvironment(
        test,
        simulation_parameters,
        experiment_dir=experiment_dir,  # Pass experiment directory
    )

    # Close the environment
    env.close()
    test_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the custom environment with MAPPO."
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Enable debugging output",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "random"],
        default="train",
        help="Mode to run: train, evaluate, or random actions",
    )
    parser.add_argument(
        "--steps", type=int, default=100000, help="Number of training steps"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/mappo_final",
        help="Path to load model from for evaluation",
    )

    args = parser.parse_args()
    env_config.DEBUG = args.debug

    if env_config.DEBUG:
        env_config.debug_print_colored("Debugging mode is enabled.", "green")

    main()
