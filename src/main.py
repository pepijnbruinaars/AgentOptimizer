import time
import argparse
import numpy as np

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


def train_mappo(env, training_steps=100000):
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
        total_timesteps=training_steps,
        eval_freq=5000,
        save_freq=20000,
        log_freq=1000,
        eval_episodes=1,
        should_eval=False,
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

        total_rewards.append(episode_reward)
        print_colored(f"Episode {ep+1} reward: {episode_reward:.2f}", "green")

    avg_reward = np.mean(total_rewards)
    print_colored(f"Average reward over {episodes} episodes: {avg_reward:.2f}", "green")

    return avg_reward


def main(args):
    """Main function to run the environment with MAPPO."""
    # Load and preprocess data
    data = load_data(config)
    preprocessed_data = remove_short_cases(data)
    train, test = split_data(preprocessed_data, split=0.8)

    simulation_parameters = SimulationParameters(
        {"start_timestamp": data["start_timestamp"].min()}
    )

    # Create environment

    if args.mode == "train":
        env = AgentOptimizerEnvironment(
            train,
            simulation_parameters,
        )

        # Train MAPPO agent
        agent = train_mappo(env, training_steps=args.steps)

        # Save the trained agent
        agent.save_models("./models/mappo_final")

        # Evaluate the trained agent
        evaluate_agent(env, agent, episodes=5)

        # Close the environment
        env.close()

    elif args.mode == "evaluate":
        env = AgentOptimizerEnvironment(
            test,
            simulation_parameters,
        )

        # Load trained agent
        agent = MAPPOAgent(env)
        agent.load_models(args.model_path)

        # Evaluate the agent
        evaluate_agent(env, agent, episodes=args.episodes)

        # Close the environment
        env.close()

    elif args.mode == "random":
        env = AgentOptimizerEnvironment(
            train,
            simulation_parameters,
        )
        # Run with random actions for baseline comparison
        i = 0
        start_time = time.perf_counter()
        try:
            observations, _ = env.reset()
            while True:
                # Random action selection
                env_config.debug_print_colored(
                    f"Step {i}, time: {env.current_time.time()}", "blue"
                )
                actions = {agent.id: np.random.choice([0, 1]) for agent in env.agents}
                observations, rewards, terminations, truncations, _ = env.step(actions)

                if any(terminations.values()):
                    print("Episode finished.")
                    break
                if any(truncations.values()):
                    print("Episode truncated.")
                    break

                i += 1

                if env_config.DEBUG:
                    env.render()
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print_colored(f"\nElapsed time: {elapsed_time:.2f} seconds", "green")

        # Close the environment
        env.close()


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

    main(args)
