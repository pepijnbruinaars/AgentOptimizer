from contextlib import contextmanager
from encodings.punycode import T
import time

import numpy as np
from CustomEnvironment.custom_environment import AgentOptimizerEnvironment
from CustomEnvironment.custom_environment.env.custom_environment import (
    SimulationParameters,
)

from config import config
from display import print_colored
from preprocessing.load_data import load_data
from preprocessing.preprocessing import remove_short_cases

import env_config
import argparse


def main() -> None:
    # Load data and show the first 20 rows for inspection
    data = load_data(config)
    simulation_parameters: SimulationParameters = {
        "start_timestamp": data["start_timestamp"].min()
    }
    preprocessed_data = remove_short_cases(data)

    # Create an instance of the custom environment
    env = AgentOptimizerEnvironment(
        preprocessed_data,
        simulation_parameters,
    )
    if env_config.DEBUG:
        env.render()

    # Perform a step in the environment
    i = 0
    start_time = time.perf_counter()
    try:
        while True:
            # Action selection using MAPPO
            # For simplicity, we use a random action here
            env_config.debug_print_colored(
                f"Step {i}, time: {env.current_time.time()}", "blue"
            )
            actions = {agent.id: np.random.choice([0, 1]) for agent in env.agents}
            # actions = {agent.id: 1 if agent.id == 1 else 0 for agent in env.agents}
            observations, rewards, terminations, truncations, _ = env.step(actions)

            if any(terminations.values()):
                print("Episode finished.")
                break
            if any(truncations.values()):
                print("Episode truncated.")
                break

            i += 1
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print_colored(f"\nElapsed time: {elapsed_time:.2f} seconds", "green")
        env.close()


DEBUG = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the custom environment.")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Enable debugging output",
    )
    args = parser.parse_args()
    env_config.DEBUG = args.debug

    env_config.debug_print_colored("Debugging mode is enabled.", "green")

    main()
