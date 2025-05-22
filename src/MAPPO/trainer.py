import numpy as np
import os
import time
from datetime import datetime

from .agent import MAPPOAgent


def map_actions_to_array(actions: dict[int, int]) -> np.ndarray:
    """Maps the actions output dict to an array with the nth key mapped to the nth index."""
    array = np.zeros(len(actions))

    for i, key in enumerate(actions.keys()):
        array[i] = np.array(actions[key])

    return array


def map_action_probs_to_array(
    action_probs: dict[int, np.ndarray],
) -> np.ndarray:
    """Maps the action probabilities output dict to an array with the nth key mapped to the nth index. This is a 3D matrix"""
    array = np.zeros((len(action_probs), len(action_probs[0])))

    for i, key in enumerate(action_probs.keys()):
        array[i] = np.array(np.array(action_probs[key]))

    return array


class MAPPOTrainer:
    def __init__(
        self,
        env,
        mappo_agent: MAPPOAgent,
        total_timesteps=1_000_000,
        total_episodes=20,
        eval_freq=10_000,
        save_freq=50_000,
        log_freq=1_000,
        eval_episodes=3,
        should_eval=True,
    ):
        self.env = env
        self.agent = mappo_agent
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.should_eval = should_eval
        self.eval_episodes = eval_episodes
        self.total_episodes = total_episodes

        # Create directories for saving models and logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"./models/mappo_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize tracking variables
        self.timesteps_done = 0
        self.episodes_done = 0
        self.best_eval_reward = -float("inf")

        # Tracking metrics
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def train(self) -> None:
        """Main training loop for MAPPO."""
        print(f"Starting MAPPO training for {self.total_timesteps} timesteps...")

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        start_time = time.perf_counter()
        episode_time = time.perf_counter()

        # Array containing the ids of the actions taken by the agents
        # Of shape n (steps) x m (agents)
        episode_actions: list[np.ndarray] = []
        # Array containg the probabilities of the actions for each agent at each step
        # Shape: n (steps) x m (agents) x k (probabilities)
        episode_action_probs: list[np.ndarray] = []

        while (
            self.episodes_done < self.total_episodes
            and self.timesteps_done < self.total_timesteps
        ):
            # Select actions using the current policy
            actions, action_probs = self.agent.select_actions(obs)
            episode_actions.append(map_actions_to_array(actions))
            episode_action_probs.append(map_action_probs_to_array(action_probs))

            # Get state value
            value = self.agent.compute_values(obs)

            # Take actions in the environment
            next_obs, rewards, dones, truncated, _ = self.env.step(actions)

            # Store experience
            # Convert dones and truncated dicts to a single "done" flag for the whole environment
            done = any(list(dones.values()) + list(truncated.values()))

            # Store the experience in the buffer
            self.agent.store_experience(
                obs, actions, action_probs, sum(rewards.values()), done, value
            )

            # Update episode tracking
            episode_reward += sum(rewards.values())
            episode_length += 1

            # Move to the next step
            obs = next_obs
            self.timesteps_done += 1

            if self.timesteps_done % self.log_freq == 0:
                # Log performance
                episode_time = time.perf_counter() - episode_time
                print(
                    f"Timestep {self.timesteps_done}/{self.total_timesteps} | "
                    f"Episode Reward: {episode_reward:.2f} | "
                    f"Episode Length: {episode_length} | "
                    f"Value: {value:.2f} | "
                    f"Time for episode: {episode_time:.2f} seconds"
                )
                episode_time = time.perf_counter()

            # Check if episode is done
            if done:
                # Update policy after each episode
                self.agent.update_policy()

                # Reset environment
                obs, _ = self.env.reset()

                # Log performance
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # Reset episode metrics
                episode_reward = 0
                episode_length = 0
                # Save episode actions and action probabilities to csv
                np.savetxt(
                    f"{self.save_dir}/episode_{self.episodes_done}_actions.csv",
                    episode_actions,
                    delimiter=";",
                )
                np.savetxt(
                    f"{self.save_dir}/episode_{self.episodes_done}_action_probs.csv",
                    episode_action_probs,
                    delimiter=";",
                )
                episode_actions = []
                episode_action_probs = []
                self.episodes_done += 1

                # Logging
                if self.episodes_done % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(
                        f"Episode {self.episodes_done} | Timestep {self.timesteps_done}/{self.total_timesteps} | "
                        f"Avg. Reward: {avg_reward:.2f} | Avg. Length: {avg_length:.2f}"
                    )

            # Periodic evaluation
            if self.should_eval and self.timesteps_done % self.eval_freq == 0:
                eval_reward = self.evaluate()
                print(
                    f"Evaluation at timestep {self.timesteps_done}: {eval_reward:.2f}"
                )

                # Save best model
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.agent.save_models(f"{self.save_dir}/best")
                    print(f"New best model saved with reward: {eval_reward:.2f}")

            # Periodic saving
            if self.timesteps_done % self.save_freq == 0:
                self.agent.save_models(
                    f"{self.save_dir}/checkpoint_{self.timesteps_done}"
                )
                print(f"Checkpoint saved at timestep {self.timesteps_done}")

        # Save final model
        self.agent.save_models(f"{self.save_dir}/final")
        print(
            f"Training completed after {self.episodes_done} episodes and {self.timesteps_done} timesteps."
        )
        print(f"Total time: {(time.perf_counter() - start_time) / 60:.2f} minutes")

    def evaluate(self, deterministic=True):
        """Evaluate the current policy."""
        eval_rewards = []
        print("starting evaluation")
        for _ in range(self.eval_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0

            iteration = 0

            while not done:
                # Select actions deterministically for evaluation
                actions, _ = self.agent.select_actions(obs, deterministic=deterministic)
                next_obs, rewards, dones, truncated, _ = self.env.step(actions)

                episode_reward += sum(rewards.values())

                # Check if episode is done
                done = any(list(dones.values()) + list(truncated.values()))
                obs = next_obs
                iteration += 1
                if iteration % 1000 == 0:
                    print(f"Evaluation iteration: {iteration}")

            eval_rewards.append(episode_reward)

        avg_reward = np.mean(eval_rewards)
        return avg_reward
