import numpy as np
import os
import time
from datetime import datetime


class MAPPOTrainer:
    def __init__(
        self,
        env,
        mappo_agent,
        total_timesteps=1_000_000,
        eval_freq=10_000,
        save_freq=50_000,
        log_freq=1_000,
        eval_episodes=3,
    ):
        self.env = env
        self.agent = mappo_agent
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.eval_episodes = eval_episodes

        # Create directories for saving models and logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"./models/mappo_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize tracking variables
        self.timesteps_done = 0
        self.episodes_done = 0
        self.best_eval_reward = -float("inf")

        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []

    def train(self):
        """Main training loop for MAPPO."""
        print(f"Starting MAPPO training for {self.total_timesteps} timesteps...")

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        start_time = time.time()

        while self.timesteps_done < self.total_timesteps:
            # Select actions using the current policy
            actions, action_probs = self.agent.select_actions(obs)

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

            # if self.timesteps_done % self.log_freq == 0:
            #     print(f"Timestep {self.timesteps_done}/{self.total_timesteps} | "
            #           f"Episode Reward: {episode_reward:.2f} | "
            #           f"Episode Length: {episode_length} | "
            #           f"Value: {value:.2f}")
            print(self.timesteps_done)

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
            if self.timesteps_done % self.eval_freq == 0:
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
        print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")

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
