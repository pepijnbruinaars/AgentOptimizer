"""TensorBoard logging utilities for training."""

import os
from typing import Optional, Dict, Any
import numpy as np


class TensorBoardLogger:
    """Handles TensorBoard logging for training metrics.

    Provides a simple interface for logging scalars, histograms, and other
    metrics to TensorBoard. Can be disabled to avoid TensorBoard overhead.
    """

    def __init__(self, log_dir: str, enabled: bool = True):
        """Initialize the TensorBoard logger.

        Args:
            log_dir: Directory to store TensorBoard logs
            enabled: If True, initialize SummaryWriter; if False, all logging is no-op
        """
        self.enabled = enabled
        self.writer = None

        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                self.enabled = False
                print("Warning: tensorboard not installed, disabling TensorBoard logging")

    def log_episode_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log episode-level metrics.

        Args:
            episode: Episode number
            metrics: Dictionary containing:
                - reward: Episode reward
                - length: Episode length
                - avg_reward: Rolling average reward
                - avg_length: Rolling average length
                - cumulative_reward: Total cumulative reward
                - time: Episode duration in seconds
        """
        if not self.enabled or self.writer is None:
            return

        if "reward" in metrics:
            self.writer.add_scalar("episode/reward", metrics["reward"], episode)
        if "length" in metrics:
            self.writer.add_scalar("episode/length", metrics["length"], episode)
        if "avg_reward" in metrics:
            self.writer.add_scalar("episode/avg_reward", metrics["avg_reward"], episode)
        if "avg_length" in metrics:
            self.writer.add_scalar("episode/avg_length", metrics["avg_length"], episode)
        if "cumulative_reward" in metrics:
            self.writer.add_scalar(
                "episode/cumulative_reward", metrics["cumulative_reward"], episode
            )
        if "time" in metrics:
            self.writer.add_scalar("episode/time", metrics["time"], episode)

    def log_training_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log training-specific metrics (loss, epsilon, etc).

        Args:
            step: Training step/update count
            metrics: Dictionary containing algorithm-specific metrics:
                - For MAPPO: actor_loss, critic_loss
                - For QMIX: loss, epsilon, buffer_size
        """
        if not self.enabled or self.writer is None:
            return

        if "actor_loss" in metrics:
            self.writer.add_scalar("training/actor_loss", metrics["actor_loss"], step)
        if "critic_loss" in metrics:
            self.writer.add_scalar("training/critic_loss", metrics["critic_loss"], step)
        if "loss" in metrics:
            self.writer.add_scalar("training/loss", metrics["loss"], step)
        if "epsilon" in metrics:
            self.writer.add_scalar("training/epsilon", metrics["epsilon"], step)
        if "buffer_size" in metrics:
            self.writer.add_scalar("training/buffer_size", metrics["buffer_size"], step)
        if "target_updates" in metrics:
            self.writer.add_scalar("training/target_updates", metrics["target_updates"], step)

    def log_evaluation_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics.

        Args:
            episode: Episode number when evaluation occurred
            metrics: Dictionary containing:
                - reward: Evaluation reward
                - cumulative_reward: Evaluation cumulative reward
        """
        if not self.enabled or self.writer is None:
            return

        if "reward" in metrics:
            self.writer.add_scalar("eval/reward", metrics["reward"], episode)
        if "cumulative_reward" in metrics:
            self.writer.add_scalar("eval/cumulative_reward", metrics["cumulative_reward"], episode)

    def log_histograms(self, step: int, data: Dict[str, np.ndarray]) -> None:
        """Log histograms for action distributions, Q-values, etc.

        Args:
            step: Training step for histogram
            data: Dictionary of name -> array pairs to log as histograms
                Common keys:
                - action_probs: Action probability distribution
                - q_values: Q-value distribution
                - value_estimates: Value function estimates
        """
        if not self.enabled or self.writer is None:
            return

        for name, array in data.items():
            if isinstance(array, np.ndarray) and array.size > 0:
                self.writer.add_histogram(f"histograms/{name}", array, step)

    def flush(self) -> None:
        """Flush all pending writes to disk."""
        if self.enabled and self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
