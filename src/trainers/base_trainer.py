"""Base trainer mixin providing common logging functionality."""

import os
from typing import Optional, Dict, Any
from utils.progress_manager import ProgressBarManager
from utils.tensorboard_logger import TensorBoardLogger


class TrainerLoggingMixin:
    """Mixin providing common logging interface for all trainers.

    This mixin should be inherited by MAPPOTrainer, MAPPOOnlineTrainer, and QMIXTrainer
    to provide consistent progress bar and TensorBoard logging functionality.
    """

    def setup_logging(
        self,
        experiment_dir: str,
        enable_tensorboard: bool = True,
        disable_progress: bool = False,
    ) -> None:
        """Initialize progress bars and TensorBoard logging.

        Args:
            experiment_dir: Root directory for the experiment
            enable_tensorboard: If True, enable TensorBoard logging
            disable_progress: If True, disable progress bars
        """
        # Initialize progress manager
        total_episodes = getattr(self, "total_training_episodes", 50)
        self.progress_manager = ProgressBarManager(total_episodes, disable=disable_progress)
        self.progress_manager.start_training()

        # Initialize TensorBoard logger
        tb_log_dir = os.path.join(experiment_dir, "tensorboard")
        self.tb_logger = TensorBoardLogger(tb_log_dir, enabled=enable_tensorboard)

    def log_episode_start(self, episode_num: int, max_steps: Optional[int] = None) -> None:
        """Called at the start of each episode.

        Args:
            episode_num: Current episode number
            max_steps: Maximum number of steps expected in this episode (if known)
        """
        if hasattr(self, "progress_manager"):
            self.progress_manager.start_episode(episode_num, max_steps)

    def log_timestep(
        self,
        timestep: int,
        step_reward: Optional[float] = None,
        cumulative_reward: Optional[float] = None,
    ) -> None:
        """Called during episode for timestep-level logging.

        Args:
            timestep: Current timestep in the episode
            step_reward: Reward for this timestep
            cumulative_reward: Cumulative reward up to this point
        """
        if hasattr(self, "progress_manager"):
            self.progress_manager.update_timestep(step_reward, cumulative_reward)

    def log_episode_end(self, episode_num: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each episode.

        Args:
            episode_num: Episode number that just completed
            metrics: Dictionary containing:
                - reward: Episode reward
                - length: Episode length
                - avg_reward: Rolling average reward
                - avg_length: Rolling average length
                - cumulative_reward: Total cumulative reward
                - time: Episode duration
                - (optional) actor_loss, critic_loss, loss, epsilon, etc.
        """
        # Log to TensorBoard
        if hasattr(self, "tb_logger"):
            # Extract common metrics
            episode_metrics = {
                k: v
                for k, v in metrics.items()
                if k
                in [
                    "reward",
                    "length",
                    "avg_reward",
                    "avg_length",
                    "cumulative_reward",
                    "time",
                ]
            }
            self.tb_logger.log_episode_metrics(episode_num, episode_metrics)

            # Extract training-specific metrics if present
            training_metrics = {
                k: v
                for k, v in metrics.items()
                if k
                in [
                    "actor_loss",
                    "critic_loss",
                    "loss",
                    "epsilon",
                    "buffer_size",
                    "target_updates",
                ]
            }
            if training_metrics:
                self.tb_logger.log_training_metrics(episode_num, training_metrics)

        # Update progress bar
        if hasattr(self, "progress_manager"):
            self.progress_manager.update_episode(
                {
                    "avg_reward": metrics.get("avg_reward"),
                    "avg_length": metrics.get("avg_length"),
                    "time_elapsed": metrics.get("time"),
                }
            )
            self.progress_manager.finish_episode()

    def log_evaluation(self, episode_num: int, eval_metrics: Dict[str, Any]) -> None:
        """Called after evaluation.

        Args:
            episode_num: Episode at which evaluation occurred
            eval_metrics: Dictionary containing:
                - reward: Evaluation reward
                - cumulative_reward: Evaluation cumulative reward
        """
        if hasattr(self, "tb_logger"):
            self.tb_logger.log_evaluation_metrics(episode_num, eval_metrics)

    def cleanup_logging(self) -> None:
        """Called at the end of training to close all logging resources."""
        if hasattr(self, "progress_manager"):
            self.progress_manager.finish_training()

        if hasattr(self, "tb_logger"):
            self.tb_logger.close()
