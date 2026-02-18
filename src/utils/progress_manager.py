"""Progress bar management for training loops using tqdm."""

import sys
from tqdm import tqdm


class ProgressBarManager:
    """Manages nested tqdm progress bars for episode and timestep tracking.

    Provides a clean interface for updating both episode-level and timestep-level
    progress bars without corrupting output.
    """

    def __init__(self, total_episodes: int, disable: bool = False):
        """Initialize the progress bar manager.

        Args:
            total_episodes: Total number of training episodes
            disable: If True, disable all progress bars (useful for non-TTY environments)
        """
        self.total_episodes = total_episodes
        self.disable = disable or not sys.stdout.isatty()
        self.episode_pbar = None
        self.timestep_pbar = None
        self.current_episode = 0

    def start_training(self) -> None:
        """Initialize the episode-level progress bar."""
        if not self.disable:
            self.episode_pbar = tqdm(
                total=self.total_episodes,
                desc="Episodes",
                position=0,
                leave=True,
                bar_format="{desc}: {n}/{total} [{percentage:3.0f}%] | {postfix}",
            )

    def start_episode(self, episode_num: int, max_steps: int = None) -> None:
        """Initialize the timestep-level progress bar for an episode.

        Args:
            episode_num: Current episode number
            max_steps: Maximum number of steps in this episode (if known)
        """
        self.current_episode = episode_num
        if not self.disable:
            total = max_steps if max_steps is not None else None
            self.timestep_pbar = tqdm(
                total=total,
                desc=f"  Timesteps",
                position=1,
                leave=False,
                bar_format="{desc}: {n}/{total} [{percentage:3.0f}%] | {postfix}",
            )

    def update_episode(self, metrics: dict) -> None:
        """Update episode progress bar with current metrics.

        Args:
            metrics: Dictionary containing metrics to display
                Expected keys: avg_reward, avg_length, time_elapsed
        """
        if not self.disable and self.episode_pbar is not None:
            postfix = []
            if "avg_reward" in metrics:
                postfix.append(f"Avg Reward: {metrics['avg_reward']:.2f}")
            if "avg_length" in metrics:
                postfix.append(f"Avg Length: {metrics['avg_length']:.1f}")
            if "time_elapsed" in metrics:
                postfix.append(f"Time: {metrics['time_elapsed']:.1f}s")

            self.episode_pbar.set_postfix_str(" | ".join(postfix))
            self.episode_pbar.update(1)

    def update_timestep(self, step_reward: float = None, cumulative_reward: float = None) -> None:
        """Update timestep progress bar.

        Args:
            step_reward: Reward for the current step
            cumulative_reward: Cumulative reward up to this point
        """
        if not self.disable and self.timestep_pbar is not None:
            postfix = []
            if step_reward is not None:
                postfix.append(f"Step Reward: {step_reward:.2f}")
            if cumulative_reward is not None:
                postfix.append(f"Cum. Reward: {cumulative_reward:.2f}")

            self.timestep_pbar.set_postfix_str(" | ".join(postfix))
            self.timestep_pbar.update(1)

    def finish_episode(self) -> None:
        """Close the timestep progress bar."""
        if not self.disable and self.timestep_pbar is not None:
            self.timestep_pbar.close()
            self.timestep_pbar = None

    def finish_training(self) -> None:
        """Close all progress bars."""
        self.finish_episode()
        if not self.disable and self.episode_pbar is not None:
            self.episode_pbar.close()
            self.episode_pbar = None
