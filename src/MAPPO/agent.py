import time
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import os

from display import print_colored


from .networks import ActorNetwork, CriticNetwork


class MAPPOAgent:
    def __init__(
        self,
        env,
        hidden_size=256,
        lr_actor=0.0001,
        lr_critic=0.0001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.1,
        batch_size=1028,
        num_epochs=10,
        device=None,
        verbose=False,
    ):
        self.env = env
        self.n_agents = len(env.agents)
        self.env_agent_ids = [agent.id for agent in env.agents]
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose

        # Set device for GPU/MPS acceleration
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print_colored(f"MAPPOAgent using device: {self.device}", "blue")

        # Create actor networks for each agent
        self.actors = {}
        for agent in env.agents:
            obs_space = env.observation_space(agent.id)
            action_space = env.action_space(agent.id)
            self.actors[agent.id] = ActorNetwork(
                obs_space,
                action_space,
                hidden_size=hidden_size,
                device=self.device,
            ).to(self.device)

        # Create centralized critic
        first_agent = env.agents[0]
        self.critic = CriticNetwork(
            env.observation_space(first_agent.id),
            self.n_agents,
            hidden_size=2 * hidden_size,
            device=self.device,
        ).to(self.device)

        # Setup optimizers
        self.actor_optimizers = {
            agent_id: optim.Adam(actor.parameters(), lr=lr_actor)
            for agent_id, actor in self.actors.items()
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize models on CPU for simulation (will be moved to device during training)
        self._models_on_cpu = False  # Track current location of models
        self._move_models_to_cpu()  # Start with models on CPU for simulation

        # Experience buffer
        self.buffer = {
            "obs": [],
            "next_obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "action_probs": [],
            "advantages": [],
            "returns": [],
        }

        # For tracking training performance
        self.episode_rewards = deque(maxlen=100)

    def _log(self, text: str, color: str = "white") -> None:
        if self.verbose:
            print_colored(text, color)

    @property
    def _is_cpu_device(self) -> bool:
        return self.device.type == "cpu"

    def select_actions_and_value(self, observations, deterministic=False):
        """Select actions for all agents and compute critic value in one pass."""
        actions = {}
        action_probs = {}

        # Ensure models are on CPU for inference during episode collection
        self._move_models_to_cpu()

        for actor in self.actors.values():
            if actor.training:
                actor.eval()
        if self.critic.training:
            self.critic.eval()

        reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None
        obs_list = [observations[agent_id] for agent_id in self.env_agent_ids]

        with torch.no_grad():
            for agent_id, obs in observations.items():
                if agent_id in self.actors:
                    action, probs = self.actors[agent_id].act(obs, reward, deterministic)
                    actions[agent_id] = action
                    action_probs[agent_id] = probs

            value = self.critic(obs_list, reward).item()

        return actions, action_probs, value

    def select_actions(self, observations, deterministic=False):
        """Select actions for all agents based on their observations."""
        actions, action_probs, _ = self.select_actions_and_value(
            observations, deterministic=deterministic
        )
        return actions, action_probs

    def compute_values(self, observations):
        """Compute values for the current observations using the critic network."""
        obs_list = [observations[agent_id] for agent_id in self.env_agent_ids]
        # Get the reward from the last step
        reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None

        self._move_models_to_cpu()

        if self.critic.training:
            self.critic.eval()

        with torch.no_grad():
            value = self.critic(obs_list, reward).item()
        return value

    def _compute_values_on_current_device(self, observations):
        """Compute values using the critic network on its current device (for use during training)."""
        obs_list = [observations[agent_id] for agent_id in self.env_agent_ids]
        # Get the reward from the last step
        reward = self.buffer["rewards"][-1] if self.buffer["rewards"] else None

        with torch.no_grad():
            value = self.critic(obs_list, reward).item()
        return value

    def store_experience(
        self, obs, next_obs, actions, action_probs, rewards, dones, values
    ):
        """Store experience in the buffer."""
        self.buffer["obs"].append(obs)
        self.buffer["next_obs"].append(next_obs)
        self.buffer["actions"].append(actions)
        self.buffer["action_probs"].append(action_probs)
        self.buffer["rewards"].append(rewards)
        self.buffer["dones"].append(dones)
        self.buffer["values"].append(values)

    def reset_history(self):
        """Reset the history buffers for all networks."""
        for actor in self.actors.values():
            actor.reset_history()
        self.critic.reset_history()

    def compute_advantages_and_returns(self, use_current_device=False):
        """Compute GAE advantages and returns for stored trajectories."""
        values = np.array(self.buffer["values"])
        rewards = np.array(self.buffer["rewards"])
        dones = np.array(self.buffer["dones"])

        # Add a final value estimate for bootstrapping from the terminal next state.
        # For terminal trajectories, V(s_T) is zero and there is nothing to bootstrap.
        if len(dones) > 0 and bool(dones[-1]):
            last_value = 0.0
        else:
            last_obs = self.buffer["next_obs"][-1]
            if use_current_device:
                # During training, use the device where models currently are
                last_value = self._compute_values_on_current_device(last_obs)
            else:
                # During normal execution, ensure models are on CPU
                last_value = self.compute_values(last_obs)
        values = np.append(values, last_value)

        # Initialize advantages and returns
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        # Compute GAE advantages and returns
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        self.buffer["advantages"] = advantages  # type: ignore
        self.buffer["returns"] = returns  # type: ignore

    def update_policy(self):
        """Update policy and value networks using PPO with GPU/MPS acceleration."""
        # Skip if no data available
        if len(self.buffer["obs"]) == 0:
            return

        self._log(f"Updating policy on {self.device}...", "yellow")
        update_start_time = time.perf_counter()

        # Move models to device for training
        self._move_models_to_device()

        # Set models to training mode to enable dropout and batch norm updates
        for actor in self.actors.values():
            if not actor.training:
                actor.train()
        if not self.critic.training:
            self.critic.train()

        # Compute advantages and returns (using current device since models are now on GPU/MPS)
        self.compute_advantages_and_returns(use_current_device=True)

        # Get buffer data
        observations = self.buffer["obs"]
        actions = self.buffer["actions"]
        old_action_probs = self.buffer["action_probs"]
        returns = self.buffer["returns"]
        advantages = self.buffer["advantages"]

        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert data to tensors and move to device
        advantages_tensor = torch.tensor(
            np.asarray(advantages, dtype=np.float32), device=self.device
        )
        returns_tensor = torch.tensor(
            np.asarray(returns, dtype=np.float32), device=self.device
        )

        # Prepare action and action probability tensors
        action_tensors = {}
        old_action_prob_tensors = {}

        for agent_id in self.actors:
            action_tensors[agent_id] = torch.tensor(
                [step_actions[agent_id] for step_actions in actions],
                dtype=torch.long,
                device=self.device,
            )
            old_action_prob_tensors[agent_id] = torch.stack(
                [step_probs[agent_id] for step_probs in old_action_probs]
            ).to(self.device)

        self._log(
            f"Data preparation complete. Starting {self.num_epochs} epochs of training...",
            "cyan",
        )

        # Extract rewards once for faster batch indexing.
        rewards_buffer = self.buffer["rewards"]

        # Reset history before batch processing
        self.reset_history()

        for epoch in range(self.num_epochs):
            epoch_start_time = time.perf_counter()
            self._log(
                f"Epoch {epoch + 1}/{self.num_epochs} - Data size: {len(observations)}",
                "blue",
            )

            # Create random permutation for mini-batching
            indices = torch.randperm(len(observations), device=self.device)

            total_critic_loss = 0.0
            total_actor_losses = {agent_id: 0.0 for agent_id in self.actors}
            num_batches = 0
            total_batches = (
                len(observations) + self.batch_size - 1
            ) // self.batch_size  # Ceiling division

            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                num_batches += 1

                batch_indices_cpu = batch_indices.tolist()

                # Get batch data
                batch_obs = [observations[i] for i in batch_indices_cpu]
                batch_rewards = [rewards_buffer[i] for i in batch_indices_cpu]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Update critic (centralized) - optimized batch processing
                if len(batch_obs) > 0:
                    batch_obs_lists = [
                        [obs_dict[agent_id] for agent_id in self.env_agent_ids]
                        for obs_dict in batch_obs
                    ]

                    # Use batch forward pass for critic - single network call for entire batch
                    value_preds = self.critic.forward_batch(
                        batch_obs_lists, batch_rewards
                    )
                    critic_loss = (
                        (value_preds.squeeze() - batch_returns.squeeze()) ** 2
                    ).mean()
                    total_critic_loss += critic_loss.item()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=0.5
                    )
                    self.critic_optimizer.step()

                # Update actors (decentralized) - optimized batch processing
                for agent_id in self.actors:
                    actor = self.actors[agent_id]
                    optimizer = self.actor_optimizers[agent_id]

                    # Get batch data for this agent
                    batch_agent_actions = action_tensors[agent_id][batch_indices]
                    batch_old_action_probs = old_action_prob_tensors[agent_id][
                        batch_indices
                    ]

                    batch_agent_obs = [obs_dict[agent_id] for obs_dict in batch_obs]

                    # Use batch forward pass for actor - single network call for entire batch
                    current_action_probs = actor.forward_batch(
                        batch_agent_obs, batch_rewards
                    )

                    # Get probabilities for taken actions
                    current_action_prob_taken = current_action_probs.gather(
                        1, batch_agent_actions.unsqueeze(1)
                    ).squeeze(1)
                    old_action_prob_taken = batch_old_action_probs.gather(
                        1, batch_agent_actions.unsqueeze(1)
                    ).squeeze(1)

                    # Compute ratio
                    ratio = current_action_prob_taken / (old_action_prob_taken + 1e-8)

                    # Compute surrogate losses
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = (
                        torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        * batch_advantages
                    )

                    # Actor loss
                    actor_loss = -torch.min(surrogate1, surrogate2).mean()
                    total_actor_losses[agent_id] += actor_loss.item()

                    # Update actor
                    optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    optimizer.step()

                # Print progress less frequently now that batching is optimized
                if self.verbose and (
                    num_batches % max(1, total_batches // 3) == 0
                    or num_batches == total_batches
                ):
                    progress_pct = (num_batches / total_batches) * 100
                    elapsed_time = time.perf_counter() - epoch_start_time
                    avg_critic_loss_so_far = (
                        total_critic_loss / num_batches if num_batches > 0 else 0
                    )
                    self._log(
                        f"Epoch {epoch + 1} - "
                        f"Batch {num_batches}/{total_batches} ({progress_pct:.1f}%) - "
                        f"Time: {elapsed_time:.1f}s - Critic Loss: {avg_critic_loss_so_far:.6f}",
                        "purple",
                    )

            epoch_time = time.perf_counter() - epoch_start_time
            avg_critic_loss = total_critic_loss / num_batches if num_batches > 0 else 0
            self._log(
                f"Epoch {epoch + 1} complete - "
                f"Time: {epoch_time:.2f}s, Critic Loss: {avg_critic_loss:.6f}",
                "green",
            )

        total_update_time = time.perf_counter() - update_start_time
        self._log(f"Policy update complete - Total time: {total_update_time:.2f}s", "yellow")

        # Move models back to CPU for next simulation episodes
        self._move_models_to_cpu()

        # Clear the buffer after updating
        self.buffer = {
            "obs": [],
            "next_obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "action_probs": [],
            "advantages": [],
            "returns": [],
        }

    def save_models(self, path):
        """Save model weights to the specified path."""
        # Save critic
        # Check if directory exists, if not create it
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")

        # Save actors
        for agent_id, actor in self.actors.items():
            torch.save(actor.state_dict(), f"{path}/actor_{agent_id}.pt")

        # Save device info
        with open(f"{path}/device.txt", "w") as f:
            f.write(str(self.device))

    def load_models(self, path):
        """Load model weights from the specified path."""
        # Load critic
        self.critic.load_state_dict(
            torch.load(f"{path}/critic.pt", map_location=self.device)
        )

        # Load actors
        for agent_id, actor in self.actors.items():
            actor.load_state_dict(
                torch.load(f"{path}/actor_{agent_id}.pt", map_location=self.device)
            )

        # After loading, move models to CPU for simulation
        self._move_models_to_cpu()

    def _move_models_to_cpu(self):
        """Move all models to CPU for inference during simulation."""
        if self._is_cpu_device:
            self._models_on_cpu = True
            return

        if not hasattr(self, "_models_on_cpu") or not self._models_on_cpu:
            self._log("Moving models to CPU for inference...", "cyan")
            start_time = time.perf_counter()
            for actor in self.actors.values():
                actor.to(torch.device("cpu"))
            self.critic.to(torch.device("cpu"))
            transfer_time = time.perf_counter() - start_time
            self._log(f"Models moved to CPU in {transfer_time:.3f}s", "cyan")
            self._models_on_cpu = True

    def _move_models_to_device(self):
        """Move all models to GPU/MPS device for training."""
        if self._is_cpu_device:
            self._models_on_cpu = True
            return

        if not hasattr(self, "_models_on_cpu") or self._models_on_cpu:
            self._log(f"Moving models to {self.device} for training...", "cyan")
            start_time = time.perf_counter()
            for actor in self.actors.values():
                actor.to(self.device)
            self.critic.to(self.device)
            transfer_time = time.perf_counter() - start_time
            self._log(f"Models moved to {self.device} in {transfer_time:.3f}s", "cyan")
            self._models_on_cpu = False

    def _ensure_optimizers_on_device(self):
        """Ensure optimizer states are on the correct device."""
        # The optimizers will automatically move their states when the parameters move
        # This method exists for potential future use if manual intervention is needed
        pass
