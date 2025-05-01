import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from .networks import ActorNetwork, CriticNetwork

class MAPPOAgent:
    def __init__(
        self,
        env,
        hidden_size=64,
        lr_actor=0.0003,
        lr_critic=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        batch_size=64,
        num_epochs=10,
        buffer_size=2048
    ):
        self.env = env
        self.n_agents = len(env.agents)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Create actor networks for each agent
        self.actors = {}
        for agent in env.agents:
            obs_space = env.observation_space(agent.id)
            action_space = env.action_space(agent.id)
            self.actors[agent.id] = ActorNetwork(obs_space, action_space, hidden_size)

        # Create centralized critic
        first_agent = env.agents[0]
        self.critic = CriticNetwork(env.observation_space(first_agent.id), self.n_agents, hidden_size)

        # Setup optimizers
        self.actor_optimizers = {
            agent_id: optim.Adam(actor.parameters(), lr=lr_actor)
            for agent_id, actor in self.actors.items()
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Experience buffer
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'action_probs': [],
            'advantages': [],
            'returns': []
        }

        # For tracking training performance
        self.episode_rewards = deque(maxlen=100)

    def select_actions(self, observations, deterministic=False):
        """Select actions for all agents based on their observations."""
        actions = {}
        action_probs = {}

        for agent_id, obs in observations.items():
            if agent_id in self.actors:
                action, probs = self.actors[agent_id].act(obs, deterministic)
                actions[agent_id] = action
                action_probs[agent_id] = probs

        return actions, action_probs

    def compute_values(self, observations):
        """Compute values for the current observations using the critic network."""
        # Convert observations dict to list in agent order
        obs_list = [observations[agent.id] for agent in self.env.agents]
        value = self.critic(obs_list).item()
        return value

    def store_experience(self, obs, actions, action_probs, rewards, dones, values):
        """Store experience in the buffer."""
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(actions)
        self.buffer['action_probs'].append(action_probs)
        self.buffer['rewards'].append(rewards)
        self.buffer['dones'].append(dones)
        self.buffer['values'].append(values)

    def compute_advantages_and_returns(self):
        """Compute GAE advantages and returns for stored trajectories."""
        values = np.array(self.buffer['values'])
        rewards = np.array(self.buffer['rewards'])
        dones = np.array(self.buffer['dones'])

        # Add a final value estimate for bootstrapping
        last_obs = self.buffer['obs'][-1]
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

        self.buffer['advantages'] = advantages
        self.buffer['returns'] = returns

    def update_policy(self):
        """Update policy and value networks using PPO."""
        # Skip if not enough data
        if len(self.buffer['obs']) < self.batch_size:
            return

        # Compute advantages and returns
        self.compute_advantages_and_returns()

        # Get buffer data
        observations = self.buffer['obs']
        actions = self.buffer['actions']
        old_action_probs = self.buffer['action_probs']
        returns = self.buffer['returns']
        advantages = self.buffer['advantages']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)

        # Perform multiple epochs of updates
        for _ in range(self.num_epochs):
            # Sample mini-batches
            indices = np.random.permutation(len(observations))

            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_obs = [observations[i] for i in batch_indices]
                batch_actions = [actions[i] for i in batch_indices]
                batch_old_probs = [old_action_probs[i] for i in batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Update critics (centralized)
                for i, obs_dict in enumerate(batch_obs):
                    obs_list = [obs_dict[agent.id] for agent in self.env.agents]
                    value_pred = self.critic(obs_list)
                    value_target = batch_returns[i]

                    critic_loss = ((value_pred - value_target) ** 2).mean()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                # Update actors (decentralized)
                for agent_id in self.actors:
                    actor = self.actors[agent_id]
                    optimizer = self.actor_optimizers[agent_id]

                    for i, obs_dict in enumerate(batch_obs):
                        if agent_id in obs_dict:
                            obs = obs_dict[agent_id]
                            action = batch_actions[i][agent_id]
                            old_action_prob = batch_old_probs[i][agent_id][action]
                            advantage = batch_advantages[i]

                            # Get current action probabilities
                            current_probs = actor(obs)
                            current_action_prob = current_probs[action]

                            # Compute ratio
                            ratio = current_action_prob / old_action_prob

                            # Compute surrogate losses
                            surrogate1 = ratio * advantage
                            surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                            # Actor loss
                            actor_loss = -torch.min(surrogate1, surrogate2)

                            # Update actor
                            optimizer.zero_grad()
                            actor_loss.backward()
                            optimizer.step()

        # Clear the buffer after updating
        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'action_probs': [],
            'advantages': [],
            'returns': []
        }

    def save_models(self, path):
        """Save model weights to the specified path."""
        # Save critic
        torch.save(self.critic.state_dict(), f"{path}/critic.pt")

        # Save actors
        for agent_id, actor in self.actors.items():
            torch.save(actor.state_dict(), f"{path}/actor_{agent_id}.pt")

    def load_models(self, path):
        """Load model weights from the specified path."""
        # Load critic
        self.critic.load_state_dict(torch.load(f"{path}/critic.pt"))

        # Load actors
        for agent_id, actor in self.actors.items():
            actor.load_state_dict(torch.load(f"{path}/actor_{agent_id}.pt"))