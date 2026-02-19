import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces


class ActorNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=128, device=None):
        super(ActorNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")

        # Calculate input size from observation space
        self.obs_input_size = 0
        self.obs_keys = []
        self.obs_key_sizes = {}

        # Process Dict observation space
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                key_size = 1
            elif isinstance(space, spaces.Box):
                key_size = int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                key_size = int(np.prod(space.shape))
            else:
                key_size = 1
            self.obs_key_sizes[key] = key_size
            self.obs_input_size += key_size

        # Add reward history to input size
        self.input_size = self.obs_input_size + 1  # +1 for current reward

        # Deeper network layers with dropout
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        # Action head for discrete action space
        self.action_head = nn.Linear(hidden_size, action_space.n)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias, 0, 0.01)

    def _write_obs_into(self, out: np.ndarray, obs_dict) -> None:
        """Write flattened observation values into a pre-allocated array."""
        cursor = 0
        for key in self.obs_keys:
            key_size = self.obs_key_sizes[key]
            value = obs_dict.get(key, None)

            if value is None:
                out[cursor : cursor + key_size] = 0.0
                cursor += key_size
                continue

            if isinstance(value, np.ndarray):
                flat = value.astype(np.float32, copy=False).reshape(-1)
            elif isinstance(value, (list, tuple)):
                flat = np.asarray(value, dtype=np.float32).reshape(-1)
            else:
                try:
                    flat = np.asarray([float(value)], dtype=np.float32)
                except (TypeError, ValueError):
                    flat = np.asarray([0.0], dtype=np.float32)

            n = min(key_size, flat.size)
            out[cursor : cursor + n] = flat[:n]
            if n < key_size:
                out[cursor + n : cursor + key_size] = 0.0
            cursor += key_size

    def forward(self, obs_dict, reward=None):
        current_device = next(self.parameters()).device
        x_np = np.empty(self.input_size, dtype=np.float32)
        self._write_obs_into(x_np[:-1], obs_dict)
        x_np[-1] = 0.0 if reward is None else float(reward)
        x = torch.from_numpy(x_np).to(current_device)

        # Process through deeper network with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        action_probs = F.softmax(self.action_head(x), dim=-1)

        return action_probs

    def forward_batch(self, obs_dicts, rewards=None):
        """Batch forward pass for multiple observations."""
        current_device = next(self.parameters()).device
        batch_size = len(obs_dicts)
        x_np = np.empty((batch_size, self.input_size), dtype=np.float32)

        for i, obs_dict in enumerate(obs_dicts):
            self._write_obs_into(x_np[i, :-1], obs_dict)
            x_np[i, -1] = (
                0.0 if rewards is None else float(rewards[i])
            )

        x_batch = torch.from_numpy(x_np).to(current_device)

        # Process through deeper network with dropout (batch processing)
        x = F.relu(self.fc1(x_batch))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        action_probs = F.softmax(self.action_head(x), dim=-1)

        return action_probs

    def act(self, obs, reward=None, deterministic=False):
        with torch.no_grad():
            action_probs = self(obs, reward)

            if deterministic:
                action = torch.argmax(action_probs).item()
            else:
                action = torch.multinomial(action_probs, 1).item()

            return action, action_probs

    def reset_history(self):
        """Reset the reward history buffer."""
        pass


class CriticNetwork(nn.Module):
    def __init__(self, obs_space, n_agents, hidden_size=128, device=None):
        super(CriticNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.n_agents = n_agents

        # For centralized critic, we combine observations from all agents
        # and potentially global state information

        # Calculate input size from observation space (for all agents)
        self.single_agent_obs_size = 0
        self.obs_keys = []
        self.obs_key_sizes = {}

        # Process Dict observation space
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                key_size = 1
            elif isinstance(space, spaces.Box):
                key_size = int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                key_size = int(np.prod(space.shape))
            else:
                key_size = 1
            self.obs_key_sizes[key] = key_size
            self.single_agent_obs_size += key_size

        # Total input size = single agent obs size * number of agents + reward
        self.input_size = self.single_agent_obs_size * n_agents + 1  # +1 for reward

        # Deeper network layers with dropout
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _write_agent_obs_into(self, out: np.ndarray, agent_obs) -> None:
        """Write one agent observation into a pre-allocated slice."""
        cursor = 0
        for key in self.obs_keys:
            key_size = self.obs_key_sizes[key]
            value = agent_obs.get(key, None) if agent_obs is not None else None

            if value is None:
                out[cursor : cursor + key_size] = 0.0
                cursor += key_size
                continue

            if isinstance(value, np.ndarray):
                flat = value.astype(np.float32, copy=False).reshape(-1)
            elif isinstance(value, (list, tuple)):
                flat = np.asarray(value, dtype=np.float32).reshape(-1)
            else:
                try:
                    flat = np.asarray([float(value)], dtype=np.float32)
                except (TypeError, ValueError):
                    flat = np.asarray([0.0], dtype=np.float32)

            n = min(key_size, flat.size)
            out[cursor : cursor + n] = flat[:n]
            if n < key_size:
                out[cursor + n : cursor + key_size] = 0.0
            cursor += key_size

    def forward(self, obs_dicts, reward=None):
        current_device = next(self.parameters()).device
        x_np = np.empty(self.input_size, dtype=np.float32)

        for agent_idx in range(self.n_agents):
            start = agent_idx * self.single_agent_obs_size
            end = start + self.single_agent_obs_size
            agent_obs = obs_dicts[agent_idx] if agent_idx < len(obs_dicts) else None
            self._write_agent_obs_into(x_np[start:end], agent_obs)

        x_np[-1] = 0.0 if reward is None else float(reward)
        x = torch.from_numpy(x_np).to(current_device)

        # Process through deeper network with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        value = self.value_head(x)

        return value

    def forward_batch(self, obs_dicts_batch, rewards=None):
        """Batch forward pass for multiple observation sets."""
        current_device = next(self.parameters()).device
        batch_size = len(obs_dicts_batch)
        x_np = np.empty((batch_size, self.input_size), dtype=np.float32)

        for i, obs_dicts in enumerate(obs_dicts_batch):
            for agent_idx in range(self.n_agents):
                start = agent_idx * self.single_agent_obs_size
                end = start + self.single_agent_obs_size
                agent_obs = obs_dicts[agent_idx] if agent_idx < len(obs_dicts) else None
                self._write_agent_obs_into(x_np[i, start:end], agent_obs)

            x_np[i, -1] = 0.0 if rewards is None else float(rewards[i])

        x_batch = torch.from_numpy(x_np).to(current_device)

        # Process through deeper network with dropout (batch processing)
        x = F.relu(self.fc1(x_batch))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        values = self.value_head(x)

        return values

    def reset_history(self):
        """Reset the reward history buffer."""
        pass
