import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces

from utils.observation_utils import (
    process_observation_to_tensor,
    process_observation_batch_to_tensors,
    calculate_observation_size,
    get_observation_keys
)


class ActorNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=128, device=None):
        super(ActorNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")

        # Calculate input size from observation space and store keys
        self.obs_keys = get_observation_keys(obs_space)
        input_size = calculate_observation_size(obs_space)

        # Add reward history to input size
        input_size += 1  # For current reward

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # Action head for discrete action space
        self.action_head = nn.Linear(hidden_size, action_space.n)

    def forward(self, obs_dict, reward=None):
        # Process observation dictionary into a flat vector using shared utility
        current_device = next(self.parameters()).device
        x_parts = process_observation_to_tensor(obs_dict, self.obs_keys, current_device)

        # Add current reward to input (create as single tensor)
        reward_tensor = torch.tensor(
            [reward if reward is not None else 0.0],
            device=current_device,
            dtype=torch.float32,
        )
        x_parts.append(reward_tensor)

        x = torch.cat(x_parts)

        # Process through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)

        return action_probs

    def forward_batch(self, obs_dicts, rewards=None):
        """Batch forward pass for multiple observations."""
        # Get the current device of the model
        current_device = next(self.parameters()).device

        # Process all observations efficiently using shared utility
        x_batch = process_observation_batch_to_tensors(obs_dicts, self.obs_keys, current_device)

        # Add rewards to each observation in batch
        if rewards is not None:
            reward_tensors = torch.tensor(rewards, device=current_device, dtype=torch.float32).unsqueeze(1)
        else:
            reward_tensors = torch.zeros(len(obs_dicts), 1, device=current_device, dtype=torch.float32)
        
        # Concatenate observations with rewards
        x_batch = torch.cat([x_batch, reward_tensors], dim=1)

        # Process through network (batch processing)
        x = F.relu(self.fc1(x_batch))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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
    def __init__(self, obs_space, n_agents, hidden_size=256, device=None):
        super(CriticNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cpu")

        # For centralized critic, we combine observations from all agents
        # and potentially global state information

        # Calculate input size from observation space (for all agents)
        single_agent_obs_size = calculate_observation_size(obs_space)
        self.obs_keys = get_observation_keys(obs_space)

        # Total input size = single agent obs size * number of agents + reward
        input_size = single_agent_obs_size * n_agents + 1  # +1 for reward

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs_dicts, reward=None):
        # Process and concatenate observations from all agents using shared utility
        current_device = next(self.parameters()).device
        all_agent_inputs = []

        for agent_obs in obs_dicts:
            agent_parts = process_observation_to_tensor(agent_obs, self.obs_keys, current_device)
            if agent_parts:
                agent_input = torch.cat(agent_parts)
                all_agent_inputs.append(agent_input)

        # Concatenate all agents' observations
        x = torch.cat(all_agent_inputs)

        # Add current reward to input (single tensor creation)
        reward_tensor = torch.tensor(
            [reward if reward is not None else 0.0],
            device=current_device,
            dtype=torch.float32,
        )
        x = torch.cat([x, reward_tensor])

        # Process through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_head(x)

        return value

    def forward_batch(self, obs_dicts_batch, rewards=None):
        """Batch forward pass for multiple observation sets."""
        # Get the current device of the model
        current_device = next(self.parameters()).device

        # Process all observation sets in batch
        batch_inputs = []

        for i, obs_dicts in enumerate(obs_dicts_batch):
            # Process and concatenate observations from all agents for this sample using shared utility
            all_agent_inputs = []

            for agent_obs in obs_dicts:
                agent_parts = process_observation_to_tensor(agent_obs, self.obs_keys, torch.device("cpu"))
                if agent_parts:
                    # Convert back to CPU for concatenation
                    agent_parts_cpu = [part.cpu() if part.device != torch.device("cpu") else part for part in agent_parts]
                    agent_input = torch.cat(agent_parts_cpu)
                    all_agent_inputs.append(agent_input)

            # Concatenate all agents' observations for this sample
            x = torch.cat(all_agent_inputs)

            # Add current reward to input
            reward = rewards[i] if rewards is not None else 0.0
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            x = torch.cat([x, reward_tensor])

            batch_inputs.append(x)

        # Move entire batch to device at once and process
        x_batch = torch.stack(batch_inputs).to(current_device)

        # Process through network (batch processing)
        x = F.relu(self.fc1(x_batch))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        values = self.value_head(x)

        return values

    def reset_history(self):
        """Reset the reward history buffer."""
        pass
