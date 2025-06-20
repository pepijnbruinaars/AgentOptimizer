import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces


class ActorNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=128):
        super(ActorNetwork, self).__init__()

        # Calculate input size from observation space
        input_size = 0
        self.obs_keys = []

        # Process Dict observation space
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                input_size += 1
            elif isinstance(space, spaces.Box):
                input_size += int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                input_size += int(np.prod(space.shape))

        # Add reward history to input size
        input_size += 1  # For current reward
        
        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # Action head for discrete action space
        self.action_head = nn.Linear(hidden_size, action_space.n)

    def forward(self, obs_dict, reward=None):
        # Process observation dictionary into a flat vector
        x_parts = []

        for key in self.obs_keys:
            if key in obs_dict:
                # Handle different observation components
                if isinstance(obs_dict[key], np.ndarray):
                    x_parts.append(torch.FloatTensor(obs_dict[key].flatten()))
                else:
                    # Handle scalar values or other types
                    x_parts.append(torch.FloatTensor([obs_dict[key]]))

        # Add current reward to input
        if reward is not None:
            x_parts.append(torch.FloatTensor([reward]))
        else:
            x_parts.append(torch.FloatTensor([0.0]))

        x = torch.cat(x_parts)

        # Process through network
        x = F.relu(self.fc1(x))
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
    def __init__(self, obs_space, n_agents, hidden_size=256):
        super(CriticNetwork, self).__init__()

        # For centralized critic, we combine observations from all agents
        # and potentially global state information

        # Calculate input size from observation space (for all agents)
        single_agent_obs_size = 0
        self.obs_keys = []

        # Process Dict observation space
        for key, space in obs_space.items():
            self.obs_keys.append(key)
            if isinstance(space, spaces.Discrete):
                single_agent_obs_size += 1  # One-hot encoding
            elif isinstance(space, spaces.Box):
                single_agent_obs_size += int(np.prod(space.shape))
            elif isinstance(space, spaces.MultiBinary):
                single_agent_obs_size += int(np.prod(space.shape))

        # Total input size = single agent obs size * number of agents + reward
        input_size = single_agent_obs_size * n_agents + 1  # +1 for reward
        
        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs_dicts, reward=None):
        # Process and concatenate observations from all agents
        all_agent_inputs = []

        for agent_obs in obs_dicts:
            agent_parts = []
            for key in self.obs_keys:
                if key in agent_obs:
                    # Handle different observation components
                    if isinstance(agent_obs[key], np.ndarray):
                        agent_parts.append(torch.FloatTensor(agent_obs[key].flatten()))
                    else:
                        # Handle scalar values
                        agent_parts.append(torch.FloatTensor([agent_obs[key]]))

            if agent_parts:
                agent_input = torch.cat(agent_parts)
                all_agent_inputs.append(agent_input)

        # Concatenate all agents' observations
        x = torch.cat(all_agent_inputs)
        
        # Add current reward to input
        if reward is not None:
            x = torch.cat([x, torch.FloatTensor([reward])])
        else:
            x = torch.cat([x, torch.FloatTensor([0.0])])

        # Process through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_head(x)

        return value

    def reset_history(self):
        """Reset the reward history buffer."""
        pass
