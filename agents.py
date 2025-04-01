import torch
import torch.nn as nn
import numpy as np


class QLearningAgent(nn.Module):
    def __init__(self):
        super().__init__()

        num_actions = 5
        num_observations = 54

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(num_actions + num_observations, 32)),
            nn.ReLU(),
            self._layer_init(nn.Linear(32, 5)),
            nn.ReLU(),
            nn.Softmax(),
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize the neural network."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_action_value(self, observation, valid_actions):
        """Given the state-action, compute the action_values (Q-function)."""

        state_action = torch.cat(
            (torch.Tensor(observation["observation"]), torch.Tensor(valid_actions)), 0
        )

        return self.network(state_action)
