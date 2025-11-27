"""
Neural Network definitions.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM Feature Extractor for Time Series.
    Input: (Batch, Window, Features)
    Output: (Batch, Features_Dim)
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, 
                 lstm_hidden_size: int = 256, lstm_num_layers: int = 1):
        
        # Calculate expected input size from observation space
        # Obs space shape is (Window, N_Features)
        input_features = observation_space.shape[1]
        
        super().__init__(observation_space, features_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        self.linear = nn.Linear(lstm_hidden_size, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations shape: (Batch, Window, Features)
        lstm_out, _ = self.lstm(observations)
        
        # Take the last time step output
        # lstm_out shape: (Batch, Window, Hidden)
        last_out = lstm_out[:, -1, :]
        
        return self.relu(self.linear(last_out))
