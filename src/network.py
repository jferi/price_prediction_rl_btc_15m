"""
Neural network architecture for PPO agent.

Implements a custom LSTM-based feature extractor for time series data,
integrated with Stable-Baselines3's policy networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym

from .config import Config


class LSTMExtractor(BaseFeaturesExtractor):
    """
    Custom LSTM-based feature extractor for time series observations.
    
    Takes a sequence of normalized features and processes them through
    LSTM layers to capture temporal dependencies.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize the LSTM feature extractor.
        
        Args:
            observation_space: Gym observation space.
            features_dim: Output dimension of the feature extractor.
            lstm_hidden_size: Hidden size of LSTM layers.
            lstm_num_layers: Number of LSTM layers.
            dropout: Dropout probability.
        """
        super().__init__(observation_space, features_dim)
        
        # Get input dimensions from observation space
        # Shape is (window_size, n_features)
        self.window_size = observation_space.shape[0]
        self.n_input_features = observation_space.shape[1]
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.n_input_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)
        
        # Fully connected layer to reduce dimensions
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)
        
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                module.bias.data.fill_(0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM extractor.
        
        Args:
            observations: Tensor of shape (batch, window_size, n_features).
            
        Returns:
            Features tensor of shape (batch, features_dim).
        """
        # LSTM expects (batch, seq_len, features)
        # observations shape: (batch, window_size, n_features)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(observations)
        
        # Use the last hidden state
        # h_n shape: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Apply layer normalization
        normalized = self.layer_norm(last_hidden)
        
        # Final projection
        features = self.fc(normalized)
        
        return features


class GRUExtractor(BaseFeaturesExtractor):
    """
    Alternative GRU-based feature extractor.
    
    GRU is computationally lighter than LSTM while maintaining
    similar performance for many tasks.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize the GRU feature extractor.
        
        Args:
            observation_space: Gym observation space.
            features_dim: Output dimension of the feature extractor.
            gru_hidden_size: Hidden size of GRU layers.
            gru_num_layers: Number of GRU layers.
            dropout: Dropout probability.
        """
        super().__init__(observation_space, features_dim)
        
        self.window_size = observation_space.shape[0]
        self.n_input_features = observation_space.shape[1]
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.n_input_features,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(gru_hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size, features_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                module.bias.data.fill_(0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GRU extractor."""
        gru_out, h_n = self.gru(observations)
        last_hidden = h_n[-1]
        normalized = self.layer_norm(last_hidden)
        features = self.fc(normalized)
        return features


def create_lstm_policy(config: Config) -> Dict:
    """
    Create policy kwargs for PPO with LSTM feature extractor.
    
    Args:
        config: Configuration object with network settings.
        
    Returns:
        Dictionary of policy keyword arguments.
    """
    network_config = config.network
    
    policy_kwargs = {
        "features_extractor_class": LSTMExtractor,
        "features_extractor_kwargs": {
            "features_dim": network_config.feature_dim,
            "lstm_hidden_size": network_config.lstm_hidden_size,
            "lstm_num_layers": network_config.lstm_num_layers,
            "dropout": network_config.lstm_dropout
        },
        "net_arch": {
            "pi": network_config.pi_hidden_sizes,
            "vf": network_config.vf_hidden_sizes
        },
        "activation_fn": _get_activation_fn(network_config.activation),
        "share_features_extractor": True
    }
    
    return policy_kwargs


def create_gru_policy(config: Config) -> Dict:
    """
    Create policy kwargs for PPO with GRU feature extractor.
    
    Args:
        config: Configuration object with network settings.
        
    Returns:
        Dictionary of policy keyword arguments.
    """
    network_config = config.network
    
    policy_kwargs = {
        "features_extractor_class": GRUExtractor,
        "features_extractor_kwargs": {
            "features_dim": network_config.feature_dim,
            "gru_hidden_size": network_config.lstm_hidden_size,
            "gru_num_layers": network_config.lstm_num_layers,
            "dropout": network_config.lstm_dropout
        },
        "net_arch": {
            "pi": network_config.pi_hidden_sizes,
            "vf": network_config.vf_hidden_sizes
        },
        "activation_fn": _get_activation_fn(network_config.activation),
        "share_features_extractor": True
    }
    
    return policy_kwargs


def _get_activation_fn(name: str) -> Type[nn.Module]:
    """Get activation function by name."""
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU
    }
    return activations.get(name.lower(), nn.Tanh)

