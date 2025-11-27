"""
Configuration parameters for PPO Trading Agent.
"""

from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class EnvironmentConfig:
    """Trading environment configuration."""
    ticker: str = "BTC-USD"
    interval: str = "15m"
    window_size: int = 60  # Lookback window
    
    # Trading costs
    # KEZDETBEN ALACSONY, hogy merjen kereskedni a modell!
    transaction_fee: float = 0.0005  # 0.05% 
    
    initial_balance: float = 10000.0
    
    # Actions: 0: Hold/Neutral, 1: Long
    n_actions: int = 2

@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    use_technical_indicators: bool = True
    use_time_features: bool = True
    
    clip_range: float = 5.0

@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 1
    features_dim: int = 128
    pi_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    vf_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"

@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

@dataclass
class TrainingConfig:
    """Training process configuration."""
    total_timesteps: int = 30000

@dataclass
class Config:
    """Main configuration class."""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    model_dir: str = "models"
    log_dir: str = "logs"
    data_dir: str = "data"
    
    def __post_init__(self):
        for dir_path in [self.model_dir, self.log_dir, self.data_dir]:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def default(cls):
        return cls()
