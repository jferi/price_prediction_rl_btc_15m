"""
Configuration parameters for PPO Trading Agent.

Contains all hyperparameters and settings for the trading environment,
neural network architecture, and training process.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os


@dataclass
class EnvironmentConfig:
    """Trading environment configuration."""
    
    # Data settings
    ticker: str = "BTC-USD"
    interval: str = "15m"
    
    # Window size for observation (lookback period)
    window_size: int = 64
    
    # Trading settings
    initial_balance: float = 10000.0
    transaction_fee: float = 0.0007  # 0.07% (Binance fee)
    
    # Position settings (discrete action space)
    # 0: Hold, 1: Buy (Long), 2: Sell (Close position)
    n_actions: int = 3
    
    # Risk management
    max_position_size: float = 1.0  # Maximum position as fraction of portfolio
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    
    # Technical indicators periods
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Additional indicators
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    
    # Volume indicators
    use_volume_indicators: bool = True
    
    # Time features (sin/cos encoding)
    use_time_features: bool = True


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    
    # LSTM settings
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # Shared feature extractor
    feature_dim: int = 64
    
    # Policy and Value network hidden layers
    pi_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    vf_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    
    # Activation function
    activation: str = "tanh"


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    
    # Learning rate
    learning_rate: float = 3e-4
    
    # PPO specific
    n_steps: int = 2048  # Steps per rollout
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # Clipping
    clip_range: float = 0.2
    clip_range_vf: float = None  # None means no clipping
    
    # Entropy coefficient for exploration (increased for more exploration)
    ent_coef: float = 0.05
    
    # Value function coefficient
    vf_coef: float = 0.5
    
    # Max gradient norm
    max_grad_norm: float = 0.5
    
    # Target KL divergence
    target_kl: float = None


@dataclass
class RewardConfig:
    """Reward function configuration."""
    
    # Risk-adjusted reward settings
    risk_aversion_factor: float = 0.2
    
    # Sharpe ratio calculation window
    sharpe_window: int = 20
    
    # Penalty for holding cash (increased to encourage trading)
    hold_penalty: float = 0.001
    
    # Reward scaling (increased for clearer signal)
    reward_scaling: float = 100.0
    
    # Use differential Sharpe ratio
    use_differential_sharpe: bool = False


@dataclass
class TrainingConfig:
    """Training process configuration."""
    
    # Total training timesteps
    total_timesteps: int = 100_000
    
    # Evaluation frequency
    eval_freq: int = 5000
    n_eval_episodes: int = 5
    
    # Walk-forward validation settings
    use_walk_forward: bool = True
    train_window_days: int = 30  # Days for training
    test_window_days: int = 7   # Days for testing
    
    # Checkpointing
    save_freq: int = 10000
    
    # Logging
    log_interval: int = 1
    verbose: int = 1
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class Config:
    """Main configuration class combining all settings."""
    
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    model_dir: str = "models"
    log_dir: str = "logs"
    data_dir: str = "data"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.model_dir, self.log_dir, self.data_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)

