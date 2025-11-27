"""
PPO Trading Agent for BTC 15-minute candles.

This package implements a Proximal Policy Optimization (PPO) based
reinforcement learning agent for cryptocurrency trading.
"""

from .config import Config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .environment import TradingEnv
from .network import LSTMExtractor, create_lstm_policy
from .agent import PPOAgent
from .training import Trainer
from .evaluation import Evaluator

__version__ = "1.0.0"
__all__ = [
    "Config",
    "DataLoader", 
    "FeatureEngineer",
    "TradingEnv",
    "LSTMExtractor",
    "create_lstm_policy",
    "PPOAgent",
    "Trainer",
    "Evaluator",
]

