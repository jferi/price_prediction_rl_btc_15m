"""
Trading Environment compatible with Gymnasium.

Simplified for robust learning:
- Action Space: 2 (0: Cash/Flat, 1: Long)
- Reward: Scaled Log Returns
- Observation: Normalized window of features
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from .config import Config

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, raw_df: pd.DataFrame, config: Config, feature_engineer):
        super().__init__()
        self.df = df
        self.config = config
        self.feature_engineer = feature_engineer
        
        # Data handling
        # Features come from the SCALED dataframe
        self.features = df[feature_engineer.feature_cols].values.astype(np.float32)
        
        # Prices and Returns come from the RAW (unscaled) dataframe
        self.prices = raw_df['close'].values.astype(np.float32)
        self.log_returns = raw_df['log_return'].values.astype(np.float32)
        
        self.window_size = config.env.window_size
        self.max_steps = len(df) - 1

        # Action Space: 0=Cash, 1=Long
        self.action_space = spaces.Discrete(config.env.n_actions)

        # Observation Space: (Window, Features)
        self.observation_space = spaces.Box(
            low=-10, high=10, 
            shape=(self.window_size, len(feature_engineer.feature_cols)), 
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.position = 0  # 0: Cash, 1: Long
        self.portfolio_value = 0.0
        self.trades = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.position = 0
        self.portfolio_value = self.config.env.initial_balance
        self.trades = []
        
        return self._get_observation(), {}

    def step(self, action):
        # 1. Decision Logic
        current_price = self.prices[self.current_step]
        
        # Reward is based on the CHANGE from this step to next step given the action taken.
        # Use UNSCALED log returns for calculation
        step_log_return = self.log_returns[self.current_step + 1]
        
        # Transaction Cost Logic
        fee = 0.0
        if action != self.position:
            fee = self.config.env.transaction_fee
            self.trades.append({
                'step': self.current_step,
                'action': action,
                'price': current_price
            })
        
        reward = 0.0
        
        if action == 1: # Long
            self.portfolio_value *= np.exp(step_log_return)
            # Reward is the log return directly
            reward = step_log_return
        else: # Cash
            reward = 0.0
            
        # Deduct Fee from Portfolio and Reward
        self.portfolio_value *= (1 - fee)
        reward -= fee

        # Update State
        self.position = action
        self.current_step += 1
        
        # Check Termination
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        if self.portfolio_value < self.config.env.initial_balance * 0.5:
            terminated = True # Stop if lost 50%
            reward -= 1.0 # Big penalty for blowing up account

        # Reward Scaling
        scaled_reward = reward * 100.0 

        return self._get_observation(), scaled_reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Window slicing
        end = self.current_step + 1
        start = end - self.window_size
        return self.features[start:end]

    def _get_info(self):
        return {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'step': self.current_step
        }

    def render(self):
        print(f"Step: {self.current_step}, Value: {self.portfolio_value:.2f}, Pos: {self.position}")

def create_env(df, raw_df, config, feature_engineer):
    return TradingEnv(df, raw_df, config, feature_engineer)
