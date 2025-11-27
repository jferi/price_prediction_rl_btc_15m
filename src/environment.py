"""
Trading environment for reinforcement learning.

Implements a Gymnasium-compatible environment for cryptocurrency trading
with realistic transaction costs and risk-adjusted rewards.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces

from .config import Config
from .feature_engineering import FeatureEngineer


class TradingEnv(gym.Env):
    """
    Cryptocurrency trading environment.
    
    Action Space:
        Discrete(3):
        - 0: Hold (do nothing)
        - 1: Buy (open/add to long position)
        - 2: Sell (close long position)
    
    Observation Space:
        Box with shape (window_size, n_features) containing
        normalized technical indicators and price features.
    
    Reward:
        Risk-adjusted returns with transaction cost penalties.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        feature_engineer: FeatureEngineer,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with OHLCV data and computed features.
            config: Configuration object.
            feature_engineer: Fitted FeatureEngineer instance.
            render_mode: Rendering mode.
        """
        super().__init__()
        
        self.config = config
        self.env_config = config.env
        self.reward_config = config.reward
        self.feature_engineer = feature_engineer
        self.render_mode = render_mode
        
        # Store data
        self.data = data.copy()
        self.n_steps = len(data)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.env_config.n_actions)
        
        obs_shape = feature_engineer.get_observation_shape()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Trading state
        self._reset_state()
        
        # For reward calculation
        self.returns_history = []
        
    def _reset_state(self) -> None:
        """Reset all trading state variables."""
        self.current_step = self.config.env.window_size
        self.balance = self.env_config.initial_balance
        self.position = 0.0  # Position in BTC
        self.entry_price = 0.0  # Average entry price
        self.total_trades = 0
        self.total_profit = 0.0
        self.returns_history = []
        self.portfolio_history = [self.env_config.initial_balance]
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        self._reset_state()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.
        
        Args:
            action: Trading action (0=Hold, 1=Buy, 2=Sell).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Get current price
        current_price = self._get_current_price()
        
        # Calculate portfolio value before action
        portfolio_before = self._get_portfolio_value(current_price)
        
        # Execute action
        self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        # Get new price and portfolio value
        new_price = self._get_current_price()
        portfolio_after = self._get_portfolio_value(new_price)
        
        # Store portfolio value
        self.portfolio_history.append(portfolio_after)
        
        # Calculate return
        step_return = (portfolio_after - portfolio_before) / portfolio_before
        self.returns_history.append(step_return)
        
        # Calculate reward
        reward = self._calculate_reward(action, step_return)
        
        # Check stop loss / take profit
        if self.position > 0:
            pnl_pct = (new_price - self.entry_price) / self.entry_price
            
            if pnl_pct <= -self.env_config.stop_loss_pct:
                # Stop loss triggered
                self._execute_action(2, new_price)  # Force sell
                reward -= 0.1  # Penalty for hitting stop loss
                
            elif pnl_pct >= self.env_config.take_profit_pct:
                # Take profit triggered
                self._execute_action(2, new_price)  # Force sell
                reward += 0.05  # Small bonus for hitting take profit
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, price: float) -> None:
        """
        Execute the trading action.
        
        Args:
            action: Trading action.
            price: Current price.
        """
        fee_rate = self.env_config.transaction_fee
        
        if action == 1:  # Buy
            if self.position == 0 and self.balance > 0:
                # Calculate position size (use all available balance)
                max_btc = self.balance / price
                position_size = max_btc * self.env_config.max_position_size
                
                # Apply transaction fee
                cost = position_size * price * (1 + fee_rate)
                
                if cost <= self.balance:
                    self.position = position_size
                    self.balance -= cost
                    self.entry_price = price
                    self.total_trades += 1
                    
        elif action == 2:  # Sell
            if self.position > 0:
                # Sell entire position
                proceeds = self.position * price * (1 - fee_rate)
                profit = proceeds - (self.position * self.entry_price)
                
                self.balance += proceeds
                self.total_profit += profit
                self.position = 0.0
                self.entry_price = 0.0
                self.total_trades += 1
    
    def _calculate_reward(self, action: int, step_return: float) -> float:
        """
        Calculate the reward for the current step.
        
        Uses direct price movement to encourage trading.
        
        Args:
            action: Taken action.
            step_return: Portfolio return for this step.
            
        Returns:
            Reward value.
        """
        current_price = self._get_current_price()
        reward = 0.0
        
        # Get previous price from data
        prev_idx = max(0, self.current_step - 1)
        prev_price = float(self.data['close'].iloc[prev_idx])
        price_change = (current_price - prev_price) / prev_price
        
        if self.position > 0:
            # We have a position - reward based on price movement
            reward = price_change * 100  # Profit/loss from holding
            
            # Bonus for selling at profit
            if action == 2:
                total_pnl = (current_price - self.entry_price) / self.entry_price
                if total_pnl > 0:
                    reward += 2.0  # Big bonus for profitable exit
                else:
                    reward -= 1.0  # Penalty for loss
        else:
            # No position
            if action == 1:
                # Buying - small reward for taking action
                reward = 0.1
            else:
                # Holding cash - penalize based on missed opportunity
                if price_change > 0:
                    # Price went up and we missed it!
                    reward = -price_change * 100  # Mirror the gain we missed
                else:
                    # Price went down - we avoided loss
                    reward = abs(price_change) * 20  # Small reward for avoiding loss
        
        # Always penalize holding cash slightly to encourage participation
        if self.position == 0 and action == 0:
            reward -= 0.05
        
        return float(reward)
    
    def _get_current_price(self) -> float:
        """Get current closing price."""
        return float(self.data['close'].iloc[self.current_step])
    
    def _get_portfolio_value(self, price: float) -> float:
        """Calculate total portfolio value."""
        return self.balance + (self.position * price)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = self.feature_engineer.create_observation_array(
            self.data, self.current_step
        )
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value(current_price)
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit,
            'current_price': current_price,
            'return_pct': (portfolio_value / self.env_config.initial_balance - 1) * 100
        }
    
    def render(self) -> None:
        """Render the environment state."""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Step {info['step']}: "
                  f"Portfolio=${info['portfolio_value']:.2f}, "
                  f"Position={info['position']:.6f} BTC, "
                  f"Return={info['return_pct']:.2f}%")
    
    def close(self) -> None:
        """Clean up resources."""
        pass


def create_env(
    data: pd.DataFrame,
    config: Config,
    feature_engineer: FeatureEngineer
) -> TradingEnv:
    """
    Factory function to create trading environment.
    
    Args:
        data: OHLCV data with computed features.
        config: Configuration object.
        feature_engineer: Fitted FeatureEngineer.
        
    Returns:
        TradingEnv instance.
    """
    return TradingEnv(data, config, feature_engineer)

