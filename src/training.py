"""
Training pipeline for PPO trading agent.

Implements the training loop with walk-forward validation,
model checkpointing, and performance tracking.
"""

import os
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from .config import Config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .environment import TradingEnv, create_env
from .agent import PPOAgent


class Trainer:
    """
    Training pipeline for PPO trading agent.
    
    Handles data preparation, environment creation, model training,
    and walk-forward validation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or Config.default()
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Store training history
        self.training_history: List[Dict[str, Any]] = []
    
    def prepare_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download and prepare data for training.
        
        Args:
            start_date: Start date for data.
            end_date: End date for data.
            
        Returns:
            Prepared DataFrame with features.
        """
        # Download data
        raw_data = self.data_loader.download(start_date, end_date)
        
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Date range: {raw_data.index[0]} to {raw_data.index[-1]}")
        
        # Compute features
        data = self.feature_engineer.compute_features(raw_data)
        
        print(f"Data with features shape: {data.shape}")
        print(f"Features: {self.feature_engineer.get_feature_columns()}")
        
        return data
    
    def train_single(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        total_timesteps: Optional[int] = None
    ) -> Tuple[PPOAgent, Dict[str, Any]]:
        """
        Train on a single data split.
        
        Args:
            train_data: Training data.
            val_data: Validation data (optional).
            total_timesteps: Training timesteps.
            
        Returns:
            Tuple of (trained agent, training metrics).
        """
        # Fit feature engineer on training data
        self.feature_engineer.fit(train_data)
        
        # Normalize data
        train_normalized = self.feature_engineer.normalize(train_data)
        
        # Create training environment
        train_env = create_env(train_normalized, self.config, self.feature_engineer)
        
        # Create validation environment if provided
        val_env = None
        if val_data is not None:
            val_normalized = self.feature_engineer.normalize(val_data)
            val_env = create_env(val_normalized, self.config, self.feature_engineer)
        
        # Create agent
        agent = PPOAgent(self.config, train_env)
        
        # Train
        print(f"\nStarting training for {total_timesteps or self.config.training.total_timesteps} timesteps...")
        agent.train(
            total_timesteps=total_timesteps,
            eval_env=val_env
        )
        
        # Evaluate on training data
        train_metrics = self._evaluate_on_data(agent, train_normalized, "train")
        
        # Evaluate on validation data
        val_metrics = {}
        if val_data is not None:
            val_metrics = self._evaluate_on_data(agent, val_normalized, "val")
        
        metrics = {**train_metrics, **val_metrics}
        
        return agent, metrics
    
    def train_walk_forward(
        self,
        data: pd.DataFrame,
        timesteps_per_split: Optional[int] = None
    ) -> List[Tuple[PPOAgent, Dict[str, Any]]]:
        """
        Train using walk-forward validation.
        
        This trains multiple models on rolling windows of data,
        simulating realistic forward testing.
        
        Args:
            data: Full dataset.
            timesteps_per_split: Timesteps for each training split.
            
        Returns:
            List of (agent, metrics) tuples for each split.
        """
        results = []
        
        # Get walk-forward splits
        splits = self.data_loader.get_walk_forward_splits(
            data,
            train_days=self.config.training.train_window_days,
            test_days=self.config.training.test_window_days
        )
        
        if not splits:
            print("Not enough data for walk-forward validation. Using single split.")
            train_data, val_data, test_data = self.data_loader.split_data(data)
            agent, metrics = self.train_single(train_data, val_data, timesteps_per_split)
            
            # Evaluate on test data
            test_normalized = self.feature_engineer.normalize(test_data)
            test_metrics = self._evaluate_on_data(agent, test_normalized, "test")
            metrics.update(test_metrics)
            
            results.append((agent, metrics))
            return results
        
        # Train on each split
        for i, (train_data, test_data) in enumerate(splits):
            print(f"\n{'='*60}")
            print(f"Walk-Forward Split {i+1}/{len(splits)}")
            print(f"Train period: {train_data.index[0]} to {train_data.index[-1]}")
            print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
            print(f"{'='*60}")
            
            # Fit on training data
            self.feature_engineer.fit(train_data)
            
            # Normalize
            train_normalized = self.feature_engineer.normalize(train_data)
            test_normalized = self.feature_engineer.normalize(test_data)
            
            # Create environments
            train_env = create_env(train_normalized, self.config, self.feature_engineer)
            test_env = create_env(test_normalized, self.config, self.feature_engineer)
            
            # Create and train agent
            agent = PPOAgent(self.config, train_env)
            agent.train(
                total_timesteps=timesteps_per_split,
                eval_env=test_env
            )
            
            # Evaluate
            train_metrics = self._evaluate_on_data(agent, train_normalized, "train")
            test_metrics = self._evaluate_on_data(agent, test_normalized, "test")
            
            metrics = {
                "split": i + 1,
                "train_start": str(train_data.index[0]),
                "train_end": str(train_data.index[-1]),
                "test_start": str(test_data.index[0]),
                "test_end": str(test_data.index[-1]),
                **train_metrics,
                **test_metrics
            }
            
            results.append((agent, metrics))
            self.training_history.append(metrics)
            
            # Save model for this split
            model_path = os.path.join(
                self.config.model_dir,
                f"ppo_trading_split_{i+1}"
            )
            agent.save(model_path)
        
        # Print summary
        self._print_walk_forward_summary(results)
        
        return results
    
    def _evaluate_on_data(
        self,
        agent: PPOAgent,
        data: pd.DataFrame,
        prefix: str
    ) -> Dict[str, Any]:
        """
        Evaluate agent on given data.
        
        Args:
            agent: Trained agent.
            data: Normalized data for evaluation.
            prefix: Prefix for metric keys.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        env = create_env(data, self.config, self.feature_engineer)
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Calculate additional metrics
        portfolio_values = env.portfolio_history
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = {
            f"{prefix}_total_reward": total_reward,
            f"{prefix}_final_portfolio": info["portfolio_value"],
            f"{prefix}_return_pct": info["return_pct"],
            f"{prefix}_total_trades": info["total_trades"],
            f"{prefix}_sharpe_ratio": self._calculate_sharpe(returns),
            f"{prefix}_max_drawdown": self._calculate_max_drawdown(portfolio_values),
            f"{prefix}_win_rate": self._calculate_win_rate(returns)
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / (365 * 96)  # 15-min periods
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (96 periods per day * 365 days)
        annualization_factor = np.sqrt(96 * 365)
        sharpe = (mean_return / std_return) * annualization_factor
        
        return float(sharpe)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return float(np.max(drawdown) * 100)
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate percentage of positive returns."""
        if len(returns) == 0:
            return 0.0
        
        positive_returns = np.sum(returns > 0)
        return float(positive_returns / len(returns) * 100)
    
    def _print_walk_forward_summary(
        self,
        results: List[Tuple[PPOAgent, Dict[str, Any]]]
    ) -> None:
        """Print summary of walk-forward validation results."""
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*60)
        
        test_returns = [r[1].get("test_return_pct", 0) for r in results]
        test_sharpes = [r[1].get("test_sharpe_ratio", 0) for r in results]
        test_drawdowns = [r[1].get("test_max_drawdown", 0) for r in results]
        
        print(f"\nNumber of splits: {len(results)}")
        print(f"\nTest Returns (%):")
        print(f"  Mean: {np.mean(test_returns):.2f}")
        print(f"  Std:  {np.std(test_returns):.2f}")
        print(f"  Min:  {np.min(test_returns):.2f}")
        print(f"  Max:  {np.max(test_returns):.2f}")
        
        print(f"\nTest Sharpe Ratios:")
        print(f"  Mean: {np.mean(test_sharpes):.2f}")
        print(f"  Std:  {np.std(test_sharpes):.2f}")
        
        print(f"\nTest Max Drawdowns (%):")
        print(f"  Mean: {np.mean(test_drawdowns):.2f}")
        print(f"  Max:  {np.max(test_drawdowns):.2f}")
        
        print("="*60)
    
    def save_training_history(self, path: Optional[str] = None) -> None:
        """Save training history to CSV."""
        if not self.training_history:
            print("No training history to save.")
            return
        
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.config.log_dir, f"training_history_{timestamp}.csv")
        
        df = pd.DataFrame(self.training_history)
        df.to_csv(path, index=False)
        print(f"Training history saved to {path}")

