"""
PPO Agent for trading.

Wraps Stable-Baselines3 PPO implementation with custom LSTM policy
and provides high-level interface for training and inference.
"""

import os
from typing import Optional, Dict, Any, Callable
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import Config
from .environment import TradingEnv
from .network import create_lstm_policy, create_gru_policy


class PPOAgent:
    """
    PPO-based trading agent.
    
    Encapsulates the PPO algorithm with custom LSTM/GRU policy
    for cryptocurrency trading.
    """
    
    def __init__(
        self,
        config: Config,
        env: TradingEnv,
        use_gru: bool = False
    ):
        """
        Initialize the PPO agent.
        
        Args:
            config: Configuration object.
            env: Trading environment.
            use_gru: Whether to use GRU instead of LSTM.
        """
        self.config = config
        self.ppo_config = config.ppo
        self.training_config = config.training
        
        # Wrap environment
        self.env = DummyVecEnv([lambda: env])
        
        # Create policy kwargs
        if use_gru:
            policy_kwargs = create_gru_policy(config)
        else:
            policy_kwargs = create_lstm_policy(config)
        
        # Initialize PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.ppo_config.learning_rate,
            n_steps=self.ppo_config.n_steps,
            batch_size=self.ppo_config.batch_size,
            n_epochs=self.ppo_config.n_epochs,
            gamma=self.ppo_config.gamma,
            gae_lambda=self.ppo_config.gae_lambda,
            clip_range=self.ppo_config.clip_range,
            clip_range_vf=self.ppo_config.clip_range_vf,
            ent_coef=self.ppo_config.ent_coef,
            vf_coef=self.ppo_config.vf_coef,
            max_grad_norm=self.ppo_config.max_grad_norm,
            target_kl=self.ppo_config.target_kl,
            policy_kwargs=policy_kwargs,
            verbose=self.training_config.verbose,
            seed=self.training_config.seed,
            tensorboard_log=config.log_dir
        )
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_env: Optional[TradingEnv] = None,
        callbacks: Optional[list] = None
    ) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps.
            eval_env: Environment for evaluation.
            callbacks: Additional callbacks.
        """
        if total_timesteps is None:
            total_timesteps = self.training_config.total_timesteps
        
        # Build callback list
        callback_list = []
        
        # Evaluation callback
        if eval_env is not None:
            eval_vec_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=self.config.model_dir,
                log_path=self.config.log_dir,
                eval_freq=self.training_config.eval_freq,
                n_eval_episodes=self.training_config.n_eval_episodes,
                deterministic=True,
                render=False
            )
            callback_list.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.training_config.save_freq,
            save_path=self.config.model_dir,
            name_prefix="ppo_trading"
        )
        callback_list.append(checkpoint_callback)
        
        # Add custom callbacks
        if callbacks:
            callback_list.extend(callbacks)
        
        # Training progress callback
        progress_callback = TrainingProgressCallback(
            log_interval=self.training_config.log_interval
        )
        callback_list.append(progress_callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callback_list) if callback_list else None,
            log_interval=self.training_config.log_interval,
            progress_bar=True
        )
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predict action for given observation.
        
        Args:
            observation: Current observation.
            deterministic: Whether to use deterministic policy.
            
        Returns:
            Tuple of (action, state).
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model.
        """
        if path is None:
            path = os.path.join(self.config.model_dir, "ppo_trading_final")
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model.
        """
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Config,
        env: TradingEnv
    ) -> "PPOAgent":
        """
        Load a pretrained agent.
        
        Args:
            path: Path to the saved model.
            config: Configuration object.
            env: Trading environment.
            
        Returns:
            PPOAgent instance with loaded model.
        """
        agent = cls(config, env)
        agent.load(path)
        return agent


class TrainingProgressCallback(BaseCallback):
    """
    Callback for logging training progress.
    
    Logs additional metrics like portfolio performance,
    trade statistics, and reward analysis.
    """
    
    def __init__(self, log_interval: int = 1, verbose: int = 0):
        """
        Initialize the callback.
        
        Args:
            log_interval: Interval for logging.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_episodes = 0
    
    def _on_step(self) -> bool:
        """Called after each step."""
        # Check for episode end
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    # Get info from environment
                    info = self.locals.get("infos", [{}])[idx]
                    
                    if "portfolio_value" in info:
                        self.logger.record(
                            "trading/portfolio_value",
                            info["portfolio_value"]
                        )
                    
                    if "total_trades" in info:
                        self.logger.record(
                            "trading/total_trades",
                            info["total_trades"]
                        )
                    
                    if "return_pct" in info:
                        self.logger.record(
                            "trading/return_pct",
                            info["return_pct"]
                        )
                    
                    self.n_episodes += 1
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        print(f"\nTraining completed. Total episodes: {self.n_episodes}")


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback based on evaluation performance.
    
    Stops training if no improvement is seen for a number of evaluations.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: int = 0
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of evaluations with no improvement.
            min_delta: Minimum improvement to qualify as improvement.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
    
    def _on_step(self) -> bool:
        """Check for early stopping condition."""
        if self.n_calls % 1000 == 0:  # Check periodically
            # Get evaluation results if available
            if hasattr(self.model, "_last_eval_reward"):
                current_reward = self.model._last_eval_reward
                
                if current_reward > self.best_mean_reward + self.min_delta:
                    self.best_mean_reward = current_reward
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print(f"Early stopping: no improvement for {self.patience} evaluations")
                    return False
        
        return True

