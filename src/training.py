"""
Training Loop and Utilities.
"""

import os
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import HParam
import numpy as np
from .visualization import DashboardCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log trades and portfolio value from environment info
        # We need to access the original environment. 
        # In VecEnv, it's inside .envs
        
        # Check if 'info' is available in locals
        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                if "portfolio_value" in info:
                    self.logger.record("trading/portfolio_value", info["portfolio_value"])
                if "position" in info:
                    self.logger.record("trading/position", info["position"])
        return True

class Trainer:
    def __init__(self, config, agent, train_env, val_env=None):
        self.config = config
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env

    def train(self):
        print(f"Starting training for {self.config.training.total_timesteps} timesteps...")
        
        # Combine callbacks
        tb_callback = TensorboardCallback()
        dash_callback = DashboardCallback(total_timesteps=self.config.training.total_timesteps)
        
        # Create callback list (Dashboard handles visualization, TB logs to file)
        callbacks = CallbackList([tb_callback, dash_callback])
        
        self.agent.learn(
            total_timesteps=self.config.training.total_timesteps,
            callback=callbacks,
            progress_bar=False # Disable default progress bar as Dashboard has one
        )
        
        print("Training finished.")
        self.agent.save(os.path.join(self.config.model_dir, "ppo_final"))
