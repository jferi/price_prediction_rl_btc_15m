"""
Training Loop and Utilities.
"""

import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import numpy as np

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
        
        callback = TensorboardCallback()
        
        self.agent.learn(
            total_timesteps=self.config.training.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print("Training finished.")
        self.agent.save(os.path.join(self.config.model_dir, "ppo_final"))
