"""
PPO Agent implementation using Stable-Baselines3.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn

from .config import Config
from .network import LSTMExtractor # Feltételezzük, hogy ez megmarad, vagy egyszerűsítjük

def create_agent(env, config: Config, verbose=1):
    """
    Creates and returns a PPO agent.
    """
    
    # Custom Network Architecture defined in config
    policy_kwargs = dict(
        features_extractor_class=LSTMExtractor,
        features_extractor_kwargs=dict(
            features_dim=config.network.features_dim,
            lstm_hidden_size=config.network.lstm_hidden_size,
            lstm_num_layers=config.network.lstm_num_layers,
        ),
        net_arch=dict(
            pi=config.network.pi_hidden_sizes,
            vf=config.network.vf_hidden_sizes
        ),
        activation_fn=nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=config.log_dir
    )
    
    return model
