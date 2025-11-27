"""
Main Entry Point.
"""

import argparse
from src.config import Config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.environment import create_env
from src.agent import create_agent
from src.training import Trainer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30000)
    args = parser.parse_args()

    # 1. Config
    config = Config.default()
    config.training.total_timesteps = args.timesteps
    
    print("Configuration loaded.")
    print(f"Training for {config.training.total_timesteps} timesteps.")

    # 2. Data
    loader = DataLoader(config)
    df_raw_full = loader.download() 
    
    if len(df_raw_full) < config.env.window_size * 2:
        print("Not enough data downloaded.")
        return

    # 3. Features
    fe = FeatureEngineer(config)
    df_processed = fe.preprocess(df_raw_full)
    
    # Split
    split_idx = int(len(df_processed) * 0.8)
    
    # train_df contains unscaled features and log_returns
    train_df = df_processed.iloc[:split_idx]
    val_df = df_processed.iloc[split_idx:]
    
    # Fit scaler on TRAIN only
    fe.fit(train_df)
    
    # Scale both (this returns dataframe with scaled features)
    train_df_scaled = fe.scale(train_df)
    val_df_scaled = fe.scale(val_df)
    
    print(f"Training data: {len(train_df_scaled)} candles")
    print(f"Validation data: {len(val_df_scaled)} candles")

    # 4. Environment
    # Pass scaled features (for observation) AND unscaled processed data (for calculation)
    train_env = DummyVecEnv([lambda: create_env(train_df_scaled, train_df, config, fe)])
    val_env = DummyVecEnv([lambda: create_env(val_df_scaled, val_df, config, fe)])
    
    # Use VecNormalize for reward normalization
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # 5. Agent
    agent = create_agent(train_env, config)
    
    # 6. Train
    trainer = Trainer(config, agent, train_env, val_env)
    trainer.train()

if __name__ == "__main__":
    main()
