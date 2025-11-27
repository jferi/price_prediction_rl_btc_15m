#!/usr/bin/env python3
"""
PPO Trading Agent for BTC 15-minute candles.

Main entry point for training and evaluating a PPO-based
reinforcement learning agent for cryptocurrency trading.

Usage:
    python main.py                    # Train with default settings
    python main.py --timesteps 50000  # Train for 50k timesteps
    python main.py --eval-only        # Evaluate pretrained model
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from src.config import Config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.environment import create_env
from src.agent import PPOAgent
from src.training import Trainer
from src.evaluation import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO Trading Agent for BTC 15-minute candles"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000)"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate pretrained model, skip training"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pretrained model for evaluation"
    )
    
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward validation"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--use-gru",
        action="store_true",
        help="Use GRU instead of LSTM"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plotting (useful for headless environments)"
    )
    
    return parser.parse_args()


def setup_config(args) -> Config:
    """Setup configuration based on arguments."""
    config = Config.default()
    
    # Update training settings
    config.training.total_timesteps = args.timesteps
    config.training.seed = args.seed
    
    # For faster iteration during development, reduce some settings
    if args.timesteps < 100000:
        config.ppo.n_steps = 1024
        config.training.eval_freq = 2500
        config.training.save_freq = 5000
    
    return config


def main():
    """Main function to run training and evaluation."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("PPO TRADING AGENT FOR BTC 15-MINUTE CANDLES")
    print("="*60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup configuration
    config = setup_config(args)
    
    print(f"\nConfiguration:")
    print(f"  Training timesteps: {config.training.total_timesteps:,}")
    print(f"  Window size: {config.env.window_size}")
    print(f"  Transaction fee: {config.env.transaction_fee*100:.2f}%")
    print(f"  Random seed: {config.training.seed}")
    print(f"  Using {'GRU' if args.use_gru else 'LSTM'} network")
    
    # Set random seeds
    np.random.seed(config.training.seed)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Prepare data
    print("\n" + "-"*60)
    print("LOADING DATA")
    print("-"*60)
    
    data = trainer.prepare_data()
    
    if len(data) < config.env.window_size * 2:
        print(f"Error: Not enough data. Need at least {config.env.window_size * 2} candles.")
        sys.exit(1)
    
    # Split data
    train_data, val_data, test_data = trainer.data_loader.split_data(data)
    
    if args.eval_only:
        # Evaluation only mode
        print("\n" + "-"*60)
        print("EVALUATION MODE")
        print("-"*60)
        
        model_path = args.model_path
        if model_path is None:
            # Try to find the latest model
            model_files = [f for f in os.listdir(config.model_dir) 
                          if f.endswith('.zip') or not '.' in f]
            if model_files:
                model_path = os.path.join(config.model_dir, sorted(model_files)[-1])
            else:
                print("Error: No model found. Please provide --model-path")
                sys.exit(1)
        
        print(f"Loading model from: {model_path}")
        
        # Fit feature engineer on training data
        trainer.feature_engineer.fit(train_data)
        
        # Create test environment
        test_normalized = trainer.feature_engineer.normalize(test_data)
        test_env = create_env(test_normalized, config, trainer.feature_engineer)
        
        # Load agent
        agent = PPOAgent(config, test_env, use_gru=args.use_gru)
        agent.load(model_path)
        
    else:
        # Training mode
        print("\n" + "-"*60)
        print("TRAINING")
        print("-"*60)
        
        if args.walk_forward:
            # Walk-forward validation
            results = trainer.train_walk_forward(
                data,
                timesteps_per_split=config.training.total_timesteps // 3
            )
            agent, metrics = results[-1]  # Use the last trained model
        else:
            # Single training run
            agent, metrics = trainer.train_single(
                train_data,
                val_data,
                total_timesteps=config.training.total_timesteps
            )
            
            print("\nTraining Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        # Save the final model
        agent.save()
        trainer.save_training_history()
    
    # Evaluation
    print("\n" + "-"*60)
    print("FINAL EVALUATION ON TEST DATA")
    print("-"*60)
    
    evaluator = Evaluator(config, trainer.feature_engineer)
    
    # Run backtest
    results = evaluator.compare_with_buy_and_hold(agent, test_data)
    
    # Print report
    evaluator.print_report(results)
    
    # Generate plots
    if not args.no_plots:
        try:
            # Performance plot
            plot_path = os.path.join(
                config.log_dir,
                f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            evaluator.plot_performance(results, save_path=plot_path, show=False)
            
            # Metrics summary
            metrics_path = os.path.join(
                config.log_dir,
                f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            evaluator.plot_metrics_summary(results, save_path=metrics_path, show=False)
            
            print(f"\nPlots saved to {config.log_dir}/")
            
        except Exception as e:
            print(f"\nWarning: Could not generate plots: {e}")
            print("This is normal in headless environments. Use --no-plots to suppress.")
    
    print("\n" + "="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    main()

