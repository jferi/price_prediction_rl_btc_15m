"""
Evaluation Module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .environment import create_env

class Evaluator:
    def __init__(self, config, feature_engineer):
        self.config = config
        self.feature_engineer = feature_engineer

    def backtest(self, agent, df_raw):
        """
        Run a backtest on the given RAW DataFrame using the trained agent.
        """
        print("Starting backtest...")
        
        # Preprocess raw data
        df_processed = self.feature_engineer.preprocess(df_raw)
        
        # Align
        df_raw_aligned = df_raw.loc[df_processed.index]
        
        # Scale data
        df_scaled = self.feature_engineer.scale(df_processed)
        
        # Create environment with BOTH scaled and raw data
        env = create_env(df_scaled, df_raw_aligned, self.config, self.feature_engineer)
        obs, _ = env.reset()
        
        # Storage
        portfolio_values = []
        positions = []
        
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(info['portfolio_value'])
            positions.append(info['position'])
            
        # Convert to DataFrame for analysis
        results = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'position': positions
        })
        
        # Calculate metrics
        initial_value = self.config.env.initial_balance
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        print(f"Backtest finished.")
        print(f"Initial Value: {initial_value}")
        print(f"Final Value: {final_value:.2f}")
        print(f"Total Return: {total_return * 100:.2f}%")
        
        self.plot_results(results)
        return results

    def plot_results(self, results):
        plt.figure(figsize=(12, 6))
        plt.plot(results['portfolio_value'], label='Portfolio Value')
        plt.title('Backtest Results')
        plt.xlabel('Steps')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.log_dir}/backtest_results.png")
        print(f"Plot saved to {self.config.log_dir}/backtest_results.png")
