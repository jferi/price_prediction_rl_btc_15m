"""
Evaluation and visualization for trading agent.

Provides comprehensive evaluation metrics, backtesting,
and visualization of trading performance.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from .config import Config
from .feature_engineering import FeatureEngineer
from .environment import TradingEnv, create_env
from .agent import PPOAgent


class Evaluator:
    """
    Evaluation and visualization for trained trading agents.
    
    Provides backtesting, performance metrics, and visualizations
    for analyzing agent performance.
    """
    
    def __init__(
        self,
        config: Config,
        feature_engineer: FeatureEngineer
    ):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object.
            feature_engineer: Fitted FeatureEngineer instance.
        """
        self.config = config
        self.feature_engineer = feature_engineer
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def backtest(
        self,
        agent: PPOAgent,
        data: pd.DataFrame,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Run backtest on given data.
        
        Args:
            agent: Trained PPO agent.
            data: Test data (will be normalized).
            render: Whether to print step-by-step info.
            
        Returns:
            Dictionary containing backtest results.
        """
        # Normalize data
        normalized_data = self.feature_engineer.normalize(data)
        
        # Create environment
        env = create_env(normalized_data, self.config, self.feature_engineer)
        
        # Run episode
        obs, info = env.reset()
        done = False
        
        # Track history
        actions = []
        portfolio_values = [self.config.env.initial_balance]
        positions = [0]
        prices = [data['close'].iloc[self.config.env.window_size]]
        timestamps = [data.index[self.config.env.window_size]]
        rewards = []
        
        step = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            actions.append(action)
            portfolio_values.append(info["portfolio_value"])
            positions.append(info["position"])
            rewards.append(reward)
            
            current_idx = self.config.env.window_size + step + 1
            if current_idx < len(data):
                prices.append(data['close'].iloc[current_idx])
                timestamps.append(data.index[current_idx])
            
            if render:
                action_names = ["Hold", "Buy", "Sell"]
                print(f"Step {step}: {action_names[action]}, "
                      f"Portfolio=${info['portfolio_value']:.2f}, "
                      f"Position={info['position']:.6f}")
            
            step += 1
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        results = {
            "final_portfolio": portfolio_values[-1],
            "total_return_pct": (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
            "total_trades": info["total_trades"],
            "sharpe_ratio": self._calculate_sharpe(returns),
            "sortino_ratio": self._calculate_sortino(returns),
            "max_drawdown_pct": self._calculate_max_drawdown(portfolio_values),
            "win_rate_pct": np.sum(np.array(returns) > 0) / len(returns) * 100 if len(returns) > 0 else 0,
            "avg_trade_return": np.mean(returns) * 100 if len(returns) > 0 else 0,
            "profit_factor": self._calculate_profit_factor(returns),
            "calmar_ratio": self._calculate_calmar_ratio(returns, portfolio_values),
            
            # History for plotting
            "timestamps": timestamps,
            "portfolio_values": portfolio_values,
            "prices": prices,
            "positions": positions,
            "actions": actions,
            "rewards": rewards
        }
        
        return results
    
    def compare_with_buy_and_hold(
        self,
        agent: PPOAgent,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare agent performance with buy-and-hold strategy.
        
        Args:
            agent: Trained agent.
            data: Test data.
            
        Returns:
            Comparison results.
        """
        # Run backtest
        agent_results = self.backtest(agent, data)
        
        # Calculate buy-and-hold
        start_price = data['close'].iloc[self.config.env.window_size]
        end_price = data['close'].iloc[-1]
        
        initial_btc = self.config.env.initial_balance / start_price
        buy_hold_final = initial_btc * end_price
        buy_hold_return = (buy_hold_final / self.config.env.initial_balance - 1) * 100
        
        # Calculate buy-hold portfolio values for plotting
        prices = data['close'].iloc[self.config.env.window_size:].values
        buy_hold_values = (prices / start_price) * self.config.env.initial_balance
        
        comparison = {
            "agent_return": agent_results["total_return_pct"],
            "buy_hold_return": buy_hold_return,
            "excess_return": agent_results["total_return_pct"] - buy_hold_return,
            "agent_sharpe": agent_results["sharpe_ratio"],
            "agent_max_drawdown": agent_results["max_drawdown_pct"],
            "buy_hold_values": buy_hold_values.tolist(),
            **agent_results
        }
        
        return comparison
    
    def plot_performance(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot comprehensive performance visualization.
        
        Args:
            results: Backtest results dictionary.
            save_path: Path to save the plot.
            show: Whether to display the plot.
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        timestamps = results.get("timestamps", range(len(results["portfolio_values"])))
        
        # 1. Portfolio Value
        ax1 = axes[0]
        ax1.plot(timestamps[:len(results["portfolio_values"])], 
                 results["portfolio_values"], 
                 'b-', linewidth=1.5, label='Agent Portfolio')
        
        if "buy_hold_values" in results:
            buy_hold_ts = timestamps[:len(results["buy_hold_values"])]
            ax1.plot(buy_hold_ts, results["buy_hold_values"], 
                     'gray', linewidth=1, alpha=0.7, linestyle='--',
                     label='Buy & Hold')
        
        ax1.axhline(y=self.config.env.initial_balance, color='r', 
                    linestyle=':', alpha=0.5, label='Initial Balance')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(f'Portfolio Performance | Return: {results["total_return_pct"]:.2f}% | '
                      f'Sharpe: {results["sharpe_ratio"]:.2f}')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. BTC Price with Trade Markers
        ax2 = axes[1]
        prices = results.get("prices", [])
        ax2.plot(timestamps[:len(prices)], prices, 'k-', linewidth=1, alpha=0.7)
        
        # Mark buy/sell actions
        actions = results.get("actions", [])
        for i, action in enumerate(actions):
            if i < len(prices) - 1:
                if action == 1:  # Buy
                    ax2.scatter(timestamps[i+1], prices[i+1], 
                               color='green', marker='^', s=50, zorder=5)
                elif action == 2:  # Sell
                    ax2.scatter(timestamps[i+1], prices[i+1], 
                               color='red', marker='v', s=50, zorder=5)
        
        ax2.set_ylabel('BTC Price ($)')
        ax2.set_title(f'BTC Price with Trades | Total Trades: {results["total_trades"]}')
        ax2.grid(True, alpha=0.3)
        
        # 3. Position Over Time
        ax3 = axes[2]
        positions = results.get("positions", [])
        ax3.fill_between(timestamps[:len(positions)], positions, 
                        alpha=0.5, color='blue', step='post')
        ax3.set_ylabel('BTC Position')
        ax3.set_title('Position Size Over Time')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Rewards
        ax4 = axes[3]
        rewards = results.get("rewards", [])
        cumulative_rewards = np.cumsum(rewards) if rewards else [0]
        ax4.plot(timestamps[:len(cumulative_rewards)], cumulative_rewards, 
                 'purple', linewidth=1.5)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Cumulative Reward')
        ax4.set_xlabel('Time')
        ax4.set_title('Cumulative Reward Over Time')
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes:
            if len(timestamps) > 0 and isinstance(timestamps[0], (pd.Timestamp, datetime)):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_metrics_summary(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot summary of key metrics.
        
        Args:
            results: Backtest results.
            save_path: Path to save the plot.
            show: Whether to display.
        """
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        
        metrics = [
            ("Total Return (%)", results["total_return_pct"]),
            ("Sharpe Ratio", results["sharpe_ratio"]),
            ("Max Drawdown (%)", results["max_drawdown_pct"]),
            ("Win Rate (%)", results["win_rate_pct"]),
            ("Profit Factor", results.get("profit_factor", 0)),
            ("Total Trades", results["total_trades"])
        ]
        
        colors = ['#2ecc71' if i < 2 else '#3498db' for i in range(6)]
        
        for idx, (ax, (name, value)) in enumerate(zip(axes.flat, metrics)):
            # Determine color based on value
            if name == "Total Return (%)" or name == "Sharpe Ratio":
                color = '#2ecc71' if value > 0 else '#e74c3c'
            elif name == "Max Drawdown (%)":
                color = '#e74c3c' if value > 20 else '#f39c12' if value > 10 else '#2ecc71'
            else:
                color = '#3498db'
            
            ax.bar([0], [value], color=color, width=0.5)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_ylabel(name.split('(')[0].strip())
            ax.set_xticks([])
            ax.text(0, value, f'{value:.2f}', ha='center', va='bottom', fontsize=11)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        plt.suptitle('Trading Performance Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close()
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """Print formatted performance report."""
        print("\n" + "="*60)
        print("TRADING PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\n{'Portfolio Performance':^60}")
        print("-"*60)
        print(f"  Initial Balance:     ${self.config.env.initial_balance:,.2f}")
        print(f"  Final Portfolio:     ${results['final_portfolio']:,.2f}")
        print(f"  Total Return:        {results['total_return_pct']:+.2f}%")
        
        if "buy_hold_return" in results:
            print(f"  Buy & Hold Return:   {results['buy_hold_return']:+.2f}%")
            print(f"  Excess Return:       {results['excess_return']:+.2f}%")
        
        print(f"\n{'Risk Metrics':^60}")
        print("-"*60)
        print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:       {results['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:        {results.get('calmar_ratio', 0):.2f}")
        print(f"  Max Drawdown:        {results['max_drawdown_pct']:.2f}%")
        
        print(f"\n{'Trading Statistics':^60}")
        print("-"*60)
        print(f"  Total Trades:        {results['total_trades']}")
        print(f"  Win Rate:            {results['win_rate_pct']:.2f}%")
        print(f"  Profit Factor:       {results.get('profit_factor', 0):.2f}")
        print(f"  Avg Trade Return:    {results['avg_trade_return']:.4f}%")
        
        print("\n" + "="*60)
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        annualization = np.sqrt(96 * 365)  # 15-min periods
        return float((mean_return / std_return) * annualization)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0 if mean_return <= 0 else float('inf')
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        annualization = np.sqrt(96 * 365)
        return float((mean_return / downside_std) * annualization)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return float(np.max(drawdown) * 100)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_profit = np.sum(profits) if len(profits) > 0 else 0
        total_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        
        return float(total_profit / total_loss)
    
    def _calculate_calmar_ratio(
        self, 
        returns: np.ndarray, 
        portfolio_values: List[float]
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(returns) < 2:
            return 0.0
        
        # Annualized return
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        n_periods = len(returns)
        periods_per_year = 96 * 365
        annualized_return = ((1 + total_return) ** (periods_per_year / n_periods)) - 1
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown(portfolio_values) / 100
        
        if max_dd == 0:
            return 0.0
        
        return float(annualized_return / max_dd)

