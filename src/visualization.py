"""
Visualization utilities using Rich library for terminal dashboard.
"""

from stable_baselines3.common.callbacks import BaseCallback
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import numpy as np
import collections

class DashboardCallback(BaseCallback):
    """
    A rich-based dashboard callback for SB3 training.
    """
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.recent_rewards = collections.deque(maxlen=100)
        self.portfolio_history = collections.deque(maxlen=50)
        self.price_history = collections.deque(maxlen=50)
        self.actions_history = collections.deque(maxlen=20)
        self.layout = self.make_layout()
        self.live = Live(self.layout, refresh_per_second=4)
        
    def make_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="stats"),
            Layout(name="chart")
        )
        return layout

    def on_training_start(self):
        self.live.start()

    def on_step(self) -> bool:
        # Extract info
        infos = self.locals.get("infos", [{}])[0]
        
        # Update metrics
        if "portfolio_value" in infos:
            self.portfolio_history.append(infos["portfolio_value"])
        
        # We don't have direct access to price easily unless we put it in info
        # Let's assume info has it, or we skip it.
        # But we can get reward
        if "rewards" in self.locals:
            self.recent_rewards.append(self.locals["rewards"][0])
            
        # Update Layout
        self.update_header()
        self.update_stats(infos)
        self.update_chart()
        
        return True

    def on_training_end(self):
        self.live.stop()

    def update_header(self):
        progress = self.num_timesteps / self.total_timesteps
        bar = "█" * int(progress * 50) + "-" * (50 - int(progress * 50))
        self.layout["header"].update(
            Panel(f"Training Progress: [{bar}] {progress:.1%} | Step: {self.num_timesteps}/{self.total_timesteps}", 
                  title="PPO Trading Agent", style="bold blue")
        )

    def update_stats(self, info):
        table = Table(show_header=False, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        current_portfolio = self.portfolio_history[-1] if self.portfolio_history else 0
        avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
        
        table.add_row("Portfolio Value", f"${current_portfolio:,.2f}")
        table.add_row("Avg Reward (last 100)", f"{avg_reward:.4f}")
        table.add_row("Position", "LONG" if info.get("position", 0) == 1 else "CASH")
        
        self.layout["stats"].update(Panel(table, title="Live Statistics"))

    def update_chart(self):
        # Simple ASCII chart for portfolio
        if len(self.portfolio_history) < 2:
            return
            
        values = list(self.portfolio_history)
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1
        
        height = 10
        chart = ""
        
        # Normalize to height
        normalized = [int((v - min_val) / range_val * (height - 1)) for v in values]
        
        cols = []
        for h in normalized:
            col = [" "] * height
            col[height - 1 - h] = "●" # Dot for the value
            cols.append(col)
            
        # Transpose to print lines
        rows = []
        for r in range(height):
            row_str = ""
            for c in range(len(cols)):
                row_str += cols[c][r]
            rows.append(row_str)
            
        chart_text = "\n".join(rows)
        self.layout["chart"].update(Panel(chart_text, title="Portfolio History (Last 50 Steps)"))

