"""
Data loader for cryptocurrency data from Yahoo Finance.

Handles downloading, caching, and preprocessing of BTC 15-minute candle data.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from .config import Config


class DataLoader:
    """
    Data loader for cryptocurrency OHLCV data.
    
    Fetches data from Yahoo Finance and provides caching functionality
    to avoid repeated API calls.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object containing data settings.
        """
        self.config = config
        self.ticker = config.env.ticker
        self.interval = config.env.interval
        self.data_dir = config.data_dir
    
    def download(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download BTC data from Yahoo Finance.
        
        Note: yfinance has limitations for intraday data:
        - 15m data is only available for the last 60 days
        - Data is automatically adjusted for splits
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (optional).
            end_date: End date in 'YYYY-MM-DD' format (optional).
            use_cache: Whether to use cached data if available.
            
        Returns:
            DataFrame with OHLCV data.
        """
        # Default to last 59 days (yfinance limit for 15m is 60 days)
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
        
        cache_file = self.get_cache_path(start_date, end_date)
        
        # Try to load from cache
        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return data
        
        print(f"Downloading {self.ticker} data from {start_date} to {end_date}...")
        
        # Download data
        ticker = yf.Ticker(self.ticker)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=self.interval
        )
        
        # Clean up the data
        data = self.clean_data(data)
        
        # Cache the data
        if use_cache:
            os.makedirs(self.data_dir, exist_ok=True)
            data.to_csv(cache_file)
            print(f"Data cached to {cache_file}")
        
        print(f"Downloaded {len(data)} candles")
        return data
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.
        
        Args:
            data: Raw DataFrame from yfinance.
            
        Returns:
            Cleaned DataFrame with standard columns.
        """
        # Select and rename columns
        columns_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Keep only OHLCV columns
        data = data[list(columns_map.keys())].copy()
        data = data.rename(columns=columns_map)
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Remove zero volume rows (likely no trading)
        data = data[data['volume'] > 0]
        
        # Ensure proper datetime index
        data.index = pd.to_datetime(data.index)
        data.index = data.index.tz_localize(None)  # Remove timezone info
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def get_cache_path(self, start_date: str, end_date: str) -> str:
        """Generate cache file path."""
        filename = f"{self.ticker}_{self.interval}_{start_date}_{end_date}.csv"
        return os.path.join(self.data_dir, filename)
    
    def split_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Uses time-based split to avoid look-ahead bias.
        
        Args:
            data: Full dataset.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            
        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
        
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def get_walk_forward_splits(
        self,
        data: pd.DataFrame,
        train_days: int = 30,
        test_days: int = 7
    ) -> list:
        """
        Generate walk-forward validation splits.
        
        Args:
            data: Full dataset.
            train_days: Number of days for each training window.
            test_days: Number of days for each test window.
            
        Returns:
            List of (train_data, test_data) tuples.
        """
        splits = []
        
        # Calculate candles per day (15-min intervals = 96 candles/day)
        candles_per_day = 24 * 4  # 96 for 15-min
        train_size = train_days * candles_per_day
        test_size = test_days * candles_per_day
        step_size = test_size
        
        start_idx = 0
        while start_idx + train_size + test_size <= len(data):
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            train_data = data.iloc[start_idx:train_end].copy()
            test_data = data.iloc[train_end:test_end].copy()
            
            splits.append((train_data, test_data))
            start_idx += step_size
        
        print(f"Created {len(splits)} walk-forward splits")
        return splits

