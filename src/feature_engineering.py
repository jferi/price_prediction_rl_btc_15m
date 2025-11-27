"""
Feature engineering for trading environment.

Computes technical indicators, normalizes features, and prepares
the observation space for the RL agent.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from .config import Config


class FeatureEngineer:
    """
    Feature engineering pipeline for trading data.
    
    Computes log returns, technical indicators, and time-based features
    to create a rich observation space for the RL agent.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration object with feature settings.
        """
        self.config = config
        self.feature_config = config.features
        self.window_size = config.env.window_size
        
        # Store normalization parameters
        self.feature_means = None
        self.feature_stds = None
        self.is_fitted = False
    
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns.
            
        Returns:
            DataFrame with computed features.
        """
        df = data.copy()
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price-based features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Technical indicators
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_sma(df)
        df = self._add_ema(df)
        
        # Volume indicators
        if self.feature_config.use_volume_indicators:
            df = self._add_volume_indicators(df)
        
        # Time features
        if self.feature_config.use_time_features:
            df = self._add_time_features(df)
        
        # Drop NaN values created by rolling calculations
        df = df.dropna()
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Relative Strength Index."""
        period = self.feature_config.rsi_period
        delta = df['close'].diff()
        
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Normalize RSI to [-1, 1]
        df['rsi_normalized'] = (df['rsi'] - 50) / 50
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD histogram."""
        fast = self.feature_config.macd_fast
        slow = self.feature_config.macd_slow
        signal = self.feature_config.macd_signal
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        df['macd_histogram'] = macd_line - signal_line
        
        # Normalize MACD histogram by price
        df['macd_normalized'] = df['macd_histogram'] / df['close']
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands features."""
        period = self.feature_config.bb_period
        std_dev = self.feature_config.bb_std
        
        sma = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        # Band width (volatility measure)
        df['bb_width'] = (upper_band - lower_band) / sma
        
        # Position within bands (normalized)
        df['bb_position'] = (df['close'] - lower_band) / (upper_band - lower_band + 1e-10)
        df['bb_position'] = df['bb_position'].clip(-1, 2)  # Clip extreme values
        
        return df
    
    def _add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        for period in self.feature_config.sma_periods:
            sma = df['close'].rolling(window=period).mean()
            # Normalized: how far is price from SMA
            df[f'sma_{period}_ratio'] = (df['close'] - sma) / sma
        
        return df
    
    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        for period in self.feature_config.ema_periods:
            ema = df['close'].ewm(span=period, adjust=False).mean()
            # Normalized: how far is price from EMA
            df[f'ema_{period}_ratio'] = (df['close'] - ema) / ema
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume ratio to moving average
        vol_ma = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (vol_ma + 1e-10)
        
        # Log volume change
        df['volume_log_change'] = np.log(df['volume'] / df['volume'].shift(1) + 1e-10)
        
        # Volume-weighted price change
        df['vwap_ratio'] = (
            (df['close'] * df['volume']).rolling(window=20).sum() /
            (df['volume'].rolling(window=20).sum() + 1e-10)
        ) / df['close']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features using sin/cos encoding."""
        # Hour of day (0-23)
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (0-6)
        day_of_week = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names used for observations."""
        feature_cols = [
            'log_return',
            'high_low_ratio',
            'close_open_ratio',
            'rsi_normalized',
            'macd_normalized',
            'bb_width',
            'bb_position',
        ]
        
        # SMA features
        for period in self.feature_config.sma_periods:
            feature_cols.append(f'sma_{period}_ratio')
        
        # EMA features
        for period in self.feature_config.ema_periods:
            feature_cols.append(f'ema_{period}_ratio')
        
        # Volume features
        if self.feature_config.use_volume_indicators:
            feature_cols.extend(['volume_ratio', 'volume_log_change', 'vwap_ratio'])
        
        # Time features
        if self.feature_config.use_time_features:
            feature_cols.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])
        
        return feature_cols
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit normalization parameters on training data.
        
        Args:
            data: Training data with computed features.
        """
        feature_cols = self.get_feature_columns()
        available_cols = [col for col in feature_cols if col in data.columns]
        
        self.feature_means = data[available_cols].mean()
        self.feature_stds = data[available_cols].std()
        
        # Avoid division by zero
        self.feature_stds = self.feature_stds.replace(0, 1)
        
        self.is_fitted = True
    
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using fitted parameters.
        
        Args:
            data: DataFrame with computed features.
            
        Returns:
            DataFrame with normalized features.
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before normalizing")
        
        df = data.copy()
        feature_cols = self.get_feature_columns()
        available_cols = [col for col in feature_cols if col in df.columns]
        
        for col in available_cols:
            df[col] = (df[col] - self.feature_means[col]) / self.feature_stds[col]
            # Clip extreme values
            df[col] = df[col].clip(-5, 5)
        
        return df
    
    def create_observation_array(
        self,
        data: pd.DataFrame,
        index: int
    ) -> np.ndarray:
        """
        Create observation array for a given index.
        
        Returns a window of normalized features.
        
        Args:
            data: DataFrame with normalized features.
            index: Current index in the data.
            
        Returns:
            Numpy array of shape (window_size, n_features).
        """
        feature_cols = self.get_feature_columns()
        available_cols = [col for col in feature_cols if col in data.columns]
        
        start_idx = max(0, index - self.window_size + 1)
        end_idx = index + 1
        
        obs = data[available_cols].iloc[start_idx:end_idx].values
        
        # Pad if necessary (at the beginning of the series)
        if obs.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - obs.shape[0], len(available_cols)))
            obs = np.vstack([padding, obs])
        
        return obs.astype(np.float32)
    
    def get_observation_shape(self) -> Tuple[int, int]:
        """Get the shape of observations."""
        return (self.window_size, len(self.get_feature_columns()))

