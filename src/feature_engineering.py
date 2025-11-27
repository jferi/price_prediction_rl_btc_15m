"""
Feature Engineering Module.

Responsible for transforming raw OHLCV data into stationary, normalized features
suitable for Reinforcement Learning.
"""

import pandas as pd
import numpy as np
import ta
from typing import List, Tuple
from .config import Config

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.feature_cols = []
        self.is_fitted = False
        self.mean = None
        self.std = None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline: Calculate technical indicators -> Handle NaN -> Normalize.
        """
        df = df.copy()
        
        # 1. Basic Returns (Stationary)
        # Log return is better for ML than percentage return
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Volatility
        df['volatility'] = df['log_return'].rolling(window=20).std()

        # 3. Technical Indicators (Normalized)
        if self.config.features.use_technical_indicators:
            # RSI
            rsi_ind = ta.momentum.RSIIndicator(df['close'], window=self.config.features.rsi_period)
            df['rsi'] = rsi_ind.rsi() / 100.0  # Normalize to [0, 1]

            # MACD
            macd_ind = ta.trend.MACD(
                df['close'], 
                window_slow=self.config.features.macd_slow, 
                window_fast=self.config.features.macd_fast, 
                window_sign=self.config.features.macd_signal
            )
            # Normalize MACD by price to make it stationary
            df['macd_diff'] = macd_ind.macd_diff() / df['close']

            # Bollinger Bands
            bb_ind = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            # Width ratio is stationary
            df['bb_width'] = (bb_ind.bollinger_hband() - bb_ind.bollinger_lband()) / df['close']
            # Position within band [0, 1]
            df['bb_position'] = (df['close'] - bb_ind.bollinger_lband()) / (bb_ind.bollinger_hband() - bb_ind.bollinger_lband())

        # 4. Time Features (Cyclical encoding)
        if self.config.features.use_time_features:
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # 5. Clean up
        df.dropna(inplace=True)
        
        # Select feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        return df

    def fit(self, df: pd.DataFrame):
        """Calculate mean and std for normalization based on training data."""
        if not self.feature_cols:
            raise ValueError("Run preprocess() first to generate features.")
            
        self.mean = df[self.feature_cols].mean()
        self.std = df[self.feature_cols].std()
        # Replace 0 std with 1 to avoid division by zero
        self.std = self.std.replace(0, 1.0)
        self.is_fitted = True

    def scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before scaling.")
            
        df_scaled = df.copy()
        
        # Scale features
        for col in self.feature_cols:
            # Z-score
            df_scaled[col] = (df[col] - self.mean[col]) / self.std[col]
            
            # Clip outliers (Critical for neural networks stability)
            df_scaled[col] = df_scaled[col].clip(
                -self.config.features.clip_range, 
                self.config.features.clip_range
            )
            
        return df_scaled

    def get_observation_shape(self) -> Tuple[int, int]:
        return (self.config.env.window_size, len(self.feature_cols))
