# PPO Trading Agent for BTC 15-minute Candles

Deep Reinforcement Learning agent using Proximal Policy Optimization (PPO) for automated Bitcoin trading on 15-minute candles.

## Overview

This project implements a PPO-based trading agent with the following features:

- **LSTM/GRU Feature Extractor**: Captures temporal patterns in price data
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMAs, EMAs
- **Risk-Adjusted Rewards**: Differential Sharpe ratio with transaction fees
- **Walk-Forward Validation**: Realistic out-of-sample testing

## Project Structure

```
ppo_trading_btc/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py             # Configuration parameters
│   ├── data_loader.py        # Yahoo Finance data loading
│   ├── feature_engineering.py # Technical indicators & normalization
│   ├── environment.py        # Gymnasium trading environment
│   ├── network.py            # LSTM/GRU neural network architecture
│   ├── agent.py              # PPO agent wrapper
│   ├── training.py           # Training pipeline
│   └── evaluation.py         # Backtesting & visualization
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
├── .gitignore               # Git ignore patterns
└── README.md                # This file
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ppo_trading_btc
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python main.py
```

### Training with Custom Settings

```bash
# Train for 100,000 timesteps
python main.py --timesteps 100000

# Use GRU instead of LSTM
python main.py --use-gru

# Walk-forward validation
python main.py --walk-forward

# Set random seed
python main.py --seed 123
```

### Evaluate Pretrained Model

```bash
python main.py --eval-only --model-path models/ppo_trading_final
```

### Headless Mode (No Plots)

```bash
python main.py --no-plots
```

## Configuration

Key configuration parameters in `src/config.py`:

### Environment

- `window_size`: Lookback window for observations (default: 64)
- `transaction_fee`: Trading fee rate (default: 0.07%)
- `stop_loss_pct`: Stop loss percentage (default: 5%)
- `take_profit_pct`: Take profit percentage (default: 10%)

### Network

- `lstm_hidden_size`: LSTM hidden dimension (default: 128)
- `lstm_num_layers`: Number of LSTM layers (default: 2)
- `feature_dim`: Feature extractor output dimension (default: 64)

### PPO

- `learning_rate`: Learning rate (default: 3e-4)
- `n_steps`: Steps per rollout (default: 2048)
- `batch_size`: Mini-batch size (default: 64)
- `gamma`: Discount factor (default: 0.99)
- `clip_range`: PPO clipping range (default: 0.2)

## Features

### State Space (Observations)

- Log returns
- RSI (normalized)
- MACD histogram
- Bollinger Band width and position
- SMA/EMA ratios
- Volume indicators
- Time features (hour, day of week with sin/cos encoding)

### Action Space

- 0: **Hold** - Do nothing
- 1: **Buy** - Open/add to long position
- 2: **Sell** - Close position

### Reward Function

Risk-adjusted differential Sharpe ratio:

```
Reward = Sharpe_ratio - (Volatility × Risk_aversion) - Transaction_fees
```

## Output

After training, the following files are generated:

- `models/ppo_trading_final.zip` - Final trained model
- `models/ppo_trading_*.zip` - Checkpoint models
- `logs/performance_*.png` - Portfolio performance chart
- `logs/metrics_*.png` - Performance metrics summary
- `logs/training_history_*.csv` - Training metrics

## Performance Metrics

The evaluation includes:

- Total Return (%)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio
- Comparison with Buy & Hold strategy

## Important Notes

⚠️ **Financial Warning**: This is an educational project. Reinforcement learning models for trading are prone to overfitting and may not generalize to live market conditions. Never use this for real trading without extensive testing and risk management.

### Known Limitations

1. **Data Availability**: Yahoo Finance limits 15-minute data to the last 60 days
2. **Overfitting Risk**: RL models can memorize historical patterns
3. **Market Regime Changes**: Models may not adapt to changing market conditions
4. **Transaction Costs**: Real-world slippage and fees may differ

## References

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## License

MIT License
