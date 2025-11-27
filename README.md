# Project Structure

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

# Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd price_prediction_rl_btc_15m
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

# Usage

```bash
python main.py
```
