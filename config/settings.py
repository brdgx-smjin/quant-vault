import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Exchange
EXCHANGE_ID = "binanceusdm"  # Binance USDT-M Futures
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# Trading
SYMBOL = "BTC/USDT:USDT"
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
DEFAULT_LEVERAGE = 5
MAX_POSITION_SIZE_PCT = 0.1   # 계좌의 10%
MAX_LOSS_PER_TRADE_PCT = 0.02  # 거래당 최대 손실 2%
DAILY_MAX_LOSS_PCT = 0.05     # 일일 최대 손실 5%

# Fibonacci Levels
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION = [1.0, 1.272, 1.618, 2.0, 2.618]

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# ML
MODEL_DIR = str(PROJECT_ROOT / "models")
RETRAIN_INTERVAL_HOURS = 24
LOOKBACK_PERIODS = 500

# Paths
DATA_DIR = str(PROJECT_ROOT / "data")
LOG_DIR = str(PROJECT_ROOT / "logs")
