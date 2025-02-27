"""
Constants for the MidasScalpingv4 bot
"""

# API endpoints
BINANCE_API_URL = 'https://api.binance.com'
BINANCE_API_TESTNET_URL = 'https://testnet.binance.vision'
BINANCE_WS_URL = 'wss://stream.binance.com:9443/ws'
BINANCE_WS_TESTNET_URL = 'wss://testnet.binance.vision/ws'

# Timeframes
TIMEFRAMES = [
    '1m', '3m', '5m', '15m', '30m',
    '1h', '2h', '4h', '6h', '8h', '12h',
    '1d', '3d', '1w', '1M'
]

# Model types
MODEL_TYPES = ['lstm', 'xgboost', 'ensemble']

# Order types
ORDER_TYPES = {
    'MARKET': 'MARKET',
    'LIMIT': 'LIMIT',
    'STOP_LOSS': 'STOP_LOSS_LIMIT',
    'TAKE_PROFIT': 'TAKE_PROFIT_LIMIT'
}

# Indicators
RSI_PERIOD = 14
SMA_PERIODS = [7, 25]
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# Risk management
MIN_ORDER_SIZE = {
    'BTC': 0.0001,  # Minimum order size for BTC
    'ETH': 0.001    # Minimum order size for ETH
}

# Trade status
TRADE_STATUS = {
    'OPEN': 'OPEN',
    'CLOSED': 'CLOSED',
    'CANCELLED': 'CANCELLED',
    'PARTIAL': 'PARTIAL'
}

# Performance metrics
METRICS = {
    'TOTAL_TRADES': 'total_trades',
    'WIN_RATE': 'win_rate',
    'PROFIT_FACTOR': 'profit_factor',
    'SHARPE_RATIO': 'sharpe_ratio',
    'MAX_DRAWDOWN': 'max_drawdown',
    'TOTAL_PROFIT': 'total_profit'
}

# Log levels
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}