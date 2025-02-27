import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class Config:
    """Configuration for the Binance scalping bot"""
    
    # API credentials
    api_key: str = ""
    api_secret: str = ""
    
    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    base_order_size: Dict[str, float] = field(default_factory=lambda: {"BTCUSDT": 0.001, "ETHUSDT": 0.01})
    max_open_trades: int = 3
    
    # Risk management
    max_risk_per_trade: float = 1.0  # Max % of capital to risk per trade
    max_exposure_pct: float = 10.0   # Max % of capital in open positions
    max_exposure_per_symbol_pct: float = 5.0  # Max % per symbol
    stop_loss_pct: float = 0.5       # Default stop loss percentage
    take_profit_pct: float = 1.0     # Default take profit percentage
    trailing_stop_pct: float = 0.3   # Trailing stop percentage
    
    # Strategy parameters
    timeframe: str = "1m"           # Default timeframe for analysis
    rsi_oversold: int = 30          # RSI oversold threshold
    rsi_overbought: int = 70        # RSI overbought threshold
    sequence_length: int = 10       # Sequence length for LSTM
    model_type: str = "xgboost"     # Model type (lstm, xgboost)
    confidence_threshold: float = 0.6  # Min confidence for signals
    min_profit_threshold: float = 0.2  # Min expected profit to trade
    
    # System parameters
    use_testnet: bool = True        # Use testnet instead of real trading
    simulation_mode: bool = False   # Run in simulation mode
    enable_dashboard: bool = True   # Enable dashboard
    min_balance: float = 10.0       # Minimum balance to trade
    log_level: str = "INFO"         # Logging level
    
    # Technical parameters
    baseline_volatility: float = 0.01  # Baseline volatility
    default_volatility: float = 0.01   # Default volatility
    take_profit_ratio: float = 2.0     # TP:SL ratio
    base_stop_loss_pct: float = 0.5    # Base stop loss percentage
    max_stop_loss_pct: float = 2.0     # Max stop loss percentage
    use_price_protection: bool = True  # Use price protection
    xgb_rounds: int = 100              # Number of boosting rounds
    feature_count: int = 25            # Number of features
    
    def validate(self) -> bool:
        """Validate that required configuration is present"""
        if not self.simulation_mode:
            if not self.api_key or not self.api_secret:
                return False
        
        if not self.symbols or len(self.symbols) == 0:
            return False
            
        return True
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        return cls(
            api_key=os.environ.get("BINANCE_API_KEY", ""),
            api_secret=os.environ.get("BINANCE_API_SECRET", ""),
            symbols=os.environ.get("TRADING_SYMBOLS", "BTCUSDT").split(","),
            max_open_trades=int(os.environ.get("MAX_OPEN_TRADES", "3")),
            max_risk_per_trade=float(os.environ.get("MAX_RISK_PER_TRADE", "1.0")),
            max_exposure_pct=float(os.environ.get("MAX_EXPOSURE_PCT", "10.0")),
            stop_loss_pct=float(os.environ.get("STOP_LOSS_PERCENT", "0.5")),
            take_profit_pct=float(os.environ.get("TAKE_PROFIT_PERCENT", "1.0")),
            use_testnet=os.environ.get("USE_TESTNET", "true").lower() == "true",
        )


def load_config(config_path: str) -> Config:
    """Load configuration from a JSON file or create default"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        # Create Config object with values from file
        config = Config(
            api_key=config_dict.get('api_key', ''),
            api_secret=config_dict.get('api_secret', ''),
            symbols=config_dict.get('symbols', ['BTCUSDT']),
            max_open_trades=config_dict.get('max_open_trades', 3),
            max_risk_per_trade=config_dict.get('max_risk_per_trade', 1.0),
            max_exposure_pct=config_dict.get('max_exposure_pct', 10.0),
            stop_loss_pct=config_dict.get('stop_loss_pct', 0.5),
            take_profit_pct=config_dict.get('take_profit_pct', 1.0),
            timeframe=config_dict.get('timeframe', '1m'),
            model_type=config_dict.get('model_type', 'xgboost'),
            use_testnet=config_dict.get('use_testnet', True),
            simulation_mode=config_dict.get('simulation_mode', False),
        )
        
        return config
    else:
        # Return default config
        return Config()


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to a JSON file"""
    config_dict = {
        'api_key': config.api_key,
        'api_secret': config.api_secret,
        'symbols': config.symbols,
        'max_open_trades': config.max_open_trades,
        'max_risk_per_trade': config.max_risk_per_trade,
        'max_exposure_pct': config.max_exposure_pct,
        'stop_loss_pct': config.stop_loss_pct,
        'take_profit_pct': config.take_profit_pct,
        'timeframe': config.timeframe,
        'model_type': config.model_type,
        'use_testnet': config.use_testnet,
        'simulation_mode': config.simulation_mode,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)