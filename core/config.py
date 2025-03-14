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
    max_risk_per_trade: float = 0.5  # Max % of capital to risk per trade (reduced from 1.0%)
    max_exposure_pct: float = 8.0    # Max % of capital in open positions (reduced from 10.0%)
    max_exposure_per_symbol_pct: float = 4.0  # Max % per symbol (reduced from 5.0%)
    stop_loss_pct: float = 1.0       # Default stop loss percentage (increased from 0.5%)
    take_profit_pct: float = 3.0     # Default take profit percentage (increased for better risk/reward)
    trailing_stop_pct: float = 0.5   # Trailing stop percentage (increased from 0.3%)
    risk_reward_ratio: float = 3.0   # Target risk/reward ratio
    
    # Strategy parameters
    timeframe: str = "1m"           # Default timeframe for analysis
    rsi_oversold: int = 35          # RSI oversold threshold (increased from 30)
    rsi_overbought: int = 65        # RSI overbought threshold (decreased from 70)
    sequence_length: int = 10       # Sequence length for LSTM
    model_type: str = "xgboost"     # Model type (lstm, xgboost)
    confidence_threshold: float = 0.65  # Min confidence for signals (increased from 0.6)
    min_profit_threshold: float = 0.3   # Min expected profit to trade (increased from 0.2)
    min_trade_quantity: float = 0.001   # Cantidad mínima para operar (ajustable según el par)
    
    # System parameters
    use_testnet: bool = True        # Use testnet instead of real trading
    simulation_mode: bool = False   # Run in simulation mode
    use_real_market_data: bool = True  # Use real market data in simulation mode
    enable_dashboard: bool = True   # Enable dashboard
    min_balance: float = 10.0       # Minimum balance to trade
    log_level: str = "INFO"         # Logging level
    
    # Simulation parameters
    sim_initial_balance: Dict[str, float] = field(default_factory=lambda: {"USDT": 10000.0, "BTC": 0.15, "ETH": 2.0, "BNB": 10.0})
    
    # Technical parameters
    baseline_volatility: float = 0.015  # Baseline volatility (increased from 0.01)
    default_volatility: float = 0.015   # Default volatility (increased from 0.01)
    take_profit_ratio: float = 3.0      # TP:SL ratio (increased from 2.0)
    base_stop_loss_pct: float = 1.0     # Base stop loss percentage (increased from 0.5)
    max_stop_loss_pct: float = 2.0      # Max stop loss percentage
    use_price_protection: bool = True   # Use price protection
    xgb_rounds: int = 100               # Number of boosting rounds
    feature_count: int = 25             # Number of features
    
    # Market regime parameters
    regime_lookback_period: int = 20        # Lookback period for regime detection
    volatility_threshold: float = 0.015     # Volatility threshold for regime classification
    trend_strength_threshold: float = 0.7   # Trend strength threshold for regime classification
    
    # Configuración específica para operaciones spot
    enforce_spot_balance: bool = True       # Asegurar que nunca se opere con saldo negativo
    safety_margin_pct: float = 1.0          # Margen de seguridad para comisiones (porcentaje)
    
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
        
        # Parse simulation parameters
        simulation_mode = os.environ.get("SIMULATION_MODE", "false").lower() in ["true", "yes", "1", "y"]
        use_real_market_data = os.environ.get("USE_REAL_MARKET_DATA", "true").lower() in ["true", "yes", "1", "y"]
        
        # Parse initial simulation balances
        sim_initial_balance = {}
        if "SIM_BALANCE_USDT" in os.environ:
            sim_initial_balance["USDT"] = float(os.environ.get("SIM_BALANCE_USDT"))
        if "SIM_BALANCE_BTC" in os.environ:
            sim_initial_balance["BTC"] = float(os.environ.get("SIM_BALANCE_BTC"))
        if "SIM_BALANCE_ETH" in os.environ:
            sim_initial_balance["ETH"] = float(os.environ.get("SIM_BALANCE_ETH"))
        if "SIM_BALANCE_BNB" in os.environ:
            sim_initial_balance["BNB"] = float(os.environ.get("SIM_BALANCE_BNB"))
            
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
            simulation_mode=simulation_mode,
            use_real_market_data=use_real_market_data,
            sim_initial_balance=sim_initial_balance or None,
        )


def load_config(config_path: str) -> Config:
    """Load configuration from a JSON file or create default"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        # Process simulation balances if present
        sim_initial_balance = config_dict.get('sim_initial_balance', None)
        
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
            use_real_market_data=config_dict.get('use_real_market_data', True),
            sim_initial_balance=sim_initial_balance,
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
        'use_real_market_data': config.use_real_market_data if hasattr(config, 'use_real_market_data') else True,
    }
    
    # Add simulation balances if present
    if hasattr(config, 'sim_initial_balance') and config.sim_initial_balance:
        config_dict['sim_initial_balance'] = config.sim_initial_balance
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)