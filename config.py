import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for the Binance scalping bot"""
    
    # API credentials
    api_key: str = ""
    api_secret: str = ""
    
    # Trading parameters
    symbols: List[str] = None  # List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
    base_order_size: Dict[str, float] = None  # Order size by trading pair
    max_open_trades: int = 3  # Maximum number of concurrent trades
    
    # Risk management
    max_capital_risk_percent: float = 2.0  # Max % of capital to risk per trade
    max_risk_per_trade: float = 1.0  # Max risk per individual trade (%)
    max_exposure_pct: float = 50.0   # Maximum exposure percentage
    max_exposure_per_symbol_pct: float = 20.0  # Maximum exposure percentage per symbol
    stop_loss_percent: float = 0.5  # Default stop loss percentage
    base_stop_loss_pct: float = 0.5  # Base stop loss percentage
    max_stop_loss_pct: float = 2.0  # Maximum stop loss percentage
    take_profit_percent: float = 1.0  # Default take profit percentage
    min_profit_threshold: float = 0.2  # Minimum profit % to enter a trade
    trailing_stop_pct: float = 0.3  # Trailing stop percentage
    
    # Strategy parameters
    timeframe: str = "1m"  # Default timeframe for analysis
    rsi_oversold: int = 30  # RSI oversold threshold
    rsi_overbought: int = 70  # RSI overbought threshold
    
    # Model parameters
    model_type: str = "llm"  # Model type: "xgboost", "lstm", "llm", or "indicator"
    confidence_threshold: float = 60.0  # Confidence threshold for signals
    sequence_length: int = 60  # Sequence length for LSTM (60 candles)
    feature_count: int = 20  # Number of features for model input
    xgb_rounds: int = 100  # Boosting rounds for XGBoost
    lstm_epochs: int = 50  # Number of epochs for LSTM training
    lstm_batch_size: int = 32  # Batch size for LSTM training
    
    # Hardware acceleration (disabled for Vultr inference)
    use_gpu: bool = False  # Using Vultr inference API instead of local GPU
    gpu_device: int = 0  # Not used with Vultr
    gpu_id: int = 0  # Not used with Vultr
    
    # Automatic training
    auto_train: bool = True  # Whether to automatically train models
    training_interval_hours: int = 24  # How often to train models in hours
    min_data_points: int = 1000  # Minimum data points required for training
    
    # Vultr AI Optimization
    vultr_api_key: str = "vultrtestkey123456789"  # API key for Vultr Inference (test key)
    ai_optimization_enabled: bool = True  # Whether to use AI for strategy optimization
    ai_optimization_interval_hours: int = 12  # How often to optimize strategy with AI
    ai_optimization_model: str = "llama-3.1-70b-instruct-fp8"  # Model to use for AI optimization
    ai_min_trades_for_optimization: int = 10  # Minimum trades required before optimization
    
    def __post_init__(self):
        """Initialize default values for collection types"""
        if self.symbols is None:
            self.symbols = ["BTCUSDT"]
        
        if self.base_order_size is None:
            self.base_order_size = {"BTCUSDT": 0.001, "ETHUSDT": 0.01}
    
    def validate(self) -> bool:
        """Validate that required configuration is present"""
        if not self.api_key or not self.api_secret:
            return False
        
        if not self.symbols or len(self.symbols) == 0:
            return False
            
        return True
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        # Try to load from .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("python-dotenv not installed. Using environment variables directly.")
        
        # GPU configuration from environment (disabled for Vultr inference)
        use_gpu = False  # Disable GPU since we're using Vultr inference API
        gpu_device = 0
        gpu_id = 0
        
        # Model configuration from environment
        model_type = os.environ.get("MODEL_TYPE", "llm")
        confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "60.0"))
        sequence_length = int(os.environ.get("SEQUENCE_LENGTH", "60"))
        feature_count = int(os.environ.get("FEATURE_COUNT", "20"))
        xgb_rounds = int(os.environ.get("XGB_ROUNDS", "100"))
        lstm_epochs = int(os.environ.get("LSTM_EPOCHS", "50"))
        lstm_batch_size = int(os.environ.get("LSTM_BATCH_SIZE", "32"))
        
        # Auto training configuration
        auto_train = os.environ.get("AUTO_TRAIN", "1").lower() in ["1", "true", "yes", "y"]
        training_interval_hours = int(os.environ.get("TRAINING_INTERVAL_HOURS", "24"))
        min_data_points = int(os.environ.get("MIN_DATA_POINTS", "1000"))
        
        # Vultr AI configuration
        vultr_api_key = os.environ.get("VULTR_API_KEY", "vultrtestkey123456789")
        ai_optimization_enabled = os.environ.get("AI_OPTIMIZATION", "1").lower() in ["1", "true", "yes", "y"]
        ai_optimization_interval_hours = int(os.environ.get("AI_OPTIMIZATION_INTERVAL_HOURS", "12"))
        ai_optimization_model = os.environ.get("AI_OPTIMIZATION_MODEL", "llama-3.1-70b-instruct-fp8")
        ai_min_trades_for_optimization = int(os.environ.get("AI_MIN_TRADES", "10"))
        
        return cls(
            api_key=os.environ.get("BINANCE_API_KEY", ""),
            api_secret=os.environ.get("BINANCE_API_SECRET", ""),
            symbols=os.environ.get("TRADING_SYMBOLS", "BTCUSDT").split(","),
            max_open_trades=int(os.environ.get("MAX_OPEN_TRADES", "3")),
            max_capital_risk_percent=float(os.environ.get("MAX_CAPITAL_RISK", "2.0")),
            max_risk_per_trade=float(os.environ.get("MAX_RISK_PER_TRADE", "1.0")),
            max_exposure_pct=float(os.environ.get("MAX_EXPOSURE_PCT", "50.0")),
            max_exposure_per_symbol_pct=float(os.environ.get("MAX_EXPOSURE_PER_SYMBOL_PCT", "20.0")),
            stop_loss_percent=float(os.environ.get("STOP_LOSS_PERCENT", "0.5")),
            base_stop_loss_pct=float(os.environ.get("BASE_STOP_LOSS_PCT", "0.5")),
            max_stop_loss_pct=float(os.environ.get("MAX_STOP_LOSS_PCT", "2.0")),
            take_profit_percent=float(os.environ.get("TAKE_PROFIT_PERCENT", "1.0")),
            min_profit_threshold=float(os.environ.get("MIN_PROFIT_THRESHOLD", "0.2")),
            trailing_stop_pct=float(os.environ.get("TRAILING_STOP_PCT", "0.3")),
            model_type=model_type,
            confidence_threshold=confidence_threshold,
            sequence_length=sequence_length,
            feature_count=feature_count,
            xgb_rounds=xgb_rounds,
            lstm_epochs=lstm_epochs,
            lstm_batch_size=lstm_batch_size,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_id=gpu_id,
            auto_train=auto_train,
            training_interval_hours=training_interval_hours,
            min_data_points=min_data_points,
            vultr_api_key=vultr_api_key,
            ai_optimization_enabled=ai_optimization_enabled,
            ai_optimization_interval_hours=ai_optimization_interval_hours,
            ai_optimization_model=ai_optimization_model,
            ai_min_trades_for_optimization=ai_min_trades_for_optimization,
        )