import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.window_sizes = [7, 14, 25]  # Diferentes ventanas para cálculos
        
    def calculate_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for use with LLM strategy"""
        # Make sure we have a copy to avoid modifying the original
        df = ohlcv_data.copy()
        
        # Calculate basic indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        stddev = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * stddev)
        df['bb_lower'] = df['bb_middle'] - (2 * stddev)
        
        # MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # Volatility
        df['atr'] = (
            df['high'] - df['low']
        ).rolling(window=14).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw price data"""
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure DataFrame has required columns
        required_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in result.columns:
                raise ValueError(f"Required column {col} not found in DataFrame")
        
        # Price-based features
        for window in self.window_sizes:
            # Simple returns
            result[f'return_{window}'] = result['close'].pct_change(window)
            
            # Moving averages
            result[f'sma_{window}'] = result['close'].rolling(window=window).mean()
            
            # Price distance from MA (normalized)
            result[f'ma_dist_{window}'] = (result['close'] - result[f'sma_{window}']) / result[f'sma_{window}']
            
            # Volatility (standard deviation of returns)
            result[f'volatility_{window}'] = result['close'].pct_change().rolling(window=window).std()
        
        # Volume-based features
        result['volume_change'] = result['volume'].pct_change()
        result['volume_ma'] = result['volume'].rolling(window=20).mean()
        result['relative_volume'] = result['volume'] / result['volume_ma']
        
        # Candlestick patterns
        result['body_size'] = abs(result['close'] - result['open']) / result['open']
        result['upper_shadow'] = (result['high'] - result['close'].clip(lower=result['open'])) / result['open']
        result['lower_shadow'] = (result['close'].clip(upper=result['open']) - result['low']) / result['open']
        
        # Add order book imbalance if available
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            result['ob_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        
        # Calculate RSI
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        result['bb_middle'] = result['close'].rolling(window=20).mean()
        stddev = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + (2 * stddev)
        result['bb_lower'] = result['bb_middle'] - (2 * stddev)
        result['bb_pct'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # Momentum indicators
        result['momentum_1'] = result['close'].diff(1)
        result['momentum_3'] = result['close'].diff(3)
        
        # Drop NaN values resulting from the calculations
        result = result.dropna()
        
        return result
    
    def prepare_lstm_sequences(self, features: pd.DataFrame, target: pd.Series) -> tuple:
        """Prepare sequences for LSTM training/prediction"""
        X, y = [], []
        for i in range(len(features) - self.config.sequence_length):
            X.append(features.iloc[i:i + self.config.sequence_length].values)
            y.append(target.iloc[i + self.config.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_ml_features(self, df: pd.DataFrame, target_col: str = 'target') -> tuple:
        """Prepare features for machine learning models (XGBoost etc.)"""
        # Create target: future price change percentage
        df['future_return'] = df['close'].pct_change(1).shift(-1)
        
        # Exclude unnecessary columns
        feature_columns = [col for col in df.columns if col not in ['open_time', 'close_time', 'future_return']]
        
        # Split into features and target
        X = df[feature_columns]
        y = df['future_return']
        
        return X, y
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for neural networks"""
        # Copy DataFrame to avoid modifying the original
        result = df.copy()
        
        # Columns to normalize (skip time-related columns and target)
        skip_cols = ['open_time', 'close_time', 'future_return']
        norm_cols = [col for col in result.columns if col not in skip_cols]
        
        # Z-score normalization
        for col in norm_cols:
            mean = result[col].mean()
            std = result[col].std()
            if std > 0:
                result[col] = (result[col] - mean) / std
        
        return result
    
    def create_target(self, df: pd.DataFrame, periods_ahead: int = 1) -> pd.DataFrame:
        """Create target variable based on future price movement"""
        result = df.copy()
        
        # Future return
        result['future_return'] = result['close'].pct_change(periods_ahead).shift(-periods_ahead)
        
        # Future direction (1 for up, 0 for down)
        result['future_direction'] = (result['future_return'] > 0).astype(int)
        
        return result