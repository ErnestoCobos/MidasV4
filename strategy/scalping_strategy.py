import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import asyncio

from strategy.signal_generator import SignalGenerator, SignalType
from strategy.risk_manager import RiskManager
from data.feature_engineer import FeatureEngineer

class ScalpingStrategy:
    """
    Main scalping strategy implementation
    
    Coordinates signal generation, risk management, and trading decisions
    """
    
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.logger = logging.getLogger('ScalpingStrategy')
        
        # Initialize components
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)
        self.feature_engineer = FeatureEngineer(config)
        
        # State variables
        self.last_predictions = {}  # symbol -> prediction
    
    async def generate_signal(self, symbol: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal for a symbol
        
        Args:
            symbol: Trading symbol
            features: Dictionary of features and indicators
            
        Returns:
            Signal dictionary
        """
        # Log received features
        self.logger.debug(f"Generating signal for {symbol} with {len(features)} features")
        
        if self.model is None:
            self.logger.warning("No prediction model available, using indicator-based signals")
            return self._generate_indicator_signal(symbol, features)
        
        try:
            # Prepare features for prediction
            model_features = self._prepare_model_features(features)
            
            # Get prediction from model
            prediction = await self._get_model_prediction(symbol, model_features)
            
            # Generate signal
            current_price = features.get('close', features.get('price', 0))
            signal = self.signal_generator.generate_signal(prediction, current_price, features)
            
            # Log signal
            if signal['type'] != SignalType.NEUTRAL:
                self.logger.info(
                    f"Signal generated for {symbol}: {signal['direction']} "
                    f"with {signal['confidence']:.2f}% confidence"
                )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
            # Return neutral signal on error
            return {
                'type': SignalType.NEUTRAL,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'error': str(e)
            }
            
    def analyze(self, indicators: Dict[str, Any]):
        """
        Analyze indicators and generate a trading signal (for compatibility with bot._process_price_update)
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Tuple of (signal, direction, params)
        """
        try:
            # Get symbol from indicators or use a default
            symbol = indicators.get('symbol', 'UNKNOWN')
            
            # Use indicator-based signal generation
            signal = self._generate_indicator_signal(symbol, indicators)
            
            if signal['type'] == SignalType.NEUTRAL:
                # No signal
                return False, None, None
            
            # Extract direction
            direction = signal['direction']
            
            # Prepare parameters
            params = {
                'entry': signal['current_price'],
                'stop_loss': signal['current_price'] * 0.99 if direction == 'BUY' else signal['current_price'] * 1.01,
                'take_profit': signal['current_price'] * 1.01 if direction == 'BUY' else signal['current_price'] * 0.99,
                'strategy_type': 'indicator',
                'confidence': signal['confidence']
            }
            
            return True, direction, params
            
        except Exception as e:
            self.logger.error(f"Error in analyze method: {str(e)}")
            return False, None, None
    
    async def _get_model_prediction(self, symbol: str, features) -> float:
        """
        Get price movement prediction from model
        
        Args:
            symbol: Trading symbol
            features: Prepared features for model input
            
        Returns:
            Predicted price movement (percentage or absolute)
        """
        try:
            # Make prediction
            prediction = self.model.predict(features)
            
            # Extract and format result
            if isinstance(prediction, np.ndarray):
                prediction = prediction.item() if prediction.size == 1 else prediction[0]
            
            # Store prediction
            self.last_predictions[symbol] = prediction
            
            self.logger.debug(f"Model prediction for {symbol}: {prediction:.6f}")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {str(e)}")
            
            # Use last prediction if available, otherwise return 0
            return self.last_predictions.get(symbol, 0)
    
    def _prepare_model_features(self, features: Dict[str, Any]):
        """
        Prepare features for model input
        
        Args:
            features: Raw features dictionary
            
        Returns:
            Features in the format expected by the model
        """
        # Implementation depends on model type
        if self.config.model_type.lower() == 'lstm':
            # For LSTM, need to reshape to (1, sequence_length, feature_count)
            # TODO: Implement sequence preparation for LSTM
            raise NotImplementedError("LSTM feature preparation not implemented yet")
            
        elif self.config.model_type.lower() == 'xgboost':
            # For XGBoost, convert to numpy array
            feature_list = []
            
            # Extract relevant features in a consistent order
            for feature in [
                'sma_7', 'sma_25',
                'bb_upper', 'bb_lower', 'bb_middle',
                'rsi', 'rsi_14',
                'volatility_14',
                'volume_sma', 'current_volume',
                'relative_volume',
                'ma_dist_7', 'ma_dist_14', 'ma_dist_25',
                'return_7', 'return_14', 'return_25',
                'body_size', 'upper_shadow', 'lower_shadow'
            ]:
                # Use 0 as default if feature not available
                if feature in features:
                    feature_list.append(features[feature])
                elif feature == 'rsi' and 'rsi_14' in features:
                    feature_list.append(features['rsi_14'])
                elif feature == 'rsi_14' and 'rsi' in features:
                    feature_list.append(features['rsi'])
                else:
                    feature_list.append(0)
            
            # Return as numpy array
            return np.array([feature_list])
        
        else:
            # Default: just return the features as is
            return features
    
    def _generate_indicator_signal(self, symbol: str, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signal based only on indicators (fallback when no model available)
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators
            
        Returns:
            Signal dictionary
        """
        # Default: neutral
        signal_type = SignalType.NEUTRAL
        confidence = 0
        predicted_move_pct = 0
        
        current_price = indicators.get('close', indicators.get('price', 0))
        
        # Need RSI and Bollinger Bands for this strategy
        if not all(k in indicators for k in ['rsi_14', 'bb_upper', 'bb_lower', 'sma_7', 'sma_25']):
            if not 'rsi_14' in indicators and 'rsi' in indicators:
                indicators['rsi_14'] = indicators['rsi']
            else:
                return {
                    'type': SignalType.NEUTRAL,
                    'direction': 'NEUTRAL',
                    'confidence': 0,
                    'current_price': current_price,
                    'predicted_move_pct': 0,
                    'reason': 'Missing required indicators'
                }
        
        # Extract indicators
        rsi = indicators['rsi_14']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        sma_7 = indicators['sma_7']
        sma_25 = indicators['sma_25']
        
        # RSI oversold + price below lower BB + SMA7 > SMA25 = buy signal
        if (rsi < self.config.rsi_oversold and 
            current_price < bb_lower * 1.01 and 
            sma_7 > sma_25):
            
            signal_type = SignalType.BUY
            confidence = 70 + (self.config.rsi_oversold - rsi)  # Higher confidence if RSI lower
            predicted_move_pct = 0.5  # Expect 0.5% upward move
            reason = f"RSI oversold ({rsi:.2f}) + price below lower BB + bullish SMA crossover"
            self.logger.info(f"BUY signal generated for {symbol}: {reason}")
        
        # RSI overbought + price above upper BB + SMA7 < SMA25 = sell signal
        elif (rsi > self.config.rsi_overbought and 
              current_price > bb_upper * 0.99 and 
              sma_7 < sma_25):
              
            signal_type = SignalType.SELL
            confidence = 70 + (rsi - self.config.rsi_overbought)  # Higher confidence if RSI higher
            predicted_move_pct = -0.5  # Expect 0.5% downward move
            reason = f"RSI overbought ({rsi:.2f}) + price above upper BB + bearish SMA crossover"
            self.logger.info(f"SELL signal generated for {symbol}: {reason}")
        
        # Just RSI oversold + price below lower BB = weaker buy signal
        elif rsi < self.config.rsi_oversold and current_price < bb_lower * 1.01:
            signal_type = SignalType.BUY
            confidence = 60 + (self.config.rsi_oversold - rsi) / 2  # Medium confidence
            predicted_move_pct = 0.3  # Expect 0.3% upward move
            reason = f"RSI oversold ({rsi:.2f}) + price below lower BB"
            self.logger.info(f"BUY signal generated for {symbol}: {reason}")
        
        # Just RSI overbought + price above upper BB = weaker sell signal
        elif rsi > self.config.rsi_overbought and current_price > bb_upper * 0.99:
            signal_type = SignalType.SELL
            confidence = 60 + (rsi - self.config.rsi_overbought) / 2  # Medium confidence
            predicted_move_pct = -0.3  # Expect 0.3% downward move
            reason = f"RSI overbought ({rsi:.2f}) + price above upper BB"
            self.logger.info(f"SELL signal generated for {symbol}: {reason}")
        
        return {
            'type': signal_type,
            'direction': signal_type.name,
            'confidence': confidence,
            'current_price': current_price,
            'predicted_move_pct': predicted_move_pct,
            'indicators': indicators
        }
    
    def should_close_position(self, 
                            symbol: str, 
                            position: Dict[str, Any],
                            current_price: float, 
                            indicators: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a position should be closed
        
        Args:
            symbol: Trading symbol
            position: Position details
            current_price: Current market price
            indicators: Technical indicators
            
        Returns:
            Tuple of (should_close, reason)
        """
        return self.signal_generator.should_close_position(position, current_price, indicators)
    
    def calculate_position_size(self, 
                              total_capital: float,
                              symbol: str, 
                              entry_price: float, 
                              direction: str,
                              volatility: float) -> Tuple[float, float, float]:
        """
        Calculate position size and stop levels
        
        Args:
            total_capital: Total capital in account
            symbol: Trading symbol
            entry_price: Entry price
            direction: Trade direction ('BUY' or 'SELL')
            volatility: Current market volatility
            
        Returns:
            Tuple of (position_size, stop_loss_price, take_profit_price)
        """
        # Calculate dynamic stop loss based on volatility
        stop_loss_price = self.risk_manager.calculate_dynamic_stop_loss(
            symbol, entry_price, direction, volatility
        )
        
        # Calculate position size based on risk
        position_sizing = self.risk_manager.calculate_position_size(
            total_capital, symbol, entry_price, stop_loss_price
        )
        
        return (
            position_sizing.position_size,
            position_sizing.stop_loss_price,
            position_sizing.take_profit_price
        )
    
    def can_open_position(self, symbol: str, position_value: float, total_capital: float) -> bool:
        """Check if a new position can be opened based on risk constraints"""
        return self.risk_manager.can_open_position(symbol, position_value, total_capital)
    
    def register_position(self, symbol: str, position_details: Dict[str, Any]) -> None:
        """Register a new position"""
        self.risk_manager.register_position(symbol, position_details)
    
    def close_position(self, symbol: str) -> None:
        """Close a position"""
        self.risk_manager.close_position(symbol)
        
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all open positions"""
        return self.risk_manager.open_positions
    
    def update_trailing_stops(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Update trailing stops for all positions"""
        return self.risk_manager.update_trailing_stops(current_prices)