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
    
    def __init__(self, config, model=None, binance_client=None):
        self.config = config
        self.model = model
        self.logger = logging.getLogger('ScalpingStrategy')
        
        # Initialize components
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)
        self.feature_engineer = FeatureEngineer(config)
        self.binance_client = binance_client
        
        # Initialize regime detector
        try:
            from strategy.market_regime import MarketRegimeDetector, MarketRegime
            self.regime_detector = MarketRegimeDetector(config)
            self.MarketRegime = MarketRegime
            self.current_regime = None
            self.logger.info("Market regime detector initialized successfully")
        except ImportError:
            self.logger.warning("Market regime detector not available, falling back to standard strategy")
            self.regime_detector = None
            self.current_regime = None
        
        # Original parameters backup for regime adaptation
        self.base_parameters = None
        
        # State variables
        self.last_predictions = {}  # symbol -> prediction
    
    def adapt_to_market_regime(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Adapt strategy parameters based on current market regime"""
        # Skip if no regime detector available
        if not hasattr(self, 'regime_detector') or self.regime_detector is None:
            self.logger.warning("No regime detector available, skipping adaptation")
            return {'regime': 'UNKNOWN', 'confidence': 0}
            
        # Detect current market regime
        regime_info = self.regime_detector.detect_regime(ohlcv_data)
        self.current_regime = regime_info['regime']
        
        # Store original parameters if not already saved
        if not self.base_parameters:
            self.base_parameters = {
                'confidence_threshold': getattr(self.config, 'confidence_threshold', 0.65),
                'rsi_oversold': getattr(self.config, 'rsi_oversold', 35),
                'rsi_overbought': getattr(self.config, 'rsi_overbought', 65),
                'base_stop_loss_pct': getattr(self.config, 'base_stop_loss_pct', 1.0),
                'trailing_stop_pct': getattr(self.config, 'trailing_stop_pct', 0.5),
                'max_risk_per_trade': getattr(self.config, 'max_risk_per_trade', 0.5)
            }
        
        # Adapt parameters based on detected regime
        if regime_info['regime'] == self.MarketRegime.TRENDING_UP:
            # In uptrend - more aggressive entries, wider stops
            self.config.rsi_oversold = max(20, self.base_parameters['rsi_oversold'] - 5)
            self.config.rsi_overbought = max(75, self.base_parameters['rsi_overbought'] + 5)
            self.config.trailing_stop_pct = self.base_parameters['trailing_stop_pct'] * 1.5
            self.config.max_risk_per_trade = min(2.0, self.base_parameters['max_risk_per_trade'] * 1.2)
            self.config.confidence_threshold = max(0.6, self.base_parameters['confidence_threshold'] * 0.9)
            
            self.logger.info(f"Adapted to TRENDING_UP regime: RSI({self.config.rsi_oversold}/{self.config.rsi_overbought}), "
                           f"trailing_stop({self.config.trailing_stop_pct:.2f}%), risk({self.config.max_risk_per_trade:.2f}%)")
            
        elif regime_info['regime'] == self.MarketRegime.TRENDING_DOWN:
            # In downtrend - focus on shorts, tighter stops on longs
            self.config.rsi_overbought = min(65, self.base_parameters['rsi_overbought'] - 5)
            self.config.base_stop_loss_pct = self.base_parameters['base_stop_loss_pct'] * 0.8
            self.config.max_risk_per_trade = max(0.5, self.base_parameters['max_risk_per_trade'] * 0.8)
            
            self.logger.info(f"Adapted to TRENDING_DOWN regime: RSI({self.config.rsi_oversold}/{self.config.rsi_overbought}), "
                           f"stop_loss({self.config.base_stop_loss_pct:.2f}%), risk({self.config.max_risk_per_trade:.2f}%)")
            
        elif regime_info['regime'] == self.MarketRegime.RANGING:
            # In range - focus on mean reversion
            self.config.rsi_oversold = max(25, self.base_parameters['rsi_oversold'] + 5)
            self.config.rsi_overbought = min(75, self.base_parameters['rsi_overbought'] - 5)
            self.config.base_stop_loss_pct = self.base_parameters['base_stop_loss_pct'] * 0.8
            self.config.trailing_stop_pct = self.base_parameters['trailing_stop_pct'] * 0.7
            
            self.logger.info(f"Adapted to RANGING regime: RSI({self.config.rsi_oversold}/{self.config.rsi_overbought}), "
                           f"stop_loss({self.config.base_stop_loss_pct:.2f}%), trailing({self.config.trailing_stop_pct:.2f}%)")
            
        elif regime_info['regime'] == self.MarketRegime.VOLATILE:
            # In volatile markets - reduce size, widen stops
            self.config.base_stop_loss_pct = self.base_parameters['base_stop_loss_pct'] * 1.5
            self.config.max_risk_per_trade = max(0.5, self.base_parameters['max_risk_per_trade'] * 0.6)
            self.config.confidence_threshold = min(0.8, self.base_parameters['confidence_threshold'] * 1.2)
            
            self.logger.info(f"Adapted to VOLATILE regime: confidence({self.config.confidence_threshold:.2f}), "
                           f"stop_loss({self.config.base_stop_loss_pct:.2f}%), risk({self.config.max_risk_per_trade:.2f}%)")
        else:
            # Reset to base parameters for unknown regime
            for key, value in self.base_parameters.items():
                setattr(self.config, key, value)
                
            self.logger.info("Reset to base parameters for UNKNOWN regime")
        
        return {
            'regime': regime_info['regime'].name,
            'confidence': regime_info['confidence'],
            'volatility': regime_info.get('volatility', 0.0),
            'trend_strength': regime_info.get('trend_strength', 0.0)
        }
    
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
        # Apply market regime detection if available
        if hasattr(self, 'regime_detector') and self.regime_detector:
            try:
                # Create OHLCV dataframe from indicators if needed
                try:
                    # Check if binance_client is available
                    if hasattr(self, 'binance_client') and self.binance_client and hasattr(self.binance_client, 'get_ohlcv'):
                        ohlcv_data = self.binance_client.get_ohlcv(symbol, self.config.timeframe, 30)
                        # Detect market regime and adapt strategy parameters
                        regime_info = self.adapt_to_market_regime(ohlcv_data)
                        self.logger.info(f"Market regime for {symbol}: {regime_info['regime']} "
                                       f"(confidence: {regime_info['confidence']:.1f}%, "
                                       f"volatility: {regime_info.get('volatility', 0):.4f})")
                    else:
                        # Si no hay binance_client, usar los indicadores directamente
                        ohlcv_data = pd.DataFrame([indicators])
                        regime_info = self.adapt_to_market_regime(ohlcv_data)
                        self.logger.info(f"Using indicators directly for market regime detection")
                except Exception as e:
                    self.logger.warning(f"Failed to get OHLCV data: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error detecting market regime: {str(e)}")
        
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
                              symbol: str,
                              account_balance: float = None,
                              entry_price: float = None, 
                              stop_loss: float = None,
                              direction: str = "BUY",
                              volatility: float = 0.01) -> float:
        """
        Calculate position size for a trade
        
        Args:
            symbol: Trading symbol
            account_balance: Total available balance for trading
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: Trade direction ('BUY' or 'SELL')
            volatility: Current market volatility (optional)
            
        Returns:
            Position size quantity
        """
        try:
            # Default values
            if account_balance is None:
                account_balance = 1000.0  # Default balance
                
            if entry_price is None:
                # Try to get current price
                entry_price = 0
                
            # Protección contra balance cero - guardar un porcentaje mínimo
            min_reserved_balance_pct = getattr(self.config, 'min_reserved_balance_pct', 10)
            reserved_balance = account_balance * (min_reserved_balance_pct / 100)
            usable_balance = account_balance - reserved_balance
            
            # No usar un balance negativo o muy bajo
            if usable_balance < 10:
                self.logger.warning(f"Balance usable muy bajo: {usable_balance:.2f}. Usando valor mínimo seguro.")
                usable_balance = 10
                
            # If stop loss is not provided, calculate it
            if stop_loss is None:
                # Usar un stop loss más conservador para spot (0.7% por defecto)
                base_stop_loss_pct = getattr(self.config, 'base_stop_loss_pct', 0.7) / 100
                if direction == "BUY":
                    stop_loss = entry_price * (1 - base_stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 + base_stop_loss_pct)
            
            # Reducir riesgo por operación para mercado spot
            conservative_risk = getattr(self.config, 'max_risk_per_trade', 0.5) / 100
            
            # Aplicar factor de volatilidad
            volatility_factor = min(1.0, 0.01 / max(0.005, volatility))
            adjusted_risk = conservative_risk * volatility_factor
            
            # Calcular el monto de riesgo considerando el balance usable
            risk_amount = usable_balance * adjusted_risk
            
            # Añadir margen de seguridad para comisiones y slippage (3% por defecto)
            safety_margin = getattr(self.config, 'safety_margin_pct', 3) / 100
            risk_amount = risk_amount * (1 - safety_margin)
            
            # Calculate price difference for stop loss
            if direction == "BUY":
                price_diff = abs(entry_price - stop_loss)
            else:
                price_diff = abs(stop_loss - entry_price)
                
            # Avoid division by zero
            if price_diff == 0:
                price_diff = entry_price * 0.01  # Default 1% movement
                
            # Calculate quantity
            quantity = risk_amount / price_diff
            
            # Round quantity to standard precision (varies by symbol)
            # For BTC, typically 5 decimal places
            if "BTC" in symbol:
                quantity = round(quantity, 5)
            # For ETH, typically 4 decimal places
            elif "ETH" in symbol:
                quantity = round(quantity, 4)
            else:
                # Default precision
                quantity = round(quantity, 2)
                
            # Default minimum amount
            if quantity < 0.001:
                quantity = 0.001
                
            self.logger.info(
                f"Calculated position size for {symbol}: {quantity} units at {entry_price} "
                f"(Risk: {adjusted_risk*100:.2f}%, Reserved: {reserved_balance:.2f})"
            )
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.001  # Minimum quantity as fallback
    
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