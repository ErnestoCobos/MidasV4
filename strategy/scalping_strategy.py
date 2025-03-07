import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import asyncio

from strategy.signal_generator import SignalGenerator, SignalType
from strategy.risk_manager import RiskManager
from data.feature_engineer import FeatureEngineer
from models.ml_module import MLModule

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
        
        # Initialize ML module (handles TensorFlow/XGBoost integration)
        self.ml_module = None
        if hasattr(config, 'use_ml') and config.use_ml:
            try:
                self.ml_module = MLModule(config)
                self.logger.info("ML module initialized successfully with slippage/commission modeling")
            except Exception as e:
                self.logger.error(f"Error initializing ML module: {str(e)}")
                self.logger.warning("Falling back to traditional model or indicators")
        
        # Configure confidence threshold (higher to reduce overtrading)
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.7)
        self.logger.info(f"Using confidence threshold: {self.confidence_threshold}")
        
        # Trading limits to control risk
        self.max_daily_trades = getattr(config, 'max_daily_trades', 30)
        self.max_daily_loss_pct = getattr(config, 'max_daily_loss_pct', 3.0)
        self.logger.info(f"Trading limits: max {self.max_daily_trades} trades/day, " 
                       f"max {self.max_daily_loss_pct}% daily loss")
        
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
        self.trades_today = {}      # symbol -> count
        self.daily_loss = 0.0       # Accumulated loss for the day
        self.last_trade_time = {}   # symbol -> timestamp
    
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
        
        # First check if trading constraints allow opening a position
        total_capital = 1000  # Default value, will be updated by bot.py
        if hasattr(self, 'binance_client') and self.binance_client:
            quote_asset = symbol[3:]  # e.g., 'USDT' from 'BTCUSDT'
            account_balance = self.binance_client.get_account_balance()
            total_capital = account_balance.get(quote_asset, 1000)
        
        # Check daily limits and cooling period (before running model to save time)
        if self._check_daily_trade_limit(symbol):
            self.logger.info(f"Daily trade limit reached for {symbol}, skipping signal generation")
            return {
                'type': SignalType.NEUTRAL,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'reason': 'Daily trade limit reached'
            }
            
        if self._check_daily_loss_limit(total_capital):
            self.logger.info(f"Daily loss limit reached ({self.daily_loss:.2f}), skipping signal generation")
            return {
                'type': SignalType.NEUTRAL,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'reason': 'Daily loss limit reached'
            }
            
        if not self._check_cooling_period(symbol):
            return {
                'type': SignalType.NEUTRAL,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'reason': 'In cooling period'
            }
        
        # Use ML module if available
        if self.ml_module is not None:
            try:
                # ML module handles feature preparation internally
                result = self.ml_module.predict(features)
                
                if 'error' not in result:
                    # Create signal based on ML prediction
                    current_price = features.get('close', features.get('price', 0))
                    
                    # Apply confidence threshold to filter low-quality signals
                    if result['confidence'] < self.confidence_threshold:
                        self.logger.info(
                            f"Signal rejected for {symbol}: {result['direction']} "
                            f"with {result['confidence']:.2f}% confidence (below threshold {self.confidence_threshold:.2f}%)"
                        )
                        return {
                            'type': SignalType.NEUTRAL,
                            'direction': 'NEUTRAL',
                            'confidence': result['confidence'],
                            'current_price': current_price,
                            'reason': 'Below confidence threshold'
                        }
                    
                    # Create signal with ML prediction
                    signal = {
                        'type': SignalType[result['direction']] if result['direction'] in ['BUY', 'SELL'] else SignalType.NEUTRAL,
                        'direction': result['direction'],
                        'confidence': result['confidence'],
                        'current_price': current_price,
                        'predicted_move_pct': result['prediction'],
                        'models_used': result.get('models_used', []),
                        'indicators': features
                    }
                    
                    # Log signal
                    if signal['type'] != SignalType.NEUTRAL:
                        self.logger.info(
                            f"ML signal generated for {symbol}: {signal['direction']} "
                            f"with {signal['confidence']:.2f}% confidence"
                        )
                    
                    return signal
            except Exception as e:
                self.logger.error(f"Error in ML signal generation for {symbol}: {str(e)}")
                # Continue to fallback methods
        
        # Fall back to traditional model if available
        if self.model is not None:
            try:
                # Prepare features for prediction
                model_features = self._prepare_model_features(features)
                
                # Get prediction from model
                prediction = await self._get_model_prediction(symbol, model_features)
                
                # Generate signal
                current_price = features.get('close', features.get('price', 0))
                signal = self.signal_generator.generate_signal(prediction, current_price, features)
                
                # Apply confidence threshold
                if signal['confidence'] < self.confidence_threshold:
                    self.logger.info(
                        f"Signal rejected for {symbol}: {signal['direction']} "
                        f"with {signal['confidence']:.2f}% confidence (below threshold {self.confidence_threshold:.2f}%)"
                    )
                    return {
                        'type': SignalType.NEUTRAL,
                        'direction': 'NEUTRAL',
                        'confidence': signal['confidence'],
                        'current_price': current_price,
                        'reason': 'Below confidence threshold'
                    }
                
                # Log signal
                if signal['type'] != SignalType.NEUTRAL:
                    self.logger.info(
                        f"Model signal generated for {symbol}: {signal['direction']} "
                        f"with {signal['confidence']:.2f}% confidence"
                    )
                
                return signal
                
            except Exception as e:
                self.logger.error(f"Error in model signal generation for {symbol}: {str(e)}")
                # Continue to indicator-based fallback
        
        # Fall back to indicators if all else fails
        self.logger.info("Using indicator-based signals as fallback")
        indicator_signal = self._generate_indicator_signal(symbol, features)
        
        # Apply confidence threshold to indicator signals too
        if indicator_signal['confidence'] < self.confidence_threshold:
            self.logger.info(
                f"Indicator signal rejected for {symbol}: {indicator_signal['direction']} "
                f"with {indicator_signal['confidence']:.2f}% confidence (below threshold {self.confidence_threshold:.2f}%)"
            )
            indicator_signal['type'] = SignalType.NEUTRAL
            indicator_signal['direction'] = 'NEUTRAL'
            indicator_signal['reason'] = 'Below confidence threshold'
        
        return indicator_signal
            
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
            # Use ML module if available
            if self.ml_module is not None:
                # Get prediction from ML module
                result = self.ml_module.predict(features)
                
                # Check if prediction was successful
                if 'error' not in result:
                    # Extract prediction value
                    prediction = result['prediction']
                    
                    # Apply realistic slippage and commission modeling
                    # First determine likely direction based on prediction sign
                    side = 'BUY' if prediction > 0 else 'SELL'
                    adjusted_prediction = self.ml_module.apply_slippage_and_commission(prediction, side)
                    
                    # Store prediction
                    self.last_predictions[symbol] = adjusted_prediction
                    
                    self.logger.debug(f"ML prediction for {symbol}: {prediction:.6f} " 
                                    f"(adjusted: {adjusted_prediction:.6f}, "
                                    f"direction: {result['direction']}, "
                                    f"confidence: {result['confidence']:.2f}%)")
                    
                    # Store confidence for later use
                    features['ml_confidence'] = result['confidence']
                    features['ml_direction'] = result['direction']
                    
                    return adjusted_prediction
                else:
                    self.logger.warning(f"Error in ML prediction: {result.get('error', 'Unknown error')}")
            
            # Fall back to traditional model if ML module failed or unavailable
            if self.model is not None:
                # Make prediction with traditional model
                prediction = self.model.predict(features)
                
                # Extract and format result
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.item() if prediction.size == 1 else prediction[0]
                
                # Store prediction
                self.last_predictions[symbol] = prediction
                
                self.logger.debug(f"Model prediction for {symbol}: {prediction:.6f}")
                return prediction
            
            # No model available
            self.logger.warning(f"No ML module or model available for prediction")
            return self.last_predictions.get(symbol, 0)
            
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
        # If using ML module, let it handle feature preparation
        if self.ml_module is not None:
            # Use ML module to prepare features based on model type
            if hasattr(self.config, 'model_type'):
                return features  # ML module will handle conversion internally in predict()
            
        # Fall back to legacy implementation
        # Implementation depends on model type
        if hasattr(self.config, 'model_type') and self.config.model_type.lower() == 'lstm':
            # For LSTM, need to reshape to (1, sequence_length, feature_count)
            if 'historical_data' in features:
                # If historical data is provided, use it directly
                historical_data = features['historical_data']
                
                # Ensure correct shape [batch, sequence_length, features]
                if len(historical_data.shape) == 2:
                    # Add batch dimension if missing
                    historical_data = np.expand_dims(historical_data, axis=0)
                    
                return historical_data
            else:
                self.logger.warning("LSTM requires historical data, falling back to XGBoost format")
                # Fall back to XGBoost format as placeholder
                return self._prepare_xgboost_features(features)
            
        elif hasattr(self.config, 'model_type') and self.config.model_type.lower() == 'xgboost':
            return self._prepare_xgboost_features(features)
        else:
            # Default: just return the features as is
            return features
            
    def _prepare_xgboost_features(self, features: Dict[str, Any]):
        """Helper method to extract XGBoost features in a consistent order"""
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
        """
        Check if a new position can be opened based on risk constraints
        
        This includes:
        1. Risk manager checks (max positions, exposure limits)
        2. Daily trade limit check
        3. Daily loss limit check
        4. Sufficient time since last trade (to prevent overtrading)
        5. Symbol-specific risk adjustments
        
        Args:
            symbol: Trading symbol
            position_value: Value of the position in quote currency
            total_capital: Total capital in account
            
        Returns:
            Boolean indicating if position can be opened, with reason in logs
        """
        # First check risk manager constraints
        if not self.risk_manager.can_open_position(symbol, position_value, total_capital):
            return False
            
        # Check daily trade limit
        if self._check_daily_trade_limit(symbol):
            self.logger.info(f"Daily trade limit reached for {symbol}: {self.trades_today.get(symbol, 0)}/{self.max_daily_trades}")
            return False
            
        # Check daily loss limit
        if self._check_daily_loss_limit(total_capital):
            self.logger.info(f"Daily loss limit reached: {self.daily_loss:.2f} ({(self.daily_loss/total_capital*100):.2f}%)")
            return False
            
        # Check cooling period (time since last trade)
        if not self._check_cooling_period(symbol):
            return False
        
        # Apply symbol-specific risk adjustments
        # ETHUSDT has been showing increased losses, so we'll be more conservative
        if symbol == 'ETHUSDT':
            # Check for higher confidence threshold for ETH
            eth_confidence_threshold = self.confidence_threshold * 1.2  # 20% higher threshold
            
            # If the ML module had a recent prediction below the elevated threshold, reject
            if hasattr(self, 'ml_module') and self.ml_module is not None:
                recent_pred = getattr(self, 'last_predictions', {}).get(symbol, {})
                if isinstance(recent_pred, dict) and 'confidence' in recent_pred:
                    if recent_pred['confidence'] < eth_confidence_threshold:
                        self.logger.info(f"ETHUSDT requires higher confidence: {recent_pred['confidence']:.2f} < {eth_confidence_threshold:.2f}")
                        return False
            
            # Increase cooling period for ETHUSDT by 50%
            eth_trades_today = self.trades_today.get(symbol, 0)
            
            # Reduce maximum trades for ETHUSDT as it generates more losses
            eth_max_trades = int(self.max_daily_trades * 0.7)  # 30% fewer trades for ETH
            
            if eth_trades_today >= eth_max_trades:
                self.logger.info(f"ETHUSDT has stricter trade limits: {eth_trades_today}/{eth_max_trades}")
                return False
                
            # If ETH has caused significant losses, be even more cautious
            if self.daily_loss < -total_capital * 0.01:  # If already lost > 1%
                eth_trades = [t for t in self.trades_today.keys() if 'ETH' in t]
                eth_trades_count = len(eth_trades)
                
                if eth_trades_count >= 3:  # Only allow 3 ETH trades when in losing territory
                    self.logger.info(f"Restricting ETHUSDT trades due to daily losses")
                    return False
            
            self.logger.info(f"Applied stricter risk controls for ETHUSDT")
            
        # All checks passed
        return True
        
    def _check_daily_trade_limit(self, symbol: str) -> bool:
        """
        Check if daily trade limit has been reached for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if limit reached, False otherwise
        """
        # Get current trade count for symbol
        trades_for_symbol = self.trades_today.get(symbol, 0)
        
        # Check against limit
        return trades_for_symbol >= self.max_daily_trades
        
    def _check_daily_loss_limit(self, total_capital: float) -> bool:
        """
        Check if daily loss limit has been reached
        
        Args:
            total_capital: Total capital in account
            
        Returns:
            True if limit reached, False otherwise
        """
        # Calculate daily loss percentage
        loss_percentage = abs(self.daily_loss) / total_capital * 100 if self.daily_loss < 0 else 0
        
        # Check against limit
        return loss_percentage >= self.max_daily_loss_pct
        
    def _check_cooling_period(self, symbol: str) -> bool:
        """
        Check if sufficient time has passed since last trade for a symbol
        to prevent overtrading
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if can trade, False if in cooling period
        """
        import time
        from datetime import datetime
        
        # Get time of last trade
        last_trade_time = self.last_trade_time.get(symbol, 0)
        
        # Get current time
        current_time = time.time()
        
        # Calculate time since last trade (in seconds)
        time_since_last_trade = current_time - last_trade_time
        
        # Define cooling period based on confidence or fixed value (in seconds)
        # For low confidence trades, use longer cooling period
        cooling_period = getattr(self.config, 'cooling_period_seconds', 60)
        
        # Check if enough time has passed
        if time_since_last_trade < cooling_period:
            remaining = cooling_period - time_since_last_trade
            self.logger.info(f"In cooling period for {symbol}, {remaining:.1f}s remaining")
            return False
            
        return True
        
    def update_trade_stats(self, symbol: str, profit_loss: float = 0):
        """
        Update trading statistics after a trade is executed or closed
        
        Args:
            symbol: Trading symbol
            profit_loss: Profit/loss from the trade (if closing)
        """
        import time
        
        # Update trade count for today
        self.trades_today[symbol] = self.trades_today.get(symbol, 0) + 1
        
        # Update last trade time
        self.last_trade_time[symbol] = time.time()
        
        # Update daily loss if trade was closed with a loss
        if profit_loss < 0:
            self.daily_loss += profit_loss
            self.logger.info(f"Updated daily loss: {self.daily_loss:.2f} after {symbol} trade ({profit_loss:.2f})")
    
    def register_position(self, symbol: str, position_details: Dict[str, Any]) -> None:
        """
        Register a new position and update trading statistics
        
        Args:
            symbol: Trading symbol
            position_details: Dictionary with position details
        """
        # Register with risk manager first
        self.risk_manager.register_position(symbol, position_details)
        
        # Update trade statistics 
        self.update_trade_stats(symbol)
        
        # Log for debugging
        self.logger.info(f"Position registered for {symbol} - Daily stats: "
                       f"{self.trades_today.get(symbol, 0)}/{self.max_daily_trades} trades, "
                       f"{self.daily_loss:.2f} daily loss")
    
    def close_position(self, symbol: str, profit_loss: float = 0) -> None:
        """
        Close a position and update trading statistics
        
        Args:
            symbol: Trading symbol
            profit_loss: Profit/loss from the trade
        """
        # Close position with risk manager
        self.risk_manager.close_position(symbol)
        
        # Update statistics with P/L
        if profit_loss != 0:
            self.update_trade_stats(symbol, profit_loss)
        
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all open positions"""
        return self.risk_manager.open_positions
    
    def update_trailing_stops(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Update trailing stops for all positions"""
        return self.risk_manager.update_trailing_stops(current_prices)