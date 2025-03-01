import logging
import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, Tuple

class SignalType(Enum):
    """Types of trading signals"""
    BUY = 1      # Long position signal
    SELL = -1    # Short position signal
    NEUTRAL = 0  # No action signal

class SignalGenerator:
    """
    Generates trading signals based on model predictions and market conditions
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('SignalGenerator')
        
        # Signal parameters
        self.confidence_threshold = config.confidence_threshold
        self.min_profit_threshold = config.min_profit_threshold
    
    def generate_signal(self, 
                       prediction: float, 
                       current_price: float, 
                       indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal based on model prediction and market conditions
        
        Args:
            prediction: Model prediction (expected price movement)
            current_price: Current market price
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary with signal details
        """
        # Convert prediction to percentage if it's an absolute value
        if abs(prediction) > 1:
            # Assuming prediction is absolute price movement
            predicted_move_pct = (prediction / current_price) * 100
        else:
            # Assuming prediction is already a percentage
            predicted_move_pct = prediction * 100
        
        # Initialize as neutral
        signal_type = SignalType.NEUTRAL
        confidence = abs(predicted_move_pct)
        
        # Adjust confidence based on current market conditions
        # Lower confidence in high volatility periods
        if 'volatility_14' in indicators:
            volatility = indicators['volatility_14']
            baseline_volatility = getattr(self.config, 'baseline_volatility', 0.01)
            normalized_volatility = volatility / baseline_volatility
            confidence = confidence / max(1, normalized_volatility)
        
        # Check if RSI suggests extreme conditions (avoid trading in overbought/oversold)
        if 'rsi_14' in indicators or 'rsi' in indicators:
            rsi = indicators.get('rsi_14', indicators.get('rsi', 50))
            rsi_oversold = getattr(self.config, 'rsi_oversold', 30)
            rsi_overbought = getattr(self.config, 'rsi_overbought', 70)
            
            # Reduce confidence for buy signals if market is overbought
            if predicted_move_pct > 0 and rsi > rsi_overbought:
                confidence *= 0.7
                self.logger.debug(f"Reduced BUY confidence due to overbought RSI: {rsi}")
            
            # Reduce confidence for sell signals if market is oversold
            elif predicted_move_pct < 0 and rsi < rsi_oversold:
                confidence *= 0.7
                self.logger.debug(f"Reduced SELL confidence due to oversold RSI: {rsi}")
        
        # Check if price is at Bollinger Band extremes
        if all(k in indicators for k in ['bb_upper', 'bb_lower', 'close']):
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            close = indicators['close']
            
            # Price near upper band, reduce buy confidence or increase sell confidence
            if close > bb_upper * 0.98:
                if predicted_move_pct > 0:
                    confidence *= 0.7  # Reduce buy confidence
                    self.logger.debug("Reduced BUY confidence due to price near upper BB")
                else:
                    confidence *= 1.2  # Increase sell confidence
                    confidence = min(confidence, 100)  # Cap at 100%
                    self.logger.debug("Increased SELL confidence due to price near upper BB")
            
            # Price near lower band, reduce sell confidence or increase buy confidence
            elif close < bb_lower * 1.02:
                if predicted_move_pct < 0:
                    confidence *= 0.7  # Reduce sell confidence
                    self.logger.debug("Reduced SELL confidence due to price near lower BB")
                else:
                    confidence *= 1.2  # Increase buy confidence
                    confidence = min(confidence, 100)  # Cap at 100%
                    self.logger.debug("Increased BUY confidence due to price near lower BB")
        
        # Generate signal only if confidence exceeds threshold and expected profit > costs
        if confidence > self.confidence_threshold:
            self.logger.debug(f"Confidence {confidence:.2f}% exceeds threshold {self.confidence_threshold:.2f}%")
            
            if predicted_move_pct > 0 and predicted_move_pct > self.min_profit_threshold:
                signal_type = SignalType.BUY
                self.logger.info(f"BUY signal generated with {confidence:.2f}% confidence")
                
            elif predicted_move_pct < 0 and abs(predicted_move_pct) > self.min_profit_threshold:
                signal_type = SignalType.SELL
                self.logger.info(f"SELL signal generated with {confidence:.2f}% confidence")
        
        # Add additional details to signal
        signal_data = {
            'type': signal_type,
            'direction': signal_type.name,
            'confidence': confidence,
            'predicted_move_pct': predicted_move_pct,
            'current_price': current_price,
            'indicators': indicators
        }
        
        return signal_data
    
    def should_close_position(self, 
                            position: Dict[str, Any], 
                            current_price: float, 
                            indicators: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if an existing position should be closed based on market conditions
        
        Args:
            position: Position details
            current_price: Current market price
            indicators: Dictionary of technical indicators
            
        Returns:
            Tuple of (should_close, reason)
        """
        # Check if stop loss or take profit is hit
        entry_price = position['entry_price']
        side = position['side']
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit')
        
        # Check stop loss
        if stop_loss:
            if (side == 'BUY' and current_price <= stop_loss) or \
               (side == 'SELL' and current_price >= stop_loss):
                return True, 'stop_loss'
        
        # Check take profit
        if take_profit:
            if (side == 'BUY' and current_price >= take_profit) or \
               (side == 'SELL' and current_price <= take_profit):
                return True, 'take_profit'
        
        # Check for trend reversal
        if 'rsi_14' in indicators or 'rsi' in indicators:
            rsi = indicators.get('rsi_14', indicators.get('rsi', 50))
            rsi_oversold = getattr(self.config, 'rsi_oversold', 30)
            rsi_overbought = getattr(self.config, 'rsi_overbought', 70)
            
            # For long positions, check for overbought conditions
            if side == 'BUY' and rsi > rsi_overbought:
                return True, 'overbought'
            
            # For short positions, check for oversold conditions
            if side == 'SELL' and rsi < rsi_oversold:
                return True, 'oversold'
        
        # Check moving average crossover
        if all(k in indicators for k in ['sma_7', 'sma_25']):
            sma_7 = indicators['sma_7']
            sma_25 = indicators['sma_25']
            
            # For long positions, check for bearish crossover
            if side == 'BUY' and sma_7 < sma_25:
                # Only trigger if profit exists
                if current_price > entry_price:
                    return True, 'ma_crossover'
            
            # For short positions, check for bullish crossover
            if side == 'SELL' and sma_7 > sma_25:
                # Only trigger if profit exists
                if current_price < entry_price:
                    return True, 'ma_crossover'
                    
        # Check for trend change using ADX if available
        if 'adx' in indicators and 'adx_direction' in indicators:
            adx = indicators['adx']
            adx_direction = indicators['adx_direction']
            
            # Strong trend (ADX > 25) in opposite direction of our position
            if adx > 25:
                if side == 'BUY' and adx_direction < 0:  # Strong downtrend
                    # Close if we are in profit, or ADX is very strong (>35)
                    if current_price > entry_price or adx > 35:
                        return True, 'trend_reversal'
                        
                elif side == 'SELL' and adx_direction > 0:  # Strong uptrend
                    # Close if we are in profit, or ADX is very strong (>35)
                    if current_price < entry_price or adx > 35:
                        return True, 'trend_reversal'
        
        # Default - don't close
        return False, None