import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from functools import wraps

from strategy.signal_generator import SignalType
from strategy.risk_manager import RiskManager
from data.feature_engineer import FeatureEngineer
from ai.vultr_client import VultrInferenceClient

def sync_async(async_func):
    """Decorator to run async functions in a synchronous context"""
    @wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(async_func(*args, **kwargs))
    return sync_wrapper

class LLMScalpingStrategy:
    """
    Scalping strategy implementation using Vultr Inference API with LLMs
    
    Uses LLM-based signal generation and analysis instead of local ML models
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('LLMScalpingStrategy')
        
        # Initialize components
        self.risk_manager = RiskManager(config)
        self.feature_engineer = FeatureEngineer(config)
        
        # Initialize Vultr client
        if hasattr(config, 'vultr_api_key') and config.vultr_api_key:
            model_name = getattr(config, 'ai_optimization_model', "llama-3.1-70b-instruct-fp8")
            self.vultr_client = VultrInferenceClient(config.vultr_api_key, model=model_name)
            self.logger.info(f"Initialized Vultr Inference client with model: {model_name}")
        else:
            self.vultr_client = None
            self.logger.warning("No Vultr API key provided - LLM strategy will not function")
        
        # State variables
        self.last_predictions = {}  # symbol -> prediction
        self.recent_signals = {}  # symbol -> last signal
    
    @sync_async
    async def generate_signal(self, symbol: str, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal for a symbol using LLM analysis
        
        Args:
            symbol: Trading symbol
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        # Log received data
        self.logger.debug(f"Generating signal for {symbol} with {len(ohlcv_data)} candles")
        
        try:
            # Calculate technical indicators if we have data
            indicators_df = pd.DataFrame()
            if not ohlcv_data.empty:
                indicators_df = self.feature_engineer.calculate_indicators(ohlcv_data)
            
            # Get the current price (or use a default for simulation)
            if not ohlcv_data.empty and 'close' in ohlcv_data.columns:
                current_price = ohlcv_data['close'].iloc[-1]
            else:
                # Use default price for simulation 
                current_price = 65000.0 if 'BTC' in symbol else 3500.0
            
            # Get the current timeframe
            timeframe = getattr(self.config, 'timeframe', '1m')
            
            # For simulation or if Vultr client is not available, generate a mock signal
            if self.vultr_client is None or getattr(self.config, 'api_key', '') == 'simulation_mode_key':
                # Generate simulated trading signal
                import random
                
                # Random signal with 30% chance of BUY, 30% chance of SELL, 40% chance of NEUTRAL
                rand_val = random.random()
                if rand_val < 0.3:
                    signal_type = SignalType.BUY
                    confidence = random.uniform(60, 80)  # Random confidence between 60-80%
                    stop_loss = current_price * 0.98  # 2% below current price
                    take_profit = current_price * 1.03  # 3% above current price
                    self.logger.info(f"Signal generated for {symbol}: BUY with {confidence:.2f}% confidence")
                elif rand_val < 0.6:
                    signal_type = SignalType.SELL
                    confidence = random.uniform(60, 80)  # Random confidence between 60-80%
                    stop_loss = current_price * 1.02  # 2% above current price
                    take_profit = current_price * 0.97  # 3% below current price
                    self.logger.info(f"Signal generated for {symbol}: SELL with {confidence:.2f}% confidence")
                else:
                    # Return neutral signal
                    return self._generate_neutral_signal(symbol)
                
                # Create the signal dictionary
                return {
                    'type': signal_type,
                    'direction': signal_type.name,
                    'confidence': confidence,
                    'current_price': current_price,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'predicted_move_pct': 2.0,
                    'analysis': f"Simulated {signal_type.name} signal for {symbol}",
                    'indicators': {} if indicators_df.empty else indicators_df.iloc[-1].to_dict()
                }
            
            # If we have a real Vultr client, use it
            llm_signal = self.vultr_client.generate_trading_signals(
                market_data=indicators_df,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Check if the LLM returned an error
            if 'error' in llm_signal:
                self.logger.error(f"Error in LLM signal generation: {llm_signal['error']}")
                return self._generate_neutral_signal(symbol)
            
            # Map the LLM signal to our internal format
            return self._map_llm_signal_to_internal(llm_signal, current_price, indicators_df)
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
            # Return neutral signal on error
            return self._generate_neutral_signal(symbol, error=str(e))
    
    def _map_llm_signal_to_internal(self, llm_signal: Dict[str, Any], current_price: float, indicators_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Map the LLM signal format to our internal signal format
        """
        # Default to neutral
        signal_type = SignalType.NEUTRAL
        
        # Get the signal direction from LLM
        direction = llm_signal.get('signal', '').upper()
        
        if direction == 'BUY':
            signal_type = SignalType.BUY
        elif direction == 'SELL':
            signal_type = SignalType.SELL
        
        # Extract confidence (convert from 0-1 to 0-100 if needed)
        confidence = llm_signal.get('confidence', 0)
        if confidence <= 1:  # If confidence is in 0-1 range
            confidence *= 100
        
        # Get entry, stop loss and take profit
        entry_price = llm_signal.get('entry_price', current_price)
        stop_loss = llm_signal.get('stop_loss', None)
        take_profit = llm_signal.get('take_profit', None)
        
        # Calculate predicted move percentage
        if take_profit and signal_type == SignalType.BUY:
            predicted_move_pct = ((take_profit - current_price) / current_price) * 100
        elif take_profit and signal_type == SignalType.SELL:
            predicted_move_pct = ((current_price - take_profit) / current_price) * 100
        else:
            predicted_move_pct = 0
        
        # Create signal dict in our internal format
        signal = {
            'type': signal_type,
            'direction': signal_type.name,
            'confidence': confidence,
            'current_price': current_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'predicted_move_pct': predicted_move_pct,
            'analysis': llm_signal.get('analysis', ''),
            'indicators': indicators_df.iloc[-1].to_dict() if not indicators_df.empty else {},
            'llm_indicators': llm_signal.get('indicators', {})
        }
        
        # Log signal if not neutral
        if signal['type'] != SignalType.NEUTRAL:
            self.logger.info(
                f"Signal generated for {llm_signal.get('symbol', 'unknown')}: {signal['direction']} "
                f"with {signal['confidence']:.2f}% confidence"
            )
            
            # Store in recent signals
            self.recent_signals[llm_signal.get('symbol')] = signal
        
        return signal
    
    def _generate_neutral_signal(self, symbol: str, error: str = None) -> Dict[str, Any]:
        """Generate a neutral signal"""
        return {
            'type': SignalType.NEUTRAL,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'current_price': 0,
            'predicted_move_pct': 0,
            'error': error
        }
    
    @sync_async
    async def predict_price_movement(self, symbol: str, ohlcv_data: pd.DataFrame, horizon: str = '1h') -> Dict[str, Any]:
        """
        Predict price movement using the LLM
        
        Args:
            symbol: Trading symbol
            ohlcv_data: DataFrame with OHLCV data
            horizon: Time horizon for prediction
            
        Returns:
            Dictionary with prediction details
        """
        # For simulation or if Vultr client is not available, generate a mock prediction
        if self.vultr_client is None or getattr(self.config, 'api_key', '') == 'simulation_mode_key':
            import random
            
            # Get current price or use default
            if not ohlcv_data.empty and 'close' in ohlcv_data.columns:
                current_price = ohlcv_data['close'].iloc[-1]
            else:
                current_price = 65000.0 if 'BTC' in symbol else 3500.0
                
            # Generate random prediction
            direction = random.choice(["up", "down", "sideways"])
            confidence = random.uniform(0.5, 0.9)
            
            if direction == "up":
                price_target = current_price * (1 + random.uniform(0.01, 0.05))
                analysis = f"Bullish trend detected for {symbol}. Price is expected to increase."
            elif direction == "down":
                price_target = current_price * (1 - random.uniform(0.01, 0.05))
                analysis = f"Bearish trend detected for {symbol}. Price is expected to decrease."
            else:
                price_target = current_price * (1 + random.uniform(-0.01, 0.01))
                analysis = f"Consolidation phase for {symbol}. Price expected to move sideways."
                
            return {
                "direction": direction,
                "price_target": float(price_target),
                "confidence": float(confidence),
                "analysis": analysis,
                "symbol": symbol,
                "timeframe": getattr(self.config, 'timeframe', '1m'),
                "horizon": horizon,
                "current_price": float(current_price),
                "timestamp": pd.Timestamp.now().isoformat()
            }
                
        try:
            # Calculate technical indicators
            indicators_df = self.feature_engineer.calculate_indicators(ohlcv_data)
            
            # Get the current timeframe
            timeframe = getattr(self.config, 'timeframe', '1m')
            
            # Get prediction from LLM
            prediction = self.vultr_client.predict_price_movement(
                market_data=indicators_df,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting price movement for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    @sync_async
    async def analyze_market_conditions(self, symbol: str, ohlcv_data: pd.DataFrame, trades_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze market conditions and suggest strategy adjustments
        
        Args:
            symbol: Trading symbol
            ohlcv_data: DataFrame with OHLCV data
            trades_history: List of recent trades
            
        Returns:
            Dictionary with analysis and parameter adjustments
        """
        # For simulation or if Vultr client is not available, generate a mock analysis
        if self.vultr_client is None or getattr(self.config, 'api_key', '') == 'simulation_mode_key':
            import random
            
            # Get current price or use default
            current_price = 65000.0 if 'BTC' in symbol else 3500.0
            if not ohlcv_data.empty and 'close' in ohlcv_data.columns:
                current_price = ohlcv_data['close'].iloc[-1]
            
            # Generate simulated market analysis
            market_trends = ["alcista", "bajista", "lateral"]
            risk_adjustments = ["aumentar", "mantener", "reducir"]
            
            # Randomly selected trend and risk adjustment
            market_trend = random.choice(market_trends)
            risk_adjustment = random.choice(risk_adjustments)
            
            # Calculate random parameter adjustments
            parameter_adjustments = {}
            
            # 50% chance to adjust parameters
            if random.random() > 0.5:
                if random.random() > 0.5:
                    parameter_adjustments["rsi_oversold"] = max(10, min(40, int(getattr(self.config, 'rsi_oversold', 30) + random.randint(-5, 5))))
                if random.random() > 0.5:
                    parameter_adjustments["rsi_overbought"] = max(60, min(90, int(getattr(self.config, 'rsi_overbought', 70) + random.randint(-5, 5))))
                if random.random() > 0.5:
                    parameter_adjustments["stop_loss_pct"] = max(0.2, min(2.0, round(getattr(self.config, 'stop_loss_percent', 0.5) + random.uniform(-0.2, 0.2), 2)))
                if random.random() > 0.5:
                    parameter_adjustments["take_profit_pct"] = max(0.5, min(3.0, round(getattr(self.config, 'take_profit_percent', 1.0) + random.uniform(-0.3, 0.3), 2)))
            
            # Generate analysis text based on market trend
            analysis_texts = {
                "alcista": f"El mercado de {symbol} muestra tendencia alcista con soporte fuerte en {current_price * 0.97:.2f}. Recomendable mantener posiciones largas.",
                "bajista": f"El mercado de {symbol} muestra señales de debilidad con resistencia en {current_price * 1.03:.2f}. Precaución con posiciones largas.",
                "lateral": f"El mercado de {symbol} está en consolidación entre {current_price * 0.98:.2f} y {current_price * 1.02:.2f}. Esperar ruptura de rango."
            }
            
            analysis = analysis_texts[market_trend]
            
            # Create full response
            return {
                "analysis": analysis,
                "parameter_adjustments": parameter_adjustments,
                "reasoning": f"Basado en las condiciones actuales del mercado y el rendimiento reciente del bot, se recomienda {risk_adjustment} la exposición al riesgo.",
                "market_trend": market_trend,
                "risk_adjustment": risk_adjustment,
                "confidence": round(random.uniform(0.7, 0.9), 2)
            }
        
        try:
            # Prepare market data for analysis
            market_data = {
                'symbol': symbol,
                'timeframe': getattr(self.config, 'timeframe', '1m'),
                'current_price': ohlcv_data['close'].iloc[-1] if not ohlcv_data.empty else 0,
                'daily_range': {
                    'high': ohlcv_data['high'].max() if not ohlcv_data.empty else 0,
                    'low': ohlcv_data['low'].min() if not ohlcv_data.empty else 0,
                    'volatility': ohlcv_data['high'].max() / ohlcv_data['low'].min() - 1 if not ohlcv_data.empty else 0
                },
                'volume': {
                    'latest': ohlcv_data['volume'].iloc[-1] if not ohlcv_data.empty else 0,
                    'average': ohlcv_data['volume'].mean() if not ohlcv_data.empty else 0
                }
            }
            
            # Prepare performance history
            total_trades = len(trades_history)
            winning_trades = sum(1 for t in trades_history if t.get('profit_loss', 0) > 0)
            
            performance_history = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'avg_profit': sum(t.get('profit_loss', 0) for t in trades_history) / total_trades if total_trades > 0 else 0,
                'avg_hold_time': 0,  # Simplified for now
                'recent_trades': trades_history[-5:] if len(trades_history) > 5 else trades_history
            }
            
            # Get current parameters
            current_parameters = {
                'confidence_threshold': getattr(self.config, 'confidence_threshold', 0.6),
                'min_profit_threshold': getattr(self.config, 'min_profit_threshold', 0.2),
                'stop_loss_pct': getattr(self.config, 'stop_loss_percent', 0.5),
                'take_profit_pct': getattr(self.config, 'take_profit_percent', 1.0),
                'trailing_stop_pct': getattr(self.config, 'trailing_stop_pct', 0.3),
                'max_risk_per_trade': getattr(self.config, 'max_risk_per_trade', 1.0)
            }
            
            # Get analysis from LLM
            analysis = self.vultr_client.analyze_market_conditions(
                market_data=market_data,
                performance_history=performance_history,
                current_parameters=current_parameters
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions for {symbol}: {str(e)}")
            return {'error': str(e)}
    
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
        
        # Check RSI for extreme conditions
        if 'rsi_14' in indicators or 'rsi' in indicators:
            rsi = indicators.get('rsi_14', indicators.get('rsi', 50))
            rsi_oversold = getattr(self.config, 'rsi_oversold', 30)
            rsi_overbought = getattr(self.config, 'rsi_overbought', 70)
            
            # For long positions, check for overbought conditions
            if side == 'BUY' and rsi > rsi_overbought:
                # Only trigger if in profit
                if current_price > entry_price:
                    return True, 'overbought'
            
            # For short positions, check for oversold conditions
            if side == 'SELL' and rsi < rsi_oversold:
                # Only trigger if in profit
                if current_price < entry_price:
                    return True, 'oversold'
        
        # Check for MA crossover
        if all(k in indicators for k in ['ema_9', 'ema_21']):
            ema_9 = indicators['ema_9']
            ema_21 = indicators['ema_21']
            
            # For long positions, check for bearish crossover
            if side == 'BUY' and ema_9 < ema_21:
                # Only trigger if in profit
                if current_price > entry_price:
                    return True, 'ma_crossover'
            
            # For short positions, check for bullish crossover
            if side == 'SELL' and ema_9 > ema_21:
                # Only trigger if in profit
                if current_price < entry_price:
                    return True, 'ma_crossover'
        
        # Default - don't close
        return False, None
    
    def calculate_position_size(self, 
                              symbol: str,
                              account_balance: float, 
                              entry_price: float, 
                              stop_loss: float) -> float:
        """
        Calculate position size for a trade
        
        Args:
            symbol: Trading symbol
            account_balance: Available balance in the quote currency
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            
        Returns:
            Calculated position size
        """
        # Calculate risk amount (1% of account balance by default)
        risk_percent = getattr(self.config, 'max_risk_per_trade', 1.0) / 100
        risk_amount = account_balance * risk_percent
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Calculate position size
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
        else:
            # Fallback to minimum size if risk is zero (should never happen)
            position_size = 0.001
            self.logger.warning(f"Risk per unit is 0, using minimum position size for {symbol}")
        
        # Apply constraints based on symbol
        min_order_size = 0.001  # Default minimum
        if 'BTC' in symbol:
            min_order_size = 0.001
        elif 'ETH' in symbol:
            min_order_size = 0.01
        
        # Never risk more than 5% of account in a single trade
        max_size = account_balance * 0.05 / entry_price
        
        # Return constrained position size
        result = max(min_order_size, min(position_size, max_size))
        self.logger.info(f"Calculated position size: {result} for {symbol}")
        return result
    
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