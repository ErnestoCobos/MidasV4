import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import asyncio

from strategy.signal_generator import SignalGenerator, SignalType
from strategy.risk_manager import RiskManager
from data.feature_engineer import FeatureEngineer
from models.deep_scalper import RLTradingModel

class RLStrategy:
    """
    Reinforcement Learning based Trading Strategy
    
    Implements a risk-aware RL framework for intraday trading that uses 
    dueling Q-networks with action branching to capture fleeting trading 
    opportunities while managing risk.
    """
    
    def __init__(self, config, model=None, binance_client=None):
        """Initialize the RL strategy with configuration"""
        self.config = config
        self.logger = logging.getLogger('RLStrategy')
        
        # Initialize components
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)
        self.feature_engineer = FeatureEngineer(config)
        self.binance_client = binance_client
        
        # Initialize RL model
        if model is not None and isinstance(model, RLTradingModel):
            self.model = model
            self.logger.info("Using provided RL trading model")
        else:
            self.logger.info("Creating new RL trading model")
            self.model = RLTradingModel(config)
            
        # State variables
        self.state_buffer = {}  # symbol -> deque of states
        self.position_history = {}  # symbol -> list of positions
        self.episode_memory = {}  # symbol -> list of (s,a,r,s',done)
        self.last_action = {}  # symbol -> (action_type, position_size)
        self.last_state = {}  # symbol -> state
        
        # Risk metrics for risk-aware decision making
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.trade_count = 0
        self.profitable_trades = 0
        
        # Initialize market regime detector if available
        try:
            from strategy.market_regime import MarketRegimeDetector, MarketRegime
            self.regime_detector = MarketRegimeDetector(config)
            self.MarketRegime = MarketRegime
            self.current_regime = None
            self.logger.info("Market regime detector initialized successfully")
        except ImportError:
            self.logger.warning("Market regime detector not available")
            self.regime_detector = None
            self.current_regime = None
    
    def preprocess_state(self, symbol: str, features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess raw features into multi-modal state representation for DeepScalper model
        
        Args:
            symbol: Trading symbol
            features: Dictionary of features and indicators
            
        Returns:
            Tuple of (micro_state, macro_state, private_state) arrays
        """
        # Initialize state buffer if needed
        if symbol not in self.state_buffer:
            self.state_buffer[symbol] = {
                'micro': [],
                'macro': [],
                'private': []
            }
            self.logger.debug(f"Initialized multi-modal state buffer for {symbol}")
        
        # --- MICRO FEATURES (high-frequency data) ---
        micro_features = []
        
        # Price information - these are the most high-frequency features
        for key in ['open', 'high', 'low', 'close', 'price']:
            if key in features:
                micro_features.append(features[key])
            else:
                micro_features.append(0)
                
        # Candle shapes
        for key in ['body_size', 'upper_shadow', 'lower_shadow']:
            if key in features:
                micro_features.append(features[key])
            else:
                micro_features.append(0)
        
        # Volume data
        for key in ['volume', 'current_volume', 'relative_volume']:
            if key in features:
                micro_features.append(features[key])
            else:
                micro_features.append(0)
                
        # Short-term indicators
        for key in ['ma_dist_7', 'ma_dist_14', 'ma_dist_25']:
            if key in features:
                micro_features.append(features[key])
            else:
                micro_features.append(0)
                
        # Fill remaining micro features with zeros
        micro_dim = getattr(self.config, 'micro_dim', 20)
        while len(micro_features) < micro_dim:
            micro_features.append(0)
            
        # Truncate if too many
        if len(micro_features) > micro_dim:
            micro_features = micro_features[:micro_dim]
            
        # --- MACRO FEATURES (technical indicators) ---
        macro_features = []
        
        # Various technical indicators
        indicator_keys = [
            'sma_7', 'sma_25',
            'bb_upper', 'bb_lower', 'bb_middle',
            'rsi', 'rsi_14',
            'volatility_14',
            'adx', 'cci',
            'macd', 'macd_signal', 'macd_hist',
            'obv',
            'return_7', 'return_14', 'return_25'
        ]
        
        for key in indicator_keys:
            if key in features:
                macro_features.append(features[key])
            elif key == 'rsi' and 'rsi_14' in features:
                macro_features.append(features['rsi_14'])
            elif key == 'rsi_14' and 'rsi' in features:
                macro_features.append(features['rsi'])
            else:
                macro_features.append(0)
                
        # Fill remaining macro features with zeros
        macro_dim = getattr(self.config, 'macro_dim', 11)
        while len(macro_features) < macro_dim:
            macro_features.append(0)
            
        # Truncate if too many
        if len(macro_features) > macro_dim:
            macro_features = macro_features[:macro_dim]
            
        # --- PRIVATE FEATURES (position, capital, time) ---
        private_features = []
        
        # Current position size (normalized)
        current_position = 0
        if 'current_position' in features:
            current_position = features['current_position']
        elif hasattr(self, 'risk_manager') and symbol in self.risk_manager.open_positions:
            position = self.risk_manager.open_positions[symbol]
            if position['side'] == 'BUY':
                current_position = position['quantity']
            elif position['side'] == 'SELL':
                current_position = -position['quantity']
        private_features.append(current_position)
        
        # Available capital (normalized)
        available_capital = 1.0  # Default to full capital
        if 'available_capital' in features:
            available_capital = features['available_capital']
        elif hasattr(self, 'binance_client') and self.binance_client:
            try:
                # Try to get from client
                if hasattr(self.binance_client, 'get_account_balance'):
                    account_balance = self.binance_client.get_account_balance()
                    quote_asset = symbol[3:]  # e.g., 'USDT' from 'BTCUSDT'
                    if quote_asset in account_balance:
                        available_capital = account_balance[quote_asset] / 10000  # Normalize
            except Exception as e:
                pass
        private_features.append(available_capital)
        
        # Time features (hour of day normalized)
        time_feature = 0.5  # Default mid-day
        if 'hour' in features:
            time_feature = features['hour'] / 24.0
        private_features.append(time_feature)
        
        # Ensure private features have correct dimension
        private_dim = getattr(self.config, 'private_dim', 3)
        while len(private_features) < private_dim:
            private_features.append(0)
            
        # Truncate if too many
        if len(private_features) > private_dim:
            private_features = private_features[:private_dim]
            
        # --- STORE FEATURE ARRAYS ---
        micro_array = np.array(micro_features)
        macro_array = np.array(macro_features)
        private_array = np.array(private_features)
        
        # Add to state buffers
        self.state_buffer[symbol]['micro'].append(micro_array)
        self.state_buffer[symbol]['macro'].append(macro_array)
        
        # Manage buffer sizes
        micro_seq_len = getattr(self.config, 'micro_seq_len', 30)
        macro_seq_len = getattr(self.config, 'macro_seq_len', 30)
        
        # Trim to sequence length
        if len(self.state_buffer[symbol]['micro']) > micro_seq_len:
            self.state_buffer[symbol]['micro'] = self.state_buffer[symbol]['micro'][-micro_seq_len:]
        if len(self.state_buffer[symbol]['macro']) > macro_seq_len:
            self.state_buffer[symbol]['macro'] = self.state_buffer[symbol]['macro'][-macro_seq_len:]
        
        # --- CREATE PADDED SEQUENCES ---
        # Pad micro sequence if needed
        if len(self.state_buffer[symbol]['micro']) < micro_seq_len:
            padding = [np.zeros_like(micro_array) for _ in range(micro_seq_len - len(self.state_buffer[symbol]['micro']))]
            micro_padded = padding + self.state_buffer[symbol]['micro']
        else:
            micro_padded = self.state_buffer[symbol]['micro']
            
        # Pad macro sequence if needed
        if len(self.state_buffer[symbol]['macro']) < macro_seq_len:
            padding = [np.zeros_like(macro_array) for _ in range(macro_seq_len - len(self.state_buffer[symbol]['macro']))]
            macro_padded = padding + self.state_buffer[symbol]['macro']
        else:
            macro_padded = self.state_buffer[symbol]['macro']
            
        # Convert to numpy arrays with shape (sequence_length, feature_count)
        micro_state = np.array(micro_padded)
        macro_state = np.array(macro_padded)
        private_state = private_array
        
        return micro_state, macro_state, private_state
    
    def _map_action_to_signal(self, action: Tuple[int, int], current_price: float) -> Dict[str, Any]:
        """
        Map RL action to trading signal
        
        Args:
            action: Tuple of (action_type, position_size)
            current_price: Current market price
            
        Returns:
            Signal dictionary
        """
        action_type, position_size = action
        
        # Map action_type to SignalType
        if action_type == 0:  # BUY
            signal_type = SignalType.BUY
            direction = 'BUY'
            predicted_move_pct = 0.2 * (position_size + 1)  # Scale by position size
        elif action_type == 1:  # SELL
            signal_type = SignalType.SELL
            direction = 'SELL'
            predicted_move_pct = -0.2 * (position_size + 1)  # Scale by position size
        else:  # HOLD
            signal_type = SignalType.NEUTRAL
            direction = 'NEUTRAL'
            predicted_move_pct = 0
        
        # Calculate confidence based on position size (0-4)
        confidence = min(100, 50 + 10 * position_size)
        
        # Create signal dictionary
        signal = {
            'type': signal_type,
            'direction': direction,
            'confidence': confidence,
            'current_price': current_price,
            'predicted_move_pct': predicted_move_pct,
            'position_size': position_size,
            'risk_adjusted': True
        }
        
        return signal
    
    async def generate_signal(self, symbol: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal for a symbol using RL model
        
        Args:
            symbol: Trading symbol
            features: Dictionary of features and indicators
            
        Returns:
            Signal dictionary
        """
        # Log received features
        self.logger.debug(f"Generating RL signal for {symbol} with {len(features)} features")
        
        try:
            # Apply market regime detection if available
            if hasattr(self, 'regime_detector') and self.regime_detector:
                try:
                    # Create OHLCV dataframe from features for regime detection
                    ohlcv_data = pd.DataFrame([features])
                    
                    # Detect market regime
                    regime_info = self.regime_detector.detect_regime(ohlcv_data)
                    self.current_regime = regime_info['regime']
                    
                    self.logger.info(f"Market regime for {symbol}: {regime_info['regime'].name} "
                                   f"(confidence: {regime_info['confidence']:.2f})")
                    
                    # Add regime info to features safely
                    if isinstance(features, dict):
                        features['market_regime'] = regime_info['regime'].value
                        features['regime_confidence'] = regime_info['confidence']
                    
                except Exception as e:
                    self.logger.warning(f"Error detecting market regime: {str(e)}")
            
            # Preprocess features to get state
            current_state = self.preprocess_state(symbol, features)
            
            # Store for experience replay
            self.last_state[symbol] = current_state
            
            # Get current price
            current_price = features.get('close', features.get('price', 0))
            
            # If we don't have enough state history, use traditional strategy
            if symbol not in self.state_buffer or (
                isinstance(self.state_buffer[symbol], dict) and (
                len(self.state_buffer[symbol]['micro']) < getattr(self.config, 'micro_seq_len', 30) or
                len(self.state_buffer[symbol]['macro']) < getattr(self.config, 'macro_seq_len', 30)
                )
            ):
                self.logger.info(f"Not enough state history for {symbol}, using indicator-based signal")
                # Fall back to indicator-based signal
                return self._generate_indicator_signal(symbol, features)
            
            # Choose action using RL model (with exploration during training)
            is_training = getattr(self.config, 'rl_training_mode', False)
            action = self.model.choose_action(current_state, explore=is_training)
            
            # Store action
            self.last_action[symbol] = action
            
            # Map RL action to signal dictionary
            signal = self._map_action_to_signal(action, current_price)
            
            # Log significant actions
            if signal['type'] != SignalType.NEUTRAL:
                self.logger.info(
                    f"RL signal for {symbol}: {signal['direction']} "
                    f"with {signal['confidence']:.2f}% confidence, "
                    f"position size {signal['position_size']}"
                )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating RL signal for {symbol}: {str(e)}")
            # Return neutral signal on error
            return {
                'type': SignalType.NEUTRAL,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'error': str(e)
            }
    
    def _generate_indicator_signal(self, symbol: str, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signal based only on indicators (fallback when RL model not ready)
        
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
        
        # Apply more conservative thresholds for spot trading
        rsi_oversold = getattr(self.config, 'rsi_oversold', 30)
        rsi_overbought = getattr(self.config, 'rsi_overbought', 70)
        
        # Add risk-awareness to avoid drawdowns
        risk_multiplier = 1.0
        
        # Reduce risk when experiencing drawdown
        if self.drawdown > 0.05:  # 5% drawdown
            risk_multiplier = 0.75
        if self.drawdown > 0.1:   # 10% drawdown
            risk_multiplier = 0.5
        if self.drawdown > 0.15:  # 15% drawdown
            risk_multiplier = 0.25
        if self.drawdown > 0.2:   # 20% drawdown
            # Very conservative, practically no trading
            return {
                'type': SignalType.NEUTRAL,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'current_price': current_price,
                'predicted_move_pct': 0,
                'reason': 'Excessive drawdown protection'
            }
        
        # RSI oversold + price below lower BB + SMA7 > SMA25 = buy signal
        if (rsi < rsi_oversold and 
            current_price < bb_lower * 1.01 and 
            sma_7 > sma_25):
            
            signal_type = SignalType.BUY
            confidence = (70 + (rsi_oversold - rsi)) * risk_multiplier  # Higher confidence if RSI lower
            predicted_move_pct = 0.5 * risk_multiplier  # Expect 0.5% upward move
            reason = f"RSI oversold ({rsi:.2f}) + price below lower BB + bullish SMA crossover"
            self.logger.info(f"BUY signal generated for {symbol}: {reason}")
        
        # RSI overbought + price above upper BB + SMA7 < SMA25 = sell signal
        elif (rsi > rsi_overbought and 
              current_price > bb_upper * 0.99 and 
              sma_7 < sma_25):
              
            signal_type = SignalType.SELL
            confidence = (70 + (rsi - rsi_overbought)) * risk_multiplier  # Higher confidence if RSI higher
            predicted_move_pct = -0.5 * risk_multiplier  # Expect 0.5% downward move
            reason = f"RSI overbought ({rsi:.2f}) + price above upper BB + bearish SMA crossover"
            self.logger.info(f"SELL signal generated for {symbol}: {reason}")
        
        # Just RSI oversold + price below lower BB = weaker buy signal
        elif rsi < rsi_oversold and current_price < bb_lower * 1.01:
            signal_type = SignalType.BUY
            confidence = (60 + (rsi_oversold - rsi) / 2) * risk_multiplier  # Medium confidence
            predicted_move_pct = 0.3 * risk_multiplier  # Expect 0.3% upward move
            reason = f"RSI oversold ({rsi:.2f}) + price below lower BB"
            self.logger.info(f"BUY signal generated for {symbol}: {reason}")
        
        # Just RSI overbought + price above upper BB = weaker sell signal
        elif rsi > rsi_overbought and current_price > bb_upper * 0.99:
            signal_type = SignalType.SELL
            confidence = (60 + (rsi - rsi_overbought) / 2) * risk_multiplier  # Medium confidence
            predicted_move_pct = -0.3 * risk_multiplier  # Expect 0.3% downward move
            reason = f"RSI overbought ({rsi:.2f}) + price above upper BB"
            self.logger.info(f"SELL signal generated for {symbol}: {reason}")
        
        return {
            'type': signal_type,
            'direction': signal_type.name,
            'confidence': confidence,
            'current_price': current_price,
            'predicted_move_pct': predicted_move_pct,
            'indicators': indicators,
            'risk_multiplier': risk_multiplier
        }
    
    def record_reward(self, symbol: str, reward: float, done: bool = False) -> None:
        """
        Record reward for last action
        
        Args:
            symbol: Trading symbol
            reward: Reward value
            done: Whether episode is done
        """
        if symbol in self.last_state and symbol in self.last_action:
            # Get current state
            current_state = self.last_state[symbol]
            
            # Store experience
            if symbol not in self.episode_memory:
                self.episode_memory[symbol] = []
                
            # Create tuple of (state, action, reward, next_state, done)
            experience = (current_state, self.last_action[symbol], reward, current_state, done)
            self.episode_memory[symbol].append(experience)
            
            # Store in model's replay buffer
            self.model.remember(*experience)
            
            self.logger.debug(f"Recorded reward {reward:.4f} for {symbol}")
    
    def calculate_position_size(self, 
                              symbol: str,
                              account_balance: float = None,
                              entry_price: float = None, 
                              stop_loss: float = None,
                              direction: str = "BUY",
                              volatility: float = 0.01) -> float:
        """
        Calculate position size for a trade using risk-aware sizing
        
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
                entry_price = 0
                
            # If stop loss is not provided, calculate it
            if stop_loss is None:
                # Calculate a conservative stop loss by default
                base_stop_loss_pct = getattr(self.config, 'base_stop_loss_pct', 1.0) / 100
                if direction == "BUY":
                    stop_loss = entry_price * (1 - base_stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 + base_stop_loss_pct)
            
            # Calculate risk-based position size
            # Start with conservative max risk (0.5% of account by default)
            base_risk_pct = getattr(self.config, 'max_risk_per_trade', 0.5) / 100
            
            # Apply risk adjustment based on drawdown
            risk_pct = base_risk_pct
            if self.drawdown > 0.05:  # 5% drawdown
                risk_pct *= 0.8
            if self.drawdown > 0.1:   # 10% drawdown
                risk_pct *= 0.6
            if self.drawdown > 0.15:  # 15% drawdown
                risk_pct *= 0.4
            if self.drawdown > 0.2:   # 20% drawdown
                risk_pct *= 0.2
                
            # Also adjust based on win rate
            if self.trade_count > 10:
                win_rate = self.profitable_trades / self.trade_count
                if win_rate < 0.4:
                    risk_pct *= 0.8
                elif win_rate > 0.6:
                    risk_pct *= 1.2
            
            # Calculate risk amount
            risk_amount = account_balance * risk_pct
            
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
            
            # Additional safety margin for spot trading
            safety_margin = getattr(self.config, 'safety_margin_pct', 2) / 100
            quantity = quantity * (1 - safety_margin)
            
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
                
            self.logger.info(f"RL calculated position size for {symbol}: {quantity} units at {entry_price} "
                           f"(risk: {risk_pct*100:.2f}%, DD: {self.drawdown*100:.1f}%)")
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.001  # Minimum quantity as fallback
    
    def update_risk_metrics(self, 
                           current_balance: float,
                           trade_result: float = None,
                           is_profitable: bool = None) -> None:
        """
        Update risk metrics for risk-aware trading
        
        Args:
            current_balance: Current account balance
            trade_result: Profit/loss from recent trade (optional)
            is_profitable: Whether recent trade was profitable (optional)
        """
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown
        if self.peak_balance > 0:
            self.drawdown = 1 - (current_balance / self.peak_balance)
            if self.drawdown > self.max_drawdown:
                self.max_drawdown = self.drawdown
        
        # Update trade statistics
        if trade_result is not None:
            self.trade_count += 1
            if is_profitable:
                self.profitable_trades += 1
                
        # Log risk metrics
        self.logger.debug(f"Risk metrics - Balance: {current_balance:.2f}, Peak: {self.peak_balance:.2f}, "
                        f"DD: {self.drawdown*100:.2f}%, Max DD: {self.max_drawdown*100:.2f}%, "
                        f"Win Rate: {self.profitable_trades}/{self.trade_count}")
    
    def should_close_position(self, 
                            symbol: str, 
                            position: Dict[str, Any],
                            current_price: float, 
                            indicators: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a position should be closed using RL model
        
        Args:
            symbol: Trading symbol
            position: Position details
            current_price: Current market price
            indicators: Technical indicators
            
        Returns:
            Tuple of (should_close, reason)
        """
        # First check stop loss and take profit (these override RL decisions)
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
        
        # If we have enough state history, use RL model
        if symbol in self.state_buffer and len(self.state_buffer[symbol]) >= self.model.sequence_length:
            try:
                # Get current state
                current_state = self.preprocess_state(symbol, indicators)
                
                # Choose action using RL model (without exploration)
                action = self.model.choose_action(current_state, explore=False)
                action_type, _ = action
                
                # If we're in a LONG position and model suggests SELL, close
                if side == 'BUY' and action_type == 1:
                    return True, 'rl_model'
                
                # If we're in a SHORT position and model suggests BUY, close
                if side == 'SELL' and action_type == 0:
                    return True, 'rl_model'
                
            except Exception as e:
                self.logger.error(f"Error in RL position closing: {str(e)}")
        
        # If RL didn't suggest closing, check indicator-based rules
        return self.signal_generator.should_close_position(position, current_price, indicators)
    
    def process_trade_result(self, 
                            symbol: str, 
                            entry_price: float, 
                            exit_price: float,
                            side: str,
                            quantity: float) -> None:
        """
        Process trade result for learning
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            side: Trade direction ('BUY' or 'SELL')
            quantity: Trade quantity
        """
        # Calculate P&L
        if side == 'BUY':
            pnl = (exit_price - entry_price) * quantity
            pct_return = (exit_price - entry_price) / entry_price * 100
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
            pct_return = (entry_price - exit_price) / entry_price * 100
        
        is_profitable = pnl > 0
        
        # Log result
        self.logger.info(f"Trade result for {symbol}: {pnl:.2f} ({pct_return:.2f}%)")
        
        # Obtener el balance actual de forma más robusta
        current_balance = 1000  # Valor predeterminado
        try:
            if hasattr(self.config, 'account_balance') and self.config.account_balance is not None:
                current_balance = self.config.account_balance
            elif hasattr(self, 'binance_client') and self.binance_client:
                # Intentar obtener balance desde cliente
                if hasattr(self.binance_client, 'get_account_balance'):
                    account_info = self.binance_client.get_account_balance()
                    if account_info and 'total' in account_info:
                        current_balance = account_info['total']
        except Exception as e:
            self.logger.warning(f"Error getting current balance: {str(e)}")
        
        # Update risk metrics
        self.update_risk_metrics(
            current_balance=current_balance,
            trade_result=pnl,
            is_profitable=is_profitable
        )
        
        # Calculate reward for RL model
        # Base reward is proportional to P&L
        base_reward = pnl / (entry_price * quantity) * 10  # Scale for better learning
        
        # Add drawdown penalty if we're in significant drawdown
        drawdown_penalty = 0
        if self.drawdown > 0.1:
            drawdown_penalty = self.drawdown * 2
        
        # Final reward calculation
        reward = base_reward - drawdown_penalty
        
        # Record result for learning
        self.record_reward(symbol, reward, done=True)
        
        # If we have a complete episode, use hindsight experience replay
        if symbol in self.episode_memory and len(self.episode_memory[symbol]) > 5:
            self.model.hindsight_experience_replay(self.episode_memory[symbol])
            self.episode_memory[symbol] = []  # Reset episode memory
            
        # Potentially update the model with batch training
        if getattr(self.config, 'rl_online_learning', False) and len(self.model.memory) > self.model.batch_size:
            loss = self.model.replay()
            self.logger.debug(f"Online learning update, loss: {np.mean(loss) if loss else 'N/A'}")
    
    async def train_online(self, 
                         symbol: str, 
                         num_updates: int = 10) -> Dict[str, Any]:
        """
        Perform online training updates
        
        Args:
            symbol: Trading symbol to train for
            num_updates: Number of training updates to perform
            
        Returns:
            Training metrics
        """
        losses = []
        
        # Skip if not enough experiences
        if len(self.model.memory) < self.model.batch_size * 2:
            return {'status': 'skipped', 'reason': 'insufficient_experiences'}
        
        self.logger.info(f"Performing {num_updates} online training updates for {symbol}")
        
        # Perform updates
        for i in range(num_updates):
            batch_loss = self.model.replay()
            if batch_loss:
                losses.append(np.mean(batch_loss))
        
        avg_loss = np.mean(losses) if losses else 0
        self.logger.info(f"Online training complete - Average loss: {avg_loss:.4f}")
        
        return {
            'status': 'success',
            'updates': num_updates,
            'avg_loss': float(avg_loss),
            'memory_size': len(self.model.memory),
            'epsilon': self.model.epsilon
        }
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all open positions from risk manager"""
        return self.risk_manager.open_positions
    
    def register_position(self, symbol: str, position_details: Dict[str, Any]) -> None:
        """Register a new position with risk manager"""
        self.risk_manager.register_position(symbol, position_details)
    
    def close_position(self, symbol: str) -> None:
        """Close a position with risk manager"""
        self.risk_manager.close_position(symbol)
    
    def update_trailing_stops(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Update trailing stops with risk manager"""
        return self.risk_manager.update_trailing_stops(current_prices)