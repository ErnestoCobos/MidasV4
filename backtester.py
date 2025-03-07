import pandas as pd
import numpy as np
import os
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from strategy.scalping_strategy import ScalpingStrategy
from models.ml_module import MLModule
from strategy.risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Backtester')

class MockBinanceClient:
    """Mock BinanceClient for backtesting"""
    
    def __init__(self, config):
        self.config = config
        self.simulation_mode = True
        
        # Commission and slippage for realistic simulation
        self.commission_rate = getattr(config, 'commission_rate', 0.0004)  # 0.04% by default
        self.slippage_pct = getattr(config, 'slippage_pct', 0.0002)       # 0.02% by default
        
        # Data storage
        self.data = {}  # symbol -> DataFrame
        self.current_idx = {}  # symbol -> current index in data
        
    def load_data(self, symbol: str, data: pd.DataFrame):
        """Load historical data for a symbol"""
        self.data[symbol] = data
        self.current_idx[symbol] = 0
        logger.info(f"Loaded {len(data)} data points for {symbol}")
        
    def get_current_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current data point for a symbol"""
        if symbol not in self.data or symbol not in self.current_idx:
            logger.error(f"No data loaded for {symbol}")
            return None
            
        idx = self.current_idx[symbol]
        if idx >= len(self.data[symbol]):
            logger.info(f"Reached end of data for {symbol}")
            return None
            
        return self.data[symbol].iloc[idx].to_dict()
        
    def advance(self, symbol: str, steps: int = 1):
        """Advance to next data point for a symbol"""
        if symbol not in self.current_idx:
            logger.error(f"No data loaded for {symbol}")
            return
            
        self.current_idx[symbol] += steps
        
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        data = self.get_current_data(symbol)
        if data is None:
            return 0.0
            
        return data.get('close', 0.0)
        
    def get_ohlcv(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data as a pandas DataFrame"""
        if symbol not in self.data:
            logger.error(f"No data loaded for {symbol}")
            return pd.DataFrame()
            
        idx = self.current_idx[symbol]
        start_idx = max(0, idx - limit + 1)
        end_idx = idx + 1
        
        return self.data[symbol].iloc[start_idx:end_idx].copy()
        
    def calculate_indicators(self, symbol: str, interval: str = '1m', limit: int = 100) -> Dict[str, Any]:
        """Calculate technical indicators for backtesting"""
        if symbol not in self.data:
            logger.error(f"No data loaded for {symbol}")
            return {}
            
        # Get current data
        data = self.get_current_data(symbol)
        if data is None:
            return {}
            
        # Get historical data for indicator calculation
        ohlcv_data = self.get_ohlcv(symbol, interval, limit)
        
        # Calculate indicators
        indicators = {}
        
        # Add basic price data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data:
                indicators[col] = data[col]
        
        # Simple Moving Averages
        indicators['sma_7'] = ohlcv_data['close'].rolling(window=7).mean().iloc[-1]
        indicators['sma_25'] = ohlcv_data['close'].rolling(window=25).mean().iloc[-1]
        
        # Bollinger Bands
        sma_20 = ohlcv_data['close'].rolling(window=20).mean()
        std_20 = ohlcv_data['close'].rolling(window=20).std()
        indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
        indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
        indicators['bb_middle'] = sma_20.iloc[-1]
        
        # RSI (14 periods)
        delta = ohlcv_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        indicators['rsi_14'] = indicators['rsi']  # alias
        
        # Volatility
        indicators['volatility_14'] = ohlcv_data['close'].pct_change().rolling(window=14).std().iloc[-1]
        
        # Volume indicators
        indicators['volume_sma'] = ohlcv_data['volume'].rolling(window=20).mean().iloc[-1]
        indicators['current_volume'] = data['volume']
        indicators['relative_volume'] = data['volume'] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0
        
        # MA Distance
        indicators['ma_dist_7'] = (data['close'] - indicators['sma_7']) / indicators['sma_7'] if indicators['sma_7'] > 0 else 0
        indicators['ma_dist_14'] = (data['close'] - indicators['bb_middle']) / indicators['bb_middle'] if indicators['bb_middle'] > 0 else 0
        indicators['ma_dist_25'] = (data['close'] - indicators['sma_25']) / indicators['sma_25'] if indicators['sma_25'] > 0 else 0
        
        # Returns
        indicators['return_7'] = (data['close'] / ohlcv_data['close'].iloc[-8] - 1) if len(ohlcv_data) >= 8 else 0
        indicators['return_14'] = (data['close'] / ohlcv_data['close'].iloc[-15] - 1) if len(ohlcv_data) >= 15 else 0
        indicators['return_25'] = (data['close'] / ohlcv_data['close'].iloc[-26] - 1) if len(ohlcv_data) >= 26 else 0
        
        # Candlestick patterns
        indicators['body_size'] = abs(data['close'] - data['open']) / data['open'] if data['open'] > 0 else 0
        indicators['upper_shadow'] = (data['high'] - max(data['open'], data['close'])) / data['open'] if data['open'] > 0 else 0
        indicators['lower_shadow'] = (min(data['open'], data['close']) - data['low']) / data['open'] if data['open'] > 0 else 0
        
        # Add symbol and price for compatibility
        indicators['symbol'] = symbol
        indicators['price'] = data['close']
        
        return indicators
        
    def apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price based on order side"""
        if side == 'BUY':
            # For buys, price slips upward (worse entry price)
            return price * (1 + self.slippage_pct)
        else:
            # For sells, price slips downward (worse entry price)
            return price * (1 - self.slippage_pct)
            
    def apply_commission(self, price: float, quantity: float) -> float:
        """Calculate commission for a trade"""
        return price * quantity * self.commission_rate
        
    def execute_trade(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Execute a simulated trade with slippage and commission"""
        current_data = self.get_current_data(symbol)
        if current_data is None:
            logger.error(f"No data available for {symbol}")
            return {
                'success': False,
                'error': 'No data available'
            }
            
        # Get current price if not provided
        if price is None:
            price = current_data['close']
            
        # Apply slippage
        execution_price = self.apply_slippage(price, side)
        
        # Calculate commission
        commission = self.apply_commission(execution_price, quantity)
        
        # Calculate trade value
        trade_value = execution_price * quantity
        
        return {
            'success': True,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'requested_price': price,
            'execution_price': execution_price,
            'slippage_pct': self.slippage_pct,
            'commission': commission,
            'commission_rate': self.commission_rate,
            'trade_value': trade_value,
            'timestamp': current_data.get('timestamp', datetime.now())
        }

class Backtester:
    """
    Backtester for MidasScalpingv4
    
    Tests strategy performance using historical data with realistic simulation
    of trading conditions including:
    
    1. Slippage
    2. Commission
    3. Trade execution delay
    4. Risk management
    """
    
    def __init__(self, config: Config):
        """Initialize backtester with configuration"""
        self.config = config
        self.logger = logging.getLogger('Backtester')
        
        # Initialize mock client
        self.client = MockBinanceClient(config)
        
        # Initialize ML module if enabled
        self.ml_module = None
        if hasattr(config, 'use_ml') and config.use_ml:
            try:
                self.ml_module = MLModule(config)
                self.logger.info("ML module initialized successfully")
                
                # Load models if paths provided
                if hasattr(config, 'xgb_model_path') and config.xgb_model_path:
                    success = self.ml_module.load_xgboost(config.xgb_model_path)
                    if success:
                        self.logger.info(f"XGBoost model loaded from {config.xgb_model_path}")
                    else:
                        self.logger.warning(f"Failed to load XGBoost model from {config.xgb_model_path}")
                        
                if hasattr(config, 'lstm_model_path') and config.lstm_model_path:
                    success = self.ml_module.load_lstm(config.lstm_model_path)
                    if success:
                        self.logger.info(f"LSTM model loaded from {config.lstm_model_path}")
                    else:
                        self.logger.warning(f"Failed to load LSTM model from {config.lstm_model_path}")
                        
            except Exception as e:
                self.logger.error(f"Error initializing ML module: {str(e)}")
                self.logger.warning("Falling back to indicator-based strategy")
                
        # Initialize strategy with ML module
        self.strategy = ScalpingStrategy(config, model=self.ml_module, binance_client=self.client)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(config)
        
        # Trade tracking
        self.open_trades = {}  # symbol -> trade_details
        self.trade_history = []
        
        # Performance tracking
        self.initial_balance = getattr(config, 'sim_balance', 10000.0)
        self.current_balance = self.initial_balance
        self.equity_curve = []
        
        self.logger.info(f"Backtester initialized with {self.initial_balance} initial balance")
        
    def load_data(self, symbol: str, filepath: str) -> bool:
        """
        Load historical data for a symbol from CSV file
        
        Args:
            symbol: Trading symbol
            filepath: Path to CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load data from CSV
            data = pd.read_csv(filepath)
            
            # Standardize column names
            column_mapping = {
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'time': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }
            
            # Rename columns to standardized format
            data = data.rename(columns={col: std_col for col, std_col in column_mapping.items() 
                                       if col in data.columns})
            
            # Ensure we have all required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Parse timestamp if it's in string format
            if isinstance(data['timestamp'].iloc[0], str):
                try:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                except Exception as e:
                    self.logger.error(f"Error parsing timestamp: {str(e)}")
                    return False
            
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = pd.to_numeric(data[col])
                
            # Sort by timestamp
            data = data.sort_values('timestamp')
            
            # Load into client
            self.client.load_data(symbol, data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False
            
    async def process_bar(self, symbol: str) -> Dict[str, Any]:
        """
        Process a single bar of data
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with processing results
        """
        # Get current bar data
        data = self.client.get_current_data(symbol)
        if data is None:
            return {'success': False, 'reason': 'No data available'}
            
        # Calculate indicators
        indicators = self.client.calculate_indicators(symbol)
        
        # Generate trading signal
        signal = await self.strategy.generate_signal(symbol, indicators)
        
        # Process signal
        result = {'success': True, 'signal': signal, 'trades': [], 'time': data.get('timestamp', None)}
        
        if signal['type'] and signal['type'] != 'NEUTRAL':
            # Check for open trades for this symbol
            if symbol in self.open_trades:
                # Check if should close
                current_price = data['close']
                
                # Get trade details
                trade = self.open_trades[symbol]
                
                # Check if stop loss or take profit hit
                should_close, reason = self.strategy.should_close_position(
                    symbol, trade, current_price, indicators
                )
                
                if should_close:
                    # Close trade
                    trade_close = self._close_trade(symbol, current_price, reason)
                    result['trades'].append(trade_close)
            else:
                # Check if can open new trade
                can_open = self.strategy.can_open_position(
                    symbol, 
                    data['close'] * 0.1,  # Estimate position value for check
                    self.current_balance
                )
                
                if can_open:
                    # Open new trade
                    direction = signal['direction']
                    
                    # Calculate position size based on risk
                    quantity = self.strategy.calculate_position_size(
                        symbol=symbol,
                        account_balance=self.current_balance,
                        entry_price=data['close'],
                        stop_loss=data['close'] * 0.99 if direction == 'BUY' else data['close'] * 1.01,
                        direction=direction,
                        volatility=indicators.get('volatility_14', 0.01)
                    )
                    
                    # Execute trade with slippage and commission
                    trade = self._open_trade(
                        symbol=symbol,
                        side=direction,
                        quantity=quantity,
                        price=data['close'],
                        stop_loss=data['close'] * 0.99 if direction == 'BUY' else data['close'] * 1.01,
                        take_profit=data['close'] * 1.01 if direction == 'BUY' else data['close'] * 0.99,
                        confidence=signal.get('confidence', 0),
                        model_used=signal.get('models_used', [])
                    )
                    
                    result['trades'].append(trade)
        
        # Update equity curve
        self._update_equity_curve(data.get('timestamp', None))
        
        # Add current balance to result
        result['balance'] = self.current_balance
        result['trade_count'] = len(self.trade_history)
        
        return result
    
    def _open_trade(self, symbol: str, side: str, quantity: float, price: float, 
                   stop_loss: float, take_profit: float, confidence: float = 0,
                   model_used: List[str] = None) -> Dict[str, Any]:
        """
        Open a new trade
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Trade quantity
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: Signal confidence (0-100)
            model_used: List of models used for signal
            
        Returns:
            Dict with trade details
        """
        # Execute trade with slippage and commission
        execution = self.client.execute_trade(symbol, side, quantity, price)
        
        if not execution['success']:
            self.logger.error(f"Failed to execute trade: {execution.get('error', 'Unknown error')}")
            return execution
            
        # Calculate effective quantity after commission (for spot trading, commission is deducted from target asset)
        effective_quantity = quantity
        
        # Create trade object
        trade = {
            'symbol': symbol,
            'side': side,
            'quantity': effective_quantity,
            'entry_price': execution['execution_price'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'time_opened': execution.get('timestamp', datetime.now()),
            'commission': execution['commission'],
            'slippage': price - execution['execution_price'],
            'confidence': confidence,
            'model_used': model_used or [],
            'status': 'open'
        }
        
        # Register trade
        self.open_trades[symbol] = trade
        
        # Update strategy
        self.strategy.register_position(symbol, trade)
        
        # Update balance accounting for trade value and commission
        trade_value = execution['execution_price'] * quantity
        commission = execution['commission']
        
        if side == 'BUY':
            # For BUY, we spend the trade value + commission
            self.current_balance -= (trade_value + commission)
            self.logger.debug(f"BUY position entry: -{trade_value:.2f} (value) -{commission:.2f} (commission) = -{trade_value + commission:.2f}")
        else:
            # For SELL, we receive the trade value - commission
            self.current_balance += (trade_value - commission)
            self.logger.debug(f"SELL position entry: +{trade_value:.2f} (value) -{commission:.2f} (commission) = +{trade_value - commission:.2f}")
            
        self.logger.info(
            f"Opened {side} trade for {symbol}: {quantity} @ {execution['execution_price']} "
            f"(slippage: {execution['execution_price'] - price:.2f}, "
            f"commission: {execution['commission']:.4f})"
        )
        
        return trade
        
    def _close_trade(self, symbol: str, current_price: float, reason: str) -> Dict[str, Any]:
        """
        Close an open trade
        
        Args:
            symbol: Trading symbol
            current_price: Current price for closing
            reason: Reason for closing (take_profit, stop_loss, etc.)
            
        Returns:
            Dict with trade details
        """
        if symbol not in self.open_trades:
            return {'success': False, 'error': 'No open trade for symbol'}
            
        # Get trade details
        trade = self.open_trades[symbol]
        side = trade['side']
        
        # Execute closing trade (opposite side)
        close_side = 'SELL' if side == 'BUY' else 'BUY'
        execution = self.client.execute_trade(symbol, close_side, trade['quantity'], current_price)
        
        if not execution['success']:
            self.logger.error(f"Failed to close trade: {execution.get('error', 'Unknown error')}")
            return execution
            
        # Calculate trade value at exit (with execution price)
        exit_value = execution['execution_price'] * trade['quantity']
        
        # Calculate original commission already paid at entry
        entry_commission = trade.get('commission', 0)
        
        # Calculate P/L based on position type
        if side == 'BUY':
            # Long position: (exit_value - exit_commission) - (entry_value + entry_commission)
            entry_value = trade['entry_price'] * trade['quantity']
            profit_loss = (exit_value - execution['commission']) - (entry_value + entry_commission)
        else:
            # Short position: (entry_value - entry_commission) - (exit_value + exit_commission)
            entry_value = trade['entry_price'] * trade['quantity']
            profit_loss = (entry_value - entry_commission) - (exit_value + execution['commission'])
        
        # Update trade with exit details
        trade.update({
            'exit_price': execution['execution_price'],
            'time_closed': execution.get('timestamp', datetime.now()),
            'profit_loss': profit_loss,
            'exit_commission': execution['commission'],
            'exit_slippage': current_price - execution['execution_price'],
            'exit_reason': reason,
            'status': 'closed',
            'success': True
        })
        
        # Add to trade history
        self.trade_history.append(trade)
        
        # Update strategy
        self.strategy.close_position(symbol, profit_loss)
        
        # Remove from open trades
        del self.open_trades[symbol]
        
        # For clarity, let's calculate the final balance update separately from P/L
        exit_value = execution['execution_price'] * trade['quantity']
        exit_commission = execution['commission']
        
        if side == 'BUY':
            # When closing a long position (BUY), we receive the exit value minus commission
            # Note: The entry cost was already deducted when opening the position
            self.current_balance += (exit_value - exit_commission)
            self.logger.debug(f"Closing BUY position: +{exit_value:.2f} (exit value) -{exit_commission:.2f} (exit commission) = P/L: {profit_loss:.2f}")
        else:
            # When closing a short position (SELL), we pay the exit value plus commission
            # Note: The entry proceeds were already added when opening the position
            self.current_balance -= (exit_value + exit_commission)
            self.logger.debug(f"Closing SELL position: -{exit_value:.2f} (exit value) -{exit_commission:.2f} (exit commission) = P/L: {profit_loss:.2f}")
        
        self.logger.info(
            f"Closed {side} trade for {symbol}: {trade['quantity']} @ {execution['execution_price']} "
            f"({reason}, P/L: {profit_loss:.4f}, "
            f"slippage: {execution['execution_price'] - current_price:.4f})"
        )
        
        return trade
        
    def _update_equity_curve(self, timestamp=None):
        """Update equity curve with current balance"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Calculate unrealized P/L of open positions
        unrealized_pl = 0
        for symbol, trade in self.open_trades.items():
            # Get current price
            current_price = self.client.get_current_price(symbol)
            
            # Calculate unrealized P/L
            if trade['side'] == 'BUY':
                # Long position: current_price - entry_price
                unrealized_pl += (current_price - trade['entry_price']) * trade['quantity']
            else:
                # Short position: entry_price - current_price
                unrealized_pl += (trade['entry_price'] - current_price) * trade['quantity']
                
        # Calculate total equity
        total_equity = self.current_balance + unrealized_pl
        
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.current_balance,
            'unrealized_pl': unrealized_pl,
            'total_equity': total_equity
        })
        
    async def run(self, symbols: List[str], max_bars: int = None, show_progress: bool = True):
        """
        Run backtest on historical data
        
        Args:
            symbols: List of trading symbols to backtest
            max_bars: Maximum number of bars to process (None for all)
            show_progress: Whether to show progress bar
            
        Returns:
            Dict with backtest results
        """
        import tqdm
        
        # Initialize results
        results = {
            'trades': [],
            'equity_curve': [],
            'performance_metrics': {},
            'symbols': symbols,
            'start_time': None,
            'end_time': None,
            'initial_balance': self.initial_balance,
            'final_balance': 0,
            'total_bars': 0
        }
        
        # Get min start time and max end time across all symbols
        start_times = []
        end_times = []
        total_bars = 0
        
        for symbol in symbols:
            if symbol not in self.client.data:
                self.logger.error(f"No data loaded for {symbol}")
                continue
                
            data = self.client.data[symbol]
            if 'timestamp' in data.columns:
                start_times.append(data['timestamp'].min())
                end_times.append(data['timestamp'].max())
                
            total_bars += len(data)
            
        if not start_times or not end_times:
            self.logger.error("No valid data found for any symbol")
            return results
            
        results['start_time'] = min(start_times)
        results['end_time'] = max(end_times)
        results['total_bars'] = total_bars
        
        # Initialize progress bar
        if show_progress:
            pbar = tqdm.tqdm(total=total_bars if max_bars is None else min(max_bars, total_bars))
            
        # Process each bar
        bar_count = 0
        while True:
            # Check if reached max bars
            if max_bars is not None and bar_count >= max_bars:
                break
                
            # Process each symbol
            any_data = False
            
            for symbol in symbols:
                # Check if data available
                data = self.client.get_current_data(symbol)
                if data is None:
                    continue
                    
                any_data = True
                
                # Process bar
                result = await self.process_bar(symbol)
                
                # Add trades to results
                if 'trades' in result and result['trades']:
                    results['trades'].extend(result['trades'])
                    
                # Advance to next bar
                self.client.advance(symbol)
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
                    
            # Increment bar count
            bar_count += 1
            
            # Break if no more data
            if not any_data:
                break
                
        # Close progress bar
        if show_progress:
            pbar.close()
            
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        results['performance_metrics'] = metrics
        
        # Add final balance and equity curve
        results['final_balance'] = self.current_balance
        results['equity_curve'] = self.equity_curve
        
        # Log summary
        self.logger.info(f"Backtest completed: {len(self.trade_history)} trades")
        self.logger.info(f"Initial balance: {self.initial_balance}, Final balance: {self.current_balance}")
        self.logger.info(f"Net profit: {self.current_balance - self.initial_balance}")
        self.logger.info(f"Win rate: {metrics['win_rate']*100:.2f}%")
        
        return results
        
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from trade history"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_profit': 0,
                'average_loss': 0,
                'largest_profit': 0,
                'largest_loss': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0,
                'total_commission': 0,
                'total_slippage': 0
            }
            
        # Basic metrics
        total_trades = len(self.trade_history)
        win_trades = [t for t in self.trade_history if t.get('profit_loss', 0) > 0]
        loss_trades = [t for t in self.trade_history if t.get('profit_loss', 0) <= 0]
        
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        # Ensure profit_loss already includes commission costs
        total_profit = sum(t.get('profit_loss', 0) for t in win_trades)
        total_loss = sum(abs(t.get('profit_loss', 0)) for t in loss_trades)
        
        # Calculate total commission costs for better reporting
        total_commission = sum(t.get('commission', 0) for t in self.trade_history)
        total_commission += sum(t.get('exit_commission', 0) for t in self.trade_history)
        
        # Pure profit factor (without commissions)
        raw_profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        # Adjusted profit factor accounting for commissions
        # This is more accurate for real-world performance assessment
        adjusted_profit = total_profit - (total_commission * (win_count / total_trades if total_trades > 0 else 0))
        adjusted_loss = total_loss + (total_commission * (loss_count / total_trades if total_trades > 0 else 0))
        adjusted_profit_factor = adjusted_profit / adjusted_loss if adjusted_loss > 0 else float('inf') if adjusted_profit > 0 else 0
        
        # Use adjusted profit factor as the main metric
        profit_factor = adjusted_profit_factor
        
        average_profit = total_profit / win_count if win_count > 0 else 0
        average_loss = total_loss / loss_count if loss_count > 0 else 0
        
        largest_profit = max((t.get('profit_loss', 0) for t in win_trades), default=0)
        largest_loss = max((abs(t.get('profit_loss', 0)) for t in loss_trades), default=0)
        
        # Drawdown calculation
        equity_values = [entry['total_equity'] for entry in self.equity_curve]
        
        if equity_values:
            # Calculate drawdown
            peak = equity_values[0]
            drawdown = 0
            max_drawdown = 0
            max_drawdown_pct = 0
            
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                else:
                    dd = peak - equity
                    dd_pct = dd / peak if peak > 0 else 0
                    
                    if dd > max_drawdown:
                        max_drawdown = dd
                        max_drawdown_pct = dd_pct
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
            
        # Commission and slippage
        total_commission = sum(t.get('commission', 0) for t in self.trade_history)
        total_commission += sum(t.get('exit_commission', 0) for t in self.trade_history)
        
        total_slippage = sum(abs(t.get('slippage', 0)) for t in self.trade_history)
        total_slippage += sum(abs(t.get('exit_slippage', 0)) for t in self.trade_history)
        
        # Calculate Sharpe ratio if we have enough data
        if len(equity_values) > 1:
            # Calculate daily returns
            daily_returns = []
            
            # Group equity curve by day
            equity_df = pd.DataFrame(self.equity_curve)
            if 'timestamp' in equity_df.columns:
                equity_df['date'] = pd.to_datetime(equity_df['timestamp']).dt.date
                daily_equity = equity_df.groupby('date')['total_equity'].last()
                
                # Calculate daily returns
                daily_returns = daily_equity.pct_change().dropna()
                
                # Calculate Sharpe ratio (assuming 252 trading days per year)
                risk_free_rate = 0.0  # Simplified
                sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / (daily_returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
            
        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'raw_profit_factor': raw_profit_factor,  # Unadjusted profit factor (for comparison)
            'average_profit': average_profit,
            'average_loss': average_loss,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'net_profit': self.current_balance - self.initial_balance,
            'net_profit_pct': (self.current_balance / self.initial_balance - 1) * 100,
            'commission_impact': total_commission / self.initial_balance * 100,
            'slippage_impact': total_slippage / self.initial_balance * 100,
            'gross_profit': total_profit,
            'gross_loss': total_loss,
            'adjusted_profit': adjusted_profit,
            'adjusted_loss': adjusted_loss
        }
        
    def plot_equity_curve(self, figsize=(10, 6), save_path=None):
        """
        Plot equity curve
        
        Args:
            figsize: Figure size (width, height)
            save_path: Path to save plot (None for display only)
        """
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        
        if 'timestamp' not in equity_df.columns or len(equity_df) == 0:
            self.logger.error("No valid equity curve data to plot")
            return
            
        # Set timestamp as index
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df = equity_df.set_index('timestamp')
        
        # Plot
        plt.figure(figsize=figsize)
        
        plt.plot(equity_df.index, equity_df['total_equity'], label='Total Equity')
        plt.plot(equity_df.index, equity_df['balance'], label='Balance', alpha=0.7)
        
        # Plot initial balance as horizontal line
        plt.axhline(self.initial_balance, linestyle='--', color='gray', alpha=0.5, 
                   label=f'Initial Balance: {self.initial_balance}')
        
        # Add trades markers
        for trade in self.trade_history:
            entry_time = pd.to_datetime(trade['time_opened'])
            exit_time = pd.to_datetime(trade['time_closed'])
            
            # Find closest equity curve point
            entry_idx = (equity_df.index - entry_time).abs().argmin()
            exit_idx = (equity_df.index - exit_time).abs().argmin()
            
            entry_equity = equity_df['total_equity'].iloc[entry_idx]
            exit_equity = equity_df['total_equity'].iloc[exit_idx]
            
            # Add markers for entry and exit
            color = 'green' if trade.get('profit_loss', 0) > 0 else 'red'
            plt.scatter(entry_time, entry_equity, color=color, marker='^', alpha=0.7)
            plt.scatter(exit_time, exit_equity, color=color, marker='v', alpha=0.7)
        
        # Add labels and title
        plt.title('Backtest Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis to show currency
        plt.gca().yaxis.set_major_formatter('${x:,.2f}')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
            
    def save_results(self, filepath: str):
        """
        Save backtest results to JSON file
        
        Args:
            filepath: Path to save results
        """
        try:
            # Create results dictionary
            results = {
                'initial_balance': self.initial_balance,
                'final_balance': self.current_balance,
                'net_profit': self.current_balance - self.initial_balance,
                'net_profit_pct': (self.current_balance / self.initial_balance - 1) * 100,
                'trade_count': len(self.trade_history),
                'performance_metrics': self._calculate_performance_metrics(),
                'trades': [
                    {k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                     for k, v in trade.items()}
                    for trade in self.trade_history
                ],
                'equity_curve': [
                    {k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                     for k, v in entry.items()}
                    for entry in self.equity_curve
                ],
                'configuration': {
                    'commission_rate': self.client.commission_rate,
                    'slippage_pct': self.client.slippage_pct,
                    'confidence_threshold': getattr(self.strategy, 'confidence_threshold', 0.7),
                    'max_daily_trades': getattr(self.strategy, 'max_daily_trades', 30),
                    'max_daily_loss_pct': getattr(self.strategy, 'max_daily_loss_pct', 3.0)
                },
                'timestamp': str(datetime.now())
            }
            
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
                
            self.logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

def main():
    """Command line interface for backtester"""
    parser = argparse.ArgumentParser(description='MidasScalpingv4 Backtester')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, help='Directory containing historical data')
    parser.add_argument('--symbols', type=str, default='BTCUSDT', help='Comma-separated list of symbols to test')
    parser.add_argument('--timeframe', type=str, default='1m', help='Timeframe for testing')
    
    # Backtest parameters
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance for testing')
    parser.add_argument('--commission', type=float, default=0.0004, help='Commission rate (0.0004 = 0.04%)')
    parser.add_argument('--slippage', type=float, default=0.0002, help='Slippage percentage (0.0002 = 0.02%)')
    
    # Strategy parameters
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Signal confidence threshold (0-1)')
    parser.add_argument('--max-daily-trades', type=int, default=30, help='Maximum trades per day')
    parser.add_argument('--max-daily-loss', type=float, default=3.0, help='Maximum daily loss percentage')
    
    # ML parameters
    parser.add_argument('--use-ml', action='store_true', help='Use ML models for prediction')
    parser.add_argument('--xgb-model', type=str, help='Path to XGBoost model')
    parser.add_argument('--lstm-model', type=str, help='Path to LSTM model')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for ML if available')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='backtest_results', help='Directory for results')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve')
    
    # Execution parameters
    parser.add_argument('--max-bars', type=int, help='Maximum bars to process')
    parser.add_argument('--no-progress', action='store_true', help='Hide progress bar')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Create configuration
    config_dict = {
        'symbols': symbols,
        'timeframe': args.timeframe,
        'sim_balance': args.initial_balance,
        'commission_rate': args.commission,
        'slippage_pct': args.slippage,
        'confidence_threshold': args.confidence_threshold,
        'max_daily_trades': args.max_daily_trades,
        'max_daily_loss_pct': args.max_daily_loss,
        'use_ml': args.use_ml,
        'use_gpu': args.use_gpu
    }
    
    if args.use_ml:
        if args.xgb_model:
            config_dict['xgb_model_path'] = args.xgb_model
        if args.lstm_model:
            config_dict['lstm_model_path'] = args.lstm_model
    
    # Create config object
    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    # Create backtester
    backtester = Backtester(config)
    
    # Load data for each symbol
    data_loaded = False
    for symbol in symbols:
        # Check for data file
        file_pattern = f"{symbol}_{args.timeframe}.csv"
        
        # Check in data directory if provided
        if args.data_dir:
            data_path = os.path.join(args.data_dir, file_pattern)
            if os.path.exists(data_path):
                success = backtester.load_data(symbol, data_path)
                if success:
                    data_loaded = True
                    continue
                    
        # Check in current directory
        if os.path.exists(file_pattern):
            success = backtester.load_data(symbol, file_pattern)
            if success:
                data_loaded = True
                continue
                
        # Check common data directories
        common_dirs = ['data', 'historical_data', 'history']
        for dir_name in common_dirs:
            data_path = os.path.join(dir_name, file_pattern)
            if os.path.exists(data_path):
                success = backtester.load_data(symbol, data_path)
                if success:
                    data_loaded = True
                    break
                    
        if not data_loaded:
            logger.error(f"No data file found for {symbol}. Looked for {file_pattern} in data directory and common locations.")
            return
    
    # Run backtest
    import asyncio
    results = asyncio.run(backtester.run(
        symbols=symbols,
        max_bars=args.max_bars,
        show_progress=not args.no_progress
    ))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"backtest_{symbols[0]}_{args.timeframe}_{timestamp}"
    
    # Save results
    results_path = os.path.join(args.output_dir, f"{base_filename}.json")
    backtester.save_results(results_path)
    
    # Plot equity curve if requested
    if args.plot:
        plot_path = os.path.join(args.output_dir, f"{base_filename}_equity.png")
        backtester.plot_equity_curve(save_path=plot_path)
    
    # Print summary
    metrics = backtester._calculate_performance_metrics()
    print("\nBacktest Results Summary:")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {results['start_time']} to {results['end_time']}")
    print(f"Initial Balance: ${args.initial_balance:.2f}")
    print(f"Final Balance: ${backtester.current_balance:.2f}")
    print(f"Net Profit: ${backtester.current_balance - args.initial_balance:.2f} ({(backtester.current_balance / args.initial_balance - 1) * 100:.2f}%)")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print("\nProfitability Metrics:")
    print(f"Raw Profit Factor: {metrics['raw_profit_factor']:.2f} (without commissions)")
    print(f"Adjusted Profit Factor: {metrics['profit_factor']:.2f} (including commissions)")
    print(f"Gross Profit: ${metrics['gross_profit']:.2f}")
    print(f"Gross Loss: ${metrics['gross_loss']:.2f}")
    print(f"Adjusted Profit: ${metrics['adjusted_profit']:.2f}")
    print(f"Adjusted Loss: ${metrics['adjusted_loss']:.2f}")
    print(f"Average Win: ${metrics['average_profit']:.2f}")
    print(f"Average Loss: ${metrics['average_loss']:.2f}")
    
    print("\nRisk Metrics:")
    print(f"Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']*100:.2f}%)")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print("\nTransaction Costs:")
    print(f"Total Commission: ${metrics['total_commission']:.2f} ({metrics['commission_impact']:.2f}% of initial capital)")
    print(f"Total Slippage: ${metrics['total_slippage']:.2f} ({metrics['slippage_impact']:.2f}% of initial capital)")
    print(f"Transaction Cost Impact: ${metrics['total_commission'] + metrics['total_slippage']:.2f} " +
          f"({(metrics['commission_impact'] + metrics['slippage_impact']):.2f}% of initial capital)")
    
    # Calculate symbol performance
    if hasattr(backtester, 'trade_history') and backtester.trade_history:
        symbol_performance = {}
        for trade in backtester.trade_history:
            symbol = trade['symbol']
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {
                    'trades': 0, 
                    'wins': 0,
                    'losses': 0,
                    'profit_loss': 0,
                    'commissions': 0
                }
            
            symbol_performance[symbol]['trades'] += 1
            if trade.get('profit_loss', 0) > 0:
                symbol_performance[symbol]['wins'] += 1
            else:
                symbol_performance[symbol]['losses'] += 1
                
            symbol_performance[symbol]['profit_loss'] += trade.get('profit_loss', 0)
            symbol_performance[symbol]['commissions'] += trade.get('commission', 0) + trade.get('exit_commission', 0)
        
        print("\nPerformance by Symbol:")
        for symbol, perf in symbol_performance.items():
            win_rate = perf['wins']/perf['trades']*100 if perf['trades'] > 0 else 0
            print(f"{symbol}: {perf['profit_loss']:.2f} USD | " +
                  f"{perf['trades']} trades | " +
                  f"Win rate: {win_rate:.1f}% | " +
                  f"Commissions: {perf['commissions']:.2f} USD")
    
    print(f"\nResults saved to: {results_path}")
    if args.plot:
        print(f"Equity curve plot saved to: {plot_path}")

if __name__ == "__main__":
    main()