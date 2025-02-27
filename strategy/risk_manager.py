import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd

@dataclass
class PositionSizing:
    """Position sizing details"""
    position_size: float
    max_risk_amount: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: Optional[float] = None

class RiskManager:
    """
    Risk management for trading strategy
    
    Manages position sizing, exposure limits, and stop losses
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('RiskManager')
        
        # Risk parameters
        self.max_risk_per_trade = config.max_risk_per_trade
        self.max_open_positions = config.max_open_trades
        self.max_exposure_pct = config.max_exposure_pct
        self.max_exposure_per_symbol_pct = config.max_exposure_per_symbol_pct
        
        # Stop loss parameters
        self.base_stop_loss_pct = config.base_stop_loss_pct
        self.max_stop_loss_pct = config.max_stop_loss_pct
        self.trailing_stop_pct = config.trailing_stop_pct
        
        # State tracking
        self.open_positions = {}  # symbol -> position_details
        self.current_exposure = 0.0
        self.symbol_exposures = {}  # symbol -> exposure_amount
    
    def calculate_position_size(self, 
                               total_capital: float,
                               symbol: str, 
                               entry_price: float, 
                               stop_loss_price: float) -> PositionSizing:
        """
        Calculate position size based on risk parameters
        
        Args:
            total_capital: Total capital in account
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            
        Returns:
            PositionSizing object with details
        """
        # Calculate risk per unit (difference between entry and stop loss)
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            self.logger.warning(f"Stop loss must be different from entry price for {symbol}")
            risk_per_unit = entry_price * 0.001  # Default to 0.1% if invalid
        
        # Maximum risk amount for this trade (percentage of capital)
        max_risk_amount = total_capital * (self.max_risk_per_trade / 100)
        
        # Calculate position size based on risk per unit
        position_size = max_risk_amount / risk_per_unit
        
        # Calculate take profit price (default: risk:reward = 1:2)
        take_profit_price = None
        risk_reward_ratio = getattr(self.config, 'risk_reward_ratio', 2.0)
        
        if entry_price > stop_loss_price:  # Long position
            risk = entry_price - stop_loss_price
            take_profit_price = entry_price + (risk * risk_reward_ratio)
        else:  # Short position
            risk = stop_loss_price - entry_price
            take_profit_price = entry_price - (risk * risk_reward_ratio)
        
        # Adjust for maximum exposure constraints
        max_position_by_exposure = (total_capital * self.max_exposure_per_symbol_pct / 100) / entry_price
        position_size = min(position_size, max_position_by_exposure)
        
        # Make sure position size is not zero
        if position_size <= 0:
            self.logger.warning(f"Calculated position size <= 0 for {symbol}, using minimum")
            position_size = 0.001  # Use a small default value
        
        # Log the calculation
        self.logger.info(
            f"Position size for {symbol}: {position_size:.4f} at {entry_price} "
            f"(risk: {max_risk_amount:.2f}, SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f})"
        )
        
        # Return the position sizing details
        return PositionSizing(
            position_size=position_size,
            max_risk_amount=max_risk_amount,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
    
    def can_open_position(self, symbol: str, position_value: float, total_capital: float) -> bool:
        """
        Check if a new position can be opened based on risk constraints
        
        Args:
            symbol: Trading symbol
            position_value: Value of the position in quote currency
            total_capital: Total capital in account
            
        Returns:
            True if position can be opened, False otherwise
        """
        # Check maximum number of open positions
        if len(self.open_positions) >= self.max_open_positions:
            self.logger.info(f"Maximum number of positions reached ({self.max_open_positions})")
            return False
        
        # Check if already have a position in this symbol
        if symbol in self.open_positions:
            self.logger.info(f"Already have an open position for {symbol}")
            return False
        
        # Check maximum total exposure
        max_total_exposure = total_capital * (self.max_exposure_pct / 100)
        
        if self.current_exposure + position_value > max_total_exposure:
            self.logger.info(
                f"Maximum total exposure reached: {self.current_exposure}/{max_total_exposure} "
                f"+ {position_value} would exceed limit"
            )
            return False
        
        # Check maximum exposure per symbol
        max_symbol_exposure = total_capital * (self.max_exposure_per_symbol_pct / 100)
        current_symbol_exposure = self.symbol_exposures.get(symbol, 0)
        
        if current_symbol_exposure + position_value > max_symbol_exposure:
            self.logger.info(
                f"Maximum symbol exposure reached for {symbol}: {current_symbol_exposure}/{max_symbol_exposure} "
                f"+ {position_value} would exceed limit"
            )
            return False
        
        return True
    
    def register_position(self, symbol: str, position_details: Dict[str, Any]) -> None:
        """
        Register a new open position
        
        Args:
            symbol: Trading symbol
            position_details: Dictionary with position details
        """
        # Remove any existing position for this symbol
        if symbol in self.open_positions:
            self.close_position(symbol)
        
        # Add the new position
        self.open_positions[symbol] = position_details
        
        # Update exposure tracking
        position_value = position_details['quantity'] * position_details['entry_price']
        self.current_exposure += position_value
        self.symbol_exposures[symbol] = self.symbol_exposures.get(symbol, 0) + position_value
        
        self.logger.info(
            f"Registered position for {symbol}: {position_details['quantity']} @ {position_details['entry_price']} "
            f"(total exposure: {self.current_exposure:.2f})"
        )
    
    def close_position(self, symbol: str) -> None:
        """
        Close and remove a position from tracking
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.open_positions:
            position_details = self.open_positions[symbol]
            position_value = position_details['quantity'] * position_details['entry_price']
            
            # Update exposure tracking
            self.current_exposure -= position_value
            self.symbol_exposures[symbol] -= position_value
            
            if self.symbol_exposures[symbol] <= 0:
                del self.symbol_exposures[symbol]
                
            del self.open_positions[symbol]
            
            self.logger.info(
                f"Closed position for {symbol} "
                f"(remaining exposure: {self.current_exposure:.2f})"
            )
    
    def calculate_dynamic_stop_loss(self, 
                                   symbol: str, 
                                   entry_price: float, 
                                   direction: str, 
                                   volatility: float) -> float:
        """
        Calculate a dynamic stop loss based on market volatility
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            direction: Trade direction ('BUY' or 'SELL')
            volatility: Current market volatility
            
        Returns:
            Stop loss price
        """
        # Adjust stop loss based on current volatility
        # Higher volatility -> wider stops to avoid noise
        baseline_volatility = getattr(self.config, 'baseline_volatility', 0.01)
        volatility_factor = volatility / baseline_volatility
        
        # Scale stop loss percentage with volatility, but cap it
        adjusted_sl_pct = min(
            self.base_stop_loss_pct * max(1, volatility_factor),
            self.max_stop_loss_pct
        )
        
        # Calculate stop loss price based on direction
        if direction == 'BUY':
            stop_loss_price = entry_price * (1 - adjusted_sl_pct / 100)
        else:  # SELL
            stop_loss_price = entry_price * (1 + adjusted_sl_pct / 100)
        
        self.logger.info(
            f"Dynamic stop loss for {symbol} {direction}: {stop_loss_price:.2f} "
            f"({adjusted_sl_pct:.2f}% from entry, volatility factor: {volatility_factor:.2f})"
        )
        
        return stop_loss_price
    
    def update_trailing_stops(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Update trailing stops for all open positions
        
        Args:
            current_prices: Dictionary of current prices by symbol
            
        Returns:
            Dictionary of updated stop loss prices by symbol
        """
        updates = {}
        
        for symbol, position in self.open_positions.items():
            current_price = current_prices.get(symbol)
            if not current_price:
                continue
                
            direction = position['side']
            original_stop = position['stop_loss']
            trailing_pct = self.trailing_stop_pct / 100
            
            # For long positions
            if direction == 'BUY':
                # If price moved up enough to trail the stop
                if current_price > position['entry_price']:
                    # New stop is the greater of: original stop or (current price - trailing distance)
                    new_stop = max(
                        original_stop,
                        current_price * (1 - trailing_pct)
                    )
                    
                    # Only update if stop improved
                    if new_stop > position['stop_loss']:
                        position['stop_loss'] = new_stop
                        updates[symbol] = new_stop
                        self.logger.info(f"Updated trailing stop for {symbol} to {new_stop:.2f}")
            
            # For short positions
            elif direction == 'SELL':
                # If price moved down enough to trail the stop
                if current_price < position['entry_price']:
                    # New stop is the lesser of: original stop or (current price + trailing distance)
                    new_stop = min(
                        original_stop,
                        current_price * (1 + trailing_pct)
                    )
                    
                    # Only update if stop improved
                    if new_stop < position['stop_loss']:
                        position['stop_loss'] = new_stop
                        updates[symbol] = new_stop
                        self.logger.info(f"Updated trailing stop for {symbol} to {new_stop:.2f}")
        
        return updates
    
    def get_total_exposure(self) -> float:
        """Get total current exposure"""
        return self.current_exposure
    
    def get_symbol_exposure(self, symbol: str) -> float:
        """Get exposure for a specific symbol"""
        return self.symbol_exposures.get(symbol, 0)