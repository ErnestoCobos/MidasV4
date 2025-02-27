import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from exceptions import StrategyError

logger = logging.getLogger('Strategy')

class ScalpingStrategy:
    """
    Scalping strategy implementation for crypto trading
    
    This strategy:
    1. Uses RSI to identify overbought/oversold conditions
    2. Checks for price breakouts of Bollinger Bands
    3. Confirms with volume indicators
    4. Looks for short-term momentum
    """
    
    def __init__(self, config, model=None):
        """Initialize the strategy with configuration parameters"""
        self.config = config
        self.timeframe = config.timeframe
        self.rsi_oversold = config.rsi_oversold
        self.rsi_overbought = config.rsi_overbought
        self.model = model
        
        # Inicializar cliente de Vultr si está habilitado
        self.vultr_client = None
        if hasattr(config, 'vultr_api_key') and config.vultr_api_key and \
           hasattr(config, 'ai_optimization_enabled') and config.ai_optimization_enabled:
            try:
                from ai.vultr_client import VultrInferenceClient
                self.vultr_client = VultrInferenceClient(config.vultr_api_key)
                logger.info("Cliente de Vultr Inference inicializado correctamente")
            except Exception as e:
                logger.error(f"Error inicializando cliente de Vultr: {str(e)}")
    
    def analyze(self, indicators: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, float]]]:
        """
        Analyze market conditions and decide on trading action
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Tuple containing:
                - Boolean indicating if a trade signal is generated
                - String with signal direction ('BUY', 'SELL', or 'NONE')
                - Optional dictionary with trade parameters (entry, stop loss, take profit)
        """
        try:
            # Extract indicators
            sma_7 = indicators.get('sma_7')
            sma_25 = indicators.get('sma_25')
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            rsi = indicators.get('rsi')
            current_price = indicators.get('close', indicators.get('price'))
            volume = indicators.get('current_volume')
            volume_sma = indicators.get('volume_sma')
            
            if not all([sma_7, sma_25, bb_upper, bb_lower, rsi, current_price]):
                logger.warning("Missing required indicators for analysis")
                return False, "NONE", None
            
            # Initialize signal variables
            signal = False
            direction = "NONE"
            params = None
            
            # Strategy logic
            # 1. RSI Conditions
            rsi_buy_signal = rsi < self.rsi_oversold
            rsi_sell_signal = rsi > self.rsi_overbought
            
            # 2. Price vs Moving Averages
            price_above_sma7 = current_price > sma_7
            price_above_sma25 = current_price > sma_25
            
            # 3. Bollinger Band breakout/bounce
            price_at_bb_lower = current_price <= bb_lower * 1.005  # Within 0.5% of lower band
            price_at_bb_upper = current_price >= bb_upper * 0.995  # Within 0.5% of upper band
            
            # 4. Volume confirmation (optional)
            high_volume = volume and volume_sma and volume > volume_sma * 1.5
            
            # Generate BUY signal
            if (rsi_buy_signal or price_at_bb_lower) and price_above_sma25:
                # Strong buy when price is above SMA25 but RSI is oversold or price near lower BB
                signal = True
                direction = "BUY"
                # Calculate trade parameters
                stop_loss = current_price * (1 - (self.config.stop_loss_percent / 100))
                take_profit = current_price * (1 + (self.config.take_profit_percent / 100))
                params = {
                    'entry': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                logger.info(f"BUY Signal generated: RSI={rsi}, Price={current_price}")
            
            # Generate SELL signal
            elif (rsi_sell_signal or price_at_bb_upper) and not price_above_sma7:
                # Strong sell when price is below SMA7 but RSI is overbought or price near upper BB
                signal = True
                direction = "SELL"
                # Calculate trade parameters
                stop_loss = current_price * (1 + (self.config.stop_loss_percent / 100))
                take_profit = current_price * (1 - (self.config.take_profit_percent / 100))
                params = {
                    'entry': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                logger.info(f"SELL Signal generated: RSI={rsi}, Price={current_price}")
            
            return signal, direction, params
            
        except Exception as e:
            logger.error(f"Error in strategy analysis: {str(e)}")
            raise StrategyError(f"Strategy analysis failed: {str(e)}")
    
    def optimize_strategy_parameters(self, market_data, performance_history):
        """Optimizar parámetros de estrategia con Vultr Inference API"""
        if not self.vultr_client:
            logger.warning("Cliente de Vultr no inicializado, omitiendo optimización")
            return False
            
        try:
            # Extraer parámetros actuales
            current_parameters = {
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "stop_loss_percent": self.config.stop_loss_percent,
                "take_profit_percent": self.config.take_profit_percent,
                "max_capital_risk_percent": self.config.max_capital_risk_percent
            }
            
            # Solicitar análisis
            suggestions = self.vultr_client.analyze_market_conditions(
                market_data=market_data,
                performance_history=performance_history,
                current_parameters=current_parameters
            )
            
            # Verificar si hay error
            if "error" in suggestions:
                logger.error(f"Error en análisis de mercado: {suggestions['error']}")
                return False
                
            # Aplicar sugerencias
            param_changes = self._apply_parameter_suggestions(suggestions)
            
            # Registrar cambios
            changes_str = ", ".join([f"{k}: {v[0]} → {v[1]}" for k, v in param_changes.items()])
            logger.info(f"Parámetros optimizados: {changes_str}")
            logger.info(f"Análisis: {suggestions.get('analysis', 'No disponible')}")
            return True
            
        except Exception as e:
            logger.error(f"Error en optimización de estrategia: {str(e)}")
            return False
            
    def _apply_parameter_suggestions(self, suggestions):
        """Aplica sugerencias de parámetros con validación"""
        if "parameter_adjustments" not in suggestions:
            return {}
            
        param_adjustments = suggestions["parameter_adjustments"]
        changes_applied = {}
        
        # Validar y aplicar cambios para cada parámetro
        if "rsi_oversold" in param_adjustments:
            new_value = int(param_adjustments["rsi_oversold"])
            if 10 <= new_value <= 40:  # Rango válido
                old_value = self.rsi_oversold
                self.rsi_oversold = new_value
                changes_applied["rsi_oversold"] = (old_value, new_value)
                
        if "rsi_overbought" in param_adjustments:
            new_value = int(param_adjustments["rsi_overbought"])
            if 60 <= new_value <= 90:  # Rango válido
                old_value = self.rsi_overbought
                self.rsi_overbought = new_value
                changes_applied["rsi_overbought"] = (old_value, new_value)
        
        if "stop_loss_percent" in param_adjustments:
            new_value = float(param_adjustments["stop_loss_percent"])
            if 0.1 <= new_value <= 2.0:  # Rango válido
                old_value = self.config.stop_loss_percent
                self.config.stop_loss_percent = new_value
                changes_applied["stop_loss_percent"] = (old_value, new_value)
        
        if "take_profit_percent" in param_adjustments:
            new_value = float(param_adjustments["take_profit_percent"])
            if 0.2 <= new_value <= 5.0:  # Rango válido
                old_value = self.config.take_profit_percent
                self.config.take_profit_percent = new_value
                changes_applied["take_profit_percent"] = (old_value, new_value)
                
        if "max_capital_risk_percent" in param_adjustments:
            new_value = float(param_adjustments["max_capital_risk_percent"])
            if 0.5 <= new_value <= 5.0:  # Rango válido
                old_value = self.config.max_capital_risk_percent
                self.config.max_capital_risk_percent = new_value
                changes_applied["max_capital_risk_percent"] = (old_value, new_value)
        
        return changes_applied

    def calculate_position_size(self, 
                               symbol: str, 
                               account_balance: float,
                               entry_price: float, 
                               stop_loss: float) -> float:
        """
        Calculate the position size based on risk management rules
        
        Args:
            symbol: Trading pair symbol
            account_balance: Current account balance in quote currency
            entry_price: Planned entry price
            stop_loss: Planned stop loss price
            
        Returns:
            Recommended position size
        """
        try:
            # Calculate risk amount
            risk_percent = self.config.max_capital_risk_percent / 100
            risk_amount = account_balance * risk_percent
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            # Calculate position size
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit
            else:
                # Fallback to default size if stop loss is equal to entry (should never happen)
                position_size = self.config.base_order_size.get(symbol, 0.001)
            
            # Apply minimum/maximum constraints
            min_order_size = self.config.base_order_size.get(symbol, 0.001)
            max_order_size = min_order_size * 10  # Maximum 10x the base size
            
            position_size = max(min_order_size, min(position_size, max_order_size))
            
            logger.info(f"Calculated position size: {position_size} for {symbol} with risk {risk_percent*100}%")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            # Return the base order size as fallback
            return self.config.base_order_size.get(symbol, 0.001)