import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Any
import logging

logger = logging.getLogger("MarketRegimeDetector")

class MarketRegime(Enum):
    TRENDING_UP = 1
    TRENDING_DOWN = 2
    RANGING = 3
    VOLATILE = 4
    UNKNOWN = 0

class MarketRegimeDetector:
    """Detector de régimen de mercado para adaptación de estrategia"""
    
    def __init__(self, config):
        """Inicializar detector con configuración"""
        self.config = config
        self.lookback_period = getattr(config, 'regime_lookback_period', 20)
        self.volatility_threshold = getattr(config, 'volatility_threshold', 0.015)
        self.trend_strength_threshold = getattr(config, 'trend_strength_threshold', 0.7)
        logger.info(f"Market Regime Detector inicializado (lookback={self.lookback_period}, "
                    f"vol_threshold={self.volatility_threshold:.4f}, "
                    f"trend_threshold={self.trend_strength_threshold:.2f})")
        
    def detect_regime(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar régimen de mercado basado en el precio y los indicadores"""
        # Verificar datos suficientes
        if len(ohlcv_data) < self.lookback_period:
            logger.warning(f"Datos insuficientes para detectar régimen: {len(ohlcv_data)}/{self.lookback_period}")
            return {'regime': MarketRegime.UNKNOWN, 'confidence': 0}
            
        # Calcular volatilidad
        recent_data = ohlcv_data.tail(self.lookback_period)
        volatility = recent_data['close'].pct_change().std() * np.sqrt(self.lookback_period)
        
        # Calcular movimiento direccional
        price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
        
        # Calcular fuerza de tendencia (ADX o cálculo simple)
        trend_strength = 0
        if 'adx' in recent_data.columns:
            trend_strength = recent_data['adx'].iloc[-1] / 100
        else:
            # Cálculo simple de fuerza de tendencia
            up_days = sum(1 for x in recent_data['close'].pct_change() if x > 0)
            trend_strength = abs((up_days / self.lookback_period) - 0.5) * 2
        
        # Verificar si está en rango con bandas de Bollinger
        bb_width = None
        is_narrowing = False
        if all(x in recent_data.columns for x in ['bb_upper', 'bb_lower', 'bb_middle']):
            bb_width = (recent_data['bb_upper'] - recent_data['bb_lower']) / recent_data['bb_middle']
            is_narrowing = bb_width.iloc[-1] < bb_width.iloc[-5] if len(bb_width) >= 5 else False
        
        # Determinar régimen
        if volatility > self.volatility_threshold * 2:
            regime = MarketRegime.VOLATILE
            confidence = min(100, volatility / self.volatility_threshold * 50)
        elif trend_strength > self.trend_strength_threshold:
            regime = MarketRegime.TRENDING_UP if price_change > 0 else MarketRegime.TRENDING_DOWN
            confidence = min(100, trend_strength / self.trend_strength_threshold * 70)
        else:
            regime = MarketRegime.RANGING
            confidence = min(100, (1 - trend_strength / self.trend_strength_threshold) * 60)
            if is_narrowing:
                confidence += 20
                
        logger.debug(f"Régimen detectado: {regime.name} (confianza: {confidence:.2f}%, "
                    f"volatilidad: {volatility:.4f}, fuerza tendencia: {trend_strength:.2f})")
                
        return {
            'regime': regime, 
            'confidence': confidence,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'price_change': price_change,
            'bb_width': bb_width.iloc[-1] if bb_width is not None else None,
            'is_narrowing': is_narrowing
        }