#!/usr/bin/env python3
"""
Script para probar la estrategia de trading basada en LLM con Vultr Inference API
"""

import asyncio
import pandas as pd
import logging
import json
from datetime import datetime

from config import Config
from strategy.llm_strategy import LLMScalpingStrategy
from binance_client import BinanceClient
from data.feature_engineer import FeatureEngineer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LLMStrategyTest')

async def main():
    """Función principal para probar la estrategia LLM"""
    try:
        # Cargar configuración desde .env
        config = Config.from_env()
        
        # Verificar que estemos usando el modelo LLM
        if config.model_type.lower() != 'llm':
            logger.warning("Cambiando MODEL_TYPE a 'llm' para este test")
            config.model_type = 'llm'
        
        # Inicializar componentes
        logger.info("Inicializando componentes...")
        llm_strategy = LLMScalpingStrategy(config)
        binance_client = BinanceClient(config)
        feature_engineer = FeatureEngineer(config)
        
        # Verificar si el cliente Vultr está disponible
        if llm_strategy.vultr_client is None:
            logger.error("Error: Cliente Vultr no disponible. Verifica tu VULTR_API_KEY")
            return
        
        # Lista de símbolos para analizar
        symbols = config.symbols
        logger.info(f"Analizando símbolos: {symbols}")
        
        for symbol in symbols:
            logger.info(f"Analizando {symbol}...")
            
            # Obtener datos OHLCV
            ohlcv_data = binance_client.get_ohlcv(
                symbol=symbol,
                interval=config.timeframe,
                limit=60
            )
            
            # Verificar datos
            if ohlcv_data is None or ohlcv_data.empty:
                logger.error(f"No se pudieron obtener datos para {symbol}")
                continue
                
            logger.info(f"Datos obtenidos: {len(ohlcv_data)} filas")
            
            # Generar señal de trading
            logger.info("Generando señal de trading con LLM...")
            signal = await llm_strategy.generate_signal(symbol, ohlcv_data)
            
            # Mostrar resultado
            logger.info(f"Señal: {signal['direction']} con confianza {signal.get('confidence', 0):.2f}%")
            if signal.get('analysis'):
                logger.info(f"Análisis: {signal['analysis']}")
            
            if signal['type'].name != 'NEUTRAL':
                logger.info(f"Precio entrada: {signal.get('entry_price')}")
                logger.info(f"Stop loss: {signal.get('stop_loss')}")
                logger.info(f"Take profit: {signal.get('take_profit')}")
                
            # Predecir movimiento de precio
            logger.info("Prediciendo movimiento de precio...")
            prediction = await llm_strategy.predict_price_movement(
                symbol=symbol,
                ohlcv_data=ohlcv_data,
                horizon='4h'
            )
            
            if 'error' not in prediction:
                logger.info(f"Predicción: {prediction.get('direction', 'unknown')} hacia {prediction.get('price_target')}")
                logger.info(f"Confianza: {prediction.get('confidence', 0):.2f}")
                logger.info(f"Análisis: {prediction.get('analysis', 'No análisis disponible')}")
            else:
                logger.error(f"Error en predicción: {prediction.get('error')}")
            
            # Simular un análisis de condiciones de mercado
            logger.info("Analizando condiciones de mercado...")
            
            # Generar datos de trades de ejemplo
            example_trades = [
                {
                    'symbol': symbol,
                    'entry_time': datetime.now(),
                    'exit_time': datetime.now(),
                    'entry_price': ohlcv_data['close'].iloc[-10],
                    'exit_price': ohlcv_data['close'].iloc[-1],
                    'profit_loss': 0.05 * ohlcv_data['close'].iloc[-10],
                    'side': 'BUY'
                },
                {
                    'symbol': symbol,
                    'entry_time': datetime.now(),
                    'exit_time': datetime.now(),
                    'entry_price': ohlcv_data['close'].iloc[-20],
                    'exit_price': ohlcv_data['close'].iloc[-15],
                    'profit_loss': -0.02 * ohlcv_data['close'].iloc[-20],
                    'side': 'BUY'
                }
            ]
            
            market_analysis = await llm_strategy.analyze_market_conditions(
                symbol=symbol,
                ohlcv_data=ohlcv_data,
                trades_history=example_trades
            )
            
            if 'error' not in market_analysis:
                logger.info(f"Análisis de mercado: {market_analysis.get('analysis', 'No análisis disponible')}")
                
                if 'parameter_adjustments' in market_analysis:
                    logger.info(f"Ajustes recomendados: {json.dumps(market_analysis['parameter_adjustments'])}")
                
                logger.info(f"Tendencia: {market_analysis.get('market_trend', 'unknown')}")
                logger.info(f"Ajuste de riesgo: {market_analysis.get('risk_adjustment', 'unknown')}")
                logger.info(f"Confianza: {market_analysis.get('confidence', 0):.2f}")
            else:
                logger.error(f"Error en análisis de mercado: {market_analysis.get('error')}")
            
            logger.info(f"Análisis de {symbol} completado\n" + "-"*40 + "\n")
            
        logger.info("Test de estrategia LLM completado")
    
    except Exception as e:
        logger.error(f"Error en el test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())