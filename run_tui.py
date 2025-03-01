#!/usr/bin/env python3
"""
MidasScalpingv4 TUI Launcher
----------------------------
Este script inicia la interfaz de usuario de terminal para el Midas Scalping Bot.
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any

# Configurar logging - solo a archivo para evitar interferir con la TUI
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/tui.log"),
    ]
)
logger = logging.getLogger('TUI')

# Intenta importar Textual para verificar que está instalado
try:
    import textual
except ImportError:
    logger.error("Textual no está instalado. Instálalo con: uv pip install textual")
    print("Error: Textual no está instalado. Instálalo con: uv pip install textual")
    sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Midas Scalping Bot - TUI')
    
    parser.add_argument('--config', type=str, default=None,
                      help='Ruta al archivo de configuración JSON')
    
    parser.add_argument('--testnet', action='store_true',
                      help='Usar Binance testnet en lugar del exchange real')
    
    parser.add_argument('--symbols', type=str, default=None,
                      help='Lista de pares de trading separados por comas (ej. "BTCUSDT,ETHUSDT")')
    
    parser.add_argument('--simulate', action='store_true',
                      help='Ejecutar en modo simulación sin conectarse a la API del exchange')
                      
    parser.add_argument('--real-data', action='store_true',
                      help='Usar datos reales del mercado en modo simulación')
                      
    parser.add_argument('--sim-balance', type=str, default=None,
                      help='Balance inicial para simulación (formato: "USDT:10000,BTC:0.5,ETH:5")')
    
    # Argumentos relacionados con modelos
    parser.add_argument('--model', type=str, default=None,
                      help='Tipo de modelo a usar (xgboost, lstm, indicator)')
    
    return parser.parse_args()

def setup_bot(args: argparse.Namespace) -> Any:
    """Configurar e inicializar el bot con los argumentos proporcionados."""
    try:
        # Importar módulos necesarios
        from config import Config
        from bot import ScalpingBot
        
        # Cargar configuración
        if args.config:
            from core.config import load_config
            config = load_config(args.config)
            logger.info(f"Configuración cargada desde: {args.config}")
        else:
            config = Config.from_env()
            logger.info("Configuración cargada desde variables de entorno")
        
        # Sobreescribir con argumentos de línea de comandos
        if args.symbols:
            config.symbols = args.symbols.split(',')
            logger.info(f"Pares a monitorear: {', '.join(config.symbols)}")
        
        # Aplicar argumentos de modelo
        if args.model:
            config.model_type = args.model
            logger.info(f"Tipo de modelo: {config.model_type}")
            
        # Manejar modo simulación
        if args.simulate:
            config.api_key = "simulation_mode_key"
            config.api_secret = "simulation_mode_secret"
            config.simulation_mode = True
            
            # Configurar datos de mercado reales en modo simulación si se solicita
            if args.real_data:
                config.use_real_market_data = True
                logger.info("Usando datos reales del mercado en modo simulación")
            else:
                config.use_real_market_data = False
                logger.info("Usando datos simulados del mercado")
                
            # Configurar balance inicial personalizado si se proporciona
            if args.sim_balance:
                try:
                    sim_balance = {}
                    for balance_str in args.sim_balance.split(','):
                        asset, amount = balance_str.split(':')
                        sim_balance[asset.strip()] = float(amount.strip())
                    
                    config.sim_initial_balance = sim_balance
                    logger.info(f"Balance inicial simulado configurado: {sim_balance}")
                except Exception as e:
                    logger.warning(f"Error al configurar balance inicial simulado: {str(e)}")
        
        elif args.testnet:
            config.use_testnet = True
            logger.info("Usando Binance Testnet")
        
        # Inicializar bot
        bot = ScalpingBot(config)
        # Agregar referencia del bot al binance_client para sincronizar P/L
        if hasattr(bot, 'binance_client'):
            bot.binance_client.bot = bot
        logger.info("Bot inicializado correctamente")
        return bot
        
    except Exception as e:
        logger.error(f"Error al configurar el bot: {str(e)}")
        raise

def main():
    """Función principal."""
    print("Iniciando Midas Scalping Bot TUI...")
    
    try:
        # Parsear argumentos
        args = parse_arguments()
        
        # Agregar flag para debug
        debug_mode = "--debug" in sys.argv
        
        # Configurar bot
        bot = setup_bot(args)
        
        # Asegurarse de que el historial de operaciones se inicializa correctamente
        logger.info(f"Bot inicializado con {len(bot.trades_history)} operaciones en historial")
        logger.info(f"Bot inicializado con {len(bot.open_trades)} operaciones abiertas")
        
        # Si no hay operaciones en el historial, imprimir advertencia
        if len(bot.trades_history) == 0:
            logger.warning("No hay operaciones en el historial del bot")
        
        # Configurar el logging para Textual
        from textual.logging import TextualHandler
        
        # Crear directorio de logs si no existe
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        # Configurar logging de Textual solo a archivo, pero no afectar el resto del sistema
        textual_log_path = os.path.join(logs_dir, 'textual.log')
        
        # Configurar TextualHandler específicamente para logs de Textual
        textual_logger = logging.getLogger("textual")
        textual_logger.handlers = []  # Limpiar handlers existentes
        
        # Añadir manejador de archivo para todos los logs de Textual
        textual_file_handler = logging.FileHandler(textual_log_path)
        textual_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        textual_logger.addHandler(textual_file_handler)
        textual_logger.propagate = False
        
        # NO eliminar los handlers de root logger, solo configurar para que escriban a archivo
        root_logger = logging.getLogger()
        
        # Remover StreamHandlers para evitar la salida a consola, mantener otros tipos
        for handler in list(root_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
        
        # Añadir un FileHandler al root logger
        root_file_handler = logging.FileHandler(os.path.join(logs_dir, 'app.log'))
        root_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(root_file_handler)
        
        # Importar e iniciar TUI
        from tui import TradingBotApp
        
        # Verificar que el historial de operaciones esté correcto
        if hasattr(bot, 'trades_history'):
            print(f"Bot tiene {len(bot.trades_history)} operaciones en historial")
            for i, trade in enumerate(bot.trades_history):
                print(f"Trade #{i}: {trade['symbol']} - P/L: {trade.get('profit_loss', 'N/A')}")
        
        # Iniciar la app TUI sin especificar log (no soportado en esta versión)
        app = TradingBotApp(bot)
        if debug_mode:
            app.run(debug=True)
        else:
            app.run()
        
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario")
        print("\nPrograma interrumpido por el usuario.")
    except Exception as e:
        logger.error(f"Error al iniciar TUI: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())