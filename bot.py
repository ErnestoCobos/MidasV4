import logging
import time
import asyncio
from typing import Dict, List, Optional, Any
import threading
import queue
import json
from datetime import datetime, timedelta

from binance.client import Client
import os
from config import Config
from exceptions import CredentialsError, APIError, StrategyError, OrderError
from strategy import ScalpingStrategy
from strategy.llm_strategy import LLMScalpingStrategy
from binance_client import BinanceClient

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

# Solo configurar logging si aún no ha sido configurado
if not logging.getLogger('ScalpingBot').handlers:
    file_handler = logging.FileHandler("logs/bot.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger('ScalpingBot')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False  # Evitar duplicación de logs
else:
    logger = logging.getLogger('ScalpingBot')

class ScalpingBot:
    """
    Main bot class for crypto scalping on Binance
    
    Responsibilities:
    1. Initialize and manage components (exchange client, strategy)
    2. Monitor markets and detect trading opportunities
    3. Execute trades based on signals
    4. Manage open positions and track performance
    """
    
    # Método auxiliar para TUI - añadir getter para precios anteriores
    def get_previous_price(self, symbol):
        """Obtener precio anterior para un símbolo (para calcular cambio porcentual)"""
        # En un caso real, tendríamos un historial de precios
        # Aquí simplemente devolvemos el precio actual reducido ligeramente
        if hasattr(self, 'real_time_prices') and symbol in self.real_time_prices:
            return self.real_time_prices[symbol] * 0.999
        return None
        
    # Método auxiliar para TUI - obtener balance actual
    def get_balance(self):
        """Obtener balance total en USDT"""
        total_balance = 0.0
        try:
            if hasattr(self, 'binance_client'):
                # Obtener P/L actual de operaciones abiertas
                open_trades_pl = sum(trade.get('profit_loss', 0) for trade in self.open_trades.values())
                
                # Para modo simulación, calcular balance basado en trades y balance inicial
                if hasattr(self.binance_client, 'simulation_mode') and self.binance_client.simulation_mode:
                    # Obtener balance de simulación
                    if hasattr(self, 'config') and hasattr(self.config, 'sim_initial_balance'):
                        # Obtener el balance USDT inicial configurado
                        usdt_balance = self.config.sim_initial_balance.get('USDT', 1000.0)
                        
                        # Calcular fondos reservados para posiciones abiertas
                        # Importante: en spot, estos fondos ya están restados del balance disponible
                        reserved_funds = 0.0
                        for trade in self.open_trades.values():
                            if trade['side'] == 'BUY':
                                # Para compras en spot, los fondos ya están utilizados
                                reserved_funds += trade['quantity'] * trade['entry_price']
                        
                        # Añadir las ganancias/pérdidas totales y restar fondos reservados
                        total_balance = usdt_balance + self.total_profit_loss - reserved_funds
                        
                        # Sumar P/L no realizado de operaciones abiertas
                        total_balance += open_trades_pl
                    else:
                        # Valor predeterminado
                        total_balance = 1000.0 + self.total_profit_loss + open_trades_pl
                else:
                    # En un exchange real, obtener balances de la API
                    balances = self.binance_client.get_account_balance()
                    quote_asset = 'USDT'  # Moneda base para calcular balance
                    
                    # Sumar todos los activos convertidos a USDT (simplificado)
                    total_balance = balances.get(quote_asset, 0.0) + open_trades_pl
            else:
                # Si no hay cliente binance, devolvemos un valor predeterminado
                total_balance = 0.0
                
            # Verificar que no sea negativo (en spot no es posible)
            if total_balance < 0:
                logger.warning(f"Balance calculado negativo ({total_balance}), ajustando a 0")
                total_balance = 0.0
                
        except Exception as e:
            logger.error(f"Error al calcular balance: {str(e)}")
            total_balance = 0.0
            
        return total_balance
        
    # Método auxiliar para obtener balances desglosados por activo
    def get_detailed_balances(self):
        """Obtener diccionario de balances por activo"""
        balances = {}
        try:
            if hasattr(self, 'binance_client'):
                if hasattr(self.binance_client, 'simulation_mode') and self.binance_client.simulation_mode:
                    # En modo simulación, usar balances simulados
                    if hasattr(self, 'config') and hasattr(self.config, 'sim_initial_balance'):
                        # Copiar balances iniciales
                        balances = dict(self.config.sim_initial_balance)
                        
                        # Actualizar balances basados en operaciones abiertas y cerradas
                        for symbol, trade in self.open_trades.items():
                            # Para compras en spot, ya hemos gastado USDT para comprar el activo
                            if trade['side'] == 'BUY':
                                base_asset = symbol[:3] if len(symbol) >= 3 else symbol  # e.g., 'BTC' from 'BTCUSDT'
                                quote_asset = symbol[3:] if len(symbol) >= 6 else 'USDT'  # e.g., 'USDT' from 'BTCUSDT'
                                
                                # Restar USDT utilizado (ya deberías haberlo hecho al abrir la posición)
                                cost = trade['quantity'] * trade['entry_price']
                                if quote_asset in balances:
                                    # Los fondos ya deberían haberse restado, pero nos aseguramos
                                    balances[quote_asset] = max(0, balances.get(quote_asset, 0) - cost)
                                
                                # Añadir cantidad del activo comprado
                                if base_asset in balances:
                                    balances[base_asset] = balances.get(base_asset, 0) + trade['quantity']
                                else:
                                    balances[base_asset] = trade['quantity']
                            
                            # Para ventas en spot, ya hemos vendido el activo y recibido USDT
                            elif trade['side'] == 'SELL':
                                base_asset = symbol[:3] if len(symbol) >= 3 else symbol
                                quote_asset = symbol[3:] if len(symbol) >= 6 else 'USDT'
                                
                                # El activo ya debería haberse restado, pero nos aseguramos
                                if base_asset in balances:
                                    balances[base_asset] = max(0, balances.get(base_asset, 0) - trade['quantity'])
                                
                                # El USDT recibido ya debería haberse añadido al abrir la posición
                                
                        # Ajustar USDT con ganancias/pérdidas
                        if 'USDT' in balances:
                            # Solo sumamos P/L de operaciones cerradas
                            balances['USDT'] += self.total_profit_loss
                            
                            # Para evitar saldos negativos en spot
                            balances['USDT'] = max(0, balances['USDT'])
                    else:
                        # Valores predeterminados
                        balances = {
                            'USDT': max(0, 1000.0 + self.total_profit_loss),
                            'BTC': 0.0,
                            'ETH': 0.0
                        }
                else:
                    # En un exchange real, llamamos a la API para obtener balances actualizados
                    balances = self.binance_client.get_account_balance()
            else:
                balances = {'USDT': 0.0}
                
            # Verificar que no haya balances negativos (no es posible en spot)
            for asset, amount in list(balances.items()):
                if amount < 0:
                    logger.warning(f"Balance negativo detectado para {asset}: {amount}, ajustando a 0")
                    balances[asset] = 0.0
                    
        except Exception as e:
            logger.error(f"Error al obtener balances detallados: {str(e)}")
            balances = {'USDT': 0.0, 'ERROR': str(e)}
            
        return balances
    
    def __init__(self, config: Config):
        """Initialize the bot with configuration"""
        self.config = config
        self.binance_client = BinanceClient(config)
        self.active = False
        self.real_time_prices = {}
        self.open_trades = {}
        self.trades_history = []
        
        # Initialize database
        try:
            from data.database import SQLiteManager
            self.db = SQLiteManager(config)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            self.db = None
            
        # Add demo trades if in simulation mode (for TUI demo)
        if hasattr(self.config, 'simulation_mode') and self.config.simulation_mode:
            self._add_demo_trades()
        
        # Initialize model if enabled
        self.model = None
        if hasattr(config, 'model_type') and config.model_type != 'indicator':
            try:
                from models.model_factory import ModelFactory
                
                # Check GPU availability and log results
                tf_gpu, xgb_gpu = ModelFactory.is_gpu_available()
                
                if config.use_gpu:
                    logger.info(f"GPU acceleration requested: TensorFlow GPU: {tf_gpu}, XGBoost GPU: {xgb_gpu}")
                    
                    # Warn if GPU requested but not available for the selected model type
                    if config.model_type == 'lstm' and not tf_gpu:
                        logger.warning("LSTM model selected but TensorFlow GPU not available. Performance will be limited.")
                    elif config.model_type == 'xgboost' and not xgb_gpu:
                        logger.warning("XGBoost model selected but XGBoost GPU not available. Performance will be limited.")
                
                # Load active model from database if available
                model_loaded = False
                if self.db:
                    active_model = self.db.get_active_model(
                        symbol=self.config.symbols[0],  # Por ahora usar primer símbolo
                        model_type=config.model_type
                    )
                    
                    if active_model and os.path.exists(active_model['file_path']):
                        try:
                            self.model = ModelFactory.create_model(config.model_type, config)
                            self.model.load(active_model['file_path'])
                            logger.info(f"Loaded active model from {active_model['file_path']}")
                            model_loaded = True
                        except Exception as e:
                            logger.error(f"Error loading active model: {str(e)}")
                
                # Create new model if no active model was loaded
                if not model_loaded:
                    self.model = ModelFactory.create_model(config.model_type, config)
                    if self.model:
                        logger.info(f"Successfully initialized new {config.model_type} model")
                        
            except Exception as e:
                logger.error(f"Error initializing model: {str(e)}")
                logger.warning("Falling back to indicator-based strategy")
                self.model = None
        
        # Initialize model trainer
        try:
            from models.model_trainer import ModelTrainer
            self.model_trainer = ModelTrainer(config, self.db, self.binance_client)
            logger.info("Model trainer initialized successfully")
            
            # Start scheduled training if enabled
            if hasattr(config, 'auto_train') and config.auto_train:
                interval_hours = getattr(config, 'training_interval_hours', 24)
                self.model_trainer.start_scheduled_training(interval_hours)
                logger.info(f"Automatic model training enabled with {interval_hours}h interval")
        except Exception as e:
            logger.error(f"Error initializing model trainer: {str(e)}")
            self.model_trainer = None
                
        # Initialize strategy
        if hasattr(config, 'model_type') and config.model_type.lower() == 'llm':
            logger.info("Initializing LLM-based strategy using Vultr Inference API")
            self.strategy = LLMScalpingStrategy(config)
        else:
            # Fallback to indicator-based strategy if LLM not selected
            logger.info("Initializing indicator-based strategy (fallback)")
            self.strategy = ScalpingStrategy(config, self.model)
        
        # State tracking
        self.active = False
        self.open_trades = {}  # symbol -> trade_details
        self.price_queue = queue.Queue()
        self.real_time_prices = {}  # symbol -> current_price
        
        # Performance tracking
        self.trades_history = []
        self.total_profit_loss = 0.0
        
        logger.info("Bot initialized successfully")
    
    def start(self):
        """Start the bot and its components"""
        if self.active:
            logger.warning("Bot is already running")
            return
        
        try:
            logger.info("Starting scalping bot...")
            self.active = True
            
            # Start price monitoring for all configured symbols
            for symbol in self.config.symbols:
                self._start_price_monitoring(symbol)
            
            # Start the main trading loop in a separate thread
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            # Iniciar optimización de IA si está habilitada
            if hasattr(self.config, 'ai_optimization_enabled') and self.config.ai_optimization_enabled:
                try:
                    self.schedule_ai_strategy_optimization(
                        interval_hours=self.config.ai_optimization_interval_hours
                    )
                    logger.info(f"Optimización IA programada cada {self.config.ai_optimization_interval_hours}h")
                except Exception as e:
                    logger.error(f"Error al programar optimización IA: {str(e)}")
            
            logger.info(f"Bot started successfully, monitoring {len(self.config.symbols)} symbols")
        
        except Exception as e:
            self.active = False
            logger.error(f"Failed to start bot: {str(e)}")
            raise
    
    def stop(self):
        """Stop the bot and close all connections"""
        if not self.active:
            logger.warning("Bot is not running")
            return
        
        try:
            logger.info("Stopping bot...")
            self.active = False
            
            # Stop price monitoring
            for symbol in self.config.symbols:
                if hasattr(self.binance_client, 'stop_symbol_ticker_socket'):
                    self.binance_client.stop_symbol_ticker_socket(symbol)
            
            # Wait for trading thread to finish
            if hasattr(self, 'trading_thread') and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5.0)
            
            logger.info("Bot stopped successfully")
        
        except Exception as e:
            logger.error(f"Error while stopping bot: {str(e)}")
            
    def _add_demo_trades(self):
        """Add demo trades for TUI visualization (simulation mode only)"""
        logger.info("Adding demo trades for TUI visualization")
        try:
            # Verificar que tenemos acceso a la base de datos
            if not hasattr(self, 'db') or self.db is None:
                logger.error("No database connection available for adding demo trades")
                return
                
            # Añadir operaciones abiertas de demostración
            open_trades = [
                {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "quantity": 0.01,
                    "entry_price": 68500.0,
                    "stop_loss": 67800.0,
                    "take_profit": 69800.0,
                    "entry_time": datetime.now() - timedelta(hours=2),
                    "order_id": "12345",
                    "strategy_type": "indicator",
                    "status": "open",
                    "is_simulation": 1
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "SELL",
                    "quantity": 0.5,
                    "entry_price": 3850.0,
                    "stop_loss": 3900.0,
                    "take_profit": 3700.0,
                    "entry_time": datetime.now() - timedelta(hours=1),
                    "order_id": "12346",
                    "strategy_type": "indicator",
                    "status": "open",
                    "is_simulation": 1
                }
            ]
            
            # Almacenar operaciones abiertas en la base de datos
            for trade in open_trades:
                trade_id = self.db.store_trade(trade)
                if trade_id:
                    logger.info(f"Added demo open trade {trade['symbol']} with ID {trade_id}")
                else:
                    logger.error(f"Failed to add demo open trade {trade['symbol']}")
            
            # Añadir historial de operaciones cerradas
            closed_trades = [
                {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "quantity": 0.02,
                    "entry_price": 65000.0,
                    "exit_price": 65500.0,
                    "stop_loss": 64500.0,
                    "take_profit": 66000.0,
                    "entry_time": datetime.now() - timedelta(days=1, hours=4),
                    "exit_time": datetime.now() - timedelta(days=1, hours=2),
                    "order_id": "12340",
                    "strategy_type": "indicator",
                    "profit_loss": 10.0,
                    "exit_reason": "take_profit",
                    "status": "closed",
                    "is_simulation": 1
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "SELL",
                    "quantity": 1.0,
                    "entry_price": 3900.0,
                    "exit_price": 3800.0,
                    "stop_loss": 4000.0,
                    "take_profit": 3750.0,
                    "entry_time": datetime.now() - timedelta(days=1, hours=6),
                    "exit_time": datetime.now() - timedelta(days=1, hours=4),
                    "order_id": "12341",
                    "strategy_type": "llm",
                    "profit_loss": 100.0,
                    "exit_reason": "take_profit",
                    "status": "closed",
                    "is_simulation": 1
                },
                {
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "quantity": 0.015,
                    "entry_price": 67000.0,
                    "exit_price": 66500.0,
                    "stop_loss": 67500.0,
                    "take_profit": 66000.0,
                    "entry_time": datetime.now() - timedelta(days=2, hours=2),
                    "exit_time": datetime.now() - timedelta(days=2),
                    "order_id": "12342",
                    "strategy_type": "indicator",
                    "profit_loss": 7.5,
                    "exit_reason": "take_profit",
                    "status": "closed",
                    "is_simulation": 1
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "BUY",
                    "quantity": 0.75,
                    "entry_price": 3700.0,
                    "exit_price": 3650.0,
                    "stop_loss": 3650.0,
                    "take_profit": 3800.0,
                    "entry_time": datetime.now() - timedelta(days=2, hours=6),
                    "exit_time": datetime.now() - timedelta(days=2, hours=4),
                    "order_id": "12343",
                    "strategy_type": "indicator",
                    "profit_loss": -37.5,
                    "exit_reason": "stop_loss",
                    "status": "closed",
                    "is_simulation": 1
                }
            ]
            
            # Almacenar operaciones cerradas en la base de datos
            for trade in closed_trades:
                trade_id = self.db.store_trade(trade)
                if trade_id:
                    logger.info(f"Added demo closed trade {trade['symbol']} with ID {trade_id}")
                else:
                    logger.error(f"Failed to add demo closed trade {trade['symbol']}")
            
            # Cargar operaciones desde la base de datos para mantener la compatibilidad con el código existente
            self._load_trades_from_database()
                
        except Exception as e:
            logger.error(f"Error adding demo trades: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _load_trades_from_database(self):
        """Cargar operaciones desde la base de datos a las estructuras en memoria"""
        logger.info("Cargando operaciones desde la base de datos")
        try:
            if not hasattr(self, 'db') or self.db is None:
                logger.error("No database connection available for loading trades")
                return
                
            # Determinar si estamos en modo simulación
            is_simulation = hasattr(self.config, 'simulation_mode') and self.config.simulation_mode
            
            # Cargar operaciones abiertas
            open_trades_list = self.db.get_open_trades(is_simulation=is_simulation)
            logger.info(f"Cargadas {len(open_trades_list)} operaciones abiertas desde la base de datos")
            
            # Convertir a diccionario para compatibilidad con el código existente
            self.open_trades = {}
            for trade in open_trades_list:
                # Crear clave única para el diccionario
                trade_key = f"{trade['symbol']}_{trade['id']}"
                
                # Convertir campos de la base de datos al formato esperado por el código existente
                formatted_trade = {
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'entry_price': trade['entry_price'],
                    'stop_loss': trade['stop_loss'],
                    'take_profit': trade['take_profit'],
                    'time_opened': datetime.fromisoformat(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time'],
                    'order_id': trade['order_id'],
                    'strategy_type': trade['strategy_type'],
                    'db_id': trade['id']
                }
                
                # Añadir campos opcionales si existen
                if 'profit_loss' in trade and trade['profit_loss'] is not None:
                    formatted_trade['profit_loss'] = trade['profit_loss']
                if 'confidence' in trade and trade['confidence'] is not None:
                    formatted_trade['confidence'] = trade['confidence']
                if 'model_used' in trade and trade['model_used'] is not None:
                    formatted_trade['model_used'] = trade['model_used']
                
                self.open_trades[trade_key] = formatted_trade
            
            # Cargar historial de operaciones cerradas
            closed_trades_list = self.db.get_trades_history(is_simulation=is_simulation)
            logger.info(f"Cargadas {len(closed_trades_list)} operaciones cerradas desde la base de datos")
            
            # Convertir a lista para compatibilidad con el código existente
            self.trades_history = []
            for trade in closed_trades_list:
                # Convertir campos de la base de datos al formato esperado por el código existente
                formatted_trade = {
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'stop_loss': trade['stop_loss'] if 'stop_loss' in trade else None,
                    'take_profit': trade['take_profit'] if 'take_profit' in trade else None,
                    'time_opened': datetime.fromisoformat(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time'],
                    'time_closed': datetime.fromisoformat(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time'],
                    'order_id': trade['order_id'] if 'order_id' in trade else None,
                    'strategy_type': trade['strategy_type'],
                    'profit_loss': trade['profit_loss'],
                    'reason': trade['exit_reason'] if 'exit_reason' in trade else 'unknown'
                }
                
                # Añadir campos opcionales si existen
                if 'confidence' in trade and trade['confidence'] is not None:
                    formatted_trade['confidence'] = trade['confidence']
                if 'model_used' in trade and trade['model_used'] is not None:
                    formatted_trade['model_used'] = trade['model_used']
                
                self.trades_history.append(formatted_trade)
            
            # Calcular P/L total
            self.total_profit_loss = sum(t.get('profit_loss', 0) for t in self.trades_history)
            logger.info(f"P/L total calculado: {self.total_profit_loss}")
            
        except Exception as e:
            logger.error(f"Error cargando operaciones desde la base de datos: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    def _load_trades_from_database(self):
        """Cargar operaciones desde la base de datos a las estructuras en memoria"""
        logger.info("Cargando operaciones desde la base de datos")
        try:
            if not hasattr(self, 'db') or self.db is None:
                logger.error("No database connection available for loading trades")
                return
                
            # Determinar si estamos en modo simulación
            is_simulation = hasattr(self.config, 'simulation_mode') and self.config.simulation_mode
            
            # Cargar operaciones abiertas
            open_trades_list = self.db.get_open_trades(is_simulation=is_simulation)
            logger.info(f"Cargadas {len(open_trades_list)} operaciones abiertas desde la base de datos")
            
            # Convertir a diccionario para compatibilidad con el código existente
            self.open_trades = {}
            for trade in open_trades_list:
                # Crear clave única para el diccionario
                trade_key = f"{trade['symbol']}_{trade['id']}"
                
                # Convertir campos de la base de datos al formato esperado por el código existente
                formatted_trade = {
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'entry_price': trade['entry_price'],
                    'stop_loss': trade['stop_loss'],
                    'take_profit': trade['take_profit'],
                    'time_opened': datetime.fromisoformat(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time'],
                    'order_id': trade['order_id'],
                    'strategy_type': trade['strategy_type'],
                    'db_id': trade['id']
                }
                
                # Añadir campos opcionales si existen
                if 'profit_loss' in trade and trade['profit_loss'] is not None:
                    formatted_trade['profit_loss'] = trade['profit_loss']
                if 'confidence' in trade and trade['confidence'] is not None:
                    formatted_trade['confidence'] = trade['confidence']
                if 'model_used' in trade and trade['model_used'] is not None:
                    formatted_trade['model_used'] = trade['model_used']
                
                self.open_trades[trade_key] = formatted_trade
            
            # Cargar historial de operaciones cerradas
            closed_trades_list = self.db.get_trades_history(is_simulation=is_simulation)
            logger.info(f"Cargadas {len(closed_trades_list)} operaciones cerradas desde la base de datos")
            
            # Convertir a lista para compatibilidad con el código existente
            self.trades_history = []
            for trade in closed_trades_list:
                # Convertir campos de la base de datos al formato esperado por el código existente
                formatted_trade = {
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'stop_loss': trade['stop_loss'] if 'stop_loss' in trade else None,
                    'take_profit': trade['take_profit'] if 'take_profit' in trade else None,
                    'time_opened': datetime.fromisoformat(trade['entry_time']) if isinstance(trade['entry_time'], str) else trade['entry_time'],
                    'time_closed': datetime.fromisoformat(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time'],
                    'order_id': trade['order_id'] if 'order_id' in trade else None,
                    'strategy_type': trade['strategy_type'],
                    'profit_loss': trade['profit_loss'],
                    'reason': trade['exit_reason'] if 'exit_reason' in trade else 'unknown'
                }
                
                # Añadir campos opcionales si existen
                if 'confidence' in trade and trade['confidence'] is not None:
                    formatted_trade['confidence'] = trade['confidence']
                if 'model_used' in trade and trade['model_used'] is not None:
                    formatted_trade['model_used'] = trade['model_used']
                
                self.trades_history.append(formatted_trade)
            
            # Calcular P/L total
            self.total_profit_loss = sum(t.get('profit_loss', 0) for t in self.trades_history)
            logger.info(f"P/L total calculado: {self.total_profit_loss}")
            
        except Exception as e:
            logger.error(f"Error cargando operaciones desde la base de datos: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_performance_summary(self):
        """Obtener resumen de rendimiento para la TUI"""
        # En un caso real, calcularíamos estadísticas basadas en el historial real
        total_trades = len(self.trades_history)
        profitable_trades = sum(1 for t in self.trades_history if t.get('profit_loss', 0) > 0)
        
        # Evitar división por cero
        win_rate = 0 if total_trades == 0 else profitable_trades / total_trades
        
        # Calcular P/L total
        total_pl = sum(t.get('profit_loss', 0) for t in self.trades_history)
        
        # Calcular promedio de ganancias y pérdidas
        profit_trades = [t.get('profit_loss', 0) for t in self.trades_history if t.get('profit_loss', 0) > 0]
        loss_trades = [t.get('profit_loss', 0) for t in self.trades_history if t.get('profit_loss', 0) < 0]
        
        avg_win = sum(profit_trades) / len(profit_trades) if profit_trades else 0
        avg_loss = sum(loss_trades) / len(loss_trades) if loss_trades else 0
        
        # Primera operación (la más antigua) para "active_since"
        active_since = "N/A"
        if self.trades_history:
            oldest_trade = min(self.trades_history, key=lambda x: x.get('time_opened', datetime.now()))
            if 'time_opened' in oldest_trade:
                active_since = oldest_trade['time_opened'].strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit_loss': total_pl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'open_trades': len(self.open_trades),
            'active_since': active_since
        }
    
    def _start_price_monitoring(self, symbol: str):
        """Start real-time price monitoring for a symbol"""
        try:
            def price_callback(ticker_data):
                # Process ticker data from websocket
                current_price = float(ticker_data['c'])
                self.real_time_prices[symbol] = current_price
                
                # Put in queue for analysis
                self.price_queue.put({
                    'symbol': symbol,
                    'price': current_price,
                    'time': datetime.now()
                })
            
            # Start the websocket connection
            self.binance_client.start_symbol_ticker_socket(symbol, price_callback)
            logger.info(f"Started price monitoring for {symbol}")
        
        except Exception as e:
            logger.error(f"Failed to start price monitoring for {symbol}: {str(e)}")
            raise APIError(f"Price monitoring failed: {str(e)}")
    
    def _trading_loop(self):
        """Main trading loop that processes price updates and generates signals"""
        logger.info("Trading loop started")
        
        while self.active:
            try:
                # Process price updates
                while not self.price_queue.empty():
                    price_data = self.price_queue.get(block=False)
                    self._process_price_update(price_data)
                
                # Check for closed trades
                self._check_open_trades()
                
                # Sleep to prevent CPU overuse
                time.sleep(0.1)
            
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)  # Sleep longer on error
        
        logger.info("Trading loop stopped")
    
    def _process_price_update(self, price_data: Dict[str, Any]):
        """Process a price update and check for trading signals"""
        symbol = price_data['symbol']
        current_price = price_data['price']
        
        # Skip if we already have an open trade for this symbol
        if symbol in self.open_trades:
            return
        
        try:
            # Get full set of indicators
            indicators = self.binance_client.calculate_indicators(
                symbol=symbol,
                interval=self.config.timeframe
            )
            
            # Add current price to indicators
            indicators['price'] = current_price
            
            # Store market data in database if available
            if hasattr(self, 'db') and self.db:
                try:
                    # Format timestamp
                    timestamp = price_data.get('time', datetime.now())
                    
                    # Prepare market data
                    market_data = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'open': indicators.get('open', current_price),
                        'high': indicators.get('high', current_price),
                        'low': indicators.get('low', current_price),
                        'close': current_price,
                        'volume': indicators.get('volume', 0),
                        'num_trades': indicators.get('num_trades', 0),
                        'timeframe': self.config.timeframe
                    }
                    
                    # Store market data
                    market_id = self.db.store_market_data(market_data)
                    
                    # Store indicators if market data was stored successfully
                    if market_id:
                        # Add symbol and timestamp to indicators if not present
                        if 'symbol' not in indicators:
                            indicators['symbol'] = symbol
                        if 'timestamp' not in indicators:
                            indicators['timestamp'] = timestamp
                            
                        self.db.store_indicators(market_id, indicators)
                except Exception as e:
                    logger.warning(f"Error storing market data in database: {str(e)}")
            
            # Get OHLCV data for analysis
            ohlcv_data = self.binance_client.get_ohlcv(
                symbol=symbol,
                interval=self.config.timeframe,
                limit=60  # Get enough data for analysis
            )
            
            # Generate trading signal - different approach based on strategy type
            if isinstance(self.strategy, LLMScalpingStrategy):
                # Use LLM strategy with full OHLCV data - no need for asyncio.run as we use the decorator
                signal = self.strategy.generate_signal(symbol, ohlcv_data)
                
                if signal['type'].name != 'NEUTRAL' and signal['confidence'] > self.config.confidence_threshold:
                    logger.info(f"LLM Signal detected: {signal['direction']} {symbol} at {current_price}")
                    
                    # Format parameters for trade execution
                    params = {
                        'entry': signal.get('entry_price', current_price),
                        'stop_loss': signal.get('stop_loss', current_price * 0.99 if signal['direction'] == 'BUY' else current_price * 1.01),
                        'take_profit': signal.get('take_profit', current_price * 1.01 if signal['direction'] == 'BUY' else current_price * 0.99),
                        'strategy_type': 'llm',
                        'confidence': signal['confidence'],
                        'analysis': signal.get('analysis', '')
                    }
                    
                    self._execute_trade(symbol, signal['direction'], params)
            else:
                # Use traditional ML or indicator strategy
                signal, direction, params = self.strategy.analyze(indicators)
                
                if signal and direction in ['BUY', 'SELL'] and params:
                    logger.info(f"Signal detected: {direction} {symbol} at {current_price}")
                    self._execute_trade(symbol, direction, params)
        
        except Exception as e:
            logger.error(f"Error processing price update for {symbol}: {str(e)}")
    
    def _execute_trade(self, symbol: str, side: str, params: Dict[str, float]):
        """Execute a trade based on a signal"""
        try:
            # Get account balance
            balances = self.binance_client.get_account_balance()
            quote_asset = symbol[3:]  # e.g., 'USDT' from 'BTCUSDT'
            quote_balance = balances.get(quote_asset, 0)
            
            entry_price = params['entry']
            stop_loss = params['stop_loss']
            take_profit = params['take_profit']
            
            # Extract strategy metadata
            strategy_type = params.get('strategy_type', 'indicator')
            model_used = params.get('model_used', None)
            confidence = params.get('confidence', 0)
            
            # Calculate position size
            quantity = self.strategy.calculate_position_size(
                symbol=symbol,
                account_balance=quote_balance,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
            
            if quantity <= 0:
                logger.warning(f"Calculated quantity is zero or negative for {symbol}")
                return
                
            # Verificar que tenemos fondos suficientes para la operación (importante para modo spot)
            required_funds = quantity * entry_price
            if required_funds > quote_balance:
                logger.warning(f"Fondos insuficientes para {symbol}: {required_funds} {quote_asset} requeridos, {quote_balance} {quote_asset} disponibles")
                # Ajustar cantidad a los fondos disponibles (con un pequeño margen de seguridad del 1%)
                available_funds = quote_balance * 0.99  # 1% de margen para comisiones
                quantity = available_funds / entry_price
                logger.info(f"Ajustando cantidad a {quantity} basado en fondos disponibles")
                
                # Verificar si la cantidad ajustada es suficiente para operar
                if quantity <= 0 or quantity < self.config.min_trade_quantity:
                    logger.warning(f"Cantidad ajustada demasiado pequeña para operar: {quantity}")
                    return
            
            # Execute the order with stop loss and take profit
            order_result = self.binance_client.create_order_with_sl_tp(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=None,  # Market order
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Track the open trade
            trade = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'time_opened': datetime.now(),
                'order_id': order_result['main_order']['orderId'],
                'strategy_type': strategy_type,
                'model_used': model_used,
                'confidence': confidence
            }
            
            # Store trade in database if available
            if hasattr(self, 'db') and self.db:
                try:
                    # Format trade data for database
                    db_trade = {
                        'symbol': symbol,
                        'entry_time': trade['time_opened'],
                        'side': side,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'strategy_type': strategy_type,
                        'confidence': confidence,
                        'model_used': model_used,
                        'status': 'open'
                    }
                    
                    # Store trade in database
                    db_id = self.db.store_trade(db_trade)
                    
                    # Add database ID to trade object for later reference
                    if db_id:
                        trade['db_id'] = db_id
                except Exception as e:
                    logger.warning(f"Error storing trade in database: {str(e)}")
            
            # Add trade to open trades
            self.open_trades[symbol] = trade
            
            logger.info(f"Executed {side} order for {quantity} {symbol} at {entry_price}")
        
        except Exception as e:
            logger.error(f"Failed to execute trade: {str(e)}")
            raise OrderError(f"Trade execution failed: {str(e)}")
    
    def _check_open_trades(self):
        """Check status of open trades and update as needed"""
        symbols_to_remove = []
        
        for symbol, trade in self.open_trades.items():
            try:
                current_price = self.real_time_prices.get(symbol)
                if not current_price:
                    continue
                
                # Update P/L for all open trades even if not closing them
                side = trade['side']
                entry_price = trade['entry_price']
                quantity = trade['quantity']
                
                # Calculate current P/L based on latest price
                if side == 'BUY':
                    trade['profit_loss'] = (current_price - entry_price) * quantity
                else:  # SELL
                    trade['profit_loss'] = (entry_price - current_price) * quantity
                
                # Check if stop loss or take profit hit
                stop_loss = trade['stop_loss']
                take_profit = trade['take_profit']
                
                # For BUY trades
                if side == 'BUY':
                    if current_price <= stop_loss:
                        self._close_trade(symbol, 'stop_loss', current_price)
                        symbols_to_remove.append(symbol)
                    elif current_price >= take_profit:
                        self._close_trade(symbol, 'take_profit', current_price)
                        symbols_to_remove.append(symbol)
                
                # For SELL trades
                elif side == 'SELL':
                    if current_price >= stop_loss:
                        self._close_trade(symbol, 'stop_loss', current_price)
                        symbols_to_remove.append(symbol)
                    elif current_price <= take_profit:
                        self._close_trade(symbol, 'take_profit', current_price)
                        symbols_to_remove.append(symbol)
            
            except Exception as e:
                logger.error(f"Error checking trade status for {symbol}: {str(e)}")
        
        # Remove closed trades
        for symbol in symbols_to_remove:
            if symbol in self.open_trades:
                del self.open_trades[symbol]
    
    def _close_trade(self, symbol: str, reason: str, price: float):
        """Close a trade and record the result"""
        trade = self.open_trades.get(symbol)
        if not trade:
            return
        
        try:
            # Calculate profit/loss
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            side = trade['side']
            
            if side == 'BUY':
                profit_loss = (price - entry_price) * quantity
            else:  # SELL
                profit_loss = (entry_price - price) * quantity
            
            # Record trade result
            trade_result = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': price,
                'quantity': quantity,
                'profit_loss': profit_loss,
                'time_opened': trade['time_opened'],
                'time_closed': datetime.now(),
                'reason': reason,
                'strategy_type': trade.get('strategy_type', 'indicator'),
                'model_used': trade.get('model_used', None),
                'confidence': trade.get('confidence', None)
            }
            
            # Store trade in database if available
            if hasattr(self, 'db') and self.db:
                try:
                    # Format trade data for database
                    db_trade = {
                        'symbol': symbol,
                        'entry_time': trade['time_opened'],
                        'exit_time': datetime.now(),
                        'side': side,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'quantity': quantity,
                        'profit_loss': profit_loss,
                        'strategy_type': trade.get('strategy_type', 'indicator'),
                        'confidence': trade.get('confidence', 0),
                        'model_used': trade.get('model_used', None),
                        'exit_reason': reason,
                        'status': 'closed'
                    }
                    
                    # If trade was already stored in database, update it
                    if 'db_id' in trade:
                        self.db.update_trade(trade['db_id'], {
                            'exit_time': datetime.now(),
                            'exit_price': price,
                            'profit_loss': profit_loss,
                            'exit_reason': reason,
                            'status': 'closed'
                        })
                    else:
                        # Otherwise store as new trade
                        self.db.store_trade(db_trade)
                        
                except Exception as e:
                    logger.warning(f"Error storing trade in database: {str(e)}")
            
            self.trades_history.append(trade_result)
            self.total_profit_loss += profit_loss
            
            logger.info(f"Closed {side} trade for {symbol}: {reason} at {price}, P/L: {profit_loss:.4f}")
            
            # Here you would place closing orders if needed
            # In this implementation we assume stop loss and take profit orders
            # are already placed when opening the position
            
        except Exception as e:
            logger.error(f"Error closing trade for {symbol}: {str(e)}")

    def schedule_ai_strategy_optimization(self, interval_hours=12):
        """Programar optimizaciones periódicas de estrategia"""
        self.optimization_thread = threading.Thread(
            target=self._ai_optimization_loop,
            args=(interval_hours,)
        )
        self.optimization_thread.daemon = True
        self.optimization_thread.start()

    def _ai_optimization_loop(self, interval_hours):
        """Loop de optimización periódica usando LLM o modelo tradicional"""
        # Esperar datos iniciales antes de primera optimización
        time.sleep(60 * 30)  # 30 minutos iniciales
        
        while self.active:
            try:
                # Ejecutar solo si tenemos suficientes datos
                min_trades = getattr(self.config, 'ai_min_trades_for_optimization', 10)
                
                if len(self.trades_history) >= min_trades:
                    logger.info("Iniciando optimización IA de estrategia...")
                    
                    # Obtener resumen de rendimiento
                    performance = self.get_performance_summary()
                    
                    # Approach depends on strategy type
                    if isinstance(self.strategy, LLMScalpingStrategy):
                        # For LLM strategy - get full OHLCV data for all symbols
                        for symbol in self.config.symbols:
                            try:
                                # Get OHLCV data
                                ohlcv_data = self.binance_client.get_ohlcv(
                                    symbol=symbol,
                                    interval=self.config.timeframe,
                                    limit=100  # More data for market analysis
                                )
                                
                                # Get recent trades for this symbol
                                symbol_trades = [t for t in self.trades_history if t['symbol'] == symbol]
                                
                                # Run market analysis with LLM - no need for asyncio.run as we use the decorator
                                analysis_result = self.strategy.analyze_market_conditions(
                                    symbol=symbol,
                                    ohlcv_data=ohlcv_data,
                                    trades_history=symbol_trades
                                )
                                
                                if 'error' not in analysis_result:
                                    logger.info(f"LLM optimización para {symbol}: {analysis_result.get('analysis', 'No analysis provided')}")
                                    
                                    # Log parameter adjustments if any
                                    if 'parameter_adjustments' in analysis_result:
                                        logger.info(f"Ajustes recomendados: {json.dumps(analysis_result['parameter_adjustments'])}")
                                        
                                    # Log market trend and risk adjustment
                                    logger.info(f"Tendencia: {analysis_result.get('market_trend', 'unknown')}, "
                                               f"Ajuste de riesgo: {analysis_result.get('risk_adjustment', 'unknown')}")
                                else:
                                    logger.warning(f"Error en optimización LLM para {symbol}: {analysis_result.get('error')}")
                            
                            except Exception as e:
                                logger.error(f"Error en optimización para {symbol}: {str(e)}")
                    else:
                        # For traditional strategy - use indicators
                        market_data = {}
                        for symbol in self.config.symbols:
                            indicators = self.binance_client.calculate_indicators(
                                symbol=symbol,
                                interval=self.config.timeframe
                            )
                            market_data[symbol] = indicators
                        
                        # Execute optimization if the strategy supports it
                        if hasattr(self.strategy, 'optimize_strategy_parameters'):
                            success = self.strategy.optimize_strategy_parameters(
                                market_data=market_data,
                                performance_history=performance
                            )
                            
                            if success:
                                logger.info("Optimización IA completada exitosamente")
                            else:
                                logger.warning("Optimización IA no pudo ser completada")
                        else:
                            logger.warning("La estrategia actual no soporta optimización IA")
                else:
                    logger.info(f"Optimización IA pospuesta: insuficientes trades ({len(self.trades_history)}/{min_trades})")
                
                # Esperar hasta el próximo intervalo
                time.sleep(interval_hours * 3600)
            
            except Exception as e:
                logger.error(f"Error en loop de optimización: {str(e)}")
                time.sleep(3600)  # Esperar 1 hora en caso de error

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of bot performance with enhanced metrics"""
        total_trades = len(self.trades_history)
        profitable_trades = sum(1 for t in self.trades_history if t['profit_loss'] > 0)
        
        # Calculate basic metrics
        if total_trades > 0:
            win_rate = (profitable_trades / total_trades) * 100
            avg_profit = sum(t['profit_loss'] for t in self.trades_history if t['profit_loss'] > 0) / max(1, profitable_trades)
            avg_loss = sum(abs(t['profit_loss']) for t in self.trades_history if t['profit_loss'] < 0) / max(1, total_trades - profitable_trades)
            
            # Calculate profit factor if there are losses
            if avg_loss > 0:
                profit_factor = avg_profit / avg_loss
            else:
                profit_factor = float('inf') if avg_profit > 0 else 0
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
        
        # Get symbols with best performance
        symbol_performance = {}
        for trade in self.trades_history:
            symbol = trade['symbol']
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {
                    'trades': 0,
                    'profit_loss': 0,
                    'wins': 0
                }
            
            symbol_performance[symbol]['trades'] += 1
            symbol_performance[symbol]['profit_loss'] += trade['profit_loss']
            if trade['profit_loss'] > 0:
                symbol_performance[symbol]['wins'] += 1
                
        # Sort symbols by profit/loss
        best_symbols = sorted(
            symbol_performance.items(), 
            key=lambda x: x[1]['profit_loss'], 
            reverse=True
        )[:3]
        
        # Calculate trading frequency
        if total_trades >= 2 and len(self.trades_history) >= 2:
            first_trade_time = self.trades_history[0]['time_opened']
            last_trade_time = self.trades_history[-1]['time_closed']
            time_diff = (last_trade_time - first_trade_time).total_seconds() / 3600  # hours
            trades_per_hour = total_trades / max(1, time_diff)
        else:
            trades_per_hour = 0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit_loss': self.total_profit_loss,
            'open_trades': len(self.open_trades),
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades_per_hour': trades_per_hour,
            'best_symbols': [{'symbol': s, 'profit': p['profit_loss'], 'trades': p['trades']} 
                             for s, p in best_symbols],
            'active_since': self.trades_history[0]['time_opened'].strftime('%Y-%m-%d %H:%M:%S') if self.trades_history else None
        }


if __name__ == "__main__":
    try:
        # Load configuration
        config = Config.from_env()
        
        # Initialize and start the bot
        bot = ScalpingBot(config)
        bot.start()
        
        # Keep the main thread running
        while True:
            time.sleep(60)
            
            # Print performance summary every minute
            performance = bot.get_performance_summary()
            logger.info(f"Performance: {json.dumps(performance)}")
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        if 'bot' in locals():
            bot.stop()
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if 'bot' in locals():
            bot.stop()
