import logging
import os
import sqlite3
import json
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

class DatabaseManager(ABC):
    """Clase base abstracta para gestores de base de datos"""
    
    @abstractmethod
    def store_market_data(self, data):
        pass
        
    @abstractmethod
    def store_indicators(self, market_id, indicators):
        pass
        
    @abstractmethod
    def store_trade(self, trade_data):
        pass
        
    @abstractmethod
    def get_training_data(self, symbol, start_date, end_date):
        pass
        
    @abstractmethod
    def register_model(self, model_data):
        pass
        
    @abstractmethod
    def get_active_model(self, symbol, model_type):
        pass


class SQLiteManager(DatabaseManager):
    """Implementación de SQLite"""
    
    def __init__(self, config):
        self.logger = logging.getLogger('SQLiteManager')
        
        # Crear directorio para la base de datos si no existe
        database_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database')
        os.makedirs(database_dir, exist_ok=True)
        
        # Definir ruta del archivo SQLite
        self.db_path = os.path.join(database_dir, 'trading_data.db')
        self.conn = None
        
        self._connect()
        self._create_tables()
        
    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # Habilitar retorno de resultados como diccionario
            self.conn.row_factory = sqlite3.Row
            return True
        except Exception as e:
            self.logger.error(f"SQLite connection error: {str(e)}")
            return False
            
    def _create_tables(self):
        """Crear tablas si no existen"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        
        try:
            # Tabla market_data
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                num_trades INTEGER,
                timeframe TEXT NOT NULL
            )
            ''')
            
            # Índice para market_data
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_symbol_time 
            ON market_data (symbol, timestamp)
            ''')
            
            # Tabla indicators
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_data_id INTEGER,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                indicator_data TEXT NOT NULL,
                FOREIGN KEY (market_data_id) REFERENCES market_data (id)
            )
            ''')
            
            # Índice para indicators
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_indicator_symbol_time 
            ON indicators (symbol, timestamp)
            ''')
            
            # Tabla trades
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                profit_loss REAL,
                strategy_type TEXT NOT NULL,
                confidence REAL,
                model_used TEXT,
                exit_reason TEXT,
                status TEXT DEFAULT 'open'
            )
            ''')
            
            # Tabla models
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                created_at TEXT NOT NULL,
                file_path TEXT NOT NULL,
                accuracy REAL,
                direction_accuracy REAL,
                rmse REAL,
                training_data_start TEXT,
                training_data_end TEXT,
                is_active INTEGER DEFAULT 0,
                performance_score REAL
            )
            ''')
            
            self.conn.commit()
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {str(e)}")
        finally:
            cursor.close()
            
    def store_market_data(self, data):
        """Almacenar datos de mercado"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        try:
            # Formatear timestamp si es objeto datetime
            timestamp = data['timestamp']
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
                
            query = '''
            INSERT INTO market_data 
            (symbol, timestamp, open, high, low, close, volume, num_trades, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            cursor.execute(query, (
                data['symbol'], timestamp, data['open'], data['high'], 
                data['low'], data['close'], data['volume'], 
                data.get('num_trades', 0), data['timeframe']
            ))
            
            self.conn.commit()
            return cursor.lastrowid
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")
            return None
        finally:
            cursor.close()
            
    def store_indicators(self, market_id, indicators):
        """Almacenar indicadores técnicos"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        try:
            # Extraer symbol y timestamp
            symbol = indicators.get('symbol')
            timestamp = indicators.get('timestamp')
            
            if not symbol or not timestamp:
                # Obtener el símbolo de market_data si no está en indicators
                cursor.execute("SELECT symbol, timestamp FROM market_data WHERE id = ?", (market_id,))
                result = cursor.fetchone()
                if result:
                    symbol = result['symbol']
                    timestamp = result['timestamp']
                else:
                    raise ValueError(f"Market data not found for id: {market_id}")
            
            # Convertir timestamp si es objeto datetime
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
                
            # Serializar indicadores a JSON - primero convertir los objetos datetime a string
            # Crear una copia para no modificar el original
            indicators_copy = indicators.copy()
            
            # Convertir todos los valores datetime a string
            for key, value in indicators_copy.items():
                if isinstance(value, datetime):
                    indicators_copy[key] = value.isoformat()
                    
            indicator_data = json.dumps(indicators_copy)
            
            query = '''
            INSERT INTO indicators 
            (market_data_id, symbol, timestamp, indicator_data)
            VALUES (?, ?, ?, ?)
            '''
            
            cursor.execute(query, (market_id, symbol, timestamp, indicator_data))
            
            self.conn.commit()
            return cursor.lastrowid
            
        except Exception as e:
            self.logger.error(f"Error storing indicators: {str(e)}")
            return None
        finally:
            cursor.close()
    
    def store_trade(self, trade_data):
        """Almacenar información de operación"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        try:
            # Formatear timestamps
            entry_time = trade_data['entry_time']
            if isinstance(entry_time, datetime):
                entry_time = entry_time.isoformat()
                
            exit_time = trade_data.get('exit_time')
            if exit_time and isinstance(exit_time, datetime):
                exit_time = exit_time.isoformat()
                
            query = '''
            INSERT INTO trades 
            (symbol, entry_time, exit_time, side, entry_price, exit_price, 
             quantity, profit_loss, strategy_type, confidence, model_used, 
             exit_reason, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            cursor.execute(query, (
                trade_data['symbol'], entry_time, exit_time, trade_data['side'],
                trade_data['entry_price'], trade_data.get('exit_price'),
                trade_data['quantity'], trade_data.get('profit_loss'),
                trade_data.get('strategy_type', 'unknown'), trade_data.get('confidence'),
                trade_data.get('model_used'), trade_data.get('exit_reason'),
                trade_data.get('status', 'open')
            ))
            
            self.conn.commit()
            return cursor.lastrowid
            
        except Exception as e:
            self.logger.error(f"Error storing trade: {str(e)}")
            return None
        finally:
            cursor.close()
    
    def get_training_data(self, symbol, start_date, end_date):
        """Obtener datos para entrenamiento"""
        if not self.conn:
            self._connect()
            
        try:
            # Formatear fechas si son objetos datetime
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
                
            # Consulta para obtener datos combinados
            query = '''
            SELECT m.*, i.indicator_data
            FROM market_data m
            LEFT JOIN indicators i ON m.id = i.market_data_id
            WHERE m.symbol = ? AND m.timestamp BETWEEN ? AND ?
            ORDER BY m.timestamp
            '''
            
            # Ejecutar consulta y convertir a DataFrame
            df = pd.read_sql_query(query, self.conn, params=(symbol, start_date, end_date))
            
            # Decodificar JSON de indicadores
            if 'indicator_data' in df.columns and len(df) > 0:
                # Convertir strings JSON a diccionarios
                df['indicators'] = df['indicator_data'].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else {}
                )
                df = df.drop('indicator_data', axis=1)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {str(e)}")
            return pd.DataFrame()  # Retornar DataFrame vacío en caso de error
    
    def register_model(self, model_data):
        """Registrar modelo entrenado"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        try:
            # Formatear timestamps
            created_at = model_data.get('created_at', datetime.now())
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
                
            training_start = model_data.get('training_data_start')
            if training_start and isinstance(training_start, datetime):
                training_start = training_start.isoformat()
                
            training_end = model_data.get('training_data_end')
            if training_end and isinstance(training_end, datetime):
                training_end = training_end.isoformat()
                
            query = '''
            INSERT INTO models 
            (model_type, symbol, created_at, file_path, accuracy, 
             direction_accuracy, rmse, training_data_start, 
             training_data_end, is_active, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            cursor.execute(query, (
                model_data['model_type'], model_data['symbol'], created_at,
                model_data['file_path'], model_data.get('accuracy'),
                model_data.get('direction_accuracy'), model_data.get('rmse'),
                training_start, training_end,
                1 if model_data.get('is_active', False) else 0,
                model_data.get('performance_score')
            ))
            
            model_id = cursor.lastrowid
            
            # Si este modelo está activo, desactivar otros modelos del mismo tipo y símbolo
            if model_data.get('is_active', False):
                self._activate_model(model_id, model_data['model_type'], model_data['symbol'])
            
            self.conn.commit()
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            return None
        finally:
            cursor.close()
            
    def _activate_model(self, model_id, model_type, symbol):
        """Activar un modelo y desactivar otros del mismo tipo/símbolo"""
        cursor = self.conn.cursor()
        try:
            # Desactivar todos los modelos existentes del mismo tipo y símbolo
            query_deactivate = '''
            UPDATE models 
            SET is_active = 0
            WHERE model_type = ? AND symbol = ? AND id != ?
            '''
            
            cursor.execute(query_deactivate, (model_type, symbol, model_id))
            
            # Activar el modelo especificado
            query_activate = '''
            UPDATE models 
            SET is_active = 1
            WHERE id = ?
            '''
            
            cursor.execute(query_activate, (model_id,))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error activating model: {str(e)}")
            return False
        finally:
            cursor.close()
            
    def get_active_model(self, symbol, model_type):
        """Obtener modelo activo para un símbolo y tipo"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        try:
            query = '''
            SELECT *
            FROM models
            WHERE symbol = ? AND model_type = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            '''
            
            cursor.execute(query, (symbol, model_type))
            result = cursor.fetchone()
            
            if result:
                return dict(result)  # Convertir Row a diccionario
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting active model: {str(e)}")
            return None
        finally:
            cursor.close()
    
    def update_trade(self, trade_id, update_data):
        """Actualizar información de una operación existente"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        try:
            # Construir la consulta dinámicamente basada en los campos proporcionados
            fields = []
            values = []
            
            for key, value in update_data.items():
                # Manejar timestamps
                if key in ['exit_time'] and isinstance(value, datetime):
                    value = value.isoformat()
                
                fields.append(f"{key} = ?")
                values.append(value)
            
            # Agregar ID a los valores
            values.append(trade_id)
            
            query = f'''
            UPDATE trades 
            SET {", ".join(fields)}
            WHERE id = ?
            '''
            
            cursor.execute(query, values)
            self.conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            self.logger.error(f"Error updating trade: {str(e)}")
            return False
        finally:
            cursor.close()
    
    def get_open_trades(self, symbol=None):
        """Obtener todas las operaciones abiertas"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        try:
            if symbol:
                query = "SELECT * FROM trades WHERE status = 'open' AND symbol = ?"
                cursor.execute(query, (symbol,))
            else:
                query = "SELECT * FROM trades WHERE status = 'open'"
                cursor.execute(query)
                
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Error getting open trades: {str(e)}")
            return []
        finally:
            cursor.close()
    
    def get_recent_trades(self, limit=100, symbol=None):
        """Obtener operaciones recientes"""
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
        try:
            if symbol:
                query = """
                SELECT * FROM trades 
                WHERE symbol = ? 
                ORDER BY entry_time DESC LIMIT ?
                """
                cursor.execute(query, (symbol, limit))
            else:
                query = "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?"
                cursor.execute(query, (limit,))
                
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {str(e)}")
            return []
        finally:
            cursor.close()