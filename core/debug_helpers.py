"""
Módulo de utilidades para la depuración del sistema de trading

Este módulo proporciona funciones y clases para facilitar la depuración de
operaciones de trading, incluyendo registros detallados y herramientas de diagnóstico.
"""

import os
import json
import time
import inspect
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from functools import wraps
import logging

# Comprobación si estamos en modo debug
def is_debug_mode() -> bool:
    """Verifica si el modo de depuración está activado"""
    return os.environ.get("MIDAS_DEBUG", "0").lower() in ("1", "true", "yes", "y")

# ID único para cada sesión de trading
SESSION_ID = datetime.now().strftime("%Y%m%d%H%M%S")

def get_session_id() -> str:
    """Retorna el ID de la sesión actual"""
    return SESSION_ID

def log_trade_operation(logger, operation_type: str, symbol: str, data: Dict[str, Any]) -> str:
    """
    Registra una operación de trading con un ID único
    
    Args:
        logger: El logger a utilizar
        operation_type: Tipo de operación (signal, order, fill, etc.)
        symbol: Símbolo del instrumento
        data: Datos de la operación
        
    Returns:
        ID de transacción único
    """
    # Generar ID de transacción basado en timestamp para seguimiento
    txid = f"{operation_type}_{symbol}_{datetime.now().strftime('%H%M%S%f')}"
    
    if is_debug_mode():
        # En modo debug, loguear todo el contexto
        caller_frame = inspect.currentframe().f_back
        filename = caller_frame.f_code.co_filename
        lineno = caller_frame.f_lineno
        function = caller_frame.f_code.co_name
        
        logger.debug(f"[TXID:{txid}] {operation_type.upper()} on {symbol} from {os.path.basename(filename)}:{lineno} in {function}()")
        logger.debug(f"[TXID:{txid}] Data: {json.dumps(data, default=str)}")
    else:
        # En modo normal, solo lo esencial
        if hasattr(logger, 'trade'):
            logger.trade(f"[TXID:{txid}] {operation_type.upper()} {symbol}: {data.get('side', 'N/A')} {data.get('quantity', 0)} @ {data.get('price', 0)}")
        else:
            logger.info(f"[TXID:{txid}] {operation_type.upper()} {symbol}: {data.get('side', 'N/A')} {data.get('quantity', 0)} @ {data.get('price', 0)}")
    
    return txid

def performance_timer(func):
    """
    Decorador para medir el tiempo de ejecución de funciones críticas
    
    Solo activo en modo debug
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_debug_mode():
            return func(*args, **kwargs)
        
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        result = None
        exception = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            exception = e
            raise
        finally:
            elapsed_time = (time.time() - start_time) * 1000  # en ms
            
            # Crear mensaje de log
            log_msg = f"PERFORMANCE: {func.__name__} completed in {elapsed_time:.2f}ms"
            
            # Añadir resultado o excepción si está disponible
            if exception:
                log_msg += f" (raised {type(exception).__name__})"
            elif result is not None:
                # Intentar obtener una representación simplificada del resultado
                if isinstance(result, dict):
                    if len(result) > 3:
                        result_repr = f"dict with {len(result)} keys"
                    else:
                        result_repr = str(result)
                elif isinstance(result, (list, tuple)):
                    result_repr = f"{type(result).__name__} with {len(result)} items"
                elif hasattr(result, '__dict__'):
                    result_repr = f"{type(result).__name__} object"
                else:
                    result_repr = str(result)
                
                log_msg += f" (returned {result_repr})"
            
            logger.debug(log_msg)
    
    return wrapper

class TradingDebugger:
    """
    Clase para depuración avanzada de operaciones de trading
    
    Proporciona métodos para:
    - Registro detallado de señales y operaciones
    - Rastreo de decisiones de trading
    - Visualización y exportación de estados del sistema
    """
    
    def __init__(self, logger=None):
        """
        Inicializa el depurador de trading
        
        Args:
            logger: Logger para registrar información (opcional)
        """
        self.logger = logger or logging.getLogger('TradingDebugger')
        self.debug_mode = is_debug_mode()
        
        # Historial de operaciones para depuración
        self.operation_history = []
        
        # Información contextual actualizada durante el trading
        self.context = {
            'session_id': SESSION_ID,
            'start_time': datetime.now(),
            'system_info': self._collect_system_info(),
        }
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Recolecta información del sistema para diagnóstico"""
        import platform
        import sys
        
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'pid': os.getpid(),
        }
        
        # Intentar obtener información adicional si están disponibles las librerías
        try:
            import psutil
            process = psutil.Process(os.getpid())
            info['memory_info'] = {
                'rss': process.memory_info().rss / (1024 * 1024),  # MB
                'vms': process.memory_info().vms / (1024 * 1024),  # MB
            }
        except ImportError:
            pass
            
        try:
            import torch
            info['torch_available'] = True
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
        except ImportError:
            info['torch_available'] = False
        
        return info
    
    def log_signal(self, strategy_name: str, symbol: str, timeframe: str, 
                  signal_data: Dict[str, Any]) -> str:
        """
        Registra una señal de trading generada por una estrategia
        
        Args:
            strategy_name: Nombre de la estrategia que generó la señal
            symbol: Símbolo del instrumento
            timeframe: Marco de tiempo analizado
            signal_data: Datos de la señal incluyendo dirección, confianza, etc.
            
        Returns:
            ID de señal único para seguimiento
        """
        if not self.debug_mode:
            return ""
            
        signal_id = f"SIG_{symbol}_{timeframe}_{datetime.now().strftime('%H%M%S%f')}"
        
        # Enriquecer datos de la señal con metadatos
        enriched_data = {
            'signal_id': signal_id,
            'strategy': strategy_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            **signal_data
        }
        
        # Guardar en historial
        self.operation_history.append(('signal', enriched_data))
        
        # Registrar en log
        self.logger.debug(f"[SIGNAL:{signal_id}] Generated by {strategy_name} for {symbol} ({timeframe}): "
                         f"Direction: {signal_data.get('direction', 'unknown')}, "
                         f"Confidence: {signal_data.get('confidence', 'N/A')}")
        
        # Detalles técnicos de la señal
        if 'indicators' in signal_data:
            indicators = signal_data['indicators']
            self.logger.debug(f"[SIGNAL:{signal_id}] Technical indicators: {json.dumps(indicators, default=str)}")
        
        return signal_id
    
    def log_order(self, order_data: Dict[str, Any], signal_id: str = None) -> str:
        """
        Registra una orden enviada al exchange
        
        Args:
            order_data: Datos de la orden incluyendo símbolo, lado, cantidad, etc.
            signal_id: ID de la señal que generó esta orden (opcional)
            
        Returns:
            ID de orden único para seguimiento
        """
        if not self.debug_mode:
            return ""
            
        order_id = f"ORD_{order_data.get('symbol', 'unknown')}_{datetime.now().strftime('%H%M%S%f')}"
        
        # Enriquecer datos de la orden con metadatos
        enriched_data = {
            'order_id': order_id,
            'signal_id': signal_id,
            'timestamp': datetime.now(),
            **order_data
        }
        
        # Guardar en historial
        self.operation_history.append(('order', enriched_data))
        
        # Registrar en log
        self.logger.debug(f"[ORDER:{order_id}] Submitted for {order_data.get('symbol', 'unknown')}: "
                         f"{order_data.get('side', 'unknown')} "
                         f"{order_data.get('quantity', 0)} @ {order_data.get('price', 'market')}")
        
        if signal_id:
            self.logger.debug(f"[ORDER:{order_id}] Based on signal: {signal_id}")
        
        return order_id
    
    def log_execution(self, execution_data: Dict[str, Any], order_id: str = None) -> None:
        """
        Registra la ejecución de una orden
        
        Args:
            execution_data: Datos de la ejecución incluyendo precio, cantidad, etc.
            order_id: ID de la orden que fue ejecutada (opcional)
        """
        if not self.debug_mode:
            return
            
        # Enriquecer datos de ejecución
        enriched_data = {
            'execution_id': f"EXEC_{datetime.now().strftime('%H%M%S%f')}",
            'order_id': order_id,
            'timestamp': datetime.now(),
            **execution_data
        }
        
        # Guardar en historial
        self.operation_history.append(('execution', enriched_data))
        
        # Registrar en log
        self.logger.debug(f"[EXEC:{enriched_data['execution_id']}] Order executed for "
                         f"{execution_data.get('symbol', 'unknown')}: "
                         f"{execution_data.get('side', 'unknown')} "
                         f"{execution_data.get('quantity', 0)} @ {execution_data.get('price', 0)}")
        
        if order_id:
            self.logger.debug(f"[EXEC:{enriched_data['execution_id']}] For order: {order_id}")
    
    def export_history(self, filename: str = None) -> str:
        """
        Exporta el historial de operaciones a un archivo JSON
        
        Args:
            filename: Nombre del archivo (opcional, genera uno por defecto)
            
        Returns:
            Ruta del archivo generado
        """
        if not filename:
            filename = f"trading_history_{SESSION_ID}.json"
            
        # Crear directorio logs si no existe
        os.makedirs('logs', exist_ok=True)
        file_path = os.path.join('logs', filename)
        
        # Convertir datos a formato JSON
        data = {
            'session_info': self.context,
            'history': [(op_type, self._serialize_entry(entry)) for op_type, entry in self.operation_history]
        }
        
        # Guardar archivo
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Trading history exported to {file_path}")
        return file_path
    
    def _serialize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convierte un entrada del historial a formato serializable"""
        result = {}
        for key, value in entry.items():
            if isinstance(value, (datetime, date)):
                result[key] = value.isoformat()
            elif hasattr(value, '__dict__'):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def record_exception(self, exception: Exception, context: str = None) -> None:
        """
        Registra una excepción ocurrida durante el trading
        
        Args:
            exception: La excepción capturada
            context: Contexto adicional sobre dónde ocurrió
        """
        if not self.debug_mode:
            # En modo normal, solo registrar el error básico
            if context:
                self.logger.error(f"Error in {context}: {str(exception)}")
            else:
                self.logger.error(str(exception))
            return
            
        # En modo debug, registrar el stack trace completo
        exception_id = f"ERR_{datetime.now().strftime('%H%M%S%f')}"
        
        self.logger.error(f"[ERROR:{exception_id}] {type(exception).__name__} in {context or 'trading operation'}: {str(exception)}")
        self.logger.debug(f"[ERROR:{exception_id}] Stack trace:\n{traceback.format_exc()}")
        
        # Guardar en historial
        error_entry = {
            'error_id': exception_id,
            'type': type(exception).__name__,
            'message': str(exception),
            'context': context,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now()
        }
        
        self.operation_history.append(('error', error_entry))
    
    def log_decision(self, decision_type: str, symbol: str, data: Dict[str, Any], 
                   reason: str = None) -> None:
        """
        Registra una decisión de trading (entrada, salida, skip)
        
        Args:
            decision_type: Tipo de decisión (entry, exit, skip)
            symbol: Símbolo del instrumento
            data: Datos relacionados con la decisión
            reason: Razón para la decisión (opcional)
        """
        if not self.debug_mode:
            return
            
        decision_id = f"DEC_{decision_type}_{symbol}_{datetime.now().strftime('%H%M%S%f')}"
        
        # Enriquecer datos
        enriched_data = {
            'decision_id': decision_id,
            'type': decision_type,
            'symbol': symbol,
            'reason': reason,
            'timestamp': datetime.now(),
            **data
        }
        
        # Guardar en historial
        self.operation_history.append(('decision', enriched_data))
        
        # Registrar en log
        msg = f"[DECISION:{decision_id}] {decision_type.upper()} for {symbol}"
        if reason:
            msg += f" - Reason: {reason}"
            
        self.logger.debug(msg)
        
        # Detalles adicionales en un segundo mensaje
        if data:
            self.logger.debug(f"[DECISION:{decision_id}] Details: {json.dumps(data, default=str)}")

# Función para crear un decorador que registre argumentos y resultados de funciones críticas
def debug_function(logger=None):
    """
    Decorador para depurar funciones críticas
    
    Args:
        logger: Logger a utilizar (opcional, usa el logger del módulo por defecto)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_debug_mode():
                return func(*args, **kwargs)
                
            # Obtener logger
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
                
            # Registrar llamada a función
            arg_str = ", ".join([repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()])
            logger.debug(f"CALL: {func.__name__}({arg_str})")
            
            # Ejecutar función
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Registrar resultado
                elapsed = (time.time() - start_time) * 1000
                result_str = str(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                logger.debug(f"RETURN: {func.__name__} -> {result_str} (took {elapsed:.2f}ms)")
                
                return result
                
            except Exception as e:
                # Registrar excepción
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"EXCEPTION: {func.__name__} -> {type(e).__name__}: {str(e)} (took {elapsed:.2f}ms)")
                logger.debug(f"TRACEBACK: {traceback.format_exc()}")
                raise
                
        return wrapper
    return decorator