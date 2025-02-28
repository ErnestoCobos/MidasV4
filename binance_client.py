import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np
import random
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
import websocket
import json
import threading
from config import Config
from exceptions import CredentialsError, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BinanceClient')

class BinanceClient:
    """
    A wrapper class for interacting with the Binance API, optimized for scalping

    Attributes:
        config (Config): Configuration object containing API credentials
        client (Client): Binance API client instance
        _ws_connections (Dict): Store websocket connections
    """

    def __init__(self, config: Config):
        """Initialize BinanceClient with configuration"""
        self.config = config
        self._ws_connections = {}
        self.simulation_mode = self.config.api_key == "simulation_mode_key"
        
        if not self.simulation_mode and not self.config.validate():
            raise CredentialsError("API credentials are not properly configured")

        try:
            if self.simulation_mode:
                self.client = None
                logger.info("Initialized in simulation mode (no API connection)")
            else:
                # Always use testnet=True to ensure we don't connect to real Binance
                self.client = Client(self.config.api_key, self.config.api_secret, testnet=True)
                logger.info("Successfully initialized Binance Testnet client")
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {str(e)}")
            raise CredentialsError(f"Failed to initialize client: {str(e)}")

    def start_symbol_ticker_socket(self, symbol: str, callback: Callable) -> None:
        """
        Start websocket connection for real-time price updates

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            callback (Callable): Callback function for price updates
        """
        # In simulation mode, create a thread that generates simulated price data
        if self.simulation_mode:
            def simulate_price_feed():
                # Generate starting price based on symbol
                if 'BTC' in symbol:
                    base_price = 65000.0
                elif 'ETH' in symbol:
                    base_price = 3500.0
                else:
                    base_price = 100.0
                    
                current_price = base_price
                
                # Mock ticker data
                while True:
                    # Generate small random price change (0.1% to -0.1%)
                    change_pct = (random.random() - 0.5) * 0.2
                    current_price = current_price * (1 + change_pct)
                    
                    # Create simulated ticker data
                    ticker_data = {
                        'e': '24hrTicker',  # Event type
                        's': symbol,        # Symbol
                        'c': str(current_price),  # Current price
                        'o': str(current_price * 0.99),  # Open price
                        'h': str(current_price * 1.02),  # High price
                        'l': str(current_price * 0.98),  # Low price
                        'v': str(random.random() * 100),  # Volume
                        'q': str(random.random() * 1000000)  # Quote volume
                    }
                    
                    # Call callback with simulated data
                    callback(ticker_data)
                    
                    # Update P/L for any open trades that match this symbol
                    if hasattr(self, 'bot') and hasattr(self.bot, 'open_trades'):
                        for trade_id, trade in self.bot.open_trades.items():
                            if trade['symbol'] == symbol:
                                # Update P/L directly based on current price
                                side = trade['side']
                                entry_price = trade['entry_price']
                                quantity = trade['quantity']
                                
                                if side == 'BUY':
                                    trade['profit_loss'] = (current_price - entry_price) * quantity
                                else:  # SELL
                                    trade['profit_loss'] = (entry_price - current_price) * quantity
                    
                    # Sleep for random time (0.5 to 3 seconds)
                    time.sleep(0.5 + random.random() * 2.5)
            
            # Start simulation thread
            sim_thread = threading.Thread(target=simulate_price_feed, daemon=True)
            sim_thread.start()
            
            self._ws_connections[symbol] = (None, sim_thread)
            logger.info(f"Started simulated price feed for {symbol}")
            return
        
        # Real mode with actual WebSocket connection
        def on_message(ws, message):
            data = json.loads(message)
            # Ignore subscription confirmation messages
            if 'result' in data and data['result'] is None:
                return
            # Only process ticker messages
            if 'e' in data and data['e'] == '24hrTicker':
                callback(data)

        def on_error(ws, error):
            logger.error(f"Websocket error: {str(error)}")

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Websocket closed for {symbol}")

        def on_open(ws):
            logger.info(f"Websocket opened for {symbol}")
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@ticker"],
                "id": 1
            }
            ws.send(json.dumps(subscribe_message))

        try:
            # Disable websocket debug output for cleaner UI
            websocket.enableTrace(False)
            ws = websocket.WebSocketApp(
                "wss://testnet.binance.vision/ws",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )

            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()

            self._ws_connections[symbol] = (ws, ws_thread)
            logger.info(f"Started real-time price feed for {symbol}")
        except Exception as e:
            logger.error(f"Failed to start websocket for {symbol}: {str(e)}")
            raise APIError(f"Websocket initialization failed: {str(e)}")

    def stop_symbol_ticker_socket(self, symbol: str) -> None:
        """Stop websocket connection for a symbol"""
        if symbol in self._ws_connections:
            ws, thread = self._ws_connections[symbol]
            ws.close()
            thread.join()
            del self._ws_connections[symbol]
            logger.info(f"Stopped real-time price feed for {symbol}")

    def create_order_with_sl_tp(
        self,
        symbol: str,
        side: str,
        quantity: Union[float, Decimal],
        price: Optional[Union[float, Decimal]] = None,
        stop_loss: Optional[Union[float, Decimal]] = None,
        take_profit: Optional[Union[float, Decimal]] = None
    ) -> Dict[str, Any]:
        """
        Create a spot order with optional stop loss and take profit orders

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            side (str): 'BUY' or 'SELL'
            quantity (float|Decimal): Amount of base asset
            price (float|Decimal, optional): Limit price. If not provided, creates market order
            stop_loss (float|Decimal, optional): Stop loss price
            take_profit (float|Decimal, optional): Take profit price

        Returns:
            Dict[str, Any]: Order details including main order and SL/TP orders
            
        Raises:
            APIError: Si no hay fondos suficientes o hay un error en la API
        """
        # Validar los fondos disponibles (especialmente para compras)
        if side.upper() == 'BUY' and not self.simulation_mode:
            # Verificar fondos disponibles para operaciones reales (no simulación)
            quote_asset = symbol[3:]  # e.g., 'USDT' from 'BTCUSDT'
            quote_balance = self.get_account_balance(quote_asset).get(quote_asset, 0)
            
            # Obtener precio actual para mercado o usar el precio especificado para limit
            current_price = price if price else self.get_market_price(symbol)
            required_funds = float(quantity) * float(current_price)
            
            if required_funds > quote_balance:
                error_msg = f"Fondos insuficientes para {symbol}: {required_funds} {quote_asset} requeridos, {quote_balance} {quote_asset} disponibles"
                logger.error(error_msg)
                raise APIError(error_msg)
        if self.simulation_mode:
            # In simulation mode, generate mock order responses
            order_id = int(time.time() * 1000)  # Use timestamp as order ID
            current_price = price if price else (
                65000.0 if 'BTC' in symbol else (3500.0 if 'ETH' in symbol else 100.0)
            )
            
            # Create mock main order
            main_order = {
                'symbol': symbol,
                'orderId': order_id,
                'clientOrderId': f'sim_{order_id}',
                'transactTime': int(time.time() * 1000),
                'price': str(price) if price else '0.0',
                'origQty': str(quantity),
                'executedQty': str(quantity),
                'status': 'FILLED',
                'type': 'LIMIT' if price else 'MARKET',
                'side': side,
                'fills': [{
                    'price': str(current_price),
                    'qty': str(quantity),
                    'commission': '0.0',
                    'commissionAsset': 'BNB'
                }]
            }
            
            orders = {'main_order': main_order}
            opposite_side = 'SELL' if side == 'BUY' else 'BUY'
            
            # Create mock stop loss order
            if stop_loss:
                sl_order = {
                    'symbol': symbol,
                    'orderId': order_id + 1,
                    'clientOrderId': f'sim_sl_{order_id}',
                    'transactTime': int(time.time() * 1000),
                    'price': str(stop_loss),
                    'origQty': str(quantity),
                    'executedQty': '0.0',
                    'status': 'NEW',
                    'type': 'STOP_LOSS_LIMIT',
                    'side': opposite_side,
                    'stopPrice': str(stop_loss)
                }
                orders['stop_loss_order'] = sl_order
            
            # Create mock take profit order
            if take_profit:
                tp_order = {
                    'symbol': symbol,
                    'orderId': order_id + 2,
                    'clientOrderId': f'sim_tp_{order_id}',
                    'transactTime': int(time.time() * 1000),
                    'price': str(take_profit),
                    'origQty': str(quantity),
                    'executedQty': '0.0',
                    'status': 'NEW',
                    'type': 'LIMIT',
                    'side': opposite_side
                }
                orders['take_profit_order'] = tp_order
            
            logger.info(f"Simulated {side} order for {quantity} {symbol}")
            return orders
        
        try:
            # Create main order
            main_order = self.create_spot_order(symbol, side, quantity, price)
            orders = {'main_order': main_order}

            if main_order['status'] == 'FILLED':
                opposite_side = 'SELL' if side == 'BUY' else 'BUY'

                # Create stop loss order
                if stop_loss:
                    try:
                        sl_order = self.client.create_order(
                            symbol=symbol,
                            side=opposite_side,
                            type='STOP_LOSS_LIMIT',
                            quantity=quantity,
                            price=float(stop_loss),
                            stopPrice=float(stop_loss),
                            timeInForce='GTC'
                        )
                        orders['stop_loss_order'] = sl_order
                    except Exception as e:
                        logger.error(f"Failed to create stop loss order: {str(e)}")

                # Create take profit order
                if take_profit:
                    try:
                        tp_order = self.client.create_order(
                            symbol=symbol,
                            side=opposite_side,
                            type='LIMIT',
                            quantity=quantity,
                            price=float(take_profit),
                            timeInForce='GTC'
                        )
                        orders['take_profit_order'] = tp_order
                    except Exception as e:
                        logger.error(f"Failed to create take profit order: {str(e)}")

            return orders

        except BinanceAPIException as e:
            error_msg = f"Failed to create order with SL/TP: {str(e)}"
            logger.error(error_msg)
            raise APIError(error_msg)

    def calculate_indicators(self, symbol: str, interval: str = '1m', limit: int = 100) -> Dict[str, Any]:
        """
        Calculate basic technical indicators for scalping

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Kline interval (default: '1m' for scalping)
            limit (int): Number of candles to analyze

        Returns:
            Dict[str, Any]: Dictionary with calculated indicators
        """
        try:
            # Get kline data
            klines = self.get_klines(symbol, interval, limit)

            # Convert to pandas DataFrame
            df = pd.DataFrame([{
                'close': float(k['close']),
                'high': float(k['high']),
                'low': float(k['low']),
                'volume': float(k['volume'])
            } for k in klines])

            # Calculate indicators
            indicators = {}

            # Simple Moving Averages
            indicators['sma_7'] = df['close'].rolling(window=7).mean().iloc[-1]
            indicators['sma_25'] = df['close'].rolling(window=25).mean().iloc[-1]

            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]

            # RSI (14 periods)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]

            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['current_volume'] = df['volume'].iloc[-1]

            return indicators

        except Exception as e:
            error_msg = f"Failed to calculate indicators: {str(e)}"
            logger.error(error_msg)
            raise APIError(error_msg)

    def get_account_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        """
        Get account balance for all assets or a specific asset

        Args:
            asset (str, optional): Specific asset to query (e.g., 'BTC')

        Returns:
            Dict[str, float]: Dictionary of asset balances
        """
        if self.simulation_mode:
            # Return simulated balances in simulation mode
            sim_balances = {
                'USDT': 10000.0,
                'BTC': 0.15,
                'ETH': 2.0,
                'BNB': 10.0
            }
            
            if asset:
                return {asset: sim_balances.get(asset, 0.0)}
            return sim_balances
            
        try:
            account = self.client.get_account()
            balances = {}

            for balance in account['balances']:
                if float(balance['free']) > 0 or float(balance['locked']) > 0:
                    total = float(balance['free']) + float(balance['locked'])
                    if asset:
                        if balance['asset'] == asset:
                            return {asset: total}
                    else:
                        balances[balance['asset']] = total

            return balances if not asset else {}

        except BinanceAPIException as e:
            logger.error(f"API error while fetching balance: {str(e)}")
            raise APIError(f"Failed to get balance: {str(e)}")

    def get_market_price(self, symbol: str) -> float:
        """Get current market price for a trading pair"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"API error while fetching price for {symbol}: {str(e)}")
            raise APIError(f"Failed to get price: {str(e)}")

    def get_trading_history(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trading history for a specific symbol"""
        try:
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            return [{
                'symbol': trade['symbol'],
                'price': float(trade['price']),
                'quantity': float(trade['qty']),
                'time': datetime.fromtimestamp(trade['time'] / 1000),
                'side': trade['isBuyer'] and 'BUY' or 'SELL'
            } for trade in trades]
        except BinanceAPIException as e:
            logger.error(f"API error while fetching trades for {symbol}: {str(e)}")
            raise APIError(f"Failed to get trading history: {str(e)}")

    def get_ohlcv(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data as a pandas DataFrame"""
        klines = self.get_klines(symbol, interval, limit)
        
        # Convert to DataFrame format
        df = pd.DataFrame(klines)
        return df
        
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get kline/candlestick data for a symbol"""
        if self.simulation_mode:
            # Generate simulated kline data
            klines = []
            base_price = 65000.0 if 'BTC' in symbol else (3500.0 if 'ETH' in symbol else 100.0)
            
            # Start from current time and go backwards
            current_time = datetime.now()
            
            # Determine time delta based on interval
            if interval == '1m':
                time_delta = 60  # seconds
            elif interval == '5m':
                time_delta = 300
            elif interval == '15m':
                time_delta = 900
            elif interval == '1h':
                time_delta = 3600
            else:
                time_delta = 86400  # default to 1 day
            
            # Generate random klines
            price = base_price
            for i in range(limit):
                # Generate random price movement
                price_change = price * (random.random() * 0.02 - 0.01)  # -1% to +1%
                price += price_change
                
                # Calculate high and low
                high = price * (1 + random.random() * 0.005)
                low = price * (1 - random.random() * 0.005)
                
                # Calculate open based on previous close
                open_price = price - price_change
                
                # Create timestamp
                timestamp = current_time - timedelta(seconds=time_delta * (limit - i))
                
                klines.append({
                    'open_time': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': random.random() * 100,
                    'close_time': timestamp + timedelta(seconds=time_delta)
                })
            
            return klines
            
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            return [{
                'open_time': datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_time': datetime.fromtimestamp(kline[6] / 1000)
            } for kline in klines]
        except BinanceAPIException as e:
            logger.error(f"API error while fetching klines for {symbol}: {str(e)}")
            raise APIError(f"Failed to get klines: {str(e)}")

    def create_spot_order(
        self,
        symbol: str,
        side: str,
        quantity: Union[float, Decimal],
        price: Optional[Union[float, Decimal]] = None
    ) -> Dict[str, Any]:
        """Create a spot order (market or limit)"""
        # Validate inputs
        side = side.upper()
        if side not in ['BUY', 'SELL']:
            raise ValueError("side must be either 'BUY' or 'SELL'")
            
        if self.simulation_mode:
            # Create simulated order
            order_id = int(time.time() * 1000)  # Use timestamp as order ID
            order_type = 'LIMIT' if price else 'MARKET'
            
            # Get current price for market orders or use the specified price for limit orders
            if order_type == 'MARKET':
                current_price = 65000.0 if 'BTC' in symbol else (3500.0 if 'ETH' in symbol else 100.0)
            else:
                current_price = float(price)
                
            # Verificar fondos suficientes para la operación (spot)
            if side == 'BUY':
                quote_asset = symbol[3:]  # e.g., 'USDT' from 'BTCUSDT'
                quote_balance = self.get_account_balance(quote_asset).get(quote_asset, 0)
                order_cost = float(quantity) * current_price
                
                # Si no hay fondos suficientes, simular error de fondos insuficientes
                if order_cost > quote_balance:
                    error_msg = f"Fondos insuficientes para {symbol}: requiere {order_cost} {quote_asset}, disponible {quote_balance} {quote_asset}"
                    logger.error(error_msg)
                    raise APIError(error_msg)
                    
                # Actualizar balances simulados - restar USDT/moneda base utilizada
                if hasattr(self, 'bot') and hasattr(self.bot, 'config') and hasattr(self.bot.config, 'sim_initial_balance'):
                    if quote_asset in self.bot.config.sim_initial_balance:
                        # Restar el costo de la orden del balance simulado
                        self.bot.config.sim_initial_balance[quote_asset] -= order_cost
            
            elif side == 'SELL':
                # Para ventas, verificar que tenemos suficiente del activo a vender
                base_asset = symbol[:3]  # e.g., 'BTC' from 'BTCUSDT'
                base_balance = self.get_account_balance(base_asset).get(base_asset, 0)
                
                if float(quantity) > base_balance:
                    error_msg = f"Fondos insuficientes para {symbol}: requiere {quantity} {base_asset}, disponible {base_balance} {base_asset}"
                    logger.error(error_msg)
                    raise APIError(error_msg)
                    
                # Actualizar balances simulados - restar la cantidad del activo vendido
                if hasattr(self, 'bot') and hasattr(self.bot, 'config') and hasattr(self.bot.config, 'sim_initial_balance'):
                    if base_asset in self.bot.config.sim_initial_balance:
                        # Restar la cantidad vendida del balance simulado
                        self.bot.config.sim_initial_balance[base_asset] -= float(quantity)
                        
                    # Añadir el importe recibido en moneda base
                    quote_asset = symbol[3:]
                    if quote_asset in self.bot.config.sim_initial_balance:
                        self.bot.config.sim_initial_balance[quote_asset] += float(quantity) * current_price
            
            # Create simulated order response
            order = {
                'symbol': symbol,
                'orderId': order_id,
                'clientOrderId': f'sim_{order_id}',
                'transactTime': int(time.time() * 1000),
                'price': str(price) if price else '0.0',
                'origQty': str(quantity),
                'executedQty': str(quantity),
                'status': 'FILLED',
                'type': order_type,
                'side': side,
                'fills': [{
                    'price': str(current_price),
                    'qty': str(quantity),
                    'commission': '0.0',
                    'commissionAsset': 'BNB'
                }]
            }
            
            logger.info(f"Simulated {order_type} {side} order for {symbol}")
            return order
            
        try:
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'quantity': float(quantity),
            }

            if price:
                # Limit order
                order_params.update({
                    'type': 'LIMIT',
                    'price': float(price),
                    'timeInForce': 'GTC'  # Good Till Cancelled
                })
            else:
                # Market order
                order_params['type'] = 'MARKET'

            # Create the order
            order = self.client.create_order(**order_params)
            logger.info(f"Successfully created {order_params['type']} {side} order for {symbol}")
            return order

        except BinanceAPIException as e:
            error_msg = f"Failed to create order: {str(e)}"
            logger.error(error_msg)
            raise APIError(error_msg)
        except ValueError as e:
            error_msg = f"Invalid input: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)