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
from commission_config import (
    standard_fee_rate,
    bnb_discount_rate,
    use_bnb_for_fees,
    pairs_with_zero_fee
)

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
        
        # Initialize slippage parameter
        self.slippage_pct = getattr(config, 'slippage_pct', 0.0002)  # 0.02% default slippage
        
        # Apply commission rates from commission_config
        # Override with config value if provided (for backward compatibility)
        if hasattr(config, 'commission_rate'):
            self.commission_rate = config.commission_rate
            logger.info(f"Using commission rate from config: {self.commission_rate*100:.4f}%")
        else:
            # Determine fee rate based on commission_config settings
            self.use_bnb_for_fees = use_bnb_for_fees
            
            if self.use_bnb_for_fees:
                self.commission_rate = bnb_discount_rate
                logger.info(f"Using BNB discounted fee rate: {self.commission_rate*100:.4f}%")
            else:
                self.commission_rate = standard_fee_rate
                logger.info(f"Using standard fee rate: {self.commission_rate*100:.4f}%")
        
        logger.info(f"Commission rate: {self.commission_rate*100:.4f}%, "
                   f"slippage: {self.slippage_pct*100:.4f}%")
        
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
        # Normalize quantity according to symbol rules
        normalized_quantity = self.normalize_quantity(symbol, quantity)
        
        # Log the normalization for debugging
        logger.debug(f"Original quantity for {symbol}: {quantity}, normalized: {normalized_quantity}")
        
        # Validar los fondos disponibles (especialmente para compras)
        if side.upper() == 'BUY' and not self.simulation_mode:
            # Verificar fondos disponibles para operaciones reales (no simulación)
            quote_asset = symbol[3:]  # e.g., 'USDT' from 'BTCUSDT'
            quote_balance = self.get_account_balance(quote_asset).get(quote_asset, 0)
            
            # Obtener precio actual para mercado o usar el precio especificado para limit
            current_price = price if price else self.get_market_price(symbol)
            required_funds = float(normalized_quantity) * float(current_price)
            
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
            
            # Apply slippage to the price based on order side with a random component
            import random
            # Base slippage rate plus small random variation (0 to 20% of base rate)
            random_factor = 1.0 + random.uniform(0, 0.2)  
            effective_slippage = self.slippage_pct * random_factor
            
            slipped_price = current_price
            if side == 'BUY':
                # For buys, price slips upward (worse entry price)
                slipped_price = current_price * (1 + effective_slippage)
                logger.debug(f"Applied BUY slippage: {current_price} -> {slipped_price} " +
                            f"(+{effective_slippage*100:.4f}%, base: {self.slippage_pct*100:.4f}%)")
            else:
                # For sells, price slips downward (worse entry price)
                slipped_price = current_price * (1 - effective_slippage)
                logger.debug(f"Applied SELL slippage: {current_price} -> {slipped_price} " +
                            f"(-{effective_slippage*100:.4f}%, base: {self.slippage_pct*100:.4f}%)")
                
            # Determine if the symbol has zero fees or should use standard/discounted rates
            if symbol in pairs_with_zero_fee:
                fee_pct = 0.0
                logger.debug(f"Applied zero fee for {symbol} (promotional pair with no fees)")
            else:
                # Calculate commission (more realistic with tiered rates)
                # Base commission plus small additional cost for large orders
                volume_factor = 1.0
                
                # Apply slightly higher rates for very large or very small orders
                trade_value = quantity * slipped_price
                if trade_value > 100000:  # Very large orders get better rates
                    volume_factor = 0.9
                elif trade_value < 100:   # Very small orders get worse rates
                    volume_factor = 1.1
                
                # Use BNB discount if enabled
                base_fee_rate = self.commission_rate  
                effective_commission_rate = base_fee_rate * volume_factor
                fee_pct = effective_commission_rate
            
            # Calculate final commission amount
            trade_value = quantity * slipped_price
            commission_amount = trade_value * fee_pct
            
            logger.debug(f"Applied commission for {side} {symbol}: {commission_amount:.4f} " +
                        f"({fee_pct*100:.4f}%)")
            
            # Create mock main order with slippage and commission
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
                    'price': str(slipped_price),  # Price with slippage applied
                    'qty': str(quantity),
                    'commission': str(commission_amount),  # Real commission
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
            # Create main order using the normalized quantity
            main_order = self.create_spot_order(symbol, side, normalized_quantity, price)
            orders = {'main_order': main_order}

            if main_order['status'] == 'FILLED':
                opposite_side = 'SELL' if side == 'BUY' else 'BUY'

                # Create stop loss order with normalized quantity
                if stop_loss:
                    try:
                        sl_order = self.client.create_order(
                            symbol=symbol,
                            side=opposite_side,
                            type='STOP_LOSS_LIMIT',
                            quantity=normalized_quantity,
                            price=float(stop_loss),
                            stopPrice=float(stop_loss),
                            timeInForce='GTC'
                        )
                        orders['stop_loss_order'] = sl_order
                    except Exception as e:
                        logger.error(f"Failed to create stop loss order: {str(e)}")

                # Create take profit order with normalized quantity
                if take_profit:
                    try:
                        tp_order = self.client.create_order(
                            symbol=symbol,
                            side=opposite_side,
                            type='LIMIT',
                            quantity=normalized_quantity,
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

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading rules and precision information for a symbol
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Dict[str, Any]: Symbol information including precision rules
        """
        if self.simulation_mode:
            # Return simulated symbol info with reasonable precision values
            base_asset = symbol[:3] if len(symbol) >= 3 else symbol
            quote_asset = symbol[3:] if len(symbol) >= 6 else 'USDT'
            
            # Assign appropriate precision values based on symbol
            if base_asset == 'BTC':
                qty_precision = 6  # 0.000001 BTC minimum
            elif base_asset == 'ETH':
                qty_precision = 5  # 0.00001 ETH minimum
            else:
                qty_precision = 2  # 0.01 units minimum for other assets
                
            if quote_asset == 'USDT' or quote_asset == 'BUSD':
                price_precision = 2  # 0.01 USDT/BUSD precision
            else:
                price_precision = 8  # 0.00000001 precision for crypto/crypto pairs
            
            return {
                'symbol': symbol,
                'baseAsset': base_asset,
                'quoteAsset': quote_asset,
                'status': 'TRADING',
                'baseAssetPrecision': 8,
                'quotePrecision': 8,
                'quoteAssetPrecision': 8,
                'orderTypes': ['LIMIT', 'MARKET'],
                'filters': [
                    {
                        'filterType': 'LOT_SIZE',
                        'minQty': f"0.{'0' * (qty_precision - 1)}1",
                        'maxQty': '9000000.00000000',
                        'stepSize': f"0.{'0' * (qty_precision - 1)}1"
                    },
                    {
                        'filterType': 'PRICE_FILTER',
                        'minPrice': f"0.{'0' * (price_precision - 1)}1",
                        'maxPrice': '1000000.00000000',
                        'tickSize': f"0.{'0' * (price_precision - 1)}1"
                    },
                ]
            }
            
        try:
            # Get exchange info
            exchange_info = self.client.get_exchange_info()
            
            # Find symbol info
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return symbol_info
                    
            # If symbol not found
            raise APIError(f"Symbol {symbol} not found in exchange info")
            
        except BinanceAPIException as e:
            logger.error(f"API error while fetching symbol info for {symbol}: {str(e)}")
            raise APIError(f"Failed to get symbol info: {str(e)}")
    
    def normalize_quantity(self, symbol: str, quantity: Union[float, Decimal]) -> str:
        """
        Normalize quantity based on symbol's lot size filter
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            quantity (float|Decimal): Original quantity
            
        Returns:
            str: Normalized quantity as string with proper precision
        """
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            
            # Find lot size filter
            lot_size_filter = next(
                (f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'),
                None
            )
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                
                # Calculate decimal places needed
                if step_size < 1:
                    step_size_str = str(step_size).rstrip('0')
                    decimal_places = len(step_size_str.split('.')[-1])
                else:
                    decimal_places = 0
                
                # Truncate quantity to match step size
                quantity_float = float(quantity)
                normalized_quantity = int(quantity_float / step_size) * step_size
                
                # Format with correct precision
                return f"{{:.{decimal_places}f}}".format(normalized_quantity)
            
            # Default to 8 decimal places if no filter found
            return f"{float(quantity):.8f}"
            
        except Exception as e:
            logger.warning(f"Error normalizing quantity, using default precision (8): {str(e)}")
            # Fall back to 8 decimal places
            return f"{float(quantity):.8f}"
    
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
            
        # Normalize quantity to proper precision for this symbol
        normalized_quantity = self.normalize_quantity(symbol, quantity)
            
        if self.simulation_mode:
            # Create simulated order
            order_id = int(time.time() * 1000)  # Use timestamp as order ID
            order_type = 'LIMIT' if price else 'MARKET'
            
            # Get current price for market orders or use the specified price for limit orders
            if order_type == 'MARKET':
                current_price = 65000.0 if 'BTC' in symbol else (3500.0 if 'ETH' in symbol else 100.0)
            else:
                current_price = float(price)
                
            # Use normalized quantity
            normalized_quantity = self.normalize_quantity(symbol, quantity)
            logger.debug(f"Normalized quantity for {symbol}: {normalized_quantity} (from {quantity})")
            
            # Verificar fondos suficientes para la operación (spot)
            if side == 'BUY':
                quote_asset = symbol[3:]  # e.g., 'USDT' from 'BTCUSDT'
                quote_balance = self.get_account_balance(quote_asset).get(quote_asset, 0)
                order_cost = float(normalized_quantity) * current_price
                
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
                
                if float(normalized_quantity) > base_balance:
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
                'origQty': normalized_quantity,
                'executedQty': normalized_quantity,
                'status': 'FILLED',
                'type': order_type,
                'side': side,
                'fills': [{
                    'price': str(current_price),
                    'qty': normalized_quantity,
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
                'quantity': normalized_quantity,
            }

            if price:
                # Limit order
                # Normalize price based on symbol rules
                price_filter = next(
                    (f for f in self.get_symbol_info(symbol)['filters'] if f['filterType'] == 'PRICE_FILTER'),
                    None
                )
                
                if price_filter:
                    tick_size = float(price_filter['tickSize'])
                    if tick_size < 1:
                        tick_size_str = str(tick_size).rstrip('0')
                        price_decimal_places = len(tick_size_str.split('.')[-1])
                    else:
                        price_decimal_places = 0
                    
                    # Format price with correct precision
                    normalized_price = f"{{:.{price_decimal_places}f}}".format(float(price))
                else:
                    normalized_price = f"{float(price):.2f}"  # Default to 2 decimal places
                
                order_params.update({
                    'type': 'LIMIT',
                    'price': normalized_price,
                    'timeInForce': 'GTC'  # Good Till Cancelled
                })
            else:
                # Market order
                order_params['type'] = 'MARKET'

            # Log normalized values for debugging
            logger.debug(f"Normalized order parameters for {symbol}: {order_params}")
            
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