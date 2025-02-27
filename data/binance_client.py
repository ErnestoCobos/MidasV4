import asyncio
import json
import time
import logging
import aiohttp
import websockets
import pandas as pd
import numpy as np
import random
from decimal import Decimal
from typing import Dict, List, Any, Callable, Optional, Union
from datetime import datetime, timedelta
from core.constants import BINANCE_API_URL, BINANCE_API_TESTNET_URL, BINANCE_WS_URL, BINANCE_WS_TESTNET_URL

class BinanceClient:
    """Optimized Binance client with WebSocket support and low latency"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        self.api_secret = config.api_secret
        self.testnet = config.use_testnet
        self.simulation_mode = config.simulation_mode
        
        # Set base URLs based on testnet flag
        self.base_url = BINANCE_API_TESTNET_URL if self.testnet else BINANCE_API_URL
        self.ws_url = BINANCE_WS_TESTNET_URL if self.testnet else BINANCE_WS_URL
        
        # HTTP session for connection pooling
        self.http_session = None
        
        # WebSocket connections
        self.ws_connections = {}
        self.callbacks = {}
        
        # Cache for latest prices
        self.price_cache = {}
        
        # Logger
        self.logger = logging.getLogger('BinanceClient')
    
    async def initialize(self):
        """Initialize resources"""
        if self.http_session is None and not self.simulation_mode:
            timeout = aiohttp.ClientTimeout(total=10, connect=2, sock_connect=2, sock_read=5)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
            self.logger.info("HTTP session initialized")
    
    async def close(self):
        """Clean up resources"""
        if self.simulation_mode:
            return
            
        # Close all websocket connections
        for stream_id, ws in self.ws_connections.items():
            if ws and not ws.closed:
                await ws.close()
                self.logger.info(f"WebSocket connection closed: {stream_id}")
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
            self.logger.info("HTTP session closed")
    
    async def _make_request(self, method, endpoint, params=None, signed=False, retry_count=3):
        """Make a request to the Binance API with retry logic"""
        if self.simulation_mode:
            return self._simulate_response(method, endpoint, params)
            
        url = f"{self.base_url}{endpoint}"
        
        # Ensure HTTP session is initialized
        await self.initialize()
        
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        if signed:
            # Add timestamp and signature (simplified - in production implement proper signing)
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            # In a real implementation, calculate HMAC signature here
        
        retry_delay = 0.5
        last_error = None
        
        for attempt in range(retry_count):
            try:
                start_time = time.monotonic()
                
                if method == 'GET':
                    async with self.http_session.get(url, params=params, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()
                elif method == 'POST':
                    async with self.http_session.post(url, params=params, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()
                elif method == 'DELETE':
                    async with self.http_session.delete(url, params=params, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Log API latency
                latency = (time.monotonic() - start_time) * 1000  # ms
                self.logger.debug(f"{method} {endpoint} completed in {latency:.2f}ms")
                
                return data
                
            except aiohttp.ClientResponseError as e:
                self.logger.warning(f"API request failed (attempt {attempt+1}/{retry_count}): {e.status} {e.message}")
                last_error = e
                if e.status in [429, 418]:  # Rate limit errors
                    retry_delay = max(retry_delay * 2, 10)  # Exponential backoff, max 10s
                elif e.status in [502, 503, 504]:  # Server errors
                    retry_delay = retry_delay * 2  # Exponential backoff for server errors
                else:
                    break  # Don't retry other errors
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.warning(f"Request error (attempt {attempt+1}/{retry_count}): {str(e)}")
                last_error = e
                retry_delay = retry_delay * 2  # Exponential backoff
            
            # Wait before retrying
            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay)
        
        # If we get here, all retries failed
        self.logger.error(f"API request failed after {retry_count} attempts: {str(last_error)}")
        raise last_error
    
    def _simulate_response(self, method, endpoint, params):
        """Generate simulated responses for simulation mode"""
        self.logger.debug(f"Simulating {method} {endpoint} request")
        
        # Simulate get_klines
        if endpoint == '/api/v3/klines':
            # If use_real_market_data is enabled, try to get real data from Binance
            if hasattr(self.config, 'use_real_market_data') and self.config.use_real_market_data:
                try:
                    # Use a temporary HTTP session to get real market data
                    self.logger.info("Using real market data for simulation")
                    
                    symbol = params.get('symbol', 'BTCUSDT')
                    interval = params.get('interval', '1m')
                    limit = int(params.get('limit', 100))
                    
                    # Create a temporary session
                    import requests
                    url = f"{self.base_url}/api/v3/klines"
                    response = requests.get(url, params={
                        'symbol': symbol,
                        'interval': interval,
                        'limit': limit
                    }, timeout=10)
                    
                    if response.status_code == 200:
                        self.logger.debug("Successfully fetched real klines data")
                        return response.json()
                    else:
                        self.logger.warning(f"Failed to fetch real klines data, status: {response.status_code}. Falling back to simulation.")
                except Exception as e:
                    self.logger.warning(f"Error fetching real klines data: {str(e)}. Falling back to simulation.")
            
            # If real data fetch failed or is disabled, generate simulated klines
            symbol = params.get('symbol', 'BTCUSDT')
            interval = params.get('interval', '1m')
            limit = int(params.get('limit', 100))
            
            # Generate simulated klines
            base_price = 65000.0 if 'BTC' in symbol else (3500.0 if 'ETH' in symbol else 100.0)
            klines = []
            
            # Start from current time and go backwards
            end_time = datetime.now()
            
            # Determine time delta based on interval
            if interval == '1m':
                delta = timedelta(minutes=1)
            elif interval == '5m':
                delta = timedelta(minutes=5)
            elif interval == '15m':
                delta = timedelta(minutes=15)
            elif interval == '1h':
                delta = timedelta(hours=1)
            else:
                delta = timedelta(days=1)
            
            # Generate klines with random walk
            current_price = base_price
            for i in range(limit):
                # Calculate timestamp for this candle
                timestamp = int((end_time - (delta * i)).timestamp() * 1000)
                close_time = timestamp + int(delta.total_seconds() * 1000) - 1
                
                # Add some randomness to price
                change_pct = (random.random() - 0.5) * 0.01  # -0.5% to +0.5%
                current_price = current_price * (1 + change_pct)
                
                # Calculate OHLC
                open_price = current_price * (1 - random.random() * 0.002)
                high_price = max(current_price, open_price) * (1 + random.random() * 0.003)
                low_price = min(current_price, open_price) * (1 - random.random() * 0.003)
                
                # Generate some random volume
                volume = random.random() * 10 + 1  # 1-11 volume
                
                # Create kline in Binance format
                kline = [
                    timestamp,                       # Open time
                    str(open_price),                 # Open
                    str(high_price),                 # High
                    str(low_price),                  # Low
                    str(current_price),              # Close
                    str(volume),                     # Volume
                    close_time,                      # Close time
                    str(volume * current_price),     # Quote asset volume
                    10,                              # Number of trades
                    str(volume * 0.6),               # Taker buy base asset volume
                    str(volume * 0.6 * current_price), # Taker buy quote asset volume
                    "0"                              # Ignore
                ]
                
                klines.append(kline)
            
            # Return in reverse order (oldest first)
            return list(reversed(klines))
        
        # Simulate get_ticker
        elif endpoint == '/api/v3/ticker/price':
            # If use_real_market_data is enabled, try to get real price data from Binance
            if hasattr(self.config, 'use_real_market_data') and self.config.use_real_market_data:
                try:
                    symbol = params.get('symbol', None)
                    # Use a temporary HTTP session to get real market data
                    import requests
                    url = f"{self.base_url}/api/v3/ticker/price"
                    
                    if symbol:
                        response = requests.get(url, params={'symbol': symbol}, timeout=5)
                    else:
                        response = requests.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Update the price cache with the real prices
                        if isinstance(data, list):
                            for ticker in data:
                                self.price_cache[ticker['symbol']] = float(ticker['price'])
                        else:
                            self.price_cache[data['symbol']] = float(data['price'])
                            
                        # Return the real data
                        return data
                    else:
                        self.logger.warning(f"Failed to fetch real ticker data, status: {response.status_code}. Falling back to simulation.")
                except Exception as e:
                    self.logger.warning(f"Error fetching real ticker data: {str(e)}. Falling back to simulation.")
                    
            # If real data fetch failed or is disabled, generate simulated tickers
            symbol = params.get('symbol', None)
            
            if symbol:
                # Return single ticker
                price = self.price_cache.get(symbol, None)
                if price is None:
                    price = 65000.0 if 'BTC' in symbol else (3500.0 if 'ETH' in symbol else 100.0)
                    self.price_cache[symbol] = price
                
                # Add some random change
                change_pct = (random.random() - 0.5) * 0.001  # -0.05% to +0.05%
                price = price * (1 + change_pct)
                self.price_cache[symbol] = price
                
                return {'symbol': symbol, 'price': str(price)}
            else:
                # Return all tickers
                symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
                result = []
                
                for sym in symbols:
                    price = self.price_cache.get(sym, None)
                    if price is None:
                        price = 65000.0 if 'BTC' in sym else (3500.0 if 'ETH' in sym else 100.0)
                        self.price_cache[sym] = price
                    
                    # Add some random change
                    change_pct = (random.random() - 0.5) * 0.001  # -0.05% to +0.05%
                    price = price * (1 + change_pct)
                    self.price_cache[sym] = price
                    
                    result.append({'symbol': sym, 'price': str(price)})
                
                return result
        
        # Simulate get_account
        elif endpoint == '/api/v3/account':
            # Use configured initial balances if available
            balances = []
            if hasattr(self.config, 'sim_initial_balance') and self.config.sim_initial_balance:
                for asset, amount in self.config.sim_initial_balance.items():
                    balances.append({
                        'asset': asset,
                        'free': str(amount),
                        'locked': '0.0'
                    })
            else:
                # Default balances
                balances = [
                    {'asset': 'BTC', 'free': '0.1', 'locked': '0.0'},
                    {'asset': 'ETH', 'free': '2.0', 'locked': '0.0'},
                    {'asset': 'USDT', 'free': '10000.0', 'locked': '0.0'},
                    {'asset': 'BNB', 'free': '10.0', 'locked': '0.0'}
                ]
            
            return {
                'makerCommission': 10,
                'takerCommission': 10,
                'buyerCommission': 0,
                'sellerCommission': 0,
                'canTrade': True,
                'canWithdraw': True,
                'canDeposit': True,
                'updateTime': int(time.time() * 1000),
                'accountType': 'SPOT',
                'balances': balances
            }
        
        # Simulate create_order
        elif endpoint == '/api/v3/order':
            symbol = params.get('symbol', 'BTCUSDT')
            side = params.get('side', 'BUY')
            order_type = params.get('type', 'LIMIT')
            quantity = float(params.get('quantity', 0.1))
            price = params.get('price', None)
            
            # Generate order ID
            order_id = int(time.time() * 1000000 + random.randint(1000, 9999))
            
            # Get current price
            current_price = self.price_cache.get(symbol, None)
            if current_price is None:
                current_price = 65000.0 if 'BTC' in symbol else (3500.0 if 'ETH' in symbol else 100.0)
                self.price_cache[symbol] = current_price
            
            # Use specified price or current price
            if price is None:
                price = current_price
            else:
                price = float(price)
            
            # Create order response
            order = {
                'symbol': symbol,
                'orderId': order_id,
                'orderListId': -1,
                'clientOrderId': f'simulated_{order_id}',
                'transactTime': int(time.time() * 1000),
                'price': str(price),
                'origQty': str(quantity),
                'executedQty': str(quantity),
                'cummulativeQuoteQty': str(quantity * price),
                'status': 'FILLED',
                'timeInForce': 'GTC',
                'type': order_type,
                'side': side,
                'fills': [
                    {
                        'price': str(price),
                        'qty': str(quantity),
                        'commission': '0',
                        'commissionAsset': 'BNB',
                        'tradeId': order_id + 1000
                    }
                ]
            }
            
            return order
        
        # Default response
        return {'error': 'Endpoint not simulated', 'endpoint': endpoint}
    
    async def start_websocket(self, stream_name, callback):
        """Start a websocket connection for a specific stream"""
        if self.simulation_mode:
            self.logger.info(f"Starting simulated WebSocket for {stream_name}")
            # Start a task to simulate WebSocket data
            asyncio.create_task(self._simulate_websocket(stream_name, callback))
            return
            
        ws_url = f"{self.ws_url}/{stream_name}"
        self.logger.info(f"Connecting to WebSocket: {ws_url}")
        
        # Store the callback
        self.callbacks[stream_name] = callback
        
        # Function to handle WebSocket connection
        async def _websocket_handler():
            while True:
                try:
                    async with websockets.connect(ws_url) as websocket:
                        self.ws_connections[stream_name] = websocket
                        self.logger.info(f"WebSocket connected: {stream_name}")
                        
                        async for message in websocket:
                            # Parse JSON message
                            data = json.loads(message)
                            # Process message
                            await callback(data)
                            
                except websockets.ConnectionClosed:
                    self.logger.warning(f"WebSocket connection closed: {stream_name}. Reconnecting...")
                    await asyncio.sleep(1)  # Wait before reconnecting
                except Exception as e:
                    self.logger.error(f"WebSocket error: {str(e)}. Reconnecting...")
                    await asyncio.sleep(5)  # Longer wait on error
        
        # Start the WebSocket handler in a background task
        asyncio.create_task(_websocket_handler())
    
    async def _simulate_websocket(self, stream_name, callback):
        """Simulate WebSocket data for simulation mode"""
        # Parse stream name to extract symbol and stream type
        parts = stream_name.split('@')
        if len(parts) != 2:
            self.logger.error(f"Invalid stream name format: {stream_name}")
            return
            
        symbol = parts[0].upper()
        stream_type = parts[1]
        
        # Get base price for symbol
        base_price = self.price_cache.get(symbol, None)
        if base_price is None:
            base_price = 65000.0 if 'BTC' in symbol else (3500.0 if 'ETH' in symbol else 100.0)
            self.price_cache[symbol] = base_price
        
        # Flag to check if we're using real market data
        use_real_data = hasattr(self.config, 'use_real_market_data') and self.config.use_real_market_data
        
        # Function to generate random price change
        def get_new_price(old_price):
            change_pct = (random.random() - 0.5) * 0.001  # -0.05% to +0.05%
            return old_price * (1 + change_pct)
        
        # Function to fetch real price data from Binance
        async def fetch_real_price(symbol_to_fetch):
            try:
                import aiohttp
                url = f"{self.base_url}/api/v3/ticker/price"
                params = {'symbol': symbol_to_fetch}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            return float(data['price'])
                        else:
                            self.logger.warning(f"Failed to fetch real price for {symbol_to_fetch}, status: {response.status}")
                            return None
            except Exception as e:
                self.logger.warning(f"Error fetching real price for {symbol_to_fetch}: {str(e)}")
                return None
        
        # Simulate different stream types
        while True:
            try:
                # Update price - either fetch real data or generate random change
                if use_real_data:
                    real_price = await fetch_real_price(symbol)
                    if real_price is not None:
                        price = real_price
                        self.price_cache[symbol] = price
                    else:
                        # Fall back to simulated price if real price fetch failed
                        price = get_new_price(self.price_cache[symbol])
                        self.price_cache[symbol] = price
                else:
                    # Use simulated price
                    price = get_new_price(self.price_cache[symbol])
                    self.price_cache[symbol] = price
                
                if stream_type == 'ticker':
                    # Simulate ticker data
                    ticker_data = {
                        'e': '24hrTicker',
                        'E': int(time.time() * 1000),
                        's': symbol,
                        'p': str(price * 0.01),  # 24h price change
                        'P': str(1.0),  # 24h price change percent
                        'w': str(price * 0.995),  # 24h weighted average price
                        'c': str(price),  # Last price
                        'Q': str(0.1),  # Last quantity
                        'o': str(price * 0.99),  # Open price
                        'h': str(price * 1.01),  # High price
                        'l': str(price * 0.98),  # Low price
                        'v': str(1000 + random.random() * 500),  # Total traded base asset volume
                        'q': str(price * 1000 + random.random() * 500 * price),  # Total traded quote asset volume
                        'O': int((time.time() - 86400) * 1000),  # Statistics open time
                        'C': int(time.time() * 1000),  # Statistics close time
                        'F': 1000,  # First trade ID
                        'L': 2000,  # Last trade ID
                        'n': 1000  # Total number of trades
                    }
                    await callback(ticker_data)
                
                elif stream_type.startswith('kline_'):
                    # Extract interval
                    interval = stream_type.split('_')[1]
                    
                    # Calculate open time based on interval
                    now = datetime.now()
                    if interval == '1m':
                        open_time = now.replace(second=0, microsecond=0)
                        close_time = open_time + timedelta(minutes=1)
                    elif interval == '5m':
                        minutes = now.minute - (now.minute % 5)
                        open_time = now.replace(minute=minutes, second=0, microsecond=0)
                        close_time = open_time + timedelta(minutes=5)
                    elif interval == '15m':
                        minutes = now.minute - (now.minute % 15)
                        open_time = now.replace(minute=minutes, second=0, microsecond=0)
                        close_time = open_time + timedelta(minutes=15)
                    elif interval == '1h':
                        open_time = now.replace(minute=0, second=0, microsecond=0)
                        close_time = open_time + timedelta(hours=1)
                    else:
                        open_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                        close_time = open_time + timedelta(days=1)
                    
                    # Simulate kline data
                    open_price = price * 0.998
                    high_price = price * 1.002
                    low_price = price * 0.997
                    
                    kline_data = {
                        'e': 'kline',
                        'E': int(time.time() * 1000),
                        's': symbol,
                        'k': {
                            't': int(open_time.timestamp() * 1000),  # Kline open time
                            'T': int(close_time.timestamp() * 1000),  # Kline close time
                            's': symbol,  # Symbol
                            'i': interval,  # Interval
                            'f': 1000,  # First trade ID
                            'L': 2000,  # Last trade ID
                            'o': str(open_price),  # Open price
                            'c': str(price),  # Close price
                            'h': str(high_price),  # High price
                            'l': str(low_price),  # Low price
                            'v': str(1000 + random.random() * 500),  # Base asset volume
                            'n': 1000,  # Number of trades
                            'x': False,  # Is this kline closed?
                            'q': str(price * 1000 + random.random() * 500 * price),  # Quote asset volume
                            'V': str(500 + random.random() * 250),  # Taker buy base asset volume
                            'Q': str(price * 500 + random.random() * 250 * price),  # Taker buy quote asset volume
                            'B': '0'  # Ignore
                        }
                    }
                    await callback(kline_data)
                
                # Sleep before next update
                # Ticker updates faster than klines
                if stream_type == 'ticker':
                    await asyncio.sleep(0.5)  # 2 updates per second
                else:
                    await asyncio.sleep(1)  # 1 update per second
                    
            except Exception as e:
                self.logger.error(f"Error in simulated WebSocket: {str(e)}")
                await asyncio.sleep(5)
    
    async def subscribe_ticker(self, symbol, callback):
        """Subscribe to real-time ticker updates for a symbol"""
        stream_name = f"{symbol.lower()}@ticker"
        await self.start_websocket(stream_name, callback)
    
    async def subscribe_klines(self, symbol, interval, callback):
        """Subscribe to real-time kline/candlestick updates"""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        await self.start_websocket(stream_name, callback)
    
    async def subscribe_depth(self, symbol, callback, level=10):
        """Subscribe to order book updates"""
        stream_name = f"{symbol.lower()}@depth{level}"
        await self.start_websocket(stream_name, callback)
    
    async def subscribe_trades(self, symbol, callback):
        """Subscribe to real-time trade updates"""
        stream_name = f"{symbol.lower()}@trade"
        await self.start_websocket(stream_name, callback)
    
    async def get_klines_async(self, symbol, interval, limit=100):
        """Get kline/candlestick data asynchronously"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        data = await self._make_request('GET', '/api/v3/klines', params=params)
        
        # Convert raw data to a more usable format
        klines = []
        for kline in data:
            klines.append({
                'open_time': datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_time': datetime.fromtimestamp(kline[6] / 1000)
            })
        
        return klines
    
    async def get_symbol_ticker_async(self, symbol=None):
        """Get latest price for a symbol or all symbols"""
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        data = await self._make_request('GET', '/api/v3/ticker/price', params=params)
        
        # Update cache with the latest prices
        if isinstance(data, list):
            for ticker in data:
                self.price_cache[ticker['symbol']] = float(ticker['price'])
            return data
        else:
            self.price_cache[data['symbol']] = float(data['price'])
            return data
    
    async def create_order_async(self, **params):
        """Create an order asynchronously"""
        return await self._make_request('POST', '/api/v3/order', params=params, signed=True)
    
    async def get_account_async(self):
        """Get account information asynchronously"""
        return await self._make_request('GET', '/api/v3/account', signed=True)
    
    async def get_open_orders_async(self, symbol=None):
        """Get open orders asynchronously"""
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        return await self._make_request('GET', '/api/v3/openOrders', params=params, signed=True)
    
    async def cancel_order_async(self, symbol, order_id=None, orig_client_order_id=None):
        """Cancel an order asynchronously"""
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Either order_id or orig_client_order_id must be provided")
            
        return await self._make_request('DELETE', '/api/v3/order', params=params, signed=True)
    
    # Synchronous methods for backwards compatibility
    def get_klines(self, symbol, interval, limit=100):
        """Get kline/candlestick data (sync)"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_klines_async(symbol, interval, limit))
    
    def get_symbol_ticker(self, symbol=None):
        """Get latest price for a symbol or all symbols (sync)"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_symbol_ticker_async(symbol))
    
    def create_order(self, **params):
        """Create an order (sync)"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.create_order_async(**params))
    
    def get_account(self):
        """Get account information (sync)"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_account_async())
    
    def get_open_orders(self, symbol=None):
        """Get open orders (sync)"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_open_orders_async(symbol))
    
    def cancel_order(self, symbol, order_id=None, orig_client_order_id=None):
        """Cancel an order (sync)"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.cancel_order_async(symbol, order_id, orig_client_order_id))
    
    def calculate_indicators(self, symbol: str, interval: str = '1m', limit: int = 100) -> Dict[str, Any]:
        """
        Calculate technical indicators for a symbol
        """
        try:
            # Get kline data
            klines = self.get_klines(symbol, interval, limit)
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(klines)
            
            # Calculate indicators
            indicators = {}
            
            # Simple Moving Averages
            for period in [7, 25]:
                indicators[f'sma_{period}'] = df['close'].rolling(window=period).mean().iloc[-1]
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (2 * std_20)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (2 * std_20)).iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Volatility
            indicators['volatility_14'] = df['close'].pct_change().rolling(window=14).std().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['current_volume'] = df['volume'].iloc[-1]
            
            # Add current price
            indicators['close'] = df['close'].iloc[-1]
            indicators['price'] = df['close'].iloc[-1]
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            raise e