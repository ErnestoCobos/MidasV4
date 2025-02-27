import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data_buffer = {}  # Symbol -> DataFrame
        
    def process_klines(self, klines: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process raw kline data from Binance"""
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(klines)
        
        # Ensure all columns have the correct types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        # Ensure time columns are datetime
        for col in ['open_time', 'close_time']:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        
        # Sort by time
        if 'open_time' in df.columns:
            df = df.sort_values('open_time')
        
        return df
    
    def process_ticker(self, ticker: Dict[str, Any]) -> Dict[str, Any]:
        """Process ticker data from Binance"""
        processed_ticker = {}
        
        # Extract price
        if 'c' in ticker:
            processed_ticker['price'] = float(ticker['c'])
        elif 'price' in ticker:
            processed_ticker['price'] = float(ticker['price'])
        
        # Extract symbol
        if 's' in ticker:
            processed_ticker['symbol'] = ticker['s']
        elif 'symbol' in ticker:
            processed_ticker['symbol'] = ticker['symbol']
        
        # Extract timestamp
        if 'E' in ticker:
            processed_ticker['timestamp'] = datetime.fromtimestamp(ticker['E'] / 1000)
        else:
            processed_ticker['timestamp'] = datetime.now()
        
        # Extract volume if available
        if 'v' in ticker:
            processed_ticker['volume'] = float(ticker['v'])
        
        return processed_ticker
    
    def update_data_buffer(self, symbol: str, new_data: Dict[str, Any]):
        """Update the data buffer with new market data"""
        max_buffer_size = self.config.max_buffer_size if hasattr(self.config, 'max_buffer_size') else 1000
        
        # Create new DataFrame with single row
        new_row = pd.DataFrame([new_data])
        
        # Initialize buffer for symbol if it doesn't exist
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = new_row
            return
        
        # Append new data
        self.data_buffer[symbol] = pd.concat([self.data_buffer[symbol], new_row])
        
        # Trim buffer if it exceeds max size
        if len(self.data_buffer[symbol]) > max_buffer_size:
            self.data_buffer[symbol] = self.data_buffer[symbol].iloc[-max_buffer_size:]
    
    def get_latest_data(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """Get the latest data for a symbol from the buffer"""
        if symbol not in self.data_buffer:
            return pd.DataFrame()
        
        # Return the requested number of rows
        return self.data_buffer[symbol].iloc[-lookback:]
    
    def aggregate_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate tick data to a specific timeframe"""
        # Ensure DataFrame has timestamp column
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")
        
        # Set timestamp as index
        df_indexed = df.set_index('timestamp')
        
        # Determine the resampling rule based on timeframe
        if timeframe == '1m':
            rule = '1min'
        elif timeframe == '5m':
            rule = '5min'
        elif timeframe == '15m':
            rule = '15min'
        elif timeframe == '1h':
            rule = '1H'
        elif timeframe == '4h':
            rule = '4H'
        elif timeframe == '1d':
            rule = '1D'
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Resample the data
        resampled = df_indexed.resample(rule).agg({
            'price': 'ohlc',
            'volume': 'sum'
        })
        
        # Flatten multi-level columns
        resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
        
        # Rename columns to match expected format
        resampled = resampled.rename(columns={
            'price_open': 'open',
            'price_high': 'high',
            'price_low': 'low',
            'price_close': 'close',
            'volume_sum': 'volume'
        })
        
        # Reset index to make timestamp a column
        resampled = resampled.reset_index()
        resampled = resampled.rename(columns={'timestamp': 'open_time'})
        
        # Add close_time column (end of the interval)
        if timeframe == '1m':
            resampled['close_time'] = resampled['open_time'] + timedelta(minutes=1)
        elif timeframe == '5m':
            resampled['close_time'] = resampled['open_time'] + timedelta(minutes=5)
        elif timeframe == '15m':
            resampled['close_time'] = resampled['open_time'] + timedelta(minutes=15)
        elif timeframe == '1h':
            resampled['close_time'] = resampled['open_time'] + timedelta(hours=1)
        elif timeframe == '4h':
            resampled['close_time'] = resampled['open_time'] + timedelta(hours=4)
        elif timeframe == '1d':
            resampled['close_time'] = resampled['open_time'] + timedelta(days=1)
        
        return resampled