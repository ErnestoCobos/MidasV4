import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os
import logging

logger = logging.getLogger('Visualization')

class TradingVisualizer:
    """
    Class for creating visual representations of trading data and performance
    """
    
    def __init__(self, output_dir='charts'):
        """Initialize the visualizer"""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_price_chart(self, klines, symbol, trades=None):
        """
        Plot price chart with indicators and trades
        
        Args:
            klines: List of kline data from Binance client
            symbol: Trading pair symbol
            trades: Optional list of trades to mark on the chart
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([{
                'open_time': k['open_time'],
                'close': float(k['close']),
                'high': float(k['high']),
                'low': float(k['low']),
                'open': float(k['open']),
                'volume': float(k['volume'])
            } for k in klines])
            
            # Calculate indicators
            df['sma_7'] = df['close'].rolling(window=7).mean()
            df['sma_25'] = df['close'].rolling(window=25).mean()
            
            # Bollinger Bands
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['std_20'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
            df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price and indicators
            ax1.plot(df.index, df['close'], label='Precio', color='black')
            ax1.plot(df.index, df['sma_7'], label='SMA 7', color='blue', alpha=0.7)
            ax1.plot(df.index, df['sma_25'], label='SMA 25', color='red', alpha=0.7)
            ax1.plot(df.index, df['bb_upper'], label='BB Superior', color='green', alpha=0.3)
            ax1.plot(df.index, df['bb_lower'], label='BB Inferior', color='green', alpha=0.3)
            ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='green', alpha=0.1)
            
            # Plot trades if available
            if trades:
                for trade in trades:
                    # Find closest index to trade time
                    entry_time = trade['time_opened']
                    exit_time = trade['time_closed']
                    
                    # Find closest indices
                    try:
                        entry_idx = df['open_time'].sub(entry_time).abs().idxmin()
                        exit_idx = df['open_time'].sub(exit_time).abs().idxmin()
                        
                        # Plot entry and exit points
                        if trade['side'] == 'BUY':
                            ax1.scatter(entry_idx, trade['entry_price'], marker='^', color='green', s=100)
                            ax1.scatter(exit_idx, trade['exit_price'], marker='v', color='red' if trade['profit_loss'] < 0 else 'green', s=100)
                        else:  # SELL
                            ax1.scatter(entry_idx, trade['entry_price'], marker='v', color='red', s=100)
                            ax1.scatter(exit_idx, trade['exit_price'], marker='^', color='red' if trade['profit_loss'] < 0 else 'green', s=100)
                    except Exception as e:
                        logger.warning(f"Couldn't plot trade: {str(e)}")
            
            # Plot volume
            ax2.bar(df.index, df['volume'], color='blue', alpha=0.5)
            ax2.set_ylabel('Volumen')
            
            # Set labels and title
            ax1.set_title(f'Análisis de {symbol}')
            ax1.set_ylabel('Precio')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Format x-axis to show dates
            last_idx = df.index[-1]
            plt.tight_layout()
            
            # Save to file
            filename = f"{self.output_dir}/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating price chart: {str(e)}")
            return None
    
    def plot_performance_chart(self, trades_history, title="Desempeño de Trading"):
        """
        Plot performance chart showing cumulative P/L
        
        Args:
            trades_history: List of completed trades
            title: Chart title
        """
        try:
            if not trades_history:
                return None
                
            # Create DataFrame with trade data
            df = pd.DataFrame([{
                'time': trade['time_closed'],
                'profit_loss': trade['profit_loss'],
                'symbol': trade['symbol']
            } for trade in trades_history])
            
            # Sort by time
            df = df.sort_values('time')
            
            # Calculate cumulative P/L
            df['cumulative_pnl'] = df['profit_loss'].cumsum()
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot cumulative P/L
            ax.plot(df['time'], df['cumulative_pnl'], label='P/L Acumulado', color='blue')
            
            # Plot individual trades as points
            colors = ['green' if pl >= 0 else 'red' for pl in df['profit_loss']]
            ax.scatter(df['time'], df['cumulative_pnl'], c=colors, s=30)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Set labels and title
            ax.set_title(title)
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Ganancia/Pérdida')
            ax.grid(True, alpha=0.3)
            
            # Format y-axis to show values with precision
            plt.tight_layout()
            
            # Save to file
            filename = f"{self.output_dir}/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {str(e)}")
            return None
            
    def plot_win_loss_distribution(self, trades_history):
        """
        Plot win/loss distribution chart
        
        Args:
            trades_history: List of completed trades
        """
        try:
            if not trades_history:
                return None
                
            # Extract profit/loss values
            profits = [t['profit_loss'] for t in trades_history if t['profit_loss'] > 0]
            losses = [t['profit_loss'] for t in trades_history if t['profit_loss'] <= 0]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot histograms
            if profits:
                ax.hist(profits, bins=10, alpha=0.7, color='green', label='Ganancias')
            if losses:
                ax.hist(losses, bins=10, alpha=0.7, color='red', label='Pérdidas')
            
            # Set labels and title
            ax.set_title('Distribución de Ganancias y Pérdidas')
            ax.set_xlabel('Ganancia/Pérdida')
            ax.set_ylabel('Frecuencia')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Save to file
            filename = f"{self.output_dir}/distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error creating win/loss distribution chart: {str(e)}")
            return None