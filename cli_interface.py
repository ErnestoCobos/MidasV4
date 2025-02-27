#!/usr/bin/env python3
import os
import argparse
import logging
import threading
import time
import json
import asyncio
from datetime import datetime, timedelta
import signal
import curses
import numpy as np
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich import box

from config import Config
from bot import ScalpingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cli_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CLI_Interface')

# Global variables
console = Console()
bot = None
stop_event = threading.Event()
update_interval = 1.0  # seconds
current_view = "dashboard"  # dashboard, performance, trades, chart
current_symbol = None

# ASCII Art Logo
LOGO = """
███╗   ███╗██╗██████╗  █████╗ ███████╗    ██╗   ██╗██╗  ██╗
████╗ ████║██║██╔══██╗██╔══██╗██╔════╝    ██║   ██║██║  ██║
██╔████╔██║██║██║  ██║███████║███████╗    ██║   ██║███████║
██║╚██╔╝██║██║██║  ██║██╔══██║╚════██║    ╚██╗ ██╔╝╚════██║
██║ ╚═╝ ██║██║██████╔╝██║  ██║███████║     ╚████╔╝      ██║
╚═╝     ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝╚══════╝      ╚═══╝       ╚═╝
                                                              
🚀 SCALPING BOT PARA BINANCE 🚀
"""

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Midas Scalping Bot v4 - CLI')
    
    parser.add_argument('--config', type=str, default=None,
                      help='Path to JSON configuration file')
    
    parser.add_argument('--testnet', action='store_true',
                      help='Use Binance testnet instead of real exchange')
    
    parser.add_argument('--symbols', type=str, default=None,
                      help='Comma-separated list of trading pairs (e.g., "BTCUSDT,ETHUSDT")')
    
    parser.add_argument('--simulate', action='store_true',
                      help='Run in simulation mode without connecting to exchange API')
                      
    parser.add_argument('--real-data', action='store_true',
                      help='Use real market data from Binance API in simulation mode')
                      
    parser.add_argument('--sim-balance', type=str, default=None,
                      help='Initial balance for simulation (format: "USDT:10000,BTC:0.5,ETH:5")')
    
    # Model-related arguments
    parser.add_argument('--model', type=str, default=None,
                      help='Model type to use (xgboost, lstm, indicator)')
    
    parser.add_argument('--no-gpu', action='store_true',
                      help='Disable GPU acceleration even if available')
    
    return parser.parse_args()

def load_config_from_file(file_path):
    """Load configuration from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create Config object from dict
        config = Config(
            api_key=config_dict.get('api_key', ''),
            api_secret=config_dict.get('api_secret', ''),
            symbols=config_dict.get('symbols', ['BTCUSDT']),
            max_open_trades=config_dict.get('max_open_trades', 3),
            max_capital_risk_percent=config_dict.get('max_capital_risk_percent', 2.0),
            stop_loss_percent=config_dict.get('stop_loss_percent', 0.5),
            take_profit_percent=config_dict.get('take_profit_percent', 1.0),
        )
        
        return config
    
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {str(e)}")
        raise

def setup_bot(args):
    """Set up and initialize the bot with given arguments"""
    global bot
    
    try:
        # Load configuration
        if args.config:
            config = load_config_from_file(args.config)
        else:
            config = Config.from_env()
        
        # Override with command line arguments
        if args.symbols:
            config.symbols = args.symbols.split(',')
        
        # Apply model-related arguments
        if args.model:
            config.model_type = args.model
        
        if args.no_gpu:
            config.use_gpu = False
            
        # Handle simulation mode
        if args.simulate:
            config.api_key = "simulation_mode_key"
            config.api_secret = "simulation_mode_secret"
            config.simulation_mode = True
            
            # Configure real market data in simulation mode if requested
            if args.real_data:
                config.use_real_market_data = True
                console.print("[bold green]Using real market data in simulation mode")
            else:
                config.use_real_market_data = False
            
            # Configure custom initial balance if provided
            if args.sim_balance:
                try:
                    sim_balance = {}
                    for balance_str in args.sim_balance.split(','):
                        asset, amount = balance_str.split(':')
                        sim_balance[asset.strip()] = float(amount.strip())
                    
                    config.sim_initial_balance = sim_balance
                    console.print(f"[bold green]Initial simulation balance set: {sim_balance}")
                except Exception as e:
                    console.print(f"[bold red]Error parsing simulation balance: {str(e)}")
                    logger.warning(f"Error parsing simulation balance: {str(e)}")
        
        # Initialize bot
        bot = ScalpingBot(config)
        return bot
        
    except Exception as e:
        logger.error(f"Error setting up bot: {str(e)}")
        console.print(f"[bold red]Error setting up bot: {str(e)}")
        return None

def start_bot():
    """Start the bot"""
    global bot
    
    if bot is None:
        console.print("[bold red]Bot not initialized")
        return False
    
    try:
        bot.start()
        console.print("[bold green]Bot started successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}")
        console.print(f"[bold red]Failed to start bot: {str(e)}")
        return False

def stop_bot():
    """Stop the bot"""
    global bot
    
    if bot is None:
        console.print("[bold red]Bot not initialized")
        return False
    
    try:
        bot.stop()
        console.print("[bold yellow]Bot stopped")
        return True
    
    except Exception as e:
        logger.error(f"Failed to stop bot: {str(e)}")
        console.print(f"[bold red]Failed to stop bot: {str(e)}")
        return False

# Dashboard Components
def create_status_panel():
    """Create bot status panel"""
    if bot is None:
        return Panel("Bot not initialized", title="[bold red]Status", border_style="red")
    
    status = "🟢 Running" if bot.active else "🔴 Stopped"
    
    content = Text()
    content.append(f"Status: ", style="bold")
    content.append(f"{status}\n", style="green" if bot.active else "red")
    
    content.append(f"Symbols: ", style="bold")
    content.append(f"{', '.join(bot.config.symbols)}\n")
    
    content.append(f"Mode: ", style="bold")
    content.append("Simulation" if bot.binance_client.simulation_mode else "TestNet\n")
    
    content.append(f"Model: ", style="bold")
    content.append(f"{bot.config.model_type}\n")
    
    content.append(f"Open trades: ", style="bold")
    content.append(f"{len(bot.open_trades)}\n")
    
    return Panel(content, title="[bold blue]Bot Status", border_style="blue")

def create_performance_panel():
    """Create performance summary panel"""
    if bot is None or not bot.active:
        return Panel("Bot not running", title="[bold yellow]Performance", border_style="yellow")
    
    try:
        perf = bot.get_performance_summary()
        
        content = Text()
        content.append(f"Total Trades: ", style="bold")
        content.append(f"{perf['total_trades']}\n")
        
        content.append(f"Profitable: ", style="bold")
        content.append(f"{perf['profitable_trades']} ")
        
        win_rate = 0 if perf['total_trades'] == 0 else (perf['profitable_trades'] / perf['total_trades']) * 100
        content.append(f"({win_rate:.2f}%)\n", style="green" if win_rate >= 50 else "red")
        
        content.append(f"Total P/L: ", style="bold")
        pl_style = "green" if perf['total_profit_loss'] >= 0 else "red"
        content.append(f"{perf['total_profit_loss']:.8f}\n", style=pl_style)
        
        if perf['total_trades'] > 0:
            content.append(f"Avg Profit: ", style="bold")
            content.append(f"{perf.get('avg_profit', 0):.8f}\n", style="green")
            
            content.append(f"Avg Loss: ", style="bold")
            content.append(f"{perf.get('avg_loss', 0):.8f}\n", style="red")
        
        return Panel(content, title="[bold green]Performance", border_style="green")
    
    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        return Panel(f"Error: {str(e)}", title="[bold red]Performance", border_style="red")

def create_prices_panel():
    """Create real-time prices panel"""
    if bot is None or not bot.active:
        return Panel("Bot not running", title="[bold cyan]Live Prices", border_style="cyan")
    
    content = Text()
    
    for symbol in bot.config.symbols:
        price = bot.real_time_prices.get(symbol, "Waiting for data...")
        
        # Add symbol icon based on open trade
        if symbol in bot.open_trades:
            trade = bot.open_trades[symbol]
            if trade['side'] == 'BUY':
                content.append(f"🔵 {symbol}: ", style="bold blue")
            else:
                content.append(f"🔴 {symbol}: ", style="bold red")
        else:
            content.append(f"⚪ {symbol}: ", style="bold")
        
        content.append(f"{price}\n")
    
    return Panel(content, title="[bold cyan]Live Prices", border_style="cyan")

def create_open_trades_panel():
    """Create open trades panel"""
    if bot is None or not bot.active:
        return Panel("Bot not running", title="[bold magenta]Open Trades", border_style="magenta")
    
    if not bot.open_trades:
        return Panel("No open trades", title="[bold magenta]Open Trades", border_style="magenta")
    
    table = Table(show_header=True, box=box.ROUNDED)
    table.add_column("Symbol", style="cyan")
    table.add_column("Side", style="magenta")
    table.add_column("Entry", style="yellow")
    table.add_column("Current", style="green")
    table.add_column("P/L", style="bold")
    table.add_column("Duration", style="blue")
    
    for symbol, trade in bot.open_trades.items():
        # Calculate current P/L
        current_price = bot.real_time_prices.get(symbol, trade['entry_price'])
        
        if trade['side'] == 'BUY':
            pl = (current_price - trade['entry_price']) * trade['quantity']
        else:
            pl = (trade['entry_price'] - current_price) * trade['quantity']
        
        # Calculate duration
        now = datetime.now()
        duration = now - trade['time_opened']
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        
        # Determine P/L style
        pl_style = "green" if pl >= 0 else "red"
        
        table.add_row(
            symbol,
            trade['side'],
            f"{trade['entry_price']:.2f}",
            f"{current_price:.2f}",
            f"[{pl_style}]{pl:.8f}",
            duration_str
        )
    
    return Panel(table, title="[bold magenta]Open Trades", border_style="magenta")

def create_recent_trades_panel():
    """Create recent trades panel"""
    if bot is None or not bot.active:
        return Panel("Bot not running", title="[bold yellow]Recent Trades", border_style="yellow")
    
    if not bot.trades_history:
        return Panel("No completed trades yet", title="[bold yellow]Recent Trades", border_style="yellow")
    
    table = Table(show_header=True, box=box.ROUNDED)
    table.add_column("Symbol", style="cyan")
    table.add_column("Side", style="magenta")
    table.add_column("P/L", style="bold")
    table.add_column("Time", style="blue")
    table.add_column("Reason", style="yellow")
    
    # Get most recent 5 trades
    recent_trades = sorted(bot.trades_history, key=lambda x: x['time_closed'], reverse=True)[:5]
    
    for trade in recent_trades:
        # Determine P/L style
        pl_style = "green" if trade['profit_loss'] >= 0 else "red"
        
        # Format time
        time_str = trade['time_closed'].strftime("%H:%M:%S")
        
        table.add_row(
            trade['symbol'],
            trade['side'],
            f"[{pl_style}]{trade['profit_loss']:.8f}",
            time_str,
            trade.get('reason', 'unknown')
        )
    
    return Panel(table, title="[bold yellow]Recent Trades", border_style="yellow")

def render_dashboard():
    """Render the main dashboard layout"""
    # Create layout
    layout = Layout()
    
    # Split into rows
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=1),
    )
    
    # Configure header
    header_text = Text(LOGO.split('\n')[1], style="bold yellow")
    layout["header"].update(Panel(header_text, style="yellow"))
    
    # Split main area into columns
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    
    # Split left column
    layout["left"].split(
        Layout(name="status"),
        Layout(name="performance"),
        Layout(name="prices"),
    )
    
    # Split right column
    layout["right"].split(
        Layout(name="open_trades", ratio=2),
        Layout(name="recent_trades", ratio=3),
    )
    
    # Update panels
    layout["status"].update(create_status_panel())
    layout["performance"].update(create_performance_panel())
    layout["prices"].update(create_prices_panel())
    layout["open_trades"].update(create_open_trades_panel())
    layout["recent_trades"].update(create_recent_trades_panel())
    
    # Update footer
    commands = "[bold cyan]Commands:[/] [green]s[/] Start | [red]x[/] Stop | [yellow]p[/] Performance | [blue]t[/] Trades | [magenta]c[/] Charts | [white]q[/] Quit"
    layout["footer"].update(Panel(commands, border_style="green"))
    
    return layout

def render_performance_view():
    """Render performance details view"""
    if bot is None or not bot.active:
        return Panel("Bot not running, no performance data available", title="[bold red]Performance", border_style="red")
    
    try:
        perf = bot.get_performance_summary()
        
        # Create layout
        layout = Layout()
        
        # Split into rows
        layout.split(
            Layout(name="header", size=1),
            Layout(name="metrics", ratio=1),
            Layout(name="symbols", ratio=1),
            Layout(name="footer", size=1),
        )
        
        # Header
        layout["header"].update(Panel("[bold green]Performance Details", border_style="green"))
        
        # Performance metrics
        metrics_text = Text()
        metrics_text.append("General Metrics\n", style="bold yellow")
        metrics_text.append("-" * 50 + "\n")
        
        metrics_text.append(f"Total Trades: ", style="bold")
        metrics_text.append(f"{perf['total_trades']}\n")
        
        metrics_text.append(f"Profitable Trades: ", style="bold")
        metrics_text.append(f"{perf['profitable_trades']} ")
        
        win_rate = 0 if perf['total_trades'] == 0 else (perf['profitable_trades'] / perf['total_trades']) * 100
        metrics_text.append(f"({win_rate:.2f}%)\n", style="green" if win_rate >= 50 else "red")
        
        metrics_text.append(f"Total P/L: ", style="bold")
        pl_style = "green" if perf['total_profit_loss'] >= 0 else "red"
        metrics_text.append(f"{perf['total_profit_loss']:.8f}\n", style=pl_style)
        
        metrics_text.append(f"Open Trades: ", style="bold")
        metrics_text.append(f"{perf['open_trades']}\n")
        
        # Add advanced metrics if available
        if perf['total_trades'] > 0:
            metrics_text.append("\nAdvanced Metrics\n", style="bold yellow")
            metrics_text.append("-" * 50 + "\n")
            
            metrics_text.append(f"Average Profit: ", style="bold")
            metrics_text.append(f"{perf.get('avg_profit', 0):.8f}\n", style="green")
            
            metrics_text.append(f"Average Loss: ", style="bold")
            metrics_text.append(f"{perf.get('avg_loss', 0):.8f}\n", style="red")
            
            metrics_text.append(f"Profit Factor: ", style="bold")
            profit_factor = perf.get('profit_factor', 0)
            metrics_text.append(f"{profit_factor:.2f}\n", style="green" if profit_factor > 1 else "red")
            
            metrics_text.append(f"Trades Per Hour: ", style="bold")
            metrics_text.append(f"{perf.get('trades_per_hour', 0):.2f}\n")
            
            if perf.get('active_since'):
                metrics_text.append(f"Active Since: ", style="bold")
                metrics_text.append(f"{perf.get('active_since')}\n")
        
        layout["metrics"].update(Panel(metrics_text, title="[bold green]Performance Metrics", border_style="green"))
        
        # Symbol performance
        if perf.get('best_symbols') and len(perf['best_symbols']) > 0:
            symbols_table = Table(show_header=True, box=box.ROUNDED)
            symbols_table.add_column("Symbol", style="cyan")
            symbols_table.add_column("Profit/Loss", style="bold")
            symbols_table.add_column("Trades", style="blue")
            
            for symbol_data in perf['best_symbols']:
                symbol = symbol_data['symbol']
                profit = symbol_data['profit']
                trades = symbol_data['trades']
                
                profit_style = "green" if profit >= 0 else "red"
                
                symbols_table.add_row(
                    symbol,
                    f"[{profit_style}]{profit:.8f}",
                    f"{trades}"
                )
            
            layout["symbols"].update(Panel(symbols_table, title="[bold blue]Symbol Performance", border_style="blue"))
        else:
            layout["symbols"].update(Panel("No symbol performance data available", title="[bold yellow]Symbol Performance", border_style="yellow"))
        
        # Footer
        commands = "[bold cyan]Commands:[/] [green]d[/] Dashboard | [blue]t[/] Trades | [magenta]c[/] Charts | [white]q[/] Quit"
        layout["footer"].update(Panel(commands, border_style="green"))
        
        return layout
    
    except Exception as e:
        logger.error(f"Error rendering performance view: {str(e)}")
        return Panel(f"Error: {str(e)}", title="[bold red]Performance", border_style="red")

def render_trades_view():
    """Render trades history view"""
    if bot is None or not bot.active:
        return Panel("Bot not running, no trades data available", title="[bold red]Trades History", border_style="red")
    
    # Create layout
    layout = Layout()
    
    # Split into rows
    layout.split(
        Layout(name="header", size=1),
        Layout(name="open_trades", ratio=1),
        Layout(name="trade_history", ratio=3),
        Layout(name="footer", size=1),
    )
    
    # Header
    layout["header"].update(Panel("[bold magenta]Trades History", border_style="magenta"))
    
    # Open trades
    if bot.open_trades:
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", style="magenta")
        table.add_column("Entry", style="yellow")
        table.add_column("Current", style="green")
        table.add_column("P/L", style="bold")
        table.add_column("Duration", style="blue")
        table.add_column("SL/TP", style="cyan")
        
        for symbol, trade in bot.open_trades.items():
            # Calculate current P/L
            current_price = bot.real_time_prices.get(symbol, trade['entry_price'])
            
            if trade['side'] == 'BUY':
                pl = (current_price - trade['entry_price']) * trade['quantity']
            else:
                pl = (trade['entry_price'] - current_price) * trade['quantity']
            
            # Calculate duration
            now = datetime.now()
            duration = now - trade['time_opened']
            duration_str = str(duration).split('.')[0]  # Remove microseconds
            
            # Format SL/TP
            sl_tp = f"SL: {trade.get('stop_loss', 'N/A'):.2f}\nTP: {trade.get('take_profit', 'N/A'):.2f}"
            
            # Determine P/L style
            pl_style = "green" if pl >= 0 else "red"
            
            table.add_row(
                symbol,
                trade['side'],
                f"{trade['entry_price']:.2f}",
                f"{current_price:.2f}",
                f"[{pl_style}]{pl:.8f}",
                duration_str,
                sl_tp
            )
        
        layout["open_trades"].update(Panel(table, title="[bold green]Open Trades", border_style="green"))
    else:
        layout["open_trades"].update(Panel("No open trades", title="[bold yellow]Open Trades", border_style="yellow"))
    
    # Trade history
    if bot.trades_history:
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Time", style="cyan")
        table.add_column("Symbol", style="bold")
        table.add_column("Side", style="magenta")
        table.add_column("Entry", style="yellow")
        table.add_column("Exit", style="yellow")
        table.add_column("P/L", style="bold")
        table.add_column("Reason", style="blue")
        
        # Sort trades by time (most recent first)
        trades = sorted(bot.trades_history, key=lambda x: x['time_closed'], reverse=True)
        
        for trade in trades:
            # Format time
            time_str = trade['time_closed'].strftime("%Y-%m-%d %H:%M:%S")
            
            # Determine P/L style
            pl_style = "green" if trade['profit_loss'] >= 0 else "red"
            
            table.add_row(
                time_str,
                trade['symbol'],
                trade['side'],
                f"{trade['entry_price']:.2f}",
                f"{trade['exit_price']:.2f}",
                f"[{pl_style}]{trade['profit_loss']:.8f}",
                trade.get('reason', 'unknown')
            )
        
        layout["trade_history"].update(Panel(table, title="[bold blue]Trade History", border_style="blue"))
    else:
        layout["trade_history"].update(Panel("No completed trades yet", title="[bold yellow]Trade History", border_style="yellow"))
    
    # Footer
    commands = "[bold cyan]Commands:[/] [green]d[/] Dashboard | [yellow]p[/] Performance | [magenta]c[/] Charts | [white]q[/] Quit"
    layout["footer"].update(Panel(commands, border_style="green"))
    
    return layout

def render_chart_symbol_selection():
    """Render chart symbol selection view"""
    if bot is None or not bot.active or not bot.config.symbols:
        return Panel("Bot not running or no symbols configured", title="[bold red]Charts", border_style="red")
    
    content = Text()
    content.append("Select a symbol to view chart:\n\n", style="bold yellow")
    
    for i, symbol in enumerate(bot.config.symbols, 1):
        content.append(f"{i}. {symbol}\n", style="cyan")
    
    content.append("\n0. Return to Dashboard", style="green")
    
    return Panel(content, title="[bold blue]Chart Selection", border_style="blue")

def render_chart_view(symbol):
    """Render ASCII chart for a symbol"""
    if bot is None or not bot.active:
        return Panel("Bot not running", title="[bold red]Chart", border_style="red")
    
    try:
        # Get price data
        klines = bot.binance_client.get_klines(
            symbol=symbol,
            interval=bot.config.timeframe,
            limit=40  # Last 40 candles
        )
        
        if not klines:
            return Panel(f"No data available for {symbol}", title="[bold yellow]Chart", border_style="yellow")
        
        # Extract prices and timestamps
        prices = [float(k['close']) for k in klines]
        timestamps = [k['open_time'].strftime("%H:%M") for k in klines]
        
        # Normalize prices to fit in terminal
        height = 15  # Chart height
        price_min = min(prices)
        price_max = max(prices)
        price_range = price_max - price_min
        
        normalized_prices = [int((p - price_min) / price_range * (height - 1)) if price_range > 0 else 0 for p in prices]
        
        # Create ASCII chart
        chart = []
        for y in range(height - 1, -1, -1):
            line = ""
            for i, p in enumerate(normalized_prices):
                if p == y:
                    # Show price point
                    line += "●"
                elif i > 0 and normalized_prices[i-1] < y < p or normalized_prices[i-1] > y > p:
                    # Show connecting line
                    line += "│"
                else:
                    line += " "
            chart.append(line)
        
        # Add price labels
        chart.insert(0, f"{price_max:.2f}")
        chart.append(f"{price_min:.2f}")
        
        # Add timestamp labels
        timestamp_line = ""
        step = max(1, len(timestamps) // 8)  # Show up to 8 labels
        for i, ts in enumerate(timestamps):
            if i % step == 0:
                timestamp_label = ts
                timestamp_line += timestamp_label.ljust(step)
            
        # Convert chart to string
        chart_str = "\n".join(chart) + "\n" + timestamp_line
        
        # Create layout
        layout = Layout()
        
        # Split into rows
        layout.split(
            Layout(name="header", size=1),
            Layout(name="info", size=4),
            Layout(name="chart", ratio=1),
            Layout(name="footer", size=1),
        )
        
        # Header
        layout["header"].update(Panel(f"[bold blue]{symbol} Chart", border_style="blue"))
        
        # Symbol info
        info = Text()
        
        current_price = bot.real_time_prices.get(symbol, prices[-1])
        percent_change = ((current_price - prices[0]) / prices[0]) * 100
        
        info.append(f"Current Price: ", style="bold")
        info.append(f"{current_price:.2f}\n")
        
        info.append(f"24h Change: ", style="bold")
        change_style = "green" if percent_change >= 0 else "red"
        change_arrow = "↑" if percent_change >= 0 else "↓"
        info.append(f"{change_arrow} {abs(percent_change):.2f}%\n", style=change_style)
        
        # Show open trade info if exists
        if symbol in bot.open_trades:
            trade = bot.open_trades[symbol]
            
            info.append(f"\nOpen Trade Details:\n", style="bold yellow")
            info.append(f"Side: ", style="bold")
            info.append(f"{trade['side']}\n")
            
            info.append(f"Entry: ", style="bold")
            info.append(f"{trade['entry_price']:.2f}\n")
            
            info.append(f"Quantity: ", style="bold")
            info.append(f"{trade['quantity']}\n")
            
            if 'stop_loss' in trade:
                info.append(f"Stop Loss: ", style="bold")
                info.append(f"{trade['stop_loss']:.2f}\n")
            
            if 'take_profit' in trade:
                info.append(f"Take Profit: ", style="bold")
                info.append(f"{trade['take_profit']:.2f}\n")
        
        layout["info"].update(Panel(info, title="Symbol Info", border_style="green"))
        
        # Chart
        layout["chart"].update(Panel(chart_str, title="Price Chart", border_style="yellow"))
        
        # Footer
        commands = "[bold cyan]Commands:[/] [green]d[/] Dashboard | [yellow]p[/] Performance | [blue]t[/] Trades | [white]b[/] Back | [white]q[/] Quit"
        layout["footer"].update(Panel(commands, border_style="green"))
        
        return layout
        
    except Exception as e:
        logger.error(f"Error rendering chart: {str(e)}")
        return Panel(f"Error rendering chart: {str(e)}", title="[bold red]Chart Error", border_style="red")

def render_current_view():
    """Render the current view based on view state"""
    global current_view, current_symbol
    
    if current_view == "dashboard":
        return render_dashboard()
    elif current_view == "performance":
        return render_performance_view()
    elif current_view == "trades":
        return render_trades_view()
    elif current_view == "chart_selection":
        return render_chart_symbol_selection()
    elif current_view == "chart" and current_symbol:
        return render_chart_view(current_symbol)
    else:
        return render_dashboard()

def handle_input(key):
    """Handle keyboard input"""
    global current_view, current_symbol, stop_event
    
    # Common commands
    if key == 'q':
        if Confirm.ask("Are you sure you want to quit?"):
            stop_event.set()
            return True
    
    # View switching
    elif key == 'd':
        current_view = "dashboard"
    elif key == 'p':
        current_view = "performance"
    elif key == 't':
        current_view = "trades"
    elif key == 'c':
        current_view = "chart_selection"
    
    # Bot control
    elif key == 's':
        if not bot.active:
            start_bot()
    elif key == 'x':
        if bot.active:
            if Confirm.ask("Are you sure you want to stop the bot?"):
                stop_bot()
    
    # Chart selection handling
    elif current_view == "chart_selection":
        if key == '0':
            current_view = "dashboard"
        elif key.isdigit() and int(key) > 0 and bot and bot.config.symbols and int(key) <= len(bot.config.symbols):
            current_symbol = bot.config.symbols[int(key) - 1]
            current_view = "chart"
    
    # Back button for chart view
    elif current_view == "chart" and key == 'b':
        current_view = "chart_selection"
    
    return False

def update_ui():
    """Main UI update loop"""
    global stop_event
    
    try:
        with Live(render_current_view(), refresh_per_second=1) as live:
            while not stop_event.is_set():
                # Update the display
                live.update(render_current_view())
                
                # Check for input (non-blocking)
                if console.input(timeout=0.1) as key:
                    if handle_input(key):
                        break
                
                # Sleep to prevent high CPU usage
                time.sleep(update_interval)
    
    except KeyboardInterrupt:
        console.print("[bold yellow]Interrupted by user.")
    except Exception as e:
        logger.error(f"Error in UI loop: {str(e)}")
        console.print(f"[bold red]Error: {str(e)}")

def show_welcome():
    """Display welcome screen with ASCII art logo"""
    console.clear()
    console.print(LOGO, style="bold yellow")
    console.print("="*72)
    console.print("  Developed for high-frequency trading in cryptocurrency markets")
    console.print("  Version: 4.0.0 - Testnet Enabled")
    console.print("  Strategy: RSI + Bollinger Bands + SMA Crossovers")
    console.print("="*72)
    console.print("\nInitializing system...\n")

def main():
    """Main entry point for the CLI interface"""
    global bot, stop_event
    
    show_welcome()
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup bot
        bot = setup_bot(args)
        if bot is None:
            console.print("[bold red]Failed to initialize bot. Exiting.")
            return
        
        console.print("[bold green]Bot initialized successfully.")
        
        # Ask if user wants to start the bot
        if Confirm.ask("Start the bot now?"):
            start_bot()
        
        # Start UI update loop
        update_ui()
    
    except KeyboardInterrupt:
        console.print("[bold yellow]Interrupted by user.")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        console.print(f"[bold red]Error: {str(e)}")
    finally:
        # Clean up
        if bot and bot.active:
            console.print("Stopping bot...")
            stop_bot()
        
        console.print("[bold green]Exiting. Goodbye!")

if __name__ == "__main__":
    main()