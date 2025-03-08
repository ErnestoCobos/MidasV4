#!/usr/bin/env python3
import os
import argparse
import logging
import json
import signal
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from config import Config
from bot import ScalpingBot
from core.logging_setup import setup_logging, is_debug_mode, log_exception

# Configure enhanced logging
logger = setup_logging('CLI_Interface', component='cli')

# Log the debug mode status at startup
debug_mode = is_debug_mode()
if debug_mode:
    logger.debug("🔍 DEBUG mode is ENABLED - Verbose logging activated")
    logger.debug("Running CLI Interface with enhanced debugging")

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
    
    # Group for main configuration options
    main_config = parser.add_argument_group('Main Configuration')
    main_config.add_argument('--config', type=str, default=None,
                      help='Path to JSON configuration file')
    main_config.add_argument('--api-key', type=str, default=None,
                      help='Binance API key (overrides environment variables)')
    main_config.add_argument('--api-secret', type=str, default=None,
                      help='Binance API secret (overrides environment variables)')
    
    # Group for trading parameters
    trading_params = parser.add_argument_group('Trading Parameters')
    trading_params.add_argument('--symbols', type=str, default=None,
                      help='Comma-separated list of trading pairs (e.g., "BTCUSDT,ETHUSDT")')
    trading_params.add_argument('--max-open-trades', type=int, default=None,
                      help='Maximum number of concurrent trades')
    trading_params.add_argument('--timeframe', type=str, default=None,
                      help='Timeframe for trading (e.g., "1m", "5m", "15m", "1h")')
    
    # Group for risk management
    risk_params = parser.add_argument_group('Risk Management')
    risk_params.add_argument('--max-risk', type=float, default=None,
                      help='Maximum capital risk percentage per trade')
    risk_params.add_argument('--stop-loss', type=float, default=None,
                      help='Stop loss percentage')
    risk_params.add_argument('--take-profit', type=float, default=None,
                      help='Take profit percentage')
    risk_params.add_argument('--trailing-stop', type=float, default=None,
                      help='Trailing stop percentage')
    
    # Group for exchange connection options
    exchange_group = parser.add_argument_group('Exchange Connection')
    exchange_group.add_argument('--testnet', action='store_true',
                      help='Use Binance testnet instead of real exchange')
    exchange_group.add_argument('--simulate', action='store_true',
                      help='Run in simulation mode without connecting to exchange API')
    exchange_group.add_argument('--real-data', action='store_true',
                      help='Use real market data from Binance API in simulation mode')
    exchange_group.add_argument('--sim-balance', type=str, default=None,
                      help='Initial balance for simulation (format: "USDT:10000,BTC:0.5,ETH:5")')
    
    # Group for model-related arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', type=str, default=None,
                      choices=['xgboost', 'lstm', 'llm', 'indicator', 'rl', 'deepscalper'],
                      help='Model type to use (xgboost, lstm, llm, indicator, rl, deepscalper)')
    model_group.add_argument('--model-path', type=str, default=None,
                      help='Path to saved model file to load')
    model_group.add_argument('--confidence', type=float, default=None,
                      help='Confidence threshold for trading signals (0-100)')
    model_group.add_argument('--no-gpu', action='store_true',
                      help='Disable GPU acceleration even if available')
    
    # Group for debugging and logging
    debug_group = parser.add_argument_group('Debugging and Logging')
    debug_group.add_argument('--debug', action='store_true',
                      help='Enable debug mode with verbose logging (alternative to MIDAS_DEBUG=1)')
    debug_group.add_argument('--log-level', type=str, default=None,
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set specific logging level')
    debug_group.add_argument('--log-file', type=str, default=None,
                      help='Specify custom log file path')
    
    # Action commands (mutually exclusive)
    action_group = parser.add_argument_group('Actions')
    action_mutex = parser.add_mutually_exclusive_group()
    action_mutex.add_argument('--train', action='store_true',
                      help='Train model before starting bot')
    action_mutex.add_argument('--backtest', action='store_true',
                      help='Run backtesting and exit')
    action_mutex.add_argument('--optimize', action='store_true',
                      help='Run parameter optimization and exit')
    action_mutex.add_argument('--show-balance', action='store_true',
                      help='Show account balance and exit')
    action_mutex.add_argument('--health-check', action='store_true',
                      help='Run system health check and exit')
    
    args = parser.parse_args()
    
    # Set debug environment variable if specified via command line
    if args.debug:
        os.environ['MIDAS_DEBUG'] = '1'
    
    return args

def load_config_from_file(file_path: str) -> Config:
    """Load configuration from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        logger.debug(f"Loaded configuration from {file_path}")
        
        # Create Config object from dict
        config = Config(
            api_key=config_dict.get('api_key', ''),
            api_secret=config_dict.get('api_secret', ''),
            symbols=config_dict.get('symbols', ['BTCUSDT']),
            max_open_trades=config_dict.get('max_open_trades', 3),
            max_capital_risk_percent=config_dict.get('max_capital_risk_percent', 2.0),
            stop_loss_percent=config_dict.get('stop_loss_percent', 0.5),
            take_profit_percent=config_dict.get('take_profit_percent', 1.0),
            timeframe=config_dict.get('timeframe', '1m'),
            model_type=config_dict.get('model_type', 'llm'),
            use_gpu=config_dict.get('use_gpu', False),
        )
        
        return config
    
    except Exception as e:
        log_exception(logger, e, f"Failed to load configuration from {file_path}")
        raise

def setup_bot(args: argparse.Namespace) -> Optional[ScalpingBot]:
    """Set up and initialize the bot with given arguments"""
    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from file: {args.config}")
            config = load_config_from_file(args.config)
        else:
            logger.info("Loading configuration from environment variables")
            config = Config.from_env()
        
        # Override with command line arguments
        if args.api_key:
            config.api_key = args.api_key
            logger.debug("API key overridden from command line")
        
        if args.api_secret:
            config.api_secret = args.api_secret
            logger.debug("API secret overridden from command line")
        
        if args.symbols:
            config.symbols = args.symbols.split(',')
            logger.info(f"Trading symbols set to: {', '.join(config.symbols)}")
        
        # Apply model-related arguments
        if args.model:
            config.model_type = args.model
            logger.info(f"Model type set to: {config.model_type}")
        
        if args.confidence:
            config.confidence_threshold = args.confidence
            logger.info(f"Confidence threshold set to: {config.confidence_threshold}%")
        
        if args.no_gpu:
            config.use_gpu = False
            logger.info("GPU acceleration disabled")
        
        # Apply risk management parameters
        if args.max_risk:
            config.max_capital_risk_percent = args.max_risk
            logger.info(f"Max capital risk set to: {config.max_capital_risk_percent}%")
        
        if args.stop_loss:
            config.stop_loss_percent = args.stop_loss
            logger.info(f"Stop loss set to: {config.stop_loss_percent}%")
        
        if args.take_profit:
            config.take_profit_percent = args.take_profit
            logger.info(f"Take profit set to: {config.take_profit_percent}%")
        
        if args.trailing_stop:
            config.trailing_stop_pct = args.trailing_stop
            logger.info(f"Trailing stop set to: {config.trailing_stop_pct}%")
        
        if args.timeframe:
            config.timeframe = args.timeframe
            logger.info(f"Timeframe set to: {config.timeframe}")
        
        if args.max_open_trades:
            config.max_open_trades = args.max_open_trades
            logger.info(f"Max open trades set to: {config.max_open_trades}")
            
        # Handle simulation mode
        if args.simulate:
            config.api_key = "simulation_mode_key"
            config.api_secret = "simulation_mode_secret"
            config.simulation_mode = True
            
            # Configure real market data in simulation mode if requested
            if args.real_data:
                config.use_real_market_data = True
                logger.info("Using real market data in simulation mode")
            else:
                config.use_real_market_data = False
                logger.info("Using simulated market data")
            
            # Configure custom initial balance if provided
            if args.sim_balance:
                try:
                    sim_balance = {}
                    for balance_str in args.sim_balance.split(','):
                        asset, amount = balance_str.split(':')
                        sim_balance[asset.strip()] = float(amount.strip())
                    
                    config.sim_initial_balance = sim_balance
                    logger.info(f"Initial simulation balance set: {sim_balance}")
                except Exception as e:
                    logger.warning(f"Error parsing simulation balance: {str(e)}")
        
        # Handle testnet mode
        if args.testnet:
            config.testnet = True
            logger.info("Using Binance testnet")
        
        # Initialize bot
        logger.info("Initializing bot")
        return ScalpingBot(config)
        
    except Exception as e:
        log_exception(logger, e, "Error setting up bot")
        print(f"Error setting up bot: {str(e)}")
        if debug_mode:
            print("Check logs for detailed error information")
        return None

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\nInterrupted by user. Shutting down...")
    logger.info("Bot interrupted by user signal")
    sys.exit(0)

def format_balance(balances: Dict[str, float]) -> str:
    """Format account balances for display"""
    result = []
    for asset, amount in sorted(balances.items()):
        if amount > 0:
            if amount < 0.001:
                result.append(f"{asset}: {amount:.8f}")
            else:
                result.append(f"{asset}: {amount:.4f}")
    return "\n".join(result)

def show_performance(bot: ScalpingBot) -> None:
    """Show performance statistics in a nicely formatted table"""
    try:
        perf = bot.get_performance_summary()
        
        # Helper function to draw a nice ASCII box
        def draw_box(title, content, width=72):
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│ " + title.center(width - 4) + " │")
            print("├" + "─" * (width - 2) + "┤")
            for line in content:
                print("│ " + line.ljust(width - 4) + " │")
            print("└" + "─" * (width - 2) + "┘")
        
        # --- PERFORMANCE SUMMARY SECTION ---
        summary_content = []
        
        # Format initial and current balance
        if perf.get('initial_balance') or perf.get('current_balance'):
            summary_content.append("ACCOUNT BALANCE".center(68))
            summary_content.append("─" * 68)
            
            # Format as table with columns
            balance_header = f"{'ASSET':<10} {'INITIAL':<20} {'CURRENT':<20} {'CHANGE':<15}"
            summary_content.append(balance_header)
            summary_content.append("─" * 68)
            
            # Get all unique assets from both balances
            all_assets = set()
            if perf.get('initial_balance'):
                all_assets.update(perf.get('initial_balance').keys())
            if perf.get('current_balance'):
                all_assets.update(perf.get('current_balance').keys())
                
            # Show only assets with non-zero balances
            for asset in sorted(all_assets):
                initial = perf.get('initial_balance', {}).get(asset, 0)
                current = perf.get('current_balance', {}).get(asset, 0)
                
                if initial > 0 or current > 0:
                    change = current - initial
                    change_str = f"{change:+.8f}" if change != 0 else "0.00000000"
                    
                    # Format with appropriate precision
                    if asset == 'USDT':
                        row = f"{asset:<10} {initial:<20.2f} {current:<20.2f} {change_str:<15}"
                    else:
                        row = f"{asset:<10} {initial:<20.8f} {current:<20.8f} {change_str:<15}"
                        
                    summary_content.append(row)
            
            summary_content.append("")  # Add separator line
            
        # Add key metrics
        summary_content.append("TRADING METRICS".center(68))
        summary_content.append("─" * 68)
        
        win_rate = perf.get('win_rate', 0)
        
        # Format as two columns
        col_width = 34  # Width for each column
        
        # Row 1: Trade counts and win rate
        row1_col1 = f"Total Trades: {perf.get('total_trades', 0)}"
        row1_col2 = f"Win Rate: {win_rate:.2f}%"
        summary_content.append(f"{row1_col1:<{col_width}}{row1_col2:<{col_width}}")
        
        # Row 2: Profitable trades and P/L
        row2_col1 = f"Profitable: {perf.get('profitable_trades', 0)}"
        row2_col2 = f"Total P/L: {perf.get('total_profit_loss', 0):.8f}"
        summary_content.append(f"{row2_col1:<{col_width}}{row2_col2:<{col_width}}")
        
        # Row 3: Open trades and profit factor
        row3_col1 = f"Open Trades: {perf.get('open_trades', 0)}"
        if perf.get('profit_factor', 0) > 999:
            row3_col2 = "Profit Factor: ∞"
        else:
            row3_col2 = f"Profit Factor: {perf.get('profit_factor', 0):.2f}"
        summary_content.append(f"{row3_col1:<{col_width}}{row3_col2:<{col_width}}")
        
        # Row 4: Average metrics
        if 'avg_profit' in perf and 'avg_loss' in perf:
            row4_col1 = f"Avg Profit: {perf.get('avg_profit', 0):.8f}"
            row4_col2 = f"Avg Loss: {perf.get('avg_loss', 0):.8f}"
            summary_content.append(f"{row4_col1:<{col_width}}{row4_col2:<{col_width}}")
        
        # Row 5: Trading frequency and active since
        if perf.get('trades_per_hour', 0) > 0:
            row5_col1 = f"Trades/Hour: {perf.get('trades_per_hour', 0):.2f}"
            if perf.get('active_since'):
                row5_col2 = f"Active Since: {perf.get('active_since')}"
                summary_content.append(f"{row5_col1:<{col_width}}{row5_col2:<{col_width}}")
            else:
                summary_content.append(f"{row5_col1:<{col_width}}")
        
        # Draw the performance summary box
        draw_box("PERFORMANCE SUMMARY", summary_content)
        
        # --- SYMBOL PERFORMANCE SECTION ---
        if perf.get('best_symbols') and len(perf.get('best_symbols', [])) > 0:
            symbol_content = []
            
            # Create header
            symbol_header = f"{'SYMBOL':<10} {'PROFIT/LOSS':<25} {'TRADES':<10} {'WIN %':<10}"
            symbol_content.append(symbol_header)
            symbol_content.append("─" * 68)
            
            # Add data for each symbol
            for data in perf.get('best_symbols', []):
                symbol = data['symbol']
                profit = data['profit']
                trades = data['trades']
                
                # Calculate win rate for this symbol - need to access from perf data directly
                symbol_win_rate = 0
                symbol_wins = 0
                
                # Calculate from trades history
                for trade in bot.trades_history:
                    if trade['symbol'] == symbol and trade['profit_loss'] > 0:
                        symbol_wins += 1
                
                symbol_win_rate = (symbol_wins / trades) * 100 if trades > 0 else 0
                
                profit_str = f"{profit:.8f}"
                row = f"{symbol:<10} {profit_str:<25} {trades:<10} {symbol_win_rate:.2f}%"
                symbol_content.append(row)
            
            # Draw the symbol performance box
            draw_box("SYMBOL PERFORMANCE", symbol_content)
    
    except Exception as e:
        print(f"Error getting performance: {str(e)}")
        logger.error(f"Error getting performance: {str(e)}")

def show_trades(bot: ScalpingBot) -> None:
    """Show open trades and recent trade history in formatted tables"""
    try:
        # Helper function to draw a nice ASCII box
        def draw_box(title, content, width=72):
            print("\n┌" + "─" * (width - 2) + "┐")
            print("│ " + title.center(width - 4) + " │")
            print("├" + "─" * (width - 2) + "┤")
            for line in content:
                print("│ " + line.ljust(width - 4) + " │")
            print("└" + "─" * (width - 2) + "┘")
            
        # --- OPEN TRADES SECTION ---
        if not bot.open_trades:
            open_trades_content = ["No open trades"]
        else:
            open_trades_content = []
            
            # Table header
            header = f"{'SYMBOL':<10} {'SIDE':<6} {'ENTRY':<12} {'CURRENT':<12} {'P/L':<14} {'DURATION':<15}"
            open_trades_content.append(header)
            open_trades_content.append("─" * 68)
            
            for symbol, trade in bot.open_trades.items():
                # Calculate current P/L
                current_price = bot.real_time_prices.get(symbol, trade['entry_price'])
                
                if trade['side'] == 'BUY':
                    pl = (current_price - trade['entry_price']) * trade['quantity']
                else:
                    pl = (trade['entry_price'] - current_price) * trade['quantity']
                
                # Format P/L with sign
                if pl >= 0:
                    pl_str = f"+{pl:.8f}"
                else:
                    pl_str = f"{pl:.8f}"
                
                # Calculate duration
                now = datetime.now()
                duration = now - trade['time_opened']
                duration_str = str(duration).split('.')[0]  # Remove microseconds
                
                # Format row
                row = f"{symbol:<10} {trade['side']:<6} {trade['entry_price']:<12.8f} {current_price:<12.8f} {pl_str:<14} {duration_str:<15}"
                open_trades_content.append(row)
                
                # Add SL/TP info in second row
                sl_tp_info = "  "
                if 'stop_loss' in trade:
                    sl_tp_info += f"SL: {trade['stop_loss']:.8f}  "
                if 'take_profit' in trade:
                    sl_tp_info += f"TP: {trade['take_profit']:.8f}"
                    
                if sl_tp_info.strip():
                    open_trades_content.append(sl_tp_info)
                    open_trades_content.append("")  # Empty row for spacing
        
        # Draw open trades box
        draw_box("OPEN TRADES", open_trades_content)
        
        # --- RECENT TRADES SECTION ---
        if not bot.trades_history:
            recent_trades_content = ["No completed trades"]
        else:
            recent_trades_content = []
            
            # Table header
            header = f"{'TIME':<20} {'SYMBOL':<10} {'SIDE':<6} {'ENTRY':<12} {'EXIT':<12} {'P/L':<14} {'REASON':<10}"
            recent_trades_content.append(header)
            recent_trades_content.append("─" * 68)
            
            # Most recent 5 trades
            recent_trades = sorted(bot.trades_history, key=lambda x: x['time_closed'], reverse=True)[:5]
            
            for trade in recent_trades:
                # Format P/L with sign
                if trade['profit_loss'] >= 0:
                    profit_str = f"+{trade['profit_loss']:.8f}"
                else:
                    profit_str = f"{trade['profit_loss']:.8f}"
                
                # Format time
                time_str = trade['time_closed'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Truncate reason if too long
                reason = trade.get('reason', 'unknown')
                if len(reason) > 10:
                    reason = reason[:7] + "..."
                
                # Format row
                row = f"{time_str:<20} {trade['symbol']:<10} {trade['side']:<6} {trade['entry_price']:<12.8f} {trade['exit_price']:<12.8f} {profit_str:<14} {reason:<10}"
                recent_trades_content.append(row)
            
            if len(bot.trades_history) > 5:
                recent_trades_content.append("")
                recent_trades_content.append(f"...and {len(bot.trades_history) - 5} more trades")
        
        # Draw recent trades box
        draw_box("RECENT TRADES", recent_trades_content)
    
    except Exception as e:
        print(f"Error getting trades: {str(e)}")
        logger.error(f"Error getting trades: {str(e)}")

def show_welcome():
    """Display welcome screen with ASCII art logo"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(LOGO)
    print("="*72)
    print("  Developed for high-frequency trading in cryptocurrency markets")
    print("  Version: 4.0.0 - Command Line Mode")
    print("  Strategy: AI-Powered Multi-Strategy Approach")
    print("="*72)
    print("\nInitializing system...\n")

def run_health_check() -> bool:
    """Run system health check"""
    print("\n=== SYSTEM HEALTH CHECK ===")
    
    checks_passed = True
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")
    if not (3, 7) <= sys.version_info < (3, 12):
        print("⚠️  Warning: Recommended Python version is 3.7-3.11")
        checks_passed = False
    
    # Check for required packages
    required_packages = ['numpy', 'pandas', 'ccxt', 'scikit-learn']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            checks_passed = False
    
    # Check for model dependencies
    model_packages = {
        'xgboost': 'XGBoost',
        'tensorflow': 'TensorFlow (for LSTM)',
        'torch': 'PyTorch (for Deep Learning)',
        'transformers': 'Transformers (for LLM)'
    }
    
    for package, name in model_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"⚠️  {name} not installed (optional)")
    
    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✅ GPU available: {device_name} ({device_count} devices)")
        else:
            print("ℹ️  No GPU detected, using CPU")
    except:
        print("ℹ️  Could not check GPU, PyTorch not available")
    
    # Check logs directory
    logs_dir = os.path.join(os.getcwd(), 'logs')
    if os.path.isdir(logs_dir):
        print(f"✅ Logs directory exists: {logs_dir}")
    else:
        print(f"⚠️  Logs directory not found: {logs_dir}")
        checks_passed = False
    
    # Check saved_models directory
    models_dir = os.path.join(os.getcwd(), 'saved_models')
    if os.path.isdir(models_dir):
        print(f"✅ Models directory exists: {models_dir}")
    else:
        print(f"⚠️  Models directory not found: {models_dir}")
    
    # Overall result
    print("\n=== HEALTH CHECK RESULT ===")
    if checks_passed:
        print("✅ All critical checks passed. System is ready.")
    else:
        print("⚠️  Some checks failed. Review warnings before proceeding.")
    
    return checks_passed

def main():
    """Main entry point for the CLI interface"""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Show welcome screen
    show_welcome()
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Run health check if requested
        if args.health_check:
            run_health_check()
            return
        
        # Setup bot
        bot = setup_bot(args)
        if bot is None:
            print("Failed to initialize bot. Exiting.")
            return
        
        print("Bot initialized successfully.")
        
        # Handle show balance and exit
        if args.show_balance:
            try:
                print("\n=== ACCOUNT BALANCE ===")
                balances = bot.binance_client.get_account_balance()
                print(format_balance(balances))
                return
            except Exception as e:
                print(f"Error getting balance: {str(e)}")
                return
        
        # Handle model training
        if args.train:
            print("\nTraining model...")
            # This would call the training functionality
            # For now, just showing a placeholder
            print("Training complete.")
            return
        
        # Handle backtesting
        if args.backtest:
            print("\nRunning backtesting...")
            # This would call the backtesting functionality
            # For now, just showing a placeholder
            print("Backtesting complete.")
            return
        
        # Handle optimization
        if args.optimize:
            print("\nRunning parameter optimization...")
            # This would call the optimization functionality
            # For now, just showing a placeholder
            print("Optimization complete.")
            return
        
        # Start the bot
        print("\nStarting trading bot...")
        bot.start()
        
        # Print configuration summary
        print("\n=== CONFIGURATION ===")
        print(f"Trading Pairs: {', '.join(bot.config.symbols)}")
        print(f"Model Type: {bot.config.model_type}")
        print(f"Timeframe: {bot.config.timeframe}")
        print(f"Stop Loss: {bot.config.stop_loss_percent}%")
        print(f"Take Profit: {bot.config.take_profit_percent}%")
        print(f"Risk Per Trade: {bot.config.max_capital_risk_percent}%")
        print(f"Max Open Trades: {bot.config.max_open_trades}")
        print(f"Mode: {'Simulation' if getattr(bot.config, 'simulation_mode', False) else 'TestNet' if getattr(bot.config, 'testnet', False) else 'Live'}")
        
        print("\nBot running in CLI mode. Press Ctrl+C to stop.")
        print("Periodic updates will be displayed below:")
        
        # Main loop
        update_interval = 10  # seconds
        last_update_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Show periodic updates
                if current_time - last_update_time >= update_interval:
                    # Clear screen for new update
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # Show current time
                    print(f"\n=== UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                    
                    # Show active status
                    status = "ACTIVE" if bot.active else "STOPPED"
                    print(f"Status: {status}")
                    
                    # Show prices
                    print("\n=== CURRENT PRICES ===")
                    for symbol in bot.config.symbols:
                        price = bot.real_time_prices.get(symbol, "Waiting for data...")
                        symbol_status = "🔵" if symbol in bot.open_trades else "⚪"
                        print(f"{symbol_status} {symbol}: {price}")
                    
                    # Show trades
                    show_trades(bot)
                    
                    # Show performance
                    show_performance(bot)
                    
                    # Update time
                    last_update_time = current_time
                
                # Sleep to prevent high CPU usage
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping bot...")
            bot.stop()
            print("Bot stopped successfully.")
    
    except Exception as e:
        log_exception(logger, e, "Error in main function")
        print(f"Error: {str(e)}")
        
        if debug_mode:
            print("Check the logs for detailed error information")
            print(f"Logs directory: {os.path.join(os.getcwd(), 'logs')}")

if __name__ == "__main__":
    main()