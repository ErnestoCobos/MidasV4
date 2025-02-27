import os
import logging
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from config import Config
from bot import ScalpingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('WebInterface')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global variables
bot = None
config = None
bot_status = "stopped"
bot_thread = None

# Create charts directory if it doesn't exist
os.makedirs("static/charts", exist_ok=True)

#######################
# Bot Control Functions
#######################

def initialize_bot(cfg):
    """Initialize the bot with configuration"""
    global bot, config
    config = cfg
    bot = ScalpingBot(config)
    return bot

def start_bot_thread():
    """Start the bot in a separate thread"""
    global bot, bot_status, bot_thread
    
    if bot_status == "running":
        logger.warning("Bot is already running")
        return False
    
    try:
        if bot is None:
            logger.error("Bot has not been initialized")
            return False
        
        # Start the bot
        bot.start()
        bot_status = "running"
        
        # Start monitoring thread
        bot_thread = threading.Thread(target=monitor_bot, daemon=True)
        bot_thread.start()
        
        logger.info("Bot started successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}")
        bot_status = "error"
        return False

def stop_bot():
    """Stop the bot"""
    global bot, bot_status
    
    if bot_status != "running":
        logger.warning("Bot is not running")
        return False
    
    try:
        bot.stop()
        bot_status = "stopped"
        logger.info("Bot stopped successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to stop bot: {str(e)}")
        bot_status = "error"
        return False

def monitor_bot():
    """Monitor bot status and performance in background"""
    global bot, bot_status
    
    logger.info("Bot monitoring thread started")
    
    while bot_status == "running":
        try:
            # Sleep to prevent high CPU usage
            time.sleep(5)
            
            # Check if bot is still active
            if not bot.active:
                logger.warning("Bot is no longer active, updating status")
                bot_status = "stopped"
                break
        
        except Exception as e:
            logger.error(f"Error in bot monitoring: {str(e)}")
            bot_status = "error"
            break
    
    logger.info("Bot monitoring thread ended")

#######################
# Chart Generation
#######################

def generate_price_chart(symbol, klines, trades=None):
    """Generate interactive price chart using Plotly"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': k['open_time'],
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
        
        # Create subplots: price and volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{symbol} Price", "Volume"))
        
        # Add price line
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['close'], name='Price', line=dict(color='black')),
            row=1, col=1
        )
        
        # Add indicators
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['sma_7'], name='SMA 7', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['sma_25'], name='SMA 25', line=dict(color='red', width=1)),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper',
                       line=dict(color='green', width=1, dash='dash')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower',
                       line=dict(color='green', width=1, dash='dash'),
                       fill='tonexty', fillcolor='rgba(0,128,0,0.1)'),
            row=1, col=1
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker=dict(color='blue', opacity=0.5)),
            row=2, col=1
        )
        
        # Add trades if available
        if trades:
            buy_x, buy_y, buy_texts = [], [], []
            sell_x, sell_y, sell_texts = [], [], []
            
            for trade in trades:
                try:
                    # Get entry/exit details
                    entry_time = trade['time_opened']
                    exit_time = trade['time_closed']
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price']
                    
                    # Format trade info
                    profit_pct = ((exit_price - entry_price) / entry_price) * 100 if trade['side'] == 'BUY' else ((entry_price - exit_price) / entry_price) * 100
                    trade_text = f"P/L: {trade['profit_loss']:.8f} ({profit_pct:.2f}%)"
                    
                    if trade['side'] == 'BUY':
                        buy_x.append(entry_time)
                        buy_y.append(entry_price)
                        buy_texts.append(f"BUY at {entry_price:.2f}")
                        
                        # Add exit point color based on profit/loss
                        color = 'green' if trade['profit_loss'] > 0 else 'red'
                        fig.add_trace(
                            go.Scatter(x=[exit_time], y=[exit_price], mode='markers',
                                       marker=dict(symbol='triangle-down', size=12, color=color),
                                       name=f"Exit {exit_time.strftime('%H:%M:%S')}",
                                       text=trade_text, hoverinfo='text'),
                            row=1, col=1
                        )
                    else:  # SELL
                        sell_x.append(entry_time)
                        sell_y.append(entry_price)
                        sell_texts.append(f"SELL at {entry_price:.2f}")
                        
                        # Add exit point color based on profit/loss
                        color = 'green' if trade['profit_loss'] > 0 else 'red'
                        fig.add_trace(
                            go.Scatter(x=[exit_time], y=[exit_price], mode='markers',
                                       marker=dict(symbol='triangle-up', size=12, color=color),
                                       name=f"Exit {exit_time.strftime('%H:%M:%S')}",
                                       text=trade_text, hoverinfo='text'),
                            row=1, col=1
                        )
                
                except Exception as e:
                    logger.warning(f"Couldn't plot trade: {str(e)}")
            
            # Add buy points
            if buy_x:
                fig.add_trace(
                    go.Scatter(x=buy_x, y=buy_y, mode='markers',
                               marker=dict(symbol='triangle-up', size=12, color='green'),
                               name='Buy Entries', text=buy_texts, hoverinfo='text'),
                    row=1, col=1
                )
            
            # Add sell points
            if sell_x:
                fig.add_trace(
                    go.Scatter(x=sell_x, y=sell_y, mode='markers',
                               marker=dict(symbol='triangle-down', size=12, color='red'),
                               name='Sell Entries', text=sell_texts, hoverinfo='text'),
                    row=1, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Analysis',
            showlegend=True,
            height=800,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        # Update Y-axes
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        # Create a temporary HTML file
        filename = f"static/charts/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        
        return filename
    
    except Exception as e:
        logger.error(f"Error generating price chart: {str(e)}")
        return None

def generate_performance_chart(trades_history):
    """Generate performance chart using Plotly"""
    try:
        if not trades_history:
            return None
        
        # Create DataFrame
        df = pd.DataFrame([{
            'time': trade['time_closed'],
            'profit_loss': trade['profit_loss'],
            'symbol': trade['symbol'],
            'side': trade['side'],
            'reason': trade.get('reason', 'unknown')
        } for trade in trades_history])
        
        # Sort by time
        df = df.sort_values('time')
        
        # Calculate cumulative P/L
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        
        # Create color array based on profit/loss
        df['color'] = df['profit_loss'].apply(lambda x: 'green' if x > 0 else 'red')
        
        # Create figure
        fig = go.Figure()
        
        # Add cumulative P/L line
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['cumulative_pnl'], mode='lines+markers',
                       name='Cumulative P/L', line=dict(color='blue', width=2),
                       marker=dict(color=df['color'], size=8))
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title='Trading Performance',
            xaxis_title='Time',
            yaxis_title='Profit/Loss',
            showlegend=True,
            height=600,
            template='plotly_white'
        )
        
        # Create a temporary HTML file
        filename = f"static/charts/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        
        return filename
    
    except Exception as e:
        logger.error(f"Error generating performance chart: {str(e)}")
        return None

def generate_distribution_chart(trades_history):
    """Generate win/loss distribution chart using Plotly"""
    try:
        if not trades_history:
            return None
        
        # Extract profit/loss values
        profits = [t['profit_loss'] for t in trades_history if t['profit_loss'] > 0]
        losses = [t['profit_loss'] for t in trades_history if t['profit_loss'] <= 0]
        
        # Create figure
        fig = go.Figure()
        
        # Add profits histogram
        if profits:
            fig.add_trace(
                go.Histogram(x=profits, name='Profits', marker_color='green', opacity=0.7,
                             autobinx=True, nbinsx=10)
            )
        
        # Add losses histogram
        if losses:
            fig.add_trace(
                go.Histogram(x=losses, name='Losses', marker_color='red', opacity=0.7,
                             autobinx=True, nbinsx=10)
            )
        
        # Update layout
        fig.update_layout(
            title='Profit/Loss Distribution',
            xaxis_title='Profit/Loss',
            yaxis_title='Frequency',
            barmode='overlay',
            showlegend=True,
            height=500,
            template='plotly_white'
        )
        
        # Create a temporary HTML file
        filename = f"static/charts/distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(filename)
        
        return filename
    
    except Exception as e:
        logger.error(f"Error generating distribution chart: {str(e)}")
        return None

#######################
# Flask Routes
#######################

@app.route('/')
def index():
    """Main dashboard route"""
    global bot, bot_status
    
    # If bot is not initialized, redirect to setup
    if bot is None:
        return redirect(url_for('setup'))
    
    # Get performance data if bot is running
    performance = bot.get_performance_summary() if bot and bot_status == "running" else {
        'total_trades': 0,
        'profitable_trades': 0,
        'win_rate': 0,
        'total_profit_loss': 0,
        'open_trades': 0,
        'best_symbols': []
    }
    
    # Get currently monitored symbols
    symbols = config.symbols if config else []
    
    # Get real-time prices if available
    prices = {}
    if bot and bot_status == "running":
        prices = {s: bot.real_time_prices.get(s, "Waiting for data...") for s in symbols}
    
    # Get open trades if available
    open_trades = {}
    if bot and bot_status == "running":
        open_trades = bot.open_trades
    
    return render_template('dashboard.html', 
                          bot_status=bot_status,
                          performance=performance,
                          symbols=symbols,
                          prices=prices,
                          open_trades=open_trades)

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Bot setup route"""
    global config
    
    if request.method == 'POST':
        try:
            # Get form data
            api_key = request.form.get('api_key', '')
            api_secret = request.form.get('api_secret', '')
            symbols = request.form.get('symbols', 'BTCUSDT').split(',')
            max_capital_risk = float(request.form.get('max_capital_risk', 2.0))
            stop_loss = float(request.form.get('stop_loss', 0.5))
            take_profit = float(request.form.get('take_profit', 1.0))
            model_type = request.form.get('model_type', 'indicator')
            simulation = 'simulation' in request.form
            
            # Create config
            if simulation:
                api_key = "simulation_mode_key"
                api_secret = "simulation_mode_secret"
            
            config = Config(
                api_key=api_key,
                api_secret=api_secret,
                symbols=symbols,
                max_capital_risk_percent=max_capital_risk,
                stop_loss_percent=stop_loss,
                take_profit_percent=take_profit,
                model_type=model_type
            )
            
            # Initialize bot
            initialize_bot(config)
            
            # Redirect to dashboard
            return redirect(url_for('index'))
        
        except Exception as e:
            logger.error(f"Error in setup: {str(e)}")
            return render_template('setup.html', error=str(e))
    
    return render_template('setup.html')

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the bot"""
    if start_bot_thread():
        return jsonify({'status': 'success', 'message': 'Bot started successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start bot'})

@app.route('/stop_bot', methods=['POST'])
def stop_bot_route():
    """Stop the bot"""
    if stop_bot():
        return jsonify({'status': 'success', 'message': 'Bot stopped successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to stop bot'})

@app.route('/bot_status')
def get_bot_status():
    """Get current bot status"""
    global bot, bot_status
    
    status_data = {
        'status': bot_status,
        'active_since': None,
        'symbols': [],
        'open_trades': 0
    }
    
    if bot and bot_status == "running":
        # Add more status details
        status_data['symbols'] = config.symbols
        status_data['open_trades'] = len(bot.open_trades)
        
        if bot.trades_history:
            status_data['active_since'] = bot.trades_history[0]['time_opened'].strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify(status_data)

@app.route('/performance')
def performance():
    """View performance data"""
    global bot, bot_status
    
    if bot is None or bot_status != "running":
        return jsonify({'status': 'error', 'message': 'Bot is not running'})
    
    # Get performance data
    performance = bot.get_performance_summary()
    
    # Generate performance chart
    chart_file = None
    if bot.trades_history:
        chart_file = generate_performance_chart(bot.trades_history)
    
    # Generate distribution chart
    distribution_file = None
    if len(bot.trades_history) >= 5:
        distribution_file = generate_distribution_chart(bot.trades_history)
    
    return render_template('performance.html',
                          performance=performance,
                          chart_file=chart_file,
                          distribution_file=distribution_file)

@app.route('/trades')
def trades():
    """View trades history"""
    global bot, bot_status
    
    if bot is None or bot_status != "running":
        return jsonify({'status': 'error', 'message': 'Bot is not running'})
    
    # Get trades history (most recent first)
    trades_history = sorted(bot.trades_history, key=lambda x: x['time_closed'], reverse=True)
    
    # Get open trades
    open_trades = bot.open_trades
    
    return render_template('trades.html',
                          trades_history=trades_history,
                          open_trades=open_trades)

@app.route('/charts/<symbol>')
def symbol_chart(symbol):
    """View chart for a specific symbol"""
    global bot, bot_status
    
    if bot is None or bot_status != "running":
        return jsonify({'status': 'error', 'message': 'Bot is not running'})
    
    # Validate symbol
    if symbol not in config.symbols:
        return redirect(url_for('index'))
    
    try:
        # Get historical data
        klines = bot.binance_client.get_klines(
            symbol=symbol, 
            interval=bot.config.timeframe,
            limit=100
        )
        
        # Get trades for this symbol
        symbol_trades = [t for t in bot.trades_history if t['symbol'] == symbol]
        
        # Generate chart
        chart_file = generate_price_chart(symbol, klines, symbol_trades)
        
        # Get current price
        current_price = bot.real_time_prices.get(symbol, "Waiting for data...")
        
        # Check if there's an open trade for this symbol
        open_trade = bot.open_trades.get(symbol)
        
        return render_template('chart.html',
                              symbol=symbol,
                              chart_file=chart_file,
                              current_price=current_price,
                              open_trade=open_trade)
    
    except Exception as e:
        logger.error(f"Error generating chart for {symbol}: {str(e)}")
        return redirect(url_for('index'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """View and modify bot settings"""
    global bot, config, bot_status
    
    if bot is None:
        return redirect(url_for('setup'))
    
    if request.method == 'POST':
        try:
            # Check if we need to restart the bot
            restart_needed = False
            if bot_status == "running":
                stop_bot()
                restart_needed = True
            
            # Update settings
            stop_loss = float(request.form.get('stop_loss', config.stop_loss_percent))
            take_profit = float(request.form.get('take_profit', config.take_profit_percent))
            max_risk = float(request.form.get('max_risk', config.max_capital_risk_percent))
            
            # Update config
            config.stop_loss_percent = stop_loss
            config.take_profit_percent = take_profit
            config.max_capital_risk_percent = max_risk
            
            # Restart bot if needed
            if restart_needed:
                start_bot_thread()
            
            return redirect(url_for('settings', updated=True))
        
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return render_template('settings.html',
                                 config=config,
                                 error=str(e))
    
    # Show settings form
    updated = request.args.get('updated', False)
    return render_template('settings.html',
                         config=config,
                         updated=updated)

#######################
# Start the Server
#######################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)