import logging
import argparse
import json
import os
import time
from config import Config
from bot import ScalpingBot
from visualization import TradingVisualizer

# Configure logging
# File handler for all logs
file_handler = logging.FileHandler("bot.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Console handler only for critical and error logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # Only show errors in console
console_handler.setFormatter(logging.Formatter('❌ %(message)s'))

# Configure root logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

logger = logging.getLogger('Main')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Midas Scalping Bot v4')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON configuration file')
    
    parser.add_argument('--testnet', action='store_true',
                        help='Use Binance testnet instead of real exchange')
    
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of trading pairs (e.g., "BTCUSDT,ETHUSDT")')
    
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtest mode instead of live trading')
                        
    parser.add_argument('--simulate', action='store_true',
                        help='Run in simulation mode without connecting to exchange API')
                        
    parser.add_argument('--real-data', action='store_true',
                        help='Use real market data from Binance API in simulation mode')
                        
    parser.add_argument('--sim-balance', type=str, default=None,
                        help='Initial balance for simulation (format: "USDT:10000,BTC:0.5,ETH:5")')
    
    # Model-related arguments
    parser.add_argument('--model', type=str, default=None,
                        help='Model type to use (xgboost, lstm, indicator)')
    
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model file to load')
    
    # GPU-related arguments
    parser.add_argument('--gpu', action='store_true', 
                        help='Enable GPU acceleration if available')
    
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration even if available')
    
    parser.add_argument('--gpu-device', type=int, default=None,
                        help='GPU device ID to use (for systems with multiple GPUs)')
                        
    # Training-related arguments
    parser.add_argument('--auto-train', action='store_true',
                        help='Enable automatic model training')
                        
    parser.add_argument('--no-train', action='store_true',
                        help='Disable automatic model training')
                        
    parser.add_argument('--train-interval', type=int, default=None,
                        help='Training interval in hours (default: 24)')
                        
    parser.add_argument('--train-now', action='store_true',
                        help='Train model immediately on startup')
    
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

def show_welcome():
    """Display welcome screen with ASCII art logo"""
    # Clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    logo = """
    ███╗   ███╗██╗██████╗  █████╗ ███████╗    ██╗   ██╗██╗  ██╗
    ████╗ ████║██║██╔══██╗██╔══██╗██╔════╝    ██║   ██║██║  ██║
    ██╔████╔██║██║██║  ██║███████║███████╗    ██║   ██║███████║
    ██║╚██╔╝██║██║██║  ██║██╔══██║╚════██║    ╚██╗ ██╔╝╚════██║
    ██║ ╚═╝ ██║██║██████╔╝██║  ██║███████║     ╚████╔╝      ██║
    ╚═╝     ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝╚══════╝      ╚═══╝       ╚═╝
                                                              
    🚀 SCALPING BOT PARA BINANCE 🚀
    """
    
    print(logo)
    print("="*72)
    print("  Desarrollado para trading de alta frecuencia en mercados de criptomonedas")
    print("  Versión: 4.0.0 - Testnet Enabled")
    print("  Estrategia: RSI + Bollinger Bands + SMA Crossovers")
    print("="*72)
    print("\nIniciando sistema...")
    print()
    
def main():
    """Main entry point for the bot"""
    # Show welcome screen
    show_welcome()
    
    args = parse_arguments()
    
    try:
        # Load configuration
        print("📋 Cargando configuración...")
        if args.config:
            config = load_config_from_file(args.config)
            print(f"   ✅ Configuración cargada desde: {args.config}")
        else:
            config = Config.from_env()
            print("   ✅ Configuración cargada desde variables de entorno")
        
        # Override with command line arguments
        if args.symbols:
            config.symbols = args.symbols.split(',')
            print(f"   ✅ Pares a monitorear: {', '.join(config.symbols)}")
        
        # Apply model-related arguments
        if args.model:
            config.model_type = args.model
            print(f"   ✅ Tipo de modelo: {config.model_type}")
        
        # Apply GPU-related arguments
        if args.gpu and args.no_gpu:
            print("   ⚠️ Opciones --gpu y --no-gpu mutuamente excluyentes. Usando configuración por defecto.")
        elif args.gpu:
            config.use_gpu = True
            print("   ✅ Aceleración GPU activada")
        elif args.no_gpu:
            config.use_gpu = False
            print("   ✅ Aceleración GPU desactivada")
            
        if args.gpu_device is not None:
            config.gpu_device = args.gpu_device
            config.gpu_id = args.gpu_device
            if config.use_gpu:
                print(f"   ✅ Dispositivo GPU: {config.gpu_device}")
        
        # Apply training-related arguments
        if args.auto_train:
            config.auto_train = True
            print("   ✅ Entrenamiento automático activado")
            
            if args.train_interval:
                config.training_interval_hours = args.train_interval
                print(f"   ✅ Intervalo de entrenamiento: {config.training_interval_hours} horas")
        elif args.no_train:
            config.auto_train = False
            print("   ✅ Entrenamiento automático desactivado")
        
        # Validate configuration (skip API validation in simulation mode)
        if not args.simulate and not config.validate():
            print("   ❌ Configuración inválida. Verifica tus claves API y parámetros.")
            logger.error("Invalid configuration. Please check your API keys and trading parameters.")
            return
            
        # Handle simulation and testnet modes
        if args.simulate:
            config.api_key = "simulation_mode_key"
            config.api_secret = "simulation_mode_secret"
            config.simulation_mode = True
            
            # Configure real market data in simulation mode if requested
            if args.real_data:
                config.use_real_market_data = True
                print("   ✅ Modo simulación activado (con datos reales de mercado)")
            else:
                config.use_real_market_data = False
                print("   ✅ Modo simulación activado (con datos simulados)")
                
            # Configure custom initial balance if provided
            if args.sim_balance:
                try:
                    sim_balance = {}
                    for balance_str in args.sim_balance.split(','):
                        asset, amount = balance_str.split(':')
                        sim_balance[asset.strip()] = float(amount.strip())
                    
                    config.sim_initial_balance = sim_balance
                    print(f"   ✅ Balance inicial simulado configurado: {sim_balance}")
                except Exception as e:
                    print(f"   ⚠️ Error al configurar balance inicial simulado: {str(e)}")
                    logger.warning(f"Error parsing simulation balance: {str(e)}")
        elif args.testnet:
            print("   ✅ Modo testnet activado (conectado a Binance Testnet)")
        
        # Initialize and start the bot
        print("\n🚀 Inicializando MidasScalpingBot v4...")
        logger.info("Initializing MidasScalpingBot v4...")
        
        if args.backtest:
            # Backtest mode not implemented yet
            logger.error("Backtest mode not implemented yet")
            return
        
        # Load model if path specified
        if args.model_path:
            try:
                from models.model_factory import ModelFactory
                print(f"   ⏳ Cargando modelo desde {args.model_path}...")
                model_type = args.model or config.model_type
                model = ModelFactory.load_model(model_type, config, args.model_path)
                print(f"   ✅ Modelo {model_type} cargado exitosamente")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                print(f"   ❌ Error al cargar modelo: {str(e)}")
                return
        
        # Check for GPU acceleration if enabled
        if config.use_gpu:
            try:
                from models.model_factory import ModelFactory
                tf_gpu, xgb_gpu = ModelFactory.is_gpu_available()
                if tf_gpu or xgb_gpu:
                    print("   ✅ Hardware GPU detectado y configurado para aceleración")
                else:
                    print("   ⚠️ GPU solicitada pero no detectada. Usando CPU.")
            except Exception as e:
                logger.warning(f"Error checking GPU: {str(e)}")
        
        # Create and start the bot
        bot = ScalpingBot(config)
        print("   ✅ Iniciando bot...")
        
        # Train model immediately if requested
        if args.train_now:
            print("   ⏳ Iniciando entrenamiento inmediato del modelo...")
            try:
                # Crear un loop asyncio para ejecutar el entrenamiento
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Entrenar modelo
                train_result = loop.run_until_complete(
                    bot.model_trainer.train_model(
                        model_type=config.model_type,
                        symbol=config.symbols[0],  # Por ahora entrenar para el primer símbolo
                        force_train=True
                    )
                )
                loop.close()
                
                # Mostrar resultado
                if train_result.get('status') == 'success':
                    accuracy = train_result['evaluation_metrics']['direction_accuracy'] * 100
                    print(f"   ✅ Entrenamiento completado. Precisión: {accuracy:.2f}%")
                    print(f"   ✅ Modelo guardado en: {train_result['model_path']}")
                else:
                    print(f"   ❌ Error en entrenamiento: {train_result.get('reason', 'desconocido')}")
            except Exception as e:
                print(f"   ❌ Error iniciando entrenamiento: {str(e)}")
                logger.error(f"Error starting immediate training: {str(e)}")
        
        # Start the bot
        bot.start()
        
        logger.info(f"Bot successfully started and monitoring: {', '.join(config.symbols)}")
        print(f"   ✅ Monitoreando: {', '.join(config.symbols)}")
        print(f"   ✅ Stop Loss: {config.stop_loss_percent}%")
        print(f"   ✅ Take Profit: {config.take_profit_percent}%")
        print(f"   ✅ Riesgo por operación: {config.max_capital_risk_percent}%")
        print("\n🎯 Bot iniciado correctamente. Presiona ENTER para continuar al menú principal...")
        input()
        
        try:
            # Run indefinitely with an improved CLI interface
            while True:
                # Clear the screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Show header
                print("\n" + "="*72)
                print("🚀 MIDAS SCALPING BOT v4 - MENU PRINCIPAL 🚀".center(72))
                print("="*72)
                print("\nEl bot está ejecutándose en segundo plano...")
                print("\nCOMANDOS DISPONIBLES:")
                print("1. 📊 Ver estadísticas")
                print("2. 💰 Ver balance")
                print("3. 📈 Ver pares monitoreados")
                print("4. 📜 Historial de operaciones")
                print("5. 📊 Visualizar gráficos")
                print("6. ⚙️  Ajustar configuración")
                print("7. 🔄 Reiniciar bot")
                print("q. 🛑 Salir")
                
                input_cmd = input("\n>> Selecciona una opción: ")
                
                if input_cmd.lower() == 'q':
                    print("\n🛑 Deteniendo el bot. Por favor espera...")
                    break
                
                elif input_cmd == '1':
                    print("\n📊 ESTADÍSTICAS DE RENDIMIENTO:")
                    stats = bot.get_performance_summary()
                    
                    print("\n  RESUMEN GENERAL:")
                    print(f"  • Operaciones totales: {stats['total_trades']}")
                    print(f"  • Operaciones rentables: {stats['profitable_trades']}")
                    print(f"  • Tasa de éxito: {stats['win_rate']:.2f}%")
                    print(f"  • Ganancia/Pérdida total: {stats['total_profit_loss']:.8f}")
                    print(f"  • Operaciones abiertas: {stats['open_trades']}")
                    
                    if stats['total_trades'] > 0:
                        print("\n  MÉTRICAS AVANZADAS:")
                        print(f"  • Ganancia media: {stats['avg_profit']:.8f}")
                        print(f"  • Pérdida media: {stats['avg_loss']:.8f}")
                        print(f"  • Factor de beneficio: {stats['profit_factor']:.2f}")
                        print(f"  • Frecuencia: {stats['trades_per_hour']:.2f} ops/hora")
                        
                        if stats['active_since']:
                            print(f"  • Activo desde: {stats['active_since']}")
                            
                        if stats['best_symbols']:
                            print("\n  MEJORES PARES:")
                            for i, symbol_data in enumerate(stats['best_symbols'], 1):
                                symbol = symbol_data['symbol']
                                profit = symbol_data['profit']
                                trades = symbol_data['trades']
                                print(f"  {i}. {symbol}: {profit:.8f} ({trades} operaciones)")
                
                elif input_cmd == '2':
                    print("\n💰 BALANCE DE CUENTA:")
                    try:
                        balances = bot.binance_client.get_account_balance()
                        for asset, amount in balances.items():
                            if amount > 0:
                                print(f"  • {asset}: {amount:.8f}")
                    except Exception as e:
                        print(f"  Error al recuperar balance: {str(e)}")
                
                elif input_cmd == '3':
                    print("\n📈 PARES MONITOREADOS:")
                    for symbol in bot.config.symbols:
                        price = bot.real_time_prices.get(symbol, "Esperando datos...")
                        if symbol in bot.open_trades:
                            trade = bot.open_trades[symbol]
                            print(f"  • {symbol}: ${price} - POSICIÓN ABIERTA ({trade['side']})")
                        else:
                            print(f"  • {symbol}: ${price}")
                
                elif input_cmd == '4':
                    print("\n📜 HISTORIAL DE OPERACIONES:")
                    if not bot.trades_history:
                        print("  No hay operaciones completadas todavía.")
                    else:
                        for i, trade in enumerate(bot.trades_history[-5:], 1):
                            profit = trade['profit_loss']
                            profit_str = f"+{profit:.8f}" if profit >= 0 else f"{profit:.8f}"
                            print(f"  {i}. {trade['symbol']} {trade['side']} - {profit_str} - {trade['reason']}")
                        if len(bot.trades_history) > 5:
                            print(f"  ... y {len(bot.trades_history) - 5} operaciones más")
                
                elif input_cmd == '5':
                    # Initialize visualizer if not already done
                    visualizer = TradingVisualizer()
                    
                    print("\n📊 VISUALIZAR GRÁFICOS:")
                    print("  • 1. Gráfico de par específico")
                    print("  • 2. Gráfico de rendimiento")
                    print("  • 3. Distribución de ganancias/pérdidas")
                    print("  • 0. Volver")
                    
                    visual_cmd = input("\n>> Selecciona una opción: ")
                    
                    if visual_cmd == '1':
                        # Symbol chart
                        print("\nPares disponibles:")
                        for i, symbol in enumerate(bot.config.symbols, 1):
                            print(f"  • {i}. {symbol}")
                        
                        try:
                            selection = int(input("\nSelecciona un par (número): "))
                            if 1 <= selection <= len(bot.config.symbols):
                                selected_symbol = bot.config.symbols[selection-1]
                                print(f"\nGenerando gráfico para {selected_symbol}...")
                                
                                # Get historical data
                                klines = bot.binance_client.get_klines(
                                    symbol=selected_symbol, 
                                    interval=bot.config.timeframe,
                                    limit=100
                                )
                                
                                # Get trades for this symbol
                                symbol_trades = [t for t in bot.trades_history if t['symbol'] == selected_symbol]
                                
                                # Generate chart
                                chart_file = visualizer.plot_price_chart(klines, selected_symbol, symbol_trades)
                                if chart_file:
                                    print(f"✅ Gráfico guardado en: {chart_file}")
                                else:
                                    print("❌ Error al generar el gráfico")
                            else:
                                print("❌ Selección inválida")
                        except ValueError:
                            print("❌ Por favor ingresa un número válido")
                    
                    elif visual_cmd == '2':
                        # Performance chart
                        if not bot.trades_history:
                            print("❌ No hay suficientes operaciones para generar un gráfico")
                        else:
                            print("\nGenerando gráfico de rendimiento...")
                            chart_file = visualizer.plot_performance_chart(bot.trades_history)
                            if chart_file:
                                print(f"✅ Gráfico guardado en: {chart_file}")
                            else:
                                print("❌ Error al generar el gráfico")
                    
                    elif visual_cmd == '3':
                        # Win/loss distribution
                        if not bot.trades_history or len(bot.trades_history) < 5:
                            print("❌ No hay suficientes operaciones para generar una distribución")
                        else:
                            print("\nGenerando gráfico de distribución...")
                            chart_file = visualizer.plot_win_loss_distribution(bot.trades_history)
                            if chart_file:
                                print(f"✅ Gráfico guardado en: {chart_file}")
                            else:
                                print("❌ Error al generar el gráfico")
                
                elif input_cmd == '6':
                    print("\n⚙️  AJUSTAR CONFIGURACIÓN:")
                    print("  • 1. Riesgo por operación (actual: {:.1f}%)".format(bot.config.max_capital_risk_percent))
                    print("  • 2. Stop Loss (actual: {:.1f}%)".format(bot.config.stop_loss_percent))
                    print("  • 3. Take Profit (actual: {:.1f}%)".format(bot.config.take_profit_percent))
                    print("  • 0. Volver")
                    
                    setting_cmd = input("\n>> Selecciona parámetro a modificar: ")
                    
                    if setting_cmd == '1':
                        try:
                            new_value = float(input("Nuevo valor (0.5-5%): "))
                            if 0.5 <= new_value <= 5:
                                bot.config.max_capital_risk_percent = new_value
                                print(f"✅ Riesgo por operación actualizado a {new_value}%")
                            else:
                                print("❌ Valor fuera de rango permitido")
                        except ValueError:
                            print("❌ Por favor ingresa un número válido")
                    
                    elif setting_cmd == '2':
                        try:
                            new_value = float(input("Nuevo valor (0.1-2%): "))
                            if 0.1 <= new_value <= 2:
                                bot.config.stop_loss_percent = new_value
                                print(f"✅ Stop Loss actualizado a {new_value}%")
                            else:
                                print("❌ Valor fuera de rango permitido")
                        except ValueError:
                            print("❌ Por favor ingresa un número válido")
                    
                    elif setting_cmd == '3':
                        try:
                            new_value = float(input("Nuevo valor (0.5-5%): "))
                            if 0.5 <= new_value <= 5:
                                bot.config.take_profit_percent = new_value
                                print(f"✅ Take Profit actualizado a {new_value}%")
                            else:
                                print("❌ Valor fuera de rango permitido")
                        except ValueError:
                            print("❌ Por favor ingresa un número válido")
                
                elif input_cmd == '7':
                    confirm = input("¿Estás seguro de reiniciar el bot? (s/n): ")
                    if confirm.lower() == 's':
                        print("🔄 Reiniciando bot...")
                        bot.stop()
                        time.sleep(2)
                        bot.start()
                        print("✅ Bot reiniciado correctamente")
                
                else:
                    print("❌ Opción no válida. Por favor intenta de nuevo.")
                
                # Pausa para que el usuario pueda leer la salida
                input("\nPresiona ENTER para continuar...")
                
                # Limpiar pantalla al volver al menú principal
                os.system('cls' if os.name == 'nt' else 'clear')
        
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
        
        finally:
            # Ensure bot is properly stopped
            bot.stop()
            logger.info("Bot ha sido detenido correctamente")
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()