from textual.widgets import Static, DataTable
from textual.reactive import reactive

# Panel de estado (Estado/Modo/Balance)
class StatusPanel(Static):
    def on_mount(self):
        # Actualizamos el contenido inicial del panel de estado
        self.update_status()
        
    def update_status(self):
        bot = self.app.bot  # referencia al bot desde la App
        estado = "🟢 Activo" if hasattr(bot, 'active') and bot.active else "🔴 Detenido"
        
        # Determinar el modo del bot
        if hasattr(bot, 'binance_client') and hasattr(bot.binance_client, 'simulation_mode'):
            modo = "🔵 Simulación" if bot.binance_client.simulation_mode else "🔴 Real"
        else:
            modo = "Desconocido"
        
        # Obtener balance total
        balance_text = "N/A"
        if hasattr(bot, 'get_balance'):
            try:
                total_balance = bot.get_balance()
                
                # Mostrar balance con colores según ganancia/pérdida
                initial_balance = 1000.0  # Asumimos 1000 USDT como balance inicial
                if hasattr(bot, 'config') and hasattr(bot.config, 'sim_initial_balance'):
                    initial_balance = bot.config.sim_initial_balance.get('USDT', 1000.0)
                
                profit_loss = total_balance - initial_balance
                change_percent = (profit_loss / initial_balance) * 100 if initial_balance > 0 else 0
                
                # Formatear el balance total
                balance_text = f"{total_balance:.2f} USDT"
                
                # Obtener balances detallados si están disponibles
                if hasattr(bot, 'get_detailed_balances'):
                    try:
                        detailed_balances = bot.get_detailed_balances()
                        detailed_text = " ("
                        for asset, amount in detailed_balances.items():
                            if asset != 'USDT' and amount > 0:
                                detailed_text += f"{asset}: {amount:.4f}, "
                        
                        if detailed_text != " (":
                            balance_text += detailed_text.rstrip(", ") + ")"
                            
                        # Añadir cambio porcentual
                        if change_percent != 0:
                            balance_text += f" | Δ {change_percent:.2f}%"
                    except Exception as e:
                        print(f"Error al obtener balances detallados: {str(e)}")
            except Exception as e:
                print(f"Error al obtener balance total: {str(e)}")
        
        # Generar texto del panel
        text = f"Estado: {estado} | Modo: {modo} | Balance: {balance_text}"
        
        self.update(text)  # actualiza el texto mostrado en el widget

# Panel de precios en vivo
class PricesPanel(DataTable):
    def compose(self):
        # Configurar columnas
        self.add_columns("Par", "Precio", "Cambio")
        self.cursor_type = "row"  # permitir seleccionar filas
        # DataTable debe devolver un iterable vacío
        return []
        
    def on_mount(self):
        # Inicializar con datos iniciales
        self._initialize_rows()
    
    def _initialize_rows(self):
        """Inicializar filas con datos disponibles"""
        if hasattr(self.app.bot, 'real_time_prices'):
            for pair, price in self.app.bot.real_time_prices.items():
                # Agregar fila con key del par
                try:
                    self.add_row(pair, f"{price:.8f}", "0.00%", key=pair)
                except Exception as e:
                    # Ignorar errores al agregar filas iniciales
                    pass
        
    def refresh_prices(self):
        """Actualizar precios en tiempo real"""
        # Verificar que bot tenga datos de precios
        if not hasattr(self.app.bot, 'real_time_prices'):
            return
        
        # Procesar cada par de trading
        for pair, price in self.app.bot.real_time_prices.items():
            try:
                # Intentar encontrar la fila existente
                row_keys = list(self.rows.keys())
                row_exists = pair in row_keys
                
                # Crear fila si no existe
                if not row_exists:
                    self.add_row(pair, f"{price:.8f}", "0.00%", key=pair)
                else:
                    # Actualizar fila existente
                    # Calcular cambio si tenemos precio anterior
                    current_price = price
                    old_price = self.app.bot.get_previous_price(pair) if hasattr(self.app.bot, 'get_previous_price') else current_price
                    
                    if old_price and old_price > 0:
                        change_pct = ((current_price - old_price) / old_price) * 100
                        change_str = f"{change_pct:+.2f}%" 
                        change_style = "green" if change_pct >= 0 else "red"
                        
                        # Actualizar las celdas
                        self.update_cell(pair, "Precio", f"{current_price:.8f}")
                        self.update_cell(pair, "Cambio", change_str, style=change_style)
                    else:
                        self.update_cell(pair, "Precio", f"{current_price:.8f}")
            except Exception as e:
                # Ignorar errores durante la actualización
                continue

# Panel de operaciones abiertas
class OpenTradesPanel(DataTable):
    def compose(self):
        # Configurar columnas
        self.add_columns("Par", "Entrada", "SL", "TP", "P/L")
        self.cursor_type = "row"
        # DataTable debe devolver un iterable vacío
        return []
    
    def on_mount(self):
        """Inicializar con datos iniciales"""
        self._initialize_rows()
    
    def _initialize_rows(self):
        """Añadir filas iniciales si hay operaciones abiertas"""
        if hasattr(self.app.bot, 'open_trades'):
            for trade_id, trade in self.app.bot.open_trades.items():
                try:
                    self._add_trade_row(trade_id, trade)
                except Exception as e:
                    # Ignorar errores al agregar filas iniciales
                    pass
        
    def _add_trade_row(self, trade_id, trade):
        # Extraer datos del trade con manejo seguro
        pair = trade.get("symbol", "Unknown")
        entry = trade.get("entry_price", 0.0)
        sl = trade.get("stop_loss", 0.0)
        tp = trade.get("take_profit", 0.0)
        pl = trade.get("profit_loss", 0.0)
        
        # Determinar estilo para P/L
        pl_style = "green" if pl >= 0 else "red"
        
        # Usamos trade_id como key de fila
        self.add_row(
            pair, 
            f"{entry:.8f}", 
            f"{sl:.8f}", 
            f"{tp:.8f}", 
            f"{pl:.8f}", 
            key=str(trade_id),
            style=pl_style
        )
    
    def refresh_trades(self):
        """Actualizar operaciones abiertas"""
        # Verificar si bot tiene open_trades
        if not hasattr(self.app.bot, 'open_trades'):
            print("Bot doesn't have open_trades attribute")
            return
            
        try:
            # Sincronizar con bot.open_trades (agregar nuevas, quitar cerradas, actualizar P/L)
            bot_trades = self.app.bot.open_trades
            
            # Obtener claves actuales
            row_keys = list(self.rows.keys())
            
            # Actualizar o agregar
            for trade_id, trade in bot_trades.items():
                trade_id_str = str(trade_id)
                
                if trade_id_str not in row_keys:  # si no existe fila, agregar nueva
                    try:
                        self._add_trade_row(trade_id, trade)
                    except Exception as e:
                        print(f"Error adding trade row {trade_id}: {str(e)}")
                else:
                    # Si ya existe, actualizar P/L y otros campos que puedan cambiar
                    try:
                        pl = trade.get("profit_loss", 0.0)
                        pl_style = "green" if pl >= 0 else "red"
                        
                        # Actualizar campos con nombres de columna
                        self.update_cell(trade_id_str, "P/L", f"{pl:.8f}", style=pl_style)
                        
                        # También actualizamos SL/TP por si fueron modificados
                        sl = trade.get("stop_loss", 0.0) 
                        tp = trade.get("take_profit", 0.0)
                        self.update_cell(trade_id_str, "SL", f"{sl:.8f}")
                        self.update_cell(trade_id_str, "TP", f"{tp:.8f}")
                    except Exception as e:
                        print(f"Error updating trade row {trade_id}: {str(e)}")
            
            # Eliminar filas de trades que ya no están abiertos (cerrados)
            for row_key in row_keys:
                if row_key not in map(str, bot_trades.keys()):
                    try:
                        self.remove_row(row_key)
                    except Exception as e:
                        print(f"Error removing trade row {row_key}: {str(e)}")
        except Exception as e:
            # Registrar errores generales durante la actualización
            print(f"Error refreshing trades panel: {str(e)}")
            import traceback
            print(traceback.format_exc())

# Panel de historial de operaciones cerradas
class HistoryPanel(DataTable):
    def compose(self):
        # Configurar columnas
        self.add_columns("Par", "Entrada", "Salida", "P/L", "Fecha")
        self.cursor_type = "row"
        # DataTable debe devolver un iterable vacío
        return []
    
    def on_mount(self):
        # Inicializar con datos existentes
        self._initialize_rows()
    
    def _initialize_rows(self, max_rows=None):
        """Añadir filas iniciales del historial de operaciones
        
        Args:
            max_rows: Opcional, número máximo de filas a mostrar (más recientes primero)
        """
        if hasattr(self.app.bot, 'trades_history'):
            # Si se especifica max_rows, tomar solo las últimas N operaciones
            trades_to_show = self.app.bot.trades_history
            if max_rows is not None and max_rows > 0:
                trades_to_show = trades_to_show[-max_rows:]
                
            # Inicializar filas
            for index, trade in enumerate(trades_to_show):
                try:
                    self._add_history_row(trade, index)
                except Exception as e:
                    # Registrar errores al agregar filas iniciales
                    print(f"Error al agregar fila de historial {index}: {str(e)}")
                    pass
    
    def _add_history_row(self, trade, row_id=None):
        # Extraer datos del trade con manejo seguro
        pair = trade.get("symbol", "Unknown")
        entry = trade.get("entry_price", 0.0)
        exit_price = trade.get("exit_price", 0.0)
        pl = trade.get("profit_loss", 0.0)
        time_closed = trade.get("time_closed", None)
        
        # Formatear fecha si está disponible
        date_str = time_closed.strftime("%Y-%m-%d %H:%M") if time_closed else "N/A"
        
        # Determinar estilo para P/L
        pl_style = "green" if pl >= 0 else "red"
        
        # Usar row_id como identificador único si se proporciona
        if row_id is not None:
            key = f"history_{row_id}"
        else:
            # Generar un ID único para la fila
            key = f"history_{len(self.rows)}"
        
        # Añadir fila
        self.add_row(
            pair,
            f"{entry:.8f}",
            f"{exit_price:.8f}",
            f"{pl:.8f}",
            date_str,
            key=key,
            style=pl_style
        )
        
    def refresh_history(self):
        """Actualizar historial de operaciones cerradas"""
        # Verificar si el bot tiene trades_history
        if not hasattr(self.app.bot, 'trades_history'):
            print("El bot no tiene trades_history")
            return
        
        try:
            # Contar trades actuales en la tabla
            current_count = self.row_count
            bot_trades_count = len(self.app.bot.trades_history)
            
            print(f"Historial: {current_count} filas en tabla, {bot_trades_count} trades en el bot")
            
            # Si la tabla está vacía pero hay datos en el bot, recrear todas las filas
            if current_count == 0 and bot_trades_count > 0:
                print("Reinicializando todas las filas del historial")
                self._initialize_rows()
                return
                
            # Verificar si hay trades nuevos para añadir
            if bot_trades_count > current_count:
                new_trades = self.app.bot.trades_history[current_count:]
                print(f"Añadiendo {len(new_trades)} nuevos trades al historial")
                
                for i, trade in enumerate(new_trades):
                    try:
                        self._add_history_row(trade, current_count + i)
                    except Exception as e:
                        # Registrar errores al añadir filas
                        print(f"Error añadiendo fila de historial: {str(e)}")
        except Exception as e:
            # Registrar errores durante la actualización
            print(f"Error refrescando historial: {str(e)}")

# Panel de estadísticas de rendimiento
class StatsPanel(Static):
    def update_stats(self):
        if not hasattr(self.app.bot, 'get_performance_summary'):
            self.update("No hay estadísticas disponibles")
            return
            
        try:
            stats = self.app.bot.get_performance_summary()
            
            win_rate = stats.get("win_rate", 0) * 100  # porcentaje
            total_pl = stats.get("total_profit_loss", 0)
            # Corregir nombres de campos para coincidir con los que devuelve el bot
            avg_win = stats.get("avg_profit", 0)  # Nombre correcto en el bot
            avg_loss = stats.get("avg_loss", 0)
            total_trades = stats.get('total_trades', 0)
            open_trades = stats.get('open_trades', 0)
            profitable_trades = stats.get('profitable_trades', 0)
            
            # Debug para verificar campos
            print(f"Stats disponibles: {', '.join(stats.keys())}")
            print(f"Ganancias promedio: {avg_win}, Pérdidas promedio: {avg_loss}")
            
            # Verificar si estamos en modo compacto (para dashboard)
            is_compact = "compact-stats" in self.classes
            
            if is_compact:
                # Versión compacta para panel de dashboard
                text = (f"💰 P/L Total: {'∆' if total_pl >= 0 else '∇'}{total_pl:.2f} USDT | "
                        f"📊 Win Rate: {win_rate:.1f}% | "
                        f"🔄 Trades: {total_trades} ({open_trades} abiertos)")
            else:
                # Versión completa para vista de estadísticas
                profit_factor = stats.get('profit_factor', 0)
                trades_per_hour = stats.get('trades_per_hour', 0)
                best_symbols = stats.get('best_symbols', [])
                active_since = stats.get('active_since', 'N/A')
                
                # Formatear estadísticas detalladas con manejo mejorado de números
                try:
                    win_rate_fmt = f"{win_rate:.1f}%" if win_rate else "N/A"
                    total_pl_fmt = f"{total_pl:.4f} USDT" if total_pl != 0 else "0.0000 USDT"
                    avg_win_fmt = f"{avg_win:.5f}" if avg_win else "0.00000"
                    avg_loss_fmt = f"{avg_loss:.5f}" if avg_loss else "0.00000"
                    profit_factor_fmt = f"{profit_factor:.2f}" if profit_factor else "N/A"
                    trades_per_hour_fmt = f"{trades_per_hour:.2f}" if trades_per_hour else "0.00"
                    
                    text = (f"📊 Win Rate: {win_rate_fmt}\n"
                            f"💰 P/L Total: {total_pl_fmt}\n"
                            f"📈 Ganancia Promedio: {avg_win_fmt} | 📉 Pérdida Promedio: {avg_loss_fmt}\n"
                            f"📊 Factor de Beneficio: {profit_factor_fmt}\n"
                            f"🔄 Trades: {total_trades} totales | {profitable_trades} ganadores | {open_trades} abiertos\n"
                            f"⏱️ Frecuencia: {trades_per_hour_fmt} trades/hora\n\n"
                            f"🏆 Mejores pares:")
                except Exception as fmt_error:
                    print(f"Error formateando estadísticas: {str(fmt_error)}")
                    # Formato más simple como fallback
                    text = (f"📊 Win Rate: {win_rate}\n"
                            f"💰 P/L Total: {total_pl}\n"
                            f"📈 Ganancia Promedio: {avg_win} | 📉 Pérdida Promedio: {avg_loss}\n"
                            f"🔄 Trades: {total_trades} totales")
                
                # Solo añadir mejores símbolos si no estamos en el modo de fallback
                if "Factor de Beneficio" in text and best_symbols:
                    try:
                        for i, symbol_data in enumerate(best_symbols):
                            symbol = symbol_data.get('symbol', 'N/A')
                            profit = symbol_data.get('profit', 0)
                            trades = symbol_data.get('trades', 0)
                            text += f"\n{'  ' if i > 0 else ''}{symbol}: {profit:.5f} USDT ({trades} trades)"
                    except Exception as symbol_err:
                        print(f"Error al formatear mejores pares: {str(symbol_err)}")
                        text += " No se pudieron mostrar"
                elif "Factor de Beneficio" in text:
                    text += " No hay suficientes datos"
                
                # Añadir tiempo activo
                if "Factor de Beneficio" in text:
                    try:
                        text += f"\n\n🕒 Activo desde: {active_since if active_since else 'N/A'}"
                    except Exception:
                        pass
            
            # Aplicar formato Rich
            self.update(text)
        except Exception as e:
            self.update(f"Error al actualizar estadísticas: {str(e)}")