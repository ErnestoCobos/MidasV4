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
        
        # Intentar obtener el balance
        balance = "N/A"
        if hasattr(bot, 'get_balance'):
            try:
                balance = bot.get_balance()
                balance = f"{balance:.2f} USDT"
            except:
                pass
        
        text = f"Estado: {estado} | Modo: {modo} | Balance: {balance}"
        self.update(text)  # actualiza el texto mostrado en el widget

# Panel de precios en vivo
class PricesPanel(DataTable):
    def compose(self):
        # Configurar columnas
        self.add_columns("Par", "Precio", "Cambio")
        self.cursor_type = "row"  # permitir seleccionar filas
        
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
                    except Exception:
                        pass
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
                    except Exception:
                        pass
            
            # Eliminar filas de trades que ya no están abiertos (cerrados)
            for row_key in row_keys:
                if row_key not in map(str, bot_trades.keys()):
                    try:
                        self.remove_row(row_key)
                    except Exception:
                        pass
        except Exception:
            # Ignorar errores generales durante la actualización
            pass

# Panel de historial de operaciones cerradas
class HistoryPanel(DataTable):
    def compose(self):
        # Configurar columnas
        self.add_columns("Par", "Entrada", "Salida", "P/L", "Fecha")
        self.cursor_type = "row"
    
    def on_mount(self):
        # Inicializar con datos existentes
        self._initialize_rows()
    
    def _initialize_rows(self):
        """Añadir filas iniciales del historial de operaciones"""
        if hasattr(self.app.bot, 'trades_history'):
            for index, trade in enumerate(self.app.bot.trades_history):
                try:
                    self._add_history_row(trade, index)
                except Exception:
                    # Ignorar errores al agregar filas iniciales
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
            return
        
        try:
            # Contar trades actuales en la tabla
            current_count = self.row_count
            
            # Verificar si hay trades nuevos para añadir
            if len(self.app.bot.trades_history) > current_count:
                new_trades = self.app.bot.trades_history[current_count:]
                for i, trade in enumerate(new_trades):
                    try:
                        self._add_history_row(trade, current_count + i)
                    except Exception:
                        # Ignorar errores al añadir nuevas filas
                        pass
        except Exception:
            # Ignorar errores generales durante la actualización
            pass

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
            avg_win = stats.get("avg_win", 0)
            avg_loss = stats.get("avg_loss", 0)
            
            text = (f"📊 Win Rate: {win_rate:.1f}%\n"
                    f"💰 P/L Total: {total_pl:.8f} USDT\n"
                    f"📈 Ganancia Promedio: {avg_win:.8f} | 📉 Pérdida Promedio: {avg_loss:.8f}\n"
                    f"🔄 Trades totales: {stats.get('total_trades', 0)}\n"
                    f"✅ Trades ganadores: {stats.get('profitable_trades', 0)}")
            
            self.update(text)
        except Exception as e:
            self.update(f"Error al actualizar estadísticas: {str(e)}")