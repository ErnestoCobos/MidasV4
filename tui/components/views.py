from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.widgets import Static, DataTable, ContentSwitcher, Label

from .panels import StatusPanel, PricesPanel, OpenTradesPanel, HistoryPanel, StatsPanel
from .charts import AsciiChart, MultiChart

# Vista Dashboard
class DashboardView(Vertical):
    def on_mount(self):
        # Sección de gráficos arriba
        symbols = []
        if hasattr(self.app.bot, 'config') and hasattr(self.app.bot.config, 'symbols'):
            symbols = self.app.bot.config.symbols[:2]  # Limitar a los primeros 2 símbolos
        
        charts = MultiChart(symbols=symbols)
        self.mount(charts)
        
        # Contenedor para las tablas
        tables = Horizontal()
        
        # Panel izquierdo con precios
        prices_container = Vertical()
        prices_container.mount(Label("Precios en tiempo real", classes="table-title"))
        prices_container.mount(PricesPanel(id="dashboard_prices"))
        tables.mount(prices_container)
        
        # Panel derecho con operaciones abiertas
        trades_container = Vertical()
        trades_container.mount(Label("Operaciones abiertas", classes="table-title"))
        trades_container.mount(OpenTradesPanel(id="dashboard_trades"))
        tables.mount(trades_container)
        
        self.mount(tables)

# Vista de Precios y Gráficos
class PricesView(Vertical):
    def on_mount(self):
        # Sección de gráficos
        symbols = []
        if hasattr(self.app.bot, 'config') and hasattr(self.app.bot.config, 'symbols'):
            symbols = self.app.bot.config.symbols[:4]  # Limitar a 4 gráficos
        
        self.mount(Label("Gráficos de Precios", classes="table-title"))
        
        # Varios gráficos en una fila
        charts = MultiChart(symbols=symbols)
        self.mount(charts)
        
        # Tabla completa de precios
        self.mount(Label("Todos los Precios", classes="table-title"))
        prices_table = PricesPanel(id="full_prices")
        self.mount(prices_table)

# Vista de Operaciones (abiertas y cerradas)
class TradesView(VerticalScroll):
    def on_mount(self):
        # Operaciones abiertas
        self.mount(Label("Operaciones Abiertas", classes="table-title"))
        open_trades_table = OpenTradesPanel(id="full_open_trades")
        self.mount(open_trades_table)
        
        # Historial de operaciones cerradas
        self.mount(Label("Historial de Operaciones", classes="table-title"))
        history_table = HistoryPanel(id="full_history")
        self.mount(history_table)

# Vista de Estadísticas
class StatsView(VerticalScroll):
    def on_mount(self):
        # Panel de estadísticas
        self.mount(Label("Estadísticas de Rendimiento", classes="table-title"))
        stats_panel = StatsPanel(id="performance_stats")
        self.mount(stats_panel)
        
        # Incluir también historial para referencia
        self.mount(Label("Historial de Operaciones", classes="table-title"))
        history_table = HistoryPanel(id="stats_history")
        self.mount(history_table)
        
        # Actualizar stats al montar
        stats_panel.update_stats()

# Vista de Configuración
class ConfigView(VerticalScroll):
    def on_mount(self):
        # Primero mostrar información del bot
        self.mount(Label("Configuración del Bot", classes="table-title"))
        
        config_info = Static("Cargando configuración...")
        self.mount(config_info)
        
        # Actualizar la información de configuración
        bot = self.app.bot
        config_text = "No se pudo cargar la configuración"
        
        if hasattr(bot, 'config'):
            config = bot.config
            config_items = []
            
            # Obtener atributos de configuración
            for attr in dir(config):
                # Omitir métodos y atributos privados
                if not attr.startswith('_') and not callable(getattr(config, attr)):
                    value = getattr(config, attr)
                    # Formatear listas y diccionarios para mejor visualización
                    if isinstance(value, list):
                        value = ', '.join(map(str, value))
                    elif isinstance(value, dict):
                        value = ', '.join(f"{k}: {v}" for k, v in value.items())
                    config_items.append(f"{attr}: {value}")
            
            config_text = "\n".join(config_items)
        
        config_info.update(config_text)