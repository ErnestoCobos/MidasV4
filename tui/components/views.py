from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.widgets import Static, DataTable, ContentSwitcher, Label

from .panels import StatusPanel, PricesPanel, OpenTradesPanel, HistoryPanel, StatsPanel
from .charts import AsciiChart, MultiChart

# Vista Dashboard
class DashboardView(Vertical):
    def compose(self):
        # Sección de gráficos arriba
        symbols = []
        if hasattr(self.app.bot, 'config') and hasattr(self.app.bot.config, 'symbols'):
            symbols = self.app.bot.config.symbols[:2]  # Limitar a los primeros 2 símbolos
        
        yield MultiChart(symbols=symbols)
        
        # Contenedor para las tablas
        with Horizontal() as tables:
            # Panel izquierdo con precios
            with Vertical():
                yield Label("Precios en tiempo real", classes="table-title")
                yield PricesPanel(id="dashboard_prices")
            
            # Panel derecho con operaciones abiertas
            with Vertical():
                yield Label("Operaciones abiertas", classes="table-title")
                yield OpenTradesPanel(id="dashboard_trades")

# Vista de Precios y Gráficos
class PricesView(Vertical):
    def compose(self):
        # Sección de gráficos
        symbols = []
        if hasattr(self.app.bot, 'config') and hasattr(self.app.bot.config, 'symbols'):
            symbols = self.app.bot.config.symbols[:4]  # Limitar a 4 gráficos
        
        yield Label("Gráficos de Precios", classes="table-title")
        
        # Varios gráficos en una fila
        yield MultiChart(symbols=symbols)
        
        # Tabla completa de precios
        yield Label("Todos los Precios", classes="table-title")
        yield PricesPanel(id="full_prices")

# Vista de Operaciones (abiertas y cerradas)
class TradesView(VerticalScroll):
    def compose(self):
        # Operaciones abiertas
        yield Label("Operaciones Abiertas", classes="table-title")
        yield OpenTradesPanel(id="full_open_trades")
        
        # Historial de operaciones cerradas
        yield Label("Historial de Operaciones", classes="table-title")
        yield HistoryPanel(id="full_history")

# Vista de Estadísticas
class StatsView(VerticalScroll):
    def compose(self):
        # Panel de estadísticas
        yield Label("Estadísticas de Rendimiento", classes="table-title")
        yield StatsPanel(id="performance_stats")
        
        # Incluir también historial para referencia
        yield Label("Historial de Operaciones", classes="table-title")
        yield HistoryPanel(id="stats_history")
    
    def on_mount(self):
        # Actualizar stats al montar la vista
        stats_panel = self.query_one(StatsPanel)
        stats_panel.update_stats()

# Vista de Configuración
class ConfigView(VerticalScroll):
    def compose(self):
        # Primero mostrar información del bot
        yield Label("Configuración del Bot", classes="table-title")
        yield Static("Cargando configuración...", id="config_info")
    
    def on_mount(self):
        # Obtener referencia al Static donde mostraremos la configuración
        config_info = self.query_one("#config_info")
        
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