from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.widgets import Static, DataTable, ContentSwitcher, Label

from .panels import StatusPanel, PricesPanel, OpenTradesPanel, HistoryPanel, StatsPanel
from .charts import AsciiChart, MultiChart

# Vista Dashboard
class DashboardView(Vertical):
    def compose(self):
        # Resumen de estado principal (estadísticas clave)
        yield Label("📈 Resumen de Trading", classes="section-title")
        yield StatsPanel(id="dashboard_stats", classes="compact-stats")
        
        # Sección de gráficos arriba
        symbols = []
        if hasattr(self.app.bot, 'config') and hasattr(self.app.bot.config, 'symbols'):
            symbols = self.app.bot.config.symbols[:2]  # Limitar a los primeros 2 símbolos
            
        yield Label("Gráficos en Tiempo Real", classes="table-title")
        yield MultiChart(symbols=symbols)
        
        # Contenedor para las tablas
        with Horizontal() as tables:
            # Panel izquierdo con precios
            with Vertical(classes="dashboard-panel"):
                yield Label("Precios en tiempo real", classes="table-title")
                yield PricesPanel(id="dashboard_prices")
            
            # Panel derecho con operaciones abiertas
            with Vertical(classes="dashboard-panel"):
                yield Label("Operaciones abiertas", classes="table-title")
                yield OpenTradesPanel(id="dashboard_trades")
                
        # Última fila de datos - últimas operaciones
        yield Label("Últimas Operaciones", classes="table-title")
        yield HistoryPanel(id="recent_trades", classes="compact-history")
        
    def on_mount(self):
        """Inicializar componentes al montar vista"""
        try:
            # Actualizar stats
            stats_panel = self.query_one("#dashboard_stats", StatsPanel)
            if stats_panel:
                stats_panel.update_stats()
                
            # Inicializar tabla de historial con número limitado de filas
            history_panel = self.query_one("#recent_trades", HistoryPanel)
            if history_panel:
                history_panel._initialize_rows(max_rows=5)  # Solo las 5 operaciones más recientes
        except Exception as e:
            print(f"Error al inicializar DashboardView: {str(e)}")

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
    
    def on_mount(self):
        """Inicializar componentes al montar la vista"""
        # Forzar inicialización de tablas cuando se monta esta vista
        try:
            history_panel = self.query_one(HistoryPanel)
            if history_panel:
                history_panel._initialize_rows()
                print(f"TradesView: Inicializado historial con {history_panel.row_count} filas")
        except Exception as e:
            print(f"Error inicializando historial en TradesView: {str(e)}")

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
        # Sección de configuración principal
        yield Label("📝 Configuración del Bot", classes="section-title")
        
        # Controles interactivos - por ahora solo para mostrar
        yield Static("[Ctrl+B] Iniciar/Detener Bot", id="config_controls", classes="controls")
        
        # Mostrar secciones de configuración agrupadas
        yield Label("Configuración General", classes="table-title")
        yield Static("Cargando configuración...", id="config_general")
        
        yield Label("Configuración de Trading", classes="table-title")
        yield Static("Cargando configuración...", id="config_trading")
        
        yield Label("Configuración de Conexión", classes="table-title")
        yield Static("Cargando configuración...", id="config_connection")
    
    def on_mount(self):
        # Actualizar la información de configuración
        bot = self.app.bot
        
        if not hasattr(bot, 'config'):
            self.query_one("#config_general").update("No se pudo cargar la configuración")
            return
        
        config = bot.config
        
        # Categorizar la configuración en secciones
        general_config = []
        trading_config = []
        connection_config = []
        
        # Palabras clave para categorizar
        trading_keywords = ['strategy', 'trade', 'position', 'risk', 'sl', 'tp', 'confidence', 
                         'model', 'timeframe', 'symbols', 'leverage', 'margin']
        connection_keywords = ['api', 'secret', 'key', 'url', 'testnet', 'simulation', 'endpoint']
        
        # Obtener atributos de configuración
        for attr in dir(config):
            # Omitir métodos y atributos privados
            if not attr.startswith('_') and not callable(getattr(config, attr)):
                try:
                    value = getattr(config, attr)
                    
                    # Formatear listas y diccionarios para mejor visualización
                    if isinstance(value, list):
                        if len(value) > 5:  # Si la lista es muy larga, truncarla
                            value = ', '.join(map(str, value[:5])) + f" ... (+{len(value)-5} más)"
                        else:
                            value = ', '.join(map(str, value))
                    elif isinstance(value, dict):
                        if len(value) > 5:  # Si el diccionario es muy largo, truncarlo
                            items = list(value.items())[:5]
                            value = ', '.join(f"{k}: {v}" for k, v in items) + f" ... (+{len(value)-5} más)"
                        else:
                            value = ', '.join(f"{k}: {v}" for k, v in value.items())
                    
                    # Formatear como texto Rich con nombres en negrita
                    config_item = f"[bold]{attr}:[/] {value}"
                    
                    # Categorizar la configuración
                    attr_lower = attr.lower()
                    if any(keyword in attr_lower for keyword in trading_keywords):
                        trading_config.append(config_item)
                    elif any(keyword in attr_lower for keyword in connection_keywords):
                        connection_config.append(config_item)
                    else:
                        general_config.append(config_item)
                except Exception:
                    # Ignorar errores al obtener atributos
                    pass
        
        # Actualizar cada sección
        self.query_one("#config_general").update("\n".join(general_config) if general_config else "No hay datos disponibles")
        self.query_one("#config_trading").update("\n".join(trading_config) if trading_config else "No hay datos disponibles")
        self.query_one("#config_connection").update("\n".join(connection_config) if connection_config else "No hay datos disponibles")