from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, ContentSwitcher
from textual.containers import Container
import time

from .components.panels import StatusPanel, PricesPanel, OpenTradesPanel, HistoryPanel, StatsPanel
from .components.views import DashboardView, PricesView, TradesView, StatsView, ConfigView
from .components.charts import MultiChart, AsciiChart

class TradingBotApp(App):
    """Aplicación TUI para el Midas Scalping Bot."""
    
    CSS_PATH = "app.tcss"
    
    BINDINGS = [
        ("d", "switch_view('dashboard')", "Dashboard"),
        ("p", "switch_view('prices')", "Precios"),
        ("t", "switch_view('trades')", "Operaciones"),
        ("s", "switch_view('stats')", "Estadísticas"),
        ("c", "switch_view('config')", "Configuración"),
        ("ctrl+b", "toggle_bot", "Iniciar/Detener Bot"),
        ("q", "quit", "Salir")
    ]

    def __init__(self, bot):
        super().__init__()
        self.bot = bot  # Referencia al bot de trading
        self.title = "Midas Scalping Bot - TUI"
        
    def compose(self) -> ComposeResult:
        """Componer la interfaz de usuario."""
        # Header con reloj
        yield Header(show_clock=True)
        
        # Panel de estado fijo arriba
        yield StatusPanel(id="status_panel")
        
        # Contenedor principal para las vistas
        with Container(id="main_container"):
            # ContentSwitcher para alternar entre vistas
            yield ContentSwitcher(id="main_view", initial="dashboard")
        
        # Footer con atajos de teclado
        yield Footer()

    def on_mount(self):
        """Se ejecuta cuando la aplicación se monta."""
        # Obtener referencia al ContentSwitcher
        switcher = self.query_one("#main_view", ContentSwitcher)
        
        # Añadir las diferentes vistas al content switcher
        switcher.add_content(DashboardView(), id="dashboard")
        switcher.add_content(PricesView(), id="prices")
        switcher.add_content(TradesView(), id="trades")
        switcher.add_content(StatsView(), id="stats")
        switcher.add_content(ConfigView(), id="config")
        
        # Vista inicial
        switcher.current = "dashboard"
        
        # Iniciar actualización periódica - más rápido para el modo simulación
        refresh_interval = 0.5  # Intervalo más rápido por defecto
        if hasattr(self.bot, 'binance_client') and hasattr(self.bot.binance_client, 'simulation_mode'):
            if not self.bot.binance_client.simulation_mode:
                refresh_interval = 1.0  # Intervalo estándar para modo real
        self.set_interval(refresh_interval, self.refresh_data)
        
        # Inicializar manualmente todas las tablas y paneles
        self.initialize_all_panels()
        
    def refresh_data(self):
        """Actualizar datos periódicamente."""
        try:
            # Actualizar panel de estado
            status_panel = self.query_one("#status_panel", StatusPanel)
            if status_panel:
                status_panel.update_status()
            
            # Actualizar todos los paneles de precios
            for panel in self.query(PricesPanel):
                try:
                    panel.refresh_prices()
                except Exception as e:
                    # Registrar el error con detalles
                    print(f"Error al actualizar panel de precios: {str(e)}")
            
            # Actualizar operaciones abiertas
            for panel in self.query(OpenTradesPanel):
                try:
                    panel.refresh_trades()
                except Exception as e:
                    print(f"Error al actualizar operaciones abiertas: {str(e)}")
            
            # Actualizar historial de operaciones
            for panel in self.query(HistoryPanel):
                try:
                    panel.refresh_history()
                except Exception as e:
                    print(f"Error al actualizar historial: {str(e)}")
                    
            # Verificar que los paneles tengan datos
            if self.query(HistoryPanel) and len(self.query(HistoryPanel)) > 0:
                history_panel = list(self.query(HistoryPanel))[0]
                print(f"Panel de historial tiene {history_panel.row_count} filas")
            
            # Actualizar gráficos
            for chart_container in self.query(MultiChart):
                try:
                    chart_container.update_charts()
                except Exception as e:
                    print(f"Error al actualizar gráficos: {str(e)}")
            
            # Actualizar estadísticas si esa vista está activa
            try:
                content_switcher = self.query_one(ContentSwitcher)
                if content_switcher and content_switcher.current == "stats":
                    for panel in self.query(StatsPanel):
                        try:
                            panel.update_stats()
                        except Exception as e:
                            print(f"Error al actualizar estadísticas: {str(e)}")
            except Exception as e:
                print(f"Error al obtener content_switcher: {str(e)}")
                
        except Exception as e:
            # Registrar el error global
            print(f"Error general en refresh_data: {str(e)}")
    
    def action_switch_view(self, view_name: str):
        """Cambiar la vista actual."""
        switcher = self.query_one(ContentSwitcher)
        switcher.current = view_name
    
    def initialize_all_panels(self):
        """Inicializar manualmente todos los paneles y tablas"""
        try:
            # Dar tiempo a que se monten todos los widgets
            time.sleep(0.5)
            
            # Inicializar tablas de precios
            for panel in self.query(PricesPanel):
                try:
                    panel._initialize_rows()
                    self.notify("Tabla de precios inicializada")
                except Exception as e:
                    print(f"Error inicializando tabla de precios: {str(e)}")
                
            # Inicializar tablas de operaciones abiertas
            for panel in self.query(OpenTradesPanel):
                try:
                    panel._initialize_rows()
                    self.notify("Tabla de operaciones abiertas inicializada")
                except Exception as e:
                    print(f"Error inicializando tabla de operaciones abiertas: {str(e)}")
                
            # Inicializar tablas de historial - Importante
            for i, panel in enumerate(self.query(HistoryPanel)):
                try:
                    print(f"Inicializando panel de historial #{i}")
                    panel._initialize_rows()
                    self.notify(f"Tabla de historial inicializada con {panel.row_count} filas")
                    print(f"Inicializado con {panel.row_count} filas")
                except Exception as e:
                    print(f"Error inicializando tabla de historial: {str(e)}")
                
            # Actualizar estadísticas
            for panel in self.query(StatsPanel):
                try:
                    panel.update_stats()
                except Exception as e:
                    print(f"Error actualizando estadísticas: {str(e)}")
                
            # Actualizar gráficos
            for chart in self.query(AsciiChart):
                try:
                    chart.update_chart()
                except Exception as e:
                    print(f"Error actualizando gráfico: {str(e)}")
            
            # Forzar la actualización de todas las vistas
            self.refresh_data()
                
        except Exception as e:
            # Registrar errores durante la inicialización
            print(f"Error general inicializando paneles: {str(e)}")
            
    def action_toggle_bot(self):
        """Iniciar o detener el bot."""
        if hasattr(self.bot, 'active'):
            if self.bot.active:
                # Detener el bot
                if hasattr(self.bot, 'stop'):
                    try:
                        self.bot.stop()
                        self.notify("Bot detenido", title="Bot Control")
                    except Exception as e:
                        self.notify(f"Error al detener el bot: {str(e)}", title="Error")
            else:
                # Iniciar el bot
                if hasattr(self.bot, 'start'):
                    try:
                        self.bot.start()
                        self.notify("Bot iniciado", title="Bot Control")
                    except Exception as e:
                        self.notify(f"Error al iniciar el bot: {str(e)}", title="Error")
        else:
            self.notify("Este bot no soporta control de estado", title="Bot Control")
        
        # Actualizar panel de estado inmediatamente
        self.query_one("#status_panel", StatusPanel).update_status()

# Importaciones adicionales que necesita el código interno
from .components.panels import PricesPanel, OpenTradesPanel, HistoryPanel, StatsPanel