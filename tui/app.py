from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, ContentSwitcher
from textual.containers import Container

from .components.panels import StatusPanel
from .components.views import DashboardView, PricesView, TradesView, StatsView, ConfigView
from .components.charts import MultiChart

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
        
        # Iniciar actualización periódica
        self.set_interval(1.0, self.refresh_data)
        
    def refresh_data(self):
        """Actualizar datos periódicamente."""
        # Actualizar panel de estado
        self.query_one("#status_panel", StatusPanel).update_status()
        
        # Actualizar todos los paneles de precios
        for panel in self.query(PricesPanel):
            panel.refresh_prices()
        
        # Actualizar operaciones abiertas
        for panel in self.query(OpenTradesPanel):
            panel.refresh_trades()
        
        # Actualizar historial de operaciones
        for panel in self.query(HistoryPanel):
            panel.refresh_history()
        
        # Actualizar gráficos
        for chart_container in self.query(MultiChart):
            chart_container.update_charts()
        
        # Actualizar estadísticas si esa vista está activa
        if self.query_one(ContentSwitcher).current == "stats":
            for panel in self.query(StatsPanel):
                panel.update_stats()
    
    def action_switch_view(self, view_name: str):
        """Cambiar la vista actual."""
        switcher = self.query_one(ContentSwitcher)
        switcher.current = view_name
    
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