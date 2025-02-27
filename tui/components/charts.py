from textual.widgets import Static
from textual.containers import Horizontal
from rich.panel import Panel

class AsciiChart(Static):
    """Widget para mostrar gráficos ASCII de precios"""
    
    def __init__(self, symbol, data_points=40, height=10, **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.data_points = data_points
        self.chart_height = height
        self.price_history = []
        
    def on_mount(self):
        self.update_chart()
        
    def update_chart(self):
        # Obtener datos del bot (últimos N precios)
        bot = self.app.bot
        
        # Obtener precio actual
        current_price = None
        if hasattr(bot, 'real_time_prices') and self.symbol in bot.real_time_prices:
            current_price = bot.real_time_prices[self.symbol]
        
        # Si no hay precio actual, mostrar mensaje de espera
        if current_price is None:
            self.update(f"Esperando datos para {self.symbol}...")
            return
            
        # Añadir precio al historial
        self.price_history.append(current_price)
        
        # Limitar la cantidad de puntos
        if len(self.price_history) > self.data_points:
            self.price_history = self.price_history[-self.data_points:]
            
        # Si no hay suficientes datos, mostrar mensaje
        if len(self.price_history) < 2:
            self.update(f"Recopilando datos para {self.symbol}...")
            return
            
        # Normalizar datos para el alto del gráfico
        min_price = min(self.price_history)
        max_price = max(self.price_history)
        price_range = max_price - min_price
        
        # Evitar división por cero
        if price_range == 0:
            price_range = 0.00001
            
        normalized = [
            int((p - min_price) / price_range * (self.chart_height - 1)) 
            for p in self.price_history
        ]
        
        # Construir el gráfico ASCII
        chart = []
        for y in range(self.chart_height - 1, -1, -1):
            line = ""
            for i, p in enumerate(normalized):
                # Punto de precio
                if p == y:
                    line += "⬤"  # Punto grueso para precio exacto
                # Línea vertical conectora
                elif i > 0 and ((normalized[i-1] < y < p) or (normalized[i-1] > y > p)):
                    line += "│"  # Línea vertical
                # Espacio vacío
                else:
                    line += " "
            chart.append(line)
            
        # Añadir etiquetas de precio
        price_label_max = f"{max_price:.2f}"
        price_label_min = f"{min_price:.2f}"
        
        # Definir tendencia
        is_positive = self.price_history[-1] > self.price_history[0]
        trend = "↗️" if is_positive else "↘️"
        percent_change = ((self.price_history[-1] - self.price_history[0]) / self.price_history[0]) * 100
        trend_class = "positive" if percent_change >= 0 else "negative"
        
        # Construir el texto completo
        header = f"{self.symbol} {trend} [bold {trend_class}]{percent_change:.2f}%[/]"
        chart_text = "\n".join(chart)
        footer = f"{price_label_max}\n{chart_text}\n{price_label_min}"
        
        # Actualizar el widget
        self.update(Panel(footer, title=header, title_align="left"))

class MultiChart(Horizontal):
    """Widget que muestra múltiples gráficos ASCII en una fila horizontal"""
    
    def __init__(self, symbols=None, **kwargs):
        super().__init__(**kwargs)
        self.symbols = symbols or []
        self.charts = {}
        
    def compose(self):
        # Crear un gráfico para cada símbolo
        for symbol in self.symbols:
            chart = AsciiChart(symbol, id=f"chart_{symbol}")
            yield chart
            self.charts[symbol] = chart
    
    def update_charts(self):
        """Actualizar todos los gráficos"""
        # Si no hay charts en self.charts, buscarlos por id
        if not self.charts and self.symbols:
            for symbol in self.symbols:
                chart_id = f"#chart_{symbol}"
                try:
                    self.charts[symbol] = self.query_one(chart_id, AsciiChart)
                except:
                    pass
        
        # Actualizar cada gráfico
        for symbol, chart in self.charts.items():
            chart.update_chart()