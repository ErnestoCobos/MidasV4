# MidasScalpingv4 - Bot de Scalping para Binance

Un bot de trading algorítmico para scalping en Binance, diseñado para operar en mercados de criptomonedas con tiempos de operación cortos.

## Características

- Conexión a la API de Binance (soporta testnet y cuenta real)
- Estrategia de trading basada en RSI, Bandas de Bollinger y medias móviles
- Gestión de riesgo integrada con cálculo automático del tamaño de posición
- Órdenes automatizadas con stop loss y take profit
- Monitoreo en tiempo real de precios mediante websockets
- Seguimiento de rendimiento y estadísticas

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tuusuario/MidasScalpingv4.git
cd MidasScalpingv4
```

2. Crea un entorno virtual e instala las dependencias:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configura tus credenciales de API:
   - Crea un archivo `.env` con tus claves de API de Binance:
   ```
   BINANCE_API_KEY=tu_clave_api
   BINANCE_API_SECRET=tu_clave_secreta
   TRADING_SYMBOLS=BTCUSDT,ETHUSDT
   MAX_CAPITAL_RISK=2.0
   ```

## Uso

### Modo de Testnet (recomendado para pruebas)

```bash
python main.py --testnet
```

### Modo de Trading en vivo

```bash
python main.py
```

### Opciones adicionales

```bash
python main.py --symbols BTCUSDT,ETHUSDT --config config.json
```

## Estrategia de Scalping

La estrategia implementada se basa en los siguientes indicadores técnicos:

1. **RSI (Índice de Fuerza Relativa)**: Identifica condiciones de sobrecompra/sobreventa
2. **Bandas de Bollinger**: Detecta breakouts y retornos a la media
3. **Medias Móviles Simples**: Identifica la tendencia a corto y medio plazo
4. **Volumen**: Confirma señales con volumen superior a la media

## Gestión de Riesgo

El bot implementa gestión de riesgo mediante:

- Límite configurable de riesgo por operación (por defecto 2% del capital)
- Stop loss y take profit automáticos
- Límite de operaciones simultáneas
- Cálculo proporcional del tamaño de posición

## Advertencia

El trading de criptomonedas conlleva riesgos. Este bot es una herramienta experimental y no garantiza resultados positivos. Utilízalo bajo tu propio riesgo y responsabilidad.