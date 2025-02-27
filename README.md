# MidasScalpingv4 - Bot de Scalping para Binance

Un bot de trading algor�tmico para scalping en Binance, dise�ado para operar en mercados de criptomonedas con tiempos de operaci�n cortos.

## Caracter�sticas

- Conexi�n a la API de Binance (soporta testnet y cuenta real)
- Estrategia de trading basada en RSI, Bandas de Bollinger y medias m�viles
- Gesti�n de riesgo integrada con c�lculo autom�tico del tama�o de posici�n
- �rdenes automatizadas con stop loss y take profit
- Monitoreo en tiempo real de precios mediante websockets
- Seguimiento de rendimiento y estad�sticas

## Instalaci�n

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

La estrategia implementada se basa en los siguientes indicadores t�cnicos:

1. **RSI (�ndice de Fuerza Relativa)**: Identifica condiciones de sobrecompra/sobreventa
2. **Bandas de Bollinger**: Detecta breakouts y retornos a la media
3. **Medias M�viles Simples**: Identifica la tendencia a corto y medio plazo
4. **Volumen**: Confirma se�ales con volumen superior a la media

## Gesti�n de Riesgo

El bot implementa gesti�n de riesgo mediante:

- L�mite configurable de riesgo por operaci�n (por defecto 2% del capital)
- Stop loss y take profit autom�ticos
- L�mite de operaciones simult�neas
- C�lculo proporcional del tama�o de posici�n

## Advertencia

El trading de criptomonedas conlleva riesgos. Este bot es una herramienta experimental y no garantiza resultados positivos. Util�zalo bajo tu propio riesgo y responsabilidad.