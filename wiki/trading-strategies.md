# Estrategias de Trading

MidasScalpingv4 incluye varias estrategias de trading que pueden utilizarse según las condiciones de mercado y preferencias del usuario. Este documento explica las estrategias implementadas, sus métodos de generación de señales y cómo se integran con otros componentes del sistema.

## Estrategias Disponibles

El sistema implementa las siguientes estrategias de trading:

1. **Estrategia de Scalping Básica**: Implementada en `strategy/scalping_strategy.py`, basada en indicadores técnicos
2. **Estrategia de Aprendizaje por Refuerzo**: Implementada en `strategy/rl_strategy.py`, utiliza redes neuronales profundas
3. **Estrategia basada en LLM** (opcional): Implementada en `strategy/llm_strategy.py`, utiliza modelos de lenguaje grandes

## 1. Estrategia de Scalping Básica

La estrategia principal de scalping está diseñada para capturar movimientos de precios pequeños pero frecuentes, principalmente en periodos de tiempo cortos.

### Generación de Señales

Las señales de trading se generan basadas en la combinación de varios indicadores técnicos:

```python
def _generate_indicator_signal(self, symbol, indicators):
    # Extraer indicadores
    rsi = indicators['rsi_14']
    bb_upper = indicators['bb_upper']
    bb_lower = indicators['bb_lower']
    sma_7 = indicators['sma_7']
    sma_25 = indicators['sma_25']
    
    # Señal de compra fuerte
    if (rsi < self.config.rsi_oversold and 
        current_price < bb_lower * 1.01 and 
        sma_7 > sma_25):
        
        signal_type = SignalType.BUY
        confidence = 70 + (self.config.rsi_oversold - rsi)
        # ...
    
    # Señal de venta fuerte
    elif (rsi > self.config.rsi_overbought and 
          current_price > bb_upper * 0.99 and 
          sma_7 < sma_25):
          
        signal_type = SignalType.SELL
        confidence = 70 + (rsi - self.config.rsi_overbought)
        # ...
```

### Condiciones de Señales

#### Señales de Compra (BUY)

1. **Señal Fuerte**: RSI oversold + precio debajo de la banda inferior de Bollinger + SMA7 cruzando por encima de SMA25
2. **Señal Moderada**: RSI oversold + precio debajo de la banda inferior de Bollinger

#### Señales de Venta (SELL)

1. **Señal Fuerte**: RSI overbought + precio por encima de la banda superior de Bollinger + SMA7 cruzando por debajo de SMA25
2. **Señal Moderada**: RSI overbought + precio por encima de la banda superior de Bollinger

### Adaptación a Regímenes de Mercado

La estrategia puede adaptarse a diferentes condiciones de mercado usando el detector de regímenes:

```python
def adapt_to_market_regime(self, ohlcv_data):
    # Detectar régimen actual
    regime_info = self.regime_detector.detect_regime(ohlcv_data)
    self.current_regime = regime_info['regime']
    
    if regime_info['regime'] == self.MarketRegime.TRENDING_UP:
        # En tendencia alcista: entradas más agresivas, stops más amplios
        self.config.rsi_oversold = max(20, self.base_parameters['rsi_oversold'] - 5)
        self.config.rsi_overbought = max(75, self.base_parameters['rsi_overbought'] + 5)
        
    elif regime_info['regime'] == self.MarketRegime.TRENDING_DOWN:
        # En tendencia bajista: enfoque en shorts, stops ajustados en longs
        self.config.rsi_overbought = min(65, self.base_parameters['rsi_overbought'] - 5)
        
    elif regime_info['regime'] == self.MarketRegime.RANGING:
        # En rango: enfoque en reversión a la media
        self.config.rsi_oversold = max(25, self.base_parameters['rsi_oversold'] + 5)
        self.config.rsi_overbought = min(75, self.base_parameters['rsi_overbought'] - 5)
```

## 2. Estrategia de Aprendizaje por Refuerzo

La estrategia RL utiliza un modelo de aprendizaje por refuerzo para tomar decisiones de trading basadas en estados del mercado, recompensas y experiencia previa.

### Flujo de Toma de Decisiones

```python
async def generate_signal(self, symbol, features):
    # Preprocesar características para crear el estado actual
    current_state = self.preprocess_state(symbol, features)
    
    # Usar modelo RL para elegir acción (con exploración si está en entrenamiento)
    is_training = getattr(self.config, 'rl_training_mode', False)
    action = self.model.choose_action(current_state, explore=is_training)
    
    # Mapear acción RL a señal de trading
    signal = self._map_action_to_signal(action, current_price)
    
    return signal
```

### Mapeo de Acciones a Señales

Las acciones del modelo RL se traducen a señales de trading concretas:

```python
def _map_action_to_signal(self, action, current_price):
    action_type, position_size = action
    
    # Mapear tipo de acción a tipo de señal
    if action_type == 0:  # BUY
        signal_type = SignalType.BUY
        direction = 'BUY'
        predicted_move_pct = 0.2 * (position_size + 1)
    elif action_type == 1:  # SELL
        signal_type = SignalType.SELL
        direction = 'SELL'
        predicted_move_pct = -0.2 * (position_size + 1)
    else:  # HOLD
        signal_type = SignalType.NEUTRAL
        direction = 'NEUTRAL'
        predicted_move_pct = 0
    
    # Calcular confianza basado en tamaño de posición
    confidence = min(100, 50 + 10 * position_size)
    
    return {
        'type': signal_type,
        'direction': direction,
        'confidence': confidence,
        'current_price': current_price,
        'predicted_move_pct': predicted_move_pct,
        'position_size': position_size,
        'risk_adjusted': True
    }
```

### Retroalimentación y Aprendizaje

Después de cerrar una posición, el sistema calcula recompensas y actualiza el modelo:

```python
def process_trade_result(self, symbol, entry_price, exit_price, side, quantity):
    # Calcular P&L
    if side == 'BUY':
        pnl = (exit_price - entry_price) * quantity
    else:  # SELL
        pnl = (entry_price - exit_price) * quantity
    
    # Calcular recompensa base proporcional al P&L
    base_reward = pnl / (entry_price * quantity) * 10
    
    # Penalización por drawdown si hay drawdown significativo
    drawdown_penalty = self.drawdown * 2 if self.drawdown > 0.1 else 0
    
    # Recompensa final
    reward = base_reward - drawdown_penalty
    
    # Registrar recompensa para aprendizaje
    self.record_reward(symbol, reward, done=True)
    
    # Usar hindsight experience replay para mejorar el aprendizaje
    if symbol in self.episode_memory and len(self.episode_memory[symbol]) > 5:
        self.model.hindsight_experience_replay(self.episode_memory[symbol])
```

## 3. Estrategia basada en LLM (Opcional)

Esta estrategia utiliza modelos de lenguaje grandes para analizar el mercado y generar señales de trading.

```python
async def generate_signal(self, symbol, features):
    # Preparar contexto para el modelo LLM
    prompt = self._prepare_analysis_prompt(symbol, features)
    
    # Llamar al modelo LLM a través de la API
    llm_response = await self.vultr_client.query_model(prompt)
    
    # Parsear respuesta del LLM
    signal = self._parse_llm_response(llm_response)
    
    return signal
```

## Integración con Generador de Señales

Todas las estrategias utilizan un `SignalGenerator` para estandarizar la representación de señales:

```python
class SignalType(Enum):
    BUY = 1      # Señal de posición larga
    SELL = -1    # Señal de posición corta
    NEUTRAL = 0  # Señal de no acción
```

Una señal completa incluye:

```python
signal = {
    'type': SignalType.BUY,           # Tipo de señal (BUY, SELL, NEUTRAL)
    'direction': 'BUY',               # Dirección como string
    'confidence': 75.5,               # Confianza de la señal (0-100)
    'current_price': 50000.0,         # Precio actual
    'predicted_move_pct': 0.5,        # Movimiento de precio predicho (%)
    'indicators': {...},              # Indicadores usados para la decisión
    'position_size': 2                # Tamaño de posición recomendado (0-4)
}
```

## Ciclo de Vida de una Operación

1. **Generación de señal**: La estrategia genera una señal de trading
2. **Validación de riesgo**: La señal pasa por el `RiskManager` para validación
3. **Dimensionamiento**: Se calcula el tamaño óptimo de la posición
4. **Ejecución**: Se crea la orden y se registra la posición
5. **Monitoreo**: Se actualizan trailing stops y se verifica si debe cerrarse
6. **Cierre**: La posición se cierra por stop loss, take profit o señal inversa
7. **Retroalimentación**: Los resultados se registran para mejorar el modelo

## Criterios de Cierre de Posiciones

Las posiciones pueden cerrarse por varias razones:

```python
def should_close_position(self, symbol, position, current_price, indicators):
    # Verificar stop loss
    if (position['side'] == 'BUY' and current_price <= position['stop_loss']) or \
       (position['side'] == 'SELL' and current_price >= position['stop_loss']):
        return True, 'stop_loss'
    
    # Verificar take profit
    if (position['side'] == 'BUY' and current_price >= position['take_profit']) or \
       (position['side'] == 'SELL' and current_price <= position['take_profit']):
        return True, 'take_profit'
    
    # Verificar reversión de RSI
    rsi = indicators.get('rsi_14', indicators.get('rsi', 50))
    if (position['side'] == 'BUY' and rsi > self.config.rsi_overbought) or \
       (position['side'] == 'SELL' and rsi < self.config.rsi_oversold):
        return True, 'rsi_reversal'
    
    # Verificar cruce de medias móviles
    if all(k in indicators for k in ['sma_7', 'sma_25']):
        sma_7 = indicators['sma_7']
        sma_25 = indicators['sma_25']
        
        if position['side'] == 'BUY' and sma_7 < sma_25 and current_price > position['entry_price']:
            return True, 'ma_crossover'
        
        if position['side'] == 'SELL' and sma_7 > sma_25 and current_price < position['entry_price']:
            return True, 'ma_crossover'
```

## Configuración de Estrategias

Las estrategias se pueden configurar a través de parámetros en `config.py`:

```python
# Parámetros de estrategia de scalping
rsi_oversold = 30
rsi_overbought = 70
confidence_threshold = 65

# Parámetros de aprendizaje por refuerzo
rl_training_mode = False
rl_state_dim = 30
sequence_length = 60
action_types = 3
position_sizes = 5
```

## Selección de Estrategia

Para seleccionar qué estrategia usar, modifica el archivo de configuración:

```python
# Estrategia basada en indicadores clásicos
strategy_type = 'indicator'

# Estrategia basada en aprendizaje por refuerzo
# strategy_type = 'rl'

# Estrategia basada en modelos de lenguaje
# strategy_type = 'llm'
```

## Extendiendo las Estrategias

Para implementar una nueva estrategia:

1. Crea una nueva clase en el directorio `strategy/`
2. Implementa los métodos principales: `generate_signal()`, `should_close_position()`, `calculate_position_size()`
3. Registra la estrategia en `model_factory.py` si usa un modelo ML específico
4. Actualiza la lógica de selección en `bot.py` para utilizar la nueva estrategia