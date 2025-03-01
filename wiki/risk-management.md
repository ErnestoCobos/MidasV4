# Gestión de Riesgo

La gestión de riesgo es una parte fundamental de MidasScalpingv4, especialmente importante para operaciones en mercado spot donde se busca proteger el capital del trader. Este documento explica los mecanismos de gestión de riesgo implementados en el sistema.

## Principios de Gestión de Riesgo

MidasScalpingv4 sigue varios principios clave de gestión de riesgo:

1. **Preservación del capital**: La prioridad máxima es evitar pérdidas significativas
2. **Gestión de exposición**: Límites en la exposición total y por símbolo
3. **Dimensionamiento dinámico**: Ajuste del tamaño de las posiciones según volatilidad y drawdown
4. **Protección de balance mínimo**: Garantizar que siempre quede un porcentaje de balance disponible
5. **Gestión adaptativa**: Ajustar parámetros de riesgo según el régimen de mercado

## Componentes del Sistema de Gestión de Riesgo

### 1. RiskManager

La clase `RiskManager` (en `strategy/risk_manager.py`) es el componente central de gestión de riesgo:

```python
class RiskManager:
    def __init__(self, config):
        # Parámetros de riesgo
        self.max_risk_per_trade = config.max_risk_per_trade  # % del balance por operación
        self.max_open_positions = config.max_open_trades     # Máximo número de posiciones abiertas
        self.max_exposure_pct = config.max_exposure_pct      # Máxima exposición total (%)
        self.max_exposure_per_symbol_pct = config.max_exposure_per_symbol_pct  # Exposición máxima por símbolo
        
        # Parámetros de stop loss
        self.base_stop_loss_pct = config.base_stop_loss_pct  # Stop loss base (%)
        self.max_stop_loss_pct = config.max_stop_loss_pct    # Stop loss máximo (%)
        self.trailing_stop_pct = config.trailing_stop_pct    # Trailing stop (%)
```

### 2. Cálculo de Posiciones

El dimensionamiento de posiciones se realiza basado en el riesgo, no en una cantidad fija:

```python
def calculate_position_size(self, account_balance, entry_price, stop_loss, direction):
    # Cálculo de la reserva de balance mínima
    min_reserved_balance_pct = getattr(self.config, 'min_reserved_balance_pct', 10)
    reserved_balance = account_balance * (min_reserved_balance_pct / 100)
    usable_balance = account_balance - reserved_balance
    
    # Ajustar riesgo según volatilidad del mercado
    adjusted_risk = self.max_risk_per_trade * volatility_factor
    
    # Calcular monto de riesgo
    risk_amount = usable_balance * adjusted_risk / 100
    
    # Calcular cantidad según diferencia de precio al stop loss
    price_diff = abs(entry_price - stop_loss)
    quantity = risk_amount / price_diff
    
    return quantity
```

### 3. Validación de Operaciones

Antes de abrir posiciones, el sistema realiza múltiples validaciones:

```python
def can_open_position(self, symbol, position_value, total_capital):
    # Verificar número máximo de posiciones
    if len(self.open_positions) >= self.max_open_positions:
        return False
    
    # Verificar si ya hay una posición abierta en este símbolo
    if symbol in self.open_positions:
        return False
    
    # Verificar balance mínimo reservado
    min_reserved_balance = total_capital * (self.min_reserved_balance_pct / 100)
    available_capital = total_capital - min_reserved_balance
    
    if position_value > available_capital:
        return False
    
    # Verificar exposición máxima total
    max_total_exposure = total_capital * (self.max_exposure_pct / 100)
    if self.current_exposure + position_value > max_total_exposure:
        return False
    
    # Verificar exposición máxima por símbolo
    max_symbol_exposure = total_capital * (self.max_exposure_per_symbol_pct / 100)
    if position_value > max_symbol_exposure:
        return False
    
    return True
```

### 4. Stop Loss Dinámico

El sistema implementa stop loss dinámicos que se ajustan según la volatilidad del mercado:

```python
def calculate_dynamic_stop_loss(self, entry_price, direction, volatility):
    # Ajustar el stop loss según volatilidad
    baseline_volatility = getattr(self.config, 'baseline_volatility', 0.01)
    volatility_factor = volatility / baseline_volatility
    
    # Escalar stop loss, pero con límites razonables
    adjusted_sl_pct = min(
        self.base_stop_loss_pct * max(1, volatility_factor),
        self.max_stop_loss_pct
    )
    
    # Calcular precio de stop loss
    if direction == 'BUY':
        stop_loss_price = entry_price * (1 - adjusted_sl_pct / 100)
    else:  # SELL
        stop_loss_price = entry_price * (1 + adjusted_sl_pct / 100)
    
    return stop_loss_price
```

### 5. Trailing Stop

Para capturar tendencias, el sistema implementa trailing stops que se mueven a medida que el precio se mueve a favor:

```python
def update_trailing_stops(self, current_prices):
    updates = {}
    
    for symbol, position in self.open_positions.items():
        direction = position['side']
        original_stop = position['stop_loss']
        trailing_pct = self.trailing_stop_pct / 100
        current_price = current_prices.get(symbol)
        
        # Para posiciones largas
        if direction == 'BUY' and current_price > position['entry_price']:
            # Nueva parada es el mayor entre: stop original o (precio actual - distancia trailing)
            new_stop = max(original_stop, current_price * (1 - trailing_pct))
            
            # Actualizar solo si mejora el stop
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
                updates[symbol] = new_stop
        
        # Para posiciones cortas
        elif direction == 'SELL' and current_price < position['entry_price']:
            # Nueva parada es el menor entre: stop original o (precio actual + distancia trailing)
            new_stop = min(original_stop, current_price * (1 + trailing_pct))
            
            # Actualizar solo si mejora el stop
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
                updates[symbol] = new_stop
    
    return updates
```

## Protección Contra Drawdown

Una característica clave para operaciones spot es la protección contra drawdown, que reduce el tamaño de las operaciones cuando hay pérdidas acumuladas:

```python
# En RLStrategy: Ajuste de riesgo basado en drawdown
def calculate_position_size(self, symbol, account_balance, entry_price, stop_loss, direction):
    # Ajustar riesgo basado en drawdown
    risk_pct = self.base_risk_pct
    
    if self.drawdown > 0.05:  # 5% drawdown
        risk_pct *= 0.8
    if self.drawdown > 0.1:   # 10% drawdown
        risk_pct *= 0.6
    if self.drawdown > 0.15:  # 15% drawdown
        risk_pct *= 0.4
    if self.drawdown > 0.2:   # 20% drawdown
        risk_pct *= 0.2
    
    # Calcular tamaño ajustado...
```

## Reserva de Balance Mínimo

Para evitar llegar a cero en operaciones spot, el sistema implementa una reserva de balance mínimo:

```python
# En RiskManager.can_open_position
def can_open_position(self, symbol, position_value, total_capital):
    # Garantizar que siempre quede un balance mínimo
    min_reserved_balance = getattr(self.config, 'min_reserved_balance_pct', 10) / 100
    min_balance = total_capital * min_reserved_balance
    available_capital = total_capital - min_balance
    
    if required_funds > available_capital:
        self.logger.info(
            f"Fondos insuficientes para {symbol}: {required_funds:.2f} {quote_asset} requeridos, "
            f"{available_capital:.2f} {quote_asset} disponibles (reserva mínima: {min_balance:.2f})"
        )
        return False
```

## Verificación de Fondos para Operaciones Spot

Para operaciones spot, es crucial verificar que haya fondos suficientes antes de abrir una posición:

```python
# Verificación específica para operaciones spot
if hasattr(self.config, 'enforce_spot_balance') and self.config.enforce_spot_balance:
    # Obtener el activo cotizado (ej: 'USDT' de 'BTCUSDT')
    quote_asset = symbol[3:]
    
    # Aplicar margen de seguridad para comisiones
    safety_margin = 1.0 + (getattr(self.config, 'safety_margin_pct', 2) / 100)
    required_funds = position_value * safety_margin
    
    # Verificar contra capital disponible (menos reserva mínima)
    if required_funds > available_capital:
        # Rechazar operación por fondos insuficientes
        return False
```

## Adaptación por Régimen de Mercado

El sistema ajusta los parámetros de riesgo según el régimen de mercado detectado:

```python
def adapt_to_market_regime(self, regime):
    if regime == MarketRegime.TRENDING_UP:
        # En tendencia alcista - entradas más agresivas, stops más amplios
        self.config.rsi_oversold = max(20, self.base_parameters['rsi_oversold'] - 5)
        self.config.max_risk_per_trade = min(2.0, self.base_parameters['max_risk_per_trade'] * 1.2)
        
    elif regime == MarketRegime.TRENDING_DOWN:
        # En tendencia bajista - enfoque en cortos, stops más ajustados en largos
        self.config.base_stop_loss_pct = self.base_parameters['base_stop_loss_pct'] * 0.8
        self.config.max_risk_per_trade = max(0.5, self.base_parameters['max_risk_per_trade'] * 0.8)
        
    elif regime == MarketRegime.RANGING:
        # En rango - enfoque en reversión a la media
        self.config.base_stop_loss_pct = self.base_parameters['base_stop_loss_pct'] * 0.8
        
    elif regime == MarketRegime.VOLATILE:
        # En mercados volátiles - reducir tamaño, ampliar stops
        self.config.max_risk_per_trade = max(0.5, self.base_parameters['max_risk_per_trade'] * 0.6)
```

## Integración con Aprendizaje por Refuerzo

El sistema de gestión de riesgo se integra con el módulo de aprendizaje por refuerzo:

1. **Función de recompensa sensible al riesgo**: Penaliza drawdowns excesivos
2. **Tarea auxiliar de riesgo**: El modelo RL aprende a estimar tanto rendimiento como riesgo
3. **Adaptación por experiencia**: El sistema optimiza parámetros basado en resultados anteriores

```python
# Cálculo de recompensa con penalización por drawdown
def calculate_reward(self, pnl, position):
    reward = pnl  # Recompensa base es P&L
    
    # Penalizar drawdowns
    if self.drawdown > 0.1:  # 10% drawdown
        drawdown_penalty = self.drawdown * 2
        reward -= drawdown_penalty
    
    return reward
```

## Configuración

Los parámetros de gestión de riesgo se pueden configurar en el archivo de configuración:

```python
# Parámetros de riesgo
max_risk_per_trade = 0.5         # Porcentaje máximo de balance en riesgo por operación
max_open_trades = 3              # Número máximo de operaciones simultáneas
max_exposure_pct = 30            # Porcentaje máximo de exposición total
max_exposure_per_symbol_pct = 15 # Porcentaje máximo por símbolo

# Parámetros de stop loss
base_stop_loss_pct = 0.7         # Porcentaje de stop loss base (conservador para spot)
max_stop_loss_pct = 2.0          # Porcentaje de stop loss máximo
trailing_stop_pct = 0.5          # Porcentaje de trailing stop

# Protección de balance
min_reserved_balance_pct = 10    # Porcentaje mínimo de balance que nunca se usa
safety_margin_pct = 3            # Margen adicional para comisiones y slippage
enforce_spot_balance = True      # Habilitar verificación estricta de balance para spot
```

## Recomendaciones de Uso

1. **Ajusta `max_risk_per_trade`**: Para spot, recomendamos valores entre 0.2% y 1%
2. **Incrementa `min_reserved_balance_pct`**: Mayor protección con valores entre 10-20%
3. **Adapta `base_stop_loss_pct`** según la volatilidad del par (más bajo para pares estables)
4. **Monitorea drawdown**: El sistema reduce automáticamente el riesgo en periodos de pérdidas
5. **Utiliza `enforce_spot_balance`**: Siempre activado para operaciones spot