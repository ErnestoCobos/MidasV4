# Fundamentos Matemáticos de MidasScalpingv4

Este documento presenta los fundamentos matemáticos que subyacen al sistema de trading algorítmico MidasScalpingv4, incluyendo el análisis técnico, la toma de decisiones basada en aprendizaje por refuerzo, y los modelos de gestión de riesgo.

## 1. Modelo de Decisión de Trading

### 1.1 Indicadores Técnicos

#### Índice de Fuerza Relativa (RSI)
El RSI es un oscilador que mide la velocidad y cambio de los movimientos de precio.

$$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$

Donde RS (Relative Strength) es:

$$\text{RS} = \frac{\text{Promedio de ganancias durante } n \text{ periodos}}{\text{Promedio de pérdidas durante } n \text{ periodos}}$$

#### Bandas de Bollinger
Las Bandas de Bollinger definen un rango de volatilidad alrededor de la media móvil.

$$\text{BB}_{\text{middle}} = \text{SMA}(n)$$
$$\text{BB}_{\text{upper}} = \text{SMA}(n) + k \times \sigma_n$$
$$\text{BB}_{\text{lower}} = \text{SMA}(n) - k \times \sigma_n$$

Donde:
- $\text{SMA}(n)$ es la media móvil simple de $n$ períodos
- $\sigma_n$ es la desviación estándar de los precios durante $n$ períodos
- $k$ es el factor multiplicador (típicamente 2)

#### Media Móvil Simple (SMA)
$$\text{SMA}(n) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}$$

Donde $P_t$ es el precio en el tiempo $t$.

### 1.2 Función de Generación de Señales

La generación de señales se basa en condiciones lógicas combinando estos indicadores. Para una señal de compra fuerte:

$$\text{BUY signal} = \begin{cases} 
\text{True}, & \text{si } \text{RSI} < \text{RSI}_{\text{oversold}} \text{ AND } P_{\text{current}} < \text{BB}_{\text{lower}} \times 1.01 \text{ AND } \text{SMA}_7 > \text{SMA}_{25} \\
\text{False}, & \text{en caso contrario}
\end{cases}$$

La confianza de la señal se calcula como:
$$\text{confidence} = 70 + (\text{RSI}_{\text{oversold}} - \text{RSI})$$

De manera similar, para señales de venta y señales moderadas.

## 2. Aprendizaje por Refuerzo (RL)

### 2.1 Formulación del Problema de RL

El trading se modela como un proceso de decisión de Markov (MDP) definido por la tupla $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- $\mathcal{S}$: Espacio de estados (características del mercado)
- $\mathcal{A}$: Espacio de acciones (comprar/vender/mantener con diferentes tamaños)
- $\mathcal{P}$: Función de transición de probabilidad
- $\mathcal{R}$: Función de recompensa
- $\gamma$: Factor de descuento

### 2.2 Arquitectura de Red Q Dual con Ramificación de Acciones

La red Q dual separa la estimación del valor de estado $V(s)$ y la ventaja de cada acción $A(s,a)$:

$$Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')\right)$$

Con ramificación de acciones, para cada tipo de acción $i$ y tamaño $j$:

$$Q(s,(i,j)) = V(s) + \left(A_i(s,j) - \frac{1}{|\mathcal{A}_i|}\sum_{j'} A_i(s,j')\right)$$

### 2.3 Función de Recompensa

La función de recompensa balancea rentabilidad y riesgo:

$$R(s,a,s') = \underbrace{r_{\text{pnl}} \cdot 10}_{\text{recompensa base}} - \underbrace{d \cdot 2}_{\text{penalización drawdown}} + \underbrace{\min(w \cdot 0.1, 0.5)}_{\text{bonificación consistencia}}$$

Donde:
- $r_{\text{pnl}}$ es el retorno normalizado (P&L dividido por el valor de la posición)
- $d$ es el drawdown actual (cuando supera el 10%)
- $w$ es la racha de operaciones ganadoras consecutivas

### 2.4 Algoritmo Q-Learning Profundo

El algoritmo actualiza los valores Q minimizando la pérdida:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ \left( r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta) \right)^2 \right]$$

Donde:
- $\theta$ son los parámetros de la red Q principal
- $\theta^-$ son los parámetros de la red Q objetivo
- $(s,a,r,s')$ son las transiciones almacenadas en el buffer de experiencia

### 2.5 Hindsight Experience Replay (HER)

La técnica HER relabela las experiencias utilizando conocimiento retrospectivo:

$$r_{\text{hindsight}}(s_t, a_t, s_T) = \text{sign}(P_T - P_t) \cdot 0.5$$

Donde $P_T$ es el precio final y $P_t$ es el precio en el momento $t$.

## 3. Detección de Regímenes de Mercado

### 3.1 Clasificación de Regímenes

El mercado se clasifica en cuatro regímenes usando una combinación de indicadores:

1. **Tendencia Alcista (TRENDING_UP)**:
   $$\text{ADX} > \text{ADX}_{\text{threshold}} \text{ AND } \text{DI+} > \text{DI-}$$

2. **Tendencia Bajista (TRENDING_DOWN)**:
   $$\text{ADX} > \text{ADX}_{\text{threshold}} \text{ AND } \text{DI+} < \text{DI-}$$

3. **Rango (RANGING)**:
   $$\text{ADX} < \text{ADX}_{\text{threshold}} \text{ AND } \sigma_{\text{price}} < \sigma_{\text{threshold}}$$

4. **Volatilidad (VOLATILE)**:
   $$\sigma_{\text{price}} > \sigma_{\text{threshold}}$$

Donde ADX es el Índice de Movimiento Direccional Promedio y $\sigma_{\text{price}}$ es la volatilidad del precio.

### 3.2 Adaptación Paramétrica

Los parámetros se ajustan dinámicamente según el régimen detectado:

$$\text{param}_{\text{adjusted}} = f_{\text{regime}}(\text{param}_{\text{base}})$$

Por ejemplo, para el régimen TRENDING_UP:
$$\text{RSI}_{\text{oversold}} = \max(20, \text{RSI}_{\text{oversold, base}} - 5)$$
$$\text{risk}_{\text{per trade}} = \min(2.0, \text{risk}_{\text{base}} \cdot 1.2)$$

## 4. Gestión de Riesgo

### 4.1 Dimensionamiento de Posiciones Basado en Riesgo

El tamaño de posición se calcula mediante:

$$\text{Quantity} = \frac{\text{Balance}_{\text{usable}} \cdot \text{risk}_{\text{adjusted}}}{\text{|Entry Price - Stop Loss|}}$$

Donde:
$$\text{Balance}_{\text{usable}} = \text{Balance}_{\text{total}} \cdot (1 - \text{reserve}_{\text{ratio}})$$
$$\text{risk}_{\text{adjusted}} = \text{risk}_{\text{base}} \cdot \text{volatility}_{\text{factor}} \cdot \text{drawdown}_{\text{factor}}$$

### 4.2 Stop Loss Dinámico

El stop loss se ajusta dinámicamente según la volatilidad del mercado:

$$\text{SL}_{\text{percentage}} = \min\left(\text{SL}_{\text{base}} \cdot \max\left(1, \frac{\text{volatility}}{\text{volatility}_{\text{baseline}}}\right), \text{SL}_{\text{max}}\right)$$

$$\text{SL}_{\text{price}} = \begin{cases} 
\text{Entry Price} \cdot (1 - \text{SL}_{\text{percentage}}), & \text{para posiciones largas} \\
\text{Entry Price} \cdot (1 + \text{SL}_{\text{percentage}}), & \text{para posiciones cortas}
\end{cases}$$

### 4.3 Trailing Stop

El trailing stop se actualiza cuando el precio se mueve a favor:

Para posiciones largas:
$$\text{NewStop} = \max(\text{OldStop}, \text{CurrentPrice} \cdot (1 - \text{trailing}_{\text{percentage}}))$$

Para posiciones cortas:
$$\text{NewStop} = \min(\text{OldStop}, \text{CurrentPrice} \cdot (1 + \text{trailing}_{\text{percentage}}))$$

### 4.4 Factor de Ajuste por Drawdown

El sistema reduce el riesgo cuando hay drawdown significativo:

$$\text{drawdown}_{\text{factor}} = \begin{cases} 
1.0, & \text{si } \text{drawdown} < 0.05 \\
0.8, & \text{si } 0.05 \leq \text{drawdown} < 0.1 \\
0.6, & \text{si } 0.1 \leq \text{drawdown} < 0.15 \\
0.4, & \text{si } 0.15 \leq \text{drawdown} < 0.2 \\
0.2, & \text{si } \text{drawdown} \geq 0.2
\end{cases}$$

## 5. Validación de Operaciones

### 5.1 Verificación de Fondos para Spot

Para operaciones spot, se verifica:

$$\text{required\_funds} = \text{position\_value} \cdot (1 + \text{safety\_margin})$$
$$\text{min\_balance} = \text{total\_capital} \cdot \text{min\_reserved\_ratio}$$
$$\text{available\_capital} = \text{total\_capital} - \text{min\_balance}$$

Una operación es válida si y solo si:
$$\text{required\_funds} \leq \text{available\_capital}$$

### 5.2 Restricciones de Exposición

Se aplican las siguientes restricciones:
1. Número máximo de posiciones:
   $$|\text{open\_positions}| < \text{max\_positions}$$

2. Exposición máxima total:
   $$\text{current\_exposure} + \text{position\_value} \leq \text{total\_capital} \cdot \text{max\_exposure\_ratio}$$

3. Exposición máxima por símbolo:
   $$\text{symbol\_exposure} + \text{position\_value} \leq \text{total\_capital} \cdot \text{max\_symbol\_exposure\_ratio}$$

## 6. Conclusión

Este framework matemático proporciona una base sólida para el sistema de trading MidasScalpingv4, combinando análisis técnico tradicional con técnicas avanzadas de aprendizaje por refuerzo y gestión de riesgo adaptativa. La implementación computacional de estos modelos permite al sistema adaptarse a diferentes condiciones de mercado mientras mantiene un estricto control del riesgo, especialmente importante en operaciones spot.