# MidasScalpingv4 - Bot de Trading con IA para Binance

Un sistema avanzado de trading algorítmico para Binance, con aprendizaje por refuerzo, arquitectura multimodal y optimizado para operaciones de scalping en mercados de criptomonedas.

## Características Principales

- Conexión a la API de Binance (soporta testnet y cuenta real)
- Modelos híbridos de trading:
  - Estrategia clásica basada en indicadores técnicos
  - Modelo XGBoost con aceleración GPU
  - LSTM para análisis de series temporales
  - **DeepScalper**: Modelo avanzado de aprendizaje por refuerzo
- Gestión de riesgo integrada con cálculo dinámico del tamaño de posición
- Interfaz CLI y TUI (Terminal User Interface)
- Backtesting con datos históricos y simulación de mercado

## Fundamentos Matemáticos del Modelo DeepScalper

DeepScalper implementa un enfoque avanzado de RL (Reinforcement Learning) con las siguientes características:

### Arquitectura Multimodal
- **Micro State**: Datos de alta frecuencia (precios OHLCV, forma de velas, volúmenes)
- **Macro State**: Indicadores técnicos (RSI, BB, ADX, medias móviles)
- **Private State**: Estado del trader (posición actual, capital disponible, tiempo)

### Branching Dueling Q-Network
Combina dos innovaciones en RL:
1. **Dueling Q-Network**: Separa la estimación del valor del estado (V) y la ventaja de cada acción (A)
   - Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
2. **Action Branching**: Factoriza el espacio de acción en dos ramas:
   - Dirección (compra/venta/retención)
   - Tamaño de posición (múltiples niveles)

### Predicción de Volatilidad como Tarea Auxiliar
Incorpora una tarea adicional para predecir la volatilidad futura, lo que mejora:
- La representación interna del estado del mercado
- La adaptación a condiciones cambiantes de riesgo
- El ajuste dinámico del tamaño de posición

### Hindsight Experience Replay (HER) Mejorado
Implementa un sistema de replay de experiencias que:
- Aplica un bonus retrospectivo para capturar movimientos más largos
- Utiliza relabeling de objetivos para aprender de trayectorias subóptimas
- Incorpora múltiples estrategias de muestreo (futuro, final, aleatorio)

### Prioritized Experience Replay (PER)
Mejora la eficiencia de aprendizaje al:
- Muestrear experiencias basándose en su error TD
- Aplicar importance sampling para corregir sesgos en el gradiente
- Ajustar dinámicamente las prioridades durante el entrenamiento

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

### Ejecutar con Interfaz CLI

```bash
python run.py --symbols BTCUSDT,ETHUSDT --timeframe 1m --model rl
```

### Ejecución con Interfaz TUI (Terminal)

```bash
python run_tui.py --config example_config.json
```

### Entrenamiento del Modelo DeepScalper

```bash
python main.py --train --model-type deep_scalper --symbols BTCUSDT --timeframe 1m --train-days 30
```

### Backtesting

```bash
python backtester.py --symbols BTCUSDT --timeframe 1m --model deep_scalper \
  --initial-balance 10000 --commission 0.001 --slippage 0.0002 --plot
```

### Modo Simulación

```bash
python run.py --simulate --symbols BTCUSDT,ETHUSDT --model deep_scalper \
  --sim-balance 10000 --real-data
```

## Modelos Disponibles

### 1. DeepScalper (Aprendizaje por Refuerzo)
- **Configuración**: `--model deep_scalper`
- Ideal para mercados volátiles y detección de patrones complejos
- Adaptación dinámica a cambios en regímenes de mercado

### 2. LSTM (Redes Neuronales Recurrentes)
- **Configuración**: `--model lstm`
- Bueno para capturar dependencias temporales a largo plazo
- Requiere gran cantidad de datos históricos para entrenamiento

### 3. XGBoost (Gradient Boosting)
- **Configuración**: `--model xgboost`
- Eficiente en términos computacionales
- Rendimiento robusto con menos datos de entrenamiento

### 4. Estrategia Clásica (Indicadores)
- **Configuración**: `--model indicator`
- Basado en RSI, Bandas de Bollinger y medias móviles
- No requiere entrenamiento previo

## Mecanismos de Seguridad

DeepScalper implementa características de robustez:

- **Guardado automático**: Checkpoints periódicos durante entrenamiento
- **Captura de señales**: Manejo de interrupciones (SIGINT, SIGTERM)
- **Preservación del estado**: El buffer de experiencias se guarda junto con los pesos
- **Recuperación automática**: Capacidad de continuar entrenamiento interrumpido

## Entorno de Ejecución Recomendado

El sistema ha sido probado y optimizado para ejecutarse en el siguiente entorno:

- **CPU**: Intel Core i9-9980XE @ 3.00GHz
- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **RAM**: 256GB DDR4
- **Sistema Operativo**: Ubuntu 24.04 LTS
- **Python**: 3.11+
- **TensorFlow**: 2.12+ con soporte GPU
- **XGBoost**: 1.7+ con soporte GPU

## Parámetros de Configuración Avanzados

Consulte `rl_config.json` para ver la configuración completa. Algunos parámetros clave:

```json
{
  "micro_dim": 20,
  "macro_dim": 11,
  "private_dim": 3,
  "micro_seq_len": 30,
  "macro_seq_len": 30,
  "action_branches": 2,
  "branch_sizes": [3, 5],
  "predict_volatility": true,
  "h_bonus": 10,
  "per_alpha": 0.6,
  "per_beta": 0.4
}
```

## Diagrama de Arquitectura

```
           ┌─────────────┐                ┌─────────────┐
           │ Micro State │                │ Macro State │
           │   (OHLCV)   │                │ (Indicators)│
           └──────┬──────┘                └──────┬──────┘
                  │                              │
        ┌─────────┴──────────┐          ┌────────┴───────┐
        │     Conv1D +       │          │    Conv1D +    │
        │   Bidirectional    │          │ Bidirectional  │
        │       LSTM         │          │     LSTM       │
        └─────────┬──────────┘          └────────┬───────┘
                  │                              │
                  │                              │     ┌─────────────┐
                  │                              │     │Private State│
                  │                              │     │ (Position)  │
                  │                              │     └───────┬─────┘
                  │                              │             │
                  └──────────────┬──────────────┘             │
                                 │                            │
                       ┌─────────┴────────────────────────────┴────┐
                       │                Fusion Layer              │
                       └─────────────────────┬───────────────────┬┘
                                             │                   │
                  ┌────────────────────┐     │                   │
                  │   Value Stream     │◄────┤                   │
                  │     V(s)          │     │                   │
                  └─────────┬─────────┘     │                   │
                            │               │                   │
       ┌────────────────────┼───────────────┘                   │
       │                    │                                   │
┌──────┴──────┐     ┌──────┴──────┐                    ┌────────┴────────┐
│ Advantage A │     │ Advantage B │                    │  Volatility     │
│  (Action)   │     │   (Size)    │                    │  Prediction     │
└─────────────┘     └─────────────┘                    └─────────────────┘
```

## Glosario de Términos

- **MDP**: Proceso de Decisión de Markov, marco matemático para problemas de decisión secuencial
- **Q-Learning**: Método de aprendizaje por refuerzo basado en la función Q(s,a)
- **Dueling**: Arquitectura que separa el valor del estado y la ventaja de cada acción
- **Replay Memory**: Buffer que almacena experiencias para entrenamiento off-policy
- **TD Error**: Error de diferencia temporal, mide la discrepancia en la ecuación de Bellman

## Colaboración y Contribuciones

Las contribuciones son bienvenidas. Para colaborar:

1. Haz fork del repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios y haz commit (`git commit -m 'Añade nueva funcionalidad'`)
4. Sube tu rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Advertencia

El trading de criptomonedas conlleva riesgos. Este sistema es una herramienta experimental y no garantiza resultados positivos. Utilízalo bajo tu propio riesgo y responsabilidad.