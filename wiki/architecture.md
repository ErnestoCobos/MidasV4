# Arquitectura del Sistema

MidasScalpingv4 está construido con una arquitectura modular que permite flexibilidad, extensibilidad y mantenimiento sencillo. Este documento explica los componentes principales del sistema y cómo interactúan entre sí.

## Estructura de Directorios

```
MidasScalpingv4/
├── core/           # Componentes centrales del sistema
├── models/         # Modelos de aprendizaje automático
├── strategy/       # Implementaciones de estrategias de trading
├── data/           # Manipulación y procesamiento de datos
├── tui/            # Interfaz de usuario textual
├── ai/             # Integraciones con IA (opcional)
```

## Componentes Principales

### 1. Core

El módulo core contiene los componentes fundamentales del sistema:

- `config.py`: Gestión de configuración global
- `constants.py`: Constantes del sistema
- `logging_setup.py`: Configuración de logging

### 2. Modelos

El directorio `models/` contiene implementaciones de diferentes modelos predictivos:

- `lstm_model.py`: Modelo de red neuronal LSTM para predicción de series temporales
- `xgboost_model.py`: Modelo de gradient boosting para clasificación/regresión
- `deep_scalper.py`: Modelo de aprendizaje por refuerzo con redes Q duales
- `model_factory.py`: Fábrica para instanciar diferentes tipos de modelos
- `model_trainer.py`: Sistema de entrenamiento automático de modelos

### 3. Estrategia

El módulo `strategy/` implementa las estrategias de trading:

- `scalping_strategy.py`: Estrategia principal de scalping basada en indicadores
- `signal_generator.py`: Generador de señales de trading
- `risk_manager.py`: Sistema de gestión de riesgo
- `market_regime.py`: Detector de regímenes de mercado
- `llm_strategy.py`: Estrategia basada en modelos de lenguaje (opcional)
- `rl_strategy.py`: Estrategia basada en aprendizaje por refuerzo

### 4. Datos

El módulo `data/` maneja la obtención y procesamiento de datos:

- `binance_client.py`: Cliente para la API de Binance
- `data_processor.py`: Procesamiento y limpieza de datos
- `database.py`: Persistencia y recuperación de datos
- `feature_engineer.py`: Generación de características para modelos

### 5. Interfaz de Usuario

El módulo `tui/` implementa la interfaz de usuario textual:

- `app.py`: Aplicación principal Textual 
- `components/`: Componentes reutilizables de la interfaz

## Flujo de Ejecución

1. **Inicialización**: 
   - Carga de configuración
   - Establecimiento de conexiones a APIs y bases de datos
   - Inicialización de modelos y estrategias

2. **Ciclo de Trading**:
   - Obtención de datos de mercado
   - Cálculo de indicadores técnicos
   - Detección de régimen de mercado
   - Generación de señales de trading
   - Validación por gestión de riesgo
   - Ejecución de órdenes (si se cumplen todos los criterios)

3. **Aprendizaje y Retroalimentación**:
   - Registro de resultados de trading
   - Cálculo de recompensas para aprendizaje por refuerzo
   - Actualización periódica de modelos

## Diagrama de Interacción

```
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│ Data Module │────▶│ Feature       │────▶│ Strategy       │
│             │◀────│ Engineering   │◀────│ Implementation │
└─────────────┘     └───────────────┘     └────────────────┘
                                                  │
                                                  ▼
┌─────────────┐                           ┌────────────────┐
│ User        │◀──────────────────────────│ Risk           │
│ Interface   │───────────────────────────▶│ Management    │
└─────────────┘                           └────────────────┘
                                                  ▲
                                                  │
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│ Order       │◀────│ Account       │◀────│ Model          │
│ Execution   │────▶│ Management    │────▶│ Predictions    │
└─────────────┘     └───────────────┘     └────────────────┘
```

## Extensibilidad

El sistema está diseñado para ser extensible:

- **Nuevas estrategias**: Implementa nuevas clases en el directorio `strategy/`
- **Nuevos modelos**: Añade implementaciones en `models/` y actualiza `model_factory.py`
- **Nuevos exchanges**: Crea nuevos clientes en el directorio `data/`

Cada componente sigue principios SOLID, facilitando la adición de nuevas funcionalidades sin modificar el código existente.