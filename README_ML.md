# MidasScalpingv4 Machine Learning Guide

Este documento explica cómo configurar, entrenar y utilizar los modelos de Machine Learning (XGBoost y TensorFlow) con aceleración GPU en MidasScalpingv4.

## Tabla de Contenidos

1. [Requisitos](#requisitos)
2. [Configuración de GPU](#configuración-de-gpu)
3. [Modelos Soportados](#modelos-soportados)
4. [Entrenamiento de Modelos](#entrenamiento-de-modelos)
5. [Integración en Estrategia](#integración-en-estrategia)
6. [Parámetros de Configuración](#parámetros-de-configuración)
7. [Backtest con Modelos ML](#backtest-con-modelos-ml)
8. [Mejores Prácticas](#mejores-prácticas)

## Requisitos

- Python 3.9+
- CUDA Toolkit 11.2+ (para GPU)
- cuDNN (para TensorFlow)
- Dependencias de Python:
  - TensorFlow 2.9+
  - XGBoost 1.6+
  - pandas
  - numpy
  - scikit-learn
  - joblib

## Configuración de GPU

### Instalación de CUDA y cuDNN

1. **Instalar CUDA Toolkit**:
   Descargar e instalar desde [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

2. **Instalar cuDNN** (para TensorFlow):
   Descargar e instalar desde [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

3. **Verificar instalación**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

### Verificar Compatibilidad en Python

```python
import tensorflow as tf
print("TensorFlow GPU disponible:", tf.config.list_physical_devices('GPU'))

import xgboost as xgb
# XGBoost verificará automáticamente la GPU al crear un modelo
```

## Modelos Soportados

MidasScalpingv4 soporta dos tipos principales de modelos:

### XGBoost
- Ventajas: Rápido, eficiente en memoria, bueno para datos tabulares
- Casos de uso: Predicción a corto plazo, clasificación de dirección de mercado
- Aceleración GPU: `tree_method='gpu_hist'`

### TensorFlow LSTM
- Ventajas: Captura patrones secuenciales, memoria a largo plazo
- Casos de uso: Predicción de tendencias, patrones complejos
- Aceleración GPU: Automática si disponible

## Entrenamiento de Modelos

### Entrenamiento de XGBoost

```python
from models.ml_module import MLModule

# Configurar parámetros
config = Config()
config.use_gpu = True
config.gpu_device = 0  # Primera GPU
config.xgb_rounds = 1000  # Número de rondas de entrenamiento

# Inicializar módulo ML
ml_module = MLModule(config)

# Preparar datos (características y etiquetas)
# X_train, y_train, X_val, y_val = ...  # Cargar o preparar datos

# Entrenar modelo
model = ml_module.train_xgboost(X_train, y_train, X_val, y_val)

# Guardar modelo
ml_module.save_xgboost('saved_models/xgboost_model.json')
```

### Entrenamiento de LSTM

```python
from models.ml_module import MLModule

# Configurar parámetros
config = Config()
config.use_gpu = True
config.gpu_device = 0
config.sequence_length = 10  # Número de pasos de tiempo para secuencia
config.feature_count = 20    # Número de características por paso de tiempo

# Inicializar módulo ML
ml_module = MLModule(config)

# Preparar datos (secuencias y etiquetas)
# Formato de X_train: [samples, sequence_length, features]
# X_train, y_train, X_val, y_val = ...  # Cargar o preparar datos

# Entrenar modelo
model = ml_module.train_lstm(
    X_train, y_train, X_val, y_val,
    epochs=50,
    batch_size=32
)

# Guardar modelo
ml_module.save_lstm('saved_models/lstm_model')
```

### Script de Preparación de Datos

Para preparar datos de entrenamiento a partir de datos históricos:

```python
import pandas as pd
from data.feature_engineer import FeatureEngineer

# Cargar datos históricos
data = pd.read_csv('data/BTCUSDT_1m.csv')

# Inicializar ingeniero de características
feature_engineer = FeatureEngineer(config)

# Generar características
features = feature_engineer.generate_features(data)

# Para XGBoost: usar directamente
X = features.drop('target', axis=1)
y = features['target']

# Para LSTM: remodelar datos a formato secuencial
sequence_length = 10
X_sequences = []
y_sequences = []

for i in range(len(features) - sequence_length):
    X_sequences.append(features.iloc[i:i+sequence_length, :-1].values)
    y_sequences.append(features.iloc[i+sequence_length, -1])

X_lstm = np.array(X_sequences)
y_lstm = np.array(y_sequences)
```

## Integración en Estrategia

El sistema utiliza la clase `MLModule` para gestionar la integración de modelos en la estrategia de trading.

### Flujo de Integración

1. **Inicialización**:
   ```python
   # En bot.py o donde inicialices la estrategia
   from models.ml_module import MLModule
   
   # Crear módulo ML
   ml_module = MLModule(config)
   
   # Cargar modelos pre-entrenados
   ml_module.load_xgboost('saved_models/xgboost_model.json')
   ml_module.load_lstm('saved_models/lstm_model')
   
   # Inicializar estrategia con módulo ML
   strategy = ScalpingStrategy(config, model=ml_module)
   ```

2. **Generación de Señales**:
   - El modelo recibe características actuales del mercado
   - Genera predicciones sobre movimiento de precios
   - La estrategia combina predicciones con indicadores técnicos
   - Aplica umbral de confianza (configurable) para reducir overtrading

3. **Ajustes de Riesgo**:
   - Incluye modelado de slippage y comisiones para simulaciones realistas
   - Implementa stop de pérdidas diario (configurable)
   - Limita el número máximo de trades por día

## Parámetros de Configuración

### Parámetros Generales

```python
config = Config()

# Parámetros de GPU
config.use_gpu = True             # Activar aceleración GPU
config.gpu_device = 0             # ID del dispositivo GPU a usar

# Parámetros de Comisiones y Slippage
config.commission_rate = 0.001   # 0.1% comisión (tarifa estándar Binance VIP 0)
config.slippage_pct = 0.0002     # 0.02% slippage
# Para usar la tarifa con descuento BNB (25% descuento): 0.00075 (0.075%)

# Parámetros de Trading
config.confidence_threshold = 0.7 # Umbral de confianza para señales (0-1)
config.max_daily_trades = 30      # Máximo de trades por día
config.max_daily_loss_pct = 3.0   # Pérdida máxima diaria (% del capital)
```

### Parámetros Específicos de XGBoost

```python
# Parámetros por defecto que se pueden ajustar
config.xgb_rounds = 100           # Número de rondas (árboles)
config.xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist',    # Usar GPU
    'predictor': 'gpu_predictor'  # Predicción en GPU
}
```

### Parámetros Específicos de LSTM

```python
# Parámetros por defecto que se pueden ajustar
config.sequence_length = 10       # Longitud de secuencia (timesteps)
config.feature_count = 20         # Número de características
```

## Backtest con Modelos ML

Para evaluar modelos con datos históricos:

```bash
python backtester.py --symbols BTCUSDT --timeframe 1m --use-ml \
  --xgb-model saved_models/xgboost_model.json \
  --lstm-model saved_models/lstm_model \
  --use-gpu \
  --initial-balance 10000 \
  --confidence-threshold 0.7 \
  --max-daily-trades 30 \
  --commission 0.001 \
  --slippage 0.0002 \
  --plot
```

O utilizando la tarifa con descuento por BNB:

```bash
python backtester.py --symbols BTCUSDT --timeframe 1m --use-ml \
  --xgb-model saved_models/xgboost_model.json \
  --lstm-model saved_models/lstm_model \
  --use-gpu \
  --initial-balance 10000 \
  --commission 0.00075 \
  --slippage 0.0002 \
  --plot
```

## Mejores Prácticas

1. **Preparación de Datos**:
   - Normalizar características para mejorar convergencia
   - Equilibrar clases para evitar sesgos
   - Usar validación cruzada para evitar sobreajuste

2. **Optimización de Modelos**:
   - Realizar búsqueda de hiperparámetros (Grid Search, Bayesian)
   - Evaluar importancia de características para seleccionar las más relevantes
   - Entrenar múltiples modelos con diferentes ventanas temporales

3. **Gestión de Riesgos**:
   - Validar modelos con datos out-of-sample
   - Implementar monitoreo constante de rendimiento
   - Reentrenar periódicamente con datos recientes
   - Incluir comisiones realistas:
     - Estándar: 0.1% (VIP 0)
     - Con BNB: 0.075% (descuento 25%)
     - Considerar los pares con comisión cero para optimización

4. **Optimización de Recursos**:
   - Usar TensorFlow-Lite para despliegue más eficiente
   - Considerar técnicas de cuantización para modelos más pequeños
   - Balance entre tamaño de modelo y precisión

5. **Integración de Señales**:
   - Combinar señales de múltiples modelos para mayor robustez
   - Adaptar umbrales de confianza según regímenes de mercado
   - Considerar el uso de ensambles para mejorar estabilidad

## Referencias y Documentación Adicional

- [Documentación de XGBoost con GPU](https://xgboost.readthedocs.io/en/latest/gpu/index.html)
- [Documentación de TensorFlow con GPU](https://www.tensorflow.org/guide/gpu)
- [Optimización de Modelos ML para Trading](https://www.investopedia.com/articles/active-trading/081315/how-machine-learning-trading-works.asp)