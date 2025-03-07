# MidasScalpingv4 Project Guide

## Commands
- **Run with CLI**: `python run.py [arguments]`
- **Run examples**:
  - Simulation: `python run.py --simulate --symbols BTCUSDT,ETHUSDT`
  - With config: `python run.py --config example_config.json`
  - Health check: `python run.py --health-check`
  - Show help: `python run.py --help`
- **Run backtester**: `python backtester.py --symbols BTCUSDT --timeframe 1m --use-ml`
- **Test**: `pytest` or `pytest tests/test_specific.py::test_function`
- **Lint**: `ruff check .` or `ruff check path/to/file.py`
- **Type check**: `mypy .` or `mypy path/to/file.py`
- **Format code**: `black .` or `black path/to/file.py`
- **Install dependencies**: `pip install -e .` or `pip install -r requirements.txt`

## CLI Interface Arguments
- **Main Config**: `--config`, `--api-key`, `--api-secret`
- **Trading Params**: `--symbols`, `--max-open-trades`, `--timeframe`
- **Risk Management**: `--max-risk`, `--stop-loss`, `--take-profit`, `--trailing-stop`, `--max-daily-loss-pct`, `--max-daily-trades`
- **Exchange**: `--testnet`, `--simulate`, `--real-data`, `--sim-balance`
- **Model**: `--model`, `--confidence-threshold`, `--use-gpu`, `--gpu-device`
- **Commissions/Slippage**: `--commission-rate`, `--slippage-pct`
- **Debug**: `--debug`, `--log-level`, `--log-file`
- **Actions**: `--train`, `--backtest`, `--optimize`, `--show-balance`, `--health-check`

## ML Module Parameters
- **GPU Configuration**:
  - `--use-gpu`: Activar aceleración GPU (True/False)
  - `--gpu-device`: ID del dispositivo GPU a usar (default: 0)
- **Model Parameters**:
  - `--xgb-model`: Ruta al modelo XGBoost guardado
  - `--lstm-model`: Ruta al modelo LSTM guardado
  - `--xgb-rounds`: Número de rondas para entrenamiento de XGBoost (default: 100)
  - `--sequence-length`: Longitud de secuencia para LSTM (default: 10)
  - `--feature-count`: Número de características para LSTM (default: 20)
- **Trading Parameters**:
  - `--confidence-threshold`: Umbral de confianza para señales (default: 0.7)
  - `--max-daily-trades`: Máximo de trades por día (default: 30)
  - `--max-daily-loss-pct`: Pérdida máxima diaria (default: 3.0%)
  - `--commission-rate`: Comisión por operación (default: 0.0004, equivalente a 0.04%)
  - `--slippage-pct`: Porcentaje de slippage (default: 0.0002, equivalente a 0.02%)

## Backtester Parameters
```bash
python backtester.py --symbols BTCUSDT --timeframe 1m --use-ml \
  --xgb-model saved_models/xgboost_model.json \
  --lstm-model saved_models/lstm_model \
  --use-gpu \
  --initial-balance 10000 \
  --confidence-threshold 0.7 \
  --max-daily-trades 30 \
  --commission 0.0004 \
  --slippage 0.0002 \
  --plot
```

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports with a blank line between groups
- **Formatting**: Follow PEP 8, use Black for auto-formatting
- **Types**: Use type hints for function parameters and return values
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Functions**: Limit to single responsibility, use descriptive names
- **Error handling**: Use try/except with specific exceptions, avoid bare except clauses
- **Docstrings**: Use Google-style docstrings for functions and classes
- **Testing**: Write unit tests for all new functionality

## Trading Risk Management
- **Stop Loss/Take Profit**: Cada operación debe tener stop loss y take profit definidos
- **Límite Diario**: Implementado límite máximo de trades por día (30 por defecto)
- **Pérdida Máxima**: Stop de pérdidas diario (3% del capital por defecto)
- **Cooling Period**: Tiempo de espera entre operaciones para evitar overtrading
- **Comisiones/Slippage**: Modelado realista de costos de transacción
- **Umbrales de Confianza**: Filtrado de señales débiles (confidence_threshold = 0.7)

## ML Module
Para más información sobre el módulo ML, consultar [README_ML.md](README_ML.md)

Remember to run tests and type checking before committing changes.