# MidasScalpingv4 Wiki

Bienvenido a la documentación del sistema de trading MidasScalping v4.

## Contenido

- [Arquitectura del Sistema](architecture.md)
- [Estrategias de Trading](trading-strategies.md)
- [Refuerzo por Aprendizaje](reinforcement-learning.md)
- [Gestión de Riesgo](risk-management.md)
- [Configuración](configuration.md)
- [Guía de Instalación](installation.md)
- [Depuración](debugging.md)
- [Solución: Error de Precisión en Binance](precision_fix.md)

## Descripción General

MidasScalping v4 es un sistema de trading algorítmico diseñado específicamente para operaciones de scalping en mercados de criptomonedas. El sistema utiliza una combinación de análisis técnico tradicional y aprendizaje por refuerzo para identificar oportunidades de trading, mientras implementa una gestión de riesgo avanzada para proteger el capital.

### Características Principales

- Estrategias basadas en indicadores técnicos (RSI, Bandas de Bollinger, medias móviles)
- Modelo de aprendizaje por refuerzo para adaptación continua
- Detección de regímenes de mercado para ajustar parámetros dinámicamente
- Gestión de riesgo avanzada con protección de balance
- Interfaz de línea de comandos (CLI) con argumentos configurables
- Interfaz de usuario textual (TUI) para monitoreo en tiempo real
- Soporte para operaciones spot con validación de balance

## Uso de la Interfaz de Línea de Comandos (CLI)

El bot puede ser controlado completamente a través de argumentos de línea de comandos. Esto facilita la automatización y la integración con otros sistemas.

### Ejemplos Básicos

```bash
# Ejecutar en modo simulación con el archivo de configuración de ejemplo
python run.py --config example_config.json --simulate

# Ejecutar con datos reales de Binance usando testnet
python run.py --testnet --symbols BTCUSDT,ETHUSDT --stop-loss 0.8 --take-profit 1.5

# Verificar el balance de la cuenta
python run.py --show-balance --testnet

# Ejecutar una revisión de salud del sistema
python run.py --health-check
```

### Grupos de Argumentos

- **Configuración Principal**: `--config`, `--api-key`, `--api-secret`
- **Parámetros de Trading**: `--symbols`, `--max-open-trades`, `--timeframe`
- **Gestión de Riesgo**: `--max-risk`, `--stop-loss`, `--take-profit`, `--trailing-stop`
- **Conexión a Exchange**: `--testnet`, `--simulate`, `--real-data`, `--sim-balance`
- **Configuración de Modelo**: `--model`, `--confidence`, `--no-gpu`
- **Depuración y Logging**: `--debug`, `--log-level`, `--log-file`
- **Acciones**: `--train`, `--backtest`, `--optimize`, `--show-balance`, `--health-check`

Para ver la lista completa de argumentos disponibles:

```bash
python run.py --help
```

Esta documentación proporciona una visión detallada de cómo funciona el sistema, cómo configurarlo y cómo extenderlo para tus necesidades específicas.