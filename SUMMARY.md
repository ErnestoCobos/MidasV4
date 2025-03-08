# Mejoras de Depuración Implementadas

## Resumen de Cambios

Hemos implementado un sistema completo de depuración que facilita el diagnóstico y seguimiento de operaciones en el bot de trading MidasScalping v4. Las principales mejoras incluyen:

### 1. Sistema de Logs Centralizado y Mejorado

- **Archivo `core/logging_setup.py`**: 
  - Formateadores de logs con colores en consola
  - Niveles de log personalizados para operaciones de trading
  - Rotación de archivos de log
  - Métodos auxiliares para registrar excepciones con stack trace completo
  - Adaptación automática al modo debug

### 2. Herramientas de Depuración Específicas para Trading

- **Archivo `core/debug_helpers.py`**:
  - Decoradores para medir rendimiento de funciones críticas
  - Sistema de tracking de transacciones mediante IDs únicos
  - Clase `TradingDebugger` para análisis detallado de operaciones
  - Exportación de historial de operaciones a JSON
  - Registro de decisiones de trading y sus razones

### 3. Integración en el Bot Principal

- **Modificaciones en `bot.py`**:
  - Inicialización del sistema de depuración
  - IDs de transacción para cada operación de trading
  - Logs detallados de cada paso del proceso de trading
  - Registro de métricas de rendimiento
  - Manejo mejorado de excepciones con contexto

### 4. Accesibilidad para el Usuario

- **Opciones de línea de comandos**:
  - Argumento `--debug` para activar modo depuración
  - Argumento `--log-level` para especificar nivel de detalle
  - Variable de entorno `MIDAS_DEBUG=1` como alternativa

- **Documentación**:
  - Nueva guía de depuración en `docs/debugging.md`
  - Actualización del README con información sobre las nuevas opciones

## Beneficios Para la Depuración

1. **Trazabilidad**: Cada operación ahora tiene IDs únicos que permiten seguir su progreso a través de múltiples componentes.

2. **Visibilidad**: Logs detallados y coloreados facilitan encontrar problemas específicos.

3. **Rendimiento**: Medición de tiempos de ejecución en funciones críticas para detectar cuellos de botella.

4. **Diagnóstico**: Stack traces completos y contexto detallado para cada error.

5. **Exportación**: Capacidad de guardar el historial de operaciones para análisis posterior.

## Cómo Utilizar

Para activar el modo debug, utiliza cualquiera de estas opciones:

```bash
# Mediante variable de entorno
MIDAS_DEBUG=1 python main.py

# O mediante argumento
python main.py --debug

# Para la interfaz TUI
python run_tui.py --debug
```

Para más información, consulta la documentación en `docs/debugging.md`.