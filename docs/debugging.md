# Guía de Depuración para MidasScalping v4

Esta guía proporciona información detallada sobre cómo depurar efectivamente el sistema MidasScalping v4 cuando se presentan problemas.

## Activación del Modo Debug

El sistema incluye un modo de depuración avanzado que proporciona logs detallados y rastreo de errores. Puedes activar este modo de varias formas:

### Mediante Variable de Entorno

```bash
MIDAS_DEBUG=1 python main.py
```

### Mediante Argumentos de Línea de Comandos

```bash
python main.py --debug
```

o

```bash
python run_tui.py --debug
```

## Ubicación de los Logs

Los logs se guardan en el directorio `logs/` con el siguiente formato:

- `Main_cli_AAAAMMDD.log` - Logs estándar de la interfaz de línea de comandos
- `Main_debug_AAAAMMDD.log` - Logs de depuración (solo disponible en modo debug)
- `CLI_Interface_tui_AAAAMMDD.log` - Logs de la interfaz de terminal
- `unhandled_exceptions_AAAAMMDD.log` - Registro de excepciones no controladas

## Niveles de Logging

El sistema utiliza los siguientes niveles de logging:

- **DEBUG**: Información detallada y de bajo nivel para depuración (solo en modo debug)
- **INFO**: Información general sobre el funcionamiento normal
- **TRADE**: Información específica sobre operaciones de trading
- **WARNING**: Advertencias que no impiden la operación pero requieren atención
- **ERROR**: Errores que pueden afectar el funcionamiento pero no detienen el programa
- **CRITICAL**: Errores críticos que pueden causar la terminación del programa

## Características Avanzadas de Depuración

### Rastreo Detallado de Excepciones

En modo debug, todas las excepciones incluyen el stack trace completo para facilitar la identificación del origen del problema.

### Monitoreo de Rendimiento

En modo debug, se registra:
- Tiempo de inicio del bot
- Tiempo de carga del modelo
- Uso de memoria
- Estadísticas de rendimiento periódicas

### Registro de Transacciones

Cada operación comercial se registra con un ID único (TXID) para facilitar el seguimiento de una operación específica en todos los logs.

### Información de GPU

Si el modo GPU está activado, se registra información detallada sobre los dispositivos GPU disponibles y su configuración.

## Comandos Útiles para Depuración

### Ver los Últimos Logs en Tiempo Real

```bash
tail -f logs/Main_debug_AAAAMMDD.log
```

### Buscar Errores en los Logs

```bash
grep -i error logs/Main_*.log
```

### Examinar Operaciones Específicas

```bash
grep -i "TXID:abc123" logs/Main_*.log
```

## Resolución de Problemas Comunes

### El Bot Se Detiene Inesperadamente
Verifica el archivo `unhandled_exceptions_AAAAMMDD.log` para identificar la causa raíz del problema.

### Error de Conexión con Binance
Comprueba la conectividad de red y verifica que las claves API sean correctas y tengan los permisos adecuados.

### Problemas con CUDA/GPU
Revisa los logs de inicialización para verificar si el sistema detectó correctamente tu GPU y si hay errores relacionados con CUDA.

### Operaciones No Ejecutadas
Busca entradas con nivel TRADE en los logs para verificar si el sistema detectó la señal pero no pudo ejecutar la operación.

## Información para Reportar Bugs

Al reportar un problema, incluye siempre:

1. Versión exacta del software
2. Sistema operativo y versión
3. Configuración utilizada (sin incluir claves API)
4. Archivos de log relevantes (con información sensible redactada)
5. Descripción detallada del problema y pasos para reproducirlo